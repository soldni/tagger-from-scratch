from typing import Sequence, Dict, Tuple
import dataclasses

import torch
import numpy as np

from tagger_from_scratch.config import Config


@dataclasses.dataclass
class ConllCourpusSample:
    tokens: list = dataclasses.field(default_factory=lambda: [])
    pos: list = dataclasses.field(default_factory=lambda: [])
    con: list = dataclasses.field(default_factory=lambda: [])
    ner: list = dataclasses.field(default_factory=lambda: [])

    def append(self, token: str, pos: str, con: str, ner: str):
        self.tokens.append(token)
        self.pos.append(pos)
        self.con.append(con)
        self.ner.append(ner)

    def __len__(self):
        return len(self.tokens)


def load_conll_corpus(path: str) -> Sequence[ConllCourpusSample]:
    raw_data = []
    with open(path, mode='r', encoding='utf-8') as f:
        for ln in f:
            if ln.startswith('-DOCSTART-'):
                continue
            elif ln == '\n':
                raw_data.append(ConllCourpusSample())
            else:
                token, pos, con, ner = ln.strip().split()
                raw_data[-1].append(token=token, pos=pos, con=con, ner=ner)

    if len(raw_data[-1]) == 0:
        raw_data.pop(-1)

    return raw_data


def load_fasttext_vectors(path: str) -> Sequence[Tuple[str, np.ndarray]]:
    fasttext_dict = []

    with open(path, mode='r', encoding='utf-8') as f:
        # skip first line, only has shape info about dim
        # and size of vocab
        next(f)
        for i, ln in enumerate(f):
            token, *embedding_values = ln.strip().split()
            embedding_values = np.array(embedding_values)
            fasttext_dict.append((token, embedding_values))

    return fasttext_dict


class ConllTokenizer:
    unk_token = '__UNK__'
    pad_token = '__PAD__'

    def __init__(self):
        self.tokens_vocab = {}
        self.pos_vocab = {}
        self.con_vocab = {}
        self.ner_vocab = {}
        self.has_trained = False
        self.max_length = -1

    def train(self, conll_corpus: Sequence[ConllCourpusSample], fasttext_vectors: Dict[str, np.ndarray] = None):
        if fasttext_vectors:
            self.tokens_vocab.update({token: i for i, (token, _) in enumerate(fasttext_vectors)})

        for sample in conll_corpus:
            if not fasttext_vectors:
                for token in sample.tokens:
                    self.tokens_vocab.setdefault(token, len(self.tokens_vocab))
            for ner in sample.ner:
                self.ner_vocab.setdefault(ner, len(self.ner_vocab))
            for con in sample.con:
                self.con_vocab.setdefault(con, len(self.con_vocab))
            for pos in sample.pos:
                self.pos_vocab.setdefault(pos, len(self.pos_vocab))
            self.max_length = max(self.max_length, len(sample))

        for vocab in (self.tokens_vocab, self.pos_vocab, self.con_vocab, self.ner_vocab):
            vocab[self.unk_token] = len(vocab)
            vocab[self.pad_token] = len(vocab)

        self.has_trained = True

    def _tokenize_field(self, sample, vocab):
        unk_token_id = vocab[self.unk_token]
        ids = tuple(vocab.get(sample[i] if i < len(sample) else self.pad_token, unk_token_id)
                    for i in range(self.max_length))
        return ids

    def tokenize_tokens(self, conll_sample: ConllCourpusSample):
        return self._tokenize_field(conll_sample.tokens, self.tokens_vocab)

    def tokenize_pos(self, conll_sample: ConllCourpusSample):
        return self._tokenize_field(conll_sample.pos, self.pos_vocab)

    def tokenize_con(self, conll_sample: ConllCourpusSample):
        return self._tokenize_field(conll_sample.con, self.con_vocab)

    def tokenize_ner(self, conll_sample: ConllCourpusSample):
        return self._tokenize_field(conll_sample.ner, self.ner_vocab)


def make_conll_dataset(config: Config, split: str, tokenizer: ConllTokenizer = None):

    assert split in {'train', 'test', 'valid'}, \
        f"Split should either 'train', 'test', or 'valid', not {split}"

    conll_corpus = load_conll_corpus(f'{config.conll_data_path}/{split}.txt')
    if not tokenizer:
        tokenizer = ConllTokenizer()
        if config.use_fasttext:
            fasttext_vectors = load_fasttext_vectors(f'{config.fasttext_data_path}/{config.fasttext_emb_file}')
        else:
            fasttext_vectors = None
        tokenizer.train(conll_corpus=conll_corpus, fasttext_vectors=fasttext_vectors)

    tokens_tensor = torch.LongTensor(tuple(tokenizer.tokenize_tokens(sample) for sample in conll_corpus))
    ner_tensor = torch.LongTensor(tuple(tokenizer.tokenize_ner(sample) for sample in conll_corpus))
    pos_tensor = torch.LongTensor(tuple(tokenizer.tokenize_pos(sample) for sample in conll_corpus))
    con_tensor = torch.LongTensor(tuple(tokenizer.tokenize_con(sample) for sample in conll_corpus))

    dataset = ConllDataset(tokens_tensor=tokens_tensor,
                           ner_tensor=ner_tensor,
                           pos_tensor=pos_tensor,
                           con_tensor=con_tensor)

    return dataset, tokenizer


@dataclasses.dataclass
class ConllTensorSample:
    tokens: torch.LongTensor
    ner: torch.LongTensor
    pos: torch.LongTensor
    con: torch.LongTensor

    @classmethod
    def collate_tensor_samples(cls, seq: Sequence):
        return cls(tokens=torch.stack(tuple(elem.tokens for elem in seq)),
                   ner=torch.stack(tuple(elem.ner for elem in seq)),
                   pos=torch.stack(tuple(elem.pos for elem in seq)),
                   con=torch.stack(tuple(elem.con for elem in seq)))

    def to(self, device: str):
        return self.__class__(tokens=self.tokens.to(device),
                              ner=self.ner.to(device),
                              pos=self.pos.to(device),
                              con=self.con.to(device))

    def keys(self):
        return (f.name for f in dataclasses.fields(self))

    def __getitem__(self, key):
        return getattr(self, key)


class ConllDataset(torch.utils.data.Dataset):
    def __init__(self,
                 tokens_tensor: torch.LongTensor,
                 ner_tensor: torch.LongTensor,
                 pos_tensor: torch.LongTensor,
                 con_tensor: torch.LongTensor):
        self.tokens_tensor = tokens_tensor
        self.ner_tensor = ner_tensor
        self.pos_tensor = pos_tensor
        self.con_tensor = con_tensor

        super().__init__()

    def __len__(self):
        return self.tokens_tensor.size(0)

    def __getitem__(self, index: int):
        return ConllTensorSample(tokens=self.tokens_tensor[index],
                                 ner=self.ner_tensor[index],
                                 pos=self.pos_tensor[index],
                                 con=self.con_tensor[index])
