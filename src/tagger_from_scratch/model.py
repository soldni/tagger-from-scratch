import dataclasses

import torch

from tagger_from_scratch.config import Config
from tagger_from_scratch.data import ConllTokenizer, load_fasttext_vectors


@dataclasses.dataclass
class BiLSTMOuput:
    output: torch.Tensor
    loss: torch.Tensor = None
    labels: torch.Tensor = None



class BiLSTMModel(torch.nn.Module):
    def __init__(self, config: Config, tokenizer: ConllTokenizer):
        self.target_task = config.target_task

        assert self.target_task in {'pos', 'ner', 'con'}, \
            "Choose between 'pos', 'ner', and 'con' as task"

        self.in_pad_token_id = tokenizer.tokens_vocab[tokenizer.pad_token]
        self.out_pad_token_id = getattr(tokenizer, f'{self.target_task}_pad_id')
        self.target_vocab_size = len(getattr(tokenizer, f'{self.target_task}_vocab'))

        super().__init__()
        if config.use_fasttext:
            fasttext_emb = load_fasttext_vectors(f'{config.fasttext_data_path}/{config.fasttext_emb_file}')
            fasttext_emb = torch.stack(tuple(emb for _, emb in fasttext_emb))
            embedding_dim = fasttext_emb.size(1)
        else:
            embedding_dim = config.embedding_size
            fasttext_emb = None

        self.embeddings = torch.nn.Embedding(num_embeddings=len(tokenizer.tokens_vocab),
                                             embedding_dim=embedding_dim,
                                             padding_idx=self.in_pad_token_id)
        if fasttext_emb is not None:
            missing_dims = max(0, self.embeddings.weight.size(0) - fasttext_emb.size(0))
            if missing_dims > 0:
                fasttext_emb = torch.vstack((fasttext_emb, self.embeddings.weight[-missing_dims:, :]))
            self.embeddings.weight = torch.nn.Parameter(fasttext_emb)

        self.lstm = torch.nn.LSTM(input_size=self.embeddings.embedding_dim,
                                  batch_first=True,
                                  hidden_size=config.lstm_hidden_size,
                                  num_layers=config.lstm_num_layers,
                                  dropout=config.lstm_dropout,
                                  bidirectional=config.lstm_bidirectional)

        self.proj = torch.nn.Linear(self.lstm.hidden_size * (2 if config.lstm_bidirectional else 1),
                                    config.classifier_proj_size)
        self.act_fn = getattr(torch.nn.functional, config.classifier_act_fn)
        self.out = torch.nn.Linear(config.classifier_proj_size,
                                   self.target_vocab_size)

    def forward(self, tokens, ner=None, pos=None, con=None) -> BiLSTMOuput:

        labels = (ner if self.target_task == 'ner' else
                  (con if self.target_task == 'con' else pos))

        # this is not ideal at all, but lengths to be sent to the packing function
        # must be on the cpu, not GPU ???? We could pass in lengths as one of the inputs
        # instead of calculating them here, but that would make the whole thing rather
        # messy so I'm going to go for this hack instead.
        lengths = torch.where(tokens != self.in_pad_token_id, 1, 0).sum(1).cpu()

        # we actually have to trim labels to maximum length for this mini-batch
        # because otherwise their size wont match the one out of `pad_packed_sequence`
        labels = labels[:, :max(lengths)]

        embeddings = self.embeddings(tokens)

        packed_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=embeddings,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False    # needs to be true only if we care about ONNX comp
        )

        packed_encodings, *_ = self.lstm(packed_embeddings)
        encodings, *_ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_encodings,
            batch_first=True,
            padding_value=self.in_pad_token_id
        )

        proj = self.act_fn(self.proj(encodings))
        output = self.out(proj)

        if labels is not None:
            labels = torch.where(labels == self.out_pad_token_id, -100, labels)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output.view(-1, self.target_vocab_size), labels.view(-1))
        elif self.training:
            raise RuntimeError('In training mode, but no labels provided')
        else:
            loss = None

        return BiLSTMOuput(output=output, loss=loss, labels=labels)
