import dataclasses

import torch

from tagger_from_scratch.config import Config
from tagger_from_scratch.data import ConllTokenizer


@dataclasses.dataclass
class BiLSTMOuput:
    output: torch.Tensor
    loss: torch.Tensor = None



class BiLSTMModel(torch.nn.Module):
    def __init__(self, config: Config, tokenizer: ConllTokenizer):
        self.target_task = config.target_task
        self.target_vocab_size = len(getattr(tokenizer, f'{self.target_task}_vocab'))

        assert self.target_task in {'pos', 'ner', 'con'}, \
            "Choose between 'pos', 'ner', and 'con' as task"

        super().__init__()
        # if config.use_fasttext:
        #     fasttext = load_fasttext_vectors(f'{config.fasttext_data_path}/{config.fasttext_emb_file}')
        self.embeddings = torch.nn.Embedding(num_embeddings=len(tokenizer.tokens_vocab),
                                             embedding_dim=config.embedding_size,
                                             padding_idx=tokenizer.tokens_vocab[tokenizer.pad_token])

        self.lstm = torch.nn.LSTM(input_size=self.embeddings.embedding_dim,
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

        embeddings = self.embeddings(tokens)
        encodings, *_ = self.lstm(embeddings)
        proj = self.act_fn(self.proj(encodings))
        output = self.out(proj)

        if self.training and labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output.view(-1, self.target_vocab_size), labels.view(-1))
        else:
            loss = None

        return BiLSTMOuput(output=output, loss=loss)
