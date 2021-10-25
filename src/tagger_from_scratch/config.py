import argparse
import os
import dataclasses
import ast

import ipdb

HOME = os.path.expanduser('~')


@dataclasses.dataclass
class Config:
    conll_data_path: str = f"{HOME}/data/tagger-from-scratch/conll2003"
    fasttext_data_path: str = f"{HOME}/data/tagger-from-scratch/fasttext"
    fasttext_emb_file: str = 'wiki-news-300d-50k.vec'
    use_fasttext: bool = False
    batch_size: int = 64
    num_workers: int = 0
    embedding_size: int = 128
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.1
    lstm_bidirectional: bool = True
    classifier_proj_size: int = 64
    classifier_act_fn: str = 'relu'
    target_task: str = 'ner'
    epochs: int = 50
    learning_rate: float = 1e-5


def create_config() -> Config:
    ap = argparse.ArgumentParser()

    for field in dataclasses.fields(Config):
        if hasattr(Config, field.name):
            field_default = getattr(Config, field.name)
            required = False
        else:
            field_default = None
            required = True

        def type_fn(local_field_name, local_field_type, local_field_default):
            def _type_fn(x):
                print(local_field_name, local_field_type, local_field_default)
                if not issubclass(local_field_type, str):
                    x = ast.literal_eval(x)
                return local_field_type(x)
            return _type_fn

        ap.add_argument(f'--{field.name}',
                        default=field_default,
                        required=required,
                        type=type_fn(field.name, field.type, field_default))

    opts = ap.parse_args()
    return Config(**opts.__dict__)
