import argparse
import os
import dataclasses
import ast


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
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1
    lstm_bidirectional: bool = True
    classifier_proj_size: int = 128
    classifier_act_fn: str = 'tanh'
    target_task: str = 'ner'
    epochs: int = 50
    learning_rate: float = 1e-4
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.


def create_config() -> Config:
    ap = argparse.ArgumentParser()

    for field in dataclasses.fields(Config):
        # for each field in the dataclass, make a argument
        # for the cli. make the argument required if the
        # config doesn't have a default value. use type
        # annotations to set the required type (more on that
        # below).
        if hasattr(Config, field.name):
            field_default = getattr(Config, field.name)
            required = False
        else:
            field_default = None
            required = True

        def type_fn(local_field_type):
            """Cast to type using a combination of ast literal eval
            and type class. Ignore literal eval if the expected type
            is string."""
            def _type_fn(x):
                if not issubclass(local_field_type, str):
                    x = ast.literal_eval(x)
                return local_field_type(x)
            return _type_fn

        # add the actual argument!
        ap.add_argument(f'--{field.name}',
                        default=field_default,
                        required=required,
                        type=type_fn(field.type))

    # parse options, chuck them in the config.
    opts = ap.parse_args()
    return Config(**opts.__dict__)
