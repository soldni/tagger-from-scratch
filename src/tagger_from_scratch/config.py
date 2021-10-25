import argparse
import os
import dataclasses


HOME = os.path.expanduser('~')


@dataclasses.dataclass
class Config:
    conll_data_path: str = f"{HOME}/data/tagger-from-scratch/conll2003"
    fasttext_data_path: str = f"{HOME}/data/tagger-from-scratch/fasttext"
    fasttext_emb_file: str = 'wiki-news-300d-50k.vec'


def create_config() -> Config:
    ap = argparse.ArgumentParser()

    for field in dataclasses.fields(Config):
        try:
            field_default = getattr(Config, field.name)
            required = False
        except AttributeError:
            field_default = None
            required = True
        ap.add_argument(f'--{field.name}',
                        default=field_default,
                        required=required,
                        type=field.type)

    opts = ap.parse_args()
    return Config(**opts.__dict__)
