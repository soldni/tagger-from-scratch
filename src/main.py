#! /usr/bin/env	python

"""

Author:
Email:
"""

from numpy.lib.twodim_base import triu_indices_from
import torch

from tagger_from_scratch.config import create_config
from tagger_from_scratch.data import make_conll_dataset, ConllTensorSample


def main():
    config = create_config()
    train_dataset, tokenizer = make_conll_dataset(config=config, split='train')
    valid_dataset, _ = make_conll_dataset(config=config, split='valid', tokenizer=tokenizer)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=ConllTensorSample.collate_tensor_samples,
        pin_memory=True,    # pinning memory is fine bc all batches are of same size
    )
    valid_data_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=ConllTensorSample.collate_tensor_samples,
        pin_memory=True,    # pinning memory is fine bc all batches are of same size
    )

    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
    main()
