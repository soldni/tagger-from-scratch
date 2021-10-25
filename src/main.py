#! /usr/bin/env	python

"""
A simple proof-of-concept tagger trained on CoNLL 2003

Author: Luca Soldaini
Email:  luca@soldaini.net
"""

import torch
import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from tagger_from_scratch.config import create_config
from tagger_from_scratch.data import make_conll_dataset, ConllTensorSample
from tagger_from_scratch.model import BiLSTMModel


def main():
    config = create_config()
    print(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    model = BiLSTMModel(config=config, tokenizer=tokenizer)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate,
                                 betas=(config.adam_beta_1, config.adam_beta_2),
                                 eps=config.adam_eps,
                                 weight_decay=config.adam_weight_decay)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.
        with tqdm.tqdm(total=len(train_data_loader), unit=' updates', desc=f'Train – epoch {epoch}') as pbar:
            for batch in train_data_loader:
                optimizer.zero_grad()
                batch = batch.to(device)
                output = model(**batch)
                output.loss.backward()
                optimizer.step()

                pbar.update(1)
                train_loss += float(output.loss)
                pbar.set_postfix({'loss': train_loss})

        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0
        with tqdm.tqdm(total=len(valid_data_loader), unit=' updates', desc=f'Valid – epoch {epoch}') as pbar,\
                torch.no_grad():
            for batch in valid_data_loader:
                batch = batch.to(device)
                output = model(**batch)

                predictions = torch.argmax(output.output, dim=2)
                predictions = torch.where(output.labels < 0, -100, predictions)

                y_true.extend([l for l in output.labels.flatten().tolist() if l >= 0])
                y_pred.extend([l for l in predictions.flatten().tolist() if l >= 0])
                pbar.update(1)
                eval_loss += float(output.loss)
                pbar.set_postfix({'loss': eval_loss})

        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        print(f"Epoch {epoch}: P={prec:02.2%} – R={rec:02.2%} – F1={f1:02.2%}")



if __name__ == '__main__':
    main()
