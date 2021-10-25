# Tagger From Scratch

Just a simple BiLSTM tagger I'm coding up with vanilla PyTorch. The goal is to eventually compare with ease of use of [PyTorch Lightning](https://www.pytorchlightning.ai/) and [PyTorch Ignite](https://pytorch.org/ignite/index.html).

## Training

To train, first prepare data with `bash prepare_data.sh`. Then run `python src/main.py`.