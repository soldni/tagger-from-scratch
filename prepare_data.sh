#! /usr/bin/env bash

# location for data
DATA_DIR="${HOME}/data/tagger-from-scratch"

# record where we are
CURRENT_DIR="$(pwd)"

CONLL_DATA_PATH="${DATA_DIR}/conll2003"
if [ ! -d "${CONLL_DATA_PATH}" ]; then
    # make directory, download tagger data
    mkdir -p "${DATA_DIR}/conll2003"
    cd ${DATA_DIR}/conll2003
    wget https://data.deepai.org/conll2003.zip
    unzip conll2003.zip
    rm -rf conll2003.zip
fi

FASTTEXT_DATA_PATH="${DATA_DIR}/fasttext"
if [ ! -d "${FASTTEXT_DATA_PATH}" ]; then
    # embeddings to init the network with
    mkdir -p "${DATA_DIR}/fasttext"
    cd ${DATA_DIR}/fasttext
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
    unzip wiki-news-300d-1M.vec.zip
    rm -rf wiki-news-300d-1M.vec.zip
fi

# create a 50k vocab version of the embeddings
# otherwise it'll take forever to load
SMALL_FASTTEXT="${FASTTEXT_DATA_PATH}/wiki-news-300d-50k.vec"
echo "" >> ${SMALL_FASTTEXT}
cat "${FASTTEXT_DATA_PATH}/wiki-news-300d-1M.vec" | tail -n +2 | head -n 50000 >> "${SMALL_FASTTEXT}"

cd ${CURRENT_DIR}