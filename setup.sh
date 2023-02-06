#!/usr/bin/env bash

pip install -r requirements.txt
pip install -e .

mkdir -p predefined
mkdir -p predefined/embeddings
mkdir -p predefined/zsl_kg_lite

cd predefined
wget -nc -q https://storage.googleapis.com/taglets-public/scads.spring2021.sqlite3
wget -nc -q https://storage.googleapis.com/taglets-public/scads.spring2023.sqlite3
wget -nc -q https://storage.googleapis.com/taglets-public/scads.imagenet22k.sqlite3
wget -nc -q https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz

cd embeddings
wget -nc -q https://storage.googleapis.com/taglets-public/embeddings/glove.840B.300d.txt.gz
gunzip --force < glove.840B.300d.txt.gz > glove.840B.300d.txt

wget -nc -q https://storage.googleapis.com/taglets-public/embeddings/numberbatch-en19.08.txt.gz
wget -nc -q https://storage.googleapis.com/taglets-public/embeddings/spring2021_processed_numberbatch.h5
wget -nc -q https://storage.googleapis.com/taglets-public/embeddings/spring2023_processed_numberbatch.h5
wget -nc -q https://storage.googleapis.com/taglets-public/embeddings/imagenet22k_processed_numberbatch.h5

cd ../zsl_kg_lite
wget -nc -q https://storage.googleapis.com/taglets-public/zsl_kg_lite/transformer.pt

cd ../
wget -nc -q https://storage.googleapis.com/taglets-public/protonet-trained.pth

