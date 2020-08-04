#!/usr/bin/env bash

pip install -r requirements.txt

mkdir -p predefined
mkdir -p predefined/embeddings
mkdir -p predefined/zsl_kg_lite

cd predefined
# wget https://storage.googleapis.com/taglets-public/scads.fall2020.sqlite3

cd embeddings
wget https://storage.googleapis.com/taglets-public/embeddings/glove.840B.300d.zip
wget https://storage.googleapis.com/taglets-public/embeddings/numberbatch-en19.08.txt.gz

cd ../zsl_kg_lite
wget https://storage.googleapis.com/taglets-public/zsl_kg_lite/transformer.pt