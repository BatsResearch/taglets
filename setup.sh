#!/usr/bin/env bash

pip install -r requirements.txt
pip install -e .

mkdir -p predefined
mkdir -p predefined/embeddings
mkdir -p predefined/zsl_kg_lite

cd predefined
wget -nc https://storage.googleapis.com/taglets-public/scads.spring2021.sqlite3

cd embeddings
wget -nc https://storage.googleapis.com/taglets-public/embeddings/glove.840B.300d.txt.gz
gunzip --force < glove.840B.300d.txt.gz > glove.840B.300d.txt

wget -nc https://storage.googleapis.com/taglets-public/embeddings/numberbatch-en19.08.txt.gz

cd ../zsl_kg_lite
wget -nc https://storage.googleapis.com/taglets-public/zsl_kg_lite/transformer.pt

cd ../
wget -nc https://storage.googleapis.com/taglets-public/protonet-trained.pth

