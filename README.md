# taglets
taglets stands for ''Tasks Algorithmically Given Labels Established via Transferred Symbols.'' A novel framework for creating weak label sources from existing datasets, combining them through a modular architecture, and selecting key examples to label under limited budgets.

[![Build Status](https://travis-ci.com/BatsResearch/taglets.svg?token=sinAgJjnTsxQ2oN3R9vi&branch=master)](https://travis-ci.com/BatsResearch/taglets)

## Description
In this package, we automatically construct labeling functions for the given
image classification problem in which there is not enough labeled data.
For each amount of labeled data, we call appropriate modules, create weak
labelers called _taglets_, and then combine their outputs to train an end model.

## Building a Docker Image
In the top level directory, run
```
docker build --tag brown_taglets:1.0 .
```

## Running the Evaluation
To start a container, run
```
docker run --rm --env-file env.list -v /lwll:/lwll:delegated --gpus all --shm-size 64G --ulimit nofile=1000000:1000000 brown_taglets:1.0
```
**Note:** "--shm-size 64G" and "--ulimit nofile=1000000:1000000" are crucial for our system to work.
