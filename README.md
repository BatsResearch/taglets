# taglets
automatically construct and integrate weak supervision sources

[![Build Status](https://travis-ci.com/BatsResearch/taglets.svg?token=sinAgJjnTsxQ2oN3R9vi&branch=master)](https://travis-ci.com/BatsResearch/taglets)

## Description
In this package, we automatically construct labeling functions for the given 
image classification problem in which there is not enough labeled data.
Based on the amount of labeled data, we call appropriate modules, create weak
labelers called _taglets_, and then combine their outputs to train an end model.

## Building a docker image
In the top level directory, run
```
docker build --tag brown_taglets:1.0 .
```

## Running the evaluation
To start a container, run 
```
docker run --rm -v /lwll:/lwll --gpus all --shm-size 64G --ulimit nofile=1000000:1000000 brown_taglets:1.0
```
Note that "--shm-size 64G" and "--ulimit nofile=1000000:1000000" are crucial for our system to work.

## Important Notes
- For the dry run, our system currently works only when LWLL_TA1_GPUS = 'all'. 
We will make sure to fix it to work with a list of GPU ids before the actual evaluation.
