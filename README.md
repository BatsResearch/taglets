# taglets
automatically construct and integrate weak supervision sources

[![Build Status](https://travis-ci.com/BatsResearch/taglets.svg?token=sinAgJjnTsxQ2oN3R9vi&branch=master)](https://travis-ci.com/BatsResearch/taglets)

## Description
In this package, we automatically construct labeling functions for the given 
image classification problem in which there is not enough labeled data.
Based on the amount of labeled data, we call appropriate modules, create weak
labelers called _taglets_, and then combine their outputs to train an end model.

## Prerequisites
1. Python 3.7 with `pip` installed in a *nix environment.
2. An internet connection is required
to download predefined data and pretrained models.
3. External datasets provided by JPL at `TODO`

## Installation
In the top level directory, run
```
./setup.sh
```

## Running the evaluation
To run the Fall 2020 LwLL evaluation, run 
```
./run_jpl.sh
```
