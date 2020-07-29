# taglets
automatically construct and integrate weak supervision sources

[![Build Status](https://travis-ci.com/BatsResearch/taglets.svg?token=sinAgJjnTsxQ2oN3R9vi&branch=master)](https://travis-ci.com/BatsResearch/taglets)

## Description
In this package, we automatically construct labeling functions for the given ```image classification problem``` in which there is not enough labeled data. Based on the amount of labeled data, we call appropriate labeling functions, train them, then combine them. We leverage many few shot and transfer learning methodologies in order to create a reliable framework for learning with less labeled data.

## Installation

1. Clone the repository
2. In the top level directory, execute
```
pip install .
```
3. Install the dependencies by executing:
```
pip install -r requirements.txt
```

## How To
To start, you will need to run 'controller.py', which is located in the 'taglets' directory. It interacts with the API to receive all the information related the given task. There are two phases: `base` and `adaptation`. Each phase has several checkpoints; in each checkpoint you can request the label for some data points, and after training the model submit the predictions on test data. 

## Additional Setup
### Zero-Shot Learning Setup
1. Download the GloVe embeddings in the scads root folder `<Scads_data_path>/glove.840B.300d.txt`.

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

2. Place the pretrained model in `<Scads_data_path>/pretrained_models/zero_shot/transformer.pt`

## Contributing
We welcome pull requests. For any changes, please open an issue first to discuss what you would like to change.
