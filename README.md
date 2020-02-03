# taglets
automatically construct and integrate weak supervision sources

## Description
In this package, we automatically construct labeling functions for the given ```image classification problem``` in which there is not enough labeled data. Based on the amount of labeled data, we call appropriate labeling functions, train them, then combine them. We leverage many few shot and transfer learning methodologies in order to create a reliable framework for learning with less labeled data.

## Installation

You could ``git clone`` to the repository, and then install the requirements using ``pip install requirements.txt`` or ``conda install requirements.txt``

## How To
To start, you will need to run 'controller.py', which is located in the 'taglets' directory. It interacts with the API to receive all the information related the given task. There are two phases: `base` and `adaptation`. Each phase has several checkpoints; in each checkpoint you can request the label for some data points, and after training the model submit the predictions on test data.  

## Contributing
We welcome pull requests. For any changes, please open an issue first to discuss what you would like to change.
