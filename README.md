# taglets
automatically construct and integrate weak supervision sources

## requirements
- python 3
- sklearn, numpy, scipy, pandas, json
- pytorch
- CUDA

## Description
In this package, we will automatically construct labeling functions for the given ```image classification problem``` in which there is not enought labeled data. Based on the amount of labeled data, we call appropriate labeling functions, train them, then combine them. We leverage many few shot and transfer learning methodolgies in order to create a relaible framework for learning with less labeled data.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install taglets.

```bash
pip install taglets
```

## How To
To start, you will need to run 'controller.py', which is located in the current directory. It interacts with the API to receive all the information related the given task.  

## Contributing
We welcome pull requests. For any changes, please open an issue first to discuss what you would like to change.

##Acknowledgments


## License
[BATS](http://stephenbach.net/)


