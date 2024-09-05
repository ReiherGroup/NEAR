# NEAR: A Training-Free Pre-Estimator of Machine Learning Model Performance
## Introduction
This repository contains an implementation of the Network Expressivity by Activation Rank (NEAR) score. 
NEAR is a zero-cost proxy for predicting the best performing neural network architecture in neural architecture search.
It is based on the effective rank of the pre- and post-activation matrix of a neural network layer. NEAR can also be
applied to identify suitable activation functions and weight initialization schemes. For a detailed description, 
we refer to our [paper](https://arxiv.org/abs/2408.08776).

## Installation
The module can be installed as follows:
```bash
git clone <near-score-repository>
cd <near-score-repository>
python3 -m pip install .
```

## Usage
A simple example on how to use the package is given in [example.py](example.py). Please note that the example requires 
the installation of `torchvision`.

## License and Copyright Information
The module near_score is distributed under the BSD 3-Clause "New" or "Revised" License. 
For more license and copyright information, see the file [LICENSE](LICENSE).

## How to Cite
When publishing results obtained with this package, please cite:
```
@Article{Husistein2024,
    title   = {{NEAR: A Training-Free Pre-Estimator of Machine Learning Model Performance}}, 
    author  = {Raphael T. Husistein and Markus Reiher and Marco Eckhoff},
    journal = {arXiv:2408.08776 [cs.LG]} 
    year    = {2024},
}
```

## Support and Contact
In case you encounter any problems or bugs, please write a message to lifelong_ml@phys.chem.ethz.ch.
