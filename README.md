# LinearL: A Simple Feedforward Neural Network for MNIST Classification

This repository contains a simple feedforward neural network implemented in PyTorch for classifying MNIST handwritten digits. The network consists of several linear layers with batch normalization, dropout, and activation functions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [References](#references)

## Installation

To run this code, you need to have Python and the required libraries installed. You can install the required packages using pip:

```bash
pip install torch torchvision tqdm matplotlib

```

## Usage

```git clone https://github.com/riteshshergill/py_linear
cd py_linear
```

## Model Architecture
The LinearL model is composed of multiple LinearLayer modules, each containing:

* A linear transformation.
* Batch normalization.
* Activation function (ReLU by default).
* Dropout for regularization.

## Training
Please refer to the train.ipynb file to understand how to train the model

## Evaluation

The test_model method evaluates the LinearL model on test data and prints the predictions and ground truths for a specified number of samples.

## References

<a href="https://pytorch.org/docs/stable/index.html">PyTorch Documentation</a>

<a href="https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html">MNIST Dataset</a>



