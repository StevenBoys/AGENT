# Balance is Essence: Accelerating Sparse Training via Adaptive Gradient Correction

This is the code for repository of BSR-Net-based models in the following paper: Balance is Essence: Accelerating Sparse Training via Adaptive Gradient Correction.

Parts of this code repository is based on the following works by the machine learning community.

* https://github.com/IGITUGraz/SparseAdversarialTraining

## Setup

You can install all required packages by:
```bash
pip install -r requirements.txt
```

## Usage

You can use `run_a_brs_net.py` to train the accelerated Bayesian sparse robust networks (A-BSR-Net) from scratch. Brief description of possible arguments are:

- `--data`: "cifar10", "cifar100", "svhn"
- `--model`: "vgg16", "resnet18", "wrn28_4"
- `--objective`: "natural" (Natural training), "at" (Standard AT), "mat" (Mixed-batch AT), trades", "mart"
- `--sparse_train`: enable end-to-end sparse training
- `--connectivity`: sparse connectivity ratio (used when `--sparse_train` is enabled)

Remarks:
* For the `--data "svhn"` option, you will need to create the directory `datasets/SVHN/` and place the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset's [train](http://ufldl.stanford.edu/housenumbers/train_32x32.mat) and [test](http://ufldl.stanford.edu/housenumbers/test_32x32.mat) `.mat` files there.

### End-to-end training and evaluation for accelerated Bayesian sparse robust networks

The following sample scripts can be used to train the accelerated Bayesian sparse robust networks (A-BSR-Net) from scratch, and also perform white box robustness evaluations using PGD attacks via [Foolbox](https://github.com/bethgelab/foolbox).

- `cifar10_vgg16_at.sh`: Standard adversarial training for a sparse VGG-16 on CIFAR-10.
