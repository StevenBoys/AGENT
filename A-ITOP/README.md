# Balance is Essence: Accelerating Sparse Training via Adaptive Gradient Correction

This is the code repository of ITOP-based models in the following paper: Balance is Essence: Accelerating Sparse Training via Adaptive Gradient Correction.

Parts of this code repository is based on the following works by the machine learning community.

* https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization

## Setup

This code requires Python 3.6.7, PyTorch v1.0.1, and CUDA v9.0.

## Usage

This code support two sparse training methods including [SET](https://www.nature.com/articles/s41467-018-04316-3) and [RigL](https://arxiv.org/abs/1911.11134). To choose which method to use, you can change the weight regorwing method: using --growth random for SET; using --growth gradient for RigL.

To train SET-ITOP-based models, run this command:

```
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 1 --lr 0.1 --density 0.05 --update_frequency 1500 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth random --death magnitude --redistribution none

```

To train RigL-ITOP-based models, run this command:

```
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 1 --lr 0.1 --density 0.05 --update_frequency 4000 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth gradient --death magnitude --redistribution none

```

To train SET-ITOP-based models with an extended training time, change the value of --multiplier (e.g., 5 times) and run this command:

```
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 5 --lr 0.1 --density 0.05 --update_frequency 1500 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth random --death magnitude --redistribution none

```