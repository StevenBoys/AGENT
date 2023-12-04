# Balance is Essence: Accelerating Sparse Training via Adaptive Gradient Correction

This is the code repository of RigL-based models in the following paper: Balance is Essence: Accelerating Sparse Training via Adaptive Gradient Correction.

Parts of this code repository is based on the following works by the machine learning community.

* https://github.com/varun19299/rigl-reproducibility

## Setup

This code requires python3.8 and pytorch: 1.7.0+ (GPU support preferable).

Then, run this command
```bash
make install
```

## W&B API key

Copy your WandB API key to `wandb_api.key`.
Will be used to login to your dashboard for visualisation. 
Alternatively, you can skip W&B visualisation, 
and set `wandb.use=False` while running the python code or `USE_WANDB=False` while running make commands.

## Usage

To train WideResNet-22-2 with RigL on CIFAR10, run this command

```
make cifar10.ERK.RigL DENSITY=0.2 SEED=0
````

You can change `DENSITY` when you want to use a different density (1 - sparsity) level.
