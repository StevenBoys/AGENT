# Balance is Essence: Accelerating Sparse Training via Adaptive Gradient Correction

This is the code repository of ITOP-based models in the following paper: Balance is Essence: Accelerating Sparse Training via Adaptive Gradient Correction.

Parts of this code repository is based on the following works by the machine learning community.

* https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization

## Requirements

The library requires Python 3.6.7, PyTorch v1.0.1, and CUDA v9.0.

You can download it via anaconda or pip, see [PyTorch/get-started](https://pytorch.org/get-started/locally/) for further information. 


## Training 
Our implementation includes the code for two dynamic sparse training methods SET (https://www.nature.com/articles/s41467-018-04316-3) and RigL (https://arxiv.org/abs/1911.11134). The main difference is the weight regorwing method: using --growth random for SET; using --growth gradient for RigL.


### CIFAR10/100
We provide the training codes for In-Time Over-Parameterization (ITOP) with our adaptive gradient correction method. 

To train a **dense model**, we just need to remove the --sparse argument.

```
python main.py --seed 18 --sparse_init ERK  --multiplier 1 --lr 0.1 --density 0.05 --update_frequency 1500 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth random --death magnitude --redistribution none

```
To train models with **SET-ITOP** with a **typical** training time, run this command:

```
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 1 --lr 0.1 --density 0.05 --update_frequency 1500 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth random --death magnitude --redistribution none

```
To train models with **RigL-ITOP** with a **typical** training time, run this command:

```
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 1 --lr 0.1 --density 0.05 --update_frequency 4000 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth gradient --death magnitude --redistribution none

```

To train models with **SET-ITOP** with an **extended** training time, change the value of --multiplier (e.g., 5 times) and run this command:

```
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 5 --lr 0.1 --density 0.05 --update_frequency 1500 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth random --death magnitude --redistribution none

```
To train models with **RigL-ITOP** with an **extended** training time, change the value of --multiplier (e.g., 5 times) and run this command:

```
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 5 --lr 0.1 --density 0.05 --update_frequency 4000 --epochs 250 --model vgg-c --data cifar10 --decay_frequency 30000 --batch-size 128 --growth gradient --death magnitude --redistribution none

```

Options:
* --sparse - Enable sparse mode (remove this if want to train dense model)
* --sparse_init - type of sparse initialization. Choose from: uniform, ERK
* --model (str) - type of networks
```
  MNIST:
	lenet5
	lenet300-100

 CIFAR-10/100：
	alexnet-s
	alexnet-b
	vgg-c
	vgg-d
	vgg-like
	wrn-28-2
	wrn-22-8
	wrn-16-8
	wrn-16-10
	ResNet-18
        ResNet-34
```
* --growth (str) - growth mode. Choose from: random, gradient, momentum
* --death (str) - removing mode. Choose from: magnitude, SET, threshold
* --redistribution (str) - redistribution mode. Choose from: magnitude, nonzeros, or none. (default none)
* --density (float) - density level (default 0.05)
* --death-rate (float) - initial pruning rate (default 0.5)

The sparse operatin is in the sparsetraining/core.py file. 

For better sparse training performance, it is suggested to decay the learning rate at the 1/2 and 3/4 training time instead of using the default learning rate schedule in main.py. 


