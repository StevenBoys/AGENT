# Balance is Essence: Accelerating Sparse Training via Adaptive Gradient Correction

This is the code repository of the following paper: Balance is Essence: Accelerating Sparse Training via Adaptive Gradient Correction.

The folders named `A-BSR-Net`, `A-ITOP`, and `A-RigL` include codes for BSR-Net-based, ITOP-based, and RigL-based models, respectively. For detailed instructions on using the code, see `README.md` in each folder.

Parts of this code repository is based on the following works by the machine learning community.

* https://github.com/IGITUGraz/SparseAdversarialTraining
* https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization
* https://github.com/varun19299/rigl-reproducibility

## BSR-Net

BSR-Net is a Bayesian sparse training method that also takes into account the adversarial training. It is proposed in paper [Training Adversarially Robust Sparse Networks via Bayesian Connectivity Sampling](https://proceedings.mlr.press/v139/ozdenizci21a.html), published at ICML 2021. 

## ITOP

ITOP is a sparse training method that uses sufficient and reliable parameter exploration to achieve in-time over-parameterization and find well-performing sparse models. It is proposed in paper [Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training](https://arxiv.org/abs/2102.02887), published at ICML 2021. 

## RigL

RigL is a popular sparse training method that uses weight and gradient magnitudes to learn the connections. It is proposed in paper [Rigging the Lottery: Making all Tickets Winners](https://arxiv.org/abs/1911.11134), published at ICML 2020. 
