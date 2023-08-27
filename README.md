# Balance is Essence: Accelerating Sparse Training via Adaptive Gradient Correction

This is the code repository of the following paper: Balance is Essence: Accelerating Sparse Training via Adaptive Gradient Correction.

The folders named `A-BSR-Net`, `A-ITOP`, and `A-RigL` include codes for BSR-Net-based, ITOP-based, and RigL-based models, respectively. For detailed instructions on using the code, see `README.md` in each folder.

## BSR-Net

Bayesian Sparse Robust Training (BSR-Net) is a Bayesian Sparse and Robust training pipeline. Based on a Bayesian posterior sampling principle, a network rewiring process simultaneously learns the sparse connectivity structure and the robustness-accuracy trade-off based on the adversarial learning objective. More specifically, regarding its mask update, it prunes all negative weights and grows new weights randomly.

## ITOP

ITOP is a recent pipeline for dynamic sparse training, which uses sufficient and reliable parameter exploration to achieve in-time over-parameterization and find well-performing sparse models. 

## RigL

RigL is a popular dynamic sparse training method that uses weight and gradient magnitudes to learn the connections. 
