#!/bin/bash

dataset="cifar10"
arch="vgg16"
classes=10
wd=1e-3

# Train robust and sparse models with Standard AT
python -u run_a_brs_net.py --data $dataset --model $arch --n_classes $classes -s -pc 0.01 -wd $wd --objective "at"
python -u run_a_brs_net.py --data $dataset --model $arch --n_classes $classes -s -pc 0.1 -wd $wd --objective "at"

# Evaluate white box adversarial robustness with Standard AT
python -u eval_robust_accuracy.py --data $dataset --model $arch --n_classes $classes -s -pc 0.01 --objective "at"
python -u reval_robust_accuracy.py --data $dataset --model $arch --n_classes $classes -s -pc 0.1 --objective "at"
