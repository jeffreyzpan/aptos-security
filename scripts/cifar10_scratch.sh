#!/bin/bash

set -e

ARCH=cifar_mobilenetv2
dataset=cifar10
epochs=350
GPU=$1

python train_models.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/test+${dataset}_${ARCH}_${epochs} --epochs ${epochs} --train --learning_rate 0.1 --optimizer sgd --schedule 150 250  --gammas 0.1 0.1 --weight_decay 4e-5


