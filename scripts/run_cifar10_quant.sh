#!/bin/bash

set -e 

ARCH=cifar_qmobilenetv2
dataset=cifar10
train_epochs=350
GPU=$1
RATIO=$2
ATTACK=$3

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/${dataset}_${ARCH}_${train_epochs}_ratio${RATIO}/model_best.pth.tar --save_path /nobackup/users/jzpan/attack_logs/cifar10/${ARCH}_${ATTACK}_${RATIO} --attacks ${3} --defences jpeg tvm --linear_quantization
