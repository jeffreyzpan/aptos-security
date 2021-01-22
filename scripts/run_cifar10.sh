#!/bin/bash

set -e 

ARCH=cifar_mobilenetv2
dataset=cifar10
train_epochs=350
GPU=$1
ATTACK=$2

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/${dataset}_${ARCH}_${train_epochs}/model_best.pth.tar --save_path /nobackup/users/jzpan/attack_logs/cifar10/${ARCH}_${ATTACK} --attacks ${2} --defences jpeg tvm
