#!/bin/bash

set -e 

ARCH=qmobilenetv2
dataset=imagenet
GPU=$1
ATTACK=$2
EPS=$3

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/qmobilenetv2_0.6.pth.tar --save_path /nobackup/users/jzpan/attack_logs/imagenet/${ARCH}_${ATTACK}_${EPS} --attacks ${ATTACK} --epsilons ${3} --linear_quantization 
