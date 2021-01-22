#!/bin/bash

set -e 

ARCH=mobilenetv2
dataset=imagenet
GPU=$1
ATTACK=$2
EPS=$3

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/mobilenetv2.pth.tar --save_path /nobackup/users/jzpan/attack_logs/imagenet/${ARCH}_${ATTACK}_${EPS} --attacks ${ATTACK} --defences jpeg tvm --epsilons ${3}
