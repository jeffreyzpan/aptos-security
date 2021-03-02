#!/bin/bash

set -e

ARCH=efficientnetb5
GPU=$1
ATTACK=$2
CONTRAST=$3
SIZE=$4

python run_adv.py --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/class_contrast_${CONTRAST}_${SIZE}_${ARCH}_30/model_best.pth.tar --save_path /nobackup/users/jzpan/attack_logs/aptos/temp_${ARCH}_${ATTACK}_contrast_${CONTRAST}_${SIZE} --attacks ${ATTACK} --defences none
