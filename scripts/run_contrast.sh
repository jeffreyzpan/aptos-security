#!/bin/bash

set -e

ARCH=efficientnetb5
GPU=$1
ATTACK=$2
CONTRAST=$3
SIZE=$4

python optimize_contrast.py --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/class_contrast_${CONTRAST}_${SIZE}_${ARCH}_30/model_best.pth.tar --save_path /nobackup/users/jzpan/contrast_logs/aptos/${ARCH}_${ATTACK}_contrast_${CONTRAST}_${SIZE} --attacks ${ATTACK}
