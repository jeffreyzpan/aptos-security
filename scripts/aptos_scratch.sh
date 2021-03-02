#!/bin/bash

set -e

ARCH=efficientnetb5
epochs=30
GPU=$1
INPUT_SIZE=$2
CONTRAST=$3

python train_models.py --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/class_contrast_${3}_${2}_${dataset}_${ARCH}_${epochs} --epochs ${epochs} --train --learning_rate 1e-4 --optimizer adam --weight_decay 1e-4 --batch_size 32 --schedule 30 --gammas 0.1 0.1 --input_size ${2} --inc_contrast ${3}

