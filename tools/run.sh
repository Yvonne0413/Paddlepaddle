#!/usr/bin/env bash

python3.7 -m paddle.distributed.launch \
    --gpus="0" \
    tools/train.py \
        -c ./configs/ResNet/ResNet50.yaml \
        -o print_interval=10
