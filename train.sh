#! /bin/bash

python train.py \
    --train_data_root /root/autodl-tmp/GT-RAIN/GT-RAIN_train \
    --val_data_root /root/autodl-tmp/GT-RAIN/GT-RAIN_val \
    --log_dir /root/tf-logs/ \
    --model_dir ./saved_model/ \
    --experiment ExampleExperiment \
    --train_bs 32 \
    --val_bs 1 \
    --crop 128 \
    --num_workers 8 \
    --n_epochs 100 \
    --val_gap 1 \
    --log_gap 2