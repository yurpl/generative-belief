#!/usr/bin/env bash

python main.py --task aste \
            --dataset modal \
            --model_name_or_path google/long-t5-local-base \
            --paradigm extraction \
            --n_gpu 3 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 4 \
            --learning_rate 3e-4  \
            --num_train_epochs 5 \
            --seed 7
