#!/bin/bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    finetune_regression.py \
        --n_batch 8 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 16 \
        --max_epochs 100 \
        --model_path '../data/checkpoints/pretrained' \
        --ckpt_filename 'VQGAN_43.pt' \
        --data_root '../data/datasets/moleculenet/qm9' \
        --grid_path '/data_npy/qm9' \
        --dataset_name qm9 \
        --measure_name 'lumo' \
        --checkpoints_folder '../data/checkpoints/finetuned/qm9/lumo' \
        --loss_fn 'mae' \
        --target_metric 'mae' \
        --save_ckpt 1 \
        --start_seed 0 \
        --save_every_epoch 0 \
        --restart_filename '' \