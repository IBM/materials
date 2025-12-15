#!/bin/bash
python -u ./scripts/extract_embeddings.py \
    --arch clip \
    --dataset_path '../data/datasets/moleculenet/train_toy.csv' \
    --save_dataset_path '../data/embeddings/clip_qm9_embeddings_train.csv' \
    --ckpt_filename 'SMILESDFT-CLIP_96.pt' \
    --data_dir '../data/qm9_files' \
    --batch_size 32 \
    --num_workers 2 \