#!/bin/bash
python -u ./scripts/extract_embeddings.py \
    --dataset_path '../data/datasets/moleculenet/qm9.csv' \
    --save_dataset_path '../data/embeddings/qm9_embeddings.csv' \
    --ckpt_filename 'VQGAN_43.pt' \
    --data_dir '../data/sample_data_schema' \
    --batch_size 2 \
    --num_workers 0 \