#!/bin/bash
python -u ./scripts/retrieval_query_script.py \
    --arch clip \
    --dataset_path '../data/catalog/catalog_retrieval.csv' \
    --save_dataset_path '../data/retrieval/results_clip_epoch=96.csv' \
    --data_dir '../data/retrieval_files' \
    --ckpt_filename 'SMILESDFT-CLIP_96.pt' \
    --topk 50 \