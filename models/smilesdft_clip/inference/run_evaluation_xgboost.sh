#!/bin/bash
python -u ./scripts/evaluate_embeddings_xgboost.py \
    --task_name r2 \
    --embeddings_path '../data/embeddings/clip_qm9_embeddings_' \