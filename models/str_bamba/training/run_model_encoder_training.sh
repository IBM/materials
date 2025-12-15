#!/bin/bash
deepspeed --num_nodes 1 --num_gpus 4  \
    train_model_encoder.py \
        --deepspeed \
        --deepspeed_config bamba_encoder_deepspeed.json \
        --config_path './str_bamba/config/config_encoder-decoder_436M.json' \
        --pubchem_files_path './data/pubchem_100k' \
        --polymer_files_path './data/polymer_100k' \
        --formulation_files_path './data/formulation' \
        --data_cache_dir './data/dataset_cache' \
        --n_batch 160 \
        --max_len 1024 \
        --lr_start 1e-4 \
        --weight_decay 0 \
        --n_workers 4 \
        --max_epochs 41 \
        --checkpoint_every 100000 \
        --tokenizer_path './str_bamba/tokenizer/str_bamba_tokenizer.json' \
        --save_checkpoint_path './checkpoints' \