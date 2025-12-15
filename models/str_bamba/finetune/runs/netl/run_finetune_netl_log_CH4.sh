python -u ../../finetune_regression.py \
    --n_batch 8 \
    --dropout 0.1 \
    --lr_encoder 1e-5 \
    --lr_predictor 1e-5 \
    --max_epochs 100 \
    --model_path '../../str_bamba' \
    --tokenizer_filename 'str_bamba_tokenizer.json' \
    --model_config_filename 'config_encoder-decoder_436M.json' \
    --ckpt_filename 'STR-Bamba_8.pt' \
    --data_root '../../../data/datasets/polymer/NETL' \
    --dataset_name NETL \
    --measure_name 'log CH4 (barrer)' \
    --inputs 'POLYMER SMILES' \
    --checkpoints_folder './checkpoints_netl_log_CH4' \
    --loss_fn 'mae' \
    --target_metric 'mae' \
    --save_ckpt 1 \
    --start_seed 40 \
    | tee $HOSTNAME.$(date +%F_%R).log