python -u ../../finetune_regression.py \
    --n_batch 16 \
    --dropout 0.1 \
    --lr_encoder 3e-6 \
    --lr_predictor 3e-6 \
    --max_epochs 1000 \
    --model_path '../../str_bamba' \
    --tokenizer_filename 'str_bamba_tokenizer.json' \
    --model_config_filename 'config_encoder-decoder_436M.json' \
    --ckpt_filename 'STR-Bamba_8.pt' \
    --data_root '../../../data/datasets/polymer/MIT-Polymer-Electrolyte-Conductivity' \
    --dataset_name 'MIT-Polymer-Electrolyte-Conductivity' \
    --measure_name 'conductivity' \
    --inputs 'POLYMER SMILES' \
    --checkpoints_folder './checkpoints_mit-polymer-electrolyte-conductivity' \
    --loss_fn 'mae' \
    --target_metric 'mae' \
    --save_ckpt 1 \
    --start_seed 0 \
    | tee $HOSTNAME.$(date +%F_%R).log