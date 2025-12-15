python -u ../../finetune_regression.py \
    --n_batch 8 \
    --dropout 0.1 \
    --lr_encoder 3e-5 \
    --lr_predictor 3e-5 \
    --max_epochs 1000 \
    --model_path '../../str_bamba' \
    --tokenizer_filename 'str_bamba_tokenizer.json' \
    --model_config_filename 'config_encoder-decoder_436M.json' \
    --ckpt_filename 'STR-Bamba_8.pt' \
    --data_root '../../../data/datasets/polymer/19F-MRI-Constrast' \
    --dataset_name 19F-MRI-Constrast \
    --measure_name '19F NMR SN Ratio' \
    --inputs 'POLYMER SMILES' \
    --checkpoints_folder './checkpoints_19f-mri-constrast' \
    --loss_fn 'mae' \
    --target_metric 'mae' \
    --save_ckpt 1 \
    --start_seed 0 \
    | tee $HOSTNAME.$(date +%F_%R).log