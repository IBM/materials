python -u ../../finetune_regression.py \
    --n_batch 16 \
    --dropout 0.1 \
    --lr_encoder 7e-7 \
    --lr_predictor 7e-7 \
    --max_epochs 200 \
    --model_path '../../str_bamba' \
    --tokenizer_filename 'str_bamba_tokenizer.json' \
    --model_config_filename 'config_encoder-decoder_436M.json' \
    --ckpt_filename 'STR-Bamba_8.pt' \
    --data_root '../../../data/datasets/polymer/CalTech' \
    --dataset_name CalTech \
    --measure_name 'O2' \
    --inputs 'POLYMER SMILES' \
    --checkpoints_folder './checkpoints_caltech_O2' \
    --loss_fn 'rmse' \
    --target_metric 'rmse' \
    --save_ckpt 1 \
    --start_seed 0 \
    | tee $HOSTNAME.$(date +%F_%R).log