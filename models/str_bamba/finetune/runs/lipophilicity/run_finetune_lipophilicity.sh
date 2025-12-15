python -u ../../finetune_regression.py \
    --n_batch 32 \
    --dropout 0.1 \
    --lr_encoder 5e-6 \
    --lr_predictor 7e-6 \
    --max_epochs 100 \
    --model_path '../../str_bamba' \
    --tokenizer_filename 'str_bamba_tokenizer.json' \
    --model_config_filename 'config_encoder-decoder_436M.json' \
    --ckpt_filename 'STR-Bamba_8.pt' \
    --data_root '../../../data/datasets/moleculenet/lipophilicity' \
    --dataset_name lipophilicity \
    --measure_name 'y' \
    --inputs 'CANONICAL_SMILES' \
    --checkpoints_folder './checkpoints_lipophilicity_smiles' \
    --loss_fn 'rmse' \
    --target_metric 'rmse' \
    --save_ckpt 1 \
    --start_seed 0 \
    | tee $HOSTNAME.$(date +%F_%R).log