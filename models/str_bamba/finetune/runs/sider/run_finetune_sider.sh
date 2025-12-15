python -u ../../finetune_classification_multitask.py \
    --n_batch 16 \
    --dropout 0.1 \
    --lr_encoder 5e-6 \
    --lr_predictor 5e-6 \
    --max_epochs 20 \
    --model_path '../../str_bamba' \
    --tokenizer_filename 'str_bamba_tokenizer.json' \
    --model_config_filename 'config_encoder-decoder_436M.json' \
    --ckpt_filename 'STR-Bamba_8.pt' \
    --data_root '../../../data/datasets/moleculenet/sider' \
    --dataset_name sider \
    --inputs 'CANONICAL_SMILES' \
    --checkpoints_folder './checkpoints_sider_smiles' \
    --loss_fn 'bceloss' \
    --target_metric 'roc-auc' \
    --save_ckpt 1 \
    --start_seed 0 \
    | tee $HOSTNAME.$(date +%F_%R).log