python -u ../../finetune_regression.py \
    --n_batch 128 \
    --dropout 0.1 \
    --lr_encoder 1e-5 \
    --lr_predictor 1e-5 \
    --max_epochs 500 \
    --model_path '../../str_bamba' \
    --tokenizer_filename 'str_bamba_tokenizer.json' \
    --model_config_filename 'config_encoder-decoder_436M.json' \
    --ckpt_filename 'STR-Bamba_8.pt' \
    --data_root '../../../data/datasets/moleculenet/qm8' \
    --dataset_name qm8 \
    --measure_name 'f2-CC2' \
    --inputs 'CANONICAL_SMILES' \
    --checkpoints_folder './checkpoints_QM8-f2-CC2_smiles' \
    --loss_fn 'mae' \
    --target_metric 'mae' \
    --save_ckpt 1 \
    --start_seed 0 \
    | tee $HOSTNAME.$(date +%F_%R).log