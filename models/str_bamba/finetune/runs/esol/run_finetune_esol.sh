python -u ../../finetune_regression.py \
    --n_batch 32 \
    --dropout 0.3 \
    --lr_encoder 1e-5 \
    --lr_predictor 1e-5 \
    --max_epochs 100 \
    --model_path '../../str_bamba' \
    --tokenizer_filename 'str_bamba_tokenizer.json' \
    --model_config_filename 'config_encoder-decoder_436M.json' \
    --ckpt_filename 'STR-Bamba_8.pt' \
    --data_root '../../../data/datasets/moleculenet/esol' \
    --dataset_name esol \
    --measure_name 'measured log solubility in mols per litre' \
    --inputs 'CANONICAL_SMILES' \
    --checkpoints_folder './checkpoints_esol_smiles' \
    --loss_fn 'rmse' \
    --target_metric 'rmse' \
    --save_ckpt 1 \
    --start_seed 0 \
    | tee $HOSTNAME.$(date +%F_%R).log