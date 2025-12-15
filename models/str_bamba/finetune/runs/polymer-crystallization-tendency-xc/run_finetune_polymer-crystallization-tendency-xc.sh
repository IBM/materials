python -u ../../finetune_regression.py \
    --n_batch 8 \
    --dropout 0.1 \
    --lr_encoder 5e-6 \
    --lr_predictor 5e-6 \
    --max_epochs 1000 \
    --model_path '../../str_bamba' \
    --tokenizer_filename 'str_bamba_tokenizer.json' \
    --model_config_filename 'config_encoder-decoder_436M.json' \
    --ckpt_filename 'STR-Bamba_8.pt' \
    --data_root '../../../data/datasets/polymer/Polymer-Crystallization-Tendency-(Xc)' \
    --dataset_name Polymer-Crystallization-Tendency-Xc \
    --measure_name 'Xc' \
    --inputs 'POLYMER SMILES' \
    --checkpoints_folder './checkpoints_polymer-crystallization-tendency-xc' \
    --loss_fn 'rmse' \
    --target_metric 'rmse' \
    --save_ckpt 1 \
    --start_seed 0 \
    | tee $HOSTNAME.$(date +%F_%R).log