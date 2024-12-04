#!/bin/bash

#SBATCH -J smi-ssed
#SBATCH -t 60:00:00
#SBATCH -o output_smi-ssed_training_epoch130_%j.out
#SBATCH --mem=64G 
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=5
#SBATCH --cpus-per-task=8

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# Load software
# module load anaconda3
source /home/USER/.bashrc
conda activate smi-ssed-env

srun torchrun \
    --nnodes 4 \
    --nproc_per_node 5 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    train_model_D.py \
        --device cuda \
        --n_batch 128 \
        --n_layer 24 \
        --n_embd 768 \
        --d_state 16 \
        --d_conv 4 \
        --expand_factor 2 \
        --dt_rank 'auto' \
        --dt_min 0.001 \
        --dt_max 0.1 \
        --dt_init 'random' \
        --dt_scale 1.0 \
        --dt_init_floor 1e-4 \
        --conv_bias 1 \
        --bias 0 \
        --max_len 202 \
        --lr_start 3e-5 \
        --lr_multiplier 1 \
        --lr_decoder 3e-5 \
        --n_workers 8 \
        --max_epochs 131 \
        --checkpoint_every 1 \
        --train_load 'pubchem' \
        --data_root './pubchem/pubchem_rd-canonical_smiles.smi' \
        --save_checkpoint_path './checkpoints' \
        --load_checkpoint_path '' \
