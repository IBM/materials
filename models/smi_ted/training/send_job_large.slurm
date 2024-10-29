#!/bin/bash

# Example of running python script in a batch mode

#SBATCH -J smi-ted-train
#SBATCH -t 30:00:00
#SBATCH -o output_smi_ted_large_epoch40_%j.out
#SBATCH --mem=64G 
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --gpus-per-task=5
#SBATCH --cpus-per-task=20

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# Load software
# module load anaconda3
source /home/.bashrc
conda activate smi-ted-env

# Run python script
srun torchrun \
    --nnodes 10 \
    --nproc_per_node 5 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    train_model_D.py \
        --device cuda \
        --n_batch 48 \
        --n_layer 24 \
        --n_head 16 \
        --n_embd 1024 \
        --max_len 202 \
        --d_dropout 0.2 \
        --lr_start 3e-5 \
        --lr_multiplier 4 \
        --lr_decoder 3e-5 \
        --n_workers 20 \
        --max_epochs 51 \
        --gpu -1 \
        --num_nodes 1 \
        --num_feats 32 \
        --root_dir . \
        --checkpoint_every 10000 \
        --grad_acc 1 \
        --train_load 'pubchem' \
        --smi_ted_version 'v2' \
        --data_root './pubchem/pubchem_rd-canonical_smiles.smi' \
        --save_checkpoint_path './large_checkpoints' \
        --load_checkpoint_path '' \
        --rotate \
        --debug \
        --model_arch 'BERT__both_rotate' \