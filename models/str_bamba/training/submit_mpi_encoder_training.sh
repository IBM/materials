#!/bin/bash -l
#PBS -l filesystems=home:grand:eagle
#PBS -l select=99:ncpus=32:system=polaris
#PBS -l place=scatter
#PBS -l walltime=06:00:00
#PBS -lselect=1:mem=384GB
#PBS -q prod
#PBS -A PROJECT
#PBS -N STR-Bamba

cd ${PBS_O_WORKDIR}

cat $PBS_NODEFILE > hostfile
sed -e 's/$/ slots=4/' -i hostfile
export DLTS_HOSTFILE=hostfile

echo "PATH=${PATH}" > .deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
echo "http_proxy=${http_proxy}" >> .deepspeed_env
echo "https_proxy=${https_proxy}" >> .deepspeed_env

cat hostfile

# Load software
# module load anaconda3
module use /soft/modulefiles
module load conda
conda activate base
source /home/user/miniforge3/bin/activate
mamba activate bamba

NNODES=`wc -l < $PBS_NODEFILE`  # num nodes
NRANKS_PER_NODE=4  # num GPUs per node
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))  # total GPUs allocated = GPUs per node * num nodes

deepspeed --hostfile=hostfile train_model_encoder.py \
        --deepspeed \
        --deepspeed_config bamba_encoder_deepspeed.json \
        --config_path './str_bamba/config/config_encoder-decoder_436M.json' \
        --pubchem_files_path './data/pubchem' \
        --polymer_files_path './data/polymer' \
        --formulation_files_path './data/formulation' \
        --data_cache_dir './data/dataset_cache' \
        --n_batch 160 \
        --max_len 1024 \
        --lr_start 1e-4 \
        --weight_decay 0 \
        --n_workers 4 \
        --max_epochs 41 \
        --checkpoint_every 100000 \
        --tokenizer_path './str_bamba/tokenizer/str_bamba_tokenizer.json' \
        --save_checkpoint_path './checkpoints' \
        --load_checkpoint_path 'Bamba_2' \
