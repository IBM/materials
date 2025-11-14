#!/bin/bash -l

# Load environment
conda activate 3dvqgan-env

export LOGLEVEL=INFO
export PYTHONPATH=$PWD
export HYDRA_FULL_ERROR=1

# MPI example w/ 4 MPI ranks per node w/ threads spread evenly across cores (1 thread per core)
NNODES=1
NRANKS_PER_NODE=4

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}"

mpirun -np ${NTOTRANKS} \
    -npernode ${NRANKS_PER_NODE} \
    -x PATH \
    --oversubscribe \
    python train_vqgan_DDP.py \
        dataset=default \
        dataset.root_dir='../data/3d_grids_sample' \
        model=vq_gan_3d  \
        model.default_root_dir_postfix='data_fm_qm9'  \
        model.precision=16  \
        model.embedding_dim=256  \
        model.n_hiddens=16 \
        model.downsample=[4,4,4] \
        model.num_workers=32 \
        model.gradient_clip_val=1.0 \
        model.lr=3e-4 \
        model.discriminator_iter_start=450 \
        model.perceptual_weight=4 \
        model.image_gan_weight=1 \
        model.gan_feat_weight=4 \
        model.batch_size=1 \
        model.n_codes=16384 \
        model.accumulate_grad_batches=1 \
        model.internal_resolution=128 \
        model.checkpoint_every=1000 \
        model.save_checkpoint_path='./checkpoints' \
        model.resume_from_checkpoint='' \
        model.max_epochs=100 \