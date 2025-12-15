#!/bin/bash -l
mpirun -np 4 \
    -npernode 4 \
    -x PATH \
    --oversubscribe \
    python main_clip.py --config config_clip.yaml