#!/bin/bash -l
mpirun -np 4 \
    -npernode 4 \
    -x PATH \
    --oversubscribe \
    python main_siglip.py --config config_siglip.yaml
