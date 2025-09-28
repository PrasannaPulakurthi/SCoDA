#!/usr/bin/env bash
set -e

python -m train --batch-size 32 --lr 0.003 --wd 5e-4 --epochs 100 --bottleneck-dim 256 --trade-off 1.0 --seed 42 --logdir runs_s2m
