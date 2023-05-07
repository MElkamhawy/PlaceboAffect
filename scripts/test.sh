#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PlaceboAffect

python3 src/main.py --mode test --test-data $1 --result $2 --prediction $3 --model $4 --train-data $5 --config $6
