#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PlaceboAffect
python3 ../src/main.py --mode train --train-data "../data/train/en/hateval2019_en_train.csv" --test-data "../data/dev/en/hateval2019_en_dev.csv" --result "../results/D3_scores_alpha.out" --prediction "../outputs/D3/pred_alpha.txt" --model "../models/D3/alpha.pkl" --config "../src/configs/alpha.yaml"
