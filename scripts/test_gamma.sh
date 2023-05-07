#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PlaceboAffect
python3 ../src/main.py --mode test --train-data "../data/train/en/hateval2019_en_train.csv" --test-data "../data/dev/en/hateval2019_en_dev.csv" --result "../results/D3_scores_gamma.out" --prediction "../outputs/D3/pred_gamma.txt" --model "../models/D3/gamma.pkl" --config "../src/configs/gamma.yaml"
