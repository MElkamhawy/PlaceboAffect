#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PlaceboAffect
python3 ../src/main.py --mode test --train-data "../data/train/en/hateval2019_en_train.csv" \
--dev-data "../data/dev/en/hateval2019_en_dev.csv" \
--test-data "../data/test/en/hateval2019_en_test.csv" \
--result "../results/D4/D4_scores_beta.out" \
--prediction "../outputs/D4/pred_beta.txt" \
--model "../models/D4/beta.pkl" \
--config "../src/configs/beta.yaml"