#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PlaceboAffect
python3 ../src/main.py --mode test --train-data "../data/train/en/hateval2019_en_train.csv" \
--dev-data "../data/dev/en/hateval2019_en_dev.csv" \
--test-data "../data/test/en/hateval2019_en_test.csv" \
--result "../results/D4/D4_scores_alpha.out" \
--prediction "../outputs/D4/pred_alpha.txt" \
--model "../models/D4/alpha.pkl" \
--config "../src/configs/alpha.yaml"