#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PlaceboAffect
python3 ../src/main.py --mode train --train-data "../data/train/en/hateval2019_en_train.csv" \
--dev-data "../data/dev/en/hateval2019_en_dev.csv" \
--test-data "../data/test/en/hateval2019_en_test.csv" \
--result "../results/D4/D4_scores_gamma.out" \
--prediction "../outputs/D4/pred_gamma.txt" \
--model "../models/D4/gamma.pkl" \
--config "../src/configs/gamma.yaml"