#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PlaceboAffect
python3 ../src/main.py --mode train --task adaptation --train-data "../data/train/es2en/hateval2019_es2en_train.csv" \
--dev-data "../data/dev/es2en/hateval2019_es2en_dev.csv" \
--test-data "../data/test/es2en/hateval2019_es2en_test.csv" \
--result "../results/D4/adaptation/D4_scores.out" \
--prediction "../outputs/D4/pred_baseline.txt" \
--model "../models/D4/baseline.pkl" \
--config "../src/configs/baseline.yaml"