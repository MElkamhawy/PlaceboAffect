#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh

python3 main.py --mode test --baseline --test-data "../data/dev/en/hateval2019_en_dev.csv" --result "../results/D2_scores.out" --prediction "../outputs/D2/pred_en_svm_baseline.txt" --model "../models/D2/svm_en_baseline.pkl"

