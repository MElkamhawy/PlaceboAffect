#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PlaceboAffect
python3 main.py --mode train --train-data "../data/train/en/hateval2019_en_train.csv" --test-data "../data/dev/en/hateval2019_en_dev.csv" --result "../results/D2_scores_embd.out" --prediction "../outputs/D2/pred_en_svm_embd.txt" --model "../models/D2/svm_en_embd.pkl"
