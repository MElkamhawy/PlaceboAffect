universe            = vanilla
executable          = scripts/test_primary.sh
myfiles             = ./
getenv              = true
error               = error.err
log                 = logs.log
arguments           = $(myfiles)/data/dev/en/hateval2019_en_test.csv $(myfiles)/results/D3_scores.out $(myfiles)/outputs/D3/pred_baseline.txt $(myfiles)/models/D3/baseline.pkl $(myfiles)/data/train/en/hateval2019_en_train.csv $(myfiles)/src/configs/baseline.yaml $(myfiles)/data/dev/en/hateval2019_en_dev.csv
queue
