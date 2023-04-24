universe            = vanilla
executable          = src/test.sh
myfiles             = ./
getenv              = true
error               = error.err
log                 = logs.log
arguments           = $(myfiles)/data/dev/en/hateval2019_en_dev.csv $(myfiles)/results/D2_scores.out $(myfiles)/outputs/D2/pred_en_svm_baseline.txt $(myfiles)/models/D2/svm_en_baseline.pkl $(myfiles)/data/train/en/hateval2019_en_train.csv
queue
