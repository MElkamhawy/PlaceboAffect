#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda remove -n PlaceboAffect --all
conda env create -f requirements.yml
conda activate PlaceboAffect
python -m nltk.downloader all
python -m spacy download en_core_web_sm
