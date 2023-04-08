"""
TBD

packages:
TBD
"""

import os
import pandas as pd
from pathlib import Path

from src.features import preprocess

from argparse import ArgumentParser


def main(training_data_file, development_data_file, test_data_file, predictions_file, results_file):
    train = preprocess.Data.from_csv(training_data_file, name='train')
    print(train.raw_df.shape)
    train.process(text_name='text', target_name='HS')
    print(train.label.shape)
    print(train.text.shape)




if __name__ == "__main__":
    """
    TBD
    """
    # arg_parser = ArgumentParser()
    # arg_parser.add_argument('--training_data_file', type=str, required=True)
    # arg_parser.add_argument('--development_data_file', type=str, required=True)
    # arg_parser.add_argument('--test_data_file', type=str, required=True)
    # arg_parser.add_argument('--predictions_file', type=str, required=True)
    # arg_parser.add_argument('--results_file', type=str, required=True)
    # args = arg_parser.parse_args()
    training_data_file = '../data/train/en/hateval2019_en_train.csv'
    development_data_file = '../data/dev/en/hateval2019_en_dev.csv'
    test_data_file = '../data/test/en/hateval2019_en_test.csv'
    predictions_file = '../outputs/pred_en.txt'
    results_file = '../outputs/res.txt'
    main(
        training_data_file=training_data_file
        , development_data_file=development_data_file
        , test_data_file=test_data_file
        , predictions_file=predictions_file
        , results_file=results_file
    )
