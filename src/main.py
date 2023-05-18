#!/usr/bin/env python

import os
import random
import sys
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import yaml

import nltk
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support

from features import preprocess, extract_features
from modeling import classifier

TARGET_LABEL_COL = 'HS'
TEXT_COL = 'text'
TRAIN_MODE = 'train'
TEST_MODE = 'test'
TRAIN_DATASET_NAME = 'train'
TRAIN_DATASET_EN_PATH = '../data/train/en/hateval2019_en_train.csv'
DEV_DATASET_NAME = 'dev'
TEST_DATASET_NAME = 'test'
CLASSIFICATION_ALGORITHM = 'SVM'
PRIMARY_TASK_LANGUAGE = 'en'
ADAPTATION_TASK_LANGUAGE = 'es'


def eprint(*args, **kwargs):
    """
    Print to stderr

    :return: Void
    """
    print(*args, file=sys.stderr, **kwargs)


def read_yaml_config(filename):
    """
    Read yaml config file.

    param filename: path to yaml file (str)
    return: config (dict)
    """
    with open(filename, 'r') as file:
        return yaml.safe_load(file)


def output_lines(lines, path):
    """
    Print results to file.

    :param lines: lines to print (list)
    :param path: path to file (str)
    :return: Void
    """
    with open(path, 'w+') as f:
        for line in lines:
            f.write(str(line) + '\n')


def create_arg_parser():
    argument_parser = ArgumentParser(description='D3 for PlaceboAffect - Course Ling 573.')
    argument_parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                                 help='Train or test the model')
    argument_parser.add_argument('--task', type=str, choices=['primary', 'adaptation'], default='primary',
                                 help='Task to perform (primary or adaptation)')
    argument_parser.add_argument('--train-data', help='Training Data File Path',
                                 default='../data/train/en/hateval2019_en_train.csv')
    argument_parser.add_argument('--dev-data', help='Development Data File Path',
                                 default='../data/dev/en/hateval2019_en_dev.csv')
    argument_parser.add_argument('--test-data', help='Test Data File Path',
                                 default='../data/test/en/hateval2019_en_test.csv')
    argument_parser.add_argument('--model', help='Model File Path', default='../models/D4/svm_{sys}.pkl')
    argument_parser.add_argument('--config', help='Config File Path', default='configs/baseline.yaml')
    argument_parser.add_argument('--result', help='Output File Path', default='../results/D4/D4_scores{sys}.out')
    argument_parser.add_argument('--predictions', help='Predictions File Path',
                                 default='../outputs/D4/pred_{sys}.txt')
    return argument_parser


def evaluate(gold, pred):
    """
    Evaluate the performance of the model

    :param gold: gold labels array (list)
    :param pred: prediction labaels array (list)
    :return: accuracy, precision, recall, f1 (float)
    """

    # Check length files
    if len(pred) != len(gold):
        eprint('Prediction and gold data have different number of lines.')
        sys.exit(1)

    # Compute Performance Measures HS
    acc_hs = accuracy_score(pred, gold)
    p_hs, r_hs, f1_hs, support = precision_recall_fscore_support(pred, gold, average="macro")

    return acc_hs, p_hs, r_hs, f1_hs


def run(mode, task, training_data_file, dev_data_file, test_data_file, result_file, predictions_file, model_file,
        config):
    """
    This is the main function that will be running the different steps for Affect Recognition System.

    param mode: train or test (str)
    param training_data_file: path to training data file (str)
    param dev_data_file: path to dev data file (str)
    param test_data_file: path to test data file (str)
    param result_file: path to result file (str)
    param predictions_file: path to predictions file (str)
    param model_file: path to model file (str)
    param config: path to config file (str)
    return: Void
    """
    # Validations
    if mode is TRAIN_MODE and not os.path.exists(training_data_file):
        eprint("Training data file does not exist!")
        sys.exit(1)

    if not os.path.exists(dev_data_file):
        eprint("Dev data file does not exist!")
        sys.exit(1)

    if not os.path.exists(test_data_file):
        eprint("Test data file does not exist!")
        sys.exit(1)

    if mode is TEST_MODE and not os.path.exists(model_file):
        eprint("Model file does not exist!")
        sys.exit(1)

    if not os.path.exists(config):
        eprint("Config file does not exist!")
        sys.exit(1)
    features_config = read_yaml_config(config)

    # Adjust File Paths
    config_name = os.path.basename(config).split('.')[0]
    model_file = model_file.format(sys=config_name)
    predictions_file = predictions_file.format(sys=config_name)
    result_file = result_file.format(sys='_' + config_name if config_name != 'baseline' else "")

    # Load Data from CSV and store as preprocess.Data object
    if task == 'primary':
        data_train = preprocess.Data.from_csv(training_data_file, name=TRAIN_DATASET_NAME)
    elif task == 'adaptation':
        train_concat_df = pd.concat([pd.read_csv(training_data_file), pd.read_csv(TRAIN_DATASET_EN_PATH)])
        data_train = preprocess.Data(raw_df=train_concat_df, name=TRAIN_DATASET_NAME)
    else:
        eprint(f'Invalid Option: {task}! - Only primary or adaptation are allowed.')

    data_dev = preprocess.Data.from_csv(dev_data_file, name=DEV_DATASET_NAME)
    data_test = preprocess.Data.from_csv(test_data_file, name=TEST_DATASET_NAME)
    print('Data Load Complete')

    # Preprocess Data
    data_train.process(text_name=TEXT_COL, target_name=TARGET_LABEL_COL)
    data_dev.process(text_name=TEXT_COL, target_name=TARGET_LABEL_COL)
    data_test.process(text_name=TEXT_COL, target_name=TARGET_LABEL_COL)
    print('Data Preprocessing Complete')

    # Extract Features from Data

    train_vector = extract_features.Vector(name=TRAIN_DATASET_NAME, text=data_train.text, config=features_config)
    dev_vector = extract_features.Vector(name=DEV_DATASET_NAME, text=data_dev.text, config=features_config)
    test_vector = extract_features.Vector(name=TEST_DATASET_NAME, text=data_test.text, config=features_config)

    train_vector.process_features()
    dev_vector.process_features(vectorizer=train_vector.vectorizer)
    test_vector.process_features(vectorizer=train_vector.vectorizer)

    clf = None
    print('Feature Extraction Complete')

    if mode == TRAIN_MODE:
        # Train Model
        # parameter_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'sigmoid']}
        parameter_grid = {"C": [0.1], "kernel": ["linear"]}
        clf = classifier.Model(parameter_grid)
        clf.fit(train_vector.vector, data_train.label, tuning=(dev_vector.vector, data_dev.label),
                algorithm=CLASSIFICATION_ALGORITHM)

        # Save Model
        clf.save_model(model_file)
    elif mode == TEST_MODE:
        clf = classifier.Model.from_file(model_file)
    else:
        eprint(f'Invalid Option: {mode}! - Only Train or Test are allowed.')

    print('Model Training Complete')

    # Predict on Test Set
    pred_labels = clf.predict(test_vector.vector)

    # Output Predictions
    pred_df = data_test.raw_df[['id']].copy()
    pred_df['pred'] = pred_labels
    pred_df.to_csv(predictions_file, sep='\t', header=False, index=False)

    # Evaluate classifier
    acc_hs, p_hs, r_hs, f1_hs = evaluate(data_test.label, pred_labels)
    report = classification_report(data_test.label, pred_labels)
    print(report)

    print(f'accuracy = {acc_hs:.2f}')
    print(f'precision = {p_hs:.2f}')
    print(f'recall = {r_hs:.2f}')
    print(f'f1_macro = {f1_hs:.2f}')

    with open(result_file, 'w') as f:
        f.write(report)
        f.write(f'\naccuracy = {acc_hs:.2f}')
        f.write(f'\nprecision = {p_hs:.2f}')
        f.write(f'\nrecall = {r_hs:.2f}')
        f.write(f'\nf1_macro = {f1_hs:.2f}')


def main():
    nltk.data.path.append("/corpora/nltk/nltk-data")
    np.random.seed(5)
    random.seed(5)
    args = create_arg_parser().parse_args()
    run(args.mode, args.task, args.train_data, args.dev_data, args.test_data, args.result, args.predictions, args.model,
        args.config)


if __name__ == '__main__':
    main()
