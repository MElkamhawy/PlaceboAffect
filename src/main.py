#!/usr/bin/env python
import random
import sys
import numpy as np
import yaml
import nltk
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
from features import preprocess, extract_features
from modeling import classifier
from args_parser import parse_args

TARGET_LABEL_COL = 'HS'
TEXT_COL = 'text'
TRAIN_MODE = 'train'
TEST_MODE = 'test'
TRAIN_DATASET_NAME = 'train'
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


def output_evaluation_results(gold_labels, pred_labels, result_file):
    acc_hs, p_hs, r_hs, f1_hs = evaluate(gold_labels, pred_labels)
    report = classification_report(gold_labels, pred_labels)
    print(report)

    print(f'accuracy = {acc_hs:.2f}')
    print(f'precision = {p_hs:.2f}')
    print(f'recall = {r_hs:.2f}')
    print(f'f1_macro = {f1_hs:.2f}\n')

    with open(result_file, 'w') as f:
        f.write(report)
        f.write(f'\naccuracy = {acc_hs:.2f}')
        f.write(f'\nprecision = {p_hs:.2f}')
        f.write(f'\nrecall = {r_hs:.2f}')
        f.write(f'\nf1_macro = {f1_hs:.2f}')


def run(args):
    """
    This is the main function that will be running the different steps for Affect Recognition System.

    param args: instance of ParsedArgs
    return: Void
    """
    features_config = read_yaml_config(args.config_path)

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

    if args.mode == TRAIN_MODE:
        # Train Model
        parameter_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'sigmoid']}
        # parameter_grid = {"C": [0.1], "kernel": ["linear"]}
        clf = classifier.Model(parameter_grid)
        clf.fit(train_vector.vector, data_train.label, tuning=(dev_vector.vector, data_dev.label),
                algorithm=CLASSIFICATION_ALGORITHM)

        # Save Model
        clf.save_model(args.model_path)
    elif args.mode == TEST_MODE:
        clf = classifier.Model.from_file(args.model_path)
    else:
        eprint(f'Invalid Option: {args.mode}! - Only Train or Test are allowed.')

    print('Model Training Complete')

    # Predict on Test Set
    dev_pred_labels = clf.predict(dev_vector.vector)
    test_pred_labels = clf.predict(test_vector.vector)

    # Output Predictions
    dev_pred_df = data_dev.raw_df[['id']].copy()
    test_pred_df = data_test.raw_df[['id']].copy()

    dev_pred_df['pred'] = dev_pred_labels
    test_pred_df['pred'] = test_pred_labels

    dev_pred_df.to_csv(args.devtest_predictions_path, sep='\t', header=False, index=False)
    test_pred_df.to_csv(args.evaltest_predictions_path, sep='\t', header=False, index=False)

    # Evaluate classifier
    print("Devtest Results:")
    output_evaluation_results(data_dev.label, dev_pred_labels, args.devtest_result_path)

    print("Evaltest Results:")
    output_evaluation_results(data_test.label, test_pred_labels, args.evaltest_result_path)


def main():
    nltk.data.path.append("/corpora/nltk/nltk-data")
    np.random.seed(5)
    random.seed(5)
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
