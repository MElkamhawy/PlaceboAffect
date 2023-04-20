"""
TBD
packages:
TBD
"""

import os
import time
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from features import preprocess, extract_features
from modeling import classifier

from argparse import ArgumentParser


def output_lines(lines, path):
    with open(path, 'w+') as f:
        for line in lines:
            f.write(str(line) + '\n')


def main(
        training_data_file,
        development_data_file,
        test_data_file,
        empath,
        train,
        strategy,
        predictions_file,
        results_file,
        model_file
):
    # Load Data from CSV and store as preprocess.Data object
    data_train = preprocess.Data.from_csv(training_data_file, name="train")
    data_dev = preprocess.Data.from_csv(development_data_file, name="dev")

    # Preprocess Data
    data_train.process(text_name="text", target_name="HS")
    data_dev.process(text_name="text", target_name="HS")
    print('preprocessing complete')

    # Extract Features from Data
    train_vector = extract_features.Vector(name="train", text=data_train.text)
    dev_vector = extract_features.Vector(name="dev", text=data_dev.text)
    train_vector.process_features(strategy=strategy, empath=empath)
    dev_vector.process_features(strategy=strategy, vectorizer=train_vector.vectorizer, empath=empath)
    print('feature extraction complete')

    if train:
        # Train Model
        parameter_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'sigmoid']}
        # parameter_grid = {"C": [0.1], "kernel": ["linear"]}
        clf = classifier.Model(parameter_grid)
        clf.fit(train_vector.vector, data_train.label, cv_folds=5, algorithm='SVM')

        # Save Model
        clf.save_model(model_file)
        print('training complete')

    else:
        clf = classifier.Model.from_file(model_file)

    # Predict on Dev Set
    pred_labels = clf.predict(dev_vector.vector)

    # Output Predictions
    output_lines(list(pred_labels), predictions_file)

    # Evaluate classifier
    accuracy = accuracy_score(data_dev.label, pred_labels)
    precision = precision_score(data_dev.label, pred_labels)
    recall = recall_score(data_dev.label, pred_labels)
    f1 = f1_score(data_dev.label, pred_labels, average="macro")

    print(f'accuracy = {accuracy:.2f}')
    print(f'precision = {precision:.2f}')
    print(f'recall = {recall:.2f}')
    print(f'f1_macro = {f1:.2f}')

    report = classification_report(data_dev.label, pred_labels)
    print(report)

    # Output Evaluation Results
    # TODO Mohamed to replace eval script provided by shared task
    with open(results_file, 'w') as f:
        f.write(report)
        f.write(f'\naccuracy = {accuracy:.2f}')
        f.write(f'\nprecision = {precision:.2f}')
        f.write(f'\nrecall = {recall:.2f}')
        f.write(f'\nf1_macro = {f1:.2f}')


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
    # arg_parser.add_argument('--empath', action='store_true')
    # args = arg_parser.parse_args()
    training_data_file = "../data/train/en/hateval2019_en_train.csv"
    development_data_file = "../data/dev/en/hateval2019_en_dev.csv"
    test_data_file = "../data/test/en/hateval2019_en_test.csv"
    empath = False
    train = True
    feature_strategy = 'w2v'
    predictions_file = f"../outputs/pred_en_{feature_strategy}.txt"
    results_file = f"../results/res_en_{feature_strategy}.txt"
    model_file = f"../models/svm_en_{feature_strategy}.pkl"

    main(
        training_data_file=training_data_file,
        development_data_file=development_data_file,
        test_data_file=test_data_file,
        empath=empath,
        train=train,
        strategy=feature_strategy,
        predictions_file=predictions_file,
        results_file=results_file,
        model_file=model_file
    )
