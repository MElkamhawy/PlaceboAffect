"""
TBD

packages:
TBD
"""

import os
import time
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from features import preprocess
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
        predictions_file,
        results_file,
        model_file,
        empath,
        train
):
    # get start time
    start = time.time()
    print('Started Data Preprocessing')
    # Load Data from CSV and store as preprocess.Data object
    data_train = preprocess.Data.from_csv(training_data_file, name="train")
    data_dev = preprocess.Data.from_csv(development_data_file, name="dev")

    # Preprocess Data
    data_train.process(
        text_name="text", target_name="HS", empath=empath)
    data_dev.process(
        text_name="text", target_name="HS", vectorizer=data_train.vectorizer, empath=empath
    )
    end = time.time()
    print(f'Completed Data Preprocessing {end - start}')

    print('Started Model Training')

    # Train Model
    # parameter_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'sigmoid']}
    parameter_grid = {"C": [0.1], "kernel": ["linear"]}
    clf = classifier.Model(parameter_grid)
    clf.fit(data_train.vector, data_train.label, cv_folds=2, algorithm='SVM') #changed from text to vector
    end = time.time()
    print(f'Completed Model Training {end - start}')
    print(clf.model.best_estimator_)

    # Save Model
    clf.save_model(model_file)


    # Predict on Dev Set
    print('Started Model Inference')
    pred_labels = clf.predict(data_dev.vector) #same
    end = time.time()
    print(f'Completed Model Inference {end - start}')
    print(list(pred_labels))

    # # Output Predictions
    output_lines(list(pred_labels), predictions_file)

    # Evaluate classifier
    accuracy = accuracy_score(data_dev.label, pred_labels)
    precision = precision_score(data_dev.label, pred_labels, average="binary")
    recall = recall_score(data_dev.label, pred_labels, average="binary")
    f1 = f1_score(data_dev.label, pred_labels, average="binary")

    print("Accuracy: {:.2f}".format(accuracy))
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("F1-score: {:.2f}".format(f1))


    # Output Evaluation Results


    end = time.time()
    print(f"total time: {end - start}")

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
    predictions_file = "../outputs/pred_en_svm_no_empath.txt"
    results_file = "../outputs/res.txt"
    model_file = "../models/svm_no_empath.pkl"
    empath = False
    train = True
    main(
        training_data_file=training_data_file,
        development_data_file=development_data_file,
        test_data_file=test_data_file,
        predictions_file=predictions_file,
        results_file=results_file,
        model_file=model_file,
        empath=empath,
        train=train
    )
