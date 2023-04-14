import re
import string
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class Model:
    def __init__(self, hyper_parameters=None):
        self.hyper_parameters = hyper_parameters
        self.text = None
        self.label = None
        self.model = None

    @classmethod
    def from_file(cls, path):
        return cls(model=joblib.load(path))

    def fit(self, text, label, cv_folds):
        # Create SVM classifier
        svm = SVC()

        # Create GridSearchCV with 5-fold cross-validation
        grid_search = GridSearchCV(svm, self.hyper_parameters, cv=cv_folds)
        grid_search.fit(text, label)

        self.model = grid_search

    def predict(self, text):
        # Get best hyperparameters and model
        best_model = self.model.best_estimator_

        # Predict on test set with best model
        y_pred = best_model.predict(text)

        return y_pred

    def save_model(self, path):
        joblib.dump(self.model, path)

