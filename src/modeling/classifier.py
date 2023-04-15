import re
import string
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class Model:
    def __init__(self, hyper_parameters=None, model=None):
        """
        Initializes the Model object.

        Args:
            hyper_parameters: A dictionary of hyperparameters for the model.
            model: A trained model object.
        """
        self.hyper_parameters = hyper_parameters
        self.model = model
        self.text = None
        self.label = None

    @classmethod
    def from_file(cls, path):
        """
        Loads a trained model object from a file.

        Args:
            path: Path to the saved model.

        Returns:
            A Model object with the loaded model.
        """
        return cls(model=joblib.load(path))

    def fit(self, text, label, cv_folds, algorithm):
        """
        Fits a model to the given data using the specified algorithm.

        Args:
            text: A numpy array of predictor variables.
            label: A numpy array of target variables.
            cv_folds: Number of cross-validation folds.
            algorithm: The name of the algorithm to use for training.

        Raises:
            ValueError: If algorithm is not 'SVM'.
        """
        if algorithm == 'SVM':
            self.model = self._fit_svm(text, label, cv_folds)

    def predict(self, text):
        """
        Uses the trained model to predict the target variable for a given set of predictor variables.

        Args:
            text: A numpy array of predictor variables.

        Returns:
            A numpy array of predicted target variables.
        """
        # Get best hyperparameters and model
        best_model = self.model.best_estimator_

        # Predict on test set with best model
        y_pred = best_model.predict(text)

        return y_pred

    def save_model(self, path):
        """
        Saves the trained model object to a file.

        Args:
            path: Path to save the model.
        """
        joblib.dump(self.model, path)

    def _fit_svm(self, text: np.ndarray, label: np.ndarray, cv_folds: int) -> SVC:
        """
        Fits an SVM classifier on the predictor variable set, text, with the target variable, label.
        Returns an SVM model object.

        Args:
            text: A numpy array of predictor variables.
            label: A numpy array of target variables.
            cv_folds: Number of cross-validation folds.

        Returns:
            An SVM model object.
        """
        svm = SVC()

        # Create GridSearchCV with k-fold cross-validation
        grid_search = GridSearchCV(svm, self.hyper_parameters, cv=cv_folds)
        grid_search.fit(text, label)

        return grid_search
