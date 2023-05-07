import joblib
import numpy as np

from sklearn.metrics import precision_recall_fscore_support

from sklearn.svm import SVC
from itertools import product


class Model:
    def __init__(self, hyper_parameters=None, model=None):
        """
        Initializes the Model object.

        Args:
            hyper_parameters: A dictionary of hyperparameters for the model. Default is None.
            model: A trained model object. Default is None.
        """
        self.hyper_parameters = hyper_parameters
        self.model = model
        self.text = None
        self.label = None
        self.algorithm = None

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

    def fit(self, text, label, tuning, algorithm):
        """
        Fits a model to the given data using the specified algorithm.

        Args:
            text: A numpy array of predictor variables.
            label: A numpy array of target variables.
            tuning: Number of cross-validation folds or tuple of (Vector.vector, Data.label)
            algorithm: The name of the algorithm to use for training.

        Raises:
            ValueError: If algorithm is not 'SVM'.
        """
        self.algorithm = algorithm

        if algorithm == 'SVM':
            self.model = self._tune_sklearn(text=text, label=label, tuning=tuning)
            print(self.model.get_params())
        else:
            ValueError('Model must be SVM')

    def predict(self, text):
        """
        Uses the trained model to predict the target variable for a given set of predictor variables.

        Args:
            text: A numpy array of predictor variables.

        Returns:
            A numpy array of predicted target variables.
        """
        # Predict on test set with  model
        y_pred = self.model.predict(text)

        return y_pred

    def save_model(self, path):
        """
        Saves the trained model object to a file.

        Args:
            path: Path to save the model.
        """
        joblib.dump(self.model, path)

    def _tune_sklearn(self, text, label, tuning):
        """
        Performs hyperparameter tuning on an SVM model using a grid search.

        Args:
            text: A numpy array of predictor variables.
            label: A numpy array of target variables.
            tuning: Number of cross-validation folds or tuple of (Vector.vector, Data.label)

        Returns:
            A trained SVM model object with the best hyperparameters found by the grid search.
        """
        # Convert parameter dict into list of parameter combinations
        param_dicts = self._get_param_combinations(self.hyper_parameters)

        # fit sklearn model
        model = self._grid_search(params=param_dicts, text=text, label=label, tuning=tuning)

        return model

    def _grid_search(self, params, text, label, tuning):
        """
        Performs a grid search on an SVM model to find the best hyperparameters.

        Args:
            params: A list of dictionaries, where each dictionary represents a unique combination of hyperparameters
                to test.
            text: A numpy array or csr_matrix of predictor variables.
            label: A numpy array of target variables.
            tuning: Number of cross-validation folds or tuple of (Vector.vector, Data.label)

        Returns:
            A trained model object with the best hyperparameters found by the grid search.
        """
        # compare each fit to the best model and best f1
        best_model = None
        best_f1 = 0

        # loop through all combinations of parameters,
        for combination in params:
            model = self._fit_sklearn(hyp_params=combination, text=text, label=label)
            preds = model.predict(tuning[0])
            p_hs, r_hs, f1_hs, support = precision_recall_fscore_support(preds, tuning[1], average="macro")
            if f1_hs > best_f1:
                best_f1 = f1_hs
                best_model = model

        return best_model

    def _fit_sklearn(self, hyp_params, text, label):
        """
        Fits an sklearn model using the given hyperparameters.

        Args
        """
        if self.algorithm == 'SVM':
            model = SVC(C=hyp_params['C'], kernel=hyp_params['kernel'])
            model.fit(text, label)
        return model

    @staticmethod
    def _get_param_combinations(param_grid):
        """
        Returns a list of every unique combination of hyperparameters in the given parameter grid,
        where each combination consists of one value for each hyperparameter.

        Args:
            param_grid (dict): A dictionary where keys are hyperparameter names and values are lists of possible
            values for each hyperparameter.

        Returns:
            list: A list of dictionaries, where each dictionary represents a unique combination of hyperparameters
            consisting of one value for each hyperparameter in the given `param_grid`.
        """
        # Get the possible values for each hyperparameter in the `param_grid` dictionary
        param_values = list(param_grid.values())

        # Get the keys for each hyperparameter in the `param_grid` dictionary
        param_keys = list(param_grid.keys())

        # Get every combination of hyperparameter values using itertools.product
        param_combinations = list(product(*param_values))

        # Create a list of dictionaries, where each dictionary represents a unique combination of hyperparameters
        # consisting of one value for each hyperparameter in the `param_grid` dictionary
        param_dicts = [dict(zip(param_keys, combination)) for combination in param_combinations]

        return param_dicts
