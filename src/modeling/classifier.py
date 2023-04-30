import joblib
import numpy as np
import scipy.sparse as sp

from sklearn.model_selection import GridSearchCV, PredefinedSplit, ShuffleSplit
from sklearn.metrics import precision_recall_fscore_support

from sklearn.svm import SVC
from itertools import product


# from src.features.extract_features import Vector

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
        if algorithm == 'SVM':
            # self.model = self._fit_svm(text, label, tuning)
            self.model = self._tune_sklearn(sklearn_model=SVC(), text=text, label=label, tuning=tuning)

    def _tune_sklearn(self, sklearn_model, text, label, tuning):
        # Convert parameter dict into list of parameter combinations
        param_dicts = self._get_param_combinations(self.hyper_parameters)

        # fit sklearn model



        return model

    def _grid_search(self, model, params, text, label, tuning):
        # compare each fit to the best model and best f1
        best_model = None
        best_f1 = 0

        # loop through all combinations of parameters,
        for combination in params:
            model = model(C=combination['C'], kernel=combination['kernel'])
            model.fit(text, label)
            preds = model.predict(tuning[0])
            p_hs, r_hs, f1_hs, support = precision_recall_fscore_support(preds, tuning[1], average="macro")
            print(f1_hs)
            if f1_hs > best_f1:
                best_f1 = f1_hs
                best_model = model

        return best_model



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

    def _prep_xy(self, text, label, tuning):
        if isinstance(tuning, int):
            cv = tuning
            x = text
            y = label
        elif isinstance(tuning, tuple):
            x, y, cv = self._prep_custom_cv(text, label, tuning)
        else:
            print("tuning argument must be int or tuple")

        return x, y, cv

    def _prep_custom_cv(self, text, label, tuning):
        if isinstance(text, sp.csr_matrix):
            text_train = text.toarray()
        elif isinstance(text, np.ndarray):
            text_train = text
        else:
            TypeError('text argument must be ndarray or csr_matrix')

        if isinstance(tuning[0], sp.csr_matrix):
            text_dev = tuning[0].toarray()
        elif isinstance(tuning[0], np.ndarray):
            text_dev = tuning[0]
        else:
            TypeError('tuning[0] argument must be ndarray or csr_matrix')

        label_dev = tuning[1]

        x = np.concatenate((text_train, text_dev))
        print(x.shape)
        y = np.concatenate((label, label_dev))
        print(y.shape)

        # test_fold = np.concatenate([
        #     np.full(text_train.shape[0], -1, dtype=np.int8),
        #     np.zeros(text_dev.shape[0], dtype=np.int8)])

        # generate a test fold array indicating the training and dev splits
        test_fold = np.zeros(len(x))
        test_fold[:len(text_train)] = -1

        # create a PredefinedSplit object using the test fold array
        ps = PredefinedSplit(test_fold)
        ps = ShuffleSplit(n_splits=2, test_size=0.5, random_state=42)

        # ps = PredefinedSplit(test_fold.tolist())

        return x, y, ps

    def _convert_to_ndarray(self, data_vector):
        pass

    def _get_param_combinations(self, param_grid):
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
