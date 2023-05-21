from argparse import ArgumentParser
from os import path, makedirs

# Project Directories
scripts_dir = path.dirname(__file__)
repo_dir = path.abspath(f"{scripts_dir}/../")
data_dir = f"{repo_dir}/data"
results_dir = f"{repo_dir}/results/D4"
outputs_dir = f"{repo_dir}/outputs/D4"
models_dir = f"{repo_dir}/models/D4"


class ParsedArgs:
    def __init__(self, argument_parser):
        """
        Initializes a ParsedArgs object by parsing the command-line arguments using the argument parser.
        Args:
        argument_parser (ArgumentParser): The argument parser object used for parsing the command-line arguments.
        Attributes:
        mode (str): The mode of operation (train or test).
        task (str): The task to perform (primary or adaptation).
        model (str): The chosen model.
        """
        args = argument_parser.parse_args()
        self.mode = args.mode
        self.task = args.task
        self.model = args.model
        # Initialize file paths
        self._init_data_paths()
        self._init_result_paths()
        self._init_output_paths()
        self._init_model_and_config_paths()

    def _validate_path(self, filename, fail_if_missing=False):
        """
        Validates a file path by checking if the file exists and creating the necessary directory if it doesn't.
        Args:
        filename (str): The file path to validate.
        fail_if_missing (bool): Flag indicating whether to raise an exception if the file is missing (default: False).
        Returns:
        str: The validated file path.
        Raises:
        Exception: If the file is missing and 'fail_if_missing' is True.
        """
        if fail_if_missing and not path.isfile(filename):
            raise Exception(f"{filename} does not exist!")
        directory = path.dirname(filename)
        makedirs(directory, exist_ok=True)
        return filename

    def _init_data_paths(self):
        """
        Initializes the file paths for the data files based on the task.
        Sets the train_data_path, dev_data_path, and test_data_path attributes.
        """
        lang = "en" if self.task == "primary" else "es2en"

        self.train_data_path = self._validate_path(
            f"{data_dir}/train/{lang}/hateval2019_{lang}_train.csv",
            self.mode == "train",
        )
        self.dev_data_path = self._validate_path(
            f"{data_dir}/dev/{lang}/hateval2019_{lang}_dev.csv", True
        )
        self.test_data_path = self._validate_path(
            f"{data_dir}/test/{lang}/hateval2019_{lang}_test.csv", True
        )

    def _init_result_paths(self):
        """
        Initializes the file paths for the result files based on the task and model.
        Sets the devtest_result_path and evaltest_result_path attributes.
        Constructs the result file names based on the model.
        """
        result_dir = f"{results_dir}/{self.task}"
        result_filename_suffix = f"_{self.model}" if self.model != "baseline" else ""
        result_filename = f"D4_scores{result_filename_suffix}.out"

        self.devtest_result_path = self._validate_path(
            f"{result_dir}/devtest/{result_filename}"
        )
        self.evaltest_result_path = self._validate_path(
            f"{result_dir}/evaltest/{result_filename}"
        )

    def _init_output_paths(self):
        """
        Initializes the file paths for the output files based on the task and model.
        Sets the devtest_predictions_path and evaltest_predictions_path attributes.
        Constructs the predictions file names based on the model.
        """
        output_dir = f"{outputs_dir}/{self.task}"
        predictions_filename = f"pred_{self.model}.txt"

        self.devtest_predictions_path = self._validate_path(
            f"{output_dir}/devtest/{predictions_filename}"
        )
        self.evaltest_predictions_path = self._validate_path(
            f"{output_dir}/evaltest/{predictions_filename}"
        )

    def _init_model_and_config_paths(self):
        """
        Initializes the file paths for the model and configuration files based on the model.
        Sets the model_path and config_path attributes.
        """
        self.model_path = self._validate_path(
            f"{repo_dir}/models/D4/{self.model}.pkl", self.mode == "test"
        )
        self.config_path = self._validate_path(
            f"{repo_dir}/src/configs/{self.model}.yaml", True
        )


def create_arg_parser():
    """
    Creates an argument parser object with predefined command-line arguments.
    Returns:
    ArgumentParser: The argument parser object.
    The command-line arguments added are:
    - -m or --mode: Specifies the mode of operation (test or train). Default is "train".
    - -t or --task: Specifies the task to perform (primary or adaptation). Default is "primary".
    - -s or --model: Specifies the chosen model (baseline, alpha, beta, delta, gamma). Default is "baseline".
    """
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        "-m",
        "--mode",
        choices=["test", "train"],
        default="train",
        help="Train or test the model",
    )
    argument_parser.add_argument(
        "-t",
        "--task",
        choices=["adaptation", "primary"],
        default="primary",
        help="Task to perform (primary or adaptation)",
    )
    argument_parser.add_argument(
        "-s",
        "--model",
        choices=["baseline", "alpha", "beta", "delta", "gamma"],
        default="baseline",
    )
    return argument_parser


def parse_args():
    """
    Creates an argument parser using the create_arg_parser function.
    Initializes a ParsedArgs object by passing the argument parser to its constructor.
    Returns:
    the created ParsedArgs object.
    """
    argument_parser = create_arg_parser()
    return ParsedArgs(argument_parser)
