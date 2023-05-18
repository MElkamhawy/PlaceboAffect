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
        args = argument_parser.parse_args()
        self.mode = args.mode
        self.task = args.task
        self.model = args.model

        self._init_data_paths()
        self._init_result_paths()
        self._init_output_paths()
        self._init_model_and_config_paths()

    def _validate_path(self, filename, fail_if_missing=False):
        if fail_if_missing and not path.isfile(filename):
            raise Exception(f"{filename} does not exist!")
        directory = path.dirname(filename)
        makedirs(directory, exist_ok=True)
        return filename

    def _init_data_paths(self):
        lang = "en" if self.task == "primary" else "es"

        self.train_data_path = self._validate_path(
            f"{data_dir}/train/{lang}/hateval2019_{lang}_train.csv", True
        )
        self.dev_data_path = self._validate_path(
            f"{data_dir}/dev/{lang}/hateval2019_{lang}_dev.csv", True
        )
        self.test_data_path = self._validate_path(
            f"{data_dir}/test/{lang}/hateval2019_{lang}_test.csv", True
        )

    def _init_result_paths(self):
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
        output_dir = f"{outputs_dir}/{self.task}"
        predictions_filename = f"pred_{self.model}.txt"

        self.devtest_predictions_path = self._validate_path(
            f"{output_dir}/devtest/{predictions_filename}"
        )
        self.evaltest_predictions_path = self._validate_path(
            f"{output_dir}/evaltest/{predictions_filename}"
        )

    def _init_model_and_config_paths(self):
        self.model_path = self._validate_path(
            f"{repo_dir}/models/D4/{self.model}.pkl", self.mode == 'test'
        )
        self.config_path = self._validate_path(
            f"{repo_dir}/src/configs/{self.model}.yaml", True
        )


def create_arg_parser():
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
    argument_parser = create_arg_parser()
    return ParsedArgs(argument_parser)
