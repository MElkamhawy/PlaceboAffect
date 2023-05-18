from argparse import ArgumentParser
from os import path, system, makedirs
import sys
import subprocess


# Project Directories
scripts_dir = path.dirname(__file__)
repo_dir = path.abspath(f"{scripts_dir}/../")
data_dir = f"{repo_dir}/data"
results_dir = f"{repo_dir}/results/D4"
outputs_dir = f"{repo_dir}/outputs/D4"
models_dir = f"{repo_dir}/models/D4"


def create_arg_parser():
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        "-m",
        "--mode",
        choices=["test", "train"],
        default="train",
    )
    argument_parser.add_argument(
        "-t",
        "--task",
        choices=["adaptation", "primary"],
        default="primary",
    )
    argument_parser.add_argument(
        "-s",
        "--model",
        choices=["baseline", "alpha", "beta", "delta", "gamma"],
        default="baseline",
    )
    return argument_parser


def validate_path(filename, fail_if_missing=False):
    if fail_if_missing and not path.isfile(filename):
        raise Exception(f"{filename} does not exist!")
    directory = path.dirname(filename)
    makedirs(directory, exist_ok=True)
    return filename


def create_script_arg(key, value):
    return [f"--{key}", value]


def create_script_args(mode, task, model):
    lang = "en" if task == "primary" else "es"

    train_data_path = validate_path(
        f"{data_dir}/train/{lang}/hateval2019_{lang}_train.csv", True
    )
    dev_data_path = validate_path(
        f"{data_dir}/dev/{lang}/hateval2019_{lang}_dev.csv", True
    )
    test_data_path = validate_path(
        f"{data_dir}/test/{lang}/hateval2019_{lang}_test.csv", True
    )

    result_dir = f"{results_dir}/{task}"
    result_filename_suffix = f"_{model}" if model != "baseline" else ""
    result_filename = f"D4_scores{result_filename_suffix}.out"

    devtest_result_path = validate_path(f"{result_dir}/devtest/{result_filename}")
    evaltest_result_path = validate_path(f"{result_dir}/evaltest/{result_filename}")

    output_dir = f"{outputs_dir}/{task}"
    predictions_filename = f"pred_{model}.txt"

    devtest_predictions_path = validate_path(
        f"{output_dir}/devtest/{predictions_filename}"
    )
    evaltest_predictions_path = validate_path(
        f"{output_dir}/evaltest/{predictions_filename}"
    )

    model_path = validate_path(f"{repo_dir}/models/D4/{model}.pkl", True)
    config_path = validate_path(f"{repo_dir}/src/configs/{model}.yaml", True)

    return (
        create_script_arg("mode", mode)
        + create_script_arg("task", task)
        + create_script_arg("train-data", train_data_path)
        + create_script_arg("dev-data", dev_data_path)
        + create_script_arg("test-data", test_data_path)
        + create_script_arg("devtest-result", devtest_result_path)
        + create_script_arg("evaltest-result", evaltest_result_path)
        + create_script_arg("devtest-predictions", devtest_predictions_path)
        + create_script_arg("evaltest-predictions", evaltest_predictions_path)
        + create_script_arg("model", model_path)
        + create_script_arg("config", config_path)
    )


def format_command(script_args):
    command_parts = [
        f"{repo_dir}/scripts/model_runner.sh",
    ] + script_args
    # # Uncomment to debug
    # debug_args = " ".join(script_args)
    # print(f"Running model_runner.sh with arguments:\n{debug_args}")
    return command_parts


def main():
    argument_parser = create_arg_parser().parse_args()

    mode = argument_parser.mode
    task = argument_parser.task
    model = argument_parser.model
    script_args = create_script_args(mode, task, model)
    command = format_command(script_args)
    subprocess.run(command)


main()
