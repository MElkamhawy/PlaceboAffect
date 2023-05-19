from os import path
import sys
import subprocess

scripts_dir = path.dirname(__file__)
repo_dir = path.abspath(f"{scripts_dir}/../")

all_modes = ["train", "test"]
all_tasks = ["primary", "adaptation"]
all_models = ["baseline", "alpha", "beta", "delta", "gamma"]


def format_args(mode, task, model):
    return ["-m", mode, "-t", task, "-s", model]


def main():
    for task in all_tasks:
        for model in all_models:
            for mode in all_modes:
                args = format_args(mode, task, model)
                command_parts = [f"{repo_dir}/scripts/model_runner.sh"] + args
                print(f"\nCalling main.py with args: {args}")
                subprocess.run(command_parts)


main()
