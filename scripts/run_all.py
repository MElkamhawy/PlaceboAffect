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
    for mode in all_modes:
        for task in all_tasks:
            for model in all_models:
                args = format_args(mode, task, model)
                command_parts = [sys.executable, f"{repo_dir}/src/main.py"] + args
                print(f"\nCalling main.py with args: {args}")
                subprocess.run(command_parts)


main()
