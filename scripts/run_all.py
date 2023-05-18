from os import path, system
from signal import signal, SIGINT
import sys
import subprocess
from time import sleep

all_modes = ["train", "test"]
all_tasks = ["primary", "adaptation"]
all_models = ["baseline", "alpha", "beta", "delta", "gamma"]


def format_args(mode, task, model):
    return ["-m", mode, "-t", task, "-s", model]


def main():
    scripts_dir = path.dirname(__file__)

    for mode in all_modes:
        for task in all_tasks:
            for model in all_models:
                args = format_args(mode, task, model)
                command_parts = [sys.executable, f"{scripts_dir}/run_model.py"] + args
                print(f"\nCalling run_model.py with args: {args}")
                subprocess.run(command_parts)


main()
