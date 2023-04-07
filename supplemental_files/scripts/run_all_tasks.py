"""This script runs all the tasks in the sample_configs directory.
This is useful for running all the tasks in a single command.

Run this from the project root directory, e.g.:
python supplemental_files/scripts/run_all_tasks.py
"""

import os
import subprocess

def run_task(config_file):
    print(f"Starting {config_file}")
    subprocess.run(["python", "cli.py", "train", "-c", config_file])

if __name__ == "__main__":
    print("Starting all tasks")

    # PT tasks
    print("Starting PT tasks")
    pt_tasks = [
        "pt_ax_densenet121",
        "pt_ax_densenet201",
        "pt_ax_resnet50",
        "pt_ax_resnet152",
        "pt_ax_vgg16",
        "pt_ax_vgg19",
    ]
    for task in pt_tasks:
        config_file = f"sample_configs/{task}.yml"
        run_task(config_file)

    # TL tasks: Adam
    print("Starting TL tasks")
    tl_adam_tasks = [
        "tl_ax_densenet121_adam",
        "tl_ax_densenet201_adam",
        "tl_ax_resnet50_adam",
        "tl_ax_resnet152_adam",
        "tl_ax_vgg16_adam",
        "tl_ax_vgg19_adam",
    ]
    for task in tl_adam_tasks:
        config_file = f"sample_configs/{task}.yml"
        run_task(config_file)

    # TL tasks: SGD
    tl_sgd_tasks = [
        "tl_ax_densenet121_sgd",
        "tl_ax_densenet201_sgd",
        "tl_ax_resnet50_sgd",
        "tl_ax_resnet152_sgd",
        "tl_ax_vgg16_sgd",
        "tl_ax_vgg19_sgd",
    ]
    for task in tl_sgd_tasks:
        config_file = f"sample_configs/{task}.yml"
        run_task(config_file)

    print("Done with all tasks")
