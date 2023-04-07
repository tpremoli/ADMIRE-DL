"""This script converts a series of evalutation yml files to a CSV file.
This should be run after using the cli to evaluate trained models.
The script will read all the yml files in the out/evals directory,
creating a new file called evals.csv in the same directory, containing
all the metrics for each model. This allows for quicker comparison.

Run this from the project root directory, i.e
python supplemental_files/scripts/evals_to_csv.py
"""

import csv
import yaml
from pathlib import Path

filedir = Path(__file__).parent.resolve()

# Set the input directory
input_directory = Path(filedir, "../../out/evals").resolve()

# Create the output CSV file
output_file = input_directory / "evals.csv"

# Write header to the output file
header = [
    "File",
    "Accuracy",
    "avg_f1_score",
    "avg_precision",
    "avg_recall",
    "avg_support",
    "weighted_f1_score",
    "weighted_precision",
    "weighted_recall",
    "weighted_support",
    "cn_f1_score",
    "cn_precision",
    "cn_recall",
    "cn_support",
    "ad_f1_score",
    "ad_precision",
    "ad_recall",
    "ad_support",
]

with open(output_file, "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(header)

    # Process each YAML file
    for file in sorted(input_directory.glob("*.yml")):
        with open(file) as f:
            content = yaml.safe_load(f)

        # Extract the metrics
        sklearn_eval = content["keras eval"]["sklearn eval"]
        accuracy = sklearn_eval["accuracy"]

        # group metrics
        w_avg_metrics = sklearn_eval["weighted avg"]
        m_avg_metrics = sklearn_eval["macro avg"]

        # class metrics
        ad_metrics = sklearn_eval["AD"]
        cn_metrics = sklearn_eval["CN"]

        # Write the metrics to the output file
        row = [
            file.stem[5:],
            accuracy,
            m_avg_metrics["f1-score"],
            m_avg_metrics["precision"],
            m_avg_metrics["recall"],
            m_avg_metrics["support"],
            w_avg_metrics["f1-score"],
            w_avg_metrics["precision"],
            w_avg_metrics["recall"],
            w_avg_metrics["support"],
            cn_metrics["f1-score"],
            cn_metrics["precision"],
            cn_metrics["recall"],
            cn_metrics["support"],
            ad_metrics["f1-score"],
            ad_metrics["precision"],
            ad_metrics["recall"],
            ad_metrics["support"],
        ]
        csv_writer.writerow(row)

print("Metrics saved in out/evals/evals.csv")