"""This script converts a series of evalutation yml files to a CSV file.
This should be run after using the cli to evaluate trained models.
The script will read all the yml files in the out/evals directory,
creating a new file called evals.csv in the same directory, containing
all the metrics for each model. This allows for quicker comparison.

"""

import csv
import re
from pathlib import Path

# Set the input directory
input_directory = Path("out/evals")

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
            content = f.read()

        # Extract the metrics using regex
        accuracy = re.search(r"^\s+accuracy:\s(.+)$", content, re.MULTILINE).group(1)

        # group metrics
        w_avg_metrics = re.findall(r"^\s+weighted avg:\n((?:\s+(?:f1-score|precision|recall|support):\s.+[\n])+)", content, re.MULTILINE)[0]
        w_avg_metrics = [m.group(2) for m in re.finditer(r"^\s+(f1-score|precision|recall|support):\s(.+)$", w_avg_metrics, re.MULTILINE)]

        m_avg_metrics = re.findall(r"^\s+macro avg:\n((?:\s+(?:f1-score|precision|recall|support):\s.+[\n])+)", content, re.MULTILINE)[0]
        m_avg_metrics = [m.group(2) for m in re.finditer(r"^\s+(f1-score|precision|recall|support):\s(.+)$", m_avg_metrics, re.MULTILINE)]

        # class metrics
        ad_metrics = re.findall(r"^\s+AD:\n((?:\s+(?:f1-score|precision|recall|support):\s.+[\n])+)", content, re.MULTILINE)[0]
        ad_metrics = [m.group(2) for m in re.finditer(r"^\s+(f1-score|precision|recall|support):\s(.+)$", ad_metrics, re.MULTILINE)]

        cn_metrics = re.findall(r"^\s+CN:\n((?:\s+(?:f1-score|precision|recall|support):\s.+[\n])+)", content, re.MULTILINE)[0]
        cn_metrics = [m.group(2) for m in re.finditer(r"^\s+(f1-score|precision|recall|support):\s(.+)$", cn_metrics, re.MULTILINE)]

        # Write the metrics to the output file
        row = [file.stem[5:], accuracy] + m_avg_metrics + w_avg_metrics + cn_metrics + ad_metrics
        csv_writer.writerow(row)

print("Metrics saved in out/evals/evals.csv")
