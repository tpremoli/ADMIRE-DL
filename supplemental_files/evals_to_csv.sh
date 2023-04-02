#!/bin/bash

# This script converts all the eval_* files in the out/eval to a csv file.
# The csv file is saved in the out/eval directory.
# The csv file is named evals.csv

# Set the input directory
input_directory="out/evals"

# Create the output CSV file
output_file="out/evals/evals.csv"

echo "File,Accuracy,avg_f1_score,avg_precision,avg_recall,avg_support,weighted_f1_score,weighted_precision,weighted_recall,weighted_support,cn_f1_score,cn_precision,cn_recall,cn_support,ad_f1_score,ad_precision,ad_recall,ad_support" > "$output_file"

for file in "$input_directory"/*.yml; do
    # Extract the metrics using grep and awk
    accuracy=$(grep -E '^\s+accuracy:' "$file" | awk -F ': ' '{print $2}')

    # group metrics
    w_avg_metrics=$(grep -E -A 4 '^\s+weighted avg:' "$file" | grep -E '^\s+(f1-score|precision|recall|support):' | awk -F ': ' '{ORS = (NR % 4 == 0) ? "," : ","; print $2}')
    m_avg_metrics=$(grep -E -A 4 '^\s+macro avg:' "$file" | grep -E '^\s+(f1-score|precision|recall|support):' | awk -F ': ' '{ORS = (NR % 4 == 0) ? "," : ","; print $2}')

    # class metrics
    ad_metrics=$(grep -E -A 4 '^\s+AD:' "$file" | grep -E '^\s+(f1-score|precision|recall|support):' | awk -F ': ' '{ORS = (NR % 4 == 0) ? "," : ","; print $2}')
    cn_metrics=$(grep -E -A 4 '^\s+CN:' "$file" | grep -E '^\s+(f1-score|precision|recall|support):' | awk -F ': ' '{ORS = (NR % 4 == 0) ? "," : ","; print $2}')

    # Write the metrics to the output file
    echo "$(basename -s .yml "$file" | sed 's/^eval_//'),$accuracy,$m_avg_metrics$w_avg_metrics$cn_metrics$ad_metrics" >> "$output_file"
done

echo "Metrics saved in out/evals/evals.csv"
