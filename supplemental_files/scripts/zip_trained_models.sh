#!/bin/bash
# This script zips all directories in the specified directory

# Check if a directory is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Check if the provided argument is a valid directory
if [ ! -d "$1" ]; then
    echo "Error: $1 is not a valid directory"
    exit 1
fi

# Change to the specified directory
cd "$1"

# Loop through each item in the directory
for item in *; do
    # Check if the item is a directory
    if [ -d "$item" ]; then
        # Compress the directory and its contents into a .zip file
        zip -r "${item}.zip" "$item"
    fi
done

echo "All directories have been zipped successfully!"
