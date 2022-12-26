# {APPNAME} Design

## Introduction

This application works via a CLI (cli.py). There are 4 subtools:

1. `prep`: The tool used to run preprocessing scripts on MRI datasets
2. `train`: The tool used to train models
3. `test`: The tool used to test models
4. `predict`: The tool used to create single predictions on existing models

## The tools

### `prep`: Preprocessing databases

The prep tool runs preprocessing steps for a given adni dataset. The prep tool is ran by using the command

    py cli.py prep <options>

where the options are

- **collection_dir**: The directory of the collection. If the collection was downloaded from ADNI, this should be the "ADNI" folder
- **collection_csv**: The directory of the collection's csv (Metadata) file. This allows the program to identify which research group the collection belongs to.
- **run_name**: The name of the run. Files will be saved in out/preprocessed_samples/{run_name}
  
This will output preprocessed MRI images and objects, ready to be trained on.

