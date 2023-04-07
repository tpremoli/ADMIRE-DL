# {APP_NAME} Design

## Introduction

This application works via a CLI (cli.py). There are 4 subtools:

1. `prep`: The tool used to run preprocessing scripts on MRI datasets
2. `train`: The tool used to train models
3. `test`: The tool used to test models
4. `predict`: The tool used to create single predictions on existing models

## The tools

### `prep`: Preprocessing datasets

The prep tool runs preprocessing steps for a given adni dataset. The prep tool is ran by using the command

    py cli.py prep <options>

where the options are

- **run_name** `-r`: The name of the run. Files will be saved in `out/preprocessed_datasets/{run_name}`
- **ratio** `--ratio`: The train/test/validation ratio. Files will be saved in `{run_name}/train`,`{run_name}/test`,`{run_name}/val` respectively.

If you're using an ADNI dataset, you must input

- **collection_dir** `-d`: The directory of the collection. If the collection was downloaded from ADNI, this should be the "ADNI" folder
- **collection_csv** `-c`: The directory of the collection's csv (Metadata) file. This allows the program to identify which research group the collection belongs to.

This will output preprocessed MRI images and objects, ready to be trained on. Each scan will be treated as a different datapoint, even if a subject has multiple scans attributed to them.

This runs the full suite of prep scripts, however this can be very time consuming (Particularly with the full 3D MRI registration). The behavior of the program can be tweaked using the settings file. More information can be found in [Settings](#Settings) 


### `train`: Preprocessing datasets

The train tool runs a training task on a prepped dataset. The train tool is ran by using the command

    py cli.py test <options>

where the options are

- **config** `-c`: The training task configuration file. This defines everything necessary in the task.

The config files are defined using `.yml` files. The following is an example.

# TODO: UPDATE THIS # TODO: add convention  
```yaml 
task_name: densenet121_transferlearn_slices 
dataset: out/adni_processed
options:
    architecture: DenseNet121
    method: transferlearn
    pooling: null # Default: None
    learning_rate: 0.001 # Default: 0.001
    epochs: 25 # Default: 25
```

Trained models will be saved in `out/trained_models/{task_name}`.

To clarify the config options:

- **task_name**: The name of the task. This should be descriptive, written with snake_case
- **dataset**: The prepped dataset to use
- **architecture**: The architecture to be used for this task. Should follow naming convention from the [Keras docs](https://keras.io/api/applications). The supported architectures are written below.
- **method**: The training method to use. This can be `pretrain` or `transferlearn`.
- **pooling**: The pooling method to use. Default is `None` (`null`), however `avg` and `max` can be used to experiment with performance.
- **learning_rate**: Learning rate to be used in the training task. Default is 0.001.

### `eval`: Evaluating created models

*TBD*

### `predict`: Using models

*TBD*

## Settings

There are a few settings that can be set to modify the behavior of the program.

TODO: Fill this in

## Suppplemental files

### Unprocessed Datasets

TODO: explain this, basically storing the unprocessed datasets. Recommended to save them
here for ease of use

### Scripts

The `supplemental_files/scripts` directory contains a collection of scripts that are generally useful
for this program, but not essential to it's functionality. Generally, all of these should be ran from 
the project's root directory, with syntax `python supplemental_files/scripts/{scriptname}.py` A quick
rundown of the scripts is given below:

1. `run_all_tasks.py`

This script runs all tasks in the sample_configs directory. This will train all models used in the final
paper for this study. 

This has a few requirements.

- The ADNI dataset should be preprocessed, with a run name of adni_processed.
- The final dataset should be in the directory `out/preprocessed_datasets/adni_processed/axial_dataset`

This will run all the tasks in the same way tasks are normally ran, with all the parameters from the given configs.

2. `evals_to_csv.py`

converts all evals to a csv for easier viewing. required after teh main eval script ran

3. `oasis_to_nii.py`

converts oasis to proper nii format and saves in oasis_nifti unprocessed datasets

4. `prep_oasis.py`

runs a prep script specifically for the oasis dataset. Stripped down

5. `prep_adni_script.sh`

which runs the default adni script that was used in sample_configs

### Extensions

Explain these

### Sample brains

Explain these