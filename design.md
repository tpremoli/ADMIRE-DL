<center>

## Alzheimer's Disease MRI Identification, Recognition, & Evaluation - Deep Learning
<img src="supplemental_files/logo.png" alt="drawing" width="300"/>
</center>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [The Tools](#the-tools)
  - [`prep`: Preprocessing Datasets](#prep-preprocessing-datasets)
  - [`train`: Training Models](#train-training-models)
  - [`eval`: Evaluating Models](#eval-evaluating-models)
- [Settings](#settings)
- [Supplemental Files](#supplemental-files)
  - [Unprocessed Datasets](#unprocessed-datasets)
  - [Sample Configs](#sample-configs)
  - [Scripts](#scripts)
  - [Extensions](#extensions)
  - [Sample brains](#sample-configs)


## Introduction

This is the repository for the deep learning portion of the Alzheimer's Disease MRI Identification, Recognition, & Evaluation project. This project aims to create a deep learning model that can accurately classify Alzheimer's Disease using MRI scans. This is done by using a deep learning model to classify the scans. 

This project uses the [Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset](https://adni.loni.usc.edu/), which contains MRI scans of patients with Alzheimer's Disease and healthy controls. The [OASIS-1](https://www.oasis-brains.org/) dataset is also used for testing purposes.

In terms of implementation, this project uses the [Keras](https://keras.io/) deep learning library, and the [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) image processing library.


## Requirements

This project uses [conda](https://docs.conda.io/en/latest/) to manage dependencies. To get started, run the following commands:

    conda env create -f environment.yml
    conda activate training_env

You must also make sure that the [NVIDIA GPU Driver](https://www.nvidia.com/Download/index.aspx) is installed. To verify that the driver is installed, run

    nvidia-smi

It is recommended that you use a GPU for training. To verify that the GPU is installed, run

    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

Linux users may need to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) to use the GPU.

This project also uses [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) for image processing. To install FSL, follow the instructions [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation).

## Getting Started

To get started, clone the repository

    git clone https://github.com/tpremoli/ADMIRE-DL

and navigate to the directory
    
    cd ADMIRE-DL

This project uses a CLI (Command Line Interface) to run tasks. The CLI is defined in `cli.py`. To get a list of all the tools, run

    python cli.py -h

To see the help for a specific command, run

    python cli.py <command> -h

These will be explained in more detail in [The Tools](#The-Tools).

To get started, you must first download the ADNI dataset. This can be done by following the instructions [here](https://adni.loni.usc.edu/data-samples/access-data/). The specific collection used in this study is the ADNI1:Screening 1.5T MRI dataset.

Once the dataset is downloaded, you must extract the zip file to `supplemental_files/unprocessed_datasets`. Once this zip file has been
extracted, it's also required that you get the metadata csv file, as this contains information about the collection. Place this file into
`supplemental_files/unprocessed_datasets`, and rename it to `test_sample.csv`.

To prep this dataset, use the provided script

    sh supplemental_files/scripts/prep_adni.sh

Which will create an adni_processed in `out/preprocessed_datasets`. After that, you can get to training models. This can be done by running

    python cli.py train -c <config>

where `<config>` is the path to a training config file. More information can be found in [`train`: Training Models](#train-training-models). This will output a model in `out/trained_models`.

After that, to evaluate the model, run

    python cli.py eval

This will output a file in `out/evals` containing the evaluation results.

## The Tools

This application works via a CLI (cli.py). There are 4 subtools:

1. `prep`: The tool used to run preprocessing scripts on MRI datasets
2. `train`: The tool used to train models
3. `test`: The tool used to test models
4. `predict`: The tool used to create single predictions on existing models

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

This runs the full suite of prep scripts, however this can be very time consuming (Particularly with the full 3D MRI registration). The behavior of the program can be tweaked using the settings file. More information on the settings file can be found in [Settings](#Settings).

### `train`: Training models

The train tool runs a training task on a prepped dataset. The train tool is ran by using the command

    py cli.py test <options>

where the options are

- **config** `-c`: The training task configuration file. This defines everything necessary in the task.

The config files are defined using `.yml` files. The following is an example.

```yaml 
task_name: tl_ax_vgg16_nopooling_Adam
dataset: out/preprocessed_datasets/adni_processed/axial_dataset
options:
    architecture: VGG16
    method: transferlearn
    pooling: null 
    epochs: 100
    batch_size: 32
    overrides:
        optimizer_name: Adam # This has given better results
```

All the options can be found in `supplemental_files/sample_configs/_options.yml`.

- **task_name**: The name of the task. This should be descriptive, written with snake_case
- **dataset**: The prepped dataset to use. Usually an axial dataset. This should be a folder containing `train`, `test`, and `val` folders, with CN and AD folders inside each of those.
- **architecture**: The architecture to be used for this task. Should follow naming convention from the [Keras docs](https://keras.io/api/applications). The supported architectures are `VGG16 | VGG19 | ResNet50 | ResNet152 | DenseNet121 | DenseNet201`
- **method**: The training method to use. This can be `pretrain` or `transferlearn`.
- **batch_size**: The batch size to use in the training task. Default is 32.
- **pooling**: The pooling method to use. Default is `None` (`null`), however `avg` and `max` can be used to experiment with performance.
- **epochs**: The number of epochs to use in the training task. Default is 50.

There are also some custom overrides that can be used. These are as follows:

- **optimizer_name**: The optimizer to use. This can be `Adam`, `SGD` or `RMSProp`. Default is `Adam`.
- **learning_rate**: The learning rate to use. Default is 0.0003 if optimizer == SGD, else 0.001.
- **l2_reg**: The L2 regularization to use. Default is null.
- **dropout**: The dropout to use. Default is null.

Trained models will be saved in `out/trained_models/{task_name}`. They can be evaluated using the `eval` tool.

### `eval`: Evaluating Models

*TBD*

## Settings

There are a few settings that can be set to modify the behavior of the program.

These can be found in `admire/settings.py`. The settings are as follows:

- `USE_S3`: Whether to use S3 for file storage. If this is set to `True`, the program will backup prepped
nii files to S3. This is useful for large datasets, as it allows for the program to be more robust to
crashes. This requires s3 to be configured for your machine.

- `AWS_S3_BUCKET_NAME`: The name of the bucket to use for S3 storage. This is only used if `USE_S3` is
set to `True`.

- `FSL_CONCURRENT_PROCESSES`: The number of concurrent processes to use for FSL. This makes use of pooling
to run multiple fsl processes at once. This is useful for speeding up the program, but can cause issues
if the machine is not powerful enough. This is set to 6 by default (for a 16GB ram system).

- `SKIP_FSL`: Whether to skip FSL processing. This is useful for if you already have the FSL processed
files, and don't want to reprocess them. This is set to `False` by default.

- `USE_NONLINEAR_REGISTRATION`: Whether to use nonlinear registration. This is set to `False` by default.
Look at the [extensions](#Extensions) section for more information.

- `SKIP_SLICE_CREATION`: Whether to skip slice creation. This is useful for if you don't want to create
slices from the 3D MRI images. This is set to `False` by default.

- `SKIP_FOLDER_SPLIT`: Whether to skip folder splitting. This is useful for if you don't want to split
the dataset into train/test/val folders. This is set to `False` by default.

## Suppplemental files

### Unprocessed Datasets

This folder contains the unprocessed datasets used in this study. These are stored here for ease of use,
and to allow for reproducibility. Although none are provided in the repo, the program works best when
the datasets are stored in the same directory structure as the ones provided here.

The datasets and recommended folder structure are as follows:

- `supplemental_files/unprocessed_datasets/ADNI`: The ADNI dataset. This contains all the individual
subject folders, as contained in the ADNI download.
- `supplemental_files/unprocessed_datasets/test_sample,csv`: The ADNI dataset csv file. This is the
metadata file that contains information about the subjects in the dataset. This is required for the
ADNI dataset to be processed. A small sample is provided here, but the full csv can be found in
the ADNI download.
- `supplemental_files/unprocessed_datasets/OASIS`: The OASIS dataset. This contains all the individual
subject scans, as contained in the OASIS download. This dataset is not provided here, as it is not
publicly available. However, the program will work with this dataset if it is placed in this folder,
along with the `oasis_cross-sectional.csv` file.

### Sample Configs

The `supplemental_files/sample_configs` directory contains a collection of sample configs that can
be used to run the program. These follow a simple naming convention to make it easy to understand.

The naming convention is as follows:

    {method}_{dataset}_{architecture}

Where:

- **method**: The config's training method. This can be `pt` (for training from scratch) or `tl`
(for transfer learning).

- **dataset**: The config's dataset. For this version of the program, this is `ax` for axial but
there is room for more options.

- **architecture**: The config's architecture. This is the name of the architecture, in all lowercase.

To run these, simply run `python cli.py train -c {config_dir}`. For example, to run the `pt_ax_vgg16`
config, you would run `python cli.py train -c supplemental_files/sample_configs/pt_ax_vgg16`.

### Scripts

The `supplemental_files/scripts` directory contains a collection of scripts that are generally useful
for this program, but not essential to it's functionality. Generally, all of these should be ran from 
the project's root directory, with syntax `python supplemental_files/scripts/{scriptname}.py` A quick
rundown of the scripts is given below:

1. `run_all_tasks.py`

This script runs all tasks in the supplemental_files/sample_configs directory. This will train all models
used in the final paper for this study. 

This has a few requirements.

- The ADNI dataset should be preprocessed, with a run name of adni_processed.
- The final dataset should be in the directory `out/preprocessed_datasets/adni_processed/axial_dataset`

This will run all the tasks in the same way tasks are normally ran, with all the parameters from the given configs.

2. `evals_to_csv.py`

This script goes through all the eval yml files in `out/evals` and places them in a csv, called `evals.csv`

This will contain all the statistics gathered from running `python cli.py eval` for each model that has been
trained and evaluated.

This is mostly for ease of use, and to have all relevant data in one place.

3. `oasis_to_nii.py`

`NOTE`: Due to discrepancies between the OASIS and ADNI datasets, combining the two is not recommended.
For real-world use, it is recommended to use only one dataset, as combining them will likely result in
poor performance. The final report associated with this repository did not use OASIS.

This script is used to convert the OASIS dataset from Analyze to current standards (NiFTi). This is due to
the fact that the Analyze format is outdated, and lacks orientation information. This script will convert
the files into NiFTi format and fix the orientation so the dataset can actually be employed in this task.

Running this script is required for the use of the OASIS dataset - if you don't convert into NiFTi you will
have problems when preprocessing and orienting images.

To run this program, use the command
    
    python supplemental_files/scripts/ oasis_to_nii.py <oasisdir>

where oasisdir is the directory where the OASIS dataset is stored. This dataset should have the `oasis_cross-sectional.csv`
file in the root folder, and the different scans should be moved out of their original `disk{n}`
folders and into the root of the oasisdir.

Running this script will create a new folder called `oasis_nifti` in 
`supplemental_files/unprocesssed_datasets`. This folder contains the converted images, and the csv file.

Once this is done, the dataset should be ready to be preprocessed using the `prep_oasis.py` script.

4. `prep_oasis.py`

This is a modified version of the prep tool found in the CLI, stripped down and modified to work with the
OASIS dataset. This script doesn't take in any arguments, and can be ran directly from the command line.
The script will run the preprocessing steps on the OASIS dataset, and save the results as typically would
be done with the prep tool, in `out/preprocessed_datasets/oasis_processed`.

This script requires that the OASIS dataset has been converted to NiFTi format, and that the `oasis_nifti`
folder is in `supplemental_files/unprocessed_datasets`. If this is not the case, the script will fail.

Run `oasis_to_nii.py` before running this script.

5. `prep_adni_script.sh`

This is a quick script that runs the prep tool on the ADNI dataset. This is useful for running the prep tool
with the same parameters as the paper, and is a quick way to get the dataset ready for training.

The dataset used in this paper is specifically the `ADNI1:Screening 1.5T` dataset, which can be downloaded
from [here](https://ida.loni.usc.edu/pages/accessing-data/). The dataset should be downloaded as a zip file,
and then extracted into the `supplemental_files/unprocessed_datasets` folder. The folder should be renamed
to `ADNI`. The csv file should be renamed to `test_sample.csv`.

This will preprocess this dataset into the `out/preprocessed_datasets/adni_processed` folder. This folder
is the main one used in the paper, and is the one that should be used for training.

### Extensions

#### T1_2_MNI_152_1mm.cnf

This is a configuration file for the `prep` tool. It is used to preprocess the ADNI dataset, and is the
configuration file used to preprocess datasets using nonlinear registration to the MNI152 1mm template.

This conf is not provided in fsl by default, and must be installed. This can be done by copying the file from
`supplemental_files/extensions` to `$FSLDIR/etc/flirtsch/T1_2_MNI152_1mm.cnf`.

The final paper does not use nonlinear registration, and instead uses linear registration. This is because
nonlinear registration is very slow, and linear registration is sufficient for this task. However, nonlinear
registration is still provided here for completeness.

### Sample brains

A couple of sample brains are provided in the `supplemental_files/sample_brains` folder. These are used
in the paper to show the results of the preprocessing steps. The brains are: 

- `mni_lin_1mm_brain.nii.gz`: A brain that has been linearly registered to the MNI152 1mm template
- `mni_nonlin_1mm_brain.nii.gz`: A brain that has been non-linearly registered to the MNI152 1mm template
- `mni_nonlin_2mm_brain.nii.gz`: A brain that has been linearly registered to the MNI152 2mm template
