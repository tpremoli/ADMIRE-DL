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

- **collection_dir** `-d`: The directory of the collection. If the collection was downloaded from ADNI, this should be the "ADNI" folder
- **collection_csv** `-c`: The directory of the collection's csv (Metadata) file. This allows the program to identify which research group the collection belongs to.
- **run_name** `-r`: The name of the run. Files will be saved in `out/preprocessed_datasets/{run_name}`

This will output preprocessed MRI images and objects, ready to be trained on. Each scan will be treated as a different datapoint, even if a subject has multiple scans attributed to them.

If you are using the kaggle data set, you can create the object files using the option

- **kaggle** `-k`: The directory of the kaggle mri images. These should be in their original folder structure (i.e nondemented, mildlydemented etc.)
  

### `train`: Preprocessing datasets

The train tool runs a training task on a prepped dataset. The train tool is ran by using the command

    py cli.py test <options>

where the options are

- **config** `-c`: The training task configuration file. This defines everything necessary in the task.

The config files are defined using `.yml` files. The following is an example.

```yaml
task_name: densenet121_finetune_slices
dataset: out/adni_processed
options:
    architecture: DenseNet121
    approach: slice
    method: finetune
    pooling: avg
    kaggle: False
    train_pct: 0.7  
    # optimizer: Adam ???
    # loss: categorical_crossentropy ??
    # learning_rate: 5e-4 ??
    # metrics: accuracy ??
    # activation: softmax ??
```

Trained models will be saved in `out/trained_models/{task_name}`.

To clarify the config options:

- **task_name**: The name of the task. This should be descriptive, written with snake_case
- **dataset**: The prepped dataset to use
- **architecture**: The architecture to be used for this task. Should follow naming convention from the [Keras docs](https://keras.io/api/applications). The supported architectures are written below.
- **approach**: The approach to use. This can be `slice` or `channel`, referring to slice-based approach or multi-channel based approach. The Kaggle dataset is incompatible with the multi-channel approach.
- **method**: The training method to use. This can be `pretrain` or `finetune`.
- **pooling**: The pooling method to use. Default is `None`, however `avg` and `max` can be used to experiment with performance.
- **train_pct**: The amount of the dataset to be used as training data. At the beginning of the training task, the dataset will be split according to this percentage and the classes. The split will be written to a file and exported with the final trained model.
  
### `test`: Testing created models

*TBD*

### `predict`: Using models

*TBD*

