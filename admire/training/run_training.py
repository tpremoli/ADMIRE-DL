import yaml
import shutil
import numpy as np
import pickle
from collections import Counter
from termcolor import cprint, colored
from tensorflow.config import list_logical_devices
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from datetime import datetime
from pathlib import Path
from ..constants import *
from .utils import gen_subsets, plot_data
from .model import create_model

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()


def load_training_task(file_loc):
    """Loads a training task from a yaml file, validates the config, and runs the training task.

    Args:
        file_loc (str): location of the yaml file containing the task config

    Raises:
        ValueError: Raised when the task config is invalid.
    """
    with open(Path(cwd, file_loc).resolve(), "r") as f:
        yamlfile = yaml.safe_load(f)
        keys = yamlfile.keys()
        options = yamlfile["options"]
        optionkeys = options.keys()

        if "task_name" not in keys:
            raise ValueError(
                colored("Task config requires a task_name attribute!", "red"))
        if "dataset" not in keys:
            raise ValueError(
                colored("Task config requires a dataset attribute!", "red"))

        if "architecture" not in optionkeys:
            raise ValueError(
                colored("Task config requires an architecture attribute!", "red"))
        if "method" not in optionkeys:
            raise ValueError(
                colored("Task config requires a method attribute!", "red"))

        # getting required parameters
        architecture = yamlfile["options"]["architecture"]
        task_name = yamlfile["task_name"]
        dataset_dir = Path(cwd, yamlfile["dataset"]).resolve()
        method = yamlfile["options"]["method"]

        # Getting optional parameters with defaults TODO: move these to overrides
        pooling = yamlfile["options"].get("pooling", None)  # Default to None
        epochs = yamlfile["options"].get("epochs", 50)  # Default to 50 epochs
        batch_size = yamlfile["options"].get("batch_size", 32)  # Default to 32 batch size

        if "overrides" in optionkeys:
            # Extra parameters
            overrides = {
                # Defaults to Adam optimizer,
                "optimizer_name": yamlfile["options"]["overrides"].get("optimizer_name", None),
                # Defaults to no l2 regularization,
                "l2reg": yamlfile["options"]["overrides"].get("l2reg", None),
                # Defaults to no dropout,
                "dropout": yamlfile["options"]["overrides"].get("dropout", None),
                # Defaults to default learning rates
                "learning_rate": yamlfile["options"]["overrides"].get("learning_rate", None)
            }
        else:
            overrides = {}

        parent_dir = Path(filedir, "../../out/trained_models").resolve()
        parent_dir.mkdir(parents=True, exist_ok=True)

        for path in parent_dir.glob(yamlfile["task_name"]):
            raise ValueError(colored(
                f"Task with name {yamlfile['task_name']} already exists in {path}!", "red"))

        model_loc = Path(parent_dir, yamlfile["task_name"]).resolve()
        model_loc.mkdir(parents=True, exist_ok=False)
        
        shutil.copyfile(Path(cwd, file_loc).resolve(), Path(
            model_loc, "task_config.yml").resolve())
        
        run_training_task(
            architecture, task_name, model_loc, dataset_dir, method, pooling, epochs, batch_size, overrides)

        


def run_training_task(architecture, task_name, model_loc, dataset_dir, method, pooling=None, epochs=25, batch_size=32, overrides={}):
    """Creates a model, trains it, and saves the model and training stats.

    Args:
        architecture (str): The architecture of the model to be trained. This must be a string corresponding to an application in Keras.applications.
        task_name (str): The name of the task. This is used to name the folder where the model and training stats are saved.
        model_loc (str): The location where the model and training stats are to be saved.
        dataset_dir (str): The Location of the dataset to be used for training.
        method (str): The method to be used in training the Model. This must be "transferlearn" or "pretrain"
        pooling (str, optional): A custom pooling method to be used. Must be from the pooling methods supported by Keras models. Defaults to None.
        epochs (int, optional): The number of epochs to train the model for. Defaults to 25.
        batch_size (int, optional): The batch size to be used for training. Defaults to 32.
        overrides (dict, optional): A dictionary of extra parameters to be passed to the model. Defaults to {}. The keys must be "optimizer_name", "l2reg", "dropout", and "learning_rate".

    Returns:
        (str): The location of the saved model and training stats.
    """
    # Check if GPU is available, and print a warning message if not
    if len(list_logical_devices('GPU')) == 0:
        cprint('WARNING: GPU is not available! Training will be slow.', "yellow")
    elif len(list_logical_devices('GPU')) == 1:
        cprint('SUCCESS: GPU is available!', "green")

    # Generating 3 datasets
    train_images, test_images, val_images = gen_subsets(
        dataset_dir, architecture, batch_size=batch_size)

    # retrieve a model created w given architecture and method
    model = create_model(
        architecture, # case sensitive!
        method.lower(), # method is case insensitive
        pooling=pooling.lower() if pooling else None, # pooling is case insensitive
        optimizer_name=overrides["optimizer_name"] if "optimizer_name" in overrides else None,
        l2reg=overrides["l2reg"] if "l2reg" in overrides else None,
        dropout=overrides["dropout"] if "dropout" in overrides else None,
        learning_rate=overrides["learning_rate"] if "learning_rate" in overrides else None
    )

    start = datetime.now()

    lr_reducer = ReduceLROnPlateau(factor=0.1,
                                   verbose=1,
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    csvlogger = CSVLogger(
        Path(Path(model_loc), "trainhistory.csv"), separator=",", append=False)

    # 15 gives a chance for reduceLR to kick in
    earlystopper = EarlyStopping(
        monitor='val_loss', mode="min", verbose=1, patience=15)

    callbacks = [lr_reducer, csvlogger, earlystopper]

    itemCt = Counter(train_images.classes)
    maxCt = float(max(itemCt.values()))
    cw = {clsID: maxCt/numImg for clsID, numImg in itemCt.items()}
    cprint(f"INFO: Sample discrepancy weights: {cw}", "blue")

    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=epochs,
        verbose=1, callbacks=callbacks,
        class_weight=cw
    )

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    score = model.evaluate(test_images)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    # Saving model and training stats
    model.save(model_loc)
    with open(Path(model_loc, "model_history"), 'wb') as pkl:
        pickle.dump(history.history, pkl)

    # Saving important data
    with open(Path(model_loc, "stats"), "w") as f:
        f.write(f"Training completed in time: {duration}\n")
        f.write(f"Test loss: {score[0]}\n")
        f.write(f"Test accuracy: {score[1]}")

    plot_data(history, Path(model_loc, f'{task_name}_plt.png'))
