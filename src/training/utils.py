from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from ..constants import *
import tensorflow.keras.applications as apps
import matplotlib.pyplot as plt


def gen_subsets(dataset_dir, is_kaggle, architecture, batch_size=32):
    """Generates 3 ImageDataGenerators for the train, test, and validation subsets of the dataset.

    Args:
        dataset_dir (str): The directory containing the preprocessed dataset
        is_kaggle (bool): If the kaggle dataset is being used.
        batch_size (int, optional): The batch size to use for the ImageDataGenerators. Defaults to 32.

    Returns:
        (tuple): returns a tuple of 3 ImageDataGenerators for the train, test, and validation subsets of the dataset.
    """
    # train_datagen = ImageDataGenerator(rescale=1./255) # preprocessing_function=apps.resnet.preprocess_input
    # test_datagen = ImageDataGenerator(rescale=1./255)
    # validation_datagen = ImageDataGenerator(rescale=1./255)

    # This gets the preprocess_input func i.e apps.resnet.preprocess_input
    preprocessing_func = apps.__dict__[
        KERAS_APP_PKG[architecture]].preprocess_input

    train_datagen = ImageDataGenerator(preprocessing_function=preprocessing_func,
                                         zoom_range=0.05, 
                                         width_shift_range=0.05, 
                                         height_shift_range=0.05, 
                                         vertical_flip=True)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_func)
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocessing_func)

    IMAGE_DIMENSIONS = KAGGLE_IMAGE_DIMENSIONS if is_kaggle else ADNI_IMAGE_DIMENSIONS

    train_images = train_datagen.flow_from_directory(
        Path(dataset_dir, "train"),
        target_size=IMAGE_DIMENSIONS,
        batch_size=batch_size, 
        class_mode='categorical' if is_kaggle else 'binary',
    )

    test_images = test_datagen.flow_from_directory(
        Path(dataset_dir, "test"),  # same directory as training data
        target_size=IMAGE_DIMENSIONS,
        batch_size=batch_size,  # What batch size??
        class_mode='categorical' if is_kaggle else 'binary',
    )

    val_images = validation_datagen.flow_from_directory(
        Path(dataset_dir, "val"),  # same directory as training data
        target_size=IMAGE_DIMENSIONS,
        batch_size=batch_size,  # What batch size??
        class_mode='categorical' if is_kaggle else 'binary',
    )

    return train_images, test_images, val_images,


def plot_data(history, save_loc):
    """Plots the accuracy and loss of the model over the epochs.

    Args:
        history (model): The history of the model
        save_loc (str): The location to save the plot
    """
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "validation loss"])
    plt.savefig(save_loc)
