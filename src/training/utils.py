from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from ..constants import *
import tensorflow.keras.applications as apps
import matplotlib.pyplot as plt

def gen_subsets(dataset_dir, is_kaggle, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255) # preprocessing_function=apps.resnet.preprocess_input
    test_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255) 

    train_images = train_datagen.flow_from_directory(
        Path(dataset_dir, "train"),
        target_size=KAGGLE_IMAGE_DIMENSIONS if is_kaggle else ADNI_IMAGE_DIMENSIONS,
        batch_size=batch_size, # What batch size??
        class_mode='categorical',
    )

    test_images = test_datagen.flow_from_directory(
        Path(dataset_dir, "test"),  # same directory as training data
        target_size=KAGGLE_IMAGE_DIMENSIONS if is_kaggle else ADNI_IMAGE_DIMENSIONS,
        batch_size=batch_size, # What batch size??
        class_mode='categorical',
    )

    val_images = validation_datagen.flow_from_directory(
        Path(dataset_dir, "val"),  # same directory as training data
        target_size=KAGGLE_IMAGE_DIMENSIONS if is_kaggle else ADNI_IMAGE_DIMENSIONS,
        batch_size=batch_size, # What batch size??
        class_mode='categorical',
    )
    
    return train_images, test_images, val_images,


def plot_data(history, save_loc):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "validation loss"])
    plt.savefig(save_loc)
