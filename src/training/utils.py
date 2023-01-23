from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from ..classes.constants import *
from .vgg import create_vgg16, create_vgg19
import matplotlib.pyplot as plt

def gen_subsets(dataset_dir, is_kaggle, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255)
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


def create_model(architecture, method, is_kaggle):
    if architecture == VGG_16:
        return create_vgg16(is_kaggle, method)
    if architecture == VGG_19:
        return create_vgg19(is_kaggle, method)
    if architecture == RES_NET_50:
        pass
    if architecture == RES_NET_157:
        pass
    if architecture == DENSE_NET_121:
        pass
    if architecture == DENSE_NET_201:
        pass

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
