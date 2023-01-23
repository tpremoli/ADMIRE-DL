from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from ..classes.constants import *
from .vgg import create_vgg16
import tensorflow.keras.applications as apps
import numpy as np

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


def create_model(architecture, is_kaggle):
    if architecture == VGG_16:
        return create_vgg16(is_kaggle)
    if architecture == VGG_19:
        pass
    if architecture == RES_NET_50:
        pass
    if architecture == RES_NET_157:
        pass
    if architecture == DENSE_NET_121:
        pass
    if architecture == DENSE_NET_201:
        pass