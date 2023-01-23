import tensorflow.keras.applications as apps
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.config import list_logical_devices
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.models import Model
from datetime import datetime
from keras import optimizers
from pathlib import Path
from ..classes.constants import *
from .utils import gen_subsets

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def create_vgg16(is_kaggle, method="finetune", pooling=None):
    if method == "finetune":
        vgg16 = apps.VGG16(
            include_top=False,  # This is if we want the final FC layers
            weights="imagenet",
            # 3D as the imgs are same across 3 channels
            input_shape=(KAGGLE_IMAGE_DIMENSIONS if is_kaggle else ADNI_IMAGE_DIMENSIONS)  + [3],
            classifier_activation="softmax",
            pooling = pooling,
        )

        for layer in vgg16.layers:
            layer.trainable = False

        x = Flatten()(vgg16.output)

        prediction = Dense(4, activation='softmax')(x)  # there's 4 categories
        model = Model(inputs=vgg16.input, outputs=prediction)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(),
                    metrics=['accuracy'])
        model.summary()
        
        return model
    
    