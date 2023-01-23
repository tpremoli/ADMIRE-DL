import tensorflow.keras.applications as apps
from tensorflow.config import list_logical_devices
from keras.layers import Dense, Flatten
from keras.models import Model
from keras import optimizers
from pathlib import Path
from ..constants import *

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

def create_vgg19(is_kaggle, method="finetune", pooling=None):
    if method == "finetune":
        vgg19 = apps.VGG19(
            include_top=False,  # This is if we want the final FC layers
            weights="imagenet",
            # 3D as the imgs are same across 3 channels
            input_shape=(KAGGLE_IMAGE_DIMENSIONS if is_kaggle else ADNI_IMAGE_DIMENSIONS)  + [3],
            classifier_activation="softmax",
            pooling = pooling,
        )

        for layer in vgg19.layers:
            layer.trainable = False

        x = Flatten()(vgg19.output)

        prediction = Dense((4 if is_kaggle else 2)  + [3], activation='softmax')(x)  # there's 4 categories
        model = Model(inputs=vgg19.input, outputs=prediction)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(),
                    metrics=['accuracy'])
        model.summary()
        
        return model

    
