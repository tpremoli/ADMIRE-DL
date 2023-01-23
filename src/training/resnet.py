import tensorflow.keras.applications as apps
from tensorflow.config import list_logical_devices
from keras.layers import Dense, Flatten
from keras.models import Model
from keras import optimizers
from pathlib import Path
from ..constants import *

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def create_resnet50(is_kaggle, method="finetune", pooling=None):
    if method == "finetune":
        resnet50 = apps.ResNet50(
            include_top=False,  # This is if we want the final FC layers
            weights="imagenet",
            # 3D as the imgs are same across 3 channels
            input_shape=(KAGGLE_IMAGE_DIMENSIONS if is_kaggle else ADNI_IMAGE_DIMENSIONS)  + [3],
            classifier_activation="softmax",
            pooling = pooling,
        )

        for layer in resnet50.layers:
            layer.trainable = False

        x = Flatten()(resnet50.output)

        prediction = Dense(4, activation='softmax')(x)  # there's 4 categories
        model = Model(inputs=resnet50.input, outputs=prediction)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(),
                    metrics=['accuracy'])
        model.summary()
        
        return model

def create_resnet157(is_kaggle, method="finetune", pooling=None):
    if method == "finetune":
        resnet157 = apps.ResNet157(
            include_top=False,  # This is if we want the final FC layers
            weights="imagenet",
            # 3D as the imgs are same across 3 channels
            input_shape=(KAGGLE_IMAGE_DIMENSIONS if is_kaggle else ADNI_IMAGE_DIMENSIONS)  + [3],
            classifier_activation="softmax",
            pooling = pooling,
        )

        for layer in resnet157.layers:
            layer.trainable = False

        x = Flatten()(resnet157.output)

        prediction = Dense(4, activation='softmax')(x)  # there's 4 categories
        model = Model(inputs=resnet157.input, outputs=prediction)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(),
                    metrics=['accuracy'])
        model.summary()
        
        return model

