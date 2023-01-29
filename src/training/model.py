import tensorflow.keras.applications as apps
from keras.layers import Dense, Flatten
from keras.models import Model
from keras import optimizers
from pathlib import Path
from ..constants import *

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def create_model(architecture, is_kaggle, method="finetune", pooling=None, fc_count=1):
    if method == "finetune":
        resnet = apps.__dict__[architecture](
            include_top=False,  # This is if we want the final FC layers
            weights="imagenet",
            # 3D as the imgs are same across 3 channels
            input_shape=(KAGGLE_IMAGE_DIMENSIONS if is_kaggle else ADNI_IMAGE_DIMENSIONS)  + [3],
            classifier_activation="softmax",
            pooling = pooling,
        )

        for layer in resnet.layers:
            layer.trainable = False

        x = Flatten()(resnet.output)

        prediction = Dense((4 if is_kaggle else 2), activation='softmax')(x)  # there's 4 categories
        model = Model(inputs=resnet.input, outputs=prediction)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(),
                    metrics=['accuracy'])
        model.summary()
        
        return model
