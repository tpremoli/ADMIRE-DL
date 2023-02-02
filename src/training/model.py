import tensorflow.keras.applications as apps
from keras.layers import Dense, Flatten
from keras.models import Model
from keras import optimizers
from pathlib import Path
from ..constants import *

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def create_model(architecture, is_kaggle, method="transferlearn", pooling=None, fc_count=1):
    """Creates a model based on the architecture and method specified.

    Args:
        architecture (str): The architecture to use in the model. Has to be a valid architecture in the keras.applications module.
        is_kaggle (bool): If the kaggle dataset is being used.
        method (str, optional): The method to use in the model. Can be "pretrain" or "transferlearn". Defaults to "transferlearn".
        pooling (str, optional): The type of pooling to use in the model. Defaults to None.
        fc_count (int, optional): The number of fully-connected layers to use. Defaults to 1.

    Returns:
        Model: returns a keras model with the specified parameters
    """
    input_shape = (KAGGLE_IMAGE_DIMENSIONS if is_kaggle else ADNI_IMAGE_DIMENSIONS)  + [3]
    
    if method == "transferlearn":

        # Create the base model
        base_model = apps.__dict__[architecture](
            include_top=False,  # This is if we want the final FC layers
            weights="imagenet",
            # 3D as the imgs are same across 3 channels
            input_shape=input_shape,
            classifier_activation="softmax",
            pooling = pooling,
        )

        # Freeze the base model
        for layer in base_model.layers:
            layer.trainable = False

        output_count = 4 if is_kaggle else 2
        
        # Converting the output of the base model to a 1D vector
        x = Flatten()(base_model.output)
        
        # Create the fully-connected layers
        for _ in range(fc_count):
            x = Dense(output_count, activation='softmax')(x)
        
        # create the model
        model = Model(inputs=base_model.input, outputs=x)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(learning_rate=0.001),
                    metrics=['accuracy'])
        model.summary()
        
        return model
