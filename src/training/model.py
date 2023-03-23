import tensorflow.keras.applications as apps
from termcolor import cprint
from keras import regularizers
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras import optimizers
from pathlib import Path
from ..constants import *

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def create_model(architecture, is_kaggle, method="transferlearn", pooling=None, optimizer_name="Adam", l2reg=None, dropout=None, learning_rate=None):
    """Creates a model based on the architecture and method specified.

    Args:
        architecture (str): The architecture to use in the model. Has to be a valid architecture in the keras.applications module.
        is_kaggle (bool): If the kaggle dataset is being used.
        method (str, optional): The method to use in the model. Can be "pretrain" or "transferlearn". Defaults to "transferlearn".
        pooling (str, optional): The type of pooling to use in the model. Defaults to None.
        optimizer_name (str, optional): The name of the optimizer to use in the model. Defaults to "Adam".
        l2reg (float, optional): The l2 regularization to use in the model. Defaults to None.
        dropout (float, optional): The dropout to use in the model. Defaults to None.
        learning_rate (float, optional): The learning rate to use in the model. Defaults to None.

    Returns:
        Model: returns a keras model with the specified parameters
    """
    input_shape = (KAGGLE_IMAGE_DIMENSIONS if is_kaggle else ADNI_IMAGE_DIMENSIONS)  + [3]
    output_count = 4 if is_kaggle else 1
    
    if method == "transferlearn":
        # Create the base model
        base_model = apps.__dict__[architecture](
            include_top=False,  # This is if we want the final FC layers
            weights="imagenet",
            # 3D as the imgs are same across 3 channels
            input_shape=input_shape,
            pooling=pooling,
        )

        # Freeze the base model
        for layer in base_model.layers:
            layer.trainable = False
        
        if dropout:
            # dropout is used to prevent overfitting
            x = Dropout(dropout)(base_model.output)
            x = Flatten()(x)
            cprint(f"INFO: Dropout layer is enabled. P={dropout}", "blue")
        else:
            # convert output of base model to a 1D vector
            cprint(f"INFO: Dropout is disabled.", "blue")
            x = Flatten()(base_model.output)
        
        # VGG models have 3 FC layers
        if architecture == "VGG16" or architecture == "VGG19":
            x = Dense(units=4096, activation='relu')(x)
            x = Dense(units=4096, activation='relu')(x)
        
        # The final layer is a softmax layer
        prediction = Dense(output_count, activation='softmax' if is_kaggle else "sigmoid")(x)
    
        model = Model(inputs=base_model.input, outputs=prediction)
        
        # adding regularization
        if l2reg:
            cprint(f"INFO: L2 regularization is enabled. Lambda={l2reg}", "blue")
            regularizer = regularizers.L2(l2reg)
            for layer in model.layers:
                for attr in ['activity_regularizer']:
                    if hasattr(layer, attr):
                        setattr(layer, attr, regularizer)

        # Setting optimizer. Adam is the default optimizer, however SGD can be stronger
        optimizer_name = optimizer_name.upper()
        if optimizer_name == "ADAM":
            cprint(f"INFO: Using Adam optimizer with learning rate {learning_rate if learning_rate else 0.001}", "blue")
            optimizer = optimizers.Adam(learning_rate=learning_rate if learning_rate else 0.001)
        elif optimizer_name == "SGD":
            cprint(f"INFO: Using SGD optimizer with learning rate {learning_rate if learning_rate else 0.0003} and momentum 0.9", "blue")
            optimizer = optimizers.SGD(learning_rate=learning_rate if learning_rate else 0.0003, momentum=0.9)
        elif optimizer_name == "RMSPROP":
            cprint(f"INFO: Using RMSprop optimizer with learning rate {learning_rate if learning_rate else 0.001}", "blue")
            optimizer = optimizers.RMSprop(learning_rate=learning_rate if learning_rate else 0.001)
        else:
            raise ValueError("Unsupported optimizer: " + optimizer_name + ". Supported optimizers are: Adam, SGD")
        
        model.compile(loss='categorical_crossentropy' if is_kaggle else 'binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])
        
        # categorical_crossentropy is used for multi-class classification:
        # https://www.sciencedirect.com/science/article/pii/S1389041718309562
        # https://link.springer.com/chapter/10.1007/978-3-319-70772-3_20
        # https://link.springer.com/chapter/10.1007/978-3-030-05587-5_34 
        # https://arxiv.org/abs/1809.03972
        
        # Accuracy metric:
        # https://www.sciencedirect.com/science/article/pii/S1389041718309562 
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7661929/
        # https://arxiv.org/abs/1809.03972 NOTE: this paper uses the sensitivity and specificity metrics as well
        
        
        return model
    
    if method == "pretrain":
        # Create the base model
        model = apps.__dict__[architecture](
            include_top=True,  # This is if we want the final FC layers
            weights=None,
            input_shape=input_shape,
            classifier_activation="softmax" if is_kaggle else "sigmoid",
            pooling = pooling,
            classes = output_count # set the number of outputs to required count
        )
        
        if l2reg:
            cprint(f"INFO: L2 regularization is enabled. Lambda={l2reg}", "blue")
            regularizer = regularizers.L2(l2reg)
            for layer in model.layers:
                for attr in ['activity_regularizer']:
                    if hasattr(layer, attr):
                        setattr(layer, attr, regularizer)
        
        model.compile(loss='categorical_crossentropy' if is_kaggle else 'binary_crossentropy',
                    optimizer=optimizers.SGD(learning_rate=0.0003, momentum=0.9), # SGD is better for pretraining
                    metrics=['accuracy'])
        
        # categorical_crossentropy is used for multi-class classification:
        # https://www.sciencedirect.com/science/article/pii/S1389041718309562
        # https://link.springer.com/chapter/10.1007/978-3-319-70772-3_20
        # https://link.springer.com/chapter/10.1007/978-3-030-05587-5_34 
        # https://arxiv.org/abs/1809.03972
        
        # Accuracy metric:
        # https://www.sciencedirect.com/science/article/pii/S1389041718309562 
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7661929/
        # https://arxiv.org/abs/1809.03972 NOTE: this paper uses the sensitivity and specificity metrics as well
        
        
        # TODO: maybe add a setting to print the model summary
        # model.summary()
        
        return model
