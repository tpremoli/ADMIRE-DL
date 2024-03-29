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


def create_model(architecture, method="transferlearn", pooling=None, optimizer_name=None, l2reg=None, dropout=None, learning_rate=None):
    """Creates a model based on the architecture and method specified.

    Args:
        architecture (str): The architecture to use in the model. Has to be a valid architecture in the keras.applications module.
        method (str, optional): The method to use in the model. Can be "pretrain" or "transferlearn". Defaults to "transferlearn".
        pooling (str, optional): The type of pooling to use in the model. Defaults to None.
        optimizer_name (str, optional): The name of the optimizer to use in the model. Defaults to None.
        l2reg (float, optional): The l2 regularization to use in the model. Defaults to None.
        dropout (float, optional): The dropout to use in the model. Defaults to None.
        learning_rate (float, optional): The learning rate to use in the model. Defaults to None.

    Returns:
        Model: returns a keras model with the specified parameters
    """
    input_shape = (ADNI_IMAGE_DIMENSIONS) + [3]
    output_count = 1
    
    # The defaults depend on the method. If transfer learning is used, Adam is the default optimizer. If pretraining is used, SGD is the default optimizer.
    if optimizer_name == None:
        optimizer_name = "Adam" if method == "transferlearn" else "SGD"
        
    optimizer_name = optimizer_name.upper()
    if optimizer_name == "ADAM":
        cprint(
            f"INFO: Using Adam optimizer with learning rate {learning_rate if learning_rate else 0.001}", "blue")
        optimizer = optimizers.Adam(
            learning_rate=learning_rate if learning_rate else 0.001)
    elif optimizer_name == "SGD":
        cprint(
            f"INFO: Using SGD optimizer with learning rate {learning_rate if learning_rate else 0.0003} and momentum 0.9", "blue")
        optimizer = optimizers.SGD(
            learning_rate=learning_rate if learning_rate else 0.0003, momentum=0.9)
    elif optimizer_name == "RMSPROP":
        cprint(
            f"INFO: Using RMSprop optimizer with learning rate {learning_rate if learning_rate else 0.001}", "blue")
        optimizer = optimizers.RMSprop(
            learning_rate=learning_rate if learning_rate else 0.001)
    else:
        raise ValueError("Unsupported optimizer: " +
                            optimizer_name + ". Supported optimizers are: Adam, SGD")

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
            # having Flatten after a global pooling layer is redundant, but doesn't hurt
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
        prediction = Dense(
            output_count, activation="sigmoid")(x)

        model = Model(inputs=base_model.input, outputs=prediction)

        # adding regularization
        if l2reg:
            cprint(
                f"INFO: L2 regularization is enabled. Lambda={l2reg}", "blue")
            regularizer = regularizers.L2(l2reg)
            for layer in model.layers:
                for attr in ['activity_regularizer']:
                    if hasattr(layer, attr):
                        setattr(layer, attr, regularizer)

        model.compile(loss='binary_crossentropy',
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
        base_model = apps.__dict__[architecture](
            include_top=False,  # This is if we want the final FC layers
            weights=None,
            input_shape=input_shape,
            pooling=pooling,
        )

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
        prediction = Dense(
            output_count, activation="sigmoid")(x)

        model = Model(inputs=base_model.input, outputs=prediction)

        # adding regularization
        if l2reg:
            cprint(
                f"INFO: L2 regularization is enabled. Lambda={l2reg}", "blue")
            regularizer = regularizers.L2(l2reg)
            for layer in model.layers:
                for attr in ['activity_regularizer']:
                    if hasattr(layer, attr):
                        setattr(layer, attr, regularizer)

        model.compile(loss='binary_crossentropy',
                      # SGD is better for pretraining
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

        # TODO: maybe add a setting to print the model summary
        # model.summary()

        return model
