import tensorflow.keras.applications as apps
from termcolor import cprint
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
    output_count = 4 if is_kaggle else 2
    
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

        # convert output of base model to a 1D vector
        x = Flatten()(base_model.output)
        
        # We create fc_count fully connected layers, relu for all but the last
        for _ in range(fc_count - 1):
            x = Dense(units=4096, activation='relu')(x) # relu avoids vanishing gradient problem
            
        # The final layer is a softmax layer
        prediction = Dense(output_count, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=prediction)
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(learning_rate=0.001),
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
    
    if method == "pretrain":
        # Create the base model
        model = apps.__dict__[architecture](
            include_top=True,  # This is if we want the final FC layers
            weights=None,
            input_shape=input_shape,
            classifier_activation="softmax",
            pooling = pooling,
            classes = output_count # set the number of outputs to required count
        )
        
        if (fc_count > 1):
            cprint("WARNING: fc_count has no effect on pretrain method", "yellow")
        
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.SGD(learning_rate=0.01), # SGD is better for pretraining
                    metrics=['accuracy'])
        model.summary()
        
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

