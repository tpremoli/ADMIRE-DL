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
from .utils import gen_subsets, create_model, plot_data

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()


def run_training_task(architecture, run_name, dataset_dir, method, is_kaggle, pooling=None):
    trained_models_path = Path(
        filedir, "../../out/trained_models").resolve().mkdir(parents=True, exist_ok=True)

    print("Devices: ", list_logical_devices())

    # Generating 3 datasets
    train_images, test_images, val_images = gen_subsets(dataset_dir, is_kaggle)

    # retrieve a model created w given architecture and method
    model = create_model(architecture, method, is_kaggle)

    start = datetime.now()
    history = model.fit(train_images,
                        steps_per_epoch=len(train_images),
                        epochs=18, verbose=5,  # idk if should change this
                        validation_data=val_images,
                        validation_steps=len(val_images))

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    score = model.evaluate(test_images)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    model_loc = Path(trained_models_path, "{}_model".format(run_name))
    model.save(model_loc)
    
    with open(Path(model_loc,"stats"), "w") as f:
        f.write("Training completed in time: {}\n".format(duration))
        f.write("Test loss: {}".format(score[0]))
        f.write("Test accuracy: {}".format(score[1]))

    plot_data(history, Path(model_loc, '{}_plt.png'.format(run_name)))
