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
from .utils import gen_subsets, create_model

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def run_training_task(architecture, dataset_dir, method, is_kaggle, pooling=None):
    print("Devices: ", list_logical_devices())

    train_images, test_images, val_images = gen_subsets(dataset_dir,is_kaggle)

    model = create_model(architecture, is_kaggle)
    
    start = datetime.now()
    history = model.fit(train_images,
                        steps_per_epoch=len(train_images),
                        epochs=18, verbose=5,
                        validation_data=val_images,
                        validation_steps=len(val_images))

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model accuracy")
    plt.ylabel("“Accuracy”")
    plt.xlabel("“Epoch”")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "validation loss"])

    score = model.evaluate(test_images)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    trained_models_path = Path(
        filedir, "../../out/trained_models").resolve().mkdir(parents=True, exist_ok=True)

    model.save(Path(trained_models_path, "experiment_model"))
    plt.savefig(Path(trained_models_path, 'experiment_model.png'))

