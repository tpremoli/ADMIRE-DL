import keras.applications as apps
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.config import list_logical_devices
from pathlib import Path
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Flatten
from keras.models import Model
from keras import optimizers
from ..classes.constants import *

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()


def run():
    print("Devices: ", list_logical_devices())

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    data_loc = "/home/tpremoli/MRI_AD_Diagnosis/out/preprocessed_datasets/new_kaggle_method"

    train_images = train_datagen.flow_from_directory(
        Path(data_loc, "train"),
        target_size=KAGGLE_IMAGE_DIMENSIONS,
        batch_size=32,
        class_mode='categorical',
    )

    test_images = test_datagen.flow_from_directory(
        Path(data_loc, "test"),  # same directory as training data
        target_size=KAGGLE_IMAGE_DIMENSIONS,
        batch_size=32,
        class_mode='categorical',
    )

    validation_images = validation_datagen.flow_from_directory(
        Path(data_loc, "val"),  # same directory as training data
        target_size=KAGGLE_IMAGE_DIMENSIONS,
        batch_size=32,
        class_mode='categorical',
    )

    vgg16 = apps.VGG16(
        include_top=False,  # This is if we want the final FC layers
        weights="imagenet",
        # 3D as the imgs are same across 3 channels
        input_shape=KAGGLE_IMAGE_DIMENSIONS + [3],
        classifier_activation="softmax",
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

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    checkpoint = ModelCheckpoint(filepath='mymodel.h5',
                                 verbose=1, save_best_only=True)

    callbacks = [checkpoint, lr_reducer]

    start = datetime.now()
    history = model.fit(train_images,
                        steps_per_epoch=len(train_images),
                        epochs=18, verbose=5,
                        validation_data=validation_images,
                        validation_steps=len(validation_images))

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
        filedir, "../../out/trained_models").resolve()
    trained_models_path.mkdir(parents=True, exist_ok=True)

    model.save(Path(trained_models_path, "experiment_model"))
    plt.savefig(Path(trained_models_path, 'experiment_model.png'))
