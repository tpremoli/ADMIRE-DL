import tensorflow.keras.applications as apps
from keras.layers import Dense, Flatten
from keras.models import Model
from keras import optimizers
from ..classes.constants import *
from ..classes.utils import load_scans_from_folder

def run():
    vgg16 = apps.VGG16(
        include_top=False, # This is if we want the final FC layers
        weights="imagenet",
        input_shape=KAGGLE_IMAGE_DIMENSIONS  + [3], # 3D as the imgs are same across 3 channels
        classifier_activation="softmax",
    )
    
    for layer in vgg16.layers:
        layer.trainable = False
        
        
    x = Flatten()(vgg16.output)
    
    prediction = Dense(4, activation='softmax')(x)
    model = Model(inputs=vgg16.input, outputs=prediction)
    model.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.Adam(),
                        metrics=['accuracy'])
    model.summary()

        
    scans = load_scans_from_folder("out/preprocessed_datasets/kaggle_processed", kaggle=True)
    
    