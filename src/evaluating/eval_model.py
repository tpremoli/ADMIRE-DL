import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import load_model
from pathlib import Path

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

model = load_model('out/trained_models/adni/vgg16_pretrain_slices')

architecture = "VGG16"
is_kaggle = False
dataset_dir = "/home/tpremoli/MRI_AD_Diagnosis/out/preprocessed_datasets/adni_processed/slice_dataset"

model.summary()
