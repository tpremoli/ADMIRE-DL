import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import applications as apps
from keras.models import load_model
from pathlib import Path
from termcolor import cprint

from .utils import load_config, calc_metrics
from ..constants import *

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def eval_all_models():
    for modelpath in sorted(Path(cwd, "out/trained_models").resolve().glob("*")):
        model = load_model(modelpath)
        config = load_config(modelpath)
        

        cprint(f"INFO: Loaded task {config['task_name']}", "blue")

        dataset = Path(cwd, config['dataset']).resolve()
        test = Path(dataset, 'test').resolve()
        val = Path(dataset, 'val').resolve()
        
        # getting the model's preprocessing function
        model_preprocessing_func = apps.__dict__[
            KERAS_APP_PKG[config["options"]["architecture"]]].preprocess_input
        
        calc_metrics(model, val, test, model_preprocessing_func, modelname=modelpath.name)
    