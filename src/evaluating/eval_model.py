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

def main():
    for modelpath in Path(cwd, "out/trained_models/adni").resolve().glob("*"):
        model = load_model(modelpath)
        config = load_config(modelpath)
        

        cprint(f"INFO: Loaded task {config['task_name']}", "blue")

        dataset = Path(cwd, config['dataset']).resolve()
        test = Path(dataset, 'test').resolve()
        val = Path(dataset, 'val').resolve()
        
        # getting the model's preprocessing function
        model_preprocessing_func = apps.__dict__[
            KERAS_APP_PKG[config["options"]["architecture"]]].preprocess_input
        
        calc_metrics(model, val, test, model_preprocessing_func, False, modelname=modelpath.name)
    
    
    # for f in sorted(test.rglob('*.png')):
    #     strpath = str(f)
    #     # The local path to our target image
    #     img = cv2.imread(strpath)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     X = np.expand_dims(img, axis=0).astype(np.float32)
    #     X = model_preprocessing_func(X)

    #     # Remove last layer's softmax
    #     preds = model.predict(X, verbose=0)
        
    #     if preds[0][0] > 0.5:
    #         label = "CN"
    #     else:
    #         label = "AD"
        
    #     conf = preds[0][0] if label == "CN" else 1 - preds[0][0]
        
    #     print(f"predicted {f.name} as {label}, expected {f.parent.name}. Confidence: {conf:.2f}")
        
