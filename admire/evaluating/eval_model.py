import numpy as np
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications as apps
from keras.models import load_model
from pathlib import Path
from termcolor import cprint

from .utils import draw_confusion_matrix, get_final_y, load_config
from ..constants import *
from ..settings import *

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def eval_all_models():
    for modelpath in sorted(Path(cwd, "out/trained_models").resolve().glob("*")):
        model = load_model(modelpath)
        config = load_config(modelpath)
        

        cprint(f"INFO: Loaded task {config['task_name']}", "blue")

        dataset = Path(cwd, config['dataset']).resolve()
        
        if EVAL_OASIS:
            test = Path(filedir, 
                        "../../out/preprocessed_datasets/oasis_processed/axial_slices").resolve()
        else:
            test = Path(dataset, 'test').resolve()
        
        # getting the model's preprocessing function
        model_preprocessing_func = apps.__dict__[
            KERAS_APP_PKG[config["options"]["architecture"]]].preprocess_input
        
        calc_metrics(model, test, model_preprocessing_func, modelname=modelpath.name)

def calc_metrics(model, testdata, preprocessing_func, modelname):
    """Calculates accuracy, F1, precision and recall, for a keras model and a dataset.

    Args:
        model (_type_): _description_
        dataset (_type_): _description_
        preprocessing_func (_type_): _description_
    """
    IMAGE_DIMENSIONS = ADNI_IMAGE_DIMENSIONS
    
    datagen = ImageDataGenerator(preprocessing_function=preprocessing_func)
    test_flow = datagen.flow_from_directory(
        testdata,
        target_size=IMAGE_DIMENSIONS,
        class_mode='binary',
        shuffle = False,
    )
    
    # Get the true and predicted labels
    y_true = test_flow.classes
    y_pred = model.predict(test_flow)
    
    # Round the predictions to the nearest integer
    y_pred = [np.rint(pred[0]) for pred in y_pred]
    
    # Aggregate the predictions into groups of 10 for final evaluation (majority vote)
    y_true, y_pred = get_final_y(y_true, y_pred)
    
    out_dir = Path(cwd, "out/evals").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate the metrics
    this_report = classification_report(y_true, y_pred, target_names=["AD", "CN"], digits=4, output_dict=True)
    
    # Write the metrics to a YAML file
    written_dict = {"keras eval": {"test accuracy": model.evaluate(test_flow)[1], "sklearn eval": this_report}}
    
    with open(Path(out_dir,f"eval_{modelname}.yml").resolve(), "w") as f:
        yaml.dump(written_dict, f)
    
    # Draw the confusion matrix
    if DRAW_CONFUSION_MATRIX:
        matrix = confusion_matrix(y_true, y_pred, normalize='pred')
        draw_confusion_matrix(matrix, modelname, out_dir)
