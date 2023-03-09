from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from ..constants import *
from ..settings import *

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def load_config(path):
    """Loads a configuration file from the given path into a dict

    Args:
        path (str): The location of the task_config.yml file.
    """
    finalpath = Path(cwd, path).resolve()
    
    if finalpath.name != "task_config.yml":
        finalpath = Path(finalpath, "task_config.yml").resolve()

    with open(finalpath, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
def draw_confusion_matrix(matrix, modelpath,out_dir):
    """Draws a confusion matrix

    Args:
        matrix (np.array): The confusion matrix
        modelpath (str): The path to the model
        out_dir (str): The path to the output directory
    """
    cm_display = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = ["AD", "CN"])

    cm_display.plot()
    plt.savefig(Path(out_dir,f"{modelpath}.png")) 
    
def get_final_y(y_true, y_pred):
    """Reduces a list of 10 predictions to a single prediction

    Args:
        y_true (list): The list of true labels
        y_pred (list): The list of predicted labels

    Returns:
        list: The reduced list of true labels
        list: The reduced list of predicted labels
    """
    final_true = []
    for i in range(0, len(y_true), 10):
        final_true.append(np.rint(np.mean(y_true[i:i+10])))
    
    final_pred = []
    for i in range(0, len(y_pred), 10):
        final_pred.append(np.rint(np.mean(y_pred[i:i+10])))
        
    return final_true, final_pred
            
def calc_metrics(model, valdata, testdata, preprocessing_func, is_kaggle, modelname):
    """Calculates accuracy, F1, precision and recall, for a keras model and a dataset.

    Args:
        model (_type_): _description_
        dataset (_type_): _description_
        preprocessing_func (_type_): _description_
    """
    IMAGE_DIMENSIONS = KAGGLE_IMAGE_DIMENSIONS if is_kaggle else ADNI_IMAGE_DIMENSIONS

    
    datagen = ImageDataGenerator(preprocessing_function=preprocessing_func)
    test_flow = datagen.flow_from_directory(
        testdata,
        target_size=IMAGE_DIMENSIONS,
        class_mode='categorical' if is_kaggle else 'binary',
        shuffle = False,
    )
    
    y_true = test_flow.classes
    y_pred = model.predict(test_flow)
    y_pred = [np.rint(pred[0]) for pred in y_pred]
    test_true, test_pred = get_final_y(y_true, y_pred)
    
    val_flow = datagen.flow_from_directory(
        valdata,
        target_size=IMAGE_DIMENSIONS,
        class_mode='categorical' if is_kaggle else 'binary',
        shuffle = False,
    )
    
    y_true = val_flow.classes
    y_pred = model.predict(val_flow)
    y_pred = [np.rint(pred[0]) for pred in y_pred]
    val_true, val_pred = get_final_y(y_true, y_pred)
    
    y_true = np.concatenate((test_true, val_true))
    y_pred = np.concatenate((test_pred, val_pred))
    
    out_dir = Path(cwd, "out/evals").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open (Path(out_dir,f"{modelname}.txt").resolve(), "w") as f:
        f.write("keras eval:")
        f.write("\ntest accuracy:")
        f.write(str(model.evaluate(test_flow)[1]))
        f.write("\nsklearn eval:\n")
        f.write(classification_report(y_true, y_pred, target_names=["AD", "CN"]))
        matrix = confusion_matrix(y_true, y_pred, normalize='pred')
        draw_confusion_matrix(matrix, modelname, out_dir)
        
