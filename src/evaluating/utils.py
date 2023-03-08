from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import yaml
import numpy as np
import matplotlib.pyplot as plt

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
            
def draw_confusion_matrix(matrix):
    """Draws a confusion matrix

    Args:
        matrix (np.array): The confusion matrix
    """
    cm_display = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = ["AD", "CN"])

    cm_display.plot()
    plt.savefig("test.png") 
            
def calc_metrics(model, dataset, preprocessing_func, is_kaggle):
    """Calculates accuracy, F1, precision and recall, for a keras model and a dataset.

    Args:
        model (_type_): _description_
        dataset (_type_): _description_
        preprocessing_func (_type_): _description_
    """
    IMAGE_DIMENSIONS = KAGGLE_IMAGE_DIMENSIONS if is_kaggle else ADNI_IMAGE_DIMENSIONS

    
    datagen = ImageDataGenerator(preprocessing_function=preprocessing_func)
    test_flow = datagen.flow_from_directory(
        dataset,
        target_size=IMAGE_DIMENSIONS,
        class_mode='categorical' if is_kaggle else 'binary',
        shuffle = False,
    )
    
    y_true = test_flow.classes
    
    y_pred = model.predict(test_flow)

    y_pred = [np.rint(pred[0]) for pred in y_pred]
    
    print("keras eval:")
    print("test accuracy:",model.evaluate(test_flow)[1])
    print("sklearn eval:")
    print(classification_report(y_true, y_pred))
    print("confusion matrix:")
    matrix = confusion_matrix(y_true, y_pred, normalize='pred')
    print(matrix)
    draw_confusion_matrix(matrix)
        
