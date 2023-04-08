from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
import yaml
import numpy as np
import matplotlib.pyplot as plt

from ..constants import *
from ..settings import *

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

# TODO finish doccing the evaluation functions
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
                    
def extract_metrics_from_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
        sklearn_eval = data['keras eval']['sklearn eval']
        accuracy = sklearn_eval['accuracy']
        avg_metrics = sklearn_eval['macro avg']
        weighted_metrics = sklearn_eval['weighted avg']
        cn_metrics = sklearn_eval['CN']
        ad_metrics = sklearn_eval['AD']
    return accuracy, avg_metrics, weighted_metrics, cn_metrics, ad_metrics

