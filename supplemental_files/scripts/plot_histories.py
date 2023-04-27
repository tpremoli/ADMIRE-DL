"""
This script plots the training and validation accuracy for each model
in the out/trained_models directory. It saves the plots in the out/model_history_plots
directory.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_history(history, plot_path):
    """Plots the training and validation accuracy for a model.

    Args:
        history (df): A dataframe containing the training history of a model.
        plot_path (str): The path to save the plot to.
    """
    # plot the training and validation accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["epoch"], history["accuracy"], label="Training Accuracy")
    plt.plot(history["epoch"], history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend(loc="lower left")
    plt.savefig(plot_path)

if __name__ == "__main__":
    print("Plotting model histories...")
    
    outpath = Path("out/model_history_plots")
    outpath.mkdir(exist_ok=True)

    for path in Path("out/trained_models").glob("*"):
        if path.is_dir():
            print("Plotting history for", path.name)
            history = pd.read_csv(path / "trainhistory.csv")
            plot_history(history, outpath/f"{path.name}.png")

