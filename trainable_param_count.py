"""This script counts the number of trainable parameters in each model
in the out/trained_models directory and writes the results to
out/trainable_param_counts.csv.
"""

from pathlib import Path
from keras.utils.layer_utils import count_params
from keras.models import load_model
from termcolor import cprint

if __name__ == "__main__":
    print("Counting trainable parameters...")
    
    # open the file to write to as csv
    with open("out/trainable_param_counts.csv", "w") as f:
        f.write("model,trainable_params\n")
    
        for path in sorted(Path("out/trained_models").glob("*")):
            if path.is_dir():
                # Load every model
                model = load_model(path)
                trainable_count = count_params(model.trainable_weights)
                f.write(f"{path.name},{trainable_count}\n")
                cprint("Counted trainable parameters for " + path.name, "green")