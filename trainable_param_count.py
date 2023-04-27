"""This script counts the number of trainable parameters in each model
in the out/trained_models directory and writes the results to
out/trainable_param_counts.txt.
"""

from pathlib import Path
from keras.utils.layer_utils import count_params
from keras.models import load_model

if __name__ == "__main__":
    print("Counting trainable parameters...")
    
    with open("out/trainable_param_counts.txt", "w") as f:
        f.write("Model: Trainable Parameters\n")
    
        for path in Path("out/trained_models").glob("*"):
            if path.is_dir():
                # Load every model
                model = load_model(path)
                trainable_count = count_params(model.trainable_weights)
                f.write(f"{path.name}: {trainable_count}")