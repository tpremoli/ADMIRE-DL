import yaml
import shutil
from tensorflow.config import list_logical_devices
from datetime import datetime
from pathlib import Path
from ..classes.constants import *
from .utils import gen_subsets, create_model, plot_data

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def load_training_task(file_loc):
    with open(Path(cwd, file_loc).resolve(), "r") as f:
        yamlfile = yaml.safe_load(f)
        
        keys = yamlfile.keys()
        if "task_name" not in keys:
            raise ValueError("Task config requires a task_name attribute!")
        if "dataset" not in keys:
            raise ValueError("Task config requires a dataset attribute!")

        for path in Path(filedir, "../../out/trained_models").glob(yamlfile["task_name"]):        
            raise ValueError("Task with name {} already exists in {}!".format(yamlfile["task_name"], path))

        options = yamlfile["options"]
        optionkeys = options.keys()
        
        if "architecture" not in optionkeys:
            raise ValueError("Task config requires an architecture attribute!")
        if "approach" not in optionkeys:
            raise ValueError("Task config requires an approach attribute!")
        if "method" not in optionkeys:
            raise ValueError("Task config requires a method attribute!")
        if "kaggle" not in optionkeys:
            raise ValueError("Task config requires a kaggle attribute!")

        architecture = yamlfile["options"]["architecture"]
        task_name = yamlfile["task_name"]
        dataset_dir = Path(cwd,yamlfile["dataset"]).resolve()
        method = yamlfile["options"]["method"]
        is_kaggle = yamlfile["options"]["kaggle"]
        
        approach = yamlfile["options"]["approach"]
        pooling = yamlfile["options"]["pooling"]

        model_loc = run_training_task(architecture, task_name, dataset_dir, method, is_kaggle)

        shutil.copyfile(Path(cwd, file_loc).resolve(), Path(model_loc, "config.yml").resolve())

def run_training_task(architecture, task_name, dataset_dir, method, is_kaggle, approach=None, pooling=None):
    trained_models_path = Path(
        filedir, "../../out/trained_models").resolve()
    
    trained_models_path.mkdir(parents=True, exist_ok=True)

    print("Devices: ", list_logical_devices())

    # Generating 3 datasets
    train_images, test_images, val_images = gen_subsets(dataset_dir, is_kaggle)

    # retrieve a model created w given architecture and method
    model = create_model(architecture, method, is_kaggle)

    start = datetime.now()
    history = model.fit(train_images,
                        steps_per_epoch=len(train_images),
                        epochs=18, verbose=5,  # idk if should change this
                        validation_data=val_images,
                        validation_steps=len(val_images))

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    score = model.evaluate(test_images)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    print(trained_models_path,task_name)
    model_loc = Path(trained_models_path, task_name)
    model.save(model_loc)
    
    with open(Path(model_loc,"stats"), "w") as f:
        f.write("Training completed in time: {}\n".format(duration))
        f.write("Test loss: {}".format(score[0]))
        f.write("Test accuracy: {}".format(score[1]))

    plot_data(history, Path(model_loc, '{}_plt.png'.format(task_name)))
    
    return model_loc
