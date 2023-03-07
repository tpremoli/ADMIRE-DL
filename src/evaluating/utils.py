from pathlib import Path
import yaml

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def load_config(path):
    """Loads a configuration file from the given path into a dict

    Args:
        path (str): The location of the task_config.yml file.
    """
    finalpath = Path(cwd, path).resolve()
    
    with open(finalpath, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)