from pathlib import Path
from .constants import *

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def load_scans_from_folder(folder):
    scan_folder = Path(cwd, folder).resolve()
    
    print("Not a kaggle dataset")
