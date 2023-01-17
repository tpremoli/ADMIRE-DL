from pathlib import Path
from ..classes.scan import Scan
from ..classes.constants import *

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def load_scans_from_folder(folder, kaggle=False):
    scan_folder = Path(cwd, folder).resolve()
    
    if kaggle:
        print("Utilizing Kaggle dataset. Loading up scans")
    else:
        print("Not a kaggle dataset")