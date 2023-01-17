from pathlib import Path
from ..classes.scan import Scan
from ..classes.constants import *

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def load_scans_from_folder(folder, kaggle=False):
    scan_folder = Path(cwd, folder).resolve()
    
    if kaggle:
        print("Utilizing Kaggle dataset. Loading up scans")
        
        scans ={
            NON_DEMENTED: [],
            VERY_MILD_DEMENTED: [],
            MILD_DEMENTED: [],
            MODERATE_DEMENTED: [],
        }
        
        for scan in scan_folder.glob("{}*".format(NON_DEMENTED)):
            print(scan.__dict__)
        
    else:
        print("Not a kaggle dataset")