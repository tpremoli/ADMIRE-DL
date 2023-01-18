from pathlib import Path
from .scan import Scan
from .constants import *

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def load_scans_from_folder(folder, kaggle=False):
    scan_folder = Path(cwd, folder).resolve()
    
    if kaggle:
        print("Utilizing Kaggle dataset. Loading up scans")
        
        loaded_scans ={
            NON_DEMENTED: [],
            VERY_MILD_DEMENTED: [],
            MILD_DEMENTED: [],
            MODERATE_DEMENTED: [],
        }
        
        # For each category
        for scan_class in loaded_scans.keys():
            print("Loading {} scans".format(scan_class))
            # Get each file for each class
            for scan in sorted(scan_folder.glob("{}*".format(scan_class))):
                # load the pickle and add to dict
                loaded_scans[scan_class].append(Scan.from_pickle(scan))
                
        return loaded_scans
    else:
        print("Not a kaggle dataset")
