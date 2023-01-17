import pandas as pd
import json
import shutil
from .constants import NON_DEMENTED, VERY_MILD_DEMENTED, MILD_DEMENTED, MODERATE_DEMENTED
from ..classes.scan import Scan
from pathlib import Path

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def prep_adni(collection_dir, collection_csv, run_name):
    # Resetting path locs
    collection_dir = Path(cwd, collection_dir).resolve()
    collection_csv = Path(cwd, collection_csv).resolve()

    if not Path.exists(collection_dir):
        raise ValueError("Collection dir {} does not exist!".format(collection_dir))
    elif not any(Path(collection_dir).iterdir()):
        raise ValueError("Collection path {} empty!".format(collection_dir))
    
    subjects = pd.read_csv(collection_csv).drop_duplicates(keep='first', subset="Subject").to_dict(orient="records")

    out_dir = Path(
        filedir, "../out/preprocessed_datasets", run_name).resolve()
    
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_count = 0
    # Do this multiprocessed
    for subject in subjects:
        subj_folder = Path(collection_dir, subject["Subject"], "MP-RAGE")
        # For each scan in the subject subject
        for count, scan_folder in enumerate(Path.glob(subj_folder, "*")):
            # This makes the name styled 002_S_0295_{no} where no is the number of sampel we're on
            scan_name = "{}_{}".format(subject["Subject"], count)

            Scan(scan_folder, run_name, scan_name, str(out_dir), kaggle=False, group=subject["Group"], sex=subject["Sex"])
            scan_count += 1

    # Writing meta file
    with open(Path(out_dir, "meta"), "w") as meta_file:
        metadata ={
            "kaggle":False,
            "run_name":run_name,
            "original_dir": str(out_dir),
            "scan_count":scan_count
        }
        json.dump(metadata, meta_file,indent=4)
        
    # Writing collection.csv file
    shutil.copyfile(collection_csv, Path(out_dir, "collection.csv"))

def prep_kaggle(kaggle_dir, run_name):
    kaggle_dir = Path(cwd, kaggle_dir).resolve()

    if not Path.exists(kaggle_dir):
        raise ValueError("Kaggle path {} does not exist!".format(kaggle_dir))
    elif not any(Path(kaggle_dir).iterdir()):
        raise ValueError("Kaggle path {} empty!".format(kaggle_dir))
    
    out_dir = Path(
        filedir, "../out/preprocessed_datasets", run_name).resolve()
    
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Launching non-demented prep")
    nondemented_count = 0
    for i, image in enumerate(Path(kaggle_dir, NON_DEMENTED).resolve().iterdir()):
        filename = "{}_{}".format(NON_DEMENTED, i)
        Scan(image, run_name, filename, str(out_dir), kaggle=True)
        nondemented_count+=1

    print("Launching very mild-demented prep")
    verymilddemented_count = 0
    for i, image in enumerate(Path(kaggle_dir, VERY_MILD_DEMENTED).resolve().iterdir()):
        filename = "{}_{}".format(VERY_MILD_DEMENTED, i)
        Scan(image, run_name, filename, str(out_dir), kaggle=True)
        verymilddemented_count+=1

    print("Launching mild-demented prep")
    milddemented_count = 0
    for i, image in enumerate(Path(kaggle_dir, MILD_DEMENTED).resolve().iterdir()):
        filename = "{}_{}".format(MILD_DEMENTED, i)
        Scan(image, run_name, filename, str(out_dir), kaggle=True)
        milddemented_count+=1

    print("Launching moderate-demented prep")
    moderatedemented_count = 0
    for i, image in enumerate(Path(kaggle_dir, MODERATE_DEMENTED).resolve().iterdir()):
        filename = "{}_{}".format(MODERATE_DEMENTED, i)
        Scan(image, run_name, filename, str(out_dir), kaggle=True)
        moderatedemented_count+=1
        
    with open(Path(out_dir, "meta"), "w") as meta_file:
        metadata ={
            "kaggle":True,
            "run_name":run_name,
            "original_dir": str(kaggle_dir),
            "nondemented":nondemented_count,
            "verymilddemented":verymilddemented_count,
            "milddemented":milddemented_count,
            "moderatedemented":moderatedemented_count
        }
        json.dump(metadata, meta_file,indent=4)
        
    print("finished prepping kaggle dataset")
