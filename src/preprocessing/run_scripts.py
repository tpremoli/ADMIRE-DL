import pandas as pd
import json
import shutil
import splitfolders
from datetime import datetime
from ..classes.constants import NON_DEMENTED, VERY_MILD_DEMENTED, MILD_DEMENTED, MODERATE_DEMENTED
from .prep_raw import prep_raw_mri
from pathlib import Path

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()


def prep_adni(collection_dir, collection_csv, run_name, split_ratio):
    # Resetting path locs
    collection_dir = Path(cwd, collection_dir).resolve()
    collection_csv = Path(cwd, collection_csv).resolve()

    if not Path.exists(collection_dir):
        raise ValueError(
            "Collection dir {} does not exist!".format(collection_dir))
    elif not any(Path(collection_dir).iterdir()):
        raise ValueError("Collection path {} empty!".format(collection_dir))

    # Getting individual subjects
    subjects = pd.read_csv(collection_csv).drop_duplicates(
        keep='first', subset="Subject").to_dict(orient="records")

    out_dir = Path(
        filedir, "../../out/preprocessed_datasets", run_name).resolve()

    print("output dir: {}".format(out_dir))

    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except:
        raise ValueError(
            "Output dir {} already exists! Pick a different run name or delete the existing directory.".format(out_dir))
    
    # Creating group subdirs for output nii images
    Path(out_dir, "nii_files/CN").resolve().mkdir(parents=True, exist_ok=True)
    Path(out_dir, "nii_files/MCI").resolve().mkdir(parents=True, exist_ok=True)
    Path(out_dir, "nii_files/AD").resolve().mkdir(parents=True, exist_ok=True)
    
    # Creating group subdirs for output image slices
    Path(out_dir, "image_slices/CN").resolve().mkdir(parents=True, exist_ok=True)
    Path(out_dir, "image_slices/MCI").resolve().mkdir(parents=True, exist_ok=True)
    Path(out_dir, "image_slices/AD").resolve().mkdir(parents=True, exist_ok=True)

    # Creating group subdirs for output multi-channel image slices
    Path(out_dir, "multi_channel/CN").resolve().mkdir(parents=True, exist_ok=True)
    Path(out_dir, "multi_channel/MCI").resolve().mkdir(parents=True, exist_ok=True)
    Path(out_dir, "multi_channel/AD").resolve().mkdir(parents=True, exist_ok=True)


    slice_count = 0 # This will be summed with slice_range[1] - slice_range[0]
    # Do this multiprocessed
    for subject in subjects:
        subj_folder = Path(collection_dir, subject["Subject"], "MP-RAGE")
        # For each scan in the subject subject
        for count, scan_folder in enumerate(Path.glob(subj_folder, "*")):
            start_time = datetime.now()
            
            # This makes the name styled 002_S_0295_{no} where no is the number of sampel we're on. min 6 chars
            scan_name = "{}_{:02d}".format(subject["Subject"], count)
            prep_raw_mri(scan_folder,scan_name,out_dir,subject["Group"],subject["Sex"])
        
            end_time = datetime.now()

            # TODO: Write to LOG. end_time - start_time

    print("Done processing raw MRIs. Saving mata data")

    scan_count = len(list(Path(out_dir, "nii_files").glob('**/*')))
    slice_count = len(list(Path(out_dir, "image_slices").glob('**/*')))

    # Writing meta file
    with open(Path(out_dir, "meta.json"), "w") as meta_file:
        metadata = {
            "kaggle": False,
            "run_name": run_name,
            "original_dir": str(collection_dir),
            "split": list(split_ratio),
            "scan_count": scan_count,
            "slice_count":slice_count,
            # "dataset_split_seed": split_seed, TODO: add split and split_seed option
            # TODO: maybe add CLI option to skip_fsl stuff and just sort into slices + folders???
        }
        json.dump(metadata, meta_file, indent=4)

    # Writing collection.csv file
    shutil.copyfile(collection_csv, Path(out_dir, "collection.csv"))
    
    print("Done! Result files found in {}".format(out_dir))


def prep_kaggle(kaggle_dir, run_name, split_ratio):
    kaggle_dir = Path(cwd, kaggle_dir).resolve()

    if not Path.exists(kaggle_dir):
        raise ValueError("Kaggle path {} does not exist!".format(kaggle_dir))
    elif not any(Path(kaggle_dir).iterdir()):
        raise ValueError("Kaggle path {} empty!".format(kaggle_dir))

    out_dir = Path(
        filedir, "../../out/preprocessed_datasets", run_name).resolve()

    print("output dir: {}".format(out_dir))

    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except:
        raise ValueError(
            "Output dir {} already exists! Pick a different run name or delete the existing directory.".format(out_dir))

    split_seed = datetime.now().timestamp()
    
    print("Splitting folders with split ratio {}".format(split_ratio))
    splitfolders.ratio(kaggle_dir, output=out_dir,
                       seed=split_seed, ratio=split_ratio)

    train_count = len(list(Path(out_dir, "train").glob('**/*')))
    test_count = len(list(Path(out_dir, "test").glob('**/*')))
    val_count = len(list(Path(out_dir, "val").glob('**/*')))

    with open(Path(out_dir, "meta.json"), "w") as meta_file:
        metadata = {
            "kaggle": True,
            "run_name": run_name,
            "original_dir": str(kaggle_dir),
            "split": list(split_ratio),
            "train_count": train_count,
            "test_count": test_count,
            "val_count": val_count,
            "dataset_split_seed": split_seed,
        }
        json.dump(metadata, meta_file, indent=4)

    print("Done! Result files found in {}".format(out_dir))
