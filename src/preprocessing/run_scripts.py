import pandas as pd
import json
import shutil
import splitfolders
from datetime import datetime
from ..classes.constants import NON_DEMENTED, VERY_MILD_DEMENTED, MILD_DEMENTED, MODERATE_DEMENTED
from ..classes.scan import Scan
from pathlib import Path

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()


def prep_adni(collection_dir, collection_csv, run_name):
    # Resetting path locs
    collection_dir = Path(cwd, collection_dir).resolve()
    collection_csv = Path(cwd, collection_csv).resolve()

    if not Path.exists(collection_dir):
        raise ValueError(
            "Collection dir {} does not exist!".format(collection_dir))
    elif not any(Path(collection_dir).iterdir()):
        raise ValueError("Collection path {} empty!".format(collection_dir))

    subjects = pd.read_csv(collection_csv).drop_duplicates(
        keep='first', subset="Subject").to_dict(orient="records")

    out_dir = Path(
        filedir, "../../out/preprocessed_datasets", run_name).resolve()

    print("output dir: {}".format(out_dir))

    out_dir.mkdir(parents=True, exist_ok=True)

    scan_count = 0
    # Do this multiprocessed
    for subject in subjects:
        subj_folder = Path(collection_dir, subject["Subject"], "MP-RAGE")
        # For each scan in the subject subject
        for count, scan_folder in enumerate(Path.glob(subj_folder, "*")):
            # This makes the name styled 002_S_0295_{no} where no is the number of sampel we're on. min 6 chars
            scan_name = "{}_{:06d}".format(subject["Subject"], count)

            Scan(scan_loc=scan_folder,
                 run_name=run_name,
                 scan_no=count,
                 scan_name=scan_name,
                 out_dir=str(out_dir),
                 group=subject["Group"],
                 sex=subject["Sex"])
            scan_count += 1

    # Writing meta file
    with open(Path(out_dir, "meta.json"), "w") as meta_file:
        metadata = {
            "kaggle": False,
            "run_name": run_name,
            "original_dir": str(out_dir),
            "scan_count": scan_count
        }
        json.dump(metadata, meta_file, indent=4)

    # Writing collection.csv file
    shutil.copyfile(collection_csv, Path(out_dir, "collection.csv"))


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

    print("finished prepping kaggle dataset")
