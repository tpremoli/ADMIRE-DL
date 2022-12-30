import pandas as pd
from .constants import NON_DEMENTED, VERY_MILD_DEMENTED, MILD_DEMENTED, MODERATE_DEMENTED
from .scan import Scan
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

    # Do this multiprocessed
    for subject in subjects:
        subj_folder = Path(collection_dir, subject["Subject"], "MP-RAGE")
        # For each scan in the subject subject
        for count, scan_folder in enumerate(Path.glob(subj_folder, "*")):
            # This makes the name styled 002_S_0295_{no} where no is the number of sampel we're on
            scan_name = "{}_{}".format(subject["Subject"], count)

            s = Scan(scan_folder=scan_folder, run_name=run_name, scan_name=scan_name, group=subject["Group"], sex=subject["Sex"])


def prep_kaggle(kaggle_dir, run_name):
    kaggle_dir = Path(cwd, kaggle_dir).resolve()

    if not Path.exists(kaggle_dir):
        raise ValueError("Kaggle path {} does not exist!".format(kaggle_dir))
    elif not any(Path(kaggle_dir).iterdir()):
        raise ValueError("Kaggle path {} empty!".format(kaggle_dir))

    print("Launching non-demented prep")
    for i, image in enumerate(Path(kaggle_dir, NON_DEMENTED).resolve().glob("*")):
        filename = "{}_{}".format(NON_DEMENTED, i)
        s = Scan(img_dir=image, run_name=run_name, category=NON_DEMENTED, name_overwrite=filename)

    print("Launching very mild-demented prep")
    for i, image in enumerate(Path(kaggle_dir, VERY_MILD_DEMENTED).resolve().glob("*")):
        filename = "{}_{}".format(VERY_MILD_DEMENTED, i)
        s = Scan(img_dir=image, run_name=run_name, category=VERY_MILD_DEMENTED, name_overwrite=filename)

    print("Launching mild-demented prep")
    for i, image in enumerate(Path(kaggle_dir, MILD_DEMENTED).resolve().glob("*")):
        filename = "{}_{}".format(MILD_DEMENTED, i)
        s = Scan(img_dir=image, run_name=run_name, category=MILD_DEMENTED, name_overwrite=filename)

    print("Launching moderate-demented prep")
    for i, image in enumerate(Path(kaggle_dir, MODERATE_DEMENTED).resolve().glob("*")):
        filename = "{}_{}".format(MODERATE_DEMENTED, i)
        s = Scan(img_dir=image, run_name=run_name, category=MODERATE_DEMENTED, name_overwrite=filename)
    
    print("finished prepping kaggle dataset")
