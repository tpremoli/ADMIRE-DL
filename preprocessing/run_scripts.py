import pandas as pd
import constants
from scan import Scan
from pathlib import Path

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def prep_adni(collection_dir, collection_csv, run_name):
    # Resetting path locs
    collection_dir = Path(cwd, collection_dir).resolve()
    collection_csv = Path(cwd, collection_csv).resolve()

    subjects = pd.read_csv(collection_csv).drop_duplicates(keep='first', subset="Subject").to_dict(orient="records")

    # Do this multiprocessed
    for subject in subjects:
        subj_folder = Path(collection_dir, subject["Subject"], "MP-RAGE")
        # For each scan in the subject subject
        for count, scan_folder in enumerate(Path.glob(subj_folder, "*")):
            # This makes the name styled 002_S_0295_{no} where no is the number of sampel we're on
            subj_name = "{}_{}".format(subject["Subject"], count)

            s = Scan(scan_folder, run_name, scan_name=subj_name, group=subject["Group"], sex=subject["Sex"])


def prep_kaggle(kaggle_dir, run_name):
    kaggle_dir = Path(cwd, kaggle_dir).resolve()

    for i, image in enumerate(Path(kaggle_dir, constants.NON_DEMENTED).resolve().glob("*")):
        filename = "{}_{}".format(i, constants.NON_DEMENTED)
        s = Scan(image, run_name, category=constants.NON_DEMENTED, name_overwrite=filename)

    for i, image in enumerate(Path(kaggle_dir, constants.VERY_MILD_DEMENTED).resolve().glob("*")):
        filename = "{}_{}".format(i, constants.VERY_MILD_DEMENTED)
        s = Scan(image, run_name, category=constants.VERY_MILD_DEMENTED, name_overwrite=filename)

    for i, image in enumerate(Path(kaggle_dir, constants.MILD_DEMENTED).resolve().glob("*")):
        filename = "{}_{}".format(i, constants.MILD_DEMENTED)
        s = Scan(image, run_name, category=constants.MILD_DEMENTED, name_overwrite=filename)

    for i, image in enumerate(Path(kaggle_dir, constants.MODERATE_DEMENTED).resolve().glob("*")):
        filename = "{}_{}".format(i, constants.MODERATE_DEMENTED)
        s = Scan(image, run_name, category=constants.MODERATE_DEMENTED, name_overwrite=filename)



prep_kaggle("unprocessed_samples/kaggle", "kaggle run")
