import pandas as pd
from subject import Subject
from pathlib import Path

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def preprocess_collection(collection_dir, collection_csv, run_name):
    # Resetting path locs
    collection_dir = Path(cwd, collection_dir).resolve()
    collection_csv = Path(cwd, collection_csv).resolve()

    subjects = pd.read_csv(collection_csv).drop_duplicates(keep='first', subset="Subject").to_dict(orient="records")

    print(subjects)

    # Do this multiprocessed
    for subject in subjects:
        subj_folder = Path(collection_dir, subject["Subject"], "MP-RAGE")
        # For each scan in the subject subject
        for count, scan_folder in enumerate(Path.glob(subj_folder, "*")):
            # This makes the name styled 002_S_0295_{no} where no is the number of sampel we're on
            subj_name = "{}_{}".format(subject["Subject"], count)

            s = Subject(scan_folder, run_name, subj_name=subj_name, group=subject["Group"], sex=subject["Sex"])


preprocess_collection("unprocessed_samples/ADNI", "unprocessed_samples/test_sample.csv", "test_sample_1")
