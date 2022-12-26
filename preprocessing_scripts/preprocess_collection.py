import pandas as pd
from .subject import Subject
from pathlib import Path

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def preprocess_collection(collection_dir, collection_csv, run_name):
    # Resetting path locs
    collection_dir = Path(cwd, collection_dir).resolve()
    collection_csv = Path(cwd, collection_csv).resolve()

    subjects = pd.read_csv(collection_csv)

    subjects = []

    # Do this multiprocessed
    for subject in subjects:
        subj_folder = Path(collection_dir, subject["Subject"])
        s = Subject(subj_folder, run_name, group=subject["Group"], sex=subject["Sex"])
        subjects.append(s.name)


preprocess_collection("unprocessed_samples/ADNI", "unprocessed_samples/test_sample.csv", "test_sample_1")
