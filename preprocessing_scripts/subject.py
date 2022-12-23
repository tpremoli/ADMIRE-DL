import pathlib
import numpy as np
import pandas as pd

cwd = pathlib.Path().resolve()
filedir = pathlib.Path(__file__).parent.resolve()


class Subject:
    def __init__(self, subj_folder: str,  collection_df=None, group=None, sex=None):
        if collection_df is None and (group is None or sex is None):
            raise ValueError("ERROR: Subject instatiation requires either a dataframe of collection info or group, sex, and age values!")
        
        subj_location = pathlib.Path(cwd, subj_folder).resolve()

        # The subject name is the parent folder (i.e 005_S_0221)
        self.name = subj_location.name

        # Setting sample sex group and age
        if collection_df is None:
            self.group = group
            self.sex = sex
        else:
            # TODO: How should we handle different scans of same subject?
            # idea: treat each scan as a different subject. identify subj by id + date.
            # Will work on in future. Keeping things simple for now.
            entry = collection_df[collection_df["Subject"] == self.name].iloc[0]

            self.group = entry["Group"]
            self.sex = entry["Sex"]


subjects_csv = pd.read_csv("unprocessed_samples/test_sample.csv")
s = Subject("unprocessed_samples/ADNI/002_S_0295", subjects_csv)

print(s.__dict__)

