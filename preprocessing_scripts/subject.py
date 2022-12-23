import pathlib
import numpy as np
import pandas as pd
import fsl.wrappers.fsl_anat as fsl_anat
# https://open.win.ox.ac.uk/pages/fsl/fslpy/fsl.wrappers.fsl_anat.html?highlight=fsl_anat#module-fsl.wrappers.fsl_anat fsl anat
import fsl.wrappers.fslmaths as fsl_maths
# https://open.win.ox.ac.uk/pages/fsl/fslpy/fsl.wrappers.fslmaths.html?highlight=maths#module-fsl.wrappers.fslmaths fsl maths

cwd = pathlib.Path().resolve()
filedir = pathlib.Path(__file__).parent.resolve()


class Subject:
    def __init__(self, subj_folder: str, run_name: str, collection_df=None, group=None, sex=None):
        if collection_df is None and (group is None or sex is None):
            raise ValueError(
                "ERROR: Subject instatiation requires either a dataframe of collection info or group, sex, and age values!")

        subj_location = pathlib.Path(cwd, subj_folder).resolve()

        # The subject name is the parent folder (i.e 005_S_0221)
        self.subj_name = subj_location.name
        self.run_name = run_name

        # Should log this in out
        print("Loading subject {} for preprocessing".format(self.subj_name))

        # Setting sample sex group and age
        if collection_df is None:
            self.group = group
            self.sex = sex
        else:
            # TODO: How should we handle different scans of same subject?
            # idea: treat each scan as a different subject. identify subj by id + date.
            # Will work on in future. Keeping things simple for now.
            entry = collection_df[collection_df["Subject"]
                                  == self.subj_name].iloc[0]
            self.group = entry["Group"]
            self.sex = entry["Sex"]

        print("Launching fsl_anat for subject {}".format(self.subj_name))
        self.data = self.preprocess_subject(subj_location)

    def preprocess_subject(self, subj_location):
        niifile = ""
        # Finding nii file. This just does it for the first file found in subfolders, but this should run for multiple samples in the future
        for p in subj_location.rglob("*"):
            if p.name.endswith(".nii"):
                niifile = p
                break

        # defining ouput dir (out/preprocessed_samples/run_name)
        out_dir = pathlib.Path(
            filedir, "../out/preprocessed_samples", self.run_name).resolve()

        fsl_anat(img=niifile, out=out_dir)


subjects_csv = pd.read_csv("unprocessed_samples/test_sample.csv")
s = Subject("unprocessed_samples/ADNI/002_S_0295", "test_run", subjects_csv)

print(s.__dict__)
