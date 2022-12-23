import shutil
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import fsl.wrappers.fsl_anat as fsl_anat
import fsl.wrappers.fslmaths as fsl_maths
from pathlib import Path

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

class Subject:
    def __init__(self, subj_folder: str, run_name: str, collection_df=None, group=None, sex=None):
        if collection_df is None and (group is None or sex is None):
            raise ValueError(
                "ERROR: Subject instatiation requires either a dataframe of collection info or group, sex, and age values!")

        subj_location = Path(cwd, subj_folder).resolve()

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

        print("Launching FSL scripts for subject {}".format(self.subj_name))
        self.nii_path, self.out_dir = self.run_fsl(subj_location)

        print("FSL scripts complete. Retrieving data from output")
        self.data = self.get_data_from_nii()

        # To access slices:
        # sagittal = self.data[26, :, :] <- 26th slice along sagittal
        # coronal = self.data[:, 30, :] <- 30th slice along coronal
        # axial = self.data[:, :, 50] <- 50th slice along axial

        # Saving our object as a pickle
        with open(Path(self.out_dir, "{}_processed.pkl".format(self.subj_name)), "wb") as pkl_file:
            pickle.dump(self, pkl_file, pickle.HIGHEST_PROTOCOL)

    def from_pickle(existing_pkl):
        with open(existing_pkl, "rb") as f:
            return pickle.load(f)

    def run_fsl(self, subj_location):
        niifile = ""
        # Finding nii file. This just does it for the first file found in subfolders, but this should run for multiple samples in the future
        for p in subj_location.rglob("*"):
            if p.name.endswith(".nii"):
                niifile = p
                break

        # defining ouput dir (out/preprocessed_samples/run_name)
        out_dir = Path(
            filedir, "../out/preprocessed_samples", self.run_name).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # The tmp_dir directory will be used to store all the fsl_anat info
        tmp_dir = Path(
            filedir, "../out/preprocessed_samples/tmp", self.run_name).resolve()

        # Running fsl_anat
        fsl_anat(img=niifile, out=tmp_dir)

        # fsl_anat adds .anat to end of output directory
        anat_dir = Path("{}.anat".format(tmp_dir))

        # This is the outputted nonlinear transformed brain
        mni_nonlin = Path(anat_dir, "T1_to_MNI_nonlin.nii.gz")

        # This is the outputted brain mask
        brain_mask = Path(anat_dir, "MNI152_T1_2mm_brain_mask_dil1.nii.gz")

        final_brain = Path(
            out_dir, "{}_processed.nii.gz".format(self.subj_name))

        # We multiply the MNI registered brain by the brain mask to have a final preprocessed brain
        fsl_maths(mni_nonlin).mul(brain_mask).run(final_brain)

        # Copy log from anat dir to out dir
        logfile = Path(anat_dir, "log.txt")
        final_logfile = Path(out_dir, "{}.log".format(self.subj_name))
        shutil.copyfile(logfile, final_logfile)

        # clearing all the .anat files (unnecessary now)
        shutil.rmtree(anat_dir)

        return final_brain, out_dir

    def get_data_from_nii(self):
        imgfile = nib.load(self.nii_path)
        return np.array(imgfile.dataobj)

    def show_img(self):
        nib.viewers.OrthoSlicer3D(self.data).show()

