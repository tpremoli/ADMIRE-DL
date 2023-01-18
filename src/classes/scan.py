import shutil
import pickle
import numpy as np
import nibabel as nib
import fsl.wrappers.fsl_anat as fsl_anat
import fsl.wrappers.fslmaths as fsl_maths
from PIL import Image
from pathlib import Path
from datetime import datetime

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

class Scan:
    def __init__(self, scan_loc:str, run_name:str, scan_no:int, scan_name:str, out_dir:str, kaggle:bool, group=None, sex=None):
        self.kaggle = kaggle
        self.scan_name = None
        self.scan_no = None # useful for sorting
        self.run_name = None # filename/run name
        self.group = None
        self.sex = None
        self.out_dir = None
        self.nii_path = None
        self.data = None

        if kaggle == True:
            img_location = Path(cwd, scan_loc).resolve()

            self.scan_name = scan_name
            self.scan_no = scan_no
            self.run_name = run_name
            self.group = group

            # should convert to b/w
            image = np.asarray(Image.open(img_location))

            self.data = image
            self.out_dir = out_dir

            # Saving our object as a pickle
            with open(Path(self.out_dir, "{}_processed.pkl".format(self.scan_name)), "wb") as pkl_file:
                pickle.dump(self, pkl_file, pickle.HIGHEST_PROTOCOL)
                
        elif kaggle == False:
            if group is None or sex is None:
                raise ValueError(
                    "ERROR: Scan instatiation requires group, sex, and age values!")

            start_time = datetime.now()

            scan_location = Path(cwd, scan_loc).resolve()

            # The scan name is the parent folder (i.e 005_S_0221_0)
            self.scan_no = scan_no
            self.scan_name = scan_name
            self.run_name = run_name

            # Should log this in out
            print("Loading scan {} for preprocessing".format(self.scan_name))

            # Setting sample sex group and age
            self.group = group
            self.sex = sex

            # defining ouput dir (out/preprocessed_datasets/run_name)
            self.out_dir = out_dir

            print("Launching FSL scripts for scan {}".format(self.scan_name))
            self.nii_path = self.run_fsl(scan_location)

            print("FSL scripts complete. Retrieving data from output")
            self.data = self.get_data_from_nii()

            # To access slices:
            # sagittal = self.data[26, :, :] <- 26th slice along sagittal
            # coronal = self.data[:, 30, :] <- 30th slice along coronal
            # axial = self.data[:, :, 50] <- 50th slice along axial

            # Saving our object as a pickle
            with open(Path(self.out_dir, "{}_processed.pkl".format(self.scan_name)), "wb") as pkl_file:
                pickle.dump(self, pkl_file, pickle.HIGHEST_PROTOCOL)
            
            end_time = datetime.now()
            with open(Path(self.out_dir, "{}.txt".format(self.scan_name)), "w") as f:
                f.write("start time: ")
                f.write(str(start_time))
                f.write("\n")
                f.write("end time: ")
                f.write(str(end_time))
                f.write("\ntotal time elapsed:")
                f.write(str(end_time-start_time))
    
    def __eq__(self, other):
        return self.scan_no == other.scan_no

    def __lt__(self, other):
        return self.scan_no < other.scan_no

    def from_pickle(existing_pkl):
        pkl_loc = Path(cwd, existing_pkl).resolve()
        with open(pkl_loc, "rb") as f:
            scan = pickle.load(f)
            return scan

    def run_fsl(self, scan_location):
        niifile = ""
        # Finding nii file. This just does it for the first file found in subfolders, but this should run for multiple samples in the future
        for p in scan_location.rglob("*"):
            if p.name.endswith(".nii"):
                niifile = p
                break

        # The tmp_dir directory will be used to store all the fsl_anat info
        tmp_dir = Path(
            filedir, "../out/preprocessed_datasets/tmp", self.scan_name).resolve()

        # Running fsl_anat (we don't need tissue segmentation nor subcortical segmentation)
        fsl_anat(img=niifile, out=tmp_dir, noseg=True,nosubcortseg=True)

        # fsl_anat adds .anat to end of output directory
        anat_dir = Path("{}.anat".format(tmp_dir))

        # This is the outputted nonlinear transformed brain
        mni_nonlin = Path(anat_dir, "T1_to_MNI_nonlin.nii.gz")

        # This is the outputted brain mask
        brain_mask = Path(anat_dir, "MNI152_T1_2mm_brain_mask_dil1.nii.gz")

        final_brain = Path(
            self.out_dir, "{}_processed.nii.gz".format(self.scan_name))

        # We multiply the MNI registered brain by the brain mask to have a final preprocessed brain
        fsl_maths(mni_nonlin).mul(brain_mask).run(final_brain)

        # Copy log from anat dir to out dir
        # logfile = Path(anat_dir, "log.txt")
        # final_logfile = Path(out_dir, "{}.log".format(self.scan_name))
        # shutil.copyfile(logfile, final_logfile)

        # clearing all the .anat files (unnecessary now)
        shutil.rmtree(anat_dir)

        return final_brain

    def get_data_from_nii(self):
        imgfile = nib.load(self.nii_path)
        return np.array(imgfile.dataobj)

    def show_img(self):
        nib.viewers.OrthoSlicer3D(self.data).show()

