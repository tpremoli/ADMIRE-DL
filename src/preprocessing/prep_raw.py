import shutil
import numpy as np
import nibabel as nib
import fsl.wrappers.fsl_anat as fsl_anat
import fsl.wrappers.fslmaths as fsl_maths
from pathlib import Path
from datetime import datetime

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()


def prep_raw_mri(scan_loc, scan_name, out_dir, group, sex):
    if group is None or sex is None:
        raise ValueError(
            "ERROR: Scan instatiation requires group, sex, and age values!")

    scan_location = Path(cwd, scan_loc).resolve()

    print("Launching FSL scripts for scan {}".format(scan_name))
    nii_path = run_fsl(scan_location, scan_name, out_dir)

    print("FSL scripts complete. Processed MRI found in {}".format(nii_path))

    # To access slices:
    # sagittal = self.data[26, :, :] <- 26th slice along sagittal
    # coronal = self.data[:, 30, :] <- 30th slice along coronal
    # axial = self.data[:, :, 50] <- 50th slice along axial

    # TODO: make sure saved into groups
    print("splitting MRI into individual slice images, slices X-X.")

    # TODO: make sure saved into groups
    print("splitting MRI into multichannal slice images, slices X-X.")
    
    # TODO: make sure saved into groups
    print("splitting images into train/test/val folders, with ratio X/X/X.")


def run_fsl(scan_location, scan_name, out_dir):
    niifile = ""
    # finding nii file. Each of a subject's scans is in a different scan_location, so this works w multiple scans for a subject
    for p in scan_location.rglob("*"):
        if p.name.endswith(".nii"):
            niifile = p
            break

    # The tmp_dir directory will be used to store all the fsl_anat info
    tmp_dir = Path(
        filedir, "../../out/preprocessed_datasets/tmp", scan_name).resolve()

    # Running fsl_anat (we don't need tissue segmentation nor subcortical segmentation)
    fsl_anat(img=niifile, out=tmp_dir, noseg=True, nosubcortseg=True)

    # fsl_anat adds .anat to end of output directory
    anat_dir = Path("{}.anat".format(tmp_dir))

    # This is the outputted nonlinear transformed brain
    mni_nonlin = Path(anat_dir, "T1_to_MNI_nonlin.nii.gz")

    # This is the outputted brain mask
    brain_mask = Path(anat_dir, "MNI152_T1_2mm_brain_mask_dil1.nii.gz")

    # TODO: split into groups
    final_brain = Path(
        out_dir, "nii_files/{}_processed.nii.gz".format(scan_name))

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

