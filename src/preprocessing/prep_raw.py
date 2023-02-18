import shutil
import numpy as np
import fsl.wrappers.fsl_anat as fsl_anat
import fsl.wrappers.fslmaths as fsl_maths
import boto3
from utils import create_slices_from_brain, create_multichannel_slices_from_brain
from termcolor import colored, cprint
from ..settings import *
from pathlib import Path

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()


def prep_raw_mri(scan_loc, scan_name, out_dir, group, run_name, slice_range=(35, 55)):
    """Prepares a singular raw mri image. This means that FSL is ran, indiviudal 
    slices are extracted, and multichannel slices are extracted. 2 datasets are 
    created, one for multichannel images, and one for simple slices.

    Args:
        scan_loc (str): The scan location
        scan_name (str): The name of the scan (format NNN_S_NNNN_NN)
        out_dir (str): output directory, where images should be placed.
        group (str): Class of the image. Can be CN, AD, or MCI
        slice_range (tuple, optional): The slices to be extracted. Defaults to (35,55).

    Raises:
        ValueError: Raised if any values are missing for the scan.
    """
    if group is None:
        raise ValueError(
            colored("ERROR: Scan instatiation requires group (class)!","red"))

    scan_location = Path(cwd, scan_loc).resolve()

    if not SKIP_FSL:
        print(f"Launching FSL scripts for scan {scan_name} in group {group}")
        
        try:
            original_brain, nii_path = run_fsl(scan_location, scan_name, group, out_dir)
        except:
            # checking which brain failed to run
            original_brain = scan_loc
            for p in scan_loc.rglob("*"):
                if p.name.endswith(".nii"):
                    original_brain = p
                    break
            # if fsl fails, we return dummy values
            # Will be logged in batches.log. just skip this scan
            return (original_brain, "FSL failed")
        
        print(f"FSL scripts complete. Processed MRI found in {nii_path}")
    else:
        nii_path = scan_location
        
    if USE_S3 and not SKIP_FSL:
        # Upload processed MRI to s3 bucket
        s3_loc = Path(f"{run_name}/{group}/{scan_name}_processed.nii.gz")
        cprint(print(f"INFO: uploading processed MRI to s3 bucket {AWS_S3_BUCKET_NAME} in {s3_loc}"), "blue")
        s3_bucket  = boto3.resource('s3').Bucket('processed-nii-files')

        try:
            s3_bucket.upload_file(str(nii_path), str(s3_loc))
        except:
            cprint("ERROR: Failed to upload file to s3 bucket", "red")
            cprint("INFO: Will attempt to sync bucket at end of run", "blue") 

    # To access slices:
    # sagittal = data[26, :, :] <- 26th slice along sagittal
    # coronal = data[:, 30, :] <- 30th slice along coronal
    # axial = data[:, :, 50] <- 50th slice along axial

    if not SKIP_SLICE_CREATION:
        # Split into individual slices
        create_slices_from_brain(nii_path, out_dir, scan_name, group, slice_range)

        # Split into multichannel slices
        create_multichannel_slices_from_brain(
            nii_path, out_dir, scan_name, group, slice_range)
    
    if not SKIP_FSL:
        return (original_brain, nii_path)


def run_fsl(scan_location, scan_name, group, out_dir):
    """Runs fsl_anat and performs brain extraction for the given scan.

    Args:
        scan_location (str): location of the scan
        scan_name (str): The name of the scan (format NNN_S_NNNN_NN)
        group (str): Class of the image. Can be CN, AD, or MCI
        out_dir (str): output directory. Nii files saved in {out_dir}/nii_files/{group}

    Returns:
        str: path of the saved nii image output by FSL
    """
    original_brain = ""
    # finding nii file. Each of a subject's scans is in a different scan_location, so this works w multiple scans for a subject
    for p in scan_location.rglob("*"):
        if p.name.endswith(".nii"):
            original_brain = p
            break

    # The tmp_dir directory will be used to store all the fsl_anat info
    tmp_dir = Path(
        filedir, "../../out/preprocessed_datasets/tmp", scan_name).resolve()

    # fsl_anat adds .anat to end of output directory
    anat_dir = Path(f"{tmp_dir}/anat")

    try:
        # Running fsl_anat (we don't need tissue segmentation nor subcortical segmentation)
        fsl_anat(img=original_brain, out=tmp_dir, noseg=True, nosubcortseg=True)
    except:
        # If fsl fails, we delete the tmp_dir and return dummy values
        shutil.rmtree(anat_dir)
        raise ValueError(f"ERROR: FSL failed to run on scan {scan_name} in group {group}. \n Original brain: {original_brain} \n tmp_dir: {tmp_dir}")

    # This is the outputted nonlinear transformed brain
    mni_nonlin = Path(anat_dir, "T1_to_MNI_nonlin.nii.gz")

    # This is the outputted brain mask
    brain_mask = Path(anat_dir, "MNI152_T1_2mm_brain_mask_dil1.nii.gz")

    # File is saved into group subfolder in nii_files output loc
    final_brain = Path(
        out_dir, f"nii_files/{group}/{scan_name}_processed.nii.gz")

    # We multiply the MNI registered brain by the brain mask to have a final preprocessed brain
    fsl_maths(mni_nonlin).mul(brain_mask).run(final_brain)

    # clearing all the .anat files (unnecessary now)
    shutil.rmtree(anat_dir)

    return (original_brain, final_brain)



