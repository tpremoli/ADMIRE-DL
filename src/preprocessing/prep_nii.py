import shutil
import numpy as np
import boto3
from .utils import create_image_slices_from_brain
from .fsl_scripts import run_fsl
from ..settings import *
from termcolor import colored, cprint
from pathlib import Path

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()


def prep_raw_mri(scan_loc, scan_name, out_dir, group, run_name, slice_range=(80, 110)):
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
            # if fsl fails, we return dummy values
            # Will be logged in batches.log. just skip this scan
            return (scan_location, "FSL failed")
        
        print(f"FSL scripts complete. Processed MRI found in {nii_path}")
    else:
        nii_path = scan_location
        
    if USE_S3 and not SKIP_FSL:
        # Upload processed MRI to s3 bucket
        s3_loc = Path(f"{run_name}/{group}/{scan_name}_processed.nii.gz")
        cprint(f"INFO: uploading processed MRI to s3 bucket {AWS_S3_BUCKET_NAME} in {s3_loc}", "blue")
        s3_bucket  = boto3.resource('s3').Bucket('processed-nii-files')

        try:
            s3_bucket.upload_file(str(nii_path), str(s3_loc))
            cprint("INFO: Successfully uploaded file to s3 bucket", "blue") 
        except:
            cprint("ERROR: Failed to upload file to s3 bucket", "red")
            cprint("INFO: Will attempt to sync bucket at end of run", "blue") 

    # To access slices:
    # sagittal = data[26, :, :] <- 26th slice along sagittal
    # coronal = data[:, 30, :] <- 30th slice along coronal
    # axial = data[:, :, 50] <- 50th slice along axial

    if not SKIP_SLICE_CREATION:
        # Split into multichannel slices
        create_image_slices_from_brain(
            nii_path, out_dir, scan_name, group, slice_range)
    
    if not SKIP_FSL:
        return (original_brain, nii_path)



