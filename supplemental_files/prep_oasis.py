"""This is a script to preprocess the OASIS dataset for use as a test dataset.
Functionality for this script is more stripped down than ADNI prep script.
This is due to the fact that it is a smaller dataset, and we don't need to
watch out for failed runs etc.

This should ideally be integrated into the main prep script, but for now it
is a separate script for ease of use and implementation.

"""


import shutil
import pandas as pd
import json
import shutil
import splitfolders
import subprocess
import boto3
import numpy as np
import nibabel as nib
import os
from PIL import Image
from termcolor import colored, cprint
from multiprocessing import Pool
from datetime import datetime
from termcolor import cprint
from pathlib import Path
from fsl.wrappers import fsl_anat, fslmaths, fnirt, flirt, bet

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()
USE_S3 = True
AWS_S3_BUCKET_NAME="processed-nii-files"
FSL_CONCURRENT_PROCESSES=6
FSLDIR = os.getenv('FSLDIR')
split_ratio = [0.8, 0.1, 0.1]

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
    
    # The tmp_dir directory will be used to store all the fsl_anat info
    tmp_dir = Path(
        filedir, "../out/preprocessed_datasets/tmp", scan_name).resolve()

    # fsl_anat adds .anat to end of output directory
    anat_dir = Path(f"{tmp_dir}.anat")

    fsl_anat(img=scan_location, out=tmp_dir, noseg=True,
                nosubcortseg=True, nononlinreg=True, noreg=True, nocleanup=True)

    cprint("INFO: fsl_anat complete. Running flirt to register to MNI space", "blue")
    # We're runnning flirt with custom parameters to improve resolution from 2mm to 1mm
    flirt(src=Path(anat_dir, "T1_biascorr"),
            ref=f"{FSLDIR}/data/standard/MNI152_T1_1mm",
            interp="spline",
            dof=12,
            v=True,
            omat=Path(anat_dir, "T1_to_MNI_lin.mat"),
            out=Path(anat_dir, "T1_to_MNI_lin")
            )

    bet(Path(anat_dir, "T1_to_MNI_lin.nii.gz"), 
        Path(anat_dir, "T1_to_MNI_lin_brain"), f=0.3, g=0, t=True)
    
    cprint("INFO: finalized flirt and brain extraction", "blue")
    
    # File is saved into group subfolder in nii_files output loc
    final_brain = Path(
        out_dir, f"nii_files/{group}/{scan_name}_processed.nii.gz")

    shutil.move(Path(anat_dir, "T1_to_MNI_lin_brain.nii.gz"), final_brain)

    # clearing all the .anat files (unnecessary now)
    shutil.rmtree(anat_dir)

    return (scan_location, final_brain)

def create_image_slices_from_brain(nii_path, out_dir, scan_name, group, slice_range=(80, 110)): # TODO: add option for saggital, coronal, and axial slices
    """Creates the axial image slices from an input nii image and slice range

    Args:
        nii_path (str): path of the nii image to extract the slices from
        out_dir (str): output directory. slices will be placed in {out_dir}/axial_slices/{group}
        scan_name (str): The name of the scan (format NNN_S_NNNN_NN)
        group (str): Class of the image. Can be CN, AD, or MCI
        slice_range (tuple, optional): The slices to be extracted. Defaults to (80, 110).
        
    """
    brain_data = get_data_from_nii(nii_path)

    for i in range(slice_range[0], slice_range[1], 3):
        # Vital to make sure that the np.float64 is correctly scaled to np.uint8
        # We do 3 slices (r=i-1,g=i,b=i+1)
        r_slice = brain_data[:, :, i-1]
        g_slice = brain_data[:, :, i]
        b_slice = brain_data[:, :, i+1]

        # We stack these into one nparray that will have shape (91,109,3)
        slice_3d = normalize_array_range(np.stack((r_slice, g_slice, b_slice), axis=2))
        
        image_data = Image.fromarray(slice_3d)
        
        # Saved as axial_slices/{group}/{subject}_slice{number}
        image_dir = Path(out_dir, f"axial_slices/{group}/{scan_name}_slice{(i-slice_range[0])//3}.png").resolve()
        image_data.save(image_dir)

def normalize_array_range(img):
    """Normalizes range of np array values, moving them to the range 0-255. Important for RGB image gen

    Args:
        img (nparray): The 2D array to normalize

    Returns:
        nparray: the normalized 2D array. 
    """
    TARGET_TYPE_MAX = 255
    TARGET_TYPE = np.uint8

    imin = np.min(img)
    imax = np.max(img)

    coeff = (img - imin) / (imax - imin) 
    newimg = (coeff * TARGET_TYPE_MAX ).astype(TARGET_TYPE)
    return newimg

def get_data_from_nii(nii_path):
    """Extracts nparray from given nii file

    Args:
        nii_path (str): The path of the nii file to get the data from

    Returns:
        nparray: The nii file's nparray
    """
    imgfile = nib.load(nii_path)
    return np.array(imgfile.dataobj)

def prep_raw_mri(scan_loc, scan_name, out_dir, group, run_name, slice_range=(80, 110)): 
    """Prepares a singular raw mri image. This means that FSL is ran, 
    and axial slices extracted. A dataset is created for the slices.
    TODO: add option for saggital, coronal, and axial slices

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

    print(f"Launching FSL scripts for scan {scan_name} in group {group}")
    
    original_brain, nii_path = run_fsl(scan_location, scan_name, group, out_dir)
    
    print(f"FSL scripts complete. Processed MRI found in {nii_path}")

    if USE_S3:
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

    # Split into slices
    create_image_slices_from_brain(
        nii_path, out_dir, scan_name, group, slice_range)

    return (original_brain, nii_path)


if __name__ == "__main__":
    collection_dir = "supplemental_files/unprocessed_datasets/OASIS"
    collection_csv = "supplemental_files/unprocessed_datasets/OASIS/OASIS.csv"
    run_name = "oasis_processed"

    prep_start_time = datetime.now()
    
    # Resetting path locs
    collection_dir = Path(cwd, collection_dir).resolve()
    collection_csv = Path(cwd, collection_csv).resolve()
    
    if not Path.exists(collection_dir):
        raise ValueError(
            colored(f"Collection dir {collection_dir} does not exist!", "red"))
    elif not any(Path(collection_dir).iterdir()):
        raise ValueError(
            colored(f"Collection path {collection_dir} empty!", "red"))

    # Getting individual subjects
    # For each subject we only really need one scan, and we're only using AD and CN, not MCI
    subjects = pd.read_csv(collection_csv)
    
    # splitting subject name into subject and scan (3rd _ is the scan name)
    subjects['ID'] = subjects['ID'].str.split('_', expand=True)[0] + "_" + subjects['ID'].str.split('_', expand=True)[1]    
    subjects.drop_duplicates(keep='first',subset=['ID'], inplace=True)
    subjects.drop(subjects[subjects['CDR'] == None].index, inplace=True)
    
    # All subjects with CDR of 0 are CN
    subjects['Group'] = subjects['CDR'].apply(lambda x: "CN" if x == 0 else "AD")
        
    est_batches = len(subjects) / FSL_CONCURRENT_PROCESSES

    out_dir = Path(
        filedir, "../out/preprocessed_datasets", run_name).resolve()

    cprint(f"INFO: output dir: {out_dir}", "blue")

    # subjects that have already been processed
    out_dir.mkdir(parents=True, exist_ok=False)

    # Creating group subdirs for output nii images
    Path(out_dir, "nii_files/CN").resolve().mkdir(parents=True, exist_ok=True)
    Path(out_dir, "nii_files/AD").resolve().mkdir(parents=True, exist_ok=True)

    # Creating group subdirs for output image slices TODO: add option for saggital, coronal, and axial slices
    Path(out_dir, "axial_slices/CN").resolve().mkdir(parents=True, exist_ok=True)
    Path(out_dir, "axial_slices/AD").resolve().mkdir(parents=True, exist_ok=True)

    # FSL prep chunk
    fsl_start_time = datetime.now()

    # queued_mris is the current queue of mris to prep concurrently
    queued_mris = []
    current_batch = 0

    cprint("INFO: Converting all scans to nii format", "blue")
    for _, subject in subjects.iterrows():
        base_folder = Path(collection_dir, subject["ID"] + "_MR1").resolve()
        specific_scan_folder = Path(base_folder, "PROCESSED", "MPRAGE","SUBJ_111").resolve()
        out_nii_path = Path(specific_scan_folder, "subj.nii").resolve()
        if out_nii_path.exists():
            subprocess.call(["rm", str(out_nii_path)])
        
        #converting to nii in same folder
        for p in sorted(specific_scan_folder.rglob("*.img")):
            scan_folder = p
            break
        
        subprocess.call(["fslchfiletype", "NIFTI", str(scan_folder), str(out_nii_path)])
        
        
    cprint("SUCCESS: All scans converted to nii format!", "green")
        
    # first loop: goes through each subject
    for _, subject in subjects.iterrows():
        base_folder = Path(collection_dir, subject["ID"] + "_MR1").resolve()
        scan_folder = Path(base_folder, "PROCESSED", "MPRAGE","SUBJ_111","subj.nii").resolve()
        
        current_subject = [scan_folder, subject["ID"],
                        out_dir, subject["Group"], run_name]

        queued_mris.append(current_subject)

        # If we've collected the # of concurrent mris we can then run the prep multiprocessed
        if len(queued_mris) == FSL_CONCURRENT_PROCESSES:
            batch_start_time = datetime.now()

            # prep all the queued MRIs at once
            pool = Pool(processes=FSL_CONCURRENT_PROCESSES)
            complete_pairs = pool.starmap(prep_raw_mri, queued_mris)

            batch_end_time = datetime.now()

            successful_str = f"SUCCESS: Batch {current_batch}/{est_batches}. It took {str(batch_end_time-batch_start_time)} to preprocess"

            cprint(successful_str, "green")

            # clear the queue
            queued_mris.clear()
            current_batch += 1

    # finishing up the last batch
    if len(queued_mris) != 0:
        batch_start_time = datetime.now()
        # prep all the queued MRIs at once
        pool = Pool(processes=len(queued_mris))
        pool.starmap(prep_raw_mri, queued_mris)
        batch_end_time = datetime.now()

        successful_str = f"SUCCESS: Batch {current_batch}/{est_batches}. It took {str(batch_end_time-batch_start_time)} to preprocess"

        cprint(successful_str, "green")

    fsl_end_time = datetime.now()
    cprint(f"SUCCESS: All FSL scripts took {str(fsl_end_time-fsl_start_time)} to run", "green")
    
    # this means we'll take care of slice creation ad-hoc
    for imgpath in collection_dir.rglob("*.nii.gz"):
        nii_path = Path(imgpath)
        
        # This will be slightly different for OASIS
        scan_name = nii_path.name[:9]
        group = nii_path.parent.name

        # Split into slices
        create_image_slices_from_brain(nii_path, out_dir, scan_name, group)
        
    # slice creation chunk TODO: check that the folder split hasn't already been done
    dataset_loc = Path(out_dir, "axial_slices")
    split_seed = datetime.now().timestamp()
    
    cprint(f"INFO: Splitting slice folders with split ratio {split_ratio}", "blue")
    splitfolders.ratio(dataset_loc, output=Path(out_dir, "axial_dataset"),
                    seed=split_seed, ratio=split_ratio, group_prefix=15)

    cprint("SUCCESS: Done processing raw MRIs. Saving meta data", "green")

    scan_count = len(list(Path(out_dir, "nii_files").glob('**/*')))
    slice_count = len(list(Path(out_dir, "axial_slices").glob('**/*'))) #TODO: check sagittal and coronals

    prep_end_time = datetime.now()
    
    # Writing meta file
    with open(Path(out_dir, "meta.json"), "w") as meta_file:
        metadata = {
            "run_name": run_name,
            "original_dir": str(collection_dir),
            "split": list(split_ratio),
            "scan_count": scan_count,
            "slice_count": slice_count,
            "dataset_split_seed": split_seed if split_seed else None, # won't always be set
            "prep_time": str(prep_end_time-prep_start_time),
        }
        json.dump(metadata, meta_file, indent=4)

    if collection_csv:
        # Writing collection.csv file
        shutil.copyfile(collection_csv, Path(out_dir, "OASIS.csv"))

    if USE_S3:
        try:  # TODO don't use subprocess, use boto3 w custom function
            cmd = f"aws s3 sync {out_dir}/nii_files s3://{AWS_S3_BUCKET_NAME}/oasis"
            subprocess.run(cmd, shell=True)
        except:
            cprint("ERROR: Failed to sync files to s3 bucket", "red")
            cprint(f"INFO: Can be done manually using command {cmd}", "blue")

    cprint(f"Done! Result files found in {out_dir}", "green")