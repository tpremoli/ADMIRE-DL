import pandas as pd
import json
import shutil
import splitfolders
import boto3
import subprocess
from termcolor import colored, cprint
from multiprocessing import Pool
from datetime import datetime
from ..constants import *
from ..settings import *
from .prep_raw import prep_raw_mri
from .utils import write_batch_to_log
from pathlib import Path

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()


def prep_adni(collection_dir, collection_csv, run_name, split_ratio):
    """Runs prep scripts for an ADNI image directory.
        Will create 5 subdirs
            -  nii_files: Containing the FSL-processed nii images, separated by class
            -  image_slices: Containing the individual image slices, separated by class
            -  multi_channel: Containing the multi-channel image slices, separated by class
            -  slice_dataset: Containing the individual image slices, separated by class and into train/test/val
            -  multichannel_dataset: Containing the multi-channel image slices, separated by class and into train/test/val

    Args:
        collection_dir (str): The image collection directory
        collection_csv (str): The image metadata file directory
        run_name (str): Name of the prep run. Outputs will be stored in out/preprocessed_datasets/{run_name}
        split_ratio (tuple): The train/test/val split ratio.

    Raises:
        ValueError: Checks if collection dir exists
        ValueError: Checks if collection dir has images
        ValueError: Checks if output dir has been used (avoids overwriting)
    """
    # Resetting path locs
    collection_dir = Path(cwd, collection_dir).resolve()
    collection_csv = Path(cwd, collection_csv).resolve()

    if not Path.exists(collection_dir):
        raise ValueError(
            colored("Collection dir {} does not exist!".format(collection_dir), "red"))
    elif not any(Path(collection_dir).iterdir()):
        raise ValueError(
            colored("Collection path {} empty!".format(collection_dir), "red"))

    # Getting individual subjects
    subjects = pd.read_csv(collection_csv)
    est_batches = len(subjects) / FSL_CONCURRENT_PROCESSES
    
    subjects = subjects.drop_duplicates(
        keep='first', subset="Subject").to_dict(orient="records")

    out_dir = Path(
        filedir, "../../out/preprocessed_datasets", run_name).resolve()

    cprint("INFO: output dir: {}".format(out_dir), "blue")

    done_subjects = []
    max_count = dict.fromkeys([subject["Subject"] for subject in subjects], -1) # This keeps track of the maxcount of each subject

    try:
        out_dir.mkdir(parents=True, exist_ok=False)
        
        # This is the csv file that will contain the original path, output path, and group. helps if we have a crash
        csv_dir = Path(out_dir, "processed.csv").resolve()
        with open(csv_dir, "w") as csv:
            csv.write('"Original Path","Output Path","Group"\n')
            
    except:
        # As we're using the same output dir, we need to check if we've already processed some of the subjects.
        done_subjects = pd.read_csv(Path(out_dir, "processed.csv"))
        cprint("INFO: {} scahs already processed. Skipping...".format(len(done_subjects)), "blue")
        
        # Converting paths to Path objects
        done_subjects["Original Path"] = [Path(s) for s in done_subjects["Original Path"]]
        done_subjects["Output Path"] = [Path(s) for s in done_subjects["Output Path"]]
        
        # Parses the count from the output path
        done_subjects["Count"] = [int(s.name[11:13]) for s in done_subjects["Output Path"]]
        done_subjects["Subject Name"] = [s.name[:10] for s in done_subjects["Output Path"]]  # removes "_NN_processsed.nii.gz"

        # This is useful for checking if we've already processed a subject
        done_subjects["Date Dir"] = [s.parent.parent for s in done_subjects["Original Path"]]

        # this updates max_count to the max count of each subject that has already been processed
        for _, subject in done_subjects.iterrows():
            if subject["Count"] > max_count[subject["Subject Name"]]:
                max_count[subject["Subject Name"]] = subject["Count"]  

    # Creating group subdirs for output nii images
    Path(out_dir, "nii_files/CN").resolve().mkdir(parents=True, exist_ok=True)
    Path(out_dir, "nii_files/AD").resolve().mkdir(parents=True, exist_ok=True)

    if not SKIP_SLICE_CREATION:
        # Creating group subdirs for output image slices
        Path(out_dir, "image_slices/CN").resolve().mkdir(parents=True, exist_ok=True)
        Path(out_dir, "image_slices/AD").resolve().mkdir(parents=True, exist_ok=True)

        # Creating group subdirs for output multi-channel image slices
        Path(out_dir, "multi_channel/CN").resolve().mkdir(parents=True, exist_ok=True)
        Path(out_dir, "multi_channel/AD").resolve().mkdir(parents=True, exist_ok=True)

    fsl_start_time = datetime.now()

    queued_mris = []  # queued_mris is the current queue of mris to prep concurrently
    current_batch = 0
    
    for subject in subjects:
        subj_folder = Path(collection_dir, subject["Subject"], "MP-RAGE") # TODO: MP-RAGE_repeat isnt covered here
        
        # For each scan in the subject subject
        for scan_folder in Path.glob(subj_folder, "*"):
            
            # This skips the scan_folder if we've already processed it
            if scan_folder in done_subjects["Date Dir"].values:
                continue
            
            # Added new subject so we increase the max count
            max_count[subject["Subject"]] += 1
            
            # This makes the name styled 002_S_0295_{no} where no is the number of sampel we're on. min 6 chars
            scan_name = "{}_{:02d}".format(subject["Subject"], max_count[subject["Subject"]])
            
            current_subject = [scan_folder, scan_name,
                               out_dir, subject["Group"], run_name]

            queued_mris.append(current_subject)

            # If we've collected the # of concurrent mris we can then run the prep multiprocessed
            if len(queued_mris) == FSL_CONCURRENT_PROCESSES:
                batch_start_time = datetime.now()

                # prep all the queued MRIs at once
                pool = Pool(processes=FSL_CONCURRENT_PROCESSES)
                complete_pairs = pool.starmap(prep_raw_mri, queued_mris)

                batch_end_time = datetime.now()

                successful_str = "SUCCESS: Batch {}/{}. It took {} to preprocess".format(
                    current_batch, est_batches, str(batch_end_time-batch_start_time))

                cprint(successful_str, "green")
                write_batch_to_log(complete_pairs, out_dir, successful_str)

                # clear the queue
                queued_mris.clear()
                current_batch += 1

    if len(queued_mris) != 0:
        batch_start_time = datetime.now()
        # prep all the queued MRIs at once
        pool = Pool(processes=len(queued_mris))
        pool.starmap(prep_raw_mri, queued_mris)
        batch_end_time = datetime.now()

        successful_str = "SUCCESS: Batch {}/{}. It took {} to preprocess".format(
            current_batch, est_batches, str(batch_end_time-batch_start_time))

        cprint(successful_str, "green")
        write_batch_to_log(complete_pairs, out_dir, successful_str)

    fsl_end_time = datetime.now()
    cprint("SUCCESS: All FSL scripts took {} to run".format(
        str(fsl_end_time-fsl_start_time)), "green")

    split_seed = datetime.now().timestamp()
    cprint("INFO: Splitting slice folders with split ratio {}".format(split_ratio), "blue")
    splitfolders.ratio(Path(out_dir, "image_slices"), output=Path(out_dir, "slice_dataset"),
                       seed=split_seed, ratio=split_ratio)

    splitfolders.ratio(Path(out_dir, "multi_channel"), output=Path(out_dir, "multichannel_dataset"),
                       seed=split_seed, ratio=split_ratio)

    cprint("SUCCESS: Done processing raw MRIs. Saving meta data", "green")

    scan_count = len(list(Path(out_dir, "nii_files").glob('**/*')))
    slice_count = len(list(Path(out_dir, "image_slices").glob('**/*')))

    # Writing meta file
    with open(Path(out_dir, "meta.json"), "w") as meta_file:
        metadata = {
            "kaggle": False,
            "run_name": run_name,
            "original_dir": str(collection_dir),
            "split": list(split_ratio),
            "scan_count": scan_count,
            "slice_count": slice_count,
            "dataset_split_seed": split_seed,
        }
        json.dump(metadata, meta_file, indent=4)

    # Writing collection.csv file
    shutil.copyfile(collection_csv, Path(out_dir, "collection.csv"))
    
    if USE_S3: 
        try: # TODO don't use subprocess, use boto3 w custom function
            cmd = "aws s3 sync {}/nii_files s3://{}/{}".format(out_dir, AWS_S3_BUCKET_NAME, run_name)
            subprocess.run(cmd, shell=True)
        except:
            cprint("ERROR: Failed to upload file to s3 bucket", "red")
            cprint("INFO: Will attempt to sync bucket at end of run", "blue") # TODO: implement this


    cprint("Done! Result files found in {}".format(out_dir), "green")

def prep_adni_nofsl(collection_dir, run_name, split_ratio): #TODO: maybe join to prep_adni
    """Preps the ADNI dataset WITHOUT running FSL scripts. Note how no collection_csv is passed in
        Will create 4 subdirs
            -  image_slices: Containing the individual image slices, separated by class
            -  multi_channel: Containing the multi-channel image slices, separated by class
            -  slice_dataset: Containing the individual image slices, separated by class and into train/test/val
            -  multichannel_dataset: Containing the multi-channel image slices, separated by class and into train/test/val

    Args:
        collection_dir (str): The image collection directory
        run_name (str): Name of the prep run. Outputs will be stored in out/preprocessed_datasets/{run_name}
        split_ratio (tuple): The train/test/val split ratio.

    Raises:
        ValueError: Checks if collection dir exists
        ValueError: Checks if collection dir has images
        ValueError: Checks if output dir has been used (avoids overwriting)
    """
    # Resetting path locs
    collection_dir = Path(cwd, collection_dir).resolve()

    if not Path.exists(collection_dir):
        raise ValueError(
            colored("Collection dir {} does not exist!".format(collection_dir), "red"))
    elif not any(Path(collection_dir).iterdir()):
        raise colored(ValueError(
            "Collection path {} empty!".format(collection_dir), "red"))

    out_dir = Path(
        filedir, "../../out/preprocessed_datasets", run_name).resolve()

    print("output dir: {}".format(out_dir))

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except:
        raise ValueError(
            colored("Output dir {} already exists! Pick a different run name or delete the existing directory.".format(out_dir), "red"))

    # Creating group subdirs for output image slices
    Path(out_dir, "image_slices/CN").resolve().mkdir(parents=True, exist_ok=True)
    Path(out_dir, "image_slices/AD").resolve().mkdir(parents=True, exist_ok=True)

    # Creating group subdirs for output multi-channel image slices
    Path(out_dir, "multi_channel/CN").resolve().mkdir(parents=True, exist_ok=True)
    Path(out_dir, "multi_channel/AD").resolve().mkdir(parents=True, exist_ok=True)

    #TODO: Add remove this log stuff
    log_file = open(Path(out_dir, "log"), "w")
    total_start_time = datetime.now()

    queued_mris = []  # queued_mris is the current queue of mris to prep concurrently
    current_batch = 0

    for scan_loc in Path.rglob(collection_dir, "*.nii.gz"):
        # 007_S_0070_00_processed.nii.gz
        scan_name = str(scan_loc.name)[:13]  # removes "_processsed.nii.gz"

        current_subject = [scan_loc, scan_name,
                           out_dir, scan_loc.parent.name, run_name]

        queued_mris.append(current_subject)

        # If we've collected the # of concurrent mris we can then run the prep multiprocessed
        if len(queued_mris) == FSL_CONCURRENT_PROCESSES:
            batch_start_time = datetime.now()

            # prep all the queued MRIs at once
            pool = Pool(processes=FSL_CONCURRENT_PROCESSES)
            pool.starmap(prep_raw_mri, queued_mris)

            batch_end_time = datetime.now()

            log_file.write("batch {} took {} to preprocess\n".format(
                current_batch, str(batch_end_time-batch_start_time)))

            # clear the queue
            queued_mris.clear()
            current_batch += 1

    if len(queued_mris) != 0:
        batch_start_time = datetime.now()
        # prep all the queued MRIs at once
        pool = Pool(processes=len(queued_mris))
        pool.starmap(prep_raw_mri, queued_mris)
        batch_end_time = datetime.now()
        log_file.write("batch {} took {} to preprocess\n".format(
            current_batch, str(batch_end_time-batch_start_time)))

    total_end_time = datetime.now()
    log_file.write("All FSL scripts took {} to run".format(
        str(total_end_time-total_start_time)))
    log_file.close()

    split_seed = datetime.now().timestamp()
    print("Splitting slice folders with split ratio {}".format(split_ratio))
    splitfolders.ratio(Path(out_dir, "image_slices"), output=Path(out_dir, "slice_dataset"),
                       seed=split_seed, ratio=split_ratio)

    splitfolders.ratio(Path(out_dir, "multi_channel"), output=Path(out_dir, "multichannel_dataset"),
                       seed=split_seed, ratio=split_ratio)

    print("Done processing raw MRIs. Saving mata data")

    scan_count = len(list(Path(out_dir, "nii_files").glob('**/*')))
    slice_count = len(list(Path(out_dir, "image_slices").glob('**/*')))

    # Writing meta file
    with open(Path(out_dir, "meta.json"), "w") as meta_file:
        metadata = {
            "kaggle": False,
            "run_name": run_name,
            "original_dir": str(collection_dir),
            "split": list(split_ratio),
            "scan_count": scan_count,
            "slice_count": slice_count,
            "dataset_split_seed": split_seed,
        }
        json.dump(metadata, meta_file, indent=4)

    cprint("Done! Result files found in {}".format(out_dir), "green")


def prep_kaggle(kaggle_dir, run_name, split_ratio):
    """Runs prep scripts for a kaggle image directory.
        Will create train/test/val split folders

    Args:
        kaggle_dir (str): directory of where to find the kaggle images
        run_name (str): Name of the prep run. Outputs will be stored in out/preprocessed_datasets/{run_name}
        split_ratio (tuple): The train/test/val split ratio.

    Raises:
        ValueError: Checks if kaggle dir exists
        ValueError: Checks if kaggle dir has images
        ValueError: Checks if output dir has been used (avoids overwriting)
    """
    kaggle_dir = Path(cwd, kaggle_dir).resolve()

    if not Path.exists(kaggle_dir):
        raise ValueError(
            colored("Kaggle path {} does not exist!".format(kaggle_dir), "red"))
    elif not any(Path(kaggle_dir).iterdir()):
        raise ValueError(
            colored("Kaggle path {} empty!".format(kaggle_dir), "red"))

    out_dir = Path(
        filedir, "../../out/preprocessed_datasets", run_name).resolve()

    print("output dir: {}".format(out_dir))

    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except:
        raise ValueError(
            colored("Output dir {} already exists! Pick a different run name or delete the existing directory.".format(out_dir), "red"))

    split_seed = datetime.now().timestamp()

    print("Splitting folders with split ratio {}".format(split_ratio))
    splitfolders.ratio(kaggle_dir, output=out_dir,
                       seed=split_seed, ratio=split_ratio)

    train_count = len(list(Path(out_dir, "train").glob('**/*')))
    test_count = len(list(Path(out_dir, "test").glob('**/*')))
    val_count = len(list(Path(out_dir, "val").glob('**/*')))

    with open(Path(out_dir, "meta.json"), "w") as meta_file:
        metadata = {
            "kaggle": True,
            "run_name": run_name,
            "original_dir": str(kaggle_dir),
            "split": list(split_ratio),
            "train_count": train_count,
            "test_count": test_count,
            "val_count": val_count,
            "dataset_split_seed": split_seed,
        }
        json.dump(metadata, meta_file, indent=4)

    cprint("Done! Result files found in {}".format(out_dir), "green")
