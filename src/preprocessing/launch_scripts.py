import pandas as pd
import json
import shutil
import splitfolders
import subprocess
from termcolor import colored, cprint
from multiprocessing import Pool
from datetime import datetime
from ..constants import *
from ..settings import *
from .prep_nii import prep_raw_mri
from .utils import write_batch_to_log, create_slices_from_brain, create_multichannel_slices_from_brain
from pathlib import Path

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

def prep_adni(collection_dir, run_name, split_ratio, collection_csv=None):
    """Runs prep scripts for an ADNI image directory.
        Will create 5 subdirs
            -  nii_files: Containing the FSL-processed nii images, separated by class
            -  image_slices: Containing the individual image slices, separated by class
            -  multi_channel: Containing the multi-channel image slices, separated by class
            -  slice_dataset: Containing the individual image slices, separated by class and into train/test/val
            -  multichannel_dataset: Containing the multi-channel image slices, separated by class and into train/test/val

    Args:
        collection_dir (str): The image collection directory
        run_name (str): Name of the prep run. Outputs will be stored in out/preprocessed_datasets/{run_name}
        split_ratio (tuple): The train/test/val split ratio.
        collection_csv (str, optional): The csv containing the collection metadata. Defaults to None.

    Raises:
        ValueError: Checks if collection dir exists
        ValueError: Checks if collection dir has images
        ValueError: Checks if output dir has been used (avoids overwriting)
    """
    prep_start_time = datetime.now()
    
    # Resetting path locs
    collection_dir = Path(cwd, collection_dir).resolve()
    if not SKIP_FSL:
        collection_csv = Path(cwd, collection_csv).resolve()
    
    if SKIP_FSL and SKIP_SLICE_CREATION and SKIP_FOLDER_SPLIT:
        raise ValueError("SKIP_FSL, SKIP_SLICE_CREATION, and SKIP_FOLDER_SPLIT are True. Nothing to do.")

    if not Path.exists(collection_dir):
        raise ValueError(
            colored(f"Collection dir {collection_dir} does not exist!", "red"))
    elif not any(Path(collection_dir).iterdir()):
        raise ValueError(
            colored(f"Collection path {collection_dir} empty!", "red"))

    if not SKIP_FSL:
        # Getting individual subjects
        # For each subject we only really need one scan, and we're only using AD and CN, not MCI
        subjects = pd.read_csv(collection_csv)
        subjects.drop_duplicates(keep='first',subset=['Subject'], inplace=True)
        subjects.drop(subjects[subjects['Group'] == 'MCI'].index, inplace=True)
        
        est_batches = len(subjects) / FSL_CONCURRENT_PROCESSES

    out_dir = Path(
        filedir, "../../out/preprocessed_datasets", run_name).resolve()

    cprint(f"INFO: output dir: {out_dir}", "blue")

    # subjects that have already been processed
    done_subjects = dict()
    done_subjects["Original Path"] = []
    done_subjects["Output Path"] = []
    done_subjects["Subject Name"] = []
    
    done_subjects = pd.DataFrame.from_dict(done_subjects)

    try:
        out_dir.mkdir(parents=True, exist_ok=False)

        if not SKIP_FSL:
            # This is the csv file that will contain the original path, output path, and group. helps if we have a crash
            # if this is done here, this is our first run, so we can create the csv
            csv_dir = Path(out_dir, "processed.csv").resolve()
            with open(csv_dir, "w") as csv:
                csv.write('"Original Path","Output Path","Group"\n')
                
    except:
        
        # If we're skipping fsl we just get rid of the dir
        if SKIP_FSL:
            # If slices exist we raise error. Else, we create the dir
            if not SKIP_SLICE_CREATION:
                if Path(out_dir, "image_slices").exists():
                    raise ValueError("Output dir already exists. Please delete or change run name")
            # TODO handle skip_folder_split
            
        else:
            # As we're using the same output dir, we need to check if we've already processed some of the subjects.
            done_subjects = pd.read_csv(Path(out_dir, "processed.csv"))
            cprint(f"INFO: {len(done_subjects)} scans already processed. Skipping...", "blue")

            # Converting paths to Path objects
            done_subjects["Original Path"] = [
                Path(s) for s in done_subjects["Original Path"]]
            done_subjects["Output Path"] = [
                Path(s) for s in done_subjects["Output Path"]]

            # removes "_NN_processsed.nii.gz"
            done_subjects["Subject Name"] = [s.name[:10]
                                            for s in done_subjects["Output Path"]]

    if not SKIP_FSL:
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

    # FSL prep chunk
    if not SKIP_FSL:
        fsl_start_time = datetime.now()

        # queued_mris is the current queue of mris to prep concurrently
        queued_mris = []
        current_batch = 0

        # first loop: goes through each subject
        for _, subject in subjects.iterrows():
            # This skips the subject if we've already processed it
            if subject["Subject"] in done_subjects["Subject Name"].values:
                continue
            
            base_folder = Path(collection_dir, subject["Subject"])
            
            # we just get the first scan. Sorted so we always get the same one per subject
            for p in sorted(base_folder.rglob("*")):
                if p.name.endswith(".nii"):
                    scan_folder = p
                    break

            current_subject = [scan_folder, subject["Subject"],
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
                write_batch_to_log(complete_pairs, out_dir, successful_str)

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
            write_batch_to_log(complete_pairs, out_dir, successful_str)

        fsl_end_time = datetime.now()
        cprint(f"SUCCESS: All FSL scripts took {str(fsl_end_time-fsl_start_time)} to run", "green")
    
    # this means we'll take care of slice creation ad-hoc
    elif not SKIP_SLICE_CREATION:
        for imgpath in collection_dir.rglob("*.nii.gz"):
            nii_path = Path(imgpath)
            scan_name = nii_path.name[:10]
            group = nii_path.parent.name
            # TODO: make slice range a setting
            # Split into individual slices
            create_slices_from_brain(nii_path, out_dir, scan_name, group)

            # Split into multichannel slices
            create_multichannel_slices_from_brain(nii_path, out_dir, scan_name, group)
        
        
    # slice creation chunk TODO: check that the folder split hasn't already been done
    if not SKIP_FOLDER_SPLIT:
        slice_split_loc = Path(collection_dir, "image_slices").resolve() if SKIP_SLICE_CREATION and SKIP_FSL else Path(out_dir, "image_slices")
        multichannel_split_loc = Path(collection_dir, "multi_channel").resolve() if SKIP_SLICE_CREATION and SKIP_FSL else Path(out_dir, "multi_channel")
        split_seed = datetime.now().timestamp()
        
        cprint(f"INFO: Splitting slice folders with split ratio {split_ratio}", "blue")
        # NOTE: group_prefix doesn't actually change results thankfully, but might be a more "realistic" way to train
        # Not to mention this opens possibility of testing with nii files without having overlaps with train dataset
        splitfolders.ratio(slice_split_loc, output=Path(out_dir, "slice_dataset"),
                        seed=split_seed, ratio=split_ratio, group_prefix=10)

        splitfolders.ratio(multichannel_split_loc, output=Path(out_dir, "multichannel_dataset"),
                        seed=split_seed, ratio=split_ratio, group_prefix=10)

    cprint("SUCCESS: Done processing raw MRIs. Saving meta data", "green")

    scan_count = len(list(Path(out_dir, "nii_files").glob('**/*')))
    slice_count = len(list(Path(out_dir, "image_slices").glob('**/*')))

    prep_end_time = datetime.now()
    
    # Writing meta file
    with open(Path(out_dir, "meta.json"), "w") as meta_file:
        metadata = {
            "kaggle": False,
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
        shutil.copyfile(collection_csv, Path(out_dir, "collection.csv"))

    if USE_S3:
        try:  # TODO don't use subprocess, use boto3 w custom function
            cmd = f"aws s3 sync {out_dir}/nii_files s3://{AWS_S3_BUCKET_NAME}/{run_name}"
            subprocess.run(cmd, shell=True)
        except:
            cprint("ERROR: Failed to sync files to s3 bucket", "red")
            cprint(f"INFO: Can be done manually using command {cmd}", "blue")

    cprint(f"Done! Result files found in {out_dir}", "green")

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
        raise ValueError(colored(f"Kaggle path {kaggle_dir} does not exist!", "red"))
    elif not any(Path(kaggle_dir).iterdir()):
        raise ValueError(colored(f"Kaggle path {kaggle_dir} empty!", "red"))

    out_dir = Path(
        filedir, "../../out/preprocessed_datasets", run_name).resolve()

    print(f"output dir: {out_dir}")

    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except:
        raise ValueError(
            colored(f"Output dir {out_dir} already exists! Pick a different run name or delete the existing directory.", "red"))

    split_seed = datetime.now().timestamp()

    print(f"Splitting folders with split ratio {split_ratio}")
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

    cprint(f"Done! Result files found in {out_dir}", "green")
