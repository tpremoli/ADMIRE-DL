from pathlib import Path
from termcolor import cprint
import shutil
import os
import subprocess
import sys

cwd = Path.cwd()

if __name__ == '__main__':
    # get rootdir from args
    if len(sys.argv) != 1:
        raise ValueError("ERROR: Need one argument - rootdir of OASIS dataset")
    
    rootdir = Path(cwd, sys.argv[1]).resolve()
    
    # Directory to save results
    niftidir = rootdir.parent / 'oasis_nifti'

    # Create directory to save the results, if not existing
    (niftidir).mkdir(parents=True, exist_ok=True)

    # For each subject
    for s in sorted(rootdir.iterdir()):
        # Skip non-directory entries
        if not s.is_dir():
            continue

        # Some feedback in the screen
        print(s.name)

        # Directory of the original, raw data
        raw_data_path = s / 'RAW'

        # For each acquisition
        for a in sorted(raw_data_path.iterdir()):
            if not a.name.endswith('.hdr'):
                continue

            output_file = niftidir / a.stem

            # Do each of the 6 steps described in the blog
            commands = [
                f"{os.environ['FSLDIR']}/bin/fslchfiletype NIFTI_GZ {a} {output_file}",
                f"{os.environ['FSLDIR']}/bin/fslorient -deleteorient {output_file}",
                f"{os.environ['FSLDIR']}/bin/fslorient -setsformcode 2 {output_file}",
                f"{os.environ['FSLDIR']}/bin/fslorient -setsform 0 0 -1.25 0 1 0 0 0 0 1 0 0 0 0 0 1 {output_file}",
                f"{os.environ['FSLDIR']}/bin/fslswapdim {output_file} RL PA IS {output_file}",
                f"{os.environ['FSLDIR']}/bin/fslorient -setqform -1.25 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 {output_file}"
            ]

            for command in commands:
                subprocess.run(command, shell=True, check=True)
                
            break # Only do the first acquisition
    
    # copy oasis.csv file to nifti directory
    # this file contains the labels for the subjects
    shutil.copyfile(rootdir.parent / 'OASIS.csv', niftidir / 'OASIS.csv')

    cprint("SUCCESS: Successfully converted all oasis files to nii!", "green")
    cprint("INFO: output files are in: " + str(niftidir), "yellow")
    