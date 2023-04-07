"""This script converts the OASIS dataset to nifti format.
The default OASIS format is Analyze 7.5, which is outdated,
missing some metadata, and not supported by many tools.
This script converts the dataset to nifti format, which is
much more compatible with modern standards.

This script should be run from the root directory of the
program, and should be ran before running prep_oasis.py.
This will create a new directory called oasis_nifti, which
will be stored in supplemental_files/unprocessed_datasets.

OASIS dataset is available at: https://www.oasis-brains.org/

To use:
    python supplemental_files/scripts/oasis_to_nii.py <oasisdir>

where <oasisdir> is the root directory of the OASIS dataset.
This should also contain the oasis_cross-sectional.csv file,
which contains the labels for the subjects.

Raises:
    ValueError: Thrown if the script is not run with the correct number of arguments.
"""

from pathlib import Path
from termcolor import cprint
import shutil
import os
import subprocess
import sys

cwd = Path.cwd()

if __name__ == '__main__':
    # get rootdir from args
    if len(sys.argv) != 2:
        raise ValueError("ERROR: Need one argument - oasisdir of OASIS dataset")
    
    oasisdir = Path(cwd, sys.argv[1]).resolve()
    
    # Directory to save results
    niftidir = oasisdir.parent / 'oasis_nifti'

    # Create directory to save the results, if not existing
    (niftidir).mkdir(parents=True, exist_ok=True)

    # For each subject
    for s in sorted(oasisdir.iterdir()):
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

            output_filename = a.stem
            if output_filename.split("_")[2] == "MR2":
                output_filename = output_filename.replace("MR2", "MR1")
                cprint("WARNING: MR2 acquisition found, renaming to MR1 instead", "yellow")

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
    shutil.copyfile(oasisdir / 'oasis_cross-sectional.csv', niftidir / 'OASIS.csv')

    cprint("SUCCESS: Successfully converted all oasis files to nii!", "green")
    cprint("INFO: output files are in: " + str(niftidir), "yellow")
    