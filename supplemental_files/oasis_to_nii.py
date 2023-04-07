from pathlib import Path
import os
import subprocess

# TODO: take args
# Directory where the OASIS data is located
rootdir = Path('/home/tpremoli/MRI_AD_Diagnosis/supplemental_files/unprocessed_datasets/OASIS')

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

print("Done!")
