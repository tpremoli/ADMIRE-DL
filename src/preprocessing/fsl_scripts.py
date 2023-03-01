import shutil
from ..constants import *
from ..settings import *
from pathlib import Path
from fsl.wrappers import fsl_anat, fslmaths

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()

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
        filedir, "../../out/preprocessed_datasets/tmp", scan_name).resolve()

    # fsl_anat adds .anat to end of output directory
    anat_dir = Path(f"{tmp_dir}.anat")

    try:
        # Running fsl_anat (we don't need tissue segmentation nor subcortical segmentation)
        fsl_anat(img=scan_location, out=tmp_dir, noseg=True, nosubcortseg=True)
    except:
        # If fsl fails, we delete the tmp_dir and return dummy values
        shutil.rmtree(anat_dir)
        raise ValueError(f"ERROR: FSL failed to run on scan {scan_name} in group {group}. \n Original brain: {scan_location} \n tmp_dir: {tmp_dir}")

    # This is the outputted nonlinear transformed brain
    mni_nonlin = Path(anat_dir, "T1_to_MNI_nonlin.nii.gz")

    # This is the outputted brain mask
    brain_mask = Path(anat_dir, "MNI152_T1_2mm_brain_mask_dil1.nii.gz")

    # File is saved into group subfolder in nii_files output loc
    final_brain = Path(
        out_dir, f"nii_files/{group}/{scan_name}_processed.nii.gz")

    # We multiply the MNI registered brain by the brain mask to have a final preprocessed brain
    fslmaths(mni_nonlin).mul(brain_mask).run(final_brain)

    # clearing all the .anat files (unnecessary now)
    shutil.rmtree(anat_dir)

    return (scan_location, final_brain)