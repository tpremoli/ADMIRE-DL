import shutil
import numpy as np
import nibabel as nib
import fsl.wrappers.fsl_anat as fsl_anat
import fsl.wrappers.fslmaths as fsl_maths
import matplotlib as plt
from pathlib import Path
from PIL import Image

cwd = Path().resolve()
filedir = Path(__file__).parent.resolve()


def prep_raw_mri(scan_loc, scan_name, out_dir, group, sex, slice_range=(35, 55)):
    """Prepares a singular raw mri image. This means that FSL is ran, indiviudal 
    slices are extracted, and multichannel slices are extracted. 2 datasets are 
    created, one for multichannel images, and one for simple slices.

    Args:
        scan_loc (str): The scan location
        scan_name (str): The name of the scan (format NNN_S_NNNN_NN)
        out_dir (str): output directory, where images should be placed.
        group (str): Class of the image. Can be CN, AD, or MCI
        sex (str): sex of the sample. Can be M or F
        slice_range (tuple, optional): The slices to be extracted. Defaults to (35,55).

    Raises:
        ValueError: Raised if any values are missing for the scan.
    """
    if group is None or sex is None:
        raise ValueError(
            "ERROR: Scan instatiation requires group, sex, and age values!")

    scan_location = Path(cwd, scan_loc).resolve()

    print("Launching FSL scripts for scan {} in group {}".format(scan_name, group))
    nii_path = run_fsl(scan_location, scan_name, group, out_dir)

    print("FSL scripts complete. Processed MRI found in {}".format(nii_path))

    # To access slices:
    # sagittal = data[26, :, :] <- 26th slice along sagittal
    # coronal = data[:, 30, :] <- 30th slice along coronal
    # axial = data[:, :, 50] <- 50th slice along axial

    print("splitting MRI into individual slice images, slices {}-{}.".format(
        slice_range[0], slice_range[1]))
    create_slices_from_brain(nii_path, out_dir, scan_name, group, slice_range)

    print("splitting MRI into multichannal slice images, slices {}-{}.".format(
        slice_range[0], slice_range[1]))
    create_multichannel_slices_from_brain(
        nii_path, out_dir, scan_name, group, slice_range)

    # TODO: make sure saved into groups
    print("splitting images into train/test/val folders, with ratio X/X/X.")


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

    # File is saved into group subfolder in nii_files output loc
    final_brain = Path(
        out_dir, "nii_files/{}/{}_processed.nii.gz".format(group, scan_name))

    # We multiply the MNI registered brain by the brain mask to have a final preprocessed brain
    fsl_maths(mni_nonlin).mul(brain_mask).run(final_brain)

    # Copy log from anat dir to out dir
    # logfile = Path(anat_dir, "log.txt")
    # final_logfile = Path(out_dir, "{}.log".format(self.scan_name))
    # shutil.copyfile(logfile, final_logfile)

    # clearing all the .anat files (unnecessary now)
    shutil.rmtree(anat_dir)

    return final_brain


def create_slices_from_brain(nii_path, out_dir, scan_name, group, slice_range=(35, 55)):
    """Creates the individual image slices from an input nii image and slice range

    Args:
        nii_path (str): path of the nii image to extract the slices from
        out_dir (str): output directory. slices will be placed in {out_dir}/image_slices/{group}
        scan_name (str): The name of the scan (format NNN_S_NNNN_NN)
        group (str): Class of the image. Can be CN, AD, or MCI
        slice_range (tuple, optional): The slices to be extracted. Defaults to (35,55).
    """
    brain_data = get_data_from_nii(nii_path)

    for i in range(slice_range[0], slice_range[1]):
        # Vital to make sure that the np.float64 is correctly scaled to np.uint8
        curr_slice = normalize_array_range(brain_data[:, :, i])

        image_data = Image.fromarray(curr_slice)

        # Saved as image_slices/{group}/{subject}_slice{number}
        image_dir = Path(out_dir, "image_slices/{}/{}_slice{}.png".format(group,
                         scan_name, (i-slice_range[0]))).resolve()
        image_data.save(image_dir)


def create_multichannel_slices_from_brain(nii_path, out_dir, scan_name, group, slice_range=(35, 55)):
    """Creates the multichannel image slices from an input nii image and slice range

    Args:
        nii_path (str): path of the nii image to extract the slices from
        out_dir (str): output directory. slices will be placed in {out_dir}/multi_channel/{group}
        scan_name (str): The name of the scan (format NNN_S_NNNN_NN)
        group (str): Class of the image. Can be CN, AD, or MCI
        slice_range (tuple, optional): The slices to be extracted. Defaults to (35,55).
    """
    brain_data = get_data_from_nii(nii_path)

    for i in range(slice_range[0], slice_range[1]):
        # Vital to make sure that the np.float64 is correctly scaled to np.uint8
        # We do 3 slices (r=i-1,g=i,b=i+1)
        r_slice = normalize_array_range(brain_data[:, :, i-1])
        g_slice = normalize_array_range(brain_data[:, :, i])
        b_slice = normalize_array_range(brain_data[:, :, i+1])

        # We stack these into one nparray that will have shape (91,109,3)
        slice_3d = np.stack((r_slice, g_slice, b_slice), axis=2)

        image_data = Image.fromarray(slice_3d)

        # Saved as image_slices/{group}/{subject}_slice{number}
        image_dir = Path(out_dir, "multi_channel/{}/{}_slice{}.png".format(group,
                         scan_name, (i-slice_range[0]))).resolve()
        image_data.save(image_dir)


def normalize_array_range(img):
    """Normalizes range of np array values, moving them to the range 0-255. Important for RGB image gen

    Args:
        img (nparray): The 2D array to normalize

    Returns:
        nparray: the normalized 2D array. 
    """
    TARGET_TYPE_MIN = 0
    TARGET_TYPE_MAX = 255
    TARGET_TYPE = np.uint8

    imin = np.min(img)
    imax = np.max(img)

    a = (TARGET_TYPE_MAX - TARGET_TYPE_MIN) / (imax - imin)
    b = TARGET_TYPE_MAX - a * imax
    new_img = (a * img + b).astype(TARGET_TYPE)
    return new_img


def get_data_from_nii(nii_path):
    """Extracts nparray from given nii file

    Args:
        nii_path (str): The path of the nii file to get the data from

    Returns:
        nparray: The nii file's nparray
    """
    imgfile = nib.load(nii_path)
    return np.array(imgfile.dataobj)
