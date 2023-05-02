import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path

def write_batch_to_log(complete_pairs, out_dir, successful_str):
    """Writes a batch of successful scans to a log file.
    
    Args:
        complete_pairs (list): A list of tuples containing the original brain and the new brain files.
        out_dir (str): The output directory.
        successful_str (str): The "success string" to be written to the log file.
    """
    logdir = Path(out_dir, "batches.log").resolve()
    csv_dir = Path(out_dir, "processed.csv").resolve()
    
    with open(logdir, "a") as log:
        log.write(successful_str)
        log.write("\n")
        for ogbrain, newbrain in complete_pairs:
            if newbrain == "FSL failed":
                log.write("\tFAIL:")
                log.write(Path(ogbrain).resolve().name)
                log.write(" failed to process. Skipped!\n")
            else:
                log.write("\t")
                log.write(Path(ogbrain).resolve().name)
                log.write(" processed to ")
                log.write(Path(newbrain).resolve().name)
                log.write(", ")
                log.write(Path(newbrain).resolve().parent.name)
                log.write("\n")
    
    with open(csv_dir, "a") as csv:
        for ogbrain, newbrain in complete_pairs:
            if newbrain == "FSL failed":
                csv.write('"')
                csv.write(str(Path(ogbrain).resolve()))
                csv.write('"')
                csv.write(",")
                
                csv.write('"')
                csv.write("FAILED")
                csv.write('"')
                csv.write(",")
                
                csv.write('"')
                csv.write("NA")
                csv.write('"')
                
                csv.write("\n")
            else:
                csv.write('"')
                csv.write(str(Path(ogbrain).resolve()))
                csv.write('"')
                csv.write(",")
                
                csv.write('"')
                csv.write(str(Path(newbrain).resolve()))
                csv.write('"')
                csv.write(",")
                
                csv.write('"')
                csv.write(str(Path(newbrain).resolve().parent.name))
                csv.write('"')

                csv.write("\n")
                
def create_image_slices_from_brain(nii_path, out_dir, scan_name, group, slice_range=(80, 110), view="axial"): # TODO: add option for saggital, coronal, and axial slices
    """Creates the axial image slices from an input nii image and slice range

    Args:
        nii_path (str): path of the nii image to extract the slices from
        out_dir (str): output directory. slices will be placed in {out_dir}/axial_slices/{group}
        scan_name (str): The name of the scan (format NNN_S_NNNN_NN)
        group (str): Class of the image. Can be CN, AD, or MCI
        slice_range (tuple, optional): The slices to be extracted. Defaults to (80, 110).
        
    """
    brain_data = get_data_from_nii(nii_path)
    
    if view == "axial":
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
    elif view == "saggital" or view == "coronal":
        raise NotImplementedError("Saggital and coronal views are not yet implemented")
    else:
        raise ValueError("Invalid view type. Must be axial, saggital, or coronal.")

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