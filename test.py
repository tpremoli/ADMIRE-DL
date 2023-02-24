import shutil
import pandas as pd
import pathlib
import splitfolders
import nibabel as nib
from PIL import Image
from pathlib import Path
import numpy as np

df = pd.read_csv("/home/tpremoli/MRI_AD_Diagnosis/unprocessed_datasets/ADNI1_Screening_1.5T_2_24_2023.csv")
# Drop duplicate subjects
df.drop_duplicates(subset=['Subject'], inplace=True)
print(len(df))
# drop subjects in MCI Group
df.drop(df[df['Group'] == 'MCI'].index, inplace=True)
print(len(df))
print(len(df[df['Group'] == 'AD']))
print(len(df[df['Group'] == 'CN']))

if False:
    adpath = "/home/tpremoli/MRI_AD_Diagnosis/out/preprocessed_datasets/adni_processed/nii_files/AD"
    cnpath = "/home/tpremoli/MRI_AD_Diagnosis/out/preprocessed_datasets/adni_processed/nii_files/CN"


    for f in pathlib.Path(adpath).glob("*.nii.gz"):
        if f.name.split("_")[3] == "00":
            shutil.copy(f, "/home/tpremoli/MRI_AD_Diagnosis/out/preprocessed_datasets/unique/nii_files/AD")

    for f in pathlib.Path(cnpath).glob("*.nii.gz"):
        if f.name.split("_")[3] == "00":
            shutil.copy(f, "/home/tpremoli/MRI_AD_Diagnosis/out/preprocessed_datasets/unique/nii_files/CN")

    splitfolders.ratio("/home/tpremoli/MRI_AD_Diagnosis/out/preprocessed_datasets/unique/nii_files", output="/home/tpremoli/MRI_AD_Diagnosis/out/preprocessed_datasets/unique/nii_split", ratio=[0.8,0.1,0.1])

    d = {'subj_name': [], 'subj_group': [], 'subj_dir':[]}
    df = pd.DataFrame(data=d)

    for f in pathlib.Path("/home/tpremoli/MRI_AD_Diagnosis/out/preprocessed_datasets/unique/nii_split").rglob("*.nii.gz"):
        subj_name = f.name[:10]
        subj_group = f.parent.name
        subj_dir = f.parent
        df = df.append({'subj_name': subj_name, 'subj_group': subj_group, 'subj_dir': subj_dir}, ignore_index=True)
        
    for f in pathlib.Path("/home/tpremoli/MRI_AD_Diagnosis/out/preprocessed_datasets/adni_processed/nii_files").rglob("*.nii.gz"):
        subj_name = f.name[:10]
        new_dir = df.loc[df['subj_name'] == subj_name, 'subj_dir'].values[0]
        shutil.copy(f, new_dir)


    
def create_slices_from_brain(f, final_path, scan_name, slice_range=(35, 55)):
    brain_data = get_data_from_nii(f)

    for i in range(slice_range[0], slice_range[1]):
        # Vital to make sure that the np.float64 is correctly scaled to np.uint8
        curr_slice = normalize_array_range(brain_data[:, :, i])

        image_data = Image.fromarray(curr_slice)

        # Saved as image_slices/{group}/{subject}_slice{number}
        image_dir = Path(final_path, f"{scan_name}_slice{(i-slice_range[0])}.png").resolve()
        
        image_data.save(image_dir)
        
def create_multichannel_slices_from_brain(f, final_path, scan_name, slice_range=(35, 55)):
    brain_data = get_data_from_nii(f)

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
        image_dir = Path(final_path, f"{scan_name}_slice{(i-slice_range[0])}.png").resolve()
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

    slice_folder = "/home/tpremoli/MRI_AD_Diagnosis/out/preprocessed_datasets/unique/final_slices"
    multichannel_folder ="/home/tpremoli/MRI_AD_Diagnosis/out/preprocessed_datasets/unique/multichannel_slices"

    for f in pathlib.Path("/home/tpremoli/MRI_AD_Diagnosis/out/preprocessed_datasets/unique/nii_split").rglob("*.nii.gz"):
        final_folder = f.parent.parent.name
        group = f.parent.name
        name = f.name
        final_slice_path = Path(slice_folder, final_folder, group).resolve()
        final_multichannel_path = Path(multichannel_folder, final_folder, group).resolve()
        create_slices_from_brain(f, final_slice_path, f.name[:13])
        create_multichannel_slices_from_brain(f, final_multichannel_path, f.name[:13])