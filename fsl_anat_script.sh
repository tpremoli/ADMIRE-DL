# The following is the fsl_anat auto pipeline
# Reference (outputs and use):
# https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/fsl_anat

# View using FSLeyes
# https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes

# Watch out for dashes!
fsl_anat -i "./002_S_0816/ADNI_002_S_0816_MR_MP_RAGE__br_raw_20060830195514279_1_S18400_I23591.nii"

# MNI152_T1_2mm_brain_mask_dil1 < brain mask output
# T1_to_MNI_nonlin < nonlin registered brain

# to apply brain mask
fslmaths "./002_S_0816/ADNI_002_S_0816_MR_MP_RAGE__br_raw_20060830195514279_1_S18400_I23591.anat/T1_to_MNI_nonlin.nii.gz" \
    -mul "./002_S_0816/ADNI_002_S_0816_MR_MP_RAGE__br_raw_20060830195514279_1_S18400_I23591.anat/MNI152_T1_2mm_brain_mask_dil1.nii.gz" \
    "new_brain.nii.gz"
