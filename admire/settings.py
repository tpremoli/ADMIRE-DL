
# Preprocessing script options
# This handles the s3 bucket upload options.
USE_S3=True
AWS_S3_BUCKET_NAME="processed-nii-files"

# This controls the number of concurrent FSL processes that can be ran at once.
FSL_CONCURRENT_PROCESSES=6

# This setting controls if FSL scripts should be ran when processing the dataset.
# If not, this assumes the FSL scripts have been ran, and the prep script has been
# ran for a folder containing processed nii.gz files, labeled and separated into
# the correct folders.
SKIP_FSL=True

# Controls if we're using nonlinear registration. This is a very expensive process,
# and is not recommended unless you have a lot of time and resources.
USE_NONLINEAR_REGISTRATION=False

# This setting controls if the program should generate image slices for the dataset
# being processed.
SKIP_SLICE_CREATION=False

# This setting controls if the program should split the dataset into train, test, and
# val folders.
SKIP_FOLDER_SPLIT=False

# This setting enables OASIS dataset model testing & evaluation. Use scripts found in
# scripts/oasis_to_nii.py and scripts/prep_oasis.py to prepare the dataset.
# The final image slices should be stored in out/preprocessed_datasets/oasis_processed/axial_slices
# NOTE: this was not used in the final report, and is not recommended due to discrepancies between
# the datasets.
EVAL_OASIS=False

# This controls if we should draw the confusion matrix for the model evaluation.
DRAW_CONFUSION_MATRIX=False