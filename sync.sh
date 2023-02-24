# This script syncs all the files stored in the out directory to a specified s3 bucket.
# This is different from the nii bucket.
# TODO: make this part of the cli tool

aws s3 sync out s3://diss-backup
