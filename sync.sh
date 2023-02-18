# This script syncs all the files stored in the out directory to a specified s3 bucket.
# This is different from the nii bucket.
# TODO: make this part of the cli tool

# The name of the bucket to sync to
$BACKUP_BUCKET="diss-backup"

aws s3 sync out s3://$BACKUP_BUCKET --recursive
