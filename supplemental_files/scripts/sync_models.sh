# This script syncs all the files stored in the out directory to a specified s3 bucket.
# This is different from the nii bucket.

# To use this script, you must have the aws cli installed and configured.
# See https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html

# Set the bucket name here
bucket_name="diss-backup"

# No need to modify anything below this line

aws s3 sync out/trained_models s3://$bucket_name/trained_models

