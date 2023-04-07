# Run this with the ADNI dataset saved in 
#       supplemental_files/unprocessed_datasets/ADNI
# and with the csv file saved in
#       supplemental_files/unprocessed_datasets/test_sample.csv

python cli.py prep -d supplemental_files/unprocessed_datasets/ADNI \
    -c supplemental_files/unprocessed_datasets/test_sample.csv \
    -r adni_processed