import argparse
from src.settings import *
from termcolor import colored

def main():
    parser = argparse.ArgumentParser(
        description='This is the CLI tool to use the {APP_NAME} tool')

    # Subparsers for prep, train, test, and predict
    subparsers = parser.add_subparsers(
        help='Prep tool to run the data', dest="tool")

    # prep parser
    parser_prep = subparsers.add_parser("prep")
    parser_prep.add_argument(
        '-r', '--run_name', required=True, type=str, help='The name of the run. Files will be saved in `out/preprocessed_datasets/{run_name}`')
    parser_prep.add_argument(
        '--ratio', nargs='+', type=float, help='The ratio of train, test, validation data. input as --ratio {train} {test} {validation}', default=[0.8, 0.1, 0.1])

    parser_prep.add_argument(
        '-d', '--collection_dir', type=str, help='The directory of the collection. If the collection was downloaded from ADNI, this should be the "ADNI" folder')
    parser_prep.add_argument(
        '-c', '--collection_csv', type=str, help='The directory of the collection\'s csv (Metadata) file. This allows the program to identify which research group the collection belongs to')
    parser_prep.add_argument(
        '-k', '--kaggle', type=str, help='The directory of the kaggle mri images. These should be in their original folder structure (i.e nondemented, mildlydemented etc.)')

    # train parser
    parser_prep = subparsers.add_parser("train")
    parser_prep.add_argument(
        '-c', '--config', required=True, type=str, help='The training task configuration file. This defines everything necessary in the task.')

    args = parser.parse_args()

    if args.tool == "prep":
        from src.preprocessing.launch_scripts import prep_adni, prep_kaggle

        # One of the options must be used
        if not args.kaggle and not args.collection_dir and not args.collection_csv:
            raise argparse.ArgumentTypeError(
                colored('Must specify COLLECTION_DIR and COLLECTION_CSV or KAGGLE to prep datasets!', "red"))

        # Ratio option errors
        if len(args.ratio) != 3:
            raise argparse.ArgumentTypeError(
                colored(f'Train/test/validation RATIO takes in exactly 3 values ({len(args.ratio)} given)!', "red"))
        if sum(args.ratio) != 1:
            raise argparse.ArgumentTypeError(
                colored('Train/test/validation RATIO must add up to 1!', "red"))

        print("Settings:")
        print("\tUSE_S3",str(USE_S3))
        print("\tAWS_S3_BUCKET_NAME",str(AWS_S3_BUCKET_NAME))
        print("\tSKIP_FSL",str(SKIP_FSL))
        print("\tSKIP_SLICE_CREATION",str(SKIP_SLICE_CREATION))

        if args.kaggle:
            # -k args are mutually exclusive with -d and -c
            if args.collection_dir or args.collection_csv:
                raise argparse.ArgumentTypeError(
                    colored('KAGGLE arg and COLLECTION_DIR/COLLECTION_CSV args are mutually exclusive! The dataset to be prepped is either kaggle or ADNI', "red"))

            print(f"Option chosen: prep kaggle dataset {args.kaggle}")
            prep_kaggle(args.kaggle, args.run_name, tuple(args.ratio))
        else:
            print(f"Option chosen: prep ADNI dataset {args.collection_dir}")

            if SKIP_FSL:
                print("SKIP_FSL option chosen! Will generate slices from passed collection_dir and split into train/test/val")
                # d is required if we're skipping FSL
                if not args.collection_dir:
                    raise argparse.ArgumentTypeError(
                        colored('Must specify COLLECTION_DIR when using ADNI dataset with SKIP_FSL enabled!', "red"))
                
                prep_adni(collection_dir=args.collection_dir,
                          run_name=args.run_name,
                          split_ratio=tuple(args.ratio))
                
            else:
                # both c and d are required if we're not skipping FSL scripts
                if not args.collection_csv or not args.collection_dir:
                    raise argparse.ArgumentTypeError(
                        colored('Must specify COLLECTION_DIR and COLLECTION_CSV when using ADNI dataset with SKIP_FSL disabled!', "red"))

                prep_adni(collection_dir=args.collection_dir,
                          run_name=args.run_name,
                          split_ratio=tuple(args.ratio),
                          collection_csv=args.collection_csv)

    elif args.tool == "train":
        from src.training.run_training import load_training_task

        if not args.config:
            raise argparse.ArgumentTypeError(
                colored("Missing config file! Training tasks require -c config.yml option", "red"))

        load_training_task(args.config)
    else:
        raise argparse.ArgumentTypeError(
            colored('Must specify CLI mode! Options: (prep, train, test, predict)', "red"))


if __name__ == "__main__":
    # from src.evaluating.eval_model import main
    main()
