import argparse
from admire.settings import *
from termcolor import colored

def main():
    parser = argparse.ArgumentParser(
        description='This is the CLI tool to use the {APP_NAME} tool')

    # Subparsers for prep, train, test, and predict
    subparsers = parser.add_subparsers(
        help='What tool to use. prep=preprocessing tool, train=training tool, eval=evaluation tool', dest="tool")

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

    # train parser
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        '-c', '--config', required=True, type=str, help='The training task configuration file. This defines everything necessary in the task.')

    # eval parser
    eval_parser = subparsers.add_parser("eval")

    args = parser.parse_args()

    if args.tool == "prep":
        from admire.preprocessing.launch_scripts import prep_adni

        # One of the options must be used
        if not args.collection_dir and not args.collection_csv:
            raise argparse.ArgumentTypeError(
                colored('Must specify COLLECTION_DIR and COLLECTION_CSV to prep datasets!', "red"))

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

        print(f"Option chosen: prep dataset {args.collection_dir}")

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
        from admire.training.run_training import load_training_task

        if not args.config:
            raise argparse.ArgumentTypeError(
                colored("Missing config file! Training tasks require -c config.yml option", "red"))

        load_training_task(args.config)
    elif args.tool == "eval":
        from admire.evaluating.eval_model import eval_all_models
        
        eval_all_models()
    else:
        raise argparse.ArgumentTypeError(
            colored('Must specify CLI mode! Options: (prep, train, test, predict)', "red"))
        


if __name__ == "__main__":
    main()
