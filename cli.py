import argparse
from src.preprocessing.run_scripts import prep_adni, prep_kaggle


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
        # -k args are mutually exclusive with -d and -c
        if args.kaggle and args.collection_dir or args.kaggle and args.collection_csv:
            raise argparse.ArgumentTypeError(
                'KAGGLE arg and COLLECTION_DIR/COLLECTION_CSV args are mutually exclusive! The dataset to be prepped is either kaggle or ADNI')
        # both c and d are required if either of them are used
        if args.collection_dir and not args.collection_csv or args.collection_csv and not args.collection_dir:
            raise argparse.ArgumentTypeError(
                'Must specify COLLECTION_DIR and COLLECTION_CSV when using ADNI dataset!')
        # One of the options must be used
        if not args.kaggle and not args.collection_dir and not args.collection_csv:
            raise argparse.ArgumentTypeError(
                'Must specify COLLECTION_DIR and COLLECTION_CSV or KAGGLE to prep datasets!')
        
        # Ratio option errors
        if len(args.ratio) != 3:
            raise argparse.ArgumentTypeError(
                'Train/test/validation RATIO takes in exactly 3 values ({} given)!'.format(len(args.ratio)))
        if sum(args.ratio) != 1:
            raise argparse.ArgumentTypeError(
                'Train/test/validation RATIO must add up to 1!')

        if args.kaggle:
            print("Option chosen: prep kaggle dataset {}".format(args.kaggle))
            prep_kaggle(args.kaggle, args.run_name, tuple(args.ratio))
        else:
            print("Option chosen: prep ADNI dataset {}".format(args.collection_dir))
            prep_adni(args.collection_dir, args.collection_csv, args.run_name)
    elif args.tool == "train":
        from src.training.experiments import run
        run()
    else:
        raise argparse.ArgumentTypeError(
            'Must specify CLI mode! Options: (prep, train, test, predict)')


if __name__ == "__main__":
    # main()
    from src.training.experiments import run
    run()
