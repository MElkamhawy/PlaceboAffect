from features import preprocess
from argparse import ArgumentParser


TARGET_LABEL_COL = "HS"
TEXT_COL = "text"
TRAIN_MODE = "train"
TEST_MODE = "test"
TRAIN_DATASET_NAME = "train"
DEV_DATASET_NAME = "dev"
CLASSIFICATION_ALGORITHM = "SVM"


def create_arg_parser():
    argument_parser = ArgumentParser(
        description="D2 for PlaceboAffect - Course Ling 573."
    )
    argument_parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Train or test the model",
    )
    argument_parser.add_argument(
        "--baseline", help="Baseline Model", action="store_true", default=False
    )
    argument_parser.add_argument(
        "--train-data",
        help="Training Data File Path",
        default="../data/train/en/hateval2019_en_train.csv",
    )
    argument_parser.add_argument(
        "--test-data",
        help="Testing Data File Path",
        default="../data/dev/en/hateval2019_en_dev.csv",
    )
    argument_parser.add_argument(
        "--model", help="Model File Path", default="../models/D2/svm_en_{sys}.pkl"
    )
    argument_parser.add_argument(
        "--empath", help="Empath Feature", action="store_true", default=False
    )
    argument_parser.add_argument(
        "--result", help="Output File Path", default="../results/res_en_svm_{sys}.txt"
    )
    argument_parser.add_argument(
        "--predictions",
        help="Predictions File Path",
        default="../outputs/D2/pred_en_svm_{sys}.txt",
    )
    return argument_parser


def run(
    mode,
    baseline,
    training_data_file,
    test_data_file,
    result_file,
    predictions_file,
    model_file,
    empath,
):
    data_train = preprocess.Data.from_csv(training_data_file, name=TRAIN_DATASET_NAME)
    data_dev = preprocess.Data.from_csv(test_data_file, name=DEV_DATASET_NAME)

    # Preprocess Data
    data_train.process(text_name=TEXT_COL, target_name=TARGET_LABEL_COL)
    data_dev.process(text_name=TEXT_COL, target_name=TARGET_LABEL_COL)

    for t in data_dev.text:
        print(t)


def main():
    args = create_arg_parser().parse_args()
    run(
        args.mode,
        args.baseline,
        args.train_data,
        args.test_data,
        args.result,
        args.predictions,
        args.model,
        args.empath,
    )


if __name__ == "__main__":
    main()
