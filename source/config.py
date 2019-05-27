import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-dp", "--data_path",
                    type=str,
                    help="input the data path",
                    default=r"../data/train/train_seg/",
                    )
parser.add_argument("-tdp", "--test_data_path",
                    type=str,
                    help="input the test data path",
                    default=r"../data/test/"
                    )
parser.add_argument("-lp", "--log_path",
                    type=str,
                    help="input the logdir path",
                    default=r"../log/")
parser.add_argument("-op", "--output_path",
                    type=str,
                    help="input the output path",
                    default=r"/data/output/")
parser.add_argument("-lr", "--learning_rate",
                    type=float,
                    help="input the learning_rate",
                    default=0.0003)
parser.add_argument("-bs", "--batch_size",
                    type=int,
                    help="input the batch_size",
                    default=128)
parser.add_argument("-opt", "--optimizer",
                    type=str,
                    help="choose the optimizer from Adam(adam), Momentum(mome), SGD(sgd)",
                    default="adam")
parser.add_argument("-ts", "--training_steps",
                    type=int,
                    help="input the training_steps",
                    default=3000)
parser.add_argument("-l2r", "--L2_regular_rate",
                    type=float,
                    help="input the l2 regular rate",
                    default=0.005)
parser.add_argument("-lrd", "--learning_rate_decay",
                    type=float,
                    help="input the learning rate decay",
                    default=0.99)
parser.add_argument("-mp", "--model_path",
                    type=str,
                    help="input the model_path",
                    default="../saved_model/large_bs_sigmoid4_xavier/")
parser.add_argument("-vp", "--validation_path",
                    type=str,
                    help="input the validation path",
                    default="../data/val/")
args = parser.parse_args()

parameters = vars(args)

