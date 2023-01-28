from argparse import ArgumentParser
from pytorch_lightning import Trainer
from baseline_model_jigsaw import Model_train



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--testing", action="store_false")
    parser.add_argument("--model", default='ReNet')
    parser.add_argument("--max_epochs", default = 70, type=int)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", default='SVHN')
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--classes", default = 500, type=int)

    #parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    model_training = Model_train(args)
    model_training()
