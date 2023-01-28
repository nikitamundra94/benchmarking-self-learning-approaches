import os
import numpy as np
import torch
import pytorch_lightning as pl
from data.cifar10_jigsaw import CIFAR10_jigsaw
from data.svhn_jigsaw import svhn_jigsaw
from pytorch_lightning.callbacks import EarlyStopping
from model.alexnet_jigsaw import AlexNet_jigsaw
from model.renet_jigsaw import ReNet_jigsaw
class Model_train(object):
    def __init__(self, args):
        self.args = args
        self.flag =0

    def __call__(self):
        args = self.args
        if args.dataset == "CIFAR10":
            dm = CIFAR10_jigsaw(
               batch_size=args.batch_size,
               data_dir='/netscratch/mundra/data/CIFAR10/origin_data/',
               num_workers=args.num_workers,
               classes = args.classes
            )
            #trainer = pl.Trainer.from_argparse_args(args,callbacks=[EarlyStopping(monitor='val_loss', patience=20)] ,gpus=1)
            if args.model == "LeNet":
                new_model = LeNet()
            elif args.model == "ReNet":
                new_model = ReNet_jigsaw()
            elif args.model == "AlexNet":
                new_model = AlexNet_jigsaw()
            else:
                new_model = nin_base()
            trainer = pl.Trainer.from_argparse_args(args,callbacks=[EarlyStopping(monitor='val_loss', patience=5)] ,gpus=1)    
            trainer.fit(new_model,dm)
            result1 = trainer.test()
            torch.save({'model_state_dict':new_model.state_dict()},
                                         '/netscratch/mundra/model/ReNetCIFAR10_jigsaw500_0.01.pt')
            print("original",result1)
        else:
            print("svhn")
            dm = svhn_jigsaw(
                batch_size=args.batch_size,
                data_dir='/netscratch/mundra/data/SVHN/origin_data/',
                num_workers=args.num_workers,
                classes = args.classes
            )

            if args.model == "LeNet":
                new_model = LeNet()
            elif args.model == "ReNet":
                new_model = ReNet_jigsaw()
            elif args.model == "AlexNet":
                new_model = AlexNet_jigsaw()
            else:
                new_model = nin_base()
            trainer = pl.Trainer.from_argparse_args(args, callbacks=[EarlyStopping(monitor='val_loss', patience=20)],gpus=1)
            trainer.fit(new_model, dm)
            result1 = trainer.test()
            torch.save({'model_state_dict':new_model.state_dict()}, '/netscratch/mundra/model/ReNetSVHN_jigsaw1000_0.01.pt')
            print("original", result1)



