import os
import numpy as np
import torch
import pytorch_lightning as pl
from data.cifar10_rotation import CIFAR10_rotation
from data.cifar10 import CIFAR10
from data.svhn_rotation import SVHN_rotation
from data.SVHN import SVHN
from model.lenet import LeNet
from model.renet import ReNet_lightning
from model.renet_4layer import ReNet_lightning_layer
from model.alexnet_conv import alexnet_conv
from model.alexnet import alexnet_base, MLP_alexnet_base
from model.nin import nin_base, MLP_nin_base
from pytorch_lightning.callbacks import EarlyStopping


class Model_base_train(object):
    def __init__(self, args):
        self.args = args
        self.flag =0

    def __call__(self):
        args = self.args
        if args.dataset == "CIFAR10":
            dm = CIFAR10(
               batch_size=args.batch_size,
               data_dir='/netscratch/mundra/data/CIFAR10/origin_data/',
               num_workers=args.num_workers
            )
            #trainer = pl.Trainer.from_argparse_args(args,callbacks=[EarlyStopping(monitor='val_loss', patience=20)] ,gpus=1)
            if args.model == "LeNet":
                new_model = LeNet()
            elif args.model == "ReNet":
                new_model = ReNet_lightning_layer()
            elif args.model == "AlexNet":
                new_model = alexnet_base()
            else:
                new_model = nin_base()
            #checkpoint = torch.load('model_AlexNet.pt')
            #new_model.load_state_dict(checkpoint['model_state_dict'])
            #model = MLP(new_model)
            trainer = pl.Trainer.from_argparse_args(args,callbacks=[EarlyStopping(monitor='val_loss', patience=20)],gpus=1)    
            trainer.fit(new_model,dm)
            result1 = trainer.test()
            print("original",result1)
        else:
            dm = SVHN(
                batch_size=args.batch_size,
                data_dir='/netscratch/mundra/data/SVHN/origin_data/',
                num_workers=args.num_workers
            )
            print("SVHN")
            trainer = pl.Trainer.from_argparse_args(args,gpus=1 )

            if args.model == "LeNet":
                new_model = LeNet()
            elif args.model == "ReNet":
                new_model = ReNet_lightning_layer()
            elif args.model == "AlexNet":
                new_model = alexnet_base()
            else:
                new_model = nin_base()
            trainer = pl.Trainer.from_argparse_args(args,callbacks=[EarlyStopping(monitor='val_loss', patience=20)],gpus=1)
            trainer.fit(new_model, dm)
            result1 = trainer.test()
            print("original", result1)



