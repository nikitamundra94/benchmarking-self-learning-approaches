import os
import numpy as np
import torch
import pytorch_lightning as pl
from data.cifar10_rotation import CIFAR10_rotation
from data.cifar10 import CIFAR10
from data.svhn_rotation import SVHN_rotation
from data.SVHN import SVHN
#from model.lenet import LeNet
from model.renet import ReNet_lightning, MLP_renet
from model.alexn import AlexNet, MLP



class Model_training(object):
    def __init__(self, args):
        self.args = args
        self.flag =0

    def __call__(self):
        args = self.args
        if args.dataset == "CIFAR10":
            dm_rotated = CIFAR10_rotation(
                batch_size=args.batch_size,
                data_dir='/netscratch/mundra/data/CIFAR10/origin_data/',
                num_workers=args.num_workers,
                rot = args.rotation
            )
            trainer = pl.Trainer.from_argparse_args(args,gpus=1)
            if args.model == "LeNet":
                model = LeNet()
            elif args.model == "ReNet":
                model = ReNet_lightning()
            else:
                model = AlexNet()
            trainer.fit(model, dm_rotated)
            torch.save({'model_state_dict':model.state_dict()},
                     'model_Renet.pt')
            if args.testing:
                result = trainer.test()
                print(result)
            dm = CIFAR10(
               batch_size=args.batch_size,
               data_dir='/netscratch/mundra/data/CIFAR10/origin_data/',
               num_workers=args.num_workers
            )
            trainer = pl.Trainer.from_argparse_args(args,gpus=1)
            checkpoint = torch.load("model_Renet.pt")
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if args.model == "LeNet":
                new_model = model
            elif args.model == "ReNet":
                new_model = MLP_renet(model)
            else:
                new_model = MLP(model)
            trainer.fit(new_model,dm)
            result1 = trainer.test()
            print("original",result1)
        else:
            dm_rotated = SVHN_rotation(
                batch_size=args.batch_size,
                data_dir='/netscratch/mundra/data/SVHN/origin_data/',
                num_workers=args.num_workers,
                rot=args.rotation
            )
            trainer = pl.Trainer.from_argparse_args(args,gpus=1 )
            if args.model=="LeNet":
                model = LeNet()
            elif args.model == "ReNet":
                model = ReNet_lightning()
            else:
                model = ALexNet()
            trainer.fit(model, dm_rotated)
            torch.save({'model_state_dict': model.state_dict()},
                       'model_SVHN.pt')
            if args.testing:
                result = trainer.test()
                print(result)
            dm = SVHN(
                batch_size=args.batch_size,
                data_dir='/netscratch/mundra/data/SVHN/origin_data/',
                num_workers=args.num_workers
            )
            trainer = pl.Trainer.from_argparse_args(args,gpus=1 )
            checkpoint = torch.load("model_SVHN.pt")
            model.load_state_dict(checkpoint['model_state_dict'])

            if args.model == "LeNet":
                new_model = model
            elif args.model == "ReNet":
                new_model = MLP(model)
            else:
                new_model = MLP(model)
            trainer.fit(new_model, dm)
            result1 = trainer.test()
            print("original", result1)



