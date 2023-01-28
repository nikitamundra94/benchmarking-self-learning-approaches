import os
import numpy as np
import torch
import pytorch_lightning as pl
from data.cifar10_rotation import CIFAR10_rotation
from data.cifar10 import CIFAR10
from data.svhn_rotation import SVHN_rotation
from data.SVHN import SVHN
from model.alexnet import alexnet_base, MLP_alexnet_base
from model.alexnet_conv import alexnet_conv, MLP_alexnet_conv, MLP_alexnet_finetune
from model.renet_finetune import ReNet_lightning_finetune, MLP_renet_finetune
from model.concat_renet_alex import concat_net
from pytorch_lightning.callbacks import EarlyStopping


class Model_train(object):
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
            
            modelA = ReNet_lightning_finetune()
            modelB = alexnet_base()
        

            checkpointA = torch.load('/netscratch/mundra/model/RENET_CIFAR4.pt')
            modelA.load_state_dict(checkpointA['model_state_dict'])
            #new_model.eval()
            checkpointB = torch.load('/netscratch/mundra/model/ALEXNET_CIFAR4.pt')
            modelB.load_state_dict(checkpointB['model_state_dict'])
            model = concat_net(modelA, modelB)
            trainer = pl.Trainer.from_argparse_args(args,callbacks=[EarlyStopping(monitor='val_loss', patience=20)],gpus=1)    
            trainer.fit(model,dm)
            result1 = trainer.test()
            print("original",result1)
        else:
            dm = SVHN(
                batch_size=args.batch_size,
                data_dir='/netscratch/mundra/data/SVHN/origin_data/',
                num_workers=args.num_workers
            )
            modelA = ReNet_lightning_finetune()
            modelB = alexnet_base()


            checkpointA = torch.load('/netscratch/mundra/model/RENET_CIFAR4.pt')
            modelA.load_state_dict(checkpointA['model_state_dict'])
            #new_model.eval()
            checkpointB = torch.load('/netscratch/mundra/model/ALEXNET_CIFAR4.pt')
            modelB.load_state_dict(checkpointB['model_state_dict'])
            model = concat_net(modelA, modelB)
            trainer = pl.Trainer.from_argparse_args(args,callbacks=[EarlyStopping(monitor='val_loss', patience=20)],gpus=1)
            trainer.fit(model,dm)
            result1 = trainer.test()
            print("original",result1)


            #new_model.eval()
            



