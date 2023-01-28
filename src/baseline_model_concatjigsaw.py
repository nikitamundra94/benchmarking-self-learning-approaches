import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from data.cifar10_resize_jigsaw import CIFAR10_resize
from data.svhn_resize import svhn_resize
from pytorch_lightning.callbacks import EarlyStopping
from model.alexnet_jigsaw import AlexNet_jigsaw
from model.jigsawconcat_renet_alex import jigsawconcat_net
from model.renet_jigsaw import ReNet_jigsaw


class Model_train(object):
    def __init__(self, args):
        self.args = args
        self.flag =0

    def __call__(self):
        args = self.args
        if args.dataset == "CIFAR10":
            dm = CIFAR10_resize(
               batch_size=args.batch_size,
               data_dir='/netscratch/mundra/data/CIFAR10/origin_data/',
               num_workers=args.num_workers
            )
            #trainer = pl.Trainer.from_argparse_args(args,callbacks=[EarlyStopping(monitor='val_loss', patience=20)] ,gpus=1)
            
            modelA = ReNet_jigsaw()
            modelB = AlexNet_jigsaw()
        

            checkpointA = torch.load('/netscratch/mundra/model/ReNetCIFAR10_jigsaw1000_0.01.pt')
            modelA.load_state_dict(checkpointA['model_state_dict'])
            #new_model.eval()
            checkpointB = torch.load('/netscratch/mundra/model/AlexNet_jigsaw1000_0.01.pt')
            modelB.load_state_dict(checkpointB['model_state_dict'])
            model = jigsawconcat_net(modelA, modelB)
            tb_logger = pl_loggers.TensorBoardLogger('/netscratch/mundra/logs/')

            trainer = pl.Trainer.from_argparse_args(args,logger=tb_logger,callbacks=[EarlyStopping(monitor='val_loss', patience=20)],gpus=1)    
            trainer.fit(model,dm)
            result1 = trainer.test()
            print("original",result1)
        else:
            dm = svhn_resize(
                batch_size=args.batch_size,
                data_dir='/netscratch/mundra/data/SVHN/origin_data/',
                num_workers=args.num_workers
            )
            modelA = ReNet_jigsaw()
            modelB = AlexNet_jigsaw()
            tb_logger = pl_loggers.TensorBoardLogger('/netscratch/mundra/logs/')
            checkpointA = torch.load('/netscratch/mundra/model/ReNetSVHN_jigsaw1000_0.01.pt')
            modelA.load_state_dict(checkpointA['model_state_dict'])
            #new_model.eval()
            checkpointB = torch.load('/netscratch/mundra/model/AlexNetSVHN_jigsaw1000_0.01.pt')
            modelB.load_state_dict(checkpointB['model_state_dict'])
            model = jigsawconcat_net(modelA, modelB)
            trainer = pl.Trainer.from_argparse_args(args,logger=tb_logger,callbacks=[EarlyStopping(monitor='val_loss', patience=20)],gpus=1)
            trainer.fit(model,dm)
            result1 = trainer.test()
            print("original",result1)


            #new_model.eval()
            



