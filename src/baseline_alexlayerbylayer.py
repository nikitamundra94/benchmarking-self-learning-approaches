import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from data.cifar10_rotation import CIFAR10_rotation
#from data.cifar10 import CIFAR10
from data.svhn_rotation import SVHN_rotation
#from data.SVHN import SVHN
#from model.lenet import LeNet
from model.alexnet_layerbylayer import alexnet_conv
#from model.nin import nin_base
#from model.alexnet import alexnet_base
from pytorch_lightning.callbacks import EarlyStopping



class Model_train(object):
    def __init__(self, args):
        self.args = args
        self.flag =0

    def __call__(self):
        args = self.args
        size =8
        if args.dataset == "CIFAR10":
            dm = CIFAR10_rotation(
               batch_size=args.batch_size,
               data_dir='/netscratch/mundra/data/CIFAR10/origin_data/',
               num_workers=args.num_workers,
               rot = args.rotation
            )
            #trainer = pl.Trainer.from_argparse_args(args,callbacks=[EarlyStopping(monitor='val_loss', patience=20)] ,gpus=1)
            if args.model == "LeNet":
                model = LeNet()
            elif args.model == "ReNet":
                model = ReNet_lightning_finetune()
            elif args.model == "AlexNet":
                model = alexnet_conv()
            else:
                model = nin_base()
            alex_layers = [nn.Conv2d(in_channels=3,out_channels= 64, kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(in_channels=64,out_channels= 192, kernel_size=3, padding=1),nn.Conv2d(in_channels=192,out_channels=384, kernel_size=3, padding=1)]
            for i, layer in enumerate(alex_layers):
                #size = int(size/2)
                if i>0:

                   size = int(size/2)
                   #print(size)
                   m_state_dict = torch.load('/netscratch/mundra/model/alexnet_mymodulesCIFAR8{}.pt'.format(i-1))
                   model.load_state_dict(m_state_dict['model_state_dict'])
                   #ReNet_lightning_finetune(size)
                   for param in model.features.parameters():
                      param.requires_grad = False
                #print(layer.out_channels)
                model.add_layer(layer,layer.out_channels,size,i)
                tb_logger = pl_loggers.TensorBoardLogger('/netscratch/mundra/logs/')
                trainer = pl.Trainer.from_argparse_args(args,logger=tb_logger,callbacks=[EarlyStopping(monitor='val_loss', patience=20)] ,gpus=1)    
                trainer.fit(model,dm)
                result1 = trainer.test()
                torch.save({'model_state_dict':model.state_dict()},
                                         '/netscratch/mundra/model/alexnet_mymodulesCIFAR8{}.pt'.format(i))
                print("original",result1)

        else:
            print("svhn")
            dm = SVHN_rotation(
                batch_size=args.batch_size,
                data_dir='/netscratch/mundra/data/SVHN/origin_data/',
                num_workers=args.num_workers,
                rot = args.rotation
                              )

            if args.model == "LeNet":
                model = LeNet()
            elif args.model == "ReNet":
                model = ReNet_lightning_finetune()
            elif args.model == "AlexNet":
                model = alexnet_conv()
            else:
                model = nin_base()
            alex_layers = [nn.Conv2d(in_channels=3,out_channels= 64, kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(in_channels=64,out_channels= 192, kernel_size=3, padding=1),nn.Conv2d(in_channels=192,out_channels=384, kernel_size=3, padding=1)]
            for i, layer in enumerate(alex_layers):
                #size = int(size/2)
                if i>0:

                   size = int(size/2)
                   #print(size)
                   m_state_dict = torch.load('/netscratch/mundra/model/alexnet_mymodulesvhn8{}.pt'.format(i-1))
                   model.load_state_dict(m_state_dict['model_state_dict'])
                   #ReNet_lightning_finetune(size)
                   for param in model.features.parameters():
                      param.requires_grad = False
                model.add_layer(layer,layer.out_channels,size,i)
                tb_logger = pl_loggers.TensorBoardLogger('/netscratch/mundra/logs/')
                trainer = pl.Trainer.from_argparse_args(args,logger=tb_logger,callbacks=[EarlyStopping(monitor='val_loss', patience=20)] ,gpus=1)
                trainer.fit(model,dm)
                result1 = trainer.test()
                torch.save({'model_state_dict':model.state_dict()},
                                         '/netscratch/mundra/model/alexnet_mymodulesvhn8{}.pt'.format(i))
                print("original",result1) 



