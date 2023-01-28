import pytorch_lightning as pl
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import random
from scipy import ndimage
import math
from PIL import Image
from torchvision.datasets import CIFAR10
from kornia.geometry import Resize
from kornia.augmentation import RandomCrop, Normalize, CenterCrop
import itertools
from torch import cat
from model.LRN import LRN
from torch.autograd import Variable

def weights_init(model):
    if type(model) in [nn.Conv2d,nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)

class AlexNet_jigsaw(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.transform_training = nn.Sequential(
 
            Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        ) 
        self.transform_eval = nn.Sequential(

             Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        ) 
        self.accuracy = pl.metrics.Accuracy()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0),
            #nn.BatchNorm2d(96),
            #nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.7),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(5, 0.0001, 0.75),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            #nn.BatchNorm2d(256),
            #nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.7),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(5, 0.0001, 0.75),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            #nn.BatchNorm2d(384),
            #nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.7),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            #nn.BatchNorm2d(384),
            #nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.7),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.7),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.fc6 = nn.Sequential( 
            nn.Linear(256*3*3, 1024),
            #nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.7),
            #nn.Linear(1024, 1024),
            #nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc7 = nn.Sequential( 
            nn.Linear(9*1024, 1024),
            #nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 1000)
        )
        self.apply(weights_init)
    def forward(self, x):
        #print(x.shape)
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)

        x_list = []
        for i in range(9):
            z = self.features(x[i])
            z = self.fc6(z.view(B,-1))
            z = z.view([B,1,-1])
            x_list.append(z)

        x = cat(x_list,1)
        x = self.fc7(x.view(B,-1))
        x = self.classifier(x)
    

        return x
    
    def cross_entropy_loss(self, logits, labels):
      return nn.CrossEntropyLoss()(logits, labels)
    
    def training_step(self, train_batch, batch_idx):

      images,labels = train_batch
      #images = torch.reshape(images,(-1,images.shape[2],images.shape[3],images.shape[4], images.shape[5]))
      #images = self.transform_training(images)
      targets = torch.reshape(labels,(1,-1))
      labels = Variable(labels)
      t = targets.squeeze(0)
      logits = self(images)
      #if batch_idx%50==0:
        #print("traiing",logits)
        #print(t)
      loss = self.cross_entropy_loss(logits, t)
      self.log('train_accuracy', self.accuracy(logits,t))
      return loss
    def validation_step(self, val_batch,batch_idx):
      images,labels = val_batch
      #labels = Variable(labels)
      #images = torch.reshape(images,(-1,images.shape[2],images.shape[3],images.shape[4], images.shape[5]))
      #images = self.transform_eval(images)
      targets = torch.reshape(labels,(1,-1))
      t = targets.squeeze(0)
      logits = self(images)

      #print(logits)
      #print("validation",t)
      loss = self.cross_entropy_loss(logits, t)
      #if batch_idx%50==0:
        #print("traiing",logits)
        #print(t)
      self.log('val_loss', loss,prog_bar=True)
      return loss
    
    def test_step(self, test_batch, batch_idx):
      images, labels = test_batch
      labels = Variable(labels)
      #images = torch.reshape(images,(-1,images.shape[2],images.shape[3],images.shape[4], images.shape[5]))
      #images = self.transform_eval(images)
      targets = torch.reshape(labels,(1,-1))
      t = targets.squeeze(0)
      logits = self(images)
      loss = self.cross_entropy_loss(logits, t)
      self.log('test_loss', loss)
      self.log('test_accuracy', self.accuracy(logits,t))

    def configure_optimizers(self):
      self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01,momentum=0.9, weight_decay = 5e-4)
      self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
      return [self.optimizer],[self.scheduler]
class AlexNet_jigsaw_finetune(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.transform_training = nn.Sequential(
 
            Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        ) 
        self.transform_eval = nn.Sequential(

             Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        ) 
        self.accuracy = pl.metrics.Accuracy()
        self.features = nn.Sequential(*list(model.features.children())[:8])
        for param in self.features.parameters():
            param.requires_grad = False
        #self.features1 = nn.Sequential(
	    #nn.Conv2d(256, 384, kernel_size=3, padding=1),
	    #nn.ReLU(inplace=True),
	#)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*26*26, 1024),
            #nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            #nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )
        self.apply(weights_init)
    def forward(self, x):
        x = self.features(x)
        x = self.features1(x)
        x = self.classifier(x)

        return x        
    
    
    def cross_entropy_loss(self, logits, labels):
      return nn.CrossEntropyLoss()(logits, labels)
    
    def training_step(self, train_batch, batch_idx):

      images,labels = train_batch
      print(images.shape)
      #images = torch.reshape(images,(-1,images.shape[2],images.shape[3],images.shape[4], images.shape[5]))
      #images = self.transform_training(images)
      targets = torch.reshape(labels,(1,-1))
      labels = Variable(labels)
      t = targets.squeeze(0)
      logits = self(images)
      #if batch_idx%50==0:
        #print("traiing",logits)
        #print(t)
      loss = self.cross_entropy_loss(logits, t)
      self.log('train_accuracy', self.accuracy(logits,t))
      return loss
    def validation_step(self, val_batch,batch_idx):
      images,labels = val_batch
      #labels = Variable(labels)
      #images = torch.reshape(images,(-1,images.shape[2],images.shape[3],images.shape[4], images.shape[5]))
      #images = self.transform_eval(images)
      targets = torch.reshape(labels,(1,-1))
      t = targets.squeeze(0)
      logits = self(images)

      #print(logits)
      #print("validation",t)
      loss = self.cross_entropy_loss(logits, t)
      #if batch_idx%50==0:
        #print("traiing",logits)
        #print(t)
      self.log('val_loss', loss,prog_bar=True)
      return loss
    
    def test_step(self, test_batch, batch_idx):
      images, labels = test_batch
      #labels = Variable(labels)
      #images = torch.reshape(images,(-1,images.shape[2],images.shape[3],images.shape[4], images.shape[5]))
      #images = self.transform_eval(images)
      targets = torch.reshape(labels,(1,-1))
      t = targets.squeeze(0)
      logits = self(images)
      loss = self.cross_entropy_loss(logits, t)
      self.log('test_loss', loss)
      self.log('test_accuracy', self.accuracy(logits,t))

    def configure_optimizers(self):
      self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01,momentum=0.9, weight_decay = 5e-4)
      self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
      return [self.optimizer],[self.scheduler]

