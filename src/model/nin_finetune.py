import pytorch_lightning as pl
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from kornia.augmentation import RandomHorizontalFlip, Normalize

def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad:
                    m.weight.data.fill_(1)
                if m.bias.requires_grad:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias.requires_grad:
                    m.bias.data.zero_()

class NIN_finetune(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.accuracy = pl.metrics.Accuracy()
        self.transform_training = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
                RandomHorizontalFlip(p=0.5)
                )
        self.transform_valid = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
                )
        print("hello")
        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels = 192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,out_channels = 160, kernel_size=1, padding=0, bias =False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.Conv2d(160,out_channels = 96, kernel_size=1, stride=1, padding= 0,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            nn.Conv2d(96, out_channels =192, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, out_channels =192, kernel_size=1, padding=0,bias= False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,out_channels =192 , kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192,out_channels =192 , kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,out_channels =192 , kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,out_channels =192 , kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
	    #nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
	    nn.Conv2d(192,out_channels =192 , kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,out_channels =192 , kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,out_channels =192 , kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(192),
            )
        self.classifier = nn.Sequential(
            #nn.Dropout(0.7),
            nn.Linear(192,10)
        )
        #self.apply(weight_initialization)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, (8,8)).view(-1,192)
        x = self.classifier(x)
        return x

    def cross_entropy_loss(self, logits, labels):
        return nn.CrossEntropyLoss()(logits, labels)

    def training_step(self, train_batch, batch_idx):

        images, labels = train_batch
        if len(images.size()) == 5:

            train_img = torch.reshape(images, (-1, images.shape[2], images.shape[3], images.shape[4]))
        else:
            train_img = images
        targets = torch.reshape(labels, (1, -1))
        train_img = self.transform_training(train_img)
        t = targets.squeeze(0)
        logits = self(train_img)
        loss = self.cross_entropy_loss(logits, t)
        return loss

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        if len(images.size()) == 5:

            valid_img = torch.reshape(images, (-1, images.shape[2], images.shape[3], images.shape[4]))
        else:
            valid_img = images
        targets = torch.reshape(labels, (1, -1))
        valid_img = self.transform_valid(valid_img)
        t = targets.squeeze(0)
        logits = self(valid_img)
        loss = self.cross_entropy_loss(logits, t)
        self.log('val_loss',loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        # print(images.shape[2])
        if len(images.size()) == 5:
            test_img = torch.reshape(images, (-1, images.shape[2], images.shape[3], images.shape[4]))
        else:
            test_img = images
        targets = torch.reshape(labels, (1, -1))
        test_img = self.transform_valid(test_img)

        t = targets.squeeze(0)
        logits = self(test_img)
        loss = self.cross_entropy_loss(logits, t)
        self.log('test_loss', loss)
        self.log('test_accuracy', self.accuracy(logits, t))
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1,momentum=0.9, weight_decay= 5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.2)
        return [self.optimizer], [self.scheduler]



class CONV_finetune(pl.LightningModule):
    def __init__(self,model):
        super().__init__()

        self.accuracy = pl.metrics.Accuracy()
        self.transform_training = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
                RandomHorizontalFlip()
                )
        self.transform_valid = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
                )
        self.features = nn.Sequential(*list(model.features.children())[:29])
        self.classifier = nn.Sequential(
            #nn.Dropout(0.4),
            nn.Linear(192 , 10),
            #nn.BatchNorm1d(200),
            #nn.ReLU(inplace=True),
            #nn.Dropout(0.4),
            #nn.Linear(200, 200),
        )
        #self.apply(weight_initialization)

    def forward(self, x):
        # print("forward")
        x = self.features(x)
        x = F.avg_pool2d(x, (8,8)).view(-1,192)
        #x = x.view(x.size(0), 192 *4* 4)
        x = self.classifier(x)
        return x

    def cross_entropy_loss(self, logits, labels):
        return nn.CrossEntropyLoss()(logits, labels)

    def training_step(self, train_batch, batch_idx):

        images, labels = train_batch
        if len(images.size()) == 5:

            train_img = torch.reshape(images, (-1, images.shape[2], images.shape[3], images.shape[4]))
        else:
            train_img = images
        train_img = self.transform_training(train_img)
        targets = torch.reshape(labels, (1, -1))
        t = targets.squeeze(0)
        logits = self(train_img)
        loss = self.cross_entropy_loss(logits, t)
        return loss

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch

        if len(images.size()) == 5:

            valid_img = torch.reshape(images, (-1, images.shape[2], images.shape[3], images.shape[4]))
        else:
            valid_img = images
        valid_img = self.transform_valid(valid_img)
        targets = torch.reshape(labels, (1, -1))
        t = targets.squeeze(0)
        logits = self(valid_img)
        loss = self.cross_entropy_loss(logits, t)
        self.log('val_loss',loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        # print(images.shape[2])
        if len(images.size()) == 5:
            test_img = torch.reshape(images, (-1, images.shape[2], images.shape[3], images.shape[4]))
        else:
            test_img = images
        test_img = self.transform_valid(test_img)
        targets = torch.reshape(labels, (1, -1))

        t = targets.squeeze(0)
        logits = self(test_img)
        loss = self.cross_entropy_loss(logits, t)
        self.log('test_loss', loss)
        self.log('test_accuracy', self.accuracy(logits, t))
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum = 0.9, weight_decay= 5e-4 )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.02)
        return [self.optimizer], [self.scheduler]
