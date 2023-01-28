import pytorch_lightning as pl
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from torch.utils.data import Dataset, DataLoader
from kornia.geometry import Resize
from kornia.augmentation import CenterCrop, Normalize, RandomHorizontalFlip, RandomGrayscale



class alexnet_conv(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.transform_training = nn.Sequential(
            
            RandomHorizontalFlip(),
            #RandomGrayscale(),
            Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        ) 
        self.transform_valid = nn.Sequential(
        
             Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        )
        self.accuracy = pl.metrics.Accuracy()
        self.features = nn.ModuleList()
    def add_layer(self, layer, channel, size, i):
        self.features.append(layer)
        self.features.append(nn.BatchNorm2d(channel))
        self.features.append(nn.ReLU(inplace=True))
        if i<=1:
            s = size
            self.features.append(nn.MaxPool2d(kernel_size=2))
        else:
            s = 4
        self.classifier = nn.Sequential(
	    nn.Flatten(),
	    nn.Dropout(0.6),
            nn.Linear(channel*s*s, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)
           )


    
    def forward(self, x):
        for i,l in enumerate(self.features):
            x = l(x)
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
        self.log('val_loss', loss)
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
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001,momentum= 0.9, weight_decay = 3e-4)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=0, last_epoch=-1, verbose=False)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.02)
        return [self.optimizer], [self.scheduler]



class MLP_alexnet_conv(pl.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.transform_training = nn.Sequential(
            RandomHorizontalFlip(),
            Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
            )
        self.transform_valid = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
                )
        self.accuracy = pl.metrics.Accuracy()
        self.features1 = nn.Sequential(*list(model.features.children())[:11])
        #self.features2 = nn.Sequential(
                 #nn.Conv2d(192, 384, kernel_size=3, padding=1),
                 #nn.BatchNorm2d(384),
                 #nn.ReLU(inplace = True),
                #)
        #self.features = nn.Sequential(self.features1,self.features2)

        #for param in self.features1.parameters():
            #param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(384 * 4 * 4, 4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.6),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        # print("forward")
        x = self.features1(x)
        x = x.view(x.size(0), 384 * 4* 4)
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
        self.log('val_loss', loss)
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
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum = 0.9, weight_decay = 3e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.02)
        return [self.optimizer], [self.scheduler]

class MLP_alexnet_finetune(pl.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.transform_training = nn.Sequential(
            RandomHorizontalFlip(),
            Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
            )
        self.transform_valid = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
                )
        self.accuracy = pl.metrics.Accuracy()
        self.features = nn.Sequential(*list(model.features.children()))
        for param in model.features.parameters():
            param.requires_grad = True
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(384 * 4 * 4, 4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.6),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        # print("forward")
        x = self.features(x)
        #x = x.view(x.size(0), 384 * 4* 4)
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
        self.log('val_loss', loss)
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
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum = 0.9, weight_decay = 3e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.02)
        return [self.optimizer], [self.scheduler]

