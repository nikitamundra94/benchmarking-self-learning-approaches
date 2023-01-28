
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
from einops import rearrange
from kornia.augmentation import Normalize, RandomHorizontalFlip,RandomVerticalFlip,RandomAffine, RandomCrop, RandomGrayscale
import itertools
from torch import cat


def weights_init(m):
    parameters = m.state_dict()
    for each_key in parameters.keys():
            print(f'Init-{each_key}')
            if 'weight_ih' in each_key:
                nn.init.orthogonal_(parameters[each_key])
            elif 'weight_hh' in each_key:
                nn.init.orthogonal_(parameters[each_key])
            elif 'bias' in each_key:
                nn.init.constant_(parameters[each_key], val=0)


class ReNet(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=(2, 2), rnn='GRU', depth=(1, 1)):
        super(ReNet, self).__init__()
        if rnn == 'GRU':
            rnn = nn.GRU
        elif rnn == 'LSTM':
            rnn = nn.LSTM

        self.lstm_h = rnn(input_size, hidden_size, bias=False, num_layers=depth[0], bidirectional=True)
        #print("Input size is", input_size)
        #print("Output size is", hidden_size)
        self.lstm_v = rnn(hidden_size * 2, hidden_size, bias=False, num_layers=depth[1], bidirectional=True)

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.lstm_h.apply(weights_init)
        self.lstm_v.apply(weights_init)

    def forward(self, x):
        k_w, k_h = self.kernel_size
        b, c, h, w = x.size()
        assert h % k_h == 0 and w % k_w == 0, 'input size does not match with kernel size'
        x = rearrange(x, 'b c (h1 h2) (w1 w2) -> h1 (b w1) (c h2 w2)', w2=k_w, h2=k_h)
        #print("The value of x", x)
        x, _ = self.lstm_h(x)
        x = rearrange(x, 'h1 (b w1) (c h2 w2) -> w1 (b h1) (c h2 w2)', b=b, w2=k_w, h2=k_h)
        x, _ = self.lstm_v(x)
        x = rearrange(x, 'w1 (b h1) (c h2 w2) -> b (c h2 w2) h1 w1', b=b, w2=k_w, h2=k_h)
        return x


class ReNet_jigsaw(pl.LightningModule):
    def __init__(self):
        super().__init__()
        print("renet finetune")
        self.transform_training = nn.Sequential(
              Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        ) 
        self.transform_valid = nn.Sequential(
             Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        ) 
        self.accuracy = pl.metrics.Accuracy()
        self.features = nn.Sequential(
            ReNet(2 * 2 * 3, 160, kernel_size=(2, 2)),
            nn.Dropout(p=0.3),
            ReNet(2 * 2 * 320, 160, kernel_size=(2, 2)),
            nn.Dropout(p=0.3),
            ReNet(2*2*320, 160, kernel_size=(2,2)),
            nn.Dropout(0.3),
           )
        self.fc6 = nn.Sequential( 
            nn.Linear(320 * 9 *9 , 4096),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
    
        self.fc7 = nn.Sequential(
            nn.Linear(9*4096 , 4096),
            nn.Dropout(0.3),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
           nn.Linear(4096, 1000),
        )

    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)
        x_list = []
        for i in range(9):
            z = self.features(x[i])
            z = self.fc6(z.reshape(B,-1))
            z = z.reshape([B,1,-1])
            x_list.append(z)

        x = cat(x_list,1)
        
        x = self.fc7(x.reshape(B,-1))
        x = self.classifier(x)
        return x


    def cross_entropy_loss(self, logits, labels):
        return nn.CrossEntropyLoss()(logits, labels)

    def training_step(self, train_batch, batch_idx):

        images, labels = train_batch      
        targets = torch.reshape(labels, (1, -1))
        t = targets.squeeze(0)
        logits = self(images)
        loss = self.cross_entropy_loss(logits, t)
        return loss

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        targets = torch.reshape(labels, (1, -1))
        t = targets.squeeze(0)
        logits = self(images)
        loss = self.cross_entropy_loss(logits, t)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        targets = torch.reshape(labels, (1, -1))

        t = targets.squeeze(0)
        logits = self(images)
        loss = self.cross_entropy_loss(logits, t)
        self.log('test_loss', loss)
        self.log('test_accuracy', self.accuracy(logits, t))
        return loss

    def configure_optimizers(self):
      self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01,momentum=0.9, weight_decay = 5e-4)
      self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
      return [self.optimizer],[self.scheduler]
class MLP_renet_finetune(pl.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.accuracy = pl.metrics.Accuracy()
        self.features = nn.Sequential(*list(model.features.children())[:6])
        #for param in self.features.parameters():
            
            #param.requires_grad = False
        self.classifier = nn.Sequential(
            #nn.Dropout(0.3),
            #ReNet(2*2*320, 160, kernel_size=(2,2)),
            #nn.Dropout(0.3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320 * 28 * 28, 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def cross_entropy_loss(self, logits, labels):
        return nn.CrossEntropyLoss()(logits, labels)

    def training_step(self, train_batch, batch_idx):
            
        images, labels = train_batch
         
        targets = torch.reshape(labels, (1, -1))
        t = targets.squeeze(0)
        logits = self(images)
        loss = self.cross_entropy_loss(logits, t)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        #print("hello")
        images, labels = val_batch
        targets = torch.reshape(labels, (1, -1))
        t = targets.squeeze(0)
        logits = self(images)
        loss = self.cross_entropy_loss(logits, t)
        self.log('val_loss', loss)
        return loss
        
    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        targets = torch.reshape(labels, (1, -1))

        t = targets.squeeze(0)
        logits = self(images)
        loss = self.cross_entropy_loss(logits, t)
        self.log('test_loss', loss)
        self.log('test_accuracy', self.accuracy(logits, t))
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01,momentum=0.9, weight_decay = 5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        return [self.optimizer],[self.scheduler]


