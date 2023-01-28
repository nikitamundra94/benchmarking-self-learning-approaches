
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


class ReNet_lightning_layer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        print("renet finetune")
        self.transform_training = nn.Sequential(
            RandomCrop((32,32), padding=4),
            RandomGrayscale(p=0.25),
            RandomHorizontalFlip(p=0.25),
            RandomVerticalFlip(p=0.25),
            RandomAffine(degrees = 0, translate = (1/16, 1/16), p=0.5),
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
            ReNet(2*2*320, 160, kernel_size=(2,2)),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(320 * 2 * 2, 4096),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
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
        print(len(images.size()))
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay =3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=0, last_epoch=-1, verbose=False)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.2)
        return [self.optimizer], [self.scheduler]
class MLP_renet_layer(pl.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.transform_training = nn.Sequential(
            RandomCrop((32,32), padding=4),
            RandomGrayscale(p=0.25),
            RandomHorizontalFlip(p=0.25),
            RandomVerticalFlip(p=0.25),
            RandomAffine(degrees = 0, translate = (1/16, 1/16), p=0.5),
            Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        )
        self.transform_valid = nn.Sequential(
                Normalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
        )
        self.accuracy = pl.metrics.Accuracy()
        self.features = nn.Sequential(*list(model.features.children())[:6])
        #for param in self.features.parameters():
            #param.requires_grad = False
        self.classifier = nn.Sequential(
            #ReNet(2*2*320, 160, kernel_size=(2,2)),
            #nn.Dropout(0.3), 
            nn.Flatten(),
            nn.Linear(320 * 4 * 4, 4096),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
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
        #print("hello")
        images, labels = val_batch
        print(len(images.size()))
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.000001, weight_decay =1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=0, last_epoch=-1, verbose=False)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.2)
        return [self.optimizer], [self.scheduler]




