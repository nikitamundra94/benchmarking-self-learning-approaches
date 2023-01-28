
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from rotation import Rotation

class CIFAR10_rotation(pl.LightningDataModule):
  
  def __init__(self, batch_size,  data_dir, num_workers, rot):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.rot = rot
        self.rot = self.rotation_func(self.rot)

  def rotation_func(self, rot):
        return{
                1 : lambda: [0],
                2 : lambda: [0,180],
                3 : lambda:[0,90,180],
                4 : lambda:[0,90,180,270],
                5 : lambda: [0,45,90,135,180,225,270,315],
              }.get(rot, lambda: 'Not a rotation')()
          
  def setup(self, stage):
    # transforms for images

    self.transform = transforms.Compose([transforms.ToTensor()])
    self.full_data = datasets.CIFAR10(self.data_dir, train=True,
                                   download=True,transform = self.transform)
    self.train_data = self.full_data.data[0:40000]
    self.valid_data = self.full_data.data[40000:50000]
    self.test_data_full =  datasets.CIFAR10(self.data_dir, train = False,transform=self.transform)
    self.test_data = self.test_data_full.data


  def train_dataloader(self):
    rotation = Rotation(self.train_data,self.rot,"CIFAR10")
    return DataLoader(rotation, batch_size=self.batch_size, shuffle=True,pin_memory=True, num_workers=self.num_workers)

  def val_dataloader(self):
    rotation = Rotation(self.valid_data, self.rot,"CIFAR10")
    return DataLoader(rotation, batch_size = self.batch_size,pin_memory=True, num_workers=self.num_workers)

  def test_dataloader(self):
    rotation = Rotation(self.test_data, self.rot,"CIFAR10")
    return DataLoader(rotation, batch_size = self.batch_size,pin_memory=True, num_workers=self.num_workers)
