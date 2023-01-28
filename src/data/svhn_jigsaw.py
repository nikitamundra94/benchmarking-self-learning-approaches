
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from jigsaw import JigsawPuzzle

class svhn_jigsaw(pl.LightningDataModule):
  
  def __init__(self, batch_size,  data_dir, num_workers, classes):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.classes = classes
          
  def setup(self, stage):
    # transforms for images
    self.transform = transforms.Compose([transforms.ToTensor()])
    self.full_data = datasets.SVHN(self.data_dir, split='train',
                                       download=True, transform=self.transform)
    self.train_data = self.full_data.data[0:60000]
    self.valid_data = self.full_data.data[60000:73257]
    self.test_data_full = datasets.SVHN(self.data_dir, split='test', download=True, transform=self.transform)
    self.test_data = self.test_data_full.data


  def train_dataloader(self):
    jigsaw = JigsawPuzzle(self.train_data,self.classes, flag=1)
    return DataLoader(jigsaw, batch_size=self.batch_size, shuffle=True,pin_memory=True, num_workers=self.num_workers)

  def val_dataloader(self):
    jigsaw = JigsawPuzzle(self.valid_data, self.classes, flag=0)
    return DataLoader(jigsaw, batch_size = self.batch_size,pin_memory=True, num_workers=self.num_workers)

  def test_dataloader(self):
    jigsaw = JigsawPuzzle(self.test_data, self.classes,flag=1)
    return DataLoader(jigsaw, batch_size = self.batch_size,pin_memory=True, num_workers=self.num_workers)
