import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch import cat
import numpy as np
from PIL import Image
from list_permutation import list_permutation

class JigsawPuzzle(Dataset):
  def __init__(self, traindata, classes, flag):
    self.image_transformer = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize(225, Image.BILINEAR),
            ])
    self.data = traindata
    self.transform_tile = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(64),
            transforms.Resize((72, 72), Image.BILINEAR),
            #transforms.Lambda(rgb_jittering),
            transforms.ToTensor(),
            
        ])
    self.get_permutation = np.load('/netscratch/mundra/svhnpermutations/permutations_%s.npy' % (classes))
    #self.get_permutation = self.list_permutation.get_permutation()
   
  def __len__(self):
        return len(self.data)

  def __getitem__(self, idx):

    #print(type(self.data[idx]))
    #data1 = torch.from_numpy(self.data[idx]).permute(0,2,1).float()
    img = self.image_transformer(self.data[idx])
    permutation = self.get_permutation
    number_permutation = permutation.shape[0]
    order_permutation = np.random.randint(number_permutation)
    tiles = [None]*9
    t=0
    s = float(img.size[0]) / 3
    a = s / 2
    #arr = np.array(img).reshape((9,75,75,3))
    labels_list = []
    for n in range(9):
      i = n / 3
      j = n % 3
      c = [a * i * 2 + a, a * j * 2 + a]
      c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
      tile = img.crop(c.tolist()) 
      tile = self.transform_tile(tile)
      #print(tile.shape)
      m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
      s[s == 0] = 1
      norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
      tile = norm(tile)
      tiles[n] = tile

    labels_list.append(order_permutation)
    labels = torch.tensor(labels_list)
    shuffled_data = [tiles[permutation[order_permutation][t]] for t in range(9)]
    data = torch.stack(shuffled_data, 0)
    return data, labels
    
def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')
