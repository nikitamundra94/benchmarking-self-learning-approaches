import torch
import torchvision
import torchvision.transforms.functional as TF
import os
from sklearn import preprocessing
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class Rotation(Dataset):

  def __init__(self,data, rotation, dataset):
  
   self.train_data = data
   self.rotation = rotation
   self.labels_list = []
   le = preprocessing.LabelEncoder()
   le.fit(rotation)
   self.class_label = le.transform(rotation)
   self.dataset = dataset
   print(self.rotation)
   
   #self.class_label = [0,1]
  def __len__(self):
        return len(self.train_data)

  def __getitem__(self, idx):
    
    imgs_list = []
    labels_list = []

    for i in range(len(self.rotation)):
      if self.dataset =="CIFAR10":
        img = TF.rotate(torch.from_numpy(self.train_data[idx]).permute(2,0,1), self.rotation[i]).float()
      else:
        img = TF.rotate(torch.from_numpy(self.train_data[idx]), self.rotation[i]).float()
      imgs_list.append(img)
      labels_list.append(self.class_label[i])
      #data = torch.stack(imgs_list, dim= 0)

    flabels = torch.tensor(labels_list)
    data = torch.stack(imgs_list, dim= 0)
    return data, flabels

  def PlotRotatedImage(self, rot_train_data):
        self.rot_train_data = rot_train_data
        fig = plt.figure(figsize=(20,20))
        for i in range(1):
            img = self.rot_train_data[i].permute(1,2,0)
            fig.add_subplot(1, 9, i+1)
            img = img / 255
            #img = img.permute(1,2,0)
            #npimg = img.numpy()
            plt.imshow(img)
