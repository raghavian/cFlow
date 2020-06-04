import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pdb

class LIDC(Dataset):
    def __init__(self, rater=4, data_dir = '/datadrive/raghav/lidc/lidcSeg/raw/', transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.rater = rater
        self.transform = transform
        self.data, self.targets = torch.load(data_dir+'lidcSeg.pt')
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label.type(torch.FloatTensor)


class Drive(Dataset):
    def __init__(self, split='train', data_dir = '/datadrive/raghav/retinaDataset/', 
            transform=None, target_transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.targets = torch.load(data_dir+'retina_'+split+'_new.pt')
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        image, label_mask = self.data[index], self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
            label_mask = self.target_transform(label_mask)
        return image, label_mask[:2], label_mask[[2]]

