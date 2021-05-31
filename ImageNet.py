import torch
import os
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import json
import pickle

class ImageNet(Dataset):
    
    def load_foldermap(self):
        f = open("imagenet_foldermap.pickle","rb")
        m = pickle.load(f)
        return m
    def load_name_foldermap(self):
        f = open("val_name_dict.pickle","rb")
        m = pickle.load(f)
        return m
    
    def get_samples(self):
        Dirs = 0
        Files = []
        for base, dirs, files in os.walk(self.img_dir):
            for file in files:
                Files.append(os.path.join(base,file).replace('\\','/'))
                
        return Files
    
    def __init__(self,img_dir,transforms):
        super().__init__()
        self.img_dir = img_dir
        self.transforms = transforms
        self.samples = self.get_samples()
        self.foldermap = self.load_foldermap()
        #self.name_foldermap = self.load_name_foldermap()
        self.labels = [i.split("/")[-2] for i in self.samples]
        #self.labels = [self.name_foldermap[(i.split("/")[-1]).split(".")[0]] for i in self.samples]    
        self.labels = torch.LongTensor([int(self.foldermap[i]) for i in self.labels])
     
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        
        sample = Image.open(self.samples[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample,label

    
class ImageNetFull(Dataset):
    
    def load_foldermap(self):
        f = open("imagenet_foldermap.pickle","rb")
        m = pickle.load(f)
        return m
    def load_name_foldermap(self):
        f = open("val_name_dict.pickle","rb")
        m = pickle.load(f)
        return m
    
    def get_samples(self):
        Dirs = 0
        Files = []
        for base, dirs, files in os.walk(self.img_dir):
            for file in files:
                Files.append(os.path.join(base,file).replace('\\','/'))
                
        return Files
    
    def __init__(self,img_dir,type,transforms):
        super().__init__()
        self.img_dir = img_dir
        self.transforms = transforms
        self.samples = self.get_samples()
        self.foldermap = self.load_foldermap()
        self.name_foldermap = self.load_name_foldermap()
        if type=="train":
            self.labels = [i.split("/")[-2] for i in self.samples]
        elif type =="val": 
            self.labels = [self.name_foldermap[(i.split("/")[-1]).split(".")[0]] for i in self.samples]    
        self.labels = torch.LongTensor([int(self.foldermap[i]) for i in self.labels])
     
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        
        sample = Image.open(self.samples[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample,label
    
    