import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms,datasets
import torch

from torch.utils.data import Subset,DataLoader
from config import INPUT_SIZE,BATCH_SIZE
from sklearn.model_selection import StratifiedKFold,KFold,train_test_split




train_transforms = transforms.Compose([
    
    transforms.Resize((600,600), Image.BILINEAR),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAffine(degrees=15,
                              translate=[0.1, 0.1],
                              scale=[0.9, 1.1],
                              shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    
])


val_transforms = transforms.Compose([
    
    transforms.Resize(INPUT_SIZE, Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    
])


def get_dsets():
        
  

    train_dset =  datasets.ImageFolder('/content/BIRD-NOT/bird_dataset_mix' + '/train_images',
                            transform=train_transforms)
    val_dset  =  datasets.ImageFolder('/content/BIRD-NOT/bird_dataset_mix' + '/val_images',
                            transform=val_transforms)
    test_dset  =  datasets.ImageFolder('/content/BIRD-NOT/cub',
                            transform=val_transforms)
    return train_dset,val_dset,test_dset


def train_val_split(dataset, val_split=0.1):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split , shuffle= True,)
    sets = {}
    sets['train'] = torch.utils.data.Subset(dataset, train_idx)
    sets['val'] = torch.utils.data.Subset(dataset, val_idx)
    return sets


def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight   

def get_loaders():
    
    train_dset,val_dset,_ = get_dsets()
    labels = [pt[1] for pt in train_dset]

    # For unbalanced dataset we create a weighted sampler                       
    weights = make_weights_for_balanced_classes(train_dset.imgs, len(train_dset.classes))                                                                
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                                                                                     
    train_loader = DataLoader(train_dset,batch_size=BATCH_SIZE, num_workers=4,drop_last=True,pin_memory=True,sampler=sampler)
    val_loader   = DataLoader(val_dset,batch_size=BATCH_SIZE, shuffle=True, num_workers=4,drop_last=True)
    return train_loader,val_loader

def get_test():

    _,_,test_dset = get_dsets()
    test_loader   = DataLoader(test_dset,batch_size=BATCH_SIZE, shuffle=True, num_workers=4,drop_last=True)
    
    return test_loader
    
def get_unlabel():
    
    
    unlabel_dset =  datasets.ImageFolder('/content/BIRD-NOT/cub',
                            transform=val_transforms)
    
    
    unlabel_loader =  DataLoader(unlabel_dset,batch_size=BATCH_SIZE, shuffle=True, num_workers=4,drop_last=True)
    
    return unlabel_loader
    
    
    
    
    