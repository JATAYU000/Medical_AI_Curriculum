import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from PIL import Image
import numpy as np
from albumentations.pytorch import ToTensorV2

import consts

class PolypDataset(Dataset):
    def __init__(self,img_list,mask_list,transform=None):
        self.img_list = img_list
        self.mask_list = mask_list
        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self,index):
        img_path = self.img_list[index]
        mask_path = self.mask_list[index]
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = np.array(img)
        mask = np.array(mask)
        mask[mask>=250.0] = 1.0
        
        if self.transform:
            augmentation = self.transform(image=img, mask=mask)
            img = augmentation["image"]
            mask = augmentation["mask"]
            mask = torch.unsqueeze(mask,0)
            
        return img,mask[:,:,:,0]


train_transform = A.Compose([A.Resize(consts.rs,consts.rs), 
                             A.Rotate(limit=45,p=0.1),
                             A.HorizontalFlip(p=1),
                             A.RandomBrightnessContrast(p=1.0,contrast_limit=(-0.3,0.3),brightness_limit=(-0.3,0.3)),
                             A.GaussNoise(std_range=(0.05, 0.2), p=1.0),
                             A.Normalize(mean=(0,0,0),std=(1,1,1),max_pixel_value=255),
                             ToTensorV2()])

val_transform = A.Compose([A.Resize(consts.rs,consts.rs),
                           A.Normalize(mean=(0,0,0),std=(1,1,1),max_pixel_value=255),
                           ToTensorV2()])

test_transform = A.Compose([A.Resize(consts.rs,consts.rs),
                            A.Normalize(mean=(0,0,0),std=(1,1,1),max_pixel_value=255),
                           ToTensorV2()])

image_paths = sorted([os.path.join(consts.img_data_path, f) for f in os.listdir(consts.img_data_path) if f.endswith('.jpg')])
mask_paths = sorted([os.path.join(consts.mask_data_path, f) for f in os.listdir(consts.mask_data_path) if f.endswith('.jpg')])

train_imgs, val_imgs, train_msks, val_msks = train_test_split(image_paths, mask_paths, test_size=0.2,random_state=42)
val_imgs, test_imgs, val_msks, test_msks = train_test_split(val_imgs, val_msks, test_size=0.25,random_state=42)

train_dataset = PolypDataset(train_imgs, train_msks, transform=train_transform)
val_dataset = PolypDataset(val_imgs, val_msks, transform=val_transform)
test_dataset = PolypDataset(test_imgs, test_msks, transform=test_transform)