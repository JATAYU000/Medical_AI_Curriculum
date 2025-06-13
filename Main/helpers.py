import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
import albumentations as A
import os
import consts

def dice_coef(inputs, targets, smooth=1e-8):
    inputs = inputs.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) 
    return dice


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)
    outputs = outputs > 0.5 
    labels = labels > 0.5  
    intersection = (outputs & labels).float().sum((1, 2))  
    union = (outputs | labels).float().sum((1, 2))         
    
    iou = (intersection + 1e-8) / (union + 1e-8)  
    return iou.mean()


class DiceLossNew(nn.Module):
    def __init__(self):
        super(DiceLossNew, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        #flatten label and prediction
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) 
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)     
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def test_losses_metrics():
    mask_path1 = f'{consts.mask_data_path}/cju0qkwl35piu0993l0dewei2.jpg'
    mask_path2 = f'{consts.mask_data_path}/cju0rx1idathl0835detmsp84.jpg'
    val_transform = A.Compose([A.Resize(consts.rs,consts.rs),
                           A.Normalize(mean=(0,0,0),std=(1,1,1),max_pixel_value=255),
                           ToTensorV2()])
    
    mask1 = np.asarray(Image.open(mask_path1))
    mask2 = np.asarray(Image.open(mask_path2))  
    augmentation = val_transform(image=mask1)
    mask_1_tensor = augmentation["image"].unsqueeze(0)
    augmentation = val_transform(image=mask2)
    mask_2_tensor = augmentation["image"].unsqueeze(0)
    bce = nn.BCEWithLogitsLoss()
    di = DiceLossNew()
    bdi = DiceBCELoss()

    print("Mask 1: ",mask1.shape)
    print("Mask 2: ",mask2.shape)
    mask_1_tensor = mask_1_tensor[:,0,:,:]
    mask_2_tensor = mask_2_tensor[:,0,:,:]

    print("\nSame Masks Testing\n")
    print("Mask 1 shape: ",mask_1_tensor.shape)
    print(f"IoU: {iou_pytorch(mask_1_tensor, mask_1_tensor).item():.4f}")
    print(f"Dice: {dice_coef(mask_1_tensor, mask_1_tensor).item():.4f}")
    print(f"BCE Loss: {bce(mask_1_tensor, mask_1_tensor).item():.4f}")
    print(f"Dice Loss: {di(mask_1_tensor, mask_1_tensor).item():.4f}")
    print(f"Dice + bce Loss: {bdi(mask_1_tensor, mask_1_tensor).item():.4f}")

    print("\nDifferent Masks Testing\n")
    print("Mask 1 shape: ",mask_1_tensor.shape)
    print("Mask 2 shape: ",mask_2_tensor.shape)
    print(f"IoU: {iou_pytorch(mask_1_tensor, mask_2_tensor).item():.4f}")
    print(f"Dice: {dice_coef(mask_1_tensor, mask_2_tensor).item():.4f}")
    print(f"BCE Loss: {bce(mask_1_tensor, mask_2_tensor).item():.4f}")
    print(f"Dice Loss: {di(mask_1_tensor, mask_2_tensor).item():.4f}")
    print(f"Dice + bce Loss: {bdi(mask_1_tensor, mask_2_tensor).item():.4f}")


