from tqdm.auto import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import consts
from helpers import dice_coef, iou_pytorch, DiceLossNew
from data_loader import train_dataset, val_dataset


def train_model(model,dataloader,criterion,optimizer):
    model.train()
    train_running_loss = 0.0
    iou_score = 0.0
    dice_score = 0.0
    batch_count = 0
    for j,img_mask in enumerate(tqdm(dataloader)):
        img = img_mask[0].float().to(consts.device)
        mask = img_mask[1].float().to(consts.device)
        y_pred = model(img)
        optimizer.zero_grad()
        loss = criterion(y_pred,mask)
        train_running_loss += loss.item()# * batch_size
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            y_prob = torch.sigmoid(y_pred)
            y_bin = (y_prob > 0.5).float()
            iou_score += iou_pytorch(mask, y_bin).item()
            dice_score += dice_coef(mask, y_bin).item()
    
        batch_count += 1
        
    mean_iou = iou_score / batch_count
    mean_dice = dice_score / batch_count
    train_loss = train_running_loss / (j+1)
    return train_loss, mean_iou, mean_dice


def val_model(model,dataloader,criterion):
    model.eval()
    val_running_loss = 0
    iou_score = 0.0
    dice_score = 0.0
    batch_count = 0
    with torch.no_grad():
        for j,img_mask in enumerate(tqdm(dataloader)):
            img = img_mask[0].float().to(consts.device)
            mask = img_mask[1].float().to(consts.device)
            y_pred = model(img)
            loss = criterion(y_pred,mask)
            
            y_prob = torch.sigmoid(y_pred)
            y_bin = (y_prob > 0.5).float()
            iou_score += iou_pytorch(mask, y_bin).item()
            dice_score += dice_coef(mask, y_bin).item()
            val_running_loss += loss.item() #* batch_size
            batch_count += 1
            
        mean_iou = iou_score / batch_count
        mean_dice = dice_score / batch_count
        val_loss = val_running_loss / (j+1)
    return val_loss, mean_iou, mean_dice


def TRAIN(custom):
    print(custom.name())
    train_dataloader = DataLoader(train_dataset,batch_size=consts.batch_size,shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=consts.batch_size,shuffle=False)

    model = custom.to(consts.device)
    optimizer = optim.Adam(model.parameters(), lr = consts.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.87)
    criterion = DiceLossNew()
    train_loss_lst = []
    val_loss_lst = []
    train_metrics = [] 
    val_metrics = []

    for i in tqdm(range(consts.epochs)):
        train_loss, train_iou, train_dice = train_model(model=model,dataloader=train_dataloader,criterion=criterion,optimizer=optimizer)
        val_loss, val_iou, val_dice = val_model(model=model,dataloader=val_dataloader,criterion=criterion)
        scheduler.step()
        train_loss_lst.append(train_loss)
        val_loss_lst.append(val_loss)
        train_metrics.append((train_iou, train_dice))
        val_metrics.append((val_iou, val_dice))
    
        
        for param_group in optimizer.param_groups:
            print(f"\nModel : {custom.name()} EPOCH : {i} with lr {param_group['lr']}")
        print(f" Train Loss : {train_loss:.4f} IOU :{train_iou:.4f} DIC : {train_dice:.4f}")
        print(f" Valid Loss : {val_loss:.4f} IOU :{val_iou:.4f} DIC : {val_dice:.4f}")
    print('Saving model....')
    torch.save(model.state_dict(), f'{consts.models_path}/{custom.name()}.pth',)
    
    plt.plot(train_loss_lst, color="green", label='train loss')
    plt.plot(val_loss_lst, color="red", label='validation loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Train and Validation LOSS")
    plt.legend()
    plt.savefig(f'{consts.save_img_path}/{custom.name()}_LOSS.png')  
    plt.show()
    
    plt.figure()
    plt.plot([m[0] for m in train_metrics], label="Train IOU", color='blue')
    plt.plot([m[1] for m in train_metrics], label="Train Dice", color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Train IOU and Dice over Epochs")
    plt.legend()
    plt.savefig(f'{consts.save_img_path}/{custom.name()}_train_metrics.png')
    plt.show()
    
    plt.figure()
    plt.plot([m[0] for m in val_metrics], label="Val IOU", color='orange')
    plt.plot([m[1] for m in val_metrics], label="Val Dice", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Validation IOU and Dice over Epochs")
    plt.legend()
    plt.savefig(f'{consts.save_img_path}/{custom.name()}_val_metrics.png')
    plt.show()

