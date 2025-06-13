import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import consts
from helpers import dice_coef, iou_pytorch, DiceLossNew
from data_loader import test_dataset
def TEST(trained_model,path):
    trained_model.load_state_dict(torch.load(path,map_location=torch.device(consts.device)))
    trained_model = trained_model.to(consts.device)
    trained_model.eval()
    criterion = DiceLossNew()
    test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False)
    avgl = 0
    avgi = 0
    avgd =0
    
    with torch.no_grad():
        for j,img_mask in enumerate(tqdm(test_dataloader)):
            img = img_mask[0].float().to(consts.device)
            mask = img_mask[1].float().to(consts.device)
            y_pred = trained_model(img)
            loss = criterion(y_pred,mask)
            avgl += loss.item()
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).float()
            iou_score = iou_pytorch(mask, y_pred).item()
            avgi+=iou_score
            dice_score = dice_coef(mask, y_pred).item()
            avgd += dice_score
            mask = (mask > 0.5).float()
            mask = mask.squeeze().cpu().numpy()
            pred_np = y_pred.squeeze().cpu().numpy()
            img_np = img.squeeze().permute(1,2,0).cpu().numpy()
            
            plt.figure(figsize=(8, 3))
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(img_np)
        
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth Mask")
            plt.imshow(mask*255 , cmap="gray")
    
            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.imshow(pred_np, cmap="gray")
            plt.show()
            print(f'{j} LOSS: {loss.item()}  IOU: {iou_score}  DICE: {dice_score} ')
        j+=1
        print(f'AVG = LOSS: {avgl/j}  IOU: {avgi/j}  DICE: {avgd/j} ')
               