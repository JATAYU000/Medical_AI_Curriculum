import torch

epochs = 2
batch_size = 4
lr = 0.005
unet_channels = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rs = 384
model_name = 'efficientunet'  # 'unet', 'resnetunet', 'efficientunet'
img_data_path = '../Kvasir-SEG/images'
mask_data_path = '../Kvasir-SEG/masks'
models_path = '../Models'
save_img_path = '../Images'
model_test_path = '../Models/EfficientUNet-ADAM-DICELOSS-30.pth'