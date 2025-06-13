from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn as nn

class EfficientUNet(nn.Module):
    def __init__(self):
        super(EfficientUNet, self).__init__()

        # Encoder (EfficientNet-Bxb)
        self.enc  = ptcv_get_model("efficientnet_b5b", pretrained=True)
        self.enc = self.enc.features
        self.enc0 = self.enc[0:2]  
        self.enc1 = self.enc[2:3]  
        self.enc2 = self.enc[3:4]  
        self.enc3 = self.enc[4:5] 
        self.enc4 = self.enc[5:6]
        # Decoder
        self.up4 = self._upsample_block(512, 176)
        self.up3 = self._upsample_block(176, 64)
        self.up2 = self._upsample_block(64, 40)
        self.up1 = self._upsample_block(40, 24)
        
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(24, 1, kernel_size=1))
        
    def name(self):
        return 'EfficientUNet-B5'
        
    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        # Encoder 
        x0 = self.enc0(x) 
        x1 = self.enc1(x0) 
        x2 = self.enc2(x1) 
        x3 = self.enc3(x2)
        x = self.enc4(x3)
        # Decoder 
        x = self.up4(x) + x3  
        x = self.up3(x) + x2  
        x = self.up2(x) + x1 
        x = self.up1(x) + x0  
        out = self.final_up(x)  
        return out


