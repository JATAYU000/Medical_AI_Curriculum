from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn as nn

class ResNetUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNetUNet, self).__init__()

        backbone = ptcv_get_model("resnet50", pretrained=True)
        self.enc = backbone.features 
        self.enc1 = self.enc[:2]  
        self.enc2 = self.enc[2:3]  
        self.enc3 = self.enc[3:4]  
        self.enc4 = self.enc[4:5]  

        self.up4 = self._upsample_block(2048, 1024)
        self.up3 = self._upsample_block(1024, 512)
        self.up2 = self._upsample_block(512, 256)
        self.up1 = self._upsample_block(256, 128)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def name(self):
        return 'ResNetUNet'
        
    def _upsample_block(self, in_channels, out_channels,sf=2):
        return nn.Sequential(
            nn.Upsample(scale_factor=sf, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x = self.enc4(x3)

        x = self.up4(x) + x3
        x = self.up3(x) + x2
        x = self.up2(x) + x1
        x = self.up1(x) #+ first enc was downscalled by 3 and next is batch or relu ig
        out = self.final_up(x)
        return out


