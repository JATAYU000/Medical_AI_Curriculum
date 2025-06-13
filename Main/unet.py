import torch.nn as nn
import torch

class UNET(nn.Module):
    def __init__(self, dropout_rate=0.1, ch = 32):
        super(UNET, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate)
            )
        
        self.encoder1 = conv_block(3, ch)
        self.encoder2 = conv_block(ch, ch*2)
        self.encoder3 = conv_block(ch*2, ch*4)
        self.encoder4 = conv_block(ch*4, ch*8)
        self.bottle_neck = conv_block(ch*8, ch*16)

        self.upsample1 = nn.ConvTranspose2d(ch*16, ch*8, kernel_size=2, stride=2)
        self.decoder1 = conv_block(ch*16, ch*8)

        self.upsample2 = nn.ConvTranspose2d(ch*8, ch*4, kernel_size=2, stride=2)
        self.decoder2 = conv_block(ch*8, ch*4)

        self.upsample3 = nn.ConvTranspose2d(ch*4, ch*2, kernel_size=2, stride=2)
        self.decoder3 = conv_block(ch*4, ch*2)

        self.upsample4 = nn.ConvTranspose2d(ch*2, ch, kernel_size=2, stride=2)
        self.decoder4 = conv_block(ch*2, ch)

        self.final = nn.Conv2d(ch, 1, kernel_size=1)

        #self.init_weights()
    def name(self):
        return 'UNet'
        
    def forward(self, x):
        c1 = self.encoder1(x)
        c2 = self.encoder2(self.pool(c1))
        c3 = self.encoder3(self.pool(c2))
        c4 = self.encoder4(self.pool(c3))
        c5 = self.bottle_neck(self.pool(c4))

        u6 = self.upsample1(c5)
        u6 = torch.cat([c4, u6], dim=1)
        c6 = self.decoder1(u6)

        u7 = self.upsample2(c6)
        u7 = torch.cat([c3, u7], dim=1)
        c7 = self.decoder2(u7)

        u8 = self.upsample3(c7)
        u8 = torch.cat([c2, u8], dim=1)
        c8 = self.decoder3(u8)

        u9 = self.upsample4(c8)
        u9 = torch.cat([c1, u9], dim=1)
        c9 = self.decoder4(u9)

        return self.final(c9)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


