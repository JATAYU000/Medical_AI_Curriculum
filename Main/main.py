from helpers import test_losses_metrics
from train import TRAIN
from test import TEST
import consts
from unet import UNET
from resnetunet import ResNetUNet
from efficientunet import EfficientUNet

print("\n---------- Testing Losses and Metrics ----------\n")
test_losses_metrics()


if consts.model_name == 'unet':
    m = UNET(ch=consts.unet_channels)
elif consts.model_name == 'resnetunet':
    m = ResNetUNet()
elif consts.model_name == 'efficientunet':
    m = EfficientUNet()
    
print(f"\n---------- Train {m.name()} Model----------\n")
#TRAIN(custom = m)


print(f"\n---------- Testing {m.name()} Model ----------\n")
TEST(m,path = consts.model_test_path)
