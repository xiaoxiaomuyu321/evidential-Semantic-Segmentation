from .unet_model import UNet
from .unetplus2 import NestedUNet

def unet(n_channels, num_classes, bilinear):
    return UNet(n_channels, num_classes, bilinear=bilinear)

def unetplus2(in_channel, num_classes):
    return NestedUNet(in_channel=in_channel,num_classes=num_classes, deepsupervision=False)
