import torch.nn as nn
import importlib

from evidence_semantic_segmentation.Model.deeplabV3plus import deeplabv3plus
from evidence_semantic_segmentation.Model.UNet import unetplus2, unet
from evidence_semantic_segmentation.Model.SETR.transformer_seg import SETRModel
from evidence_semantic_segmentation.Model.segnet.segnet import SegNet
from evidence_semantic_segmentation.Model.SegFormer.segformer import SegFormerVariants

class Segmentation(nn.Module):
    def __init__(self,
                 model_name: str,
                 img_size:int=512,
                 num_classes: int = 1000,
                 **kwargs):
        """
        model_name: 支持的模型名称，如：
            - convnet: MobileNetv3, EfficientNetv2, FasterNet, ResNet, ShuffleNetv2
            - transformer: MobileVIT, CAS_VIT
            - conv_vit: RMT
        kwargs: 传入各个模型构造函数的其他参数（如 input_size 等）
        """
        super(Segmentation, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = img_size


        if model_name == "Deeplabv3plus":
            self.model = deeplabv3plus(num_classes=self.num_classes, backbone="xception", pretrained=False, downsample_factor=16)
        elif model_name == "UNetPlus2":
            self.model = unetplus2(in_channel=3, num_classes=self.num_classes)
        elif model_name == "UNet":
            self.model = unet(n_channels=3, num_classes=self.num_classes, bilinear=True)
        elif model_name == "STER":
            self.model = SETRModel(out_channels=self.num_classes)
        elif model_name == "SegNet":
            self.model =  SegNet(input_channel=3, num_classes=self.num_classes)
        elif model_name == "SegFormer":
            self.model = SegFormerVariants(variant='B1', img_size=640, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")


    def forward(self, x):
        return self.model(x)


