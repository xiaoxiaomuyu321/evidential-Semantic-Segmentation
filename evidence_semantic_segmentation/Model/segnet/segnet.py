import torch
import torch.nn as nn
import torch.nn.functional as F

# 模块封装：Conv + BN + ReLU
def conv_bn_relu(in_channels, out_channels, kernel_size=3, padding=1, momentum=0.1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels, momentum=momentum),
        nn.ReLU(inplace=True)
    )

class SegNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(SegNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = nn.Sequential(
            conv_bn_relu(input_channel, 64),
            conv_bn_relu(64, 64)
        )
        self.enc2 = nn.Sequential(
            conv_bn_relu(64, 128),
            conv_bn_relu(128, 128)
        )
        self.enc3 = nn.Sequential(
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 256)
        )
        self.enc4 = nn.Sequential(
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512)
        )
        self.enc5 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512)
        )

        # Decoder
        self.dec5 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512)
        )
        self.dec4 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 256)
        )
        self.dec3 = nn.Sequential(
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 128)
        )
        self.dec2 = nn.Sequential(
            conv_bn_relu(128, 128),
            conv_bn_relu(128, 64)
        )
        self.dec1 = nn.Sequential(
            conv_bn_relu(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x1p, id1 = self.pool(x1)

        x2 = self.enc2(x1p)
        x2p, id2 = self.pool(x2)

        x3 = self.enc3(x2p)
        x3p, id3 = self.pool(x3)

        x4 = self.enc4(x3p)
        x4p, id4 = self.pool(x4)

        x5 = self.enc5(x4p)
        x5p, id5 = self.pool(x5)

        # Decoder
        x5d = self.unpool(x5p, id5, output_size=x5.shape)
        x5d = self.dec5(x5d)

        x4d = self.unpool(x5d, id4, output_size=x4.shape)
        x4d = self.dec4(x4d)

        x3d = self.unpool(x4d, id3, output_size=x3.shape)
        x3d = self.dec3(x3d)

        x2d = self.unpool(x3d, id2, output_size=x2.shape)
        x2d = self.dec2(x2d)

        x1d = self.unpool(x2d, id1, output_size=x1.shape)
        out = self.dec1(x1d)

        return out


if __name__ == "__main__":
    batch_size = 2
    channels = 3
    height, width = 640, 640  # 推荐先用小图测试
    num_classes = 100

    model = SegNet(input_channel=channels, num_classes=num_classes)
    x = torch.randn(batch_size, channels, height, width)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # 应该是 [2, num_classes, 256, 256]
