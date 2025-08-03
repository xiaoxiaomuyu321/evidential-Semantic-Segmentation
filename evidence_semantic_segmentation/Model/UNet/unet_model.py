""" Full assembly of the parts to form the complete network """

from evidence_semantic_segmentation.Model.UNet.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, num_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, num_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



if __name__ == "__main__":
    # 定义输入参数
    batch_size = 2
    channels = 3  # 例如RGB图像
    height, width = 640, 640  # 输入图像大小，建议是2的幂次方便于下采样
    num_classes = 100
    # 实例化UNet，假设是3通道输入，2类输出（例如二分类语义分割）
    model = UNet(n_channels=channels, num_classes=num_classes, bilinear=True)

    # 构造输入张量
    x = torch.randn(batch_size, channels, height, width)

    # 前向推理
    output = model(x)

    # 输出形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
