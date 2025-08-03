import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. 自定义颜色映射表
CUSTOM_COLORMAP = [
    (0, 0, 0),           # 0 background
    (128, 0, 0),         # 1 person
    (0, 128, 0),         # 2 cable
    (128, 128, 0),       # 3 tube
    (0, 0, 128),         # 4 indicator
    (128, 0, 128),       # 5 electrical equipment
    (0, 128, 128),       # 6 electronic equipment
    (128, 128, 128),     # 7 mining equipment
    (64, 0, 0),          # 8 rail area
    (128, 64, 0),        # 9 support equipment
    (64, 128, 0),        # 10 door
    (192, 128, 0),       # 11 tools and materials
    (64, 0, 128),        # 12 rescue equipment
    (192, 0, 128),       # 13 container
    (64, 128, 128),      # 14 metal fixture
    (192, 128, 128),     # 15 anchoring equipment
]

def get_custom_colormap2label():
    """构建从 RGB 颜色到类别索引的映射表"""
    colormap2label = np.zeros(256 ** 3, dtype=np.uint8)
    for idx, (r, g, b) in enumerate(CUSTOM_COLORMAP):
        colormap2label[(r * 256 + g) * 256 + b] = idx
    return colormap2label


# 2. 数据集类
class VOCSegDataset(Dataset):
    def __init__(self, voc_root, image_set='train', transform=None):
        self.voc_root = voc_root  # 不包含 VOC2007 层
        self.image_dir = os.path.join(self.voc_root, 'JPEGImages')
        self.mask_dir = os.path.join(self.voc_root, 'SegmentationClass')
        self.split_file = os.path.join(self.voc_root, 'ImageSets', 'Segmentation', f'{image_set}.txt')
        assert os.path.isfile(self.split_file), f"No such file: {self.split_file}"

        with open(self.split_file, 'r') as f:
            self.file_names = [x.strip() for x in f.readlines()]

        self.colormap2label = get_custom_colormap2label()
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        img_path = os.path.join(self.image_dir, name + '.jpg')
        mask_path = os.path.join(self.mask_dir, name + '_color.png')

        image = cv2.imread(img_path)[:, :, ::-1]  # BGR -> RGB
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        label_map = self.mask_to_label(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=label_map)
            image = augmented['image']
            label_map = augmented['mask'].long()

        return image, label_map

    def mask_to_label(self, mask):
        mask = mask.astype('int32')
        idx = (mask[:, :, 0] * 256 + mask[:, :, 1]) * 256 + mask[:, :, 2]
        return self.colormap2label[idx]


# 3. 增强函数
def get_voc_train_transform(size=256):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_voc_val_transform(size=256):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# 4. DataLoader 封装
def get_voc_seg_loaders(voc_root, batch_size=8, size=256, num_workers=4):
    train_set = VOCSegDataset(
        voc_root=voc_root,
        image_set='train',
        transform=get_voc_train_transform(size)
    )
    val_set = VOCSegDataset(
        voc_root=voc_root,
        image_set='val',
        transform=get_voc_val_transform(size)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# 5. 测试代码
if __name__ == '__main__':
    voc_root = 'path/to/your/dataset'  # 替换为你自己的路径
    train_loader, val_loader = get_voc_seg_loaders(voc_root, batch_size=4, size=256)

    for images, masks in train_loader:
        print('Image batch shape:', images.shape)  # [B, 3, H, W]
        print('Mask batch shape:', masks.shape)    # [B, H, W]
        print('Unique classes in mask:', torch.unique(masks))
        break
