import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, beta=1.0, smooth=1e-5, num_classes=21, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes
        self.ignore_index = ignore_index  # 可选：支持忽略某类标签

    def forward(self, inputs, target):
        """
        inputs: (N, C, H, W) - 模型输出，softmax前
        target: (N, H, W) - 整数标签
        """
        n, c, h, w = inputs.size()

        # 将 target 转换为 one-hot 编码
        target_onehot = F.one_hot(target.long(), num_classes=self.num_classes)  # (N, H, W, C)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        # 可选：排除 ignore_index 类
        if self.ignore_index is not None and self.ignore_index < self.num_classes:
            mask = torch.ones(self.num_classes, device=inputs.device, dtype=torch.bool)
            mask[self.ignore_index] = False
            inputs = inputs[:, mask, :, :]
            target_onehot = target_onehot[:, mask, :, :]

        # softmax + flatten
        inputs_soft = F.softmax(inputs, dim=1)
        inputs_flat = inputs_soft.contiguous().view(n, -1, inputs_soft.shape[1])  # (N, H*W, C)
        targets_flat = target_onehot.contiguous().view(n, -1, target_onehot.shape[1])  # (N, H*W, C)

        # 计算 dice loss
        tp = torch.sum(targets_flat * inputs_flat, dim=[0, 1])
        fp = torch.sum(inputs_flat, dim=[0, 1]) - tp
        fn = torch.sum(targets_flat, dim=[0, 1]) - tp

        score = ((1 + self.beta ** 2) * tp + self.smooth) / \
                ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)

        dice_loss = 1 - score.mean()
        return dice_loss


# -------------------------------
# Focal Loss 封装为类
# -------------------------------
class FocalLoss(nn.Module):
    def __init__(self, cls_weights=None, num_classes=21, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.cls_weights = cls_weights
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, target):
        n, c, h, w = inputs.size()

        # reshape
        temp_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, c)
        temp_target = target.view(-1)

        # CrossEntropy (不平均)
        logpt = -F.cross_entropy(temp_inputs, temp_target, weight=self.cls_weights,
                                 ignore_index=self.num_classes, reduction='none')
        pt = torch.exp(logpt)

        # Focal Loss computation
        if self.alpha is not None:
            logpt = self.alpha * logpt
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_weights=None, num_classes=21):
        """
        cls_weights: Tensor of shape (num_classes,) or None
        ignore_index: int or None, the label to ignore in loss computation
        """
        super(CrossEntropyLoss, self).__init__()
        self.cls_weights = cls_weights
        self.num_classes = num_classes

    def forward(self, inputs, target):
        """
        inputs: (N, C, H, W) - raw logits
        target: (N, H, W) - label indices
        """
        n, c, h, w = inputs.size()

        # reshape：转为二维用于 loss 计算
        temp_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, c)  # (N*H*W, C)
        temp_target = target.view(-1)  # (N*H*W,)

        loss = F.cross_entropy(
            temp_inputs,
            temp_target,
            weight=self.cls_weights
        )
        return loss


class CEDiCELoss(nn.Module):
    def __init__(self, cls_weights=None, num_classes=21):
        super(CEDiCELoss, self).__init__()
        self.cls_weights = cls_weights
        self.num_classes = num_classes
        self.CELoss = CrossEntropyLoss(cls_weights=cls_weights, num_classes=num_classes)
        self.dice_loss = DiceLoss(beta=1.0, smooth=1e-5, num_classes=self.num_classes, ignore_index=None)

    def forward(self, inputs, target):
        x1 = self.CELoss(inputs, target)
        x2 = self.dice_loss(inputs, target)
        loss = x1 + x2
        return loss

