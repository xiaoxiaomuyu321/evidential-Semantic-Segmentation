import torch.nn as nn
from evidence_semantic_segmentation.losses.evidence_loss import EDLMSELoss, EDLLogLoss, EDLDigammaLoss
from evidence_semantic_segmentation.losses.base import FocalLoss, CrossEntropyLoss, DiceLoss, CEDiCELoss


class SegLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 loss_type: str = "ce",
                 cls_weights=None,
                 annealing_step: int = 10,
                 ignore_index: int = None):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "ce":
            self.loss = CrossEntropyLoss(cls_weights=cls_weights, num_classes=num_classes, ignore_index=ignore_index)
        elif loss_type == "dice":
            self.loss = DiceLoss(beta=1.0, smooth=1e-5, num_classes=num_classes, ignore_index=ignore_index)
        elif loss_type == "focal":
            self.loss = FocalLoss(cls_weights=cls_weights, num_classes=num_classes, alpha=0.5, gamma=2.0)
        elif loss_type == "ce_dice":
            self.loss = CEDiCELoss(cls_weights=cls_weights, num_classes=num_classes)
        elif loss_type == "edl_mse":
            self.loss = EDLMSELoss(num_classes=num_classes, annealing_step=annealing_step)
        elif loss_type == "edl_log":
            self.loss = EDLLogLoss(num_classes=num_classes, annealing_step=annealing_step)
        elif loss_type == "edl_digamma":
            self.loss = EDLDigammaLoss(num_classes=num_classes, annealing_step=annealing_step)
        else:
            raise ValueError(f"Unsupported loss type '{loss_type}'.")

    def forward(self, output, target, epoch_num=None):
        if self.loss_type in ["ce", "dice", "focal", "ce_dice"]:
            return self.loss(output, target)
        else:
            return self.loss(output, target, epoch_num)


