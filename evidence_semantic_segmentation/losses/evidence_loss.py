import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- 基础函数 ----------------------

def softplus_evidence(y):
    return F.softplus(y)


def to_one_hot(label, num_classes):
    """
    label: (B, H, W) → (B, num_classes, H, W)
    """
    return F.one_hot(label.long(), num_classes).permute(0, 3, 1, 2).float()


def kl_divergence(alpha, num_classes):
    ones = torch.ones_like(alpha)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        * (torch.digamma(alpha) - torch.digamma(sum_alpha))
    ).sum(dim=1, keepdim=True)
    return first_term + second_term


def loglikelihood_loss(y, alpha):
    S = torch.sum(alpha, dim=1, keepdim=True)
    err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    return err + var


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step):
    loglikelihood = loglikelihood_loss(y, alpha)
    annealing_coef = min(1.0, epoch_num / annealing_step)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1.0, epoch_num / annealing_step)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return A + kl_div


# ---------------------- 三种证据损失类 ----------------------

class EDLMSELoss(nn.Module):
    def __init__(self, num_classes, annealing_step=10):
        super(EDLMSELoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step

    def forward(self, output, target, epoch_num):
        target = to_one_hot(target, self.num_classes)
        evidence = softplus_evidence(output)
        alpha = evidence + 1
        return torch.mean(
            mse_loss(target, alpha, epoch_num, self.num_classes, self.annealing_step)
        )


class EDLLogLoss(nn.Module):
    def __init__(self, num_classes, annealing_step=10):
        super(EDLLogLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step

    def forward(self, output, target, epoch_num):
        target = to_one_hot(target, self.num_classes)
        evidence = softplus_evidence(output)
        alpha = evidence + 1
        return torch.mean(
            edl_loss(torch.log, target, alpha, epoch_num, self.num_classes, self.annealing_step)
        )


class EDLDigammaLoss(nn.Module):
    def __init__(self, num_classes, annealing_step=10):
        super(EDLDigammaLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step

    def forward(self, output, target, epoch_num):
        target = to_one_hot(target, self.num_classes)
        evidence = softplus_evidence(output)
        alpha = evidence + 1
        return torch.mean(
            edl_loss(torch.digamma, target, alpha, epoch_num, self.num_classes, self.annealing_step)
        )

if __name__ == "__main__":
    torch.manual_seed(42)

    # 假数据
    batch_size, num_classes, H, W = 1, 2, 4, 4
    output = torch.randn(batch_size, num_classes, H, W)   # 网络 raw logits
    target = torch.randint(0, num_classes, (batch_size, H, W))  # 标签

    # 损失
    loss_fn = EDLMSELoss(num_classes=num_classes, annealing_step=10)
    epoch_num = 5
    loss = loss_fn(output, target, epoch_num)
    print("EDL loss:", loss.item())