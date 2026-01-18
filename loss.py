# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


# ... (dice_bce_loss 和 TemporalConsistencyLoss 保持不变) ...
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCEWithLogitsLoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1e-5
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def forward(self, y_pred, y_true):
        loss_bce = self.bce_loss(y_pred, y_true)
        y_pred_sigmoid = torch.sigmoid(y_pred)
        loss_dice = self.soft_dice_loss(y_true, y_pred_sigmoid)
        return loss_bce + loss_dice


class TemporalConsistencyLoss(nn.Module):
    def __init__(self):
        super(TemporalConsistencyLoss, self).__init__()

    def forward(self, f_t1, f_t2):
        return torch.mean(torch.abs(f_t1 - f_t2))


# --- [修改] 对比分离损失 (修复维度报错) ---
class ContrastiveSeparationLoss(nn.Module):
    """
    对比分离损失 (L_cs)
    修复了 F.pairwise_distance 在 4D 张量上维度计算错误的问题，
    改为手动计算欧氏距离。
    """

    def __init__(self, margin=2.0):
        super(ContrastiveSeparationLoss, self).__init__()
        self.margin = margin

    def forward(self, f_t1, f_t2, label):
        # f_t1, f_t2: [B, C, H, W]
        # label: [B, 1, H, W]

        # 1. 手动计算欧氏距离 (Euclidean Distance)
        # 显式指定 dim=1 (通道维度) 进行求和，保证输出为 [B, 1, H, W]
        diff = f_t1 - f_t2
        dist_sq = torch.sum(torch.pow(diff, 2), dim=1, keepdim=True)  # 距离平方
        dist = torch.sqrt(dist_sq + 1e-8)  # 距离 (加 1e-8 防止梯度爆炸)

        # 2. 展平以便计算 mean (Flatten to [B, N])
        # 使用 label.shape[0] 获取 batch size
        dist_sq = dist_sq.view(label.shape[0], -1)
        dist = dist.view(label.shape[0], -1)
        label = label.view(label.shape[0], -1)

        # 3. 计算 Loss
        # 不变区域 (Label=0): 最小化距离平方 -> 0.5 * D^2
        loss_unchanged = torch.sum((1 - label) * 0.5 * dist_sq) / (torch.sum(1 - label) + 1e-8)

        # 变化区域 (Label=1): 最大化距离 -> 0.5 * {max(0, m - D)}^2
        loss_changed = torch.sum(label * 0.5 * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)) / (
                    torch.sum(label) + 1e-8)

        loss = loss_unchanged + loss_changed
        return loss


class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()

    def forward(self, f_ci, f_cs):
        cos_sim = F.cosine_similarity(f_ci, f_cs, dim=1, eps=1e-8)
        abs_cos_sim = torch.abs(cos_sim)
        loss = torch.mean(abs_cos_sim)
        return loss