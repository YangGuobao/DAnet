# coding: utf-8
# @Author : YGB
# @Time : 2025/10/20

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 主任务损失 (Dice + BCE) ---
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        # 使用 BCEWithLogitsLoss 以提高数值稳定性 (它直接接收 logits)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def soft_dice_coeff(self, y_true, y_pred_prob):
        smooth = 1e-5 # 平滑项，防止分母为零
        # 确保 target 是 float 类型
        y_true = y_true.float()

        # 展平 tensors 以进行计算
        y_true_f = y_true.contiguous().view(-1)
        y_pred_f = y_pred_prob.contiguous().view(-1)

        intersection = torch.sum(y_true_f * y_pred_f) # 计算交集
        score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth) # Dice 系数公式
        return score

    def soft_dice_loss(self, y_true, y_pred_prob):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred_prob) # Dice 损失
        return loss

    def forward(self, y_pred_logits, y_true):
        # 对 logits 应用 sigmoid 得到概率，用于 Dice loss
        y_pred_prob = torch.sigmoid(y_pred_logits)

        # 确保 y_true 类型和形状匹配
        if y_true.dtype != y_pred_prob.dtype:
            y_true = y_true.type_as(y_pred_prob)
        if y_true.ndim == y_pred_prob.ndim - 1 and y_pred_prob.shape[1] == 1:
            y_true = y_true.unsqueeze(1) # 添加通道维度
        elif y_true.shape != y_pred_prob.shape:
             # 如果尺寸不匹配 (例如，模型在低分辨率预测)，进行插值
            y_pred_logits = F.interpolate(y_pred_logits, size=y_true.shape[2:], mode='bilinear', align_corners=False)
            y_pred_prob = torch.sigmoid(y_pred_logits) # 插值后重新计算概率

        # BCEWithLogitsLoss 直接接收 logits
        loss_bce = self.bce_loss(y_pred_logits, y_true)
        loss_dice = self.soft_dice_loss(y_true, y_pred_prob)

        return loss_bce + loss_dice # 返回组合损失

# --- 时序一致性损失 (L_tc) ---
class TemporalConsistencyLoss(nn.Module):
    """
    计算两个特征图之间的 L1 损失以强制一致性。
    应用于变化无关特征 (F_ci_t1, F_ci_t2)。
    """
    def __init__(self):
        super(TemporalConsistencyLoss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean') # 使用 L1 损失

    def forward(self, feature_t1, feature_t2):
        """
        Args:
            feature_t1 (torch.Tensor): 时相1的变化无关特征。
            feature_t2 (torch.Tensor): 时相2的变化无关特征。
        Returns:
            torch.Tensor: L1 损失值。
        """
        # 确保输入是浮点类型
        if feature_t1.dtype != torch.float32: feature_t1 = feature_t1.float()
        if feature_t2.dtype != torch.float32: feature_t2 = feature_t2.float()

        return self.l1_loss(feature_t1, feature_t2)

# --- 对比分离损失 (L_cs) ---
class ContrastiveSeparationLoss(nn.Module):
    """
    在变化区域推远变化敏感特征，在不变区域拉近它们。
    应用于变化敏感特征 (F_cs_t1, F_cs_t2)。
    """
    def __init__(self, margin=1.0):
        """
        Args:
            margin (float): 变化区域特征之间期望的最小距离。
        """
        super(ContrastiveSeparationLoss, self).__init__()
        self.margin = margin
        # 使用无规约的L1损失，以便按像素加权
        self.l1_loss_no_reduction = nn.L1Loss(reduction='none')

    def forward(self, feature_t1, feature_t2, target):
        """
        Args:
            feature_t1 (torch.Tensor): 时相1的变化敏感特征。
            feature_t2 (torch.Tensor): 时相2的变化敏感特征。
            target (torch.Tensor): 真实变化标签 (1代表变化, 0代表不变)。
                                   形状: (B, 1, H, W) 或 (B, H, W)。
        Returns:
            torch.Tensor: 对比分离损失值。
        """
        # 确保输入是浮点类型
        if feature_t1.dtype != torch.float32: feature_t1 = feature_t1.float()
        if feature_t2.dtype != torch.float32: feature_t2 = feature_t2.float()

        # 确保 target 是 float 类型并具有与特征图匹配的空间维度
        target = target.float()
        if target.ndim == feature_t1.ndim - 1: # 如果缺少通道维度，则添加
             target = target.unsqueeze(1)
        if target.shape[2:] != feature_t1.shape[2:]:
            # 如果空间尺寸不匹配，使用最近邻插值调整 target 尺寸
            target = F.interpolate(target, size=feature_t1.shape[2:], mode='nearest')

        # 计算像素级的 L1 距离
        distance = self.l1_loss_no_reduction(feature_t1, feature_t2)
        # 在通道维度上取平均距离，保留空间维度
        distance_spatial = distance.mean(dim=1, keepdim=True)

        # 不变区域损失 (target=0): 拉近特征 (最小化距离)
        loss_unchanged = distance_spatial * (1 - target)
        # 仅对不变像素计算平均损失，避免除以零
        num_unchanged = torch.sum(1 - target)
        # 仅当存在不变像素时才计算损失的平均值
        loss_unchanged_mean = torch.sum(loss_unchanged) / (num_unchanged + 1e-6) if num_unchanged > 0 else torch.tensor(0.0, device=feature_t1.device)


        # 变化区域损失 (target=1): 推远特征，直到距离达到 margin
        # 使用 ReLU(margin - distance) 计算推力
        loss_changed = F.relu(self.margin - distance_spatial) * target
        # 仅对变化像素计算平均损失，避免除以零
        num_changed = torch.sum(target)
        # 仅当存在变化像素时才计算损失的平均值
        loss_changed_mean = torch.sum(loss_changed) / (num_changed + 1e-6) if num_changed > 0 else torch.tensor(0.0, device=feature_t1.device)

        # 合并损失 (可以考虑加权，但这里简单相加)
        total_loss = loss_unchanged_mean + loss_changed_mean
        return total_loss