# coding: '''zuoyong'''
# @Author : 杨国宝
# @Time : 2025/10/20 15:54


import numpy as np
import logging # 添加日志记录

class Evaluator(object):
    def __init__(self, num_class=2):
        self.num_class = num_class
        # 使用int64避免大矩阵求和时溢出
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.int64)

    def _get_class_wise_tp_fp_fn(self):
        """为每个类别计算 TP, FP, FN。"""
        # 确保计算时使用浮点数
        tp = np.diag(self.confusion_matrix).astype(np.float64)
        fp = (self.confusion_matrix.sum(axis=0) - tp).astype(np.float64)
        fn = (self.confusion_matrix.sum(axis=1) - tp).astype(np.float64)
        tn = (self.confusion_matrix.sum() - tp - fp - fn).astype(np.float64) # 计算 TN
        return tp, fp, fn, tn

    def Precision(self, epsilon=1e-8):
        """计算每个类别的精确率 (Precision)"""
        tp, fp, _, _ = self._get_class_wise_tp_fp_fn()
        precision = tp / (tp + fp + epsilon)
        return precision # (num_class,)

    def Recall(self, epsilon=1e-8):
        """计算每个类别的召回率 (Recall)"""
        tp, _, fn, _ = self._get_class_wise_tp_fp_fn()
        recall = tp / (tp + fn + epsilon)
        return recall # (num_class,)

    def F1(self, epsilon=1e-8):
        """计算每个类别的 F1 分数"""
        precision = self.Precision(epsilon)
        recall = self.Recall(epsilon)
        # 使用 np.where 避免 0/0 的情况
        f1_scores = np.where(
            (precision + recall) == 0,
            0.0,
            (2.0 * precision * recall) / (precision + recall + epsilon)
        )
        return f1_scores # (num_class,)

    def Intersection_over_Union(self, epsilon=1e-8):
        """计算每个类别的交并比 (IoU)"""
        tp, fp, fn, _ = self._get_class_wise_tp_fp_fn()
        iou = tp / (tp + fp + fn + epsilon)
        return iou # (num_class,)

    def OA(self, epsilon=1e-8): # Overall Accuracy
        correct_predictions = np.diag(self.confusion_matrix).sum()
        total_predictions = self.confusion_matrix.sum()
        # 避免总数为0的情况
        return correct_predictions / (total_predictions + epsilon) if total_predictions > 0 else 0.0

    def Kappa(self, epsilon=1e-8):
        po = self.OA(epsilon)
        sum_total = np.sum(self.confusion_matrix)
        if sum_total == 0: # 如果混淆矩阵为空
            return 0.0
        sum_row = np.sum(self.confusion_matrix, axis=1)
        sum_col = np.sum(self.confusion_matrix, axis=0)
        # 使用浮点数进行计算避免整数除法问题
        pe = np.sum(sum_row.astype(np.float64) * sum_col.astype(np.float64)) / (sum_total * sum_total)
        # 避免 (1 - pe) 为零
        kappa = (po - pe) / (1 - pe + epsilon)
        return kappa

    def Mean_Intersection_over_Union(self, epsilon=1e-8):
        """计算平均 IoU"""
        MIoU_array = self.Intersection_over_Union(epsilon)
        MIoU = np.nanmean(MIoU_array) # nanmean 会自动忽略 NaN 值
        return MIoU if not np.isnan(MIoU) else 0.0 # 如果所有IoU都是NaN，返回0

    # --- 私有方法 ---
    def _generate_matrix(self, gt_image, pre_image):
        """从单个图像对生成混淆矩阵"""
        # 确保输入是整数类型
        gt_image = gt_image.astype(np.int64)
        pre_image = pre_image.astype(np.int64)
        # 过滤掉标签中无效的值 (例如 > num_class 的值)
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        # 计算混淆矩阵的索引: label = num_class * ground_truth + prediction
        label = self.num_class * gt_image[mask] + pre_image[mask]
        # 计算每个索引的数量
        count = np.bincount(label, minlength=self.num_class ** 2)
        # 重塑为 num_class x num_class 的矩阵
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    # --- 公共方法 ---
    def add_batch(self, gt_image_batch, pre_image_batch):
        """将一个批次的预测和标签添加到混淆矩阵"""
        # 确保输入是 NumPy 数组
        if not isinstance(gt_image_batch, np.ndarray): gt_image_batch = np.array(gt_image_batch)
        if not isinstance(pre_image_batch, np.ndarray): pre_image_batch = np.array(pre_image_batch)

        # 确保数据类型正确
        gt_image_batch = gt_image_batch.astype(np.int64)
        pre_image_batch = pre_image_batch.astype(np.int64)

        if gt_image_batch.shape != pre_image_batch.shape:
             logging.warning(f"Evaluator: 批次标签和预测形状不匹配: {gt_image_batch.shape} vs {pre_image_batch.shape}. 跳过此批次。")
             return

        # 假设输入是 (B, H, W) 或 (B, 1, H, W)
        if gt_image_batch.ndim == 4 and gt_image_batch.shape[1] == 1:
            gt_image_batch = gt_image_batch.squeeze(1)
        if pre_image_batch.ndim == 4 and pre_image_batch.shape[1] == 1:
            pre_image_batch = pre_image_batch.squeeze(1)

        # 检查维度是否正确 (现在应该是 B, H, W)
        if gt_image_batch.ndim != 3:
             logging.warning(f"Evaluator: 输入的批次维度不正确: {gt_image_batch.ndim} (期望 3). 跳过此批次。")
             return

        # 逐个样本处理
        for i in range(gt_image_batch.shape[0]):
            self.confusion_matrix += self._generate_matrix(gt_image_batch[i], pre_image_batch[i])

    def reset(self):
        """重置混淆矩阵为零"""
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.int64)