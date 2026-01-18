# coding: utf-8
# @Author : 杨国宝
# @Time : 2025/10/20 15:50

import json
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import logging

import data_loader
from metrics import Evaluator
from model_da_cdnet import D_A_CDNet


def create_comparison_visualization_array(label_image_np, prediction_image_np, change_value=1, no_change_value=0):
    """
    生成对比可视化图像 (TP/FP/FN/TN)
    """
    rows, cols = label_image_np.shape
    visualization_array = np.zeros((rows, cols, 3), dtype=np.uint8)

    # 定义颜色 (RGB)
    color_tp = [0, 255, 0]  # 绿色: 正确检测 (True Positive)
    color_fp = [255, 255, 0]  # 黄色: 误检 (False Positive) -> 虚警
    color_fn = [255, 0, 0]  # 红色: 漏检 (False Negative)
    color_tn = [0, 0, 0]  # 黑色: 正确背景 (True Negative)

    # 向量化操作代替双重循环，大幅提升速度
    # 1. TP: Pred=1, Label=1
    mask_tp = (prediction_image_np == change_value) & (label_image_np == change_value)
    visualization_array[mask_tp] = color_tp

    # 2. FP: Pred=1, Label=0
    mask_fp = (prediction_image_np == change_value) & (label_image_np == no_change_value)
    visualization_array[mask_fp] = color_fp

    # 3. FN: Pred=0, Label=1
    mask_fn = (prediction_image_np == no_change_value) & (label_image_np == change_value)
    visualization_array[mask_fn] = color_fn

    # 4. TN: Pred=0, Label=0 (默认黑色，可省略，或显式赋值)
    # mask_tn = (prediction_image_np == no_change_value) & (label_image_np == no_change_value)
    # visualization_array[mask_tn] = color_tn

    return visualization_array


def test_model(test_loader, model, device, evaluator, opt):
    model.eval()  # 必须调用，确保 forward 只返回 logits
    evaluator.reset()

    pred_save_dir = None
    compare_save_dir = None

    # 创建保存目录
    if opt.save_predictions or opt.save_comparisons:
        if not opt.results_path:
            logging.error("需要指定 --results_path 来保存预测或比较图。")
            return 0, 0, 0, 0
        os.makedirs(opt.results_path, exist_ok=True)

        if opt.save_predictions:
            pred_save_dir = os.path.join(opt.results_path, "predictions_binary")
            os.makedirs(pred_save_dir, exist_ok=True)

        if opt.save_comparisons:
            compare_save_dir = os.path.join(opt.results_path, "comparisons_visual")
            os.makedirs(compare_save_dir, exist_ok=True)

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Test")

    with torch.no_grad():
        for i, (t1_img, t2_img, target_mask_tensor, filenames) in pbar:
            t1_img = t1_img.to(device, non_blocking=True)
            t2_img = t2_img.to(device, non_blocking=True)
            target_mask_tensor = target_mask_tensor.to(device, non_blocking=True)

            # --- 模型推理 ---
            # 在 eval 模式下，model_da_cdnet 只返回 logits
            logits_final = model(t1_img, t2_img)

            pred_prob = torch.sigmoid(logits_final)
            pred_binary_tensor = (pred_prob >= 0.5)

            # --- 评估指标累积 ---
            # 转为 numpy
            pred_binary_np = pred_binary_tensor.cpu().numpy().astype(np.uint8)
            target_np = target_mask_tensor.cpu().numpy().astype(np.uint8)

            # 压缩通道维度 (B, 1, H, W) -> (B, H, W)
            if pred_binary_np.ndim == 4: pred_binary_np = pred_binary_np.squeeze(1)
            if target_np.ndim == 4: target_np = target_np.squeeze(1)

            evaluator.add_batch(target_np, pred_binary_np)

            # --- 结果图像保存 ---
            if opt.save_predictions or opt.save_comparisons:
                batch_size = pred_binary_tensor.shape[0]
                for idx in range(batch_size):
                    current_filename = filenames[idx]

                    # 获取单张图 (H, W)
                    single_pred = pred_binary_np[idx]
                    single_gt = target_np[idx]

                    # 1. 保存二值预测图
                    if pred_save_dir:
                        # 0 -> 0, 1 -> 255
                        pred_img_pil = Image.fromarray(single_pred * 255, mode='L')
                        save_path = os.path.join(pred_save_dir,
                                                 f"{current_filename}.png")  # 建议去掉 _pred 后缀以便某些评测工具直接读取，或保留
                        pred_img_pil.save(save_path)

                    # 2. 保存彩色对比图 (TP/FP/FN)
                    if compare_save_dir:
                        vis_array = create_comparison_visualization_array(single_gt, single_pred)
                        vis_img_pil = Image.fromarray(vis_array, 'RGB')
                        save_path = os.path.join(compare_save_dir, f"{current_filename}_vis.png")
                        vis_img_pil.save(save_path)

    # --- 计算最终指标 ---
    iou_test = evaluator.Intersection_over_Union()[1]
    f1_test = evaluator.F1()[1]
    pre_test = evaluator.Precision()[1]
    rec_test = evaluator.Recall()[1]
    oa_test = evaluator.OA()
    kappa_test = evaluator.Kappa()

    logging.info("------------------------------------------------")
    logging.info(f"[Test Summary] F1: {f1_test:.4f} | IoU: {iou_test:.4f} | OA: {oa_test:.4f}")
    logging.info(f"Precision: {pre_test:.4f} | Recall: {rec_test:.4f} | Kappa: {kappa_test:.4f}")
    logging.info("------------------------------------------------")

    # 保存 JSON 指标
    if opt.results_path:
        metrics_data = {
            "F1": f1_test, "IoU": iou_test,
            "Precision": pre_test, "Recall": rec_test,
            "OA": oa_test, "Kappa": kappa_test,
            "Args": vars(opt)
        }
        with open(os.path.join(opt.results_path, "test_metrics_ago.json"), 'w') as f:
            json.dump(metrics_data, f, indent=4)

    return f1_test, iou_test, pre_test, rec_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="D&A-CDNet Testing")

    # 基础参数
    parser.add_argument('--dataset_root', type=str, default='./data', help='数据集根目录')
    parser.add_argument('--data_name', type=str, default='WHU-CD', help='数据集名称')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='模型权重路径 (.pth)')
    parser.add_argument('--results_path', type=str, default='./results/', help='结果保存路径')

    # 模型参数
    parser.add_argument('--backbone', type=str, default='pvt_v2_b1', help='Backbone名称')

    # 测试参数
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--testsize', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_predictions', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--save_comparisons', type=lambda x: (str(x).lower() == 'true'), default=True)

    opt = parser.parse_args()

    # --- 路径处理 ---
    checkpoint_name = os.path.splitext(os.path.basename(opt.checkpoint_path))[0]
    opt.results_path = os.path.join(opt.results_path, opt.data_name, f"Test_{checkpoint_name}")
    os.makedirs(opt.results_path, exist_ok=True)

    # 日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(opt.results_path, 'test_log.txt')),
                            logging.StreamHandler()
                        ])
    logging.info(f"Test Configuration:\n{json.dumps(vars(opt), indent=2)}")

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 数据加载 ---
    # 自动推断测试路径
    test_root = os.path.join(opt.dataset_root, opt.data_name, 'test/')
    # 兼容性回退
    if not os.path.isdir(test_root):
        test_root = os.path.join(opt.dataset_root, 'test/')

    if not os.path.exists(test_root):
        logging.error(f"Test data root not found: {test_root}")
        exit(1)

    test_loader = data_loader.get_test_loader(test_root, opt.batchsize, opt.testsize,
                                              num_workers=opt.num_workers, shuffle=False, pin_memory=True)

    # --- 评估器 ---
    evaluator = Evaluator(num_class=2)

    # --- 模型加载 ---
    # 注意: pretrained=False 因为我们加载的是自己微调后的权重
    model = D_A_CDNet(backbone_name=opt.backbone, pretrained=False).to(device)

    if os.path.isfile(opt.checkpoint_path):
        logging.info(f"Loading checkpoint: {opt.checkpoint_path}")
        checkpoint = torch.load(opt.checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # 智能处理 DataParallel 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # 加载权重
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing: logging.warning(f"Missing keys: {missing}")
        if unexpected: logging.warning(f"Unexpected keys: {unexpected}")
    else:
        logging.error(f"Checkpoint not found: {opt.checkpoint_path}")
        exit(1)

    # --- 开始测试 ---
    test_model(test_loader, model, device, evaluator, opt)