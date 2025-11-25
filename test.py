# coding: '''zuoyong'''
# @Author : 杨国宝
# @Time : 2025/10/20 15:50
# coding: utf-8
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
    rows, cols = label_image_np.shape
    visualization_array = np.zeros((rows, cols, 3), dtype=np.uint8)
    color_tp = [0, 255, 0]  # 绿色: 正确检测 (True Positive)
    color_fp = [255, 255, 0]  # 黄色: 误检 (False Positive)
    color_fn = [255, 0, 0]  # 红色: 漏检 (False Negative)
    color_tn = [0, 0, 0]  # 黑色: 正确未检测 (True Negative)
    # 仅记录一次不匹配值的警告标志
    mismatched_values_logged = {} # 使用字典记录已警告的文件名

    for r in range(rows):
        for c in range(cols):
            label_val = label_image_np[r, c]
            pred_val = prediction_image_np[r, c]

            if pred_val == change_value and label_val == change_value:  # TP
                visualization_array[r, c] = color_tp
            elif pred_val == change_value and label_val == no_change_value:  # FP
                visualization_array[r, c] = color_fp
            elif pred_val == no_change_value and label_val == change_value:  # FN
                visualization_array[r, c] = color_fn
            elif pred_val == no_change_value and label_val == no_change_value:  # TN
                visualization_array[r, c] = color_tn
            else:
                # 获取当前文件名 (需要从 test_model 函数传递)
                current_filename = "未知文件" # 默认值
                # 检查 locals() 中是否有 filenames 和 idx
                # 注意: 这不是一个好的实践，最好将 filenames[idx] 作为参数传递
                # if 'filenames' in locals() and 'idx' in locals() and idx < len(filenames):
                #    current_filename = filenames[idx]

                # if current_filename not in mismatched_values_logged:
                #     logging.warning(
                #         f"图像 {current_filename} 中存在未预期的像素值组合。例如，在某个像素点：标签值={label_val}, 预测值={pred_val}。"
                #         f"请确保 change_value ({change_value}) 和 no_change_value ({no_change_value}) 设置正确。"
                #         "异常像素将以灰色显示。")
                #     mismatched_values_logged[current_filename] = True
                visualization_array[r, c] = [128, 128, 128] # 灰色标记
    return visualization_array


def test_model(test_loader, model, device, evaluator, opt):
    model.eval() # 设置为评估模式
    evaluator.reset()

    pred_save_dir = None
    compare_save_dir = None
    # 创建保存目录
    if opt.save_predictions or opt.save_comparisons:
        if not opt.results_path:
            logging.error("需要指定 --results_path 来保存预测或比较图。")
            return 0,0,0,0 # 返回默认值表示失败
        os.makedirs(opt.results_path, exist_ok=True) # 确保根目录存在

        if opt.save_predictions:
            pred_save_dir = os.path.join(opt.results_path, "predictions_binary") # 保存二值图
            os.makedirs(pred_save_dir, exist_ok=True)
            logging.info(f"二值预测结果将保存到: {pred_save_dir}")

        if opt.save_comparisons:
            compare_save_dir = os.path.join(opt.results_path, "comparisons_visual") # 保存彩色比较图
            os.makedirs(compare_save_dir, exist_ok=True)
            logging.info(f"可视化比较结果将保存到: {compare_save_dir}")

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Test")
    with torch.no_grad():
        for i, (t1_img, t2_img, target_mask_tensor, filenames) in pbar:
            t1_img = t1_img.to(device, non_blocking=True)
            t2_img = t2_img.to(device, non_blocking=True)
            target_mask_tensor = target_mask_tensor.to(device, non_blocking=True)

            # --- 模型推理 (评估模式只返回 logits) ---
            logits_final = model(t1_img, t2_img)
            # --- 模型推理结束 ---

            pred_prob = torch.sigmoid(logits_final)
            pred_binary_tensor = (pred_prob >= 0.5)

            # --- 评估 ---
            pred_binary_numpy_for_eval = pred_binary_tensor.cpu().numpy().astype(np.uint8)
            target_numpy_for_eval = target_mask_tensor.cpu().numpy().astype(np.uint8)
            # 确保 Evaluator 输入的是 (B, H, W) 或 (B, 1, H, W)
            if pred_binary_numpy_for_eval.shape[1] == 1:
                 pred_binary_numpy_for_eval = pred_binary_numpy_for_eval.squeeze(1)
            if target_numpy_for_eval.shape[1] == 1:
                 target_numpy_for_eval = target_numpy_for_eval.squeeze(1)
            evaluator.add_batch(target_numpy_for_eval, pred_binary_numpy_for_eval)
            # --- 评估结束 ---

            # --- 结果保存 ---
            for idx in range(pred_binary_tensor.shape[0]):
                current_filename_base = filenames[idx] # 获取不带后缀的文件名
                # (B, 1, H, W) -> (H, W) numpy uint8
                single_pred_numpy = pred_binary_tensor[idx, 0].cpu().numpy().astype(np.uint8)
                single_gt_numpy = target_mask_tensor[idx, 0].cpu().numpy().astype(np.uint8)

                # 保存二值预测图
                if pred_save_dir:
                    pred_img_pil = Image.fromarray(single_pred_numpy * 255, mode='L') # 0->黑, 1->白
                    save_name_pred = os.path.join(pred_save_dir, f"{current_filename_base}_pred.png")
                    try:
                        pred_img_pil.save(save_name_pred)
                    except Exception as e:
                        logging.error(f"保存预测图像失败 {save_name_pred}: {e}")

                # 保存彩色比较图
                if compare_save_dir:
                    comparison_np_array = create_comparison_visualization_array(
                        single_gt_numpy, single_pred_numpy, change_value=1, no_change_value=0
                    )
                    compare_img_pil = Image.fromarray(comparison_np_array, 'RGB')
                    save_name_compare = os.path.join(compare_save_dir, f"{current_filename_base}_compare.png")
                    try:
                        compare_img_pil.save(save_name_compare)
                    except Exception as e:
                        logging.error(f"保存比较图像失败 {save_name_compare}: {e}")
            # --- 结果保存结束 ---

    # --- 指标计算与保存 ---
    try:
        iou_test = evaluator.Intersection_over_Union()[1]
        f1_test = evaluator.F1()[1]
        pre_test = evaluator.Precision()[1]
        rec_test = evaluator.Recall()[1]
        oa_test = evaluator.OA()
        kappa_test = evaluator.Kappa()
        # 获取完整数组以保存
        iou_test_array = evaluator.Intersection_over_Union()
        f1_test_array = evaluator.F1()
        pre_test_array = evaluator.Precision()
        rec_test_array = evaluator.Recall()

    except Exception as e:
        logging.error(f"计算测试指标时出错: {e}. 指标设为 0。")
        iou_test, f1_test, pre_test, rec_test, oa_test, kappa_test = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        iou_test_array, f1_test_array, pre_test_array, rec_test_array = np.array([0.,0.]), np.array([0.,0.]), np.array([0.,0.]), np.array([0.,0.])

    logging.info("--- 测试结果 ---")
    logging.info(f"F1 Score (Change): {f1_test:.4f}")
    logging.info(f"IoU (Change):      {iou_test:.4f}")
    logging.info(f"Precision (Change):{pre_test:.4f}")
    logging.info(f"Recall (Change):   {rec_test:.4f}")
    logging.info(f"Overall Acc:     {oa_test:.4f}")
    logging.info(f"Kappa:           {kappa_test:.4f}")
    logging.info("--------------------")

    # 保存指标到 JSON 文件
    if opt.results_path:
        metrics_data = {
            "F1_Change": f1_test, "IoU_Change": iou_test,
            "Precision_Change": pre_test, "Recall_Change": rec_test,
            "OA": oa_test, "Kappa": kappa_test,
            "F1_AllClasses": f1_test_array.tolist(), "IoU_AllClasses": iou_test_array.tolist(),
            "Precision_AllClasses": pre_test_array.tolist(), "Recall_AllClasses": rec_test_array.tolist(),
            "Options": vars(opt)
        }
        metrics_filename = os.path.join(opt.results_path, "test_metrics.json")
        try:
            with open(metrics_filename, 'w') as f:
                json.dump(metrics_data, f, indent=4)
            logging.info(f"测试指标已保存到: {metrics_filename}")
        except Exception as e:
            logging.error(f"保存测试指标JSON文件失败 {metrics_filename}: {e}")

    return f1_test, iou_test, pre_test, rec_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="D&A-CDNet Testing")
    # --- 数据和路径 ---
    parser.add_argument('--dataset_root', type=str, default='./data', help='数据集根目录')
    parser.add_argument('--data_name', type=str, default='S2Looking', choices=['LEVIR', 'CDD', 'WHU', 'SYSU', 'S2Looking', 'DSIFN','custom'], help='数据集名称')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='训练好的 D&A-CDNet 模型权重文件路径 (.pth)')
    parser.add_argument('--results_path', type=str, default='./results/', help='测试结果保存根路径')

    # --- 测试参数 ---
    parser.add_argument('--batchsize', type=int, default=8, help='测试批次大小 (根据显存调整)') # 减小默认批大小
    parser.add_argument('--testsize', type=int, default=256, help='测试图像输入尺寸')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作进程数')
    parser.add_argument('--save_predictions', type=lambda x: (str(x).lower() == 'true'), default=True, help='是否保存二值预测图')
    parser.add_argument('--save_comparisons', type=lambda x: (str(x).lower() == 'true'), default=True, help='是否保存可视化比较图')


    parser.add_argument('--backbone', type=str, default='pvt_v2_b1', help='模型使用的主干网络 (需要与 checkpoint 匹配)')


    opt = parser.parse_args()


    current_time = time.strftime("%Y%m%d-%H%M%S")
    if opt.checkpoint_path and os.path.exists(opt.checkpoint_path):
        # 从检查点文件名提取基础信息，避免路径过长或特殊字符
        checkpoint_name_base = os.path.splitext(os.path.basename(opt.checkpoint_path))[0]
        # 可以进一步清理 checkpoint_name_base 中的非法字符
        valid_chars = "-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        checkpoint_name_cleaned = ''.join(c for c in checkpoint_name_base if c in valid_chars)
        if not checkpoint_name_cleaned: checkpoint_name_cleaned = "checkpoint" # 避免空名
    else:
        checkpoint_name_cleaned = "unknown_checkpoint"
        logging.warning(f"提供的检查点路径无效或不存在: {opt.checkpoint_path}")
        # exit(1) # 可以选择在此处退出

    # 更新 results_path 结构
    opt.results_path = os.path.join(opt.results_path, opt.data_name, f"test_{checkpoint_name_cleaned}_{current_time}")
    os.makedirs(opt.results_path, exist_ok=True)

    # 配置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(opt.results_path, 'testing_log.log')),
                            logging.StreamHandler()
                        ])
    logging.info("命令行参数:\n%s", json.dumps(vars(opt), indent=2))

    # --- 数据集路径 ---
    dataset_base_paths = {
        'LEVIR': os.path.join(opt.dataset_root, 'LEVIR-CD/'),
        'CDD': os.path.join(opt.dataset_root, 'CDD/'),
        'WHU': os.path.join(opt.dataset_root, 'WHU-CD/'),
        'SYSU': os.path.join(opt.dataset_root, 'SYSU-CD/'),
        'S2Looking': os.path.join(opt.dataset_root, 'S2Looking/'),
        'DSIFN': os.path.join(opt.dataset_root, 'DSIFN/'),
        'custom': opt.dataset_root
    }
    base_path_for_data = dataset_base_paths.get(opt.data_name, opt.dataset_root)
    if opt.data_name not in dataset_base_paths:
        logging.warning(f"数据集名称 {opt.data_name} 未在预设路径中。")

    opt.test_root = os.path.join(base_path_for_data, 'test/')
    logging.info(f"使用数据集: {opt.data_name}, 测试集路径: {opt.test_root}")

    if not os.path.isdir(opt.test_root):
        logging.error(f"错误: 测试目录不存在: {opt.test_root}")
        exit(1)

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # --- 数据加载器 ---
    logging.info("初始化测试数据加载器...")
    try:
        test_loader = data_loader.get_test_loader(opt.test_root, opt.batchsize, opt.testsize,
                                                  num_workers=opt.num_workers, shuffle=False, pin_memory=True)
        logging.info(f"测试数据加载器完成。测试批次数: {len(test_loader)}")
    except Exception as e:
        logging.error(f"测试数据加载器初始化失败: {e}")
        exit(1)

    # --- 评估器 ---
    evaluator = Evaluator(num_class=2)

    # --- 模型加载 ---
    logging.info(f"初始化模型 D&A-CDNet (Backbone: {opt.backbone})...")
    try:
        # pretrained=False 因为我们要加载自己的 checkpoint
        model = D_A_CDNet(backbone_name=opt.backbone, pretrained=False).to(device)
    except Exception as e:
        logging.error(f"模型结构初始化失败: {e}")
        exit(1)

    if not opt.checkpoint_path or not os.path.isfile(opt.checkpoint_path):
        logging.error(f"错误: 检查点文件未找到或路径无效: {opt.checkpoint_path}")
        exit(1)

    logging.info(f"加载模型权重从: {opt.checkpoint_path}")
    try:
        checkpoint = torch.load(opt.checkpoint_path, map_location=device)
        state_dict_to_load = checkpoint.get('model_state_dict', checkpoint)


        is_parallel_state = all(key.startswith('module.') for key in state_dict_to_load.keys())
        if is_parallel_state and (device.type == 'cpu' or torch.cuda.device_count() == 1):
            logging.info("检测到 DataParallel 模型权重，移除 'module.' 前缀...")
            new_state_dict = {k[len('module.'):]: v for k, v in state_dict_to_load.items()}
            model.load_state_dict(new_state_dict)
        elif not is_parallel_state and device.type == 'cuda' and torch.cuda.device_count() > 1:

             logging.info("权重为单 GPU 保存，但当前为多 GPU 环境，使用 DataParallel 包装模型...")
             model = torch.nn.DataParallel(model) # 先包装再加载
             model.load_state_dict(state_dict_to_load) # 加载到包装后的模型
        else:

            model.load_state_dict(state_dict_to_load)

        logging.info("模型权重加载成功。")
    except Exception as e:
        logging.error(f"加载模型权重失败: {e}")
        exit(1)

    if device.type == 'cuda' and torch.cuda.device_count() > 1 and not isinstance(model, torch.nn.DataParallel):

        pass # 上面加载逻辑已处理多GPU包装

    # --- 开始测试 ---
    logging.info("开始测试...")
    start_time_test = time.time()
    # 调用测试函数
    test_f1, test_iou, test_pre, test_rec = test_model(test_loader, model, device, evaluator, opt)
    total_testing_time_seconds = time.time() - start_time_test
    logging.info(
        f"测试完成！总耗时: {total_testing_time_seconds // 3600:.0f}h "
        f"{(total_testing_time_seconds % 3600) // 60:.0f}m "
        f"{total_testing_time_seconds % 60:.2f}s"
    )
    logging.info(f"最终测试指标: F1={test_f1:.4f}, IoU={test_iou:.4f}, Precision={test_pre:.4f}, Recall={test_rec:.4f}")
    logging.info(f"结果保存在: {opt.results_path}")