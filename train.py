# coding: utf-8
import json
import os
import torch
import torch.nn as nn # 导入 nn 以便使用 DataParallel
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import time
import argparse
import logging

# --- 导入自定义模块 ---
import data_loader
# 导入所有需要的损失函数
from loss import dice_bce_loss, TemporalConsistencyLoss, ContrastiveSeparationLoss
from metrics import Evaluator # 使用 metrics.py 中的 Evaluator

from model_da_cdnet import D_A_CDNet # 注意文件名和类名

# --- 辅助函数 ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保cudnn的确定性，可能会牺牲一些性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(train_loader, model, criterion_main, criterion_tc, criterion_cs,
                    optimizer, epoch, num_epochs, eva_train, device, opt):
    model.train() # 设置模型为训练模式
    epoch_loss = 0.0
    num_batches = len(train_loader)
    eva_train.reset()


    aux_loss_scales = [2, 3] # 对应 features_t1[2], features_t1[3] 等

    pbar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch}/{num_epochs} [T]")
    for i, (t1_img, t2_img, target_mask) in pbar:
        t1_img = t1_img.to(device, non_blocking=True)
        t2_img = t2_img.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        # 模型前向传播 (训练模式下返回 logits 和解耦特征)
        logits_final, decoupled_features = model(t1_img, t2_img)

        # 1. 计算主任务损失
        loss_main = criterion_main(logits_final, target_mask)

        # 2. 计算辅助损失 (时序一致性 L_tc 和 对比分离 L_cs)
        loss_tc = torch.tensor(0.0, device=device)
        loss_cs = torch.tensor(0.0, device=device)
        num_aux_losses = 0

        for scale_idx in aux_loss_scales:
            # 检查特征是否存在 (以防 backbone 输出层数不同)
            ci_t1_key = f'ci{scale_idx+1}_t1' # e.g., ci3_t1 for scale_idx=2
            ci_t2_key = f'ci{scale_idx+1}_t2'
            cs_t1_key = f'cs{scale_idx+1}_t1'
            cs_t2_key = f'cs{scale_idx+1}_t2'

            if all(k in decoupled_features for k in [ci_t1_key, ci_t2_key, cs_t1_key, cs_t2_key]):
                # 计算 L_tc
                loss_tc += criterion_tc(decoupled_features[ci_t1_key], decoupled_features[ci_t2_key])
                # 计算 L_cs (需要将 target_mask 缩放到对应尺度)
                target_mask_resized = F.interpolate(target_mask,
                                                    size=decoupled_features[cs_t1_key].shape[2:],
                                                    mode='nearest') # 标签用 nearest
                loss_cs += criterion_cs(decoupled_features[cs_t1_key],
                                        decoupled_features[cs_t2_key],
                                        target_mask_resized)
                num_aux_losses += 1

        # 平均辅助损失 (如果计算了的话)
        if num_aux_losses > 0:
            loss_tc /= num_aux_losses
            loss_cs /= num_aux_losses

        # 3. 计算总损失
        total_loss = loss_main + opt.lambda_tc * loss_tc + opt.lambda_cs * loss_cs

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item() # 记录总损失

        # 计算训练指标 (基于最终预测 logits_final)
        with torch.no_grad():
            pred_prob = torch.sigmoid(logits_final)
            pred_binary = (pred_prob >= 0.5).cpu().numpy().astype(np.uint8)
            target_numpy = target_mask.cpu().numpy().astype(np.uint8)
            # 确保 Evaluator 输入的是 (B, H, W) 或 (B, 1, H, W)
            if pred_binary.shape[1] == 1:
                 pred_binary = pred_binary.squeeze(1)
            if target_numpy.shape[1] == 1:
                 target_numpy = target_numpy.squeeze(1)
            eva_train.add_batch(target_numpy, pred_binary)

        pbar.set_postfix(Loss=f"{total_loss.item():.4f}", Lmain=f"{loss_main.item():.4f}", Ltc=f"{loss_tc.item():.4f}", Lcs=f"{loss_cs.item():.4f}")

    avg_epoch_loss = epoch_loss / num_batches

    # 计算 epoch 指标
    try:
        # 使用 metrics.py 中定义的更详细的方法
        iou_train = eva_train.Intersection_over_Union()[1] # Index 1 for change class
        f1_train = eva_train.F1()[1]
        pre_train = eva_train.Precision()[1]
        rec_train = eva_train.Recall()[1]
        oa_train = eva_train.OA()
        kappa_train = eva_train.Kappa()
    except Exception as e:
        logging.error(f"计算训练指标时出错: {e}")
        iou_train, f1_train, pre_train, rec_train, oa_train, kappa_train = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    logging.info(f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_epoch_loss:.4f}, F1: {f1_train:.4f}, IoU: {iou_train:.4f}, Pre: {pre_train:.4f}, Rec: {rec_train:.4f}, OA: {oa_train:.4f}, Kappa: {kappa_train:.4f}")
    return avg_epoch_loss, f1_train, iou_train


def validate_one_epoch(val_loader, model, criterion_main, epoch, num_epochs, eva_val, device, opt):
    model.eval() # 设置模型为评估模式
    epoch_loss = 0.0
    num_batches = len(val_loader)
    eva_val.reset()

    pbar = tqdm(enumerate(val_loader), total=num_batches, desc=f"Epoch {epoch}/{num_epochs} [V]")
    with torch.no_grad():
        for i, data_batch in pbar:
            # 处理可能的 data_loader 返回值 (带文件名或不带)
            if len(data_batch) == 4:
                t1_img, t2_img, target_mask, _ = data_batch # 忽略文件名
            elif len(data_batch) == 3:
                t1_img, t2_img, target_mask = data_batch
            else:
                logging.warning(f"验证数据加载器返回了意外数量的元素: {len(data_batch)}")
                continue # 跳过这个批次

            t1_img = t1_img.to(device, non_blocking=True)
            t2_img = t2_img.to(device, non_blocking=True)
            target_mask = target_mask.to(device, non_blocking=True)

            # 模型前向传播 (评估模式下只返回 logits)
            logits_final = model(t1_img, t2_img)

            # 只计算主任务损失用于监控
            loss_main = criterion_main(logits_final, target_mask)
            epoch_loss += loss_main.item()

            # 计算验证指标
            pred_prob = torch.sigmoid(logits_final)
            pred_binary = (pred_prob >= 0.5).cpu().numpy().astype(np.uint8)
            target_numpy = target_mask.cpu().numpy().astype(np.uint8)
            # 确保 Evaluator 输入的是 (B, H, W) 或 (B, 1, H, W)
            if pred_binary.shape[1] == 1:
                 pred_binary = pred_binary.squeeze(1)
            if target_numpy.shape[1] == 1:
                 target_numpy = target_numpy.squeeze(1)
            eva_val.add_batch(target_numpy, pred_binary)

            pbar.set_postfix(Loss=f"{loss_main.item():.4f}")

    avg_epoch_loss = epoch_loss / num_batches

    # 计算 epoch 指标
    try:
        iou_val = eva_val.Intersection_over_Union()[1]
        f1_val = eva_val.F1()[1]
        pre_val = eva_val.Precision()[1]
        rec_val = eva_val.Recall()[1]
        oa_val = eva_val.OA()
        kappa_val = eva_val.Kappa()
    except Exception as e:
        logging.error(f"计算验证指标时出错: {e}")
        iou_val, f1_val, pre_val, rec_val, oa_val, kappa_val = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    logging.info(f"Epoch [{epoch}/{num_epochs}] Val Loss: {avg_epoch_loss:.4f}, Val F1: {f1_val:.4f}, Val IoU: {iou_val:.4f}, Val Pre: {pre_val:.4f}, Val Rec: {rec_val:.4f}, Val OA: {oa_val:.4f}, Val Kappa: {kappa_val:.4f}")
    return avg_epoch_loss, f1_val, iou_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="D&A-CDNet Training")
    # --- 数据和路径参数 ---
    parser.add_argument('--dataset_root', type=str, default='./data', help='数据集根目录')
    parser.add_argument('--data_name', type=str, default='SYSU', choices=['LEVIR', 'CDD', 'WHU', 'SYSU', 'S2Looking', 'DSIFN','custom'], help='数据集名称')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='模型和日志保存根路径')
    parser.add_argument('--experiment_name', type=str, default='DA_CDNet_Run', help='当前实验的名称')

    # --- 训练参数 ---
    parser.add_argument('--epoch', type=int, default=150, help='总训练轮数') # 减少默认轮数以快速测试
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--batchsize', type=int, default=8, help='训练批次大小 (根据显存调整)') # 减小默认批大小
    parser.add_argument('--trainsize', type=int, default=256, help='训练图像输入尺寸')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作进程数 (Windows下设为0可能更稳定)')
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    # --- 新增：损失权重和参数 ---
    parser.add_argument('--lambda_tc', type=float, default=0.5, help='时序一致性损失 L_tc 的权重')
    parser.add_argument('--lambda_cs', type=float, default=0.5, help='对比分离损失 L_cs 的权重')
    parser.add_argument('--contrastive_margin', type=float, default=1.0, help='对比分离损失 L_cs 中的 margin 值')


    # --- 模型参数 ---
    parser.add_argument('--backbone', type=str, default='pvt_v2_b1', help='主干网络名称 (来自timm库, e.g., mit_b0, mit_b1, mit_b2)')
    parser.add_argument('--pretrained', type=lambda x: (str(x).lower() == 'true'), default=True, help='是否加载主干网络的预训练权重')
    parser.add_argument('--resume', type=str, default=None, help='要从中恢复训练的检查点 (.pth) 文件路径')

    opt = parser.parse_args()
    seed_everything(opt.seed)

    # --- 创建保存路径和日志 ---
    current_time = time.strftime("%Y%m%d-%H%M%S")
    experiment_save_path = os.path.join(opt.save_path, opt.data_name, f"{opt.experiment_name}_{opt.backbone}_{current_time}")
    os.makedirs(experiment_save_path, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(experiment_save_path, 'training_log.log')),
                            logging.StreamHandler() # 同时输出到控制台
                        ])
    logging.info("命令行参数:\n%s", json.dumps(vars(opt), indent=2))


    # --- 数据集路径 ---
    dataset_base_paths = {
        'LEVIR': os.path.join(opt.dataset_root, 'LEVIR-CD/'),
        'CDD': os.path.join(opt.dataset_root, 'CDD/'),
        'WHU': os.path.join(opt.dataset_root, 'WHU-CD/'), # 假设您的WHU是256尺寸的
        'SYSU': os.path.join(opt.dataset_root, 'SYSU-CD/'),
        'S2Looking': os.path.join(opt.dataset_root, 'S2Looking/'),
        'DSIFN': os.path.join(opt.dataset_root, 'DSIFN/'),# 添加 S2Looking
        'custom': opt.dataset_root # 自定义数据集
    }
    base_path_for_data = dataset_base_paths.get(opt.data_name, opt.dataset_root)
    if opt.data_name not in dataset_base_paths:
        logging.warning(f"数据集名称 {opt.data_name} 未在预设路径中，将直接使用 dataset_root。")

    opt.train_root = os.path.join(base_path_for_data, 'train/')
    opt.val_root = os.path.join(base_path_for_data, 'val/')
    logging.info(f"使用数据集: {opt.data_name}, 训练集: {opt.train_root}, 验证集: {opt.val_root}")

    if not os.path.isdir(opt.train_root) or not os.path.isdir(opt.val_root):
        logging.error(f"错误: 训练或验证目录不存在!")
        exit(1)

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    if device.type == 'cuda':
        logging.info(f"CUDA 设备数量: {torch.cuda.device_count()}")

    # --- 数据加载器 ---
    logging.info("初始化数据加载器...")
    train_loader = data_loader.get_loader(opt.train_root, opt.batchsize, opt.trainsize,
                                          num_workers=opt.num_workers, shuffle=True, pin_memory=True)
    val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, # 验证集通常用 test loader
                                             num_workers=opt.num_workers, shuffle=False, pin_memory=True)
    logging.info(f"数据加载器完成。训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")

    # --- 评估器 ---
    eva_train = Evaluator(num_class=2)
    eva_val = Evaluator(num_class=2)

    # --- 模型 ---
    logging.info(f"初始化模型 D&A-CDNet (Backbone: {opt.backbone}, Pretrained: {opt.pretrained})...")
    try:
        model = D_A_CDNet(backbone_name=opt.backbone, pretrained=opt.pretrained).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"模型初始化成功。可训练参数量: {total_params / 1e6:.2f} M")
    except Exception as e:
        logging.error(f"模型初始化失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        exit(1)

    # --- 多GPU ---
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        logging.info(f"使用 {torch.cuda.device_count()} 个 GPUs 进行 DataParallel 训练!")
        model = nn.DataParallel(model)

    # --- 损失函数 ---
    criterion_main = dice_bce_loss().to(device)
    criterion_tc = TemporalConsistencyLoss().to(device)
    criterion_cs = ContrastiveSeparationLoss(margin=opt.contrastive_margin).to(device)
    logging.info(f"损失函数初始化完成 (L_main: Dice+BCE, L_tc weight: {opt.lambda_tc}, L_cs weight: {opt.lambda_cs}, L_cs margin: {opt.contrastive_margin})")

    # --- 优化器和学习率调度器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.01)
    # 使用余弦退火学习率，更平滑
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=opt.lr * 0.01)
    logging.info(f"优化器: AdamW, 学习率调度器: CosineAnnealingLR")

    # --- 在这里插入加载逻辑 ---
    start_epoch = 1
    best_val_iou_for_saving = 0.0
    best_val_f1_for_saving = 0.0
    best_epoch_for_saving = 0

    if opt.resume and os.path.isfile(opt.resume):
        logging.info(f"正在从断点恢复训练: {opt.resume}")
        try:
            checkpoint = torch.load(opt.resume, map_location=device)

            # 1. 加载模型权重
            model_state_to_load = checkpoint.get('model_state_dict', checkpoint)

            # 处理 DataParallel 'module.' 前缀 (以防保存和加载时的GPU数量不同)
            is_parallel_state = all(key.startswith('module.') for key in model_state_to_load.keys())
            is_parallel_model = isinstance(model, nn.DataParallel)

            if is_parallel_state and not is_parallel_model:  # 权重是DP，模型是单GPU
                logging.info("检测到 DataParallel 权重，正在移除 'module.' 前缀以加载到单GPU模型...")
                new_state_dict = {k[len('module.'):]: v for k, v in model_state_to_load.items()}
                model.load_state_dict(new_state_dict)
            elif not is_parallel_state and is_parallel_model:  # 权重是单GPU，模型是DP
                logging.warning("权重为单GPU保存，但当前为多GPU环境。正在加载到 'module.'...")
                model.module.load_state_dict(model_state_to_load)
            else:  # 状态匹配 (DP->DP or 单->单)
                model.load_state_dict(model_state_to_load)
            logging.info("成功加载模型状态。")

            # 2. 加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logging.info("成功加载优化器状态。")
            else:
                logging.warning("未在检查点中找到 'optimizer_state_dict'，优化器将从头开始。")

            # 3. 加载调度器状态
            if 'scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logging.info("成功加载调度器状态。")
            else:
                logging.warning("未在检查点中找到 'scheduler_state_dict'，调度器将从头开始。")

            # 4. 加载起始 Epoch 和最佳指标
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1  # 从保存的 epoch 的 *下一个* epoch 开始
                logging.info(f"将从 Epoch {start_epoch} 开始训练。")
            else:
                logging.warning("未在检查点中找到 'epoch'，将从 Epoch 1 开始。")

            if 'best_val_iou' in checkpoint:
                best_val_iou_for_saving = checkpoint.get('best_val_iou', 0.0)
                best_val_f1_for_saving = checkpoint.get('best_val_f1', 0.0)
                best_epoch_for_saving = checkpoint.get('epoch', 0)
                logging.info(
                    f"已加载之前的最佳 IoU: {best_val_iou_for_saving:.4f} (来自 Epoch {best_epoch_for_saving})")

        except Exception as e:
            logging.error(f"加载检查点失败: {e}。将从头开始训练。")
            start_epoch = 1
    else:
        if opt.resume:
            logging.warning(f"指定的检查点路径无效: {opt.resume}。将从头开始训练。")
        else:
            logging.info("未指定 --resume 参数，将从头开始训练。")
    # --- 训练循环 ---

    start_time_total_train = time.time()

    logging.info("="*20 + " 开始训练 " + "="*20)
    for epoch_num_current in range(start_epoch, opt.epoch + 1):
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"--- Epoch {epoch_num_current}/{opt.epoch}, LR: {current_lr:.2e} ---")

        # 训练
        train_loss, train_f1, train_iou = train_one_epoch(
            train_loader, model, criterion_main, criterion_tc, criterion_cs,
            optimizer, epoch_num_current, opt.epoch, eva_train, device, opt)

        # 验证
        val_loss, val_f1, val_iou = validate_one_epoch(
            val_loader, model, criterion_main, epoch_num_current, opt.epoch, eva_val, device, opt)

        lr_scheduler.step()

        # 保存最佳模型 (基于验证集 IoU)
        if val_iou > best_val_iou_for_saving:
            best_val_iou_for_saving = val_iou
            best_val_f1_for_saving = val_f1
            best_epoch_for_saving = epoch_num_current

            # 区分 DataParallel 和单 GPU 保存
            model_state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

            best_model_filename = f"best_model_epoch{epoch_num_current:03d}_iou{val_iou:.4f}_f1{val_f1:.4f}.pth"
            save_checkpoint_path = os.path.join(experiment_save_path, best_model_filename)
            try:
                torch.save({
                    'epoch': epoch_num_current,
                    'model_state_dict': model_state_to_save,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_val_iou': best_val_iou_for_saving,
                    'best_val_f1': best_val_f1_for_saving,
                    'options': vars(opt) # 保存命令行参数
                }, save_checkpoint_path)
                logging.info(f"已保存新的最佳模型 (Epoch {epoch_num_current}): Val IoU: {val_iou:.4f}, Val F1: {val_f1:.4f} 到 {save_checkpoint_path}")

                # 可选：同时保存一份名为 'best_model.pth' 的副本，方便后续测试脚本直接加载
                # torch.save(model_state_to_save, os.path.join(experiment_save_path, 'best_model.pth'))

            except Exception as e:
                logging.error(f"保存模型权重失败: {e}")

        logging.info(f"当前最佳 Val IoU: {best_val_iou_for_saving:.4f} (来自 Epoch {best_epoch_for_saving}), 对应F1: {best_val_f1_for_saving:.4f}")
        logging.info("-" * 60)

    total_training_time_seconds = time.time() - start_time_total_train
    logging.info(
        f"训练完成！总耗时: {total_training_time_seconds // 3600:.0f}h "
        f"{(total_training_time_seconds % 3600) // 60:.0f}m "
        f"{total_training_time_seconds % 60:.2f}s"
    )
    logging.info(f"最终最佳验证集 IoU: {best_val_iou_for_saving:.4f} (在第 {best_epoch_for_saving} 轮), 对应 F1: {best_val_f1_for_saving:.4f}")
    logging.info(f"模型和日志保存在: {experiment_save_path}")