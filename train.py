# coding: utf-8
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import time
import argparse
import logging
import datetime  # [新增] 用于格式化时间

# --- 导入自定义模块 ---
import data_loader
# [重要] 请确保 loss.py 中已包含 OrthogonalityLoss
from loss import dice_bce_loss, TemporalConsistencyLoss, ContrastiveSeparationLoss, OrthogonalityLoss
from metrics import Evaluator
from model_da_cdnet import D_A_CDNet


# --- 辅助函数 ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# [新增] 计算模型 FLOPs 和 参数量
def cal_flops_params(model, input_size, device):
    try:
        from thop import profile
        # 创建虚拟输入 (B=1, C=3, H, W)
        input_t1 = torch.randn(1, 3, input_size, input_size).to(device)
        input_t2 = torch.randn(1, 3, input_size, input_size).to(device)

        logging.info("正在计算模型 FLOPs 和参数量...")
        # 将模型设为 eval 模式以避免影响 BatchNorm 等统计信息
        model.eval()
        flops, params = profile(model, inputs=(input_t1, input_t2), verbose=False)
        model.train()  # 恢复训练模式

        flops_g = flops / 1e9
        params_m = params / 1e6
        return flops_g, params_m
    except ImportError:
        logging.warning("未检测到 'thop' 库，跳过 FLOPs 计算。(请运行 pip install thop)")
        params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        return 0.0, params_m
    except Exception as e:
        logging.warning(f"计算 FLOPs 时出错: {e}")
        params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        return 0.0, params_m


def train_one_epoch(train_loader, model, criterion_main, criterion_tc, criterion_cs, criterion_ortho,
                    optimizer, epoch, num_epochs, eva_train, device, opt):
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)
    eva_train.reset()

    aux_loss_scales = [2, 3]

    pbar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch}/{num_epochs} [T]")
    for i, (t1_img, t2_img, target_mask) in pbar:
        t1_img = t1_img.to(device, non_blocking=True)
        t2_img = t2_img.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits_final, decoupled_features = model(t1_img, t2_img)

        loss_main = criterion_main(logits_final, target_mask)

        loss_tc = torch.tensor(0.0, device=device)
        loss_cs = torch.tensor(0.0, device=device)
        loss_ortho = torch.tensor(0.0, device=device)
        num_aux_losses = 0

        for scale_idx in aux_loss_scales:
            ci_t1_key = f'ci{scale_idx + 1}_t1'
            ci_t2_key = f'ci{scale_idx + 1}_t2'
            cs_t1_key = f'cs{scale_idx + 1}_t1'
            cs_t2_key = f'cs{scale_idx + 1}_t2'

            if all(k in decoupled_features for k in [ci_t1_key, ci_t2_key, cs_t1_key, cs_t2_key]):
                f_ci_t1 = decoupled_features[ci_t1_key]
                f_ci_t2 = decoupled_features[ci_t2_key]
                f_cs_t1 = decoupled_features[cs_t1_key]
                f_cs_t2 = decoupled_features[cs_t2_key]

                loss_tc += criterion_tc(f_ci_t1, f_ci_t2)

                target_mask_resized = F.interpolate(
                    target_mask.float(),
                    size=f_cs_t1.shape[2:],
                    mode='nearest'
                )

                loss_cs += criterion_cs(f_cs_t1, f_cs_t2, target_mask_resized)

                l_ortho_t1 = criterion_ortho(f_ci_t1, f_cs_t1)
                l_ortho_t2 = criterion_ortho(f_ci_t2, f_cs_t2)
                loss_ortho += (l_ortho_t1 + l_ortho_t2)

                num_aux_losses += 1

        if num_aux_losses > 0:
            loss_tc /= num_aux_losses
            loss_cs /= num_aux_losses
            loss_ortho /= num_aux_losses

        total_loss = loss_main + \
                     opt.lambda_tc * loss_tc + \
                     opt.lambda_cs * loss_cs + \
                     opt.lambda_ortho * loss_ortho

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

        with torch.no_grad():
            pred_prob = torch.sigmoid(logits_final)
            pred_binary = (pred_prob >= 0.5).cpu().numpy().astype(np.uint8)
            target_numpy = target_mask.cpu().numpy().astype(np.uint8)
            if pred_binary.shape[1] == 1: pred_binary = pred_binary.squeeze(1)
            if target_numpy.shape[1] == 1: target_numpy = target_numpy.squeeze(1)
            eva_train.add_batch(target_numpy, pred_binary)

        pbar.set_postfix(Loss=f"{total_loss.item():.4f}",
                         Main=f"{loss_main.item():.4f}",
                         Ortho=f"{loss_ortho.item():.4f}")

    avg_epoch_loss = epoch_loss / num_batches

    try:
        iou_train = eva_train.Intersection_over_Union()[1]
        f1_train = eva_train.F1()[1]
    except:
        iou_train, f1_train = 0.0, 0.0

    # 日志输出移至主循环，这里仅返回数值
    return avg_epoch_loss, f1_train, iou_train


def validate_one_epoch(val_loader, model, criterion_main, epoch, num_epochs, eva_val, device, opt):
    model.eval()
    epoch_loss = 0.0
    num_batches = len(val_loader)
    eva_val.reset()

    pbar = tqdm(enumerate(val_loader), total=num_batches, desc=f"Epoch {epoch}/{num_epochs} [V]")
    with torch.no_grad():
        for i, data_batch in pbar:
            if len(data_batch) == 4:
                t1_img, t2_img, target_mask, _ = data_batch
            elif len(data_batch) == 3:
                t1_img, t2_img, target_mask = data_batch
            else:
                continue

            t1_img = t1_img.to(device, non_blocking=True)
            t2_img = t2_img.to(device, non_blocking=True)
            target_mask = target_mask.to(device, non_blocking=True)

            logits_final = model(t1_img, t2_img)
            loss_main = criterion_main(logits_final, target_mask)
            epoch_loss += loss_main.item()

            pred_prob = torch.sigmoid(logits_final)
            pred_binary = (pred_prob >= 0.5).cpu().numpy().astype(np.uint8)
            target_numpy = target_mask.cpu().numpy().astype(np.uint8)
            if pred_binary.shape[1] == 1: pred_binary = pred_binary.squeeze(1)
            if target_numpy.shape[1] == 1: target_numpy = target_numpy.squeeze(1)
            eva_val.add_batch(target_numpy, pred_binary)

            pbar.set_postfix(Loss=f"{loss_main.item():.4f}")

    avg_epoch_loss = epoch_loss / num_batches
    try:
        iou_val = eva_val.Intersection_over_Union()[1]
        f1_val = eva_val.F1()[1]
    except:
        iou_val, f1_val = 0.0, 0.0

    # 日志输出移至主循环，这里仅返回数值
    return avg_epoch_loss, f1_val, iou_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="D&A-CDNet Training")
    parser.add_argument('--dataset_root', type=str, default='./data', help='数据集根目录')
    parser.add_argument('--data_name', type=str, default='WHU-CD', help='数据集名称')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='保存路径')
    parser.add_argument('--experiment_name', type=str, default='DA_CDNet_Ortho', help='实验名称')

    parser.add_argument('--epoch', type=int, default=200, help='总Epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batchsize', type=int, default=8, help='Batch Size')
    parser.add_argument('--trainsize', type=int, default=256, help='Image Size')
    parser.add_argument('--num_workers', type=int, default=4, help='Workers')
    parser.add_argument('--seed', type=int, default=42, help="Seed")

    parser.add_argument('--lambda_tc', type=float, default=0.5, help='Weight for L_tc')
    parser.add_argument('--lambda_cs', type=float, default=0.5, help='Weight for L_cs')
    parser.add_argument('--lambda_ortho', type=float, default=0.1, help='Weight for L_ortho')
    parser.add_argument('--contrastive_margin', type=float, default=1.0, help='Margin for L_cs')

    parser.add_argument('--backbone', type=str, default='pvt_v2_b1', help='Backbone')
    parser.add_argument('--pretrained', type=lambda x: (str(x).lower() == 'true'), default=True, help='Pretrained')
    parser.add_argument('--resume', type=str, default=None, help='Resume Path')

    opt = parser.parse_args()
    seed_everything(opt.seed)

    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    experiment_save_path = os.path.join(opt.save_path, opt.data_name,
                                        f"{opt.experiment_name}_{opt.backbone}_{current_time_str}")
    os.makedirs(experiment_save_path, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(experiment_save_path, 'training_log.log')),
                            logging.StreamHandler()
                        ])
    logging.info("命令行参数:\n%s", json.dumps(vars(opt), indent=2))

    # --- 路径与数据加载 ---
    opt.train_root = os.path.join(opt.dataset_root, opt.data_name, 'train/')
    opt.val_root = os.path.join(opt.dataset_root, opt.data_name, 'val/')
    if not os.path.isdir(opt.train_root):
        opt.train_root = os.path.join(opt.dataset_root, 'train/')
        opt.val_root = os.path.join(opt.dataset_root, 'val/')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = data_loader.get_loader(opt.train_root, opt.batchsize, opt.trainsize, num_workers=opt.num_workers,
                                          shuffle=True, pin_memory=True)
    val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=opt.num_workers,
                                             shuffle=False, pin_memory=True)

    eva_train = Evaluator(num_class=2)
    eva_val = Evaluator(num_class=2)

    model = D_A_CDNet(backbone_name=opt.backbone, pretrained=opt.pretrained).to(device)

    # [修改] 计算并打印模型参数量和FLOPs
    flops_g, params_m = cal_flops_params(model, opt.trainsize, device)
    logging.info(
        f"Model Info - Params: {params_m:.2f} M, FLOPs: {flops_g:.2f} G (Input: {opt.trainsize}x{opt.trainsize})")

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion_main = dice_bce_loss().to(device)
    criterion_tc = TemporalConsistencyLoss().to(device)
    criterion_cs = ContrastiveSeparationLoss(margin=opt.contrastive_margin).to(device)
    criterion_ortho = OrthogonalityLoss().to(device)

    logging.info(
        f"Loss Config: L_main(Dice+BCE), L_tc({opt.lambda_tc}), L_cs({opt.lambda_cs}), L_ortho({opt.lambda_ortho})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=opt.lr * 0.01)

    start_epoch = 1
    best_val_iou = 0.0
    best_val_f1 = 0.0
    if opt.resume and os.path.isfile(opt.resume):
        checkpoint = torch.load(opt.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
        if 'best_val_iou' in checkpoint: best_val_iou = checkpoint['best_val_iou']
        logging.info(f"Resume training from epoch {start_epoch}")

    logging.info("Start Training...")

    # [新增] 记录总训练开始时间
    total_train_start_time = time.time()

    for epoch in range(start_epoch, opt.epoch + 1):
        # [新增] 记录 Epoch 开始时间
        epoch_start_time = time.time()

        train_loss, train_f1, train_iou = train_one_epoch(
            train_loader, model, criterion_main, criterion_tc, criterion_cs, criterion_ortho,
            optimizer, epoch, opt.epoch, eva_train, device, opt)

        val_loss, val_f1, val_iou = validate_one_epoch(
            val_loader, model, criterion_main, epoch, opt.epoch, eva_val, device, opt)

        lr_scheduler.step()

        # [新增] 计算 Epoch 耗时
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_duration)))

        # [修改] 日志输出包含时间、Params 和 FLOPs
        logging.info(f"Epoch [{epoch}/{opt.epoch}] Time: {epoch_time_str} | "
                     f"Train [L:{train_loss:.4f} F1:{train_f1:.4f} IoU:{train_iou:.4f}] | "
                     f"Val [L:{val_loss:.4f} F1:{val_f1:.4f} IoU:{val_iou:.4f}]")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_f1 = val_f1
            save_path = os.path.join(experiment_save_path, f"best_model_iou{val_iou:.4f}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model,
                                                                            nn.DataParallel) else model.state_dict(),
                'best_val_iou': best_val_iou,
                'best_val_f1': best_val_f1
            }, save_path)
            logging.info(f"Saved Best Model: {save_path}")

    # [新增] 计算总耗时
    total_train_end_time = time.time()
    total_duration = total_train_end_time - total_train_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_duration)))

    logging.info("=" * 60)
    logging.info("Training Finished.")
    logging.info(f"Total Training Time: {total_time_str}")
    logging.info(f"Model Complexity - Params: {params_m:.2f} M, FLOPs: {flops_g:.2f} G")
    logging.info(f"Best Val IoU: {best_val_iou:.4f}, F1: {best_val_f1:.4f}")
    logging.info("=" * 60)