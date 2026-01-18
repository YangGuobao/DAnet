# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import logging
from safetensors.torch import load_file


# --- [新增] 空间注意力模块 (对应 Source 54) ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # 输入通道为2 (Max + Avg)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿着通道维度做 AvgPool 和 MaxPool
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)


# --- [修改] 通道注意力模块 (保持不变，但为了完整性列出) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # [可选] 论文常结合Max和Avg，这里保留您原本的Avg也可以，或者增强一下

        hidden_dim = max(1, in_planes // ratio)
        # 使用 Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 改进：结合 Avg 和 Max (更强的通道注意力)
        avg_out = self.fc(self.avg_pool(x))
        # max_out = self.fc(self.max_pool(x)) # 如果显存够，建议加上 Max
        # out = avg_out + max_out
        return self.sigmoid(avg_out)


# --- [重写] 特征解耦模块 (FDM -> 对应论文 ARGD) ---
class FDM(nn.Module):
    """
    对应论文中的: Adaptive Residual Gated Disentanglement Module (ARGD)
    包含:
    1. Dual-Attention Background Modeling (双重注意力背景建模)
    2. Adaptive Gated Feature Sifting (自适应门控特征筛选)
    """

    def __init__(self, in_channels):
        super(FDM, self).__init__()

        # 混合特征通道数 (Concat后是2倍)
        mix_channels = in_channels * 2

        # 1. 双重注意力 (用于提取 F_ci)
        self.channel_att = ChannelAttention(mix_channels)
        self.spatial_att = SpatialAttention(kernel_size=7)

        # 2. 自适应门控 (用于提取 F_cs)
        # 输入是 [F_mix; F_ci] -> 4倍通道
        # 输出是 Gate (1通道或mix_channels通道，通常逐通道加权效果更好)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(mix_channels * 2, mix_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mix_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mix_channels, mix_channels, kernel_size=1, bias=True),
            nn.Sigmoid()  # 生成 0~1 的门控系数
        )

    def forward(self, f_t1, f_t2):
        # A. 混合特征 F_mix
        f_mix = torch.cat([f_t1, f_t2], dim=1)  # [B, 2C, H, W]

        # B. 背景建模 (Dual Attention) -> 提取 F_ci
        # Source 53: F'ci = Ac(Fmix) * Fmix
        att_c = self.channel_att(f_mix)
        f_ci_temp = f_mix * att_c

        # Source 54: Fci = As(F'ci) * F'ci
        att_s = self.spatial_att(f_ci_temp)
        f_ci_concat = f_ci_temp * att_s  # 最终的背景特征 (变化无关)

        # C. 自适应门控筛选 -> 提取 F_cs
        # Source 56: G = Sigmoid(Conv([F_mix; F_ci]))
        gate_input = torch.cat([f_mix, f_ci_concat], dim=1)  # [B, 4C, H, W]
        gate = self.gate_conv(gate_input)

        # Source 58: Fcs = Fmix * G
        f_cs_concat = f_mix * gate  # 最终的变化特征 (变化敏感)

        # D. 拆分回 T1/T2 (为了计算 Loss)
        c = f_t1.shape[1]
        f_ci_t1, f_ci_t2 = f_ci_concat.split(c, dim=1)
        f_cs_t1, f_cs_t2 = f_cs_concat.split(c, dim=1)

        return f_ci_t1, f_ci_t2, f_cs_t1, f_cs_t2




class DecoderBlock(nn.Module):
    def __init__(self, in_channels_skip, in_channels_up, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels_skip + in_channels_up, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, skip_feature, up_feature):
        x = self.upsample(up_feature)
        if x.shape[2:] != skip_feature.shape[2:]:
            diff_h = skip_feature.shape[2] - x.shape[2]
            diff_w = skip_feature.shape[3] - x.shape[3]
            x = F.pad(x, (diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2))
        x = torch.cat([skip_feature, x], dim=1)
        x = self.conv_block(x)
        return x


class PredictionHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(PredictionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x


class D_A_CDNet(nn.Module):
    def __init__(self, backbone_name='pvt_v2_b1', pretrained=True, num_classes=1, decoder_channels=(256, 128, 64, 64)):
        super(D_A_CDNet, self).__init__()

        self.backbone = timm.create_model(backbone_name, pretrained=False, features_only=True)

        if pretrained:
            try:
                script_dir = os.path.dirname(__file__)
                # 您的权重文件名
                weight_file = 'pvt_v2_b1_weights.safetensors'  # 或 pvt_v2_b1_weights.safetensors
                weight_path = os.path.join(script_dir, weight_file)

                if not os.path.exists(weight_path):
                    logging.warning(f"本地权重文件未找到: {weight_path}, 将尝试在线下载或跳过。")
                else:
                    logging.info(f"正在从本地文件加载权重: {weight_path}")
                    state_dict = load_file(weight_path)
                    self.backbone.load_state_dict(state_dict, strict=False)
                    logging.info("权重加载成功。")

            except Exception as e:
                logging.error(f"权重加载失败: {e}")

        # 获取backbone各阶段输出通道数
        encoder_channels = self.backbone.feature_info.channels()

        # --- 特征解耦模块 (使用新的 FDM/ARGD) ---
        self.fdm1 = FDM(encoder_channels[0])  # Stride 4
        self.fdm2 = FDM(encoder_channels[1])  # Stride 8
        self.fdm3 = FDM(encoder_channels[2])  # Stride 16
        self.fdm4 = FDM(encoder_channels[3])  # Stride 32

        # --- 解码器 ---
        self.decoder_block1 = DecoderBlock(encoder_channels[2], encoder_channels[3], decoder_channels[0])
        self.decoder_block2 = DecoderBlock(encoder_channels[1], decoder_channels[0], decoder_channels[1])
        self.decoder_block3 = DecoderBlock(encoder_channels[0], decoder_channels[1], decoder_channels[2])

        # --- 预测头 ---
        self.prediction_head = PredictionHead(decoder_channels[2], num_classes)

        # --- 最终上采样 ---
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        features_t1 = self.backbone(x1)
        features_t2 = self.backbone(x2)

        # ARGD 解耦
        ci1_t1, ci1_t2, cs1_t1, cs1_t2 = self.fdm1(features_t1[0], features_t2[0])
        ci2_t1, ci2_t2, cs2_t1, cs2_t2 = self.fdm2(features_t1[1], features_t2[1])
        ci3_t1, ci3_t2, cs3_t1, cs3_t2 = self.fdm3(features_t1[2], features_t2[2])
        ci4_t1, ci4_t2, cs4_t1, cs4_t2 = self.fdm4(features_t1[3], features_t2[3])

        # 计算差值用于解码 (只用 cs 特征)
        d_cs1 = torch.abs(cs1_t1 - cs1_t2)
        d_cs2 = torch.abs(cs2_t1 - cs2_t2)
        d_cs3 = torch.abs(cs3_t1 - cs3_t2)
        d_cs4 = torch.abs(cs4_t1 - cs4_t2)

        fused3 = self.decoder_block1(skip_feature=d_cs3, up_feature=d_cs4)
        fused2 = self.decoder_block2(skip_feature=d_cs2, up_feature=fused3)
        fused1 = self.decoder_block3(skip_feature=d_cs1, up_feature=fused2)

        logits_low_res = self.prediction_head(fused1)
        logits_final = self.final_upsample(logits_low_res)

        if self.training:
            decoupled_features = {
                'ci1_t1': ci1_t1, 'ci1_t2': ci1_t2, 'cs1_t1': cs1_t1, 'cs1_t2': cs1_t2,
                'ci2_t1': ci2_t1, 'ci2_t2': ci2_t2, 'cs2_t1': cs2_t1, 'cs2_t2': cs2_t2,
                'ci3_t1': ci3_t1, 'ci3_t2': ci3_t2, 'cs3_t1': cs3_t1, 'cs3_t2': cs3_t2,
                'ci4_t1': ci4_t1, 'ci4_t2': ci4_t2, 'cs4_t1': cs4_t1, 'cs4_t2': cs4_t2,
            }
            return logits_final, decoupled_features
        else:
            return logits_final