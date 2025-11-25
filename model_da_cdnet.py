# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os  # 导入 os 库
import logging  # 导入日志库
from safetensors.torch import load_file  # 导入 safetensors 加载器


# --- 特征解耦模块 (FDM) ---
# (FDM, ChannelAttention, DecoderBlock, PredictionHead 类的代码保持不变...)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class FDM(nn.Module):
    def __init__(self, in_channels):
        super(FDM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels * 2)

    def forward(self, f_t1, f_t2):
        f_concat = torch.cat([f_t1, f_t2], dim=1)
        attention_weights = self.channel_attention(f_concat)
        f_ci_concat = f_concat * attention_weights
        f_cs_concat = f_concat - f_ci_concat
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


# --- D&A-CDNet 整体网络 ---
class D_A_CDNet(nn.Module):
    def __init__(self, backbone_name='pvt_v2_b1', pretrained=True, num_classes=1, decoder_channels=(256, 128, 64, 64)):
        super(D_A_CDNet, self).__init__()

        self.backbone = timm.create_model(backbone_name, pretrained=False, features_only=True)

        if pretrained:
            try:
                # 2. 定义本地权重文件的路径
                # 假设权重文件在项目根目录 (与 train.py 同级)
                script_dir = os.path.dirname(__file__)
                weight_file = 'pvt_v2_b1_weights.safetensors'  # 使用您重命名的文件
                weight_path = os.path.join(script_dir, weight_file)

                if not os.path.exists(weight_path):
                    logging.error(f"--- 权重文件未找到! ---")
                    logging.error(f"请将下载的 .safetensors 权重文件复制到项目根目录并重命名为: {weight_file}")
                    logging.error(f"检查路径: {weight_path}")
                    logging.error("正在退出。")
                    exit(1)  # 找不到文件则退出

                logging.info(f"正在从本地文件手动加载权重: {weight_path}")
                # 3. 加载 .safetensors 文件
                state_dict = load_file(weight_path)

                # 4. 将权重加载到 self.backbone
                # 我们使用 strict=False，因为 timm.create_model 创建的 features_only 模型
                # 可能不包含原始模型中的 'head' (分类头) 部分，这会导致 'unexpected_keys'
                missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)

                if unexpected_keys:
                    logging.warning(f"加载权重时忽略了以下键 (通常是分类头，正常现象): {unexpected_keys}")
                if missing_keys:
                    logging.warning(f"加载权重时缺少了以下键 (这可能不正常): {missing_keys}")

                logging.info(f"成功从 {weight_path} 加载了权重到 backbone。")

            except ImportError:
                logging.error("请安装 `safetensors` 库 (pip install safetensors) 以便从 .safetensors 文件加载权重。")
                exit(1)
            except Exception as e:
                logging.error(f"手动加载权重失败: {e}")
                import traceback
                logging.error(traceback.format_exc())
                exit(1)

        # --- 修改结束 ---

        # 获取backbone各阶段输出通道数
        encoder_channels = self.backbone.feature_info.channels()

        # --- 特征解耦模块 (每个尺度一个) ---
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
        # (forward 函数保持不变...)
        features_t1 = self.backbone(x1)
        features_t2 = self.backbone(x2)
        ci1_t1, ci1_t2, cs1_t1, cs1_t2 = self.fdm1(features_t1[0], features_t2[0])
        ci2_t1, ci2_t2, cs2_t1, cs2_t2 = self.fdm2(features_t1[1], features_t2[1])
        ci3_t1, ci3_t2, cs3_t1, cs3_t2 = self.fdm3(features_t1[2], features_t2[2])
        ci4_t1, ci4_t2, cs4_t1, cs4_t2 = self.fdm4(features_t1[3], features_t2[3])
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

# --- 示例用法 ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型实例 (使用 PVT v2 B1)
    model = D_A_CDNet(backbone_name='pvt_v2_b1', pretrained=False).to(device)
    model.train()

    # 创建虚拟输入数
    dummy_t1 = torch.randn(2, 3, 256, 256).to(device)
    dummy_t2 = torch.randn(2, 3, 256, 256).to(device)

    # 前向传播 (训练模式)
    final_logits, features_dict = model(dummy_t1, dummy_t2)

    print("--- 训练模式输出 (使用 pvt_v2_b1) ---")
    print(f"最终预测 Logits 形状: {final_logits.shape}")  # 应为 (2, 1, 256, 256)
    print("解耦特征字典键:", features_dict.keys())
    print(f"F_ci1_t1 形状: {features_dict['ci1_t1'].shape}")  # 应为 (2, 64, 64, 64) for pvt_v2_b1
    print(f"F_cs4_t2 形状: {features_dict['cs4_t2'].shape}")  # 应为 (2, 512, 8, 8) for pvt_v2_b1

    # 前向传播 (评估模式)
    model.eval()
    with torch.no_grad():
        final_logits_eval = model(dummy_t1, dummy_t2)

    print("\n--- 评估模式输出 ---")
    print(f"最终预测 Logits 形状: {final_logits_eval.shape}")  # 应为 (2, 1, 256, 256)