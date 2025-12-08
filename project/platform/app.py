import gradio as gr
import torch
import timm
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import zipfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
import io
import time



"""使用占位模型，增加视频处理功能"""

# 导入视频处理模块
from video_processor import CystoscopyVideoProcessor

# --- Device setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Helper function to crop borders and pad to square ---
def crop_borders_and_pad_to_square(img_pil, border_color_threshold=20):
    """
    Crops black/dark borders from a PIL image and then pads it to a square.

    Args:
        img_pil (PIL.Image.Image): Input image.
        border_color_threshold (int): Pixels with all channels below this value are considered border.

    Returns:
        PIL.Image.Image: Processed image.
    """
    img_np = np.array(img_pil)

    if len(img_np.shape) == 2:  # Grayscale image
        img_np_rgb_like = np.stack((img_np,) * 3, axis=-1)
    elif img_np.shape[2] == 4:  # RGBA image, convert to RGB
        img_pil_original_mode_is_rgba = True
        temp_img_pil = img_pil.convert("RGB")
        img_np_rgb_like = np.array(temp_img_pil)
    elif img_np.shape[2] == 1:  # Grayscale with 3rd dim e.g. (H, W, 1)
        img_np_rgb_like = np.concatenate([img_np] * 3, axis=-1)
    else:  # Already RGB or similar 3-channel
        img_np_rgb_like = img_np

    # Find non-border rows: rows where at least one pixel is above threshold in any channel
    row_has_content = np.any(img_np_rgb_like > border_color_threshold, axis=(1, 2))
    # Find non-border columns: columns where at least one pixel is above threshold in any channel
    col_has_content = np.any(img_np_rgb_like > border_color_threshold, axis=(0, 2))

    non_border_rows = np.where(row_has_content)[0]
    non_border_cols = np.where(col_has_content)[0]

    if non_border_rows.size == 0 or non_border_cols.size == 0:
        # Image is likely all dark or empty, proceed without cropping
        # However, we still need to ensure it's converted to RGB if it was RGBA for consistent padding mode
        if img_np.shape[2] == 4:
            cropped_img_pil = img_pil.convert("RGB")
        else:
            cropped_img_pil = img_pil.copy()
    else:
        first_row, last_row = non_border_rows[0], non_border_rows[-1]
        first_col, last_col = non_border_cols[0], non_border_cols[-1]

        # Perform cropping on the original numpy array to preserve mode if possible
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:  # Original was RGB
            cropped_img_np = img_np[first_row:last_row + 1, first_col:last_col + 1, :]
        elif len(img_np.shape) == 2:  # Original was Grayscale
            cropped_img_np = img_np[first_row:last_row + 1, first_col:last_col + 1]
        elif img_np.shape[2] == 4:  # Original was RGBA, crop the converted RGB version
            img_rgb_for_crop = np.array(img_pil.convert("RGB"))
            cropped_img_np = img_rgb_for_crop[first_row:last_row + 1, first_col:last_col + 1, :]
        else:  # Fallback to the RGB-like version for cropping for other multi-channel cases
            cropped_img_np = img_np_rgb_like[first_row:last_row + 1, first_col:last_col + 1, :]

        cropped_img_pil = Image.fromarray(cropped_img_np)

    # Pad the cropped image to a square
    width, height = cropped_img_pil.size
    if width == height:
        return cropped_img_pil

    max_dim = max(width, height)

    current_mode = cropped_img_pil.mode
    # Ensure mode is suitable for padding (L or RGB)
    if current_mode not in ['L', 'RGB']:
        # print(f"Warning: Cropped image mode {current_mode} not L or RGB. Converting to RGB for padding.")
        cropped_img_pil = cropped_img_pil.convert('RGB')
        current_mode = 'RGB'

    fill_color = (0, 0, 0) if current_mode == 'RGB' else 0
    padded_img = Image.new(current_mode, (max_dim, max_dim), fill_color)

    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2
    padded_img.paste(cropped_img_pil, (paste_x, paste_y))

    return padded_img


# --- ModelConfig definition (copied from segmentation_model.py for loading checkpoint) ---
@dataclass
class ModelConfig:
    # Basic Config
    MODEL_NAME: str = "ResNet34UNetPlusPlus"
    RANDOM_SEED: int = 42
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset Config
    DATA_DIR: str = "dataset"
    TRAIN_IMG_DIR: str = "dataset/images"
    TRAIN_MASK_DIR: str = "dataset/masks"
    IMAGE_SIZE: int = 256
    BATCH_SIZE: int = 4
    TRAIN_VAL_SPLIT: float = 0.8
    NUM_WORKERS: int = 1

    # Model Config (ResNet34UNetPlusPlus specific)
    IN_CHANNELS: int = 3
    OUT_CHANNELS: int = 1
    ENCODER_PRETRAINED: bool = True
    DECODER_FEATURES_START: int = 32
    DEEP_SUPERVISION: bool = True
    DROPOUT_P: float = 0.4

    # Training Config
    NUM_EPOCHS: int = 200
    LEARNING_RATE: float = 1e-4
    OPTIMIZER_NAME: str = "Adam"
    LOSS_FUNCTION: str = "BCEWithLogitsLoss"
    USE_AMP: bool = True

    # LR Scheduler Config
    USE_LR_SCHEDULER: bool = False
    LR_SCHEDULER_PATIENCE: int = 10
    LR_SCHEDULER_FACTOR: float = 0.1
    LR_SCHEDULER_MIN_LR: float = 1e-7

    # MixUp Config
    USE_MIXUP: bool = False
    MIXUP_ALPHA: float = 0.4

    # Early Stopping Config
    EARLY_STOPPING_PATIENCE: int = 40
    EARLY_STOPPING_MIN_DELTA: float = 0.0001
    EARLY_STOPPING_METRIC: str = "dice"

    # Save Config
    SAVE_DIR: str = "checkpoints_resnet34_unetplusplus"

    # Visualization Config
    VISUALIZE_PREDICTIONS: bool = True
    NUM_VISUALIZATION_SAMPLES: int = 5
    FIGURE_DIR: str = "figures_resnet34_unetplusplus"

    def __post_init__(self):
        if self.DEVICE == 'cuda' and not torch.cuda.is_available():
            # print("WARNING: CUDA not available, switching to CPU.") # Avoid print during app load
            self.DEVICE = 'cpu'


# --- Model definitions ---

# Helper Class CBAM for TumorClassifierCBAM (from classification_model.py)
class CBAM(torch.nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        # 通道注意力
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(channel, channel // reduction, 1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid_channel = torch.nn.Sigmoid()
        # 空间注意力
        self.conv_spatial = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = torch.nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        x = x * self.sigmoid_channel(avg_out + max_out)
        # 空间注意力
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        x = x * self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_map, max_map], dim=1)))
        return x


# Updated TumorClassifier (from classification_model.py as TumorClassifierCBAM)
class TumorClassifierCBAM(torch.nn.Module):
    def __init__(self, dropout_p=0.7):
        super().__init__()
        # 1) 预训练 EfficientNet-B0 (changed from B3 to match checkpoint),不要它的分类头
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)  # Changed to B0 and use weights
        self.features = backbone.features
        # 2) 在主干最后插一个 CBAM
        self.cbam = CBAM(channel=1280, reduction=16, kernel_size=7)  # Changed channel to 1280 for EfficientNet-B0
        # 3) 全局池化
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        # 4) 分类头：BatchNorm1d -> Dropout -> Linear
        in_feats = backbone.classifier[1].in_features
        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_feats),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.Linear(in_feats, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


# Helper classes for ResNet34UNetPlusPlus (from segmentation_model.py)
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(ConvBlock, self).__init__()
        layers = [
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        ]
        if dropout_p > 0 and dropout_p <= 1:
            layers.append(torch.nn.Dropout2d(dropout_p))
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        inter_channels = in_channels // 8
        if inter_channels == 0:
            inter_channels = 1

        self.query_conv = torch.nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out


class AttentionGate(torch.nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = torch.nn.Sequential(
            torch.nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(F_int)
        )
        self.W_x = torch.nn.Sequential(
            torch.nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(F_int)
        )
        self.psi = torch.nn.Sequential(
            torch.nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid()
        )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_input = self.relu(g1 + x1)
        alpha = self.psi(psi_input)
        return x * alpha


# Updated Segmentation Model (from segmentation_model.py as ResNet34UNetPlusPlus)
class ResNet34UNetPlusPlus(torch.nn.Module):
    def __init__(self, out_channels=1, decoder_features_start=32, deep_supervision=False,
                 pretrained_encoder=True, dropout_p=0.2):
        super(ResNet34UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained_encoder else None)

        self.encoder_x0_0_prepool = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.encoder_x1_0 = resnet.layer1
        self.encoder_x2_0 = resnet.layer2
        self.encoder_x3_0 = resnet.layer3
        self.encoder_x4_0 = resnet.layer4

        enc0_0_ch = 64
        enc1_0_ch = 64
        enc2_0_ch = 128
        enc3_0_ch = 256
        enc4_0_ch = 512

        self.self_attention_bottleneck = SelfAttention(enc4_0_ch)

        self.filters = [
            decoder_features_start,
            decoder_features_start * 2,
            decoder_features_start * 4,
            decoder_features_start * 8
        ]

        self.att_enc0 = AttentionGate(F_g=enc1_0_ch, F_l=enc0_0_ch, F_int=enc0_0_ch // 2)
        self.att_enc1 = AttentionGate(F_g=enc2_0_ch, F_l=enc1_0_ch, F_int=enc1_0_ch // 2)
        self.att_enc2 = AttentionGate(F_g=enc3_0_ch, F_l=enc2_0_ch, F_int=enc2_0_ch // 2)
        self.att_enc3 = AttentionGate(F_g=enc4_0_ch, F_l=enc3_0_ch, F_int=enc3_0_ch // 2)

        self.conv0_1 = ConvBlock(enc0_0_ch + enc1_0_ch, self.filters[0], dropout_p=dropout_p)
        self.conv1_1 = ConvBlock(enc1_0_ch + enc2_0_ch, self.filters[1], dropout_p=dropout_p)
        self.conv2_1 = ConvBlock(enc2_0_ch + enc3_0_ch, self.filters[2], dropout_p=dropout_p)
        self.conv3_1 = ConvBlock(enc3_0_ch + enc4_0_ch, self.filters[3], dropout_p=dropout_p)

        self.conv0_2 = ConvBlock(enc0_0_ch + self.filters[0] + self.filters[1], self.filters[0], dropout_p=dropout_p)
        self.conv1_2 = ConvBlock(enc1_0_ch + self.filters[1] + self.filters[2], self.filters[1], dropout_p=dropout_p)
        self.conv2_2 = ConvBlock(enc2_0_ch + self.filters[2] + self.filters[3], self.filters[2], dropout_p=dropout_p)

        self.conv0_3 = ConvBlock(enc0_0_ch + self.filters[0] * 2 + self.filters[1], self.filters[0],
                                 dropout_p=dropout_p)
        self.conv1_3 = ConvBlock(enc1_0_ch + self.filters[1] * 2 + self.filters[2], self.filters[1],
                                 dropout_p=dropout_p)

        self.conv0_4 = ConvBlock(enc0_0_ch + self.filters[0] * 3 + self.filters[1], self.filters[0],
                                 dropout_p=dropout_p)

        if self.deep_supervision:
            self.final1 = torch.nn.Conv2d(self.filters[0], out_channels, kernel_size=1)
            self.final2 = torch.nn.Conv2d(self.filters[0], out_channels, kernel_size=1)
            self.final3 = torch.nn.Conv2d(self.filters[0], out_channels, kernel_size=1)
            self.final4 = torch.nn.Conv2d(self.filters[0], out_channels, kernel_size=1)
        else:
            self.final = torch.nn.Conv2d(self.filters[0], out_channels, kernel_size=1)

    def _upsample_to_match(self, x_to_upsample, x_target_spatial_map):
        return F.interpolate(x_to_upsample, size=x_target_spatial_map.shape[2:], mode='bilinear', align_corners=True)

    def forward(self, x):
        x0_0_orig = self.encoder_x0_0_prepool(x)
        pooled_orig = self.pool(x0_0_orig)
        x1_0_orig = self.encoder_x1_0(pooled_orig)
        x2_0_orig = self.encoder_x2_0(x1_0_orig)
        x3_0_orig = self.encoder_x3_0(x2_0_orig)
        x4_0_orig_before_sa = self.encoder_x4_0(x3_0_orig)
        x4_0_orig = self.self_attention_bottleneck(x4_0_orig_before_sa)

        g_for_x0_0 = self._upsample_to_match(x1_0_orig, x0_0_orig)
        x0_0_feat = self.att_enc0(g=g_for_x0_0, x=x0_0_orig)
        g_for_x1_0 = self._upsample_to_match(x2_0_orig, x1_0_orig)
        x1_0_feat = self.att_enc1(g=g_for_x1_0, x=x1_0_orig)
        g_for_x2_0 = self._upsample_to_match(x3_0_orig, x2_0_orig)
        x2_0_feat = self.att_enc2(g=g_for_x2_0, x=x2_0_orig)
        g_for_x3_0 = self._upsample_to_match(x4_0_orig, x3_0_orig)
        x3_0_feat = self.att_enc3(g=g_for_x3_0, x=x3_0_orig)
        x4_0_feat = x4_0_orig

        up_x1_0 = self._upsample_to_match(x1_0_feat, x0_0_feat)
        x0_1 = self.conv0_1(torch.cat([x0_0_feat, up_x1_0], dim=1))
        up_x2_0 = self._upsample_to_match(x2_0_feat, x1_0_feat)
        x1_1 = self.conv1_1(torch.cat([x1_0_feat, up_x2_0], dim=1))
        up_x3_0 = self._upsample_to_match(x3_0_feat, x2_0_feat)
        x2_1 = self.conv2_1(torch.cat([x2_0_feat, up_x3_0], dim=1))
        up_x4_0 = self._upsample_to_match(x4_0_feat, x3_0_feat)
        x3_1 = self.conv3_1(torch.cat([x3_0_feat, up_x4_0], dim=1))

        up_x1_1 = self._upsample_to_match(x1_1, x0_0_feat)
        x0_2 = self.conv0_2(torch.cat([x0_0_feat, x0_1, up_x1_1], dim=1))
        up_x2_1 = self._upsample_to_match(x2_1, x1_0_feat)
        x1_2 = self.conv1_2(torch.cat([x1_0_feat, x1_1, up_x2_1], dim=1))
        up_x3_1 = self._upsample_to_match(x3_1, x2_0_feat)
        x2_2 = self.conv2_2(torch.cat([x2_0_feat, x2_1, up_x3_1], dim=1))

        up_x1_2 = self._upsample_to_match(x1_2, x0_0_feat)
        x0_3 = self.conv0_3(torch.cat([x0_0_feat, x0_1, x0_2, up_x1_2], dim=1))
        up_x2_2 = self._upsample_to_match(x2_2, x1_0_feat)
        x1_3 = self.conv1_3(torch.cat([x1_0_feat, x1_1, x1_2, up_x2_2], dim=1))

        up_x1_3 = self._upsample_to_match(x1_3, x0_0_feat)
        x0_4 = self.conv0_4(torch.cat([x0_0_feat, x0_1, x0_2, x0_3, up_x1_3], dim=1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)

            out1 = self._upsample_to_match(out1, x)
            out2 = self._upsample_to_match(out2, x)
            out3 = self._upsample_to_match(out3, x)
            out4 = self._upsample_to_match(out4, x)

            return [out4, out3, out2, out1]
        else:
            final_output = self.final(x0_4)
            final_output = self._upsample_to_match(final_output, x)
            return final_output


# ConvNeXtClassifier remains unchanged
class ConvNeXtClassifier(torch.nn.Module):
    def __init__(self, dropout_rate=0.6, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=0, global_pool='')
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.backbone.num_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


# --- Load models ---
# --- Load Classification Model (Optional) ---
classification_model = None
try:
    classification_model = TumorClassifierCBAM().to(device)
    classification_model.load_state_dict(torch.load('models/classification_model.pth', map_location=device))
    classification_model.eval()
    print("✅ Classification model loaded.")
except Exception as e:
    print(f"⚠️ Classification model not loaded: {e}")

try:
    segmentation_model = ResNet34UNetPlusPlus(out_channels=1,
                                              deep_supervision=True,
                                              pretrained_encoder=True,
                                              ).to(device)

    loaded_data = torch.load('models/segmentation_model.pth', map_location=device, weights_only=False)
    print("✅ Segmentation model loaded.")
    if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
        segmentation_model.load_state_dict(loaded_data['model_state_dict'])
    elif isinstance(loaded_data, dict) and not ('model_state_dict' in loaded_data) and all(isinstance(k, str) for k in
                                                                                           loaded_data.keys()):  # Heuristic: if it's a flat dict of tensors, it's likely a state_dict itself
        segmentation_model.load_state_dict(loaded_data)
    else:

        raise RuntimeError("Loaded segmentation model data is not a recognized checkpoint or state_dict format.")

    segmentation_model.eval()
except Exception as e:
    print(f"Segmentation model loading failed: {e}")
    segmentation_model = None

# --- Load Grading Model (Optional) ---
grading_model = None
try:
    grading_model = ConvNeXtClassifier().to(device)
    grading_model.load_state_dict(torch.load('models/grading_model.pth', map_location=device))
    grading_model.eval()
    print("✅ Grading model loaded.")
except Exception as e:
    print(f"⚠️ Grading model not loaded: {e}")

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 初始化视频处理器 ---
video_processor = CystoscopyVideoProcessor(
    classification_model=classification_model,
    segmentation_model=segmentation_model,
    grading_model=grading_model,
    preprocess_transform=preprocess,
    device=device,
    # 平滑参数，使用指数加权移动平均 (EWMA)，smoothed_value = α × current + (1-α) × previous_smoothed
    enable_temporal_smoothing=True,      # 启用平滑
    smoothing_window=5,                  # 窗口大小：5帧
    smoothing_method='exponential'       # 方法：指数加权
)

# --- Language dictionary ---
language_texts = {
    "中文": {
        "title": "膀胱癌AI诊断平台",
        "upload_label": "上传图片或压缩包",
        "predict_button": "开始预测",
        "result_label": "预测结果",
        "threshold_slider_label": "分类阈值 (肿瘤概率)",
        "download_masks_button": "下载所有掩码",
        "clear_button": "一键清除",
        "download_file_label": "下载掩码压缩包",
        "error_box_label": "状态/错误信息",
        "enable_classification_label": "启用分类模型",
        "enable_segmentation_label": "启用分割模型",
        "enable_grading_label": "启用分级模型",
        "lang_code": "中文",
        "肿瘤": "肿瘤",
        "正常": "正常",
        "classification_label_prefix": "分类",
        "tumor_prob_label_prefix": "肿瘤概率",
        "normal_prob_label_prefix": "正常概率",
        "classification_disabled_label": "分类: 未启用",
        "classification_failed_label": "分类: 模型加载失败",
        "segmentation_failed_label": "分割: 模型加载失败或未执行",
        "grading_label_prefix": "分级",
        "grading_disabled_label_short": "未启用",
        "grading_failed_label": "分级: 模型加载失败",
        "segmentation_info_prefix": "分割",
# 新增视频相关文本
        "video_tab": "视频分析",
        "image_tab": "图像分析",
        "video_upload_label": "上传膀胱镜视频",
        "video_process_button": "开始处理视频",
        "video_output_label": "处理后的视频",
        "video_stats_label": "统计信息",
        "video_skip_frames_label": "跳帧处理",
        "video_skip_frames_info": "0=处理所有帧, 1=每隔1帧处理",
        "video_processing": "正在处理视频...",
        "video_no_file": "请先上传视频文件",
        "video_error": "视频处理失败",
        "video_stats_template": """
### 视频分析统计
- **总帧数**: {total_frames}
- **处理帧数**: {processed_frames}
- **检测到肿瘤的帧数**: {tumor_frames} ({tumor_rate:.1f}%)
- **高级别肿瘤帧数**: {high_grade_frames}
- **低级别肿瘤帧数**: {low_grade_frames}

**建议**: {"检测到可疑病变，建议进一步检查" if "{tumor_frames}" != "0" else "未检测到明显异常"}
        """,
    },
    "English": {
        "title": "Bladder Cancer AI Diagnostic Platform",
        "upload_label": "Upload Images or Zip",
        "predict_button": "Start Prediction",
        "result_label": "Prediction Results",
        "threshold_slider_label": "Classification Threshold (Tumor Probability)",
        "download_masks_button": "Download All Masks",
        "clear_button": "Clear All",
        "download_file_label": "Download Masks ZIP",
        "error_box_label": "Status/Error Messages",
        "enable_classification_label": "Enable Classification Model",
        "enable_segmentation_label": "Enable Segmentation Model",
        "enable_grading_label": "Enable Grading Model",
        "lang_code": "English",
        "肿瘤": "Tumor",
        "正常": "Normal",
        "classification_label_prefix": "Classification",
        "tumor_prob_label_prefix": "Tumor Probability",
        "normal_prob_label_prefix": "Normal Probability",
        "classification_disabled_label": "Classification: Disabled",
        "classification_failed_label": "Classification: Model Load Failed",
        "segmentation_failed_label": "Segmentation: Model Load Failed or Not Performed",
        "grading_label_prefix": "Grading",
        "grading_disabled_label_short": "Disabled",
        "grading_failed_label": "Grading: Model Load Failed",
        "segmentation_info_prefix": "Segmentation",
# 新增视频相关文本
        "video_tab": "Video Analysis",
        "image_tab": "Image Analysis",
        "video_upload_label": "Upload Cystoscopy Video",
        "video_process_button": "Process Video",
        "video_output_label": "Processed Video",
        "video_stats_label": "Statistics",
        "video_skip_frames_label": "Skip Frames",
        "video_skip_frames_info": "0=process all frames, 1=process every other frame",
        "video_processing": "Processing video...",
        "video_no_file": "Please upload a video file first",
        "video_error": "Video processing failed",
        "video_stats_template": """
### Video Analysis Statistics
- **Total Frames**: {total_frames}
- **Processed Frames**: {processed_frames}
- **Tumor Detected Frames**: {tumor_frames} ({tumor_rate:.1f}%)
- **High Grade Frames**: {high_grade_frames}
- **Low Grade Frames**: {low_grade_frames}

**Recommendation**: {"Suspicious lesions detected, further examination recommended" if "{tumor_frames}" != "0" else "No obvious abnormality detected"}
        """,
    }
}


# --- Single image prediction ---
def predict_single(img_pil, lang, original_filename, classification_threshold,
                   enable_classification, enable_segmentation, enable_grading):
    img_pil = crop_borders_and_pad_to_square(img_pil)
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
    mask_pil_for_download = None
    filename_for_mask = Path(original_filename).stem + "_mask.png"

    label = "-"
    grade_label = "-"
    description_parts = []

    classification_performed_and_is_tumor = False  # Helper flag
    classification_performed_successfully = False

    if enable_classification:
        if classification_model is not None:
            with torch.no_grad():
                cls_out = classification_model(img_tensor)
                prob = torch.sigmoid(cls_out).item()  # Calculate actual probability
                is_tumor_based_on_threshold = prob > classification_threshold
                classification_performed_successfully = True

                if is_tumor_based_on_threshold:
                    current_label_text = language_texts[lang].get("肿瘤", "肿瘤")
                    classification_performed_and_is_tumor = True
                else:
                    current_label_text = language_texts[lang].get("正常", "正常")

                tumor_prob_percent = prob * 100
                normal_prob_percent = (1 - prob) * 100

            description_parts.append(f"{language_texts[lang]['classification_label_prefix']}: {current_label_text}")
            description_parts.append(f"{language_texts[lang]['tumor_prob_label_prefix']}: {tumor_prob_percent:.1f}%")
            description_parts.append(f"{language_texts[lang]['normal_prob_label_prefix']}: {normal_prob_percent:.1f}%")
        else:
            description_parts.append(language_texts[lang]['classification_failed_label'])

    else:  # Classification is disabled
        description_parts.append(language_texts[lang]['classification_disabled_label'])

    mask_display_img = Image.new('RGB', img_pil.size, (0, 0, 0))

    should_attempt_segmentation = enable_segmentation and \
                                  (not enable_classification or (
                                              enable_classification and classification_performed_and_is_tumor))

    if should_attempt_segmentation:
        if segmentation_model is not None:
            with torch.no_grad():
                seg_outputs = segmentation_model(img_tensor)
                if isinstance(seg_outputs, list):
                    seg_out = seg_outputs[0]
                else:
                    seg_out = seg_outputs
                mask_values = torch.sigmoid(seg_out).squeeze().cpu().numpy()
                if np.any(mask_values > 0.5):
                    mask_binary = (mask_values > 0.5).astype(np.uint8) * 255
                    mask_display_img = Image.fromarray(mask_binary).convert("RGB").resize(img_pil.size)
                    mask_pil_for_download = Image.fromarray(mask_binary).convert("L").resize(img_pil.size)

        else:
            description_parts.append(language_texts[lang]['segmentation_failed_label'])
    elif enable_segmentation and not should_attempt_segmentation and enable_classification and classification_performed_successfully and not classification_performed_and_is_tumor:
        # Segmentation enabled, but classification was Normal, so no segmentation performed by design
        description_parts.append(
            f"{language_texts[lang]['segmentation_info_prefix']}: {language_texts[lang]['正常']}")  # e.g. "Segmentation: Normal"
    elif not enable_segmentation:
        description_parts.append(
            f"{language_texts[lang]['segmentation_info_prefix']}: {language_texts[lang].get('grading_disabled_label_short', '未启用')}")

    should_attempt_grading = enable_grading and \
                             (not enable_classification or (
                                         enable_classification and classification_performed_and_is_tumor))

    grading_prefix = language_texts[lang]['grading_label_prefix']
    if should_attempt_grading:
        if grading_model is not None:
            with torch.no_grad():
                grade_out = grading_model(img_tensor)
                grade_prob_tensor = torch.softmax(grade_out, dim=1)
                grade_idx = torch.argmax(grade_prob_tensor).item()
                if lang == "中文":
                    grade_label = "高级别" if grade_idx == 1 else "低级别"
                else:
                    grade_label = "High Grade" if grade_idx == 1 else "Low Grade"
            description_parts.append(f"{grading_prefix}: {grade_label}")
        else:
            description_parts.append(language_texts[lang]['grading_failed_label'])
            description_parts.append(f"{grading_prefix}: -")
    elif enable_grading and not should_attempt_grading and enable_classification and classification_performed_successfully and not classification_performed_and_is_tumor:
        description_parts.append(f"{grading_prefix}: -")
    elif not enable_grading:
        description_parts.append(
            f"{grading_prefix}: {language_texts[lang].get('grading_disabled_label_short', '未启用')}")

    width_orig, height_orig = img_pil.size
    composite_image = Image.new('RGB', (width_orig * 2, height_orig))
    composite_image.paste(img_pil, (0, 0))
    composite_image.paste(mask_display_img, (width_orig, 0))

    description = "\n".join(filter(None, description_parts))

    return (composite_image, description, mask_pil_for_download, filename_for_mask)


# --- Main predict function ---
def predict(files, lang, classification_threshold,
            enable_classification, enable_segmentation, enable_grading):
    if not files:
        return [], [], "请先上传文件。" if lang == "中文" else "Please upload files first.", gr.update(visible=False)

    gallery_results = []
    masks_data_for_download = []
    error_messages = []
    processed_files_count = 0

    for file_data in files:
        actual_file_path = file_data.name
        original_uploaded_filename = getattr(file_data, 'orig_name', os.path.basename(actual_file_path))
        if not original_uploaded_filename or original_uploaded_filename.startswith("tmp"):
            original_uploaded_filename = os.path.basename(file_data.name)

        current_file_display_name_for_error = original_uploaded_filename

        try:
            if original_uploaded_filename.lower().endswith('.zip'):
                temp_dir_for_extraction = tempfile.mkdtemp()
                current_zip_errors = []

                try:
                    with zipfile.ZipFile(actual_file_path, 'r') as zip_ref:
                        zip_content_names = zip_ref.namelist()
                        image_files_in_zip = [name for name in zip_content_names if
                                              name.lower().endswith(('jpg', 'jpeg', 'png')) and not name.startswith(
                                                  '__MACOSX')]

                        if not image_files_in_zip:
                            error_messages.append(
                                f"ZIP文件 '{original_uploaded_filename}' 中未找到有效图片。" if lang == "中文" else f"No valid images found in ZIP file '{original_uploaded_filename}'.")
                            continue

                        zip_ref.extractall(temp_dir_for_extraction, members=image_files_in_zip)

                    for img_name_in_zip in image_files_in_zip:
                        img_path_in_zip = os.path.join(temp_dir_for_extraction, img_name_in_zip)
                        current_file_display_name_for_error = f"{original_uploaded_filename} -> {img_name_in_zip}"  # Update for error context

                        try:
                            img_pil = Image.open(img_path_in_zip).convert('RGB')
                            composite_img, desc, mask_pil, mask_fname = predict_single(
                                img_pil, lang, img_name_in_zip, classification_threshold,
                                enable_classification, enable_segmentation, enable_grading
                            )
                            gallery_results.append((composite_img, desc))
                            if mask_pil is not None:
                                masks_data_for_download.append((mask_fname, mask_pil))
                            processed_files_count += 1

                        except Image.UnidentifiedImageError:
                            err_msg = f"无法识别 '{current_file_display_name_for_error}' 的图像格式或文件已损坏。" if lang == "中文" else f"Cannot identify image format for '{current_file_display_name_for_error}' or file is corrupted."
                            current_zip_errors.append(err_msg)
                        except Exception as e_inner:
                            err_msg = f"处理 '{current_file_display_name_for_error}' 失败: {type(e_inner).__name__} - {e_inner}" if lang == "中文" else f"Error processing '{current_file_display_name_for_error}': {type(e_inner).__name__} - {e_inner}"
                            current_zip_errors.append(err_msg)
                finally:
                    if os.path.exists(temp_dir_for_extraction):
                        for root, dirs, files_in_dir_loop in os.walk(temp_dir_for_extraction,
                                                                     topdown=False):  # Renamed files to files_in_dir_loop
                            for name in files_in_dir_loop:
                                try:
                                    os.remove(os.path.join(root, name))
                                except OSError:
                                    pass
                            for name_dir in dirs:  # Renamed name to name_dir
                                try:
                                    os.rmdir(os.path.join(root, name_dir))
                                except OSError:
                                    pass
                        try:
                            os.rmdir(temp_dir_for_extraction)
                        except OSError:
                            pass

                if current_zip_errors:
                    error_messages.extend(current_zip_errors)

            else:  # This is the 'else' for 'if .zip'
                img_pil = Image.open(actual_file_path).convert('RGB')
                composite_img, desc, mask_pil, mask_fname = predict_single(
                    img_pil, lang, original_uploaded_filename, classification_threshold,
                    enable_classification, enable_segmentation, enable_grading
                )
                gallery_results.append((composite_img, desc))
                if mask_pil is not None:
                    masks_data_for_download.append((mask_fname, mask_pil))
                processed_files_count += 1

        except FileNotFoundError:
            err_msg = f"文件 '{current_file_display_name_for_error}' 未找到或路径无效。" if lang == "中文" else f"File '{current_file_display_name_for_error}' not found or path is invalid."
            error_messages.append(err_msg)
        except Image.UnidentifiedImageError:
            err_msg = f"无法识别文件 '{current_file_display_name_for_error}' 的图像格式，或文件已损坏。" if lang == "中文" else f"Cannot identify image format for '{current_file_display_name_for_error}', or file is corrupted."
            error_messages.append(err_msg)
        except Exception as e_outer:
            err_msg = f"处理文件 '{current_file_display_name_for_error}' 时发生未知错误: {type(e_outer).__name__} - {e_outer}" if lang == "中文" else f"Unknown error processing file '{current_file_display_name_for_error}': {type(e_outer).__name__} - {e_outer}"
            error_messages.append(err_msg)

    final_status_message = ""
    if error_messages:
        final_status_message += ("错误汇总:\n" if lang == "中文" else "Error Summary:\n") + "\n".join(error_messages)
        if processed_files_count > 0:
            final_status_message += "\n\n"

    if processed_files_count > 0:
        final_status_message += f"{processed_files_count} 张图片处理完成。" if lang == "中文" else f"{processed_files_count} image(s) processed successfully."
    elif not error_messages:
        final_status_message = "未找到有效图像进行处理，或上传内容为空。" if lang == "中文" else "No valid images found to process, or upload was empty."

    download_masks_button_visibility = gr.update(visible=bool(masks_data_for_download))

    return gallery_results, masks_data_for_download, final_status_message.strip(), download_masks_button_visibility


# --- Function to create a zip file for mask download ---
def create_mask_zip(masks_data):
    if not masks_data:
        return None

    temp_zip_dir = tempfile.mkdtemp()
    zip_file_path = os.path.join(temp_zip_dir, "predicted_masks.zip")

    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        for mask_filename, mask_pil_image in masks_data:
            if mask_pil_image:
                img_byte_arr = io.BytesIO()
                mask_pil_image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                zf.writestr(mask_filename, img_byte_arr)
    return zip_file_path


# --- Function to clear all inputs and outputs ---
def clear_all_outputs():
    return (None, None, None, [], 0.5, "", True, True, True, \
            gr.update(visible=False), gr.update(value=None, visible=False))


# --- 新增视频处理函数 ---

def process_video(video_file, lang, classification_threshold,
                  enable_classification, enable_segmentation, enable_grading,
                  show_mask, mask_alpha, skip_frames, progress=gr.Progress()):
    """
    处理视频文件 - 修复版本

    Args:
        video_file: 上传的视频文件路径
        lang: 语言
        classification_threshold: 分类阈值
        enable_classification: 启用分类
        enable_segmentation: 启用分割
        enable_grading: 启用分级
        show_mask: 显示掩膜
        mask_alpha: 掩膜透明度
        skip_frames: 跳帧数
        progress: Gradio进度条

    Returns:
        output_video_path: 处理后的视频路径
        stats_text: 统计信息文本
        error_message: 错误信息
    """
    texts = language_texts[lang]

    if video_file is None:
        return None, "", texts["video_no_file"]

    try:
        progress(0, desc=texts["video_processing"])

        # 创建临时输出文件 - 确保使用绝对路径
        temp_dir = tempfile.gettempdir()
        output_filename = f"processed_video_{os.getpid()}_{int(time.time())}.mp4"
        output_path = os.path.join(temp_dir, output_filename)

        print(f"输入视频: {video_file}")
        print(f"输出路径: {output_path}")

        # 进度回调
        def update_progress(p):
            progress(p, desc=f"{texts['video_processing']} {int(p * 100)}%")

        # 处理视频
        stats = video_processor.process_video(
            input_path=video_file,
            output_path=output_path,
            cls_threshold=classification_threshold,
            show_mask=show_mask,
            mask_alpha=mask_alpha,
            skip_frames=skip_frames,
            enable_classification=enable_classification,
            enable_segmentation=enable_segmentation,
            enable_grading=enable_grading,
            lang=lang,
            progress_callback=update_progress
        )

        # 验证输出文件
        if not os.path.exists(output_path):
            error_msg = f"输出文件不存在: {output_path}"
            print(f"错误: {error_msg}")
            return None, "", error_msg

        file_size = os.path.getsize(output_path)
        print(f"输出文件大小: {file_size / 1024 / 1024:.2f} MB")

        if file_size < 1000:
            error_msg = f"输出文件异常小 ({file_size} bytes)"
            print(f"错误: {error_msg}")
            return None, "", error_msg

        # 生成统计信息
        tumor_rate = (stats['tumor_frames'] / stats['processed_frames'] * 100
                      if stats['processed_frames'] > 0 else 0)

        # 修复字符串格式化问题
        recommendation = "检测到可疑病变,建议进一步检查" if stats['tumor_frames'] > 0 else "未检测到明显异常"

        if lang == "中文":
            stats_text = f"""
### 视频分析统计
- **总帧数**: {stats['total_frames']}
- **处理帧数**: {stats['processed_frames']}
- **检测到肿瘤的帧数**: {stats['tumor_frames']} ({tumor_rate:.1f}%)
- **高级别肿瘤帧数**: {stats['high_grade_frames']}
- **低级别肿瘤帧数**: {stats['low_grade_frames']}

**建议**: {recommendation}
            """
        else:
            recommendation_en = "Suspicious lesions detected, further examination recommended" if stats[
                                                                                                      'tumor_frames'] > 0 else "No obvious abnormality detected"
            stats_text = f"""
### Video Analysis Statistics
- **Total Frames**: {stats['total_frames']}
- **Processed Frames**: {stats['processed_frames']}
- **Tumor Detected Frames**: {stats['tumor_frames']} ({tumor_rate:.1f}%)
- **High Grade Frames**: {stats['high_grade_frames']}
- **Low Grade Frames**: {stats['low_grade_frames']}

**Recommendation**: {recommendation_en}
            """

        progress(1.0, desc="处理完成!" if lang == "中文" else "Processing complete!")

        print(f"✓ 视频处理成功，返回路径: {output_path}")
        return output_path, stats_text.strip(), ""

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"视频处理异常:")
        print(error_details)

        error_msg = f"{texts['video_error']}: {str(e)}"
        return None, "", error_msg

# --- Language switch function ---
def switch_language(selected_lang):
    """扩展的语言切换函数，包含视频组件"""
    texts = language_texts[selected_lang]
    # 调用原有的switch_language函数获取基础更新
    # 添加视频相关组件的更新
    # video_updates = [
    #     gr.update(label=texts["video_upload_label"]),
    #     gr.update(value=texts["video_process_button"]),
    #     gr.update(label=texts["video_output_label"]),
    #     gr.update(label=texts["video_stats_label"]),
    #     gr.update(label=texts["video_skip_frames_label"],
    #               info=texts["video_skip_frames_info"]),
    # ]
    return (
        # 图像分析 Tab 的组件
        gr.update(value=f"<h1 style='font-weight:bold; font-size:32px;'>{texts['title']}</h1>"),
        gr.update(label=texts["upload_label"]),
        gr.update(value=texts["predict_button"]),
        gr.update(label=texts["result_label"]),
        gr.update(label=texts["threshold_slider_label"]),
        gr.update(value=texts["download_masks_button"]),
        gr.update(value=texts["clear_button"]),
        gr.update(label=texts["download_file_label"]),
        gr.update(label=texts["error_box_label"]),
        gr.update(label=texts["enable_classification_label"]),
        gr.update(label=texts["enable_segmentation_label"]),
        gr.update(label=texts["enable_grading_label"]),
        # 视频分析 Tab 的组件
        gr.update(label=texts["video_upload_label"]),
        gr.update(value=texts["video_process_button"]),
        gr.update(label=texts["video_output_label"]),
        gr.update(label=texts["video_stats_label"]),
        gr.update(label=texts["video_skip_frames_label"],
                  info=texts["video_skip_frames_info"]),
    )


# --- Interface(扩展原有界面) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    stored_masks_state = gr.State([])
    initial_lang = "中文"  # Default language
    current_lang_texts = language_texts[initial_lang]

    # 顶部标题和语言选择 (保持原有)
    with gr.Row():
        title = gr.Markdown(f"<h1 style='font-weight:bold; font-size:32px;'>{current_lang_texts['title']}</h1>")
        lang_choice = gr.Radio(["中文", "English"], value=initial_lang, label="语言/Language", interactive=True,
                               container=False)
        # 使用Tab组织图像和视频功能
    with gr.Tabs():
    # Tab 1: 图像分析 (保持原有功能)
        with gr.Tab(current_lang_texts.get("image_tab", "图像分析")):
            with gr.Row():
                with gr.Column(scale=1):
                    upload = gr.Files(label=current_lang_texts["upload_label"], file_types=["image", ".zip"], interactive=True)

                with gr.Row():
                    enable_classification_switch = gr.Checkbox(label=current_lang_texts["enable_classification_label"],
                                                               value=True, interactive=True)
                    enable_segmentation_switch = gr.Checkbox(label=current_lang_texts["enable_segmentation_label"],
                                                             value=True, interactive=True)
                    enable_grading_switch = gr.Checkbox(label=current_lang_texts["enable_grading_label"], value=True,
                                                        interactive=True)

                threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                             label=current_lang_texts["threshold_slider_label"], interactive=True)
                with gr.Row():
                    predict_btn = gr.Button(current_lang_texts["predict_button"])
                    download_masks_btn = gr.Button(current_lang_texts["download_masks_button"], visible=False)
                    clear_btn = gr.Button(current_lang_texts["clear_button"])

            with gr.Column(scale=2):
                output_gallery = gr.Gallery(label=current_lang_texts["result_label"], columns=2, object_fit="contain",
                                            height="auto")
                download_file_output = gr.File(label=current_lang_texts["download_file_label"], interactive=False,
                                               visible=False)
                error_display_box = gr.Textbox(label=current_lang_texts["error_box_label"], interactive=False, lines=3,
                                               value="")

    # Tab 2: 视频分析 (新增)
    with gr.Tab(current_lang_texts.get("video_tab", "视频分析")):
        gr.Markdown("""
        ### 📹 视频处理功能
        上传膀胱镜检查视频，系统将逐帧分析并生成带标注的视频。
        - 自动检测每一帧中的肿瘤
        - 实时分割肿瘤区域
        - 预测肿瘤分级
        """)
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label=current_lang_texts.get("video_upload_label", "上传膀胱镜视频")
                )

                with gr.Accordion("处理参数", open=True):
                    video_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                        label="分类阈值",
                        info="肿瘤检测的置信度阈值"
                    )

                    with gr.Row():
                        video_enable_cls = gr.Checkbox(
                            label="启用分类", value=True
                        )
                        video_enable_seg = gr.Checkbox(
                            label="启用分割", value=True
                        )
                        video_enable_grade = gr.Checkbox(
                            label="启用分级", value=True
                        )

                    video_show_mask = gr.Checkbox(
                        value=True, label="显示分割掩膜"
                    )
                    video_mask_alpha = gr.Slider(
                        0, 1, 0.4, step=0.1, label="掩膜透明度"
                    )
                    video_skip_frames = gr.Slider(
                        0, 10, 0, step=1,
                        label=current_lang_texts.get("video_skip_frames_label", "跳帧处理"),
                        info=current_lang_texts.get("video_skip_frames_info", "0=处理所有帧")
                    )

                video_process_btn = gr.Button(
                    current_lang_texts.get("video_process_button", "开始处理视频"),
                    variant="primary", size="lg"
                )

            with gr.Column(scale=2):
                video_output = gr.Video(
                    label=current_lang_texts.get("video_output_label", "处理后的视频")
                )
                video_stats = gr.Markdown(
                    label=current_lang_texts.get("video_stats_label", "统计信息")
                )
                video_error_box = gr.Textbox(
                    label="错误信息", interactive=False, lines=2, value=""
                )
        #使用提示
        with gr.Accordion("💡 使用提示", open=False):
            gr.Markdown("""
            1. **视频格式**: 支持 MP4, AVI, MOV 等常见格式
            2. **处理时间**: 取决于视频长度，30秒视频约需1-2分钟
            3. **跳帧功能**: 跳帧=1时速度提升约50%，但可能漏检部分帧
            4. **建议设置**: 
               - 初次分析: 跳帧=0，完整分析
               - 快速预览: 跳帧=2-3，快速浏览
            5. **注意**: 长视频(>5分钟)建议使用跳帧功能
            """)
    # --- 事件绑定 ---
    # Predict button click handler
    predict_btn.click(
        fn=predict,
        inputs=[
            upload, lang_choice, threshold_slider,
            enable_classification_switch,
            enable_segmentation_switch,
            enable_grading_switch
        ],
        outputs=[output_gallery, stored_masks_state, error_display_box, download_masks_btn]
    )

    # Download masks button click handler
    download_masks_btn.click(
        fn=create_mask_zip,
        inputs=[stored_masks_state],
        outputs=download_file_output,
    ).then(
        fn=lambda file_path: gr.update(visible=bool(file_path), value=file_path if file_path else None),
        inputs=[download_file_output],
        outputs=[download_file_output]
    )

    # Clear button click handler
    clear_btn.click(
        fn=clear_all_outputs,
        inputs=None,
        outputs=[
            upload, output_gallery, download_file_output, stored_masks_state,
            threshold_slider, error_display_box,
            enable_classification_switch,
            enable_segmentation_switch,
            enable_grading_switch,
            download_masks_btn,
            download_file_output  # Ensure this is also targeted by clear
        ]
    )

    # 视频处理按钮 (新增)
    video_process_btn.click(
        fn=process_video,
        inputs=[
            video_input, lang_choice, video_threshold,
            video_enable_cls, video_enable_seg, video_enable_grade,
            video_show_mask, video_mask_alpha, video_skip_frames
        ],
        outputs=[video_output, video_stats, video_error_box]
    )

    # Language choice change handler
    lang_choice.change(
        fn=switch_language,
        inputs=lang_choice,
        outputs=[
            title,
            upload,
            predict_btn,
            output_gallery,
            threshold_slider,
            download_masks_btn,
            clear_btn,
            download_file_output,
            error_display_box,
            enable_classification_switch,
            enable_segmentation_switch,
            enable_grading_switch,
            # 视频组件
            video_input, video_process_btn, video_output,
            video_stats, video_skip_frames
        ]
    )

# --- Launch ---
demo.launch(share=True)

