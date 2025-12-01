# %% [markdown]
# # UNet 3+ 膀胱镜图像分割训练脚本
# 本脚本实现了基于UNet 3+的医学图像分割模型训练。
# 主要功能：
# 1. 数据加载和预处理
# 2. 模型训练和验证 (支持深度监督)
# 3. 指标计算和可视化
# 4. 模型保存和早停

# %% [markdown]
# ## 导入必要的库

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，确保在无显示设备的环境下也能保存图片
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass, field
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from torchvision import models # Not strictly needed for UNet3+ from scratch
import torch.nn.functional as F

# %% [markdown]
# ## 配置类

# %%
@dataclass
class ModelConfig:
    # 基础配置
    MODEL_NAME: str = "UNet3Plus" # 模型名称，用于保存和加载
    RANDOM_SEED: int = 42
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据集配置
    DATA_DIR: str = "dataset" 
    TRAIN_IMG_DIR: str = "dataset/images"
    TRAIN_MASK_DIR: str = "dataset/masks"
    IMAGE_SIZE: int = 256
    BATCH_SIZE: int = 4 # UNet3+ can be memory intensive
    TRAIN_VAL_SPLIT: float = 0.8
    NUM_WORKERS: int = 1 
    
    # 模型配置 (UNet 3+)
    IN_CHANNELS: int = 3
    OUT_CHANNELS: int = 1
    FEATURES_START: int = 64 # 初始特征数
    DEEP_SUPERVISION: bool = True # 是否使用深度监督

    # 训练配置
    NUM_EPOCHS: int = 100
    LEARNING_RATE: float = 1e-4 
    OPTIMIZER_NAME: str = "Adam" # "Adam", "AdamW"
    LOSS_FUNCTION: str = "BCEWithLogitsLoss" # "BCEWithLogitsLoss", "DiceLoss", "FocalLoss", "DiceBCELoss"
    USE_AMP: bool = True # 是否使用混合精度训练
    
    # 早停配置
    EARLY_STOPPING_PATIENCE: int = 15 # Increased patience for potentially complex model
    EARLY_STOPPING_MIN_DELTA: float = 0.001 
    EARLY_STOPPING_METRIC: str = "dice" 
    
    # 保存配置
    SAVE_DIR: str = "checkpoints_unet3plus"
    
    # 可视化配置
    VISUALIZE_PREDICTIONS: bool = True
    NUM_VISUALIZATION_SAMPLES: int = 5
    FIGURE_DIR: str = "figures_unet3plus"

    def __post_init__(self):
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(self.FIGURE_DIR, exist_ok=True)
        if self.DEVICE == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA不可用，切换到CPU")
            self.DEVICE = 'cpu'

# %% [markdown]
# ## 数据增强

# %%
class DataTransforms:
    def __init__(self, image_size: int):
        self.image_size = image_size
        # For UNet 3+ trained from scratch, ImageNet normalization might not be optimal,
        # but it's a common default. Consider dataset-specific normalization if performance is low.
        self.train_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Or ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ToTensorV2(),
        ])
        self.val_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])

    def get_train_transforms(self):
        return self.train_transform

    def get_val_transforms(self):
        return self.val_transform

# %% [markdown]
# ## 数据集类

# %%
class BladderDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        base_name, img_ext = os.path.splitext(img_name)
        mask_name_options = [
            f"{base_name}_mask.png",
            f"{base_name}_mask{img_ext}",
            f"{base_name}.png",
        ]
        
        mask_path = None
        for mn_opt in mask_name_options:
            potential_mask_path = os.path.join(self.mask_dir, mn_opt)
            if os.path.exists(potential_mask_path):
                mask_path = potential_mask_path
                break
        
        if mask_path is None:
            raise FileNotFoundError(f"Mask for image {img_name} not found. Looked for: {mask_name_options}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) 
        mask[mask > 0] = 1.0 

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0) 

        return image, mask

# %% [markdown]
# ## UNet 3+ 模型定义

# %%
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_convs - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNet3Plus(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, feature_scale=64, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        filters = [feature_scale, feature_scale * 2, feature_scale * 4, feature_scale * 8, feature_scale * 16]

        # Encoder
        self.enc1_conv = ConvBlock(in_channels, filters[0])
        self.enc2_conv = ConvBlock(filters[0], filters[1])
        self.enc3_conv = ConvBlock(filters[1], filters[2])
        self.enc4_conv = ConvBlock(filters[2], filters[3])
        self.enc5_conv = ConvBlock(filters[3], filters[4]) # Bottleneck

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder components
        # For transforming skip connections from encoder
        # To Decoder 4 (filters[3] output channels ideally, but paper uses fixed num_filters for concat inputs)
        # Let's use `feature_scale` (F) as the channel count for each component before concatenation
        F_concat = feature_scale 
        
        # Decoder 4 (output H/8 x W/8, aiming for filters[3] or F channels output)
        self.hd4_conv_e1 = nn.Sequential(nn.MaxPool2d(8), nn.Conv2d(filters[0], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd4_conv_e2 = nn.Sequential(nn.MaxPool2d(4), nn.Conv2d(filters[1], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd4_conv_e3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(filters[2], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd4_conv_e4 = nn.Sequential(nn.Conv2d(filters[3], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd4_conv_d5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                                         nn.Conv2d(filters[4], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.dec4_conv = ConvBlock(F_concat * 5, filters[3]) # Output filters[3] channels

        # Decoder 3 (output H/4 x W/4)
        self.hd3_conv_e1 = nn.Sequential(nn.MaxPool2d(4), nn.Conv2d(filters[0], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd3_conv_e2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(filters[1], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd3_conv_e3 = nn.Sequential(nn.Conv2d(filters[2], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd3_conv_d4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                         nn.Conv2d(filters[3], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd3_conv_d5 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                         nn.Conv2d(filters[4], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.dec3_conv = ConvBlock(F_concat * 5, filters[2]) # Output filters[2] channels

        # Decoder 2 (output H/2 x W/2)
        self.hd2_conv_e1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(filters[0], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd2_conv_e2 = nn.Sequential(nn.Conv2d(filters[1], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd2_conv_d3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                         nn.Conv2d(filters[2], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd2_conv_d4 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                         nn.Conv2d(filters[3], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd2_conv_d5 = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
                                         nn.Conv2d(filters[4], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.dec2_conv = ConvBlock(F_concat * 5, filters[1]) # Output filters[1] channels

        # Decoder 1 (output H x W)
        self.hd1_conv_e1 = nn.Sequential(nn.Conv2d(filters[0], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd1_conv_d2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                         nn.Conv2d(filters[1], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd1_conv_d3 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                         nn.Conv2d(filters[2], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd1_conv_d4 = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
                                         nn.Conv2d(filters[3], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.hd1_conv_d5 = nn.Sequential(nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
                                         nn.Conv2d(filters[4], F_concat, 1), nn.BatchNorm2d(F_concat), nn.ReLU(inplace=True))
        self.dec1_conv = ConvBlock(F_concat * 5, filters[0]) # Output filters[0] channels

        # Deep supervision classifiers
        if self.deep_supervision:
            self.ds_out5 = nn.Conv2d(filters[4], n_classes, kernel_size=3, padding=1)
            self.ds_out4 = nn.Conv2d(filters[3], n_classes, kernel_size=3, padding=1)
            self.ds_out3 = nn.Conv2d(filters[2], n_classes, kernel_size=3, padding=1)
            self.ds_out2 = nn.Conv2d(filters[1], n_classes, kernel_size=3, padding=1)
            self.ds_out1 = nn.Conv2d(filters[0], n_classes, kernel_size=3, padding=1)
        else:
            self.final_conv = nn.Conv2d(filters[0], n_classes, kernel_size=1) # Or 3x3 like ds_out1
            

    def forward(self, x):
        # Encoder
        e1 = self.enc1_conv(x)    # H, W, F0
        e1_pool = self.pool(e1)

        e2 = self.enc2_conv(e1_pool) # H/2, W/2, F1
        e2_pool = self.pool(e2)

        e3 = self.enc3_conv(e2_pool) # H/4, W/4, F2
        e3_pool = self.pool(e3)

        e4 = self.enc4_conv(e3_pool) # H/8, W/8, F3
        e4_pool = self.pool(e4)

        e5_bottleneck = self.enc5_conv(e4_pool) # H/16, W/16, F4 (bottleneck)

        # Decoder
        # Decoder 4
        hd4_skip_e1 = self.hd4_conv_e1(e1)
        hd4_skip_e2 = self.hd4_conv_e2(e2)
        hd4_skip_e3 = self.hd4_conv_e3(e3)
        hd4_skip_e4 = self.hd4_conv_e4(e4)
        hd4_skip_d5 = self.hd4_conv_d5(e5_bottleneck)
        hd4_input = torch.cat((hd4_skip_e1, hd4_skip_e2, hd4_skip_e3, hd4_skip_e4, hd4_skip_d5), dim=1)
        hd4_out = self.dec4_conv(hd4_input) # H/8, W/8, F3 channels

        # Decoder 3
        hd3_skip_e1 = self.hd3_conv_e1(e1)
        hd3_skip_e2 = self.hd3_conv_e2(e2)
        hd3_skip_e3 = self.hd3_conv_e3(e3)
        hd3_skip_d4 = self.hd3_conv_d4(hd4_out)
        hd3_skip_d5 = self.hd3_conv_d5(e5_bottleneck)
        hd3_input = torch.cat((hd3_skip_e1, hd3_skip_e2, hd3_skip_e3, hd3_skip_d4, hd3_skip_d5), dim=1)
        hd3_out = self.dec3_conv(hd3_input) # H/4, W/4, F2 channels
        
        # Decoder 2
        hd2_skip_e1 = self.hd2_conv_e1(e1)
        hd2_skip_e2 = self.hd2_conv_e2(e2)
        hd2_skip_d3 = self.hd2_conv_d3(hd3_out)
        hd2_skip_d4 = self.hd2_conv_d4(hd4_out)
        hd2_skip_d5 = self.hd2_conv_d5(e5_bottleneck)
        hd2_input = torch.cat((hd2_skip_e1, hd2_skip_e2, hd2_skip_d3, hd2_skip_d4, hd2_skip_d5), dim=1)
        hd2_out = self.dec2_conv(hd2_input) # H/2, W/2, F1 channels

        # Decoder 1
        hd1_skip_e1 = self.hd1_conv_e1(e1)
        hd1_skip_d2 = self.hd1_conv_d2(hd2_out)
        hd1_skip_d3 = self.hd1_conv_d3(hd3_out)
        hd1_skip_d4 = self.hd1_conv_d4(hd4_out)
        hd1_skip_d5 = self.hd1_conv_d5(e5_bottleneck)
        hd1_input = torch.cat((hd1_skip_e1, hd1_skip_d2, hd1_skip_d3, hd1_skip_d4, hd1_skip_d5), dim=1)
        hd1_out = self.dec1_conv(hd1_input) # H, W, F0 channels

        if self.deep_supervision:
            out5 = self.ds_out5(e5_bottleneck)
            out5 = F.interpolate(out5, size=x.size()[2:], mode='bilinear', align_corners=True)

            out4 = self.ds_out4(hd4_out)
            out4 = F.interpolate(out4, size=x.size()[2:], mode='bilinear', align_corners=True)

            out3 = self.ds_out3(hd3_out)
            out3 = F.interpolate(out3, size=x.size()[2:], mode='bilinear', align_corners=True)

            out2 = self.ds_out2(hd2_out)
            out2 = F.interpolate(out2, size=x.size()[2:], mode='bilinear', align_corners=True)
            
            out1 = self.ds_out1(hd1_out)
            # out1 is already at full resolution if input is HxW

            return [out1, out2, out3, out4, out5] # Main output first
        else:
            # final_out = self.final_conv(hd1_out) # If using a different final conv
            final_out = self.ds_out1(hd1_out) # Re-use the same structure as one of the DS heads
            return final_out


# %% [markdown]
# ## 早停类

# %%
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value = None
        self.early_stop = False

        if self.mode == 'min':
            self.delta_op = lambda current, best: best - current > self.min_delta
            self.best_op = lambda current, best: current < best
        elif self.mode == 'max':
            self.delta_op = lambda current, best: current - best > self.min_delta
            self.best_op = lambda current, best: current > best
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Choose 'min' or 'max'.")

    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.best_op(current_value, self.best_value):
            if self.delta_op(current_value, self.best_value):
                self.best_value = current_value
                self.counter = 0
                if self.verbose:
                    print(f"EarlyStopping: New best value: {self.best_value:.6f}")
            else:
                self.counter += 1
                if self.verbose:
                    print(f"EarlyStopping: No significant improvement. Counter: {self.counter}/{self.patience}. Best: {self.best_value:.6f}, Current: {current_value:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: Metric did not improve. Counter: {self.counter}/{self.patience}. Best: {self.best_value:.6f}, Current: {current_value:.6f}")

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"EarlyStopping: Stopping early after {self.patience} epochs of no improvement.")
        
        return self.early_stop

# %% [markdown]
# ## 指标计算

# %%
class MetricsCalculator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def _calculate_stats(self, y_pred_sigmoid, y_true):
        y_pred = (y_pred_sigmoid > self.threshold).float()
        
        tp = torch.sum((y_pred == 1) & (y_true == 1)).item()
        fp = torch.sum((y_pred == 1) & (y_true == 0)).item()
        fn = torch.sum((y_pred == 0) & (y_true == 1)).item()
        tn = torch.sum((y_pred == 0) & (y_true == 0)).item()
        
        return tp, fp, fn, tn

    def calculate_metrics(self, outputs, masks):
        # Outputs are logits. If deep supervision, this should be the primary output.
        y_pred_sigmoid = torch.sigmoid(outputs)
        y_true = masks
        
        tp, fp, fn, tn = self._calculate_stats(y_pred_sigmoid, y_true)
        
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        
        return {
            "dice": dice,
            "iou": iou,
            "sensitivity": sensitivity,
            "specificity": specificity
        }

# %% [markdown]
# ## 可视化工具

# %%
class Visualizer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.figure_dir = config.FIGURE_DIR
        os.makedirs(self.figure_dir, exist_ok=True)

    def plot_training_curves(self, history, epoch):
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Training Progress - Epoch {epoch+1} - Model: {self.config.MODEL_NAME}', fontsize=16)

        axs[0, 0].plot(history['train_loss'], label='Train Loss')
        axs[0, 0].plot(history['val_loss'], label='Validation Loss')
        axs[0, 0].set_title('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].plot(history['train_dice'], label='Train Dice')
        axs[0, 1].plot(history['val_dice'], label='Validation Dice')
        axs[0, 1].set_title('Dice Coefficient')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Dice')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        axs[0, 2].plot(history['train_iou'], label='Train IoU')
        axs[0, 2].plot(history['val_iou'], label='Validation IoU')
        axs[0, 2].set_title('IoU (Jaccard)')
        axs[0, 2].set_xlabel('Epoch')
        axs[0, 2].set_ylabel('IoU')
        axs[0, 2].legend()
        axs[0, 2].grid(True)

        axs[1, 0].plot(history['train_sensitivity'], label='Train Sensitivity')
        axs[1, 0].plot(history['val_sensitivity'], label='Validation Sensitivity')
        axs[1, 0].set_title('Sensitivity (Recall)')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Sensitivity')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        axs[1, 1].plot(history['train_specificity'], label='Train Specificity')
        axs[1, 1].plot(history['val_specificity'], label='Validation Specificity')
        axs[1, 1].set_title('Specificity')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Specificity')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        fig.delaxes(axs[1, 2])
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        save_path = os.path.join(self.figure_dir, f"{self.config.MODEL_NAME}_training_curves_epoch_{epoch+1}.png")
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
        plt.close(fig)

    def visualize_predictions(self, model, dataloader, device, num_samples=5, epoch="final"):
        if not self.config.VISUALIZE_PREDICTIONS:
            return
            
        model.eval()
        samples_shown = 0
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        fig.suptitle(f"Sample Predictions ({self.config.MODEL_NAME} - Epoch {epoch})", fontsize=16)
        axes_flat = axes.flat if num_samples > 1 else axes 

        with torch.no_grad():
            for i, (images, masks_true) in enumerate(dataloader):
                if samples_shown >= num_samples:
                    break
                
                images = images.to(device)
                masks_true = masks_true.to(device) 
                
                model_outputs = model(images)
                # If deep supervision, model_outputs is a list. Use the primary output for visualization.
                outputs_logits = model_outputs[0] if isinstance(model_outputs, list) else model_outputs
                
                outputs_sigmoid = torch.sigmoid(outputs_logits)
                masks_pred = (outputs_sigmoid > 0.5).float()

                for j in range(images.size(0)):
                    if samples_shown >= num_samples:
                        break

                    img_np = images[j].cpu().permute(1, 2, 0).numpy()
                    # Unnormalize based on the normalization used in DataTransforms
                    # If mean=[0.5,...], std=[0.5,...] -> (val * 0.5) + 0.5
                    # If ImageNet -> (val * std) + mean
                    current_mean = np.array([0.5, 0.5, 0.5]) # Match DataTransforms
                    current_std = np.array([0.5, 0.5, 0.5])
                    img_np = current_std * img_np + current_mean
                    img_np = np.clip(img_np, 0, 1)

                    mask_true_np = masks_true[j].cpu().squeeze().numpy()
                    mask_pred_np = masks_pred[j].cpu().squeeze().numpy()

                    ax_idx = samples_shown * 3 if num_samples > 1 else 0
                    axes_flat[ax_idx].imshow(img_np)
                    axes_flat[ax_idx].set_title("Image")
                    axes_flat[ax_idx].axis('off')

                    axes_flat[ax_idx + 1].imshow(mask_true_np, cmap='gray')
                    axes_flat[ax_idx + 1].set_title("True Mask")
                    axes_flat[ax_idx + 1].axis('off')

                    axes_flat[ax_idx + 2].imshow(mask_pred_np, cmap='gray')
                    axes_flat[ax_idx + 2].set_title("Predicted Mask")
                    axes_flat[ax_idx + 2].axis('off')
                    
                    samples_shown += 1
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.figure_dir, f"{self.config.MODEL_NAME}_sample_predictions_epoch_{epoch}.png")
        plt.savefig(save_path)
        print(f"Sample predictions saved to {save_path}")
        plt.close(fig)

# %% [markdown]
# ## 模型训练器

# %%
class ModelTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._set_seed()
        
        self.device = torch.device(config.DEVICE)
        self.data_transforms = DataTransforms(config.IMAGE_SIZE)
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer(config)

        self.model = self._get_model().to(self.device)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_loss_function()
        self.scaler = GradScaler(enabled=config.USE_AMP)
        
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            mode='max' if config.EARLY_STOPPING_METRIC in ["dice", "iou", "sensitivity"] else 'min',
            verbose=True
        )
        
        self.history = {
            metric: [] for metric in [
                "train_loss", "val_loss", 
                "train_dice", "val_dice",
                "train_iou", "val_iou",
                "train_sensitivity", "val_sensitivity",
                "train_specificity", "val_specificity"
            ]
        }

    def _set_seed(self):
        torch.manual_seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.RANDOM_SEED)

    def _get_model(self):
        if self.config.MODEL_NAME == "UNet3Plus":
            model = UNet3Plus(
                in_channels=self.config.IN_CHANNELS,
                n_classes=self.config.OUT_CHANNELS,
                feature_scale=self.config.FEATURES_START,
                deep_supervision=self.config.DEEP_SUPERVISION
            )
        else:
            raise ValueError(f"Unsupported model: {self.config.MODEL_NAME}")
        return model

    def _get_optimizer(self):
        if self.config.OPTIMIZER_NAME.lower() == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER_NAME.lower() == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.OPTIMIZER_NAME}")
        return optimizer

    def _get_loss_function(self):
        if self.config.LOSS_FUNCTION == "BCEWithLogitsLoss":
            criterion = nn.BCEWithLogitsLoss()
        # Add other loss functions here if needed
        else:
            raise ValueError(f"Unsupported loss function: {self.config.LOSS_FUNCTION}")
        return criterion

    def _train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0
        epoch_metrics = {"dice": 0, "iou": 0, "sensitivity": 0, "specificity": 0}
        
        for images, masks in tqdm(dataloader, desc="Training"):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            with autocast(enabled=self.config.USE_AMP):
                outputs = self.model(images) # Can be a list if deep_supervision is True
                
                if self.config.DEEP_SUPERVISION and isinstance(outputs, list):
                    current_loss = 0
                    for out_sup in outputs:
                        current_loss += self.criterion(out_sup, masks)
                    current_loss /= len(outputs)
                    # For metrics, use the primary output (e.g., the first one)
                    primary_output_for_metrics = outputs[0]
                else:
                    current_loss = self.criterion(outputs, masks)
                    primary_output_for_metrics = outputs
            
            self.scaler.scale(current_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += current_loss.item()
            batch_metrics = self.metrics_calculator.calculate_metrics(primary_output_for_metrics.detach(), masks)
            for key in epoch_metrics:
                epoch_metrics[key] += batch_metrics[key]
        
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_metrics = {key: val / num_batches for key, val in epoch_metrics.items()}
        return avg_loss, avg_metrics

    def _validate_epoch(self, dataloader):
        self.model.eval()
        epoch_loss = 0
        epoch_metrics = {"dice": 0, "iou": 0, "sensitivity": 0, "specificity": 0}
        
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="Validating"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                with autocast(enabled=self.config.USE_AMP):
                    outputs = self.model(images) # Can be a list
                    if self.config.DEEP_SUPERVISION and isinstance(outputs, list):
                        current_loss = 0
                        for out_sup in outputs:
                            current_loss += self.criterion(out_sup, masks)
                        current_loss /= len(outputs)
                        primary_output_for_metrics = outputs[0]
                    else:
                        current_loss = self.criterion(outputs, masks)
                        primary_output_for_metrics = outputs
                
                epoch_loss += current_loss.item()
                batch_metrics = self.metrics_calculator.calculate_metrics(primary_output_for_metrics, masks)
                for key in epoch_metrics:
                    epoch_metrics[key] += batch_metrics[key]
            
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_metrics = {key: val / num_batches for key, val in epoch_metrics.items()}
        return avg_loss, avg_metrics

    def _save_checkpoint(self, epoch, metric_value, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            f'val_{self.config.EARLY_STOPPING_METRIC}': metric_value,
            'config': self.config # Save config with checkpoint
        }
        if self.scaler:
             checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        filename_suffix = "best" if is_best else f"epoch_{epoch+1}"
        save_path = os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_NAME}_{filename_suffix}.pth")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path} (Validation {self.config.EARLY_STOPPING_METRIC}: {metric_value:.4f})")
        if is_best:
             print(f"*** New best model saved based on validation {self.config.EARLY_STOPPING_METRIC}! ***")

    def train(self):
        print(f"Starting training for {self.config.MODEL_NAME} on {self.device}")
        print(f"Configuration: {self.config}")

        full_dataset = BladderDataset(
            image_dir=self.config.TRAIN_IMG_DIR,
            mask_dir=self.config.TRAIN_MASK_DIR,
            transform=self.data_transforms.get_train_transforms()
        )
        
        val_transform = self.data_transforms.get_val_transforms()
        
        train_size = int(self.config.TRAIN_VAL_SPLIT * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        generator = torch.Generator().manual_seed(self.config.RANDOM_SEED)
        # Ensure split is done on indices, then create Subsets with respective transforms
        all_indices = list(range(len(full_dataset)))
        train_indices, val_indices = torch.utils.data.random_split(all_indices, [train_size, val_size], generator=generator)

        # Create dataset instances for train and val with their specific transforms
        train_dataset_obj = BladderDataset(image_dir=self.config.TRAIN_IMG_DIR, mask_dir=self.config.TRAIN_MASK_DIR, transform=self.data_transforms.get_train_transforms())
        val_dataset_obj = BladderDataset(image_dir=self.config.TRAIN_IMG_DIR, mask_dir=self.config.TRAIN_MASK_DIR, transform=val_transform)
        
        train_dataset = torch.utils.data.Subset(train_dataset_obj, train_indices.indices)
        val_dataset = torch.utils.data.Subset(val_dataset_obj, val_indices.indices)


        train_loader = DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, 
            num_workers=self.config.NUM_WORKERS, pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True # May help if batch size is small and last batch causes issues with BN
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False,
            num_workers=self.config.NUM_WORKERS, pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
        if len(train_loader) == 0 or len(val_loader) == 0:
            print("ERROR: One of the dataloaders is empty. Check dataset paths, sizes, and batch_size.")
            return
        print(f"Train loader batches: {len(train_loader)}, Validation loader batches: {len(val_loader)}")

        best_metric_val = -float('inf') if self.early_stopping.mode == 'max' else float('inf')
        last_epoch = 0

        for epoch in range(self.config.NUM_EPOCHS):
            last_epoch = epoch
            print(f"\n--- Epoch {epoch+1}/{self.config.NUM_EPOCHS} ---")
            
            train_loss, train_metrics = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate_epoch(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for metric_name in train_metrics: # dice, iou, sensitivity, specificity
                self.history[f'train_{metric_name}'].append(train_metrics[metric_name])
                self.history[f'val_{metric_name}'].append(val_metrics[metric_name])

            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Train Metrics: Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}, Sens: {train_metrics['sensitivity']:.4f}, Spec: {train_metrics['specificity']:.4f}")
            print(f"  Val Metrics:   Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, Sens: {val_metrics['sensitivity']:.4f}, Spec: {val_metrics['specificity']:.4f}")

            self.visualizer.plot_training_curves(self.history, epoch)
            current_metric_val = val_metrics[self.config.EARLY_STOPPING_METRIC]
            
            is_improvement = (self.early_stopping.mode == 'max' and current_metric_val > best_metric_val) or \
                             (self.early_stopping.mode == 'min' and current_metric_val < best_metric_val)
            
            if is_improvement and (abs(current_metric_val - best_metric_val) > self.config.EARLY_STOPPING_MIN_DELTA if best_metric_val != -float('inf') and best_metric_val != float('inf') else True) :
                best_metric_val = current_metric_val
                self._save_checkpoint(epoch, current_metric_val, is_best=True)
            else:
                # Save regular checkpoint less frequently
                if (epoch + 1) % 10 == 0 : 
                     self._save_checkpoint(epoch, current_metric_val, is_best=False)

            if self.early_stopping(current_metric_val): # This call updates early_stopping.counter and best_value
                print("Early stopping triggered.")
                break
        
        print("\nTraining finished.")
        print(f"Best validation {self.config.EARLY_STOPPING_METRIC}: {self.early_stopping.best_value:.4f}")
        
        best_model_path = os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_NAME}_best.pth")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} for final visualization.")
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            # Ensure model config matches if loading, or re-init model with saved config
            loaded_config = checkpoint.get('config', self.config) # Fallback to current if not saved
            if loaded_config.MODEL_NAME != self.config.MODEL_NAME or \
               loaded_config.FEATURES_START != self.config.FEATURES_START or \
               loaded_config.DEEP_SUPERVISION != self.config.DEEP_SUPERVISION:
                 print("Warning: Loaded model config differs. Re-initializing model with loaded config for state_dict compatibility.")
                 self.model = UNet3Plus(in_channels=loaded_config.IN_CHANNELS, n_classes=loaded_config.OUT_CHANNELS,
                                        feature_scale=loaded_config.FEATURES_START, deep_supervision=loaded_config.DEEP_SUPERVISION).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.visualizer.visualize_predictions(self.model, val_loader, self.device, 
                                                  num_samples=self.config.NUM_VISUALIZATION_SAMPLES, epoch="best_model")
        else:
            final_epoch_str = f"epoch_{last_epoch+1}_final_state"
            print(f"No best model checkpoint found or training completed. Visualizing model state from {final_epoch_str}.")
            self.visualizer.visualize_predictions(self.model, val_loader, self.device, 
                                                 num_samples=self.config.NUM_VISUALIZATION_SAMPLES, epoch=final_epoch_str)

# %% [markdown]
# ## 主执行函数

# %%
def main():
    config = ModelConfig(
        MODEL_NAME="UNet3Plus",
        BATCH_SIZE=2, # UNet3+ is very memory intensive, start with a small batch size
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=150, # May need more epochs
        FEATURES_START=32, # Start with fewer features if memory is an issue
        DEEP_SUPERVISION=True,
        EARLY_STOPPING_PATIENCE=20, 
        DATA_DIR="dataset", 
        TRAIN_IMG_DIR="dataset/images",
        TRAIN_MASK_DIR="dataset/masks",
        SAVE_DIR="checkpoints_unet3plus", 
        FIGURE_DIR="figures_unet3plus"
    )

    trainer = ModelTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 