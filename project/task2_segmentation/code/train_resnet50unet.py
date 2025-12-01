# %% [markdown]
# # ResNet50-UNet 膀胱镜图像分割训练脚本
# 本脚本实现了基于ResNet50-UNet的医学图像分割模型训练。
# 主要功能：
# 1. 数据加载和预处理
# 2. 模型训练和验证
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
from torchvision import models
import torch.nn.functional as F

# %% [markdown]
# ## 配置类

# %%
@dataclass
class ModelConfig:
    # 基础配置
    MODEL_NAME: str = "ResNet50UNet" # 模型名称，用于保存和加载
    RANDOM_SEED: int = 42
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据集配置
    DATA_DIR: str = "dataset" 
    TRAIN_IMG_DIR: str = "dataset/images"
    TRAIN_MASK_DIR: str = "dataset/masks"
    IMAGE_SIZE: int = 256
    BATCH_SIZE: int = 8 # ResNet50UNet 可能需要更小的批量大小
    TRAIN_VAL_SPLIT: float = 0.8
    NUM_WORKERS: int = 1 
    
    # 模型配置
    IN_CHANNELS: int = 3
    OUT_CHANNELS: int = 1
    RESNET_PRETRAINED: bool = True # 是否使用ResNet的预训练权重
    
    # 训练配置
    NUM_EPOCHS: int = 100
    LEARNING_RATE: float = 1e-4 # 对于预训练模型，初始学习率可能需要小一些
    OPTIMIZER_NAME: str = "Adam" # "Adam", "AdamW"
    LOSS_FUNCTION: str = "BCEWithLogitsLoss" # "BCEWithLogitsLoss", "DiceLoss", "FocalLoss", "DiceBCELoss"
    USE_AMP: bool = True # 是否使用混合精度训练
    
    # 早停配置
    EARLY_STOPPING_PATIENCE: int = 10
    EARLY_STOPPING_MIN_DELTA: float = 0.001 # Dice Score 的最小提升
    EARLY_STOPPING_METRIC: str = "dice" # 基于哪个指标进行早停
    
    # 保存配置
    SAVE_DIR: str = "checkpoints"
    
    # 可视化配置
    VISUALIZE_PREDICTIONS: bool = True
    NUM_VISUALIZATION_SAMPLES: int = 5

    FIGURE_DIR: str = "figures" # 保存训练曲线图的目录

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
        self.train_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], # ImageNet mean and std
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
        self.val_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
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
        
        # 构建可能的掩码文件名
        base_name, img_ext = os.path.splitext(img_name)
        mask_name_options = [
            f"{base_name}_mask.png",
            f"{base_name}_mask{img_ext}", # e.g., image.jpg -> image_mask.jpg
            f"{base_name}.png", # In case mask has same name but different ext
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
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # Convert to grayscale
        mask[mask > 0] = 1.0 # Binarize mask

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0) # Add channel dimension for mask (H, W) -> (1, H, W)

        return image, mask

# %% [markdown]
# ## ResNet50-UNet 模型定义

# %%
class ResNet50UNet(nn.Module):
    def __init__(self, n_classes=1, pretrained_resnet=True):
        super().__init__()
        self.n_classes = n_classes

        # Encoder (ResNet50)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained_resnet else None)
        
        self.base_model = resnet
        self.base_layers = list(resnet.children())

        # Encoder layers
        self.layer0 = nn.Sequential(*self.base_layers[:3]) #Conv, BN, ReLU
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # MaxPool, Layer1
        self.layer2 = self.base_layers[5]  # Layer2
        self.layer3 = self.base_layers[6]  # Layer3
        self.layer4 = self.base_layers[7]  # Layer4

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024 + 1024, 1024) # Skip connection from layer3 (1024 channels)

        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512 + 512, 512) # Skip connection from layer2 (512 channels)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256 + 256, 256) # Skip connection from layer1 (actual output of resnet.layer1 is 256)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Skip connection from layer0. Output of resnet.maxpool (self.base_layers[3]) has 64 channels.
        # The output of self.layer0 (Conv, BN, ReLU) is 64 channels
        self.dec1 = self.conv_block(128 + 64, 128)

        # Final upsampling & conv
        # After dec1 (output 128 channels), image size is H/2 x W/2 if original input to U-Net is H x W
        # ResNet50 downsamples by 32. If original image is 256x256:
        # layer0 out: 128x128 (after initial conv stride 2, before maxpool)
        # layer1 out: 64x64 (after maxpool and layer1)
        # layer2 out: 32x32
        # layer3 out: 16x16
        # layer4 out: 8x8 (bottleneck)
        # Decoder brings it back:
        # upconv4 + dec4 out: 16x16
        # upconv3 + dec3 out: 32x32
        # upconv2 + dec2 out: 64x64
        # upconv1 + dec1 out: 128x128
        # We need one more upsampling to get to 256x256
        self.upconv0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec0 = self.conv_block(64, 64) # No skip connection here, or could take one from initial input if needed

        self.conv_last = nn.Conv2d(64, n_classes, kernel_size=1)
        # Sigmoid/Softmax is not applied here, will be handled by loss function (BCEWithLogitsLoss) or during inference

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e0 = self.layer0(x)       # Output: 64 channels, H/2 x W/2 (due to initial Conv stride 2)
        e1 = self.layer1(e0)      # Output: 256 channels, H/4 x W/4 (due to MaxPool in layer1)
        e2 = self.layer2(e1)      # Output: 512 channels, H/8 x W/8
        e3 = self.layer3(e2)      # Output: 1024 channels, H/16 x W/16
        e4 = self.layer4(e3)      # Output: 2048 channels, H/32 x W/32 (bottleneck)
        
        # Decoder
        d4 = self.upconv4(e4)               # H/16 x W/16
        d4 = torch.cat((d4, e3), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)               # H/8 x W/8
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)               # H/4 x W/4
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)               # H/2 x W/2
        d1 = torch.cat((d1, e0), dim=1) # e0 is 64 channels
        d1 = self.dec1(d1)
        
        d0 = self.upconv0(d1)               # H x W
        d0 = self.dec0(d0)

        out = self.conv_last(d0)          # H x W
        return out

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
        # outputs are logits, apply sigmoid
        y_pred_sigmoid = torch.sigmoid(outputs)
        y_true = masks
        
        tp, fp, fn, tn = self._calculate_stats(y_pred_sigmoid, y_true)
        
        # Dice Coefficient
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        
        # IoU (Jaccard Index)
        iou = tp / (tp + fp + fn + 1e-8)
        
        # Sensitivity (Recall)
        sensitivity = tp / (tp + fn + 1e-8)
        
        # Specificity
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
        """Plots and saves training and validation metrics."""
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Training Progress - Epoch {epoch+1} - Model: {self.config.MODEL_NAME}', fontsize=16)

        # Loss
        axs[0, 0].plot(history['train_loss'], label='Train Loss')
        axs[0, 0].plot(history['val_loss'], label='Validation Loss')
        axs[0, 0].set_title('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Dice
        axs[0, 1].plot(history['train_dice'], label='Train Dice')
        axs[0, 1].plot(history['val_dice'], label='Validation Dice')
        axs[0, 1].set_title('Dice Coefficient')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Dice')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # IoU
        axs[0, 2].plot(history['train_iou'], label='Train IoU')
        axs[0, 2].plot(history['val_iou'], label='Validation IoU')
        axs[0, 2].set_title('IoU (Jaccard)')
        axs[0, 2].set_xlabel('Epoch')
        axs[0, 2].set_ylabel('IoU')
        axs[0, 2].legend()
        axs[0, 2].grid(True)

        # Sensitivity
        axs[1, 0].plot(history['train_sensitivity'], label='Train Sensitivity')
        axs[1, 0].plot(history['val_sensitivity'], label='Validation Sensitivity')
        axs[1, 0].set_title('Sensitivity (Recall)')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Sensitivity')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Specificity
        axs[1, 1].plot(history['train_specificity'], label='Train Specificity')
        axs[1, 1].plot(history['val_specificity'], label='Validation Specificity')
        axs[1, 1].set_title('Specificity')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Specificity')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # Remove empty subplot
        fig.delaxes(axs[1, 2])

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
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
        
        axes_flat = axes.flat if num_samples > 1 else axes # Handle single sample case

        with torch.no_grad():
            for i, (images, masks_true) in enumerate(dataloader):
                if samples_shown >= num_samples:
                    break
                
                images = images.to(device)
                masks_true = masks_true.to(device) # (B, 1, H, W)
                
                outputs_logits = model(images) # (B, 1, H, W)
                outputs_sigmoid = torch.sigmoid(outputs_logits)
                masks_pred = (outputs_sigmoid > 0.5).float()

                for j in range(images.size(0)):
                    if samples_shown >= num_samples:
                        break

                    img_np = images[j].cpu().permute(1, 2, 0).numpy()
                    # Unnormalize image for visualization
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = std * img_np + mean
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
            mode='max' if config.EARLY_STOPPING_METRIC in ["dice", "iou", "sensitivity"] else 'min', # Adjust mode based on metric
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
        # Potentially add for full determinism, but can slow down training
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


    def _get_model(self):
        if self.config.MODEL_NAME == "ResNet50UNet":
            model = ResNet50UNet(n_classes=self.config.OUT_CHANNELS, pretrained_resnet=self.config.RESNET_PRETRAINED)
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
        # Add other loss functions here if needed (DiceLoss, FocalLoss, etc.)
        # elif self.config.LOSS_FUNCTION == "DiceLoss":
        #     criterion = DiceLoss() 
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
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            batch_metrics = self.metrics_calculator.calculate_metrics(outputs.detach(), masks)
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
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                epoch_loss += loss.item()
                batch_metrics = self.metrics_calculator.calculate_metrics(outputs, masks)
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
            f'val_{self.config.EARLY_STOPPING_METRIC}': metric_value
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

        # Data
        full_dataset = BladderDataset(
            image_dir=self.config.TRAIN_IMG_DIR,
            mask_dir=self.config.TRAIN_MASK_DIR,
            transform=self.data_transforms.get_train_transforms() # Use train transforms for the full dataset initially
        )
        
        # Temporarily set transform to None for val_dataset to apply val_transform later
        # This is a common pattern: split first, then assign different transforms
        val_transform = self.data_transforms.get_val_transforms()
        
        train_size = int(self.config.TRAIN_VAL_SPLIT * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Use a generator for reproducibility in splitting
        generator = torch.Generator().manual_seed(self.config.RANDOM_SEED)
        train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size], generator=generator)

        # Create subsets with correct transforms
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
        # For val_dataset, we need to wrap it to apply the validation transform
        # One way is to create a new BladderDataset instance just for validation, or modify the Subset's __getitem__
        # Simpler: ensure BladderDataset's transform can be updated or make a new one.
        # For now, we'll create a new BladderDataset for validation with validation transforms
        
        # Create validation dataset with validation transforms
        # This requires access to the file lists used by the original full_dataset.
        # A cleaner way is often to have BladderDataset accept indices and apply transform based on mode.
        # Or, more simply, ensure the transform for the Subset is correctly set.
        
        # Re-creating dataset objects for train/val with specific transforms
        train_dataset = BladderDataset(image_dir=self.config.TRAIN_IMG_DIR, mask_dir=self.config.TRAIN_MASK_DIR, transform=self.data_transforms.get_train_transforms())
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices.indices)

        val_dataset_full = BladderDataset(image_dir=self.config.TRAIN_IMG_DIR, mask_dir=self.config.TRAIN_MASK_DIR, transform=val_transform)
        val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices.indices)


        train_loader = DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, 
            num_workers=self.config.NUM_WORKERS, pin_memory=True if self.device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False,
            num_workers=self.config.NUM_WORKERS, pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
        print(f"Train loader batches: {len(train_loader)}, Validation loader batches: {len(val_loader)}")

        best_metric_val = -float('inf') if self.early_stopping.mode == 'max' else float('inf')

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{self.config.NUM_EPOCHS} ---")
            
            train_loss, train_metrics = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate_epoch(val_loader)

            # Store metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for metric_name in train_metrics:
                self.history[f'train_{metric_name}'].append(train_metrics[metric_name])
                self.history[f'val_{metric_name}'].append(val_metrics[metric_name])

            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Train Metrics: Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}, Sens: {train_metrics['sensitivity']:.4f}, Spec: {train_metrics['specificity']:.4f}")
            print(f"  Val Metrics:   Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, Sens: {val_metrics['sensitivity']:.4f}, Spec: {val_metrics['specificity']:.4f}")

            # Update and save training curves plot
            self.visualizer.plot_training_curves(self.history, epoch)

            current_metric_val = val_metrics[self.config.EARLY_STOPPING_METRIC]
            
            if (self.early_stopping.mode == 'max' and current_metric_val > best_metric_val) or \
               (self.early_stopping.mode == 'min' and current_metric_val < best_metric_val):
                best_metric_val = current_metric_val
                self._save_checkpoint(epoch, current_metric_val, is_best=True)
            else:
                # Save regular checkpoint (optional, can be done less frequently)
                if (epoch + 1) % 5 == 0 : # Save every 5 epochs for example
                     self._save_checkpoint(epoch, current_metric_val, is_best=False)


            if self.early_stopping(current_metric_val):
                print("Early stopping triggered.")
                break
        
        print("\nTraining finished.")
        print(f"Best validation {self.config.EARLY_STOPPING_METRIC}: {self.early_stopping.best_value:.4f}")
        
        # Load best model for final visualization
        best_model_path = os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_NAME}_best.pth")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} for final visualization.")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.visualizer.visualize_predictions(self.model, val_loader, self.device, 
                                                  num_samples=self.config.NUM_VISUALIZATION_SAMPLES, epoch="best_model")
        else:
            print("No best model checkpoint found for final visualization.")
            self.visualizer.visualize_predictions(self.model, val_loader, self.device, 
                                                 num_samples=self.config.NUM_VISUALIZATION_SAMPLES, epoch=f"epoch_{epoch+1}_final")


# %% [markdown]
# ## 主执行函数

# %%
def main():
    # 1. 配置
    config = ModelConfig(
        MODEL_NAME="ResNet50UNet",
        BATCH_SIZE=4, # Adjusted for potentially higher memory usage
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=100, # Start with fewer epochs for testing
        EARLY_STOPPING_PATIENCE=15,
        DATA_DIR="dataset", # Make sure this path is correct
        TRAIN_IMG_DIR="dataset/images",
        TRAIN_MASK_DIR="dataset/masks",
        SAVE_DIR="checkpoints_resnet50unet",
        FIGURE_DIR="figures_resnet50unet"
    )

    # 2. 初始化训练器
    trainer = ModelTrainer(config)

    # 3. 开始训练
    trainer.train()

if __name__ == "__main__":
    main() 