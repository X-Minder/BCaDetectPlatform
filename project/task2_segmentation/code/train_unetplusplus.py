# %% [markdown]
# # UNet++ (Nested U-Net) 膀胱镜图像分割训练脚本
# 本脚本实现了基于UNet++的医学图像分割模型训练。
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

# %% [markdown]
# ## 配置类

# %%
@dataclass
class ModelConfig:
    # 基础配置
    MODEL_NAME: str = "UNetPlusPlus"
    RANDOM_SEED: int = 42
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据集配置
    DATA_DIR: str = "dataset" 
    TRAIN_IMG_DIR: str = "dataset/images"
    TRAIN_MASK_DIR: str = "dataset/masks"
    IMAGE_SIZE: int = 256
    BATCH_SIZE: int = 8 
    TRAIN_VAL_SPLIT: float = 0.8
    NUM_WORKERS: int = 1 
    
    # 模型配置 (UNet++ specific)
    IN_CHANNELS: int = 3
    OUT_CHANNELS: int = 1
    FEATURES_START: int = 32 # Initial features, e.g., 32 for UNet++ standard
    DEEP_SUPERVISION: bool = False # Set to True to enable deep supervision (requires loss and trainer modification)

    # 训练配置
    NUM_EPOCHS: int = 100
    LEARNING_RATE: float = 1e-4 
    OPTIMIZER_NAME: str = "Adam"
    LOSS_FUNCTION: str = "BCEWithLogitsLoss"
    USE_AMP: bool = True
    
    # 早停配置
    EARLY_STOPPING_PATIENCE: int = 10
    EARLY_STOPPING_MIN_DELTA: float = 0.001
    EARLY_STOPPING_METRIC: str = "dice"
    
    # 保存配置
    SAVE_DIR: str = "checkpoints" 
    
    # 可视化配置
    VISUALIZE_PREDICTIONS: bool = True
    NUM_VISUALIZATION_SAMPLES: int = 5
    FIGURE_DIR: str = "figures"

    def __post_init__(self):
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
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        self.val_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    def get_train_transforms(self): return self.train_transform
    def get_val_transforms(self): return self.val_transform

# %% [markdown]
# ## 数据集类

# %%
class BladderDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        base_name, img_ext = os.path.splitext(img_name)
        mask_name_options = [f"{base_name}_mask.png", f"{base_name}_mask{img_ext}", f"{base_name}.png"]
        mask_path = next((os.path.join(self.mask_dir, mn_opt) for mn_opt in mask_name_options if os.path.exists(os.path.join(self.mask_dir, mn_opt))), None)
        if mask_path is None: raise FileNotFoundError(f"Mask for image {img_name} not found. Looked for: {mask_name_options}")
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask > 0] = 1.0
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask'].unsqueeze(0)
        return image, mask

# %% [markdown]
# ## UNet++ 模型定义

# %%
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0): # Added dropout
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features_start=32, deep_supervision=False, dropout_p=0.1):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        self.dropout_p = dropout_p

        filters = [features_start, features_start*2, features_start*4, features_start*8, features_start*16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder (X_i_0)
        self.conv0_0 = ConvBlock(in_channels, filters[0], dropout_p=self.dropout_p)
        self.conv1_0 = ConvBlock(filters[0], filters[1], dropout_p=self.dropout_p)
        self.conv2_0 = ConvBlock(filters[1], filters[2], dropout_p=self.dropout_p)
        self.conv3_0 = ConvBlock(filters[2], filters[3], dropout_p=self.dropout_p)
        self.conv4_0 = ConvBlock(filters[3], filters[4], dropout_p=self.dropout_p) # Bottleneck

        # Skip path L1 (X_i_1)
        # X0_0 (filters[0]) + Up(X1_0) (filters[1]) -> filters[0]
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0], dropout_p=self.dropout_p)
        # X1_0 (filters[1]) + Up(X2_0) (filters[2]) -> filters[1]
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1], dropout_p=self.dropout_p)
        # X2_0 (filters[2]) + Up(X3_0) (filters[3]) -> filters[2]
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2], dropout_p=self.dropout_p)
        # X3_0 (filters[3]) + Up(X4_0) (filters[4]) -> filters[3]
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3], dropout_p=self.dropout_p)

        # Skip path L2 (X_i_2)
        # X0_0 (filters[0]) + X0_1 (filters[0]) + Up(X1_1) (filters[1]) -> filters[0]
        self.conv0_2 = ConvBlock(filters[0]*2 + filters[1], filters[0], dropout_p=self.dropout_p)
        # X1_0 (filters[1]) + X1_1 (filters[1]) + Up(X2_1) (filters[2]) -> filters[1]
        self.conv1_2 = ConvBlock(filters[1]*2 + filters[2], filters[1], dropout_p=self.dropout_p)
        # X2_0 (filters[2]) + X2_1 (filters[2]) + Up(X3_1) (filters[3]) -> filters[2]
        self.conv2_2 = ConvBlock(filters[2]*2 + filters[3], filters[2], dropout_p=self.dropout_p)
        
        # Skip path L3 (X_i_3)
        # X0_0 (filters[0]) + X0_1 (filters[0]) + X0_2 (filters[0]) + Up(X1_2) (filters[1]) -> filters[0]
        self.conv0_3 = ConvBlock(filters[0]*3 + filters[1], filters[0], dropout_p=self.dropout_p)
        # X1_0 (filters[1]) + X1_1 (filters[1]) + X1_2 (filters[1]) + Up(X2_2) (filters[2]) -> filters[1]
        self.conv1_3 = ConvBlock(filters[1]*3 + filters[2], filters[1], dropout_p=self.dropout_p)

        # Skip path L4 (X_i_4)
        # X0_0 (filters[0]) + X0_1 (filters[0]) + X0_2 (filters[0]) + X0_3 (filters[0]) + Up(X1_3) (filters[1]) -> filters[0]
        self.conv0_4 = ConvBlock(filters[0]*4 + filters[1], filters[0], dropout_p=self.dropout_p)

        # Output convolutions for deep supervision (if enabled)
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0)) # Bottleneck

        # Decoder - Level 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        # Decoder - Level 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        # Decoder - Level 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        
        # Decoder - Level 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return [out1, out2, out3, out4] # Or torch.stack([out1, out2, out3, out4], dim=0) for easier loss computation
        else:
            return self.final(x0_4)

# %% [markdown]
# ## 早停类

# %%
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=True): # Changed min_delta to float
        self.patience = patience; self.min_delta = min_delta; self.mode = mode
        self.verbose = verbose; self.counter = 0; self.best_value = None
        self.early_stop = False; self.best_epoch = -1

        if self.mode == 'min': self.delta_op, self.best_op = lambda c,b: b-c > self.min_delta, lambda c,b: c < b
        elif self.mode == 'max': self.delta_op, self.best_op = lambda c,b: c-b > self.min_delta, lambda c,b: c > b
        else: raise ValueError(f"Unsupported mode: {self.mode}")

    def __call__(self, current_value, epoch): # Added epoch
        if self.best_value is None:
            self.best_value = current_value; self.best_epoch = epoch; return False
        
        if self.best_op(current_value, self.best_value):
            if self.delta_op(current_value, self.best_value):
                self.best_value = current_value; self.counter = 0; self.best_epoch = epoch
                if self.verbose: print(f"EarlyStopping: New best value: {self.best_value:.6f} at epoch {epoch+1}")
            else:
                self.counter += 1
                if self.verbose: print(f"EarlyStopping: No sig. improvement. Counter: {self.counter}/{self.patience}. Best: {self.best_value:.6f} (epoch {self.best_epoch+1}), Current: {current_value:.6f} (epoch {epoch+1})")
        else:
            self.counter += 1
            if self.verbose: print(f"EarlyStopping: Metric not improved. Counter: {self.counter}/{self.patience}. Best: {self.best_value:.6f} (epoch {self.best_epoch+1}), Current: {current_value:.6f} (epoch {epoch+1})")

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose: print(f"EarlyStopping: Stopping early after {self.patience} epochs of no improvement from epoch {self.best_epoch+1}.")
        return self.early_stop

# %% [markdown]
# ## 指标计算

# %%
class MetricsCalculator:
    def __init__(self, threshold=0.5): self.threshold = threshold
    def _calculate_stats(self, y_pred_sigmoid, y_true):
        y_pred = (y_pred_sigmoid > self.threshold).float()
        tp = torch.sum((y_pred == 1) & (y_true == 1)).item()
        fp = torch.sum((y_pred == 1) & (y_true == 0)).item()
        fn = torch.sum((y_pred == 0) & (y_true == 1)).item()
        tn = torch.sum((y_pred == 0) & (y_true == 0)).item()
        return tp, fp, fn, tn
    def calculate_metrics(self, outputs, masks):
        y_pred_sigmoid = torch.sigmoid(outputs)
        tp, fp, fn, tn = self._calculate_stats(y_pred_sigmoid, masks)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        return {"dice": dice, "iou": iou, "sensitivity": sensitivity, "specificity": specificity}

# %% [markdown]
# ## 可视化工具

# %%
class Visualizer:
    def __init__(self, config: ModelConfig):
        self.config = config; self.figure_dir = config.FIGURE_DIR
        os.makedirs(self.figure_dir, exist_ok=True)

    def plot_training_curves(self, history, epoch):
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Training Progress - Epoch {epoch+1} - Model: {self.config.MODEL_NAME}', fontsize=16)
        metrics_to_plot = ["loss", "dice", "iou", "sensitivity", "specificity"]
        for i, metric in enumerate(metrics_to_plot):
            ax = axs[i//3, i%3]
            ax.plot(history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
            ax.plot(history[f'val_{metric}'], label=f'Val {metric.capitalize()}')
            ax.set_title(metric.capitalize()); ax.set_xlabel('Epoch'); ax.set_ylabel(metric.capitalize())
            ax.legend(); ax.grid(True)
        if len(metrics_to_plot) < axs.size: fig.delaxes(axs[1,2]) # Remove last if 5 plots
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.figure_dir, f"training_curves_epoch_{epoch+1}.png")
        plt.savefig(save_path); print(f"Training curves saved to {save_path}"); plt.close(fig)

    def visualize_predictions(self, model, dataloader, device, num_samples=5, epoch="final"):
        if not self.config.VISUALIZE_PREDICTIONS: return
        model.eval(); samples_shown = 0
        if num_samples == 1: fig, axes = plt.subplots(1, 3, figsize=(12,4)); axes_flat = [axes[0], axes[1], axes[2]]
        else: fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples)); axes_flat = axes.flat
        fig.suptitle(f"Sample Predictions ({self.config.MODEL_NAME} - Epoch {epoch})", fontsize=16)
        with torch.no_grad():
            for images, masks_true in dataloader:
                if samples_shown >= num_samples: break
                images, masks_true = images.to(device), masks_true.to(device)
                outputs = model(images)
                if isinstance(outputs, list): outputs = outputs[-1] # Use last output if deep supervision list
                masks_pred = (torch.sigmoid(outputs) > 0.5).float()
                for j in range(images.size(0)):
                    if samples_shown >= num_samples: break
                    img_np = images[j].cpu().permute(1,2,0).numpy(); mean=np.array([0.485,0.456,0.406]); std=np.array([0.229,0.224,0.225])
                    img_np = np.clip(std*img_np+mean,0,1)
                    ax_base = samples_shown*3
                    axes_flat[ax_base].imshow(img_np); axes_flat[ax_base].set_title("Image"); axes_flat[ax_base].axis('off')
                    axes_flat[ax_base+1].imshow(masks_true[j].cpu().squeeze().numpy(), cmap='gray'); axes_flat[ax_base+1].set_title("True Mask"); axes_flat[ax_base+1].axis('off')
                    axes_flat[ax_base+2].imshow(masks_pred[j].cpu().squeeze().numpy(), cmap='gray'); axes_flat[ax_base+2].set_title("Predicted Mask"); axes_flat[ax_base+2].axis('off')
                    samples_shown +=1
        plt.tight_layout(rect=[0,0,1,0.96]); save_path = os.path.join(self.figure_dir, f"sample_predictions_epoch_{epoch}.png")
        plt.savefig(save_path); print(f"Sample predictions saved to {save_path}"); plt.close(fig)

# %% [markdown]
# ## 模型训练器

# %%
class ModelTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.config.SAVE_DIR = os.path.join(config.SAVE_DIR, config.MODEL_NAME)
        self.config.FIGURE_DIR = os.path.join(config.FIGURE_DIR, config.MODEL_NAME)
        os.makedirs(self.config.SAVE_DIR, exist_ok=True); os.makedirs(self.config.FIGURE_DIR, exist_ok=True)
        self._set_seed(); self.device = torch.device(config.DEVICE)
        self.data_transforms = DataTransforms(config.IMAGE_SIZE)
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer(self.config)
        self.model = self._get_model().to(self.device)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_loss_function()
        self.scaler = GradScaler(enabled=config.USE_AMP)
        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, min_delta=config.EARLY_STOPPING_MIN_DELTA, 
                                            mode='max' if config.EARLY_STOPPING_METRIC in ["dice","iou","sensitivity"] else 'min', verbose=True)
        self.history = {m:[] for m in ["train_loss","val_loss","train_dice","val_dice","train_iou","val_iou",
                                       "train_sensitivity","val_sensitivity","train_specificity","val_specificity"]}
    def _set_seed(self):
        torch.manual_seed(self.config.RANDOM_SEED); np.random.seed(self.config.RANDOM_SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.config.RANDOM_SEED)
    def _get_model(self):
        if self.config.MODEL_NAME == "UNetPlusPlus":
            return UNetPlusPlus(in_channels=self.config.IN_CHANNELS, out_channels=self.config.OUT_CHANNELS,
                                features_start=self.config.FEATURES_START, deep_supervision=self.config.DEEP_SUPERVISION,
                                dropout_p=0.1) # Example dropout, can be configured
        raise ValueError(f"Unsupported model: {self.config.MODEL_NAME}")
    def _get_optimizer(self):
        if self.config.OPTIMIZER_NAME.lower() == "adam": return optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        if self.config.OPTIMIZER_NAME.lower() == "adamw": return optim.AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        raise ValueError(f"Unsupported optimizer: {self.config.OPTIMIZER_NAME}")
    def _get_loss_function(self):
        if self.config.LOSS_FUNCTION == "BCEWithLogitsLoss": return nn.BCEWithLogitsLoss()
        # Add other losses: DiceLoss, FocalLoss, etc.
        raise ValueError(f"Unsupported loss: {self.config.LOSS_FUNCTION}")

    def _calculate_loss_and_metrics(self, outputs, masks, is_train=True):
        if self.config.DEEP_SUPERVISION and isinstance(outputs, list): # Deep supervision
            total_loss = 0
            # Consider only the last output for metrics, or average them if appropriate
            # For simplicity with current metrics calculator, use last output for metrics calculation
            final_output_for_metrics = outputs[-1] 
            for i, output in enumerate(outputs):
                total_loss += self.criterion(output, masks) * (0.5 + (i * 0.5 / (len(outputs)-1)) if len(outputs)>1 else 1.0) # Weighted sum, simple avg: * (1/len(outputs))
            avg_loss = total_loss / len(outputs) if len(outputs) > 0 else total_loss # if only one output due to no DS
        else: # Single output
            final_output_for_metrics = outputs
            avg_loss = self.criterion(final_output_for_metrics, masks)
        
        # Detach for metrics if in training to prevent graph issues with detached parts
        metrics = self.metrics_calculator.calculate_metrics(final_output_for_metrics.detach() if is_train else final_output_for_metrics, masks)
        return avg_loss, metrics

    def _train_epoch(self, dataloader):
        self.model.train(); epoch_loss = 0; epoch_metrics = {k:0 for k in self.history if k.startswith('train_') and 'loss' not in k}
        metric_keys_stripped = [k.replace('train_','') for k in epoch_metrics.keys()]

        for images, masks in tqdm(dataloader, desc="Training"):
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            with autocast(enabled=self.config.USE_AMP):
                outputs = self.model(images)
                loss, batch_metrics = self._calculate_loss_and_metrics(outputs, masks, is_train=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer); self.scaler.update()
            epoch_loss += loss.item()
            for key in metric_keys_stripped: epoch_metrics[f'train_{key}'] += batch_metrics[key]
        
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_metrics = {key: val / num_batches for key, val in epoch_metrics.items()}
        return avg_loss, avg_metrics

    def _validate_epoch(self, dataloader):
        self.model.eval(); epoch_loss = 0; epoch_metrics = {k:0 for k in self.history if k.startswith('val_') and 'loss' not in k}
        metric_keys_stripped = [k.replace('val_','') for k in epoch_metrics.keys()]

        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="Validating"):
                images, masks = images.to(self.device), masks.to(self.device)
                with autocast(enabled=self.config.USE_AMP):
                    outputs = self.model(images)
                    loss, batch_metrics = self._calculate_loss_and_metrics(outputs, masks, is_train=False)
                epoch_loss += loss.item()
                for key in metric_keys_stripped: epoch_metrics[f'val_{key}'] += batch_metrics[key]
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_metrics = {key: val/num_batches for key, val in epoch_metrics.items()}
        return avg_loss, avg_metrics

    def _save_checkpoint(self, epoch, metric_value, is_best=False):
        chkpt = {'epoch':epoch, 'model_state_dict':self.model.state_dict(), 'optimizer_state_dict':self.optimizer.state_dict(),
                 f'val_{self.config.EARLY_STOPPING_METRIC}': metric_value}
        if self.scaler: chkpt['scaler_state_dict'] = self.scaler.state_dict()
        suffix = "best" if is_best else f"epoch_{epoch+1}"
        save_path = os.path.join(self.config.SAVE_DIR, f"{suffix}.pth")
        torch.save(chkpt, save_path)
        print(f"Checkpoint saved: {save_path} (Val {self.config.EARLY_STOPPING_METRIC}: {metric_value:.4f})")
        if is_best: print(f"*** New best model (Val {self.config.EARLY_STOPPING_METRIC}: {metric_value:.4f} at epoch {epoch+1}) ***")

    def train(self):
        print(f"Starting training for {self.config.MODEL_NAME} on {self.device}")
        print(f"Save Dir: {self.config.SAVE_DIR}, Figure Dir: {self.config.FIGURE_DIR}")
        print(f"Config: {self.config}")

        full_dataset = BladderDataset(self.config.TRAIN_IMG_DIR, self.config.TRAIN_MASK_DIR, self.data_transforms.get_train_transforms())
        train_size = int(self.config.TRAIN_VAL_SPLIT * len(full_dataset))
        val_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(self.config.RANDOM_SEED)
        train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size], generator=generator)
        
        train_d_instance = BladderDataset(self.config.TRAIN_IMG_DIR, self.config.TRAIN_MASK_DIR, self.data_transforms.get_train_transforms())
        train_dataset = torch.utils.data.Subset(train_d_instance, train_indices.indices)
        val_d_instance = BladderDataset(self.config.TRAIN_IMG_DIR, self.config.TRAIN_MASK_DIR, self.data_transforms.get_val_transforms())
        val_dataset = torch.utils.data.Subset(val_d_instance, val_indices.indices)

        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=self.config.NUM_WORKERS, pin_memory=self.device.type=='cuda')
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=self.config.NUM_WORKERS, pin_memory=self.device.type=='cuda')
        print(f"Data: Train {len(train_dataset)} (batches {len(train_loader)}), Val {len(val_dataset)} (batches {len(val_loader)})")

        best_metric_val = -float('inf') if self.early_stopping.mode == 'max' else float('inf')
        final_epoch = 0

        for epoch in range(self.config.NUM_EPOCHS):
            final_epoch = epoch; print(f"\n--- Epoch {epoch+1}/{self.config.NUM_EPOCHS} ---")
            train_loss, train_metrics_dict = self._train_epoch(train_loader) # train_metrics is now a dict like {'train_dice': val, ...}
            val_loss, val_metrics_dict = self._validate_epoch(val_loader)   # val_metrics is now a dict like {'val_dice': val, ...}

            self.history['train_loss'].append(train_loss); self.history['val_loss'].append(val_loss)
            for k,v in train_metrics_dict.items(): self.history[k].append(v) # k is already 'train_dice' etc.
            for k,v in val_metrics_dict.items(): self.history[k].append(v) # k is already 'val_dice' etc.

            print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            # Access metrics directly from the returned dicts, keys are already prefixed
            print(f"  Train Metrics: Dice: {train_metrics_dict.get('train_dice',0):.4f}, IoU: {train_metrics_dict.get('train_iou',0):.4f}, Sens: {train_metrics_dict.get('train_sensitivity',0):.4f}, Spec: {train_metrics_dict.get('train_specificity',0):.4f}")
            print(f"  Val Metrics:   Dice: {val_metrics_dict.get('val_dice',0):.4f}, IoU: {val_metrics_dict.get('val_iou',0):.4f}, Sens: {val_metrics_dict.get('val_sensitivity',0):.4f}, Spec: {val_metrics_dict.get('val_specificity',0):.4f}")
            
            self.visualizer.plot_training_curves(self.history, epoch)
            current_val_metric = val_metrics_dict['val_' + self.config.EARLY_STOPPING_METRIC] # Get the specific metric for early stopping

            if (self.early_stopping.mode == 'max' and current_val_metric > best_metric_val and (current_val_metric - best_metric_val) > self.config.EARLY_STOPPING_MIN_DELTA) or \
               (self.early_stopping.mode == 'min' and current_val_metric < best_metric_val and (best_metric_val - current_val_metric) > self.config.EARLY_STOPPING_MIN_DELTA) or \
               (self.early_stopping.best_value is None): # Initial best assignment
                best_metric_val = current_val_metric
                self._save_checkpoint(epoch, current_val_metric, is_best=True)
            
            if (epoch+1)%10 ==0: self._save_checkpoint(epoch, current_val_metric, is_best=False)
            if self.early_stopping(current_val_metric, epoch): print("Early stopping triggered."); break
        
        print(f"\nTraining finished. Best validation {self.config.EARLY_STOPPING_METRIC}: {self.early_stopping.best_value:.4f} at epoch {self.early_stopping.best_epoch+1}")
        best_model_path = os.path.join(self.config.SAVE_DIR, "best.pth")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} for final visualization.")
            chkpt = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(chkpt['model_state_dict'])
            self.visualizer.visualize_predictions(self.model,val_loader,self.device,num_samples=self.config.NUM_VISUALIZATION_SAMPLES,epoch="best_model")
        else:
            print(f"No best model at {best_model_path}. Visualizing model from last epoch {final_epoch+1}.")
            self.visualizer.visualize_predictions(self.model,val_loader,self.device,num_samples=self.config.NUM_VISUALIZATION_SAMPLES,epoch=f"epoch_{final_epoch+1}_final")

# %% [markdown]
# ## 主执行函数

# %%
def main():
    config = ModelConfig(
        MODEL_NAME="UNetPlusPlus",
        FEATURES_START=32, 
        BATCH_SIZE=4, # UNet++ can be memory intensive
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=100,
        EARLY_STOPPING_PATIENCE=15, # Increased patience
        DEEP_SUPERVISION=False, # Keep False for now for simpler trainer logic matching template
        DATA_DIR="dataset", 
        TRAIN_IMG_DIR="dataset/images",
        TRAIN_MASK_DIR="dataset/masks",
        SAVE_DIR="checkpoints", 
        FIGURE_DIR="figures"  
    )
    trainer = ModelTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 