
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
from dataclasses import dataclass, field
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn.functional as F

@dataclass
class ModelConfig:

    MODEL_NAME: str = "AttentionUNet" 
    RANDOM_SEED: int = 42
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
 
    DATA_DIR: str = "dataset" 
    TRAIN_IMG_DIR: str = "dataset/images"
    TRAIN_MASK_DIR: str = "dataset/masks"
    IMAGE_SIZE: int = 256
    BATCH_SIZE: int = 8 
    TRAIN_VAL_SPLIT: float = 0.8
    NUM_WORKERS: int = 1 
    

    IN_CHANNELS: int = 3
    OUT_CHANNELS: int = 1
    FEATURES_START: int = 64 

    # 训练配置
    NUM_EPOCHS: int = 100
    LEARNING_RATE: float = 1e-4 
    OPTIMIZER_NAME: str = "Adam" # "Adam", "AdamW"
    LOSS_FUNCTION: str = "BCEWithLogitsLoss" # "BCEWithLogitsLoss", "DiceLoss", "FocalLoss", "DiceBCELoss"
    USE_AMP: bool = True # 是否使用混合精度训练
    
    # 早停配置
    EARLY_STOPPING_PATIENCE: int = 10
    EARLY_STOPPING_MIN_DELTA: float = 0.001 # Dice Score 的最小提升
    EARLY_STOPPING_METRIC: str = "dice" # 基于哪个指标进行早停
    
    # 保存配置
    SAVE_DIR: str = "checkpoints" # Base directory, will be adapted in main
    
    # 可视化配置
    VISUALIZE_PREDICTIONS: bool = True
    NUM_VISUALIZATION_SAMPLES: int = 5

    FIGURE_DIR: str = "figures" # Base directory, will be adapted in main

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
# ## Attention Gate 和 Attention U-Net 模型定义

# %%
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # g: gating signal from the coarser scale
        # x: signal from the skip connection (encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features_start=64):
        super(AttentionUNet, self).__init__()
        
        fs = features_start
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, fs)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = ConvBlock(fs, fs*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = ConvBlock(fs*2, fs*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = ConvBlock(fs*4, fs*8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(fs*8, fs*16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(fs*16, fs*8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=fs*8, F_l=fs*8, F_int=fs*4)
        self.dec4 = ConvBlock(fs*16, fs*8) # fs*8 (from upconv) + fs*8 (from att_gate*enc4)
        
        self.upconv3 = nn.ConvTranspose2d(fs*8, fs*4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=fs*4, F_l=fs*4, F_int=fs*2)
        self.dec3 = ConvBlock(fs*8, fs*4) # fs*4 + fs*4
        
        self.upconv2 = nn.ConvTranspose2d(fs*4, fs*2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=fs*2, F_l=fs*2, F_int=fs)
        self.dec2 = ConvBlock(fs*4, fs*2) # fs*2 + fs*2
        
        self.upconv1 = nn.ConvTranspose2d(fs*2, fs, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=fs, F_l=fs, F_int=fs//2)
        self.dec1 = ConvBlock(fs*2, fs) # fs + fs
        
        # Output layer
        self.conv_out = nn.Conv2d(fs, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)    # fs
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)   # fs*2
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)   # fs*4
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)   # fs*8
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4) # fs*16
        
        # Decoder
        d4 = self.upconv4(b)             # fs*8
        att_e4 = self.att4(g=d4, x=e4)   # fs*8 (attended e4)
        d4 = torch.cat((att_e4, d4), dim=1) # fs*16
        d4 = self.dec4(d4)               # fs*8
        
        d3 = self.upconv3(d4)            # fs*4
        att_e3 = self.att3(g=d3, x=e3)   # fs*4
        d3 = torch.cat((att_e3, d3), dim=1) # fs*8
        d3 = self.dec3(d3)               # fs*4
        
        d2 = self.upconv2(d3)            # fs*2
        att_e2 = self.att2(g=d2, x=e2)   # fs*2
        d2 = torch.cat((att_e2, d2), dim=1) # fs*4
        d2 = self.dec2(d2)               # fs*2
        
        d1 = self.upconv1(d2)            # fs
        att_e1 = self.att1(g=d1, x=e1)   # fs
        d1 = torch.cat((att_e1, d1), dim=1) # fs*2
        d1 = self.dec1(d1)               # fs
        
        out = self.conv_out(d1)
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
        # Specific figure_dir with model name is set in ModelTrainer now
        self.figure_dir = config.FIGURE_DIR 
        os.makedirs(self.figure_dir, exist_ok=True)


    def plot_training_curves(self, history, epoch):
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Training Progress - Epoch {epoch+1} - Model: {self.config.MODEL_NAME}', fontsize=16)

        axs[0, 0].plot(history['train_loss'], label='Train Loss')
        axs[0, 0].plot(history['val_loss'], label='Validation Loss')
        axs[0, 0].set_title('Loss'); axs[0, 0].set_xlabel('Epoch'); axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend(); axs[0, 0].grid(True)

        axs[0, 1].plot(history['train_dice'], label='Train Dice')
        axs[0, 1].plot(history['val_dice'], label='Validation Dice')
        axs[0, 1].set_title('Dice Coefficient'); axs[0, 1].set_xlabel('Epoch'); axs[0, 1].set_ylabel('Dice')
        axs[0, 1].legend(); axs[0, 1].grid(True)

        axs[0, 2].plot(history['train_iou'], label='Train IoU')
        axs[0, 2].plot(history['val_iou'], label='Validation IoU')
        axs[0, 2].set_title('IoU (Jaccard)'); axs[0, 2].set_xlabel('Epoch'); axs[0, 2].set_ylabel('IoU')
        axs[0, 2].legend(); axs[0, 2].grid(True)

        axs[1, 0].plot(history['train_sensitivity'], label='Train Sensitivity')
        axs[1, 0].plot(history['val_sensitivity'], label='Validation Sensitivity')
        axs[1, 0].set_title('Sensitivity (Recall)'); axs[1, 0].set_xlabel('Epoch'); axs[1, 0].set_ylabel('Sensitivity')
        axs[1, 0].legend(); axs[1, 0].grid(True)
        
        axs[1, 1].plot(history['train_specificity'], label='Train Specificity')
        axs[1, 1].plot(history['val_specificity'], label='Validation Specificity')
        axs[1, 1].set_title('Specificity'); axs[1, 1].set_xlabel('Epoch'); axs[1, 1].set_ylabel('Specificity')
        axs[1, 1].legend(); axs[1, 1].grid(True)

        fig.delaxes(axs[1, 2])
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # Save path now includes model name from config, handled by ModelTrainer or main call
        save_path = os.path.join(self.figure_dir, f"training_curves_epoch_{epoch+1}.png")
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
        plt.close(fig)

    def visualize_predictions(self, model, dataloader, device, num_samples=5, epoch="final"):
        if not self.config.VISUALIZE_PREDICTIONS:
            return
            
        model.eval()
        samples_shown = 0
        # Adjust subplot creation if num_samples is 1 to avoid error with axes.flat
        if num_samples == 1:
            fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4))
            axes_flat = [axes[0], axes[1], axes[2]] # make it iterable
        else:
            fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
            axes_flat = axes.flat


        fig.suptitle(f"Sample Predictions ({self.config.MODEL_NAME} - Epoch {epoch})", fontsize=16)
        
        with torch.no_grad():
            for i, (images, masks_true) in enumerate(dataloader):
                if samples_shown >= num_samples:
                    break
                
                images = images.to(device)
                masks_true = masks_true.to(device)
                outputs_logits = model(images)
                outputs_sigmoid = torch.sigmoid(outputs_logits)
                masks_pred = (outputs_sigmoid > 0.5).float()

                for j in range(images.size(0)):
                    if samples_shown >= num_samples:
                        break

                    img_np = images[j].cpu().permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = std * img_np + mean
                    img_np = np.clip(img_np, 0, 1)

                    mask_true_np = masks_true[j].cpu().squeeze().numpy()
                    mask_pred_np = masks_pred[j].cpu().squeeze().numpy()

                    ax_idx_base = samples_shown * 3

                    axes_flat[ax_idx_base].imshow(img_np)
                    axes_flat[ax_idx_base].set_title("Image"); axes_flat[ax_idx_base].axis('off')

                    axes_flat[ax_idx_base + 1].imshow(mask_true_np, cmap='gray')
                    axes_flat[ax_idx_base + 1].set_title("True Mask"); axes_flat[ax_idx_base + 1].axis('off')

                    axes_flat[ax_idx_base + 2].imshow(mask_pred_np, cmap='gray')
                    axes_flat[ax_idx_base + 2].set_title("Predicted Mask"); axes_flat[ax_idx_base + 2].axis('off')
                    
                    samples_shown += 1
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.figure_dir, f"sample_predictions_epoch_{epoch}.png")
        plt.savefig(save_path)
        print(f"Sample predictions saved to {save_path}")
        plt.close(fig)

# %% [markdown]
# ## 模型训练器

# %%
class ModelTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        # Update save and figure directories to include model name
        self.config.SAVE_DIR = os.path.join(config.SAVE_DIR, config.MODEL_NAME)
        self.config.FIGURE_DIR = os.path.join(config.FIGURE_DIR, config.MODEL_NAME)
        os.makedirs(self.config.SAVE_DIR, exist_ok=True)
        os.makedirs(self.config.FIGURE_DIR, exist_ok=True)

        self._set_seed()
        
        self.device = torch.device(config.DEVICE)
        self.data_transforms = DataTransforms(config.IMAGE_SIZE)
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer(self.config) # Pass updated config

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
        if self.config.MODEL_NAME == "AttentionUNet":
            model = AttentionUNet(
                in_channels=self.config.IN_CHANNELS, 
                out_channels=self.config.OUT_CHANNELS,
                features_start=self.config.FEATURES_START
            )
        # Add other models here if necessary for this script
        # elif self.config.MODEL_NAME == "SomeOtherModel":
        #     pass
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
        # elif self.config.LOSS_FUNCTION == "DiceLoss":
        #     from .losses import DiceLoss # Assuming DiceLoss is defined elsewhere
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
        # Save path uses the model-specific SAVE_DIR from config
        save_path = os.path.join(self.config.SAVE_DIR, f"{filename_suffix}.pth") 
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path} (Validation {self.config.EARLY_STOPPING_METRIC}: {metric_value:.4f})")
        if is_best:
             print(f"*** New best model saved based on validation {self.config.EARLY_STOPPING_METRIC}! ***")


    def train(self):
        print(f"Starting training for {self.config.MODEL_NAME} on {self.device}")
        print(f"Using Save Directory: {self.config.SAVE_DIR}")
        print(f"Using Figure Directory: {self.config.FIGURE_DIR}")
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
        train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size], generator=generator)


        train_dataset_instance = BladderDataset(image_dir=self.config.TRAIN_IMG_DIR, mask_dir=self.config.TRAIN_MASK_DIR, transform=self.data_transforms.get_train_transforms())
        train_dataset = torch.utils.data.Subset(train_dataset_instance, train_indices.indices)

        val_dataset_instance = BladderDataset(image_dir=self.config.TRAIN_IMG_DIR, mask_dir=self.config.TRAIN_MASK_DIR, transform=val_transform)
        val_dataset = torch.utils.data.Subset(val_dataset_instance, val_indices.indices)

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
        final_epoch = 0

        for epoch in range(self.config.NUM_EPOCHS):
            final_epoch = epoch
            print(f"\n--- Epoch {epoch+1}/{self.config.NUM_EPOCHS} ---")
            
            train_loss, train_metrics = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate_epoch(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for metric_name in train_metrics:
                self.history[f'train_{metric_name}'].append(train_metrics[metric_name])
                self.history[f'val_{metric_name}'].append(val_metrics[metric_name])

            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Train Metrics: Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}, Sens: {train_metrics['sensitivity']:.4f}, Spec: {train_metrics['specificity']:.4f}")
            print(f"  Val Metrics:   Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, Sens: {val_metrics['sensitivity']:.4f}, Spec: {val_metrics['specificity']:.4f}")

            self.visualizer.plot_training_curves(self.history, epoch)

            current_metric_val = val_metrics[self.config.EARLY_STOPPING_METRIC]
            
            if (self.early_stopping.mode == 'max' and current_metric_val > best_metric_val) or \
               (self.early_stopping.mode == 'min' and current_metric_val < best_metric_val):
                # Check if it's a significant improvement for 'max' mode
                if self.early_stopping.mode == 'max' and (current_metric_val - best_metric_val) > self.config.EARLY_STOPPING_MIN_DELTA :
                    best_metric_val = current_metric_val
                    self._save_checkpoint(epoch, current_metric_val, is_best=True)
                # Check if it's a significant improvement for 'min' mode
                elif self.early_stopping.mode == 'min' and (best_metric_val - current_metric_val) > self.config.EARLY_STOPPING_MIN_DELTA :
                    best_metric_val = current_metric_val
                    self._save_checkpoint(epoch, current_metric_val, is_best=True)
                elif best_metric_val == -float('inf') or best_metric_val == float('inf'): # First assignment
                     best_metric_val = current_metric_val
                     self._save_checkpoint(epoch, current_metric_val, is_best=True)


            # Save regular checkpoint less frequently
            if (epoch + 1) % 10 == 0 : # Save every 10 epochs for example
                    self._save_checkpoint(epoch, current_metric_val, is_best=False)


            if self.early_stopping(current_metric_val):
                print("Early stopping triggered.")
                break
        
        print("\nTraining finished.")
        print(f"Best validation {self.config.EARLY_STOPPING_METRIC}: {self.early_stopping.best_value:.4f} at epoch {self.early_stopping.best_epoch if hasattr(self.early_stopping, 'best_epoch') else 'N/A'}")
        
        best_model_path = os.path.join(self.config.SAVE_DIR, "best.pth")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} for final visualization.")
            checkpoint_data = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            self.visualizer.visualize_predictions(self.model, val_loader, self.device, 
                                                  num_samples=self.config.NUM_VISUALIZATION_SAMPLES, epoch="best_model")
        else:
            print(f"No best model checkpoint found at {best_model_path}. Using model from last epoch ({final_epoch+1}) for visualization.")
            self.visualizer.visualize_predictions(self.model, val_loader, self.device, 
                                                 num_samples=self.config.NUM_VISUALIZATION_SAMPLES, epoch=f"epoch_{final_epoch+1}_final")


# %% [markdown]
# ## 主执行函数

# %%
def main():
    config = ModelConfig(
        MODEL_NAME="AttentionUNet",
        FEATURES_START=32, # Can be 32 or 64, adjust based on memory
        BATCH_SIZE=8, 
        LEARNING_RATE=1e-4, # Typical for Attention U-Net
        NUM_EPOCHS=100, 
        EARLY_STOPPING_PATIENCE=15,
        DATA_DIR="dataset", 
        TRAIN_IMG_DIR="dataset/images",
        TRAIN_MASK_DIR="dataset/masks",
        # Base directories, ModelTrainer will append MODEL_NAME
        SAVE_DIR="checkpoints", 
        FIGURE_DIR="figures"  
    )

    trainer = ModelTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 