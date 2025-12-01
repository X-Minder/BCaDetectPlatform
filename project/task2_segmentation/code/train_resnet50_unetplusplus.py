# %% [markdown]
# # UNet++ (Nested U-Net) with ResNet50 Backbone - Bladder Segmentation Training
# This script implements UNet++ with a pre-trained ResNet50 backbone for medical image segmentation.
# Key Features:
# 1. Data Loading and Preprocessing
# 2. Model Training and Validation (with ResNet50 encoder)
# 3. Metrics Calculation and Visualization
# 4. Model Saving and Early Stopping
# 5. Deep Supervision (optional)

# %% [markdown]
# ## Import Libraries

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
from dataclasses import dataclass, field
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import torch.nn.functional as F

# %% [markdown]
# ## Configuration Class

# %%
@dataclass
class ModelConfig:
    # Basic Config
    MODEL_NAME: str = "ResNet50UNetPlusPlus"
    RANDOM_SEED: int = 42
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dataset Config
    DATA_DIR: str = "dataset" 
    TRAIN_IMG_DIR: str = "dataset/images"
    TRAIN_MASK_DIR: str = "dataset/masks"
    IMAGE_SIZE: int = 256
    BATCH_SIZE: int = 2 # Adjust based on GPU memory with ResNet50 backbone (potentially very low)
    TRAIN_VAL_SPLIT: float = 0.8
    NUM_WORKERS: int = 1 
    
    # Model Config (ResNet50UNetPlusPlus specific)
    IN_CHANNELS: int = 3
    OUT_CHANNELS: int = 1
    ENCODER_PRETRAINED: bool = True
    DECODER_FEATURES_START: int = 64 
    DEEP_SUPERVISION: bool = True 
    DROPOUT_P: float = 0.2 

    # Training Config
    NUM_EPOCHS: int = 100
    LEARNING_RATE: float = 1e-4 
    OPTIMIZER_NAME: str = "Adam" 
    LOSS_FUNCTION: str = "BCEWithLogitsLoss" 
    USE_AMP: bool = True
    
    # Early Stopping Config
    EARLY_STOPPING_PATIENCE: int = 15
    EARLY_STOPPING_MIN_DELTA: float = 0.001
    EARLY_STOPPING_METRIC: str = "dice"
    
    # Save Config
    SAVE_DIR: str = "checkpoints_resnet50_unetplusplus"
    
    # Visualization Config
    VISUALIZE_PREDICTIONS: bool = True
    NUM_VISUALIZATION_SAMPLES: int = 5
    FIGURE_DIR: str = "figures_resnet50_unetplusplus"

    def __post_init__(self):
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(self.FIGURE_DIR, exist_ok=True)
        if self.DEVICE == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA not available, switching to CPU.")
            self.DEVICE = 'cpu'

# %% [markdown]
# ## Data Augmentation

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
# ## Dataset Class

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
# ## ResNet50-UNet++ Model Definition

# %%
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
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

class ResNet50UNetPlusPlus(nn.Module):
    def __init__(self, out_channels=1, decoder_features_start=64, deep_supervision=False, 
                 pretrained_encoder=True, dropout_p=0.1):
        super(ResNet50UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        
        # --- Encoder (ResNet50) ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained_encoder else None)
        
        self.encoder_x0_0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # Output: 64 ch, H/2
        self.pool = resnet.maxpool # H/4
        self.encoder_x1_0 = resnet.layer1 # Output: 256 ch, H/4 (bottleneck x4 of 64)
        self.encoder_x2_0 = resnet.layer2 # Output: 512 ch, H/8 (bottleneck x4 of 128)
        self.encoder_x3_0 = resnet.layer3 # Output: 1024 ch, H/16 (bottleneck x4 of 256)
        self.encoder_x4_0 = resnet.layer4 # Output: 2048 ch, H/32 (bottleneck x4 of 512) - Bottleneck

        # Channels from ResNet50 stages:
        enc0_0_ch = 64
        enc1_0_ch = 256
        enc2_0_ch = 512
        enc3_0_ch = 1024
        enc4_0_ch = 2048
        
        self.filters = [
            decoder_features_start,      
            decoder_features_start * 2,  
            decoder_features_start * 4,  
            decoder_features_start * 8   
        ]

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Skip path L1 (Nodes X_i_1)
        self.conv0_1 = ConvBlock(enc0_0_ch + enc1_0_ch, self.filters[0], dropout_p=dropout_p)
        self.conv1_1 = ConvBlock(enc1_0_ch + enc2_0_ch, self.filters[1], dropout_p=dropout_p)
        self.conv2_1 = ConvBlock(enc2_0_ch + enc3_0_ch, self.filters[2], dropout_p=dropout_p)
        self.conv3_1 = ConvBlock(enc3_0_ch + enc4_0_ch, self.filters[3], dropout_p=dropout_p)

        # Skip path L2 (Nodes X_i_2)
        self.conv0_2 = ConvBlock(enc0_0_ch + self.filters[0] + self.filters[1], self.filters[0], dropout_p=dropout_p)
        self.conv1_2 = ConvBlock(enc1_0_ch + self.filters[1] + self.filters[2], self.filters[1], dropout_p=dropout_p)
        self.conv2_2 = ConvBlock(enc2_0_ch + self.filters[2] + self.filters[3], self.filters[2], dropout_p=dropout_p)
        
        # Skip path L3 (Nodes X_i_3)
        self.conv0_3 = ConvBlock(enc0_0_ch + self.filters[0]*2 + self.filters[1], self.filters[0], dropout_p=dropout_p)
        self.conv1_3 = ConvBlock(enc1_0_ch + self.filters[1]*2 + self.filters[2], self.filters[1], dropout_p=dropout_p)

        # Skip path L4 (Nodes X_i_4)
        self.conv0_4 = ConvBlock(enc0_0_ch + self.filters[0]*3 + self.filters[1], self.filters[0], dropout_p=dropout_p)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(self.filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(self.filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(self.filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(self.filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(self.filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        x_enc0_0_prepool = self.encoder_x0_0(x) 
        x_enc_pooled = self.pool(x_enc0_0_prepool) 
        
        x_enc1_0 = self.encoder_x1_0(x_enc_pooled) 
        x_enc2_0 = self.encoder_x2_0(x_enc1_0)   
        x_enc3_0 = self.encoder_x3_0(x_enc2_0)   
        x_enc4_0 = self.encoder_x4_0(x_enc3_0)   

        x0_1 = self.conv0_1(torch.cat([x_enc0_0_prepool, self.up(x_enc1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x_enc1_0, self.up(x_enc2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x_enc2_0, self.up(x_enc3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x_enc3_0, self.up(x_enc4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x_enc0_0_prepool, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x_enc1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x_enc2_0, x2_1, self.up(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x_enc0_0_prepool, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x_enc1_0, x1_1, x1_2, self.up(x2_2)], 1))
        
        x0_4 = self.conv0_4(torch.cat([x_enc0_0_prepool, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            
            out1 = F.interpolate(out1, size=x.size()[2:], mode='bilinear', align_corners=True)
            out2 = F.interpolate(out2, size=x.size()[2:], mode='bilinear', align_corners=True)
            out3 = F.interpolate(out3, size=x.size()[2:], mode='bilinear', align_corners=True)
            out4 = F.interpolate(out4, size=x.size()[2:], mode='bilinear', align_corners=True)
            
            return [out4, out3, out2, out1] 
        else:
            final_output = self.final(x0_4)
            final_output = F.interpolate(final_output, size=x.size()[2:], mode='bilinear', align_corners=True)
            return final_output


# %% [markdown]
# ## Early Stopping Class

# %%
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=True):
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
            if self.verbose: print(f"EarlyStopping: Initialized best value to {self.best_value:.6f}")
            return False 

        improved = False
        significant_improvement = False

        if self.best_op(current_value, self.best_value):
            improved = True
            if self.delta_op(current_value, self.best_value):
                significant_improvement = True

        if significant_improvement:
            if self.verbose: print(f"EarlyStopping: Metric significantly improved from {self.best_value:.6f} to {current_value:.6f}. Resetting counter.")
            self.best_value = current_value
            self.counter = 0
        elif improved: 
            self.counter +=1
            if self.verbose: print(f"EarlyStopping: Metric improved to {current_value:.6f}, but not significantly. Counter: {self.counter}/{self.patience}.")
        else: 
            self.counter += 1
            if self.verbose: print(f"EarlyStopping: Metric did not improve from {self.best_value:.6f}. Counter: {self.counter}/{self.patience}.")
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose: print(f"EarlyStopping: Stopping early after {self.patience} epochs of no significant improvement.")
        
        return self.early_stop


# %% [markdown]
# ## Metrics Calculator

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

    def calculate_metrics(self, outputs_logits, masks):
        y_pred_sigmoid = torch.sigmoid(outputs_logits)
        y_true = masks
        tp, fp, fn, tn = self._calculate_stats(y_pred_sigmoid, y_true)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        return {"dice": dice, "iou": iou, "sensitivity": sensitivity, "specificity": specificity}

# %% [markdown]
# ## Visualizer

# %%
class Visualizer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.figure_dir = config.FIGURE_DIR
        os.makedirs(self.figure_dir, exist_ok=True)

    def plot_training_curves(self, history, epoch):
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Training Progress - Epoch {epoch+1} - Model: {self.config.MODEL_NAME}', fontsize=16)
        metrics_to_plot = ["loss", "dice", "iou", "sensitivity", "specificity"]
        plot_coords = [(0,0), (0,1), (0,2), (1,0), (1,1)]
        titles = ["Loss", "Dice Coefficient", "IoU (Jaccard)", "Sensitivity (Recall)", "Specificity"]

        for i, metric in enumerate(metrics_to_plot):
            ax = axs[plot_coords[i]]
            ax.plot(history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
            ax.plot(history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
            ax.set_title(titles[i])
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True)
        
        if len(metrics_to_plot) < axs.size: fig.delaxes(axs[1,2]) 
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        save_path = os.path.join(self.figure_dir, f"{self.config.MODEL_NAME}_training_curves_epoch_{epoch+1}.png")
        plt.savefig(save_path)
        plt.close(fig)

    def visualize_predictions(self, model, dataloader, device, num_samples=5, epoch="final"):
        if not self.config.VISUALIZE_PREDICTIONS: return
        model.eval()
        samples_shown = 0
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1: axes = [axes] 
        fig.suptitle(f"Sample Predictions ({self.config.MODEL_NAME} - Epoch {epoch})", fontsize=16)
        
        with torch.no_grad():
            for images, masks_true in dataloader:
                if samples_shown >= num_samples: break
                images, masks_true = images.to(device), masks_true.to(device)
                
                model_outputs = model(images)
                outputs_logits = model_outputs[0] if isinstance(model_outputs, list) else model_outputs 
                
                outputs_sigmoid = torch.sigmoid(outputs_logits)
                masks_pred = (outputs_sigmoid > 0.5).float()

                for j in range(images.size(0)):
                    if samples_shown >= num_samples: break
                    img_np = images[j].cpu().permute(1, 2, 0).numpy()
                    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
                    img_np = std * img_np + mean
                    img_np = np.clip(img_np, 0, 1)

                    mask_true_np = masks_true[j].cpu().squeeze().numpy()
                    mask_pred_np = masks_pred[j].cpu().squeeze().numpy()
                    
                    ax_row = axes[samples_shown]
                    ax_row[0].imshow(img_np); ax_row[0].set_title("Image"); ax_row[0].axis('off')
                    ax_row[1].imshow(mask_true_np, cmap='gray'); ax_row[1].set_title("True Mask"); ax_row[1].axis('off')
                    ax_row[2].imshow(mask_pred_np, cmap='gray'); ax_row[2].set_title("Predicted Mask"); ax_row[2].axis('off')
                    samples_shown += 1
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.figure_dir, f"{self.config.MODEL_NAME}_sample_predictions_epoch_{epoch}.png")
        plt.savefig(save_path)
        print(f"Sample predictions saved to {save_path}")
        plt.close(fig)

# %% [markdown]
# ## Model Trainer

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
            mode='max' if config.EARLY_STOPPING_METRIC in ["dice", "iou", "sensitivity"] else 'min'
        )
        self.history = {m: [] for m in ["train_loss", "val_loss", "train_dice", "val_dice", "train_iou", "val_iou", 
                                        "train_sensitivity", "val_sensitivity", "train_specificity", "val_specificity"]}

    def _set_seed(self):
        torch.manual_seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.config.RANDOM_SEED)

    def _get_model(self):
        if self.config.MODEL_NAME == "ResNet50UNetPlusPlus":
            model = ResNet50UNetPlusPlus(
                out_channels=self.config.OUT_CHANNELS,
                decoder_features_start=self.config.DECODER_FEATURES_START,
                deep_supervision=self.config.DEEP_SUPERVISION,
                pretrained_encoder=self.config.ENCODER_PRETRAINED,
                dropout_p=self.config.DROPOUT_P
            )
        # Add elif for ResNet34UNetPlusPlus if this trainer is to be multi-model
        # else: raise ValueError(f"Unsupported model: {self.config.MODEL_NAME}")
        # For now, keeping it specific to ResNet50 version in this script.
        elif self.config.MODEL_NAME == "ResNet34UNetPlusPlus": # For safety, but should not be hit if MODEL_NAME is correct
             raise ValueError("This script is for ResNet50UNetPlusPlus. Use train_resnet34_unetplusplus.py for ResNet34.")
        else: raise ValueError(f"Unsupported model configured: {self.config.MODEL_NAME}")
        return model

    def _get_optimizer(self):
        if self.config.OPTIMIZER_NAME.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER_NAME.lower() == "adamw":
            return optim.AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        else: raise ValueError(f"Unsupported optimizer: {self.config.OPTIMIZER_NAME}")

    def _get_loss_function(self):
        if self.config.LOSS_FUNCTION == "BCEWithLogitsLoss": return nn.BCEWithLogitsLoss()
        else: raise ValueError(f"Unsupported loss function: {self.config.LOSS_FUNCTION}")

    def _run_epoch(self, dataloader, is_train=True):
        if is_train: self.model.train()
        else: self.model.eval()
        
        epoch_loss = 0.0
        epoch_metrics = {k: 0.0 for k in ["dice", "iou", "sensitivity", "specificity"]}
        
        progress_bar_desc = "Training" if is_train else "Validating"
        
        for images, masks in tqdm(dataloader, desc=progress_bar_desc):
            images, masks = images.to(self.device), masks.to(self.device)

            if is_train: self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_train): 
                with autocast(enabled=self.config.USE_AMP):
                    outputs = self.model(images) 
                    
                    current_loss = 0
                    if self.config.DEEP_SUPERVISION and isinstance(outputs, list):
                        for i, out_sup in enumerate(outputs):
                            current_loss += self.criterion(out_sup, masks)
                        current_loss /= len(outputs)
                        primary_output_for_metrics = outputs[0] 
                    else:
                        current_loss = self.criterion(outputs, masks)
                        primary_output_for_metrics = outputs
                
                if is_train:
                    self.scaler.scale(current_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            epoch_loss += current_loss.item()
            batch_metrics = self.metrics_calculator.calculate_metrics(primary_output_for_metrics.detach() if is_train else primary_output_for_metrics, masks)
            for key in epoch_metrics: epoch_metrics[key] += batch_metrics[key]
            
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
            'config': self.config 
        }
        if self.config.USE_AMP: checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        filename_suffix = "best" if is_best else f"epoch_{epoch+1}"
        save_path = os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_NAME}_{filename_suffix}.pth")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path} (Val {self.config.EARLY_STOPPING_METRIC}: {metric_value:.4f}){' **Best**' if is_best else ''}")

    def train(self):
        print(f"Starting training for {self.config.MODEL_NAME} on {self.config.DEVICE}")
        print(f"Config: {self.config}")

        full_dataset = BladderDataset(self.config.TRAIN_IMG_DIR, self.config.TRAIN_MASK_DIR) 
        train_size = int(self.config.TRAIN_VAL_SPLIT * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        generator = torch.Generator().manual_seed(self.config.RANDOM_SEED)
        train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size], generator=generator)

        train_dataset_transformed = BladderDataset(self.config.TRAIN_IMG_DIR, self.config.TRAIN_MASK_DIR, transform=self.data_transforms.get_train_transforms())
        val_dataset_transformed = BladderDataset(self.config.TRAIN_IMG_DIR, self.config.TRAIN_MASK_DIR, transform=self.data_transforms.get_val_transforms())
        
        train_dataset = torch.utils.data.Subset(train_dataset_transformed, train_indices.indices)
        val_dataset = torch.utils.data.Subset(val_dataset_transformed, val_indices.indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=self.config.NUM_WORKERS, pin_memory=self.config.DEVICE=='cuda', drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=self.config.NUM_WORKERS, pin_memory=self.config.DEVICE=='cuda')
        
        print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches. Validation: {len(val_dataset)} samples, {len(val_loader)} batches.")
        if not train_loader or not val_loader:
            print("ERROR: DataLoader is empty. Check dataset paths, split, or batch size.")
            return

        best_metric_val = -float('inf') if self.early_stopping.mode == 'max' else float('inf')
        final_epoch_completed = 0

        for epoch in range(self.config.NUM_EPOCHS):
            final_epoch_completed = epoch
            print(f"--- Epoch {epoch+1}/{self.config.NUM_EPOCHS} ---")
            train_loss, train_metrics = self._run_epoch(train_loader, is_train=True)
            val_loss, val_metrics = self._run_epoch(val_loader, is_train=False)

            self.history['train_loss'].append(train_loss); self.history['val_loss'].append(val_loss)
            for m in ["dice", "iou", "sensitivity", "specificity"]:
                self.history[f'train_{m}'].append(train_metrics[m]); self.history[f'val_{m}'].append(val_metrics[m])
            
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Train Metrics: Dice {train_metrics['dice']:.4f}, IoU {train_metrics['iou']:.4f}, Sensitivity {train_metrics['sensitivity']:.4f}, Specificity {train_metrics['specificity']:.4f}")
            print(f"  Val Metrics:   Dice {val_metrics['dice']:.4f}, IoU {val_metrics['iou']:.4f}, Sensitivity {val_metrics['sensitivity']:.4f}, Specificity {val_metrics['specificity']:.4f}")
            
            self.visualizer.plot_training_curves(self.history, epoch)
            current_val_metric = val_metrics[self.config.EARLY_STOPPING_METRIC]

            improved_significantly = False
            if self.early_stopping.best_value is not None:
                if self.early_stopping.mode == 'max':
                    if current_val_metric > self.early_stopping.best_value + self.early_stopping.min_delta:
                        improved_significantly = True
                else: 
                    if current_val_metric < self.early_stopping.best_value - self.early_stopping.min_delta:
                        improved_significantly = True
            
            if self.early_stopping.best_value is None or improved_significantly:
                 self._save_checkpoint(epoch, current_val_metric, is_best=True)
            elif (epoch + 1) % 10 == 0: 
                self._save_checkpoint(epoch, current_val_metric, is_best=False)

            if self.early_stopping(current_val_metric): 
                print("Early stopping triggered.")
                break
        
        print(f"\nTraining finished. Best validation {self.config.EARLY_STOPPING_METRIC}: {self.early_stopping.best_value:.4f}")
        
        best_model_path = os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_NAME}_best.pth")
        if os.path.exists(best_model_path):
            print(f"Loading best model: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False) 
            
            loaded_cfg = checkpoint.get('config')
            if loaded_cfg and (loaded_cfg.MODEL_NAME != self.config.MODEL_NAME or \
                               loaded_cfg.DECODER_FEATURES_START != self.config.DECODER_FEATURES_START or \
                               loaded_cfg.DEEP_SUPERVISION != self.config.DEEP_SUPERVISION or \
                               loaded_cfg.OUT_CHANNELS != self.config.OUT_CHANNELS):
                print("Warning: Loaded model config differs. Re-init model with loaded config.")
                # Ensure correct model class is instantiated here
                if loaded_cfg.MODEL_NAME == "ResNet50UNetPlusPlus":
                    self.model = ResNet50UNetPlusPlus(
                        out_channels=loaded_cfg.OUT_CHANNELS,
                        decoder_features_start=loaded_cfg.DECODER_FEATURES_START,
                        deep_supervision=loaded_cfg.DEEP_SUPERVISION,
                        pretrained_encoder=False, 
                        dropout_p=loaded_cfg.DROPOUT_P
                    ).to(self.device)
                else:
                    print(f"ERROR: Checkpoint config MODEL_NAME is {loaded_cfg.MODEL_NAME}, but current script expects ResNet50UNetPlusPlus. Cannot safely re-init.")
                    # Potentially fall back to current config model or raise error
                    # For now, we proceed with the model loaded by state_dict, but it might be structurally different.

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.visualizer.visualize_predictions(self.model, val_loader, self.device, num_samples=self.config.NUM_VISUALIZATION_SAMPLES, epoch="best_model")
        else:
            print(f"No best model found. Visualizing model from epoch {final_epoch_completed+1}.")
            self.visualizer.visualize_predictions(self.model, val_loader, self.device, num_samples=self.config.NUM_VISUALIZATION_SAMPLES, epoch=f"epoch_{final_epoch_completed+1}_final")

# %% [markdown]
# ## Main Execution

# %%
def main():
    config = ModelConfig(
        MODEL_NAME="ResNet50UNetPlusPlus",
        BATCH_SIZE=2, # ResNet50UNet++ is very memory heavy
        LEARNING_RATE=5e-5, # May need lower LR for deeper pretrained backbone & smaller batch
        NUM_EPOCHS=120,
        DECODER_FEATURES_START=64, # Can experiment with 32 or 64
        DEEP_SUPERVISION=True, 
        ENCODER_PRETRAINED=True,
        DROPOUT_P = 0.3, # Slightly higher dropout for deeper model
        EARLY_STOPPING_PATIENCE=20,
        DATA_DIR="dataset", 
        TRAIN_IMG_DIR="dataset/images",
        TRAIN_MASK_DIR="dataset/masks",
        SAVE_DIR="checkpoints_resnet50_unetplusplus", 
        FIGURE_DIR="figures_resnet50_unetplusplus"
    )

    trainer = ModelTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 