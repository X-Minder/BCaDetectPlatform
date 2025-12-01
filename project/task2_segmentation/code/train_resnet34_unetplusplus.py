# %% [markdown]
# # UNet++ (Nested U-Net) with ResNet34 Backbone - Bladder Segmentation Training
# This script implements UNet++ with a pre-trained ResNet34 backbone for medical image segmentation.
# Key Features:
# 1. Data Loading and Preprocessing
# 2. Model Training and Validation (with ResNet34 encoder)
# 3. Metrics Calculation and Visualization
# 4. Model Saving and Early Stopping
# 5. Deep Supervision (optional)
# 6. Learning Rate Scheduler (optional)
# 7. MixUp Augmentation (optional)

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
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass, field
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# %% [markdown]
# ## Configuration Class

# %%
@dataclass
class ModelConfig:
    # Basic Config
    MODEL_NAME: str = "ResNet34UNetPlusPlus"
    RANDOM_SEED: int = 42
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dataset Config
    DATA_DIR: str = "D:\project3\image_dataset\data_task2"
    TRAIN_IMG_DIR: str = "D:\project3\image_dataset\data_task2\images"
    TRAIN_MASK_DIR: str = "D:\project3\image_dataset\data_task2\masks"
    IMAGE_SIZE: int = 256
    BATCH_SIZE: int = 4 # Adjusted for ResNet34
    TRAIN_VAL_SPLIT: float = 0.8
    NUM_WORKERS: int = 1 # Set to 0 if DataLoader issues on Windows
    
    # Model Config (ResNet34UNetPlusPlus specific)
    IN_CHANNELS: int = 3
    OUT_CHANNELS: int = 1
    ENCODER_PRETRAINED: bool = True
    DECODER_FEATURES_START: int = 32 # For ResNet34, 32 is a common start
    DEEP_SUPERVISION: bool = True 
    DROPOUT_P: float = 0.2 

    # Training Config
    NUM_EPOCHS: int = 120 # Best score config
    LEARNING_RATE: float = 1e-4 # Best score config
    OPTIMIZER_NAME: str = "Adam" # Best score config: "Adam", "AdamW"
    LOSS_FUNCTION: str = "BCEWithLogitsLoss" # Best score config: "BCEWithLogitsLoss", "DiceLoss", "DiceBCELoss"
    USE_AMP: bool = True # Mixed precision training
    
    # LR Scheduler Config
    USE_LR_SCHEDULER: bool = False # Best score config
    LR_SCHEDULER_PATIENCE: int = 10 # Epochs to wait for improvement before reducing LR
    LR_SCHEDULER_FACTOR: float = 0.1 # Factor by which LR is reduced
    LR_SCHEDULER_MIN_LR: float = 1e-7 # Minimum learning rate

    # MixUp Config
    USE_MIXUP: bool = False # Best score config
    MIXUP_ALPHA: float = 0.2 # Alpha parameter for Beta distribution (e.g., 0.1 to 0.4)

    # Early Stopping Config
    EARLY_STOPPING_PATIENCE: int = 40 
    EARLY_STOPPING_MIN_DELTA: float = 0.0001 
    EARLY_STOPPING_METRIC: str = "dice" # "dice", "val_loss", "iou", etc.
    
    # Save Config
    SAVE_DIR: str = "checkpoints_resnet34_unetplusplus"
    
    # Visualization Config
    VISUALIZE_PREDICTIONS: bool = True
    NUM_VISUALIZATION_SAMPLES: int = 5
    FIGURE_DIR: str = "figures_resnet34_unetplusplus"

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
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats for ResNet
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
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        base_name, img_ext = os.path.splitext(img_name)
        mask_name_options = [
            f"{base_name}_mask.png", 
            f"{base_name}_mask{img_ext}", 
            f"{base_name}.png",
            f"{base_name}.jpeg", # Common alternatives
            f"{base_name}.jpg"
        ]
        mask_path = None
        for mn_opt in mask_name_options:
            potential_path = os.path.join(self.mask_dir, mn_opt)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
        
        if mask_path is None: 
            raise FileNotFoundError(f"Mask for image {img_name} not found in {self.mask_dir}. Looked for options like: {base_name}_mask.png, {base_name}.png etc.")
            
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path} or mask {mask_path}: {e}")
            
        mask[mask > 0] = 1.0 
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask'].unsqueeze(0)
        return image, mask

# %% [markdown]
# ## MixUp Augmentation Helper Function

# %%
def apply_mixup_batch(images, masks, alpha):
    if alpha <= 0: 
        return images, masks

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size()[0]
    index = torch.randperm(batch_size).to(images.device)

    mixed_images = lam * images + (1 - lam) * images[index, :]
    mixed_masks = lam * masks + (1 - lam) * masks[index, :]
    
    return mixed_images, mixed_masks

# %% [markdown]
# ## ResNet34-UNet++ Model Definition

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
        if dropout_p > 0 and dropout_p <=1: 
            layers.append(nn.Dropout2d(dropout_p))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class ResNet34UNetPlusPlus(nn.Module):
    def __init__(self, out_channels=1, decoder_features_start=32, deep_supervision=False, 
                 pretrained_encoder=True, dropout_p=0.2):
        super(ResNet34UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        
        # --- Encoder (ResNet34) ---
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained_encoder else None)
        
        self.encoder_x0_0_prepool = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) 
        self.pool = resnet.maxpool 
        self.encoder_x1_0 = resnet.layer1 
        self.encoder_x2_0 = resnet.layer2 
        self.encoder_x3_0 = resnet.layer3 
        self.encoder_x4_0 = resnet.layer4 

        # Channels from ResNet34 stages:
        enc0_0_ch = 64  
        enc1_0_ch = 64  
        enc2_0_ch = 128 
        enc3_0_ch = 256 
        enc4_0_ch = 512 
        
        # --- Decoder (UNet++ specific part) ---
        self.filters = [
            decoder_features_start,      
            decoder_features_start * 2,  
            decoder_features_start * 4,  
            decoder_features_start * 8   
        ]

        # --- Attention Gates for Encoder Features ---
        # AG for x0_0_feat (F_l=enc0_0_ch=64) gated by upsampled x1_0_feat (F_g=enc1_0_ch=64)
        self.att_enc0 = AttentionGate(F_g=enc1_0_ch, F_l=enc0_0_ch, F_int=enc0_0_ch // 2)
        # AG for x1_0_feat (F_l=enc1_0_ch=64) gated by upsampled x2_0_feat (F_g=enc2_0_ch=128)
        self.att_enc1 = AttentionGate(F_g=enc2_0_ch, F_l=enc1_0_ch, F_int=enc1_0_ch // 2)
        # AG for x2_0_feat (F_l=enc2_0_ch=128) gated by upsampled x3_0_feat (F_g=enc3_0_ch=256)
        self.att_enc2 = AttentionGate(F_g=enc3_0_ch, F_l=enc2_0_ch, F_int=enc2_0_ch // 2)
        # AG for x3_0_feat (F_l=enc3_0_ch=256) gated by upsampled x4_0_feat (F_g=enc4_0_ch=512)
        self.att_enc3 = AttentionGate(F_g=enc4_0_ch, F_l=enc3_0_ch, F_int=enc3_0_ch // 2)

        # --- Convolution Blocks for UNet++ Decoder Paths ---
        # These input channel calculations remain the same as the original UNet++
        # because attention is applied *before* concatenation to one of the inputs.

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

        # Final output layers for deep supervision
        if self.deep_supervision:
            self.final1 = nn.Conv2d(self.filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(self.filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(self.filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(self.filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(self.filters[0], out_channels, kernel_size=1)

    def _upsample_to_match(self, x_to_upsample, x_target_spatial_map):
        """Helper to upsample x_to_upsample to match spatial dimensions of x_target_spatial_map."""
        return F.interpolate(x_to_upsample, size=x_target_spatial_map.shape[2:], mode='bilinear', align_corners=True)

    def forward(self, x):
        # --- Encoder Path (Original Features) ---
        x0_0_orig = self.encoder_x0_0_prepool(x)      
        pooled_orig = self.pool(x0_0_orig)                 
        x1_0_orig = self.encoder_x1_0(pooled_orig)         
        x2_0_orig = self.encoder_x2_0(x1_0_orig)      
        x3_0_orig = self.encoder_x3_0(x2_0_orig)      
        x4_0_orig = self.encoder_x4_0(x3_0_orig) # This is the bottleneck      

        # --- Apply Attention to Encoder Features ---
        # The gating signal 'g' for X_i_0 comes from an upsampled version of X_{i+1}_0
        
        # Upsample deeper features to serve as gating signals
        g_for_x0_0 = self._upsample_to_match(x1_0_orig, x0_0_orig)
        x0_0_feat = self.att_enc0(g=g_for_x0_0, x=x0_0_orig) # Attended X_0_0

        g_for_x1_0 = self._upsample_to_match(x2_0_orig, x1_0_orig)
        x1_0_feat = self.att_enc1(g=g_for_x1_0, x=x1_0_orig) # Attended X_1_0

        g_for_x2_0 = self._upsample_to_match(x3_0_orig, x2_0_orig)
        x2_0_feat = self.att_enc2(g=g_for_x2_0, x=x2_0_orig) # Attended X_2_0

        g_for_x3_0 = self._upsample_to_match(x4_0_orig, x3_0_orig)
        x3_0_feat = self.att_enc3(g=g_for_x3_0, x=x3_0_orig) # Attended X_3_0
        
        x4_0_feat = x4_0_orig # Bottleneck feature, not gated further by a deeper one in this setup

        # --- Decoder Path (using attended encoder features) ---
        # The UNet++ structure now uses the attention-modified xN_0_feat versions.
        
        # Level 1
        up_x1_0 = self._upsample_to_match(x1_0_feat, x0_0_feat) # Use attended x1_0_feat for upsampling if preferred, or orig for gating
        x0_1 = self.conv0_1(torch.cat([x0_0_feat, up_x1_0], dim=1))
        
        up_x2_0 = self._upsample_to_match(x2_0_feat, x1_0_feat)
        x1_1 = self.conv1_1(torch.cat([x1_0_feat, up_x2_0], dim=1))
        
        up_x3_0 = self._upsample_to_match(x3_0_feat, x2_0_feat)
        x2_1 = self.conv2_1(torch.cat([x2_0_feat, up_x3_0], dim=1))
        
        up_x4_0 = self._upsample_to_match(x4_0_feat, x3_0_feat)
        x3_1 = self.conv3_1(torch.cat([x3_0_feat, up_x4_0], dim=1))

        # Level 2
        up_x1_1 = self._upsample_to_match(x1_1, x0_0_feat) # x1_1 is from decoder path
        x0_2 = self.conv0_2(torch.cat([x0_0_feat, x0_1, up_x1_1], dim=1))
        
        up_x2_1 = self._upsample_to_match(x2_1, x1_0_feat)
        x1_2 = self.conv1_2(torch.cat([x1_0_feat, x1_1, up_x2_1], dim=1))
        
        up_x3_1 = self._upsample_to_match(x3_1, x2_0_feat)
        x2_2 = self.conv2_2(torch.cat([x2_0_feat, x2_1, up_x3_1], dim=1))
        
        # Level 3
        up_x1_2 = self._upsample_to_match(x1_2, x0_0_feat)
        x0_3 = self.conv0_3(torch.cat([x0_0_feat, x0_1, x0_2, up_x1_2], dim=1))
        
        up_x2_2 = self._upsample_to_match(x2_2, x1_0_feat)
        x1_3 = self.conv1_3(torch.cat([x1_0_feat, x1_1, x1_2, up_x2_2], dim=1))
        
        # Level 4
        up_x1_3 = self._upsample_to_match(x1_3, x0_0_feat)
        x0_4 = self.conv0_4(torch.cat([x0_0_feat, x0_1, x0_2, x0_3, up_x1_3], dim=1))

        # --- Output ---
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

# %% [markdown]
# ## Custom Loss Functions
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Attention Gate.
        Args:
            F_g: Number of channels in the gating signal (from coarser scale).
            F_l: Number of channels in the input feature map (from encoder, finer scale).
            F_int: Number of channels in the intermediate layer.
        """
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
        """
        Args:
            g: Gating signal from the coarser scale (e.g., Up(X_{i+1}_j) ).
            x: Input feature map from the encoder path (e.g., X_i_0).
        Returns:
            Attention-weighted x.
        """
        # g (e.g., from X_1_0 upsampled) should be processed to match x's channels for W_g, or F_int
        # x (e.g., X_0_0) should be processed for W_x
        
        g1 = self.W_g(g)  # Gate processing
        x1 = self.W_x(x)  # Input feature processing
        
        # Add and ReLU
        # Ensure g1 and x1 have same spatial dimensions before adding.
        # Typically, g is upsampled to x's spatial size *before* entering the AG.
        psi_input = self.relu(g1 + x1)
        
        # Compute attention coefficients
        alpha = self.psi(psi_input) # Spatial attention map (values between 0 and 1)
        
        # Apply attention to the input feature map x
        return x * alpha
# %%
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice_coeff

class DiceBCELoss(nn.Module):
    def __init__(self, smooth_dice=1e-6, bce_weight=0.5, dice_weight=0.5): 
        super(DiceBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth_dice)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice

# %% [markdown]
# ## Early Stopping Class

# %%
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=True, metric_name="val_metric"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.metric_name = metric_name 
        self.counter = 0
        self.best_value = None
        self.early_stop = False

        if self.mode == 'min':
            self.delta_op = lambda current, best: best - current >= self.min_delta 
            self.best_op = lambda current, best: current <= best
        elif self.mode == 'max':
            self.delta_op = lambda current, best: current - best >= self.min_delta 
            self.best_op = lambda current, best: current >= best
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Choose 'min' or 'max'.")

    def __call__(self, current_value): 
        if self.best_value is None: 
            self.best_value = current_value
            if self.verbose: print(f"EarlyStopping for {self.metric_name}: Initialized best value to {self.best_value:.6f}")
            return False 

        is_better = self.best_op(current_value, self.best_value)
        is_significant_improvement = False
        if is_better:
            is_significant_improvement = self.delta_op(current_value, self.best_value)

        if is_significant_improvement:
            if self.verbose: print(f"EarlyStopping for {self.metric_name}: Metric significantly improved from {self.best_value:.6f} to {current_value:.6f}. Resetting counter.")
            self.best_value = current_value
            self.counter = 0
        else: 
            self.counter += 1
            if is_better: 
                 if self.verbose: print(f"EarlyStopping for {self.metric_name}: Metric improved to {current_value:.6f} (from {self.best_value:.6f}), but not significantly. Counter: {self.counter}/{self.patience}.")
            else: 
                 if self.verbose: print(f"EarlyStopping for {self.metric_name}: Metric did not improve from {self.best_value:.6f} (current: {current_value:.6f}). Counter: {self.counter}/{self.patience}.")
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose: print(f"EarlyStopping for {self.metric_name}: Stopping early after {self.patience} epochs of no significant improvement.")
        
        return self.early_stop


# %% [markdown]
# ## Metrics Calculator

# %%
class MetricsCalculator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def _calculate_stats(self, y_pred_sigmoid, y_true):
        y_pred = (y_pred_sigmoid > self.threshold).float()
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)

        tp = torch.sum((y_pred_flat == 1) & (y_true_flat == 1)).item()
        fp = torch.sum((y_pred_flat == 1) & (y_true_flat == 0)).item()
        fn = torch.sum((y_pred_flat == 0) & (y_true_flat == 1)).item()
        tn = torch.sum((y_pred_flat == 0) & (y_true_flat == 0)).item()
        return tp, fp, fn, tn

    def calculate_metrics(self, outputs_logits, masks):
        y_pred_sigmoid = torch.sigmoid(outputs_logits)
        tp, fp, fn, tn = self._calculate_stats(y_pred_sigmoid, masks)
        
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

    def plot_training_curves(self, history, epoch, lr_history=None):
        metrics_to_plot = ["loss", "dice", "iou", "sensitivity", "specificity"]
        num_plots = len(metrics_to_plot)
        if lr_history and len(lr_history) > 0 : num_plots +=1 

        cols = 3
        rows = (num_plots + cols -1) // cols
        
        fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False) # squeeze=False to always get 2D array
        axs = axs.flatten() 

        fig.suptitle(f'Training Progress - Epoch {epoch+1} - Model: {self.config.MODEL_NAME}', fontsize=16)
        
        titles = {
            "loss": "Loss", "dice": "Dice Coefficient", "iou": "IoU (Jaccard)",
            "sensitivity": "Sensitivity (Recall)", "specificity": "Specificity", "lr": "Learning Rate"
        }

        plot_idx = 0
        for metric_key in metrics_to_plot:
            if f'train_{metric_key}' not in history or f'val_{metric_key}' not in history:
                if plot_idx < len(axs): fig.delaxes(axs[plot_idx]) 
                plot_idx +=1
                continue

            ax = axs[plot_idx]
            ax.plot(history[f'train_{metric_key}'], label=f'Train {titles.get(metric_key, metric_key.capitalize())}')
            ax.plot(history[f'val_{metric_key}'], label=f'Validation {titles.get(metric_key, metric_key.capitalize())}')
            ax.set_title(titles.get(metric_key, metric_key.capitalize()))
            ax.set_xlabel('Epoch')
            ax.set_ylabel(titles.get(metric_key, metric_key.capitalize()))
            ax.legend()
            ax.grid(True)
            plot_idx +=1
        
        if lr_history and len(lr_history) > 0 and plot_idx < len(axs):
            ax = axs[plot_idx]
            ax.plot(lr_history, label='Learning Rate', color='green')
            ax.set_title(titles["lr"])
            ax.set_xlabel('Epoch')
            ax.set_ylabel('LR')
            ax.set_yscale('log') 
            ax.legend()
            ax.grid(True)
            plot_idx +=1

        for j in range(plot_idx, len(axs)): # Remove any remaining empty subplots
            fig.delaxes(axs[j])
            
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        save_path = os.path.join(self.figure_dir, f"{self.config.MODEL_NAME}_training_curves_epoch_{epoch+1}.png")
        plt.savefig(save_path)
        plt.close(fig)

    def visualize_predictions(self, model, dataloader, device, num_samples=5, epoch="final"):
        if not self.config.VISUALIZE_PREDICTIONS or num_samples == 0: return
        model.eval()
        samples_shown = 0
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples), squeeze=False) # squeeze=False for consistency
        
        fig.suptitle(f"Sample Predictions ({self.config.MODEL_NAME} - Epoch {epoch})", fontsize=16)
        
        with torch.no_grad():
            for images_batch, masks_batch_true in dataloader: # Use distinct names
                if samples_shown >= num_samples: break
                images_vis, masks_true_vis = images_batch.to(device), masks_batch_true.to(device)
                
                model_outputs = model(images_vis)
                outputs_logits = model_outputs[0] if isinstance(model_outputs, list) and self.config.DEEP_SUPERVISION else model_outputs 
                
                outputs_sigmoid = torch.sigmoid(outputs_logits)
                masks_pred = (outputs_sigmoid > 0.5).float()

                for j in range(images_vis.size(0)):
                    if samples_shown >= num_samples: break
                    
                    img_np = images_vis[j].cpu().permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = std * img_np + mean
                    img_np = np.clip(img_np, 0, 1)

                    mask_true_np = masks_true_vis[j].cpu().squeeze().numpy()
                    mask_pred_np = masks_pred[j].cpu().squeeze().numpy()
                    
                    axes[samples_shown, 0].imshow(img_np)
                    axes[samples_shown, 0].set_title("Image")
                    axes[samples_shown, 0].axis('off')
                    
                    axes[samples_shown, 1].imshow(mask_true_np, cmap='gray')
                    axes[samples_shown, 1].set_title("True Mask")
                    axes[samples_shown, 1].axis('off')
                    
                    axes[samples_shown, 2].imshow(mask_pred_np, cmap='gray')
                    axes[samples_shown, 2].set_title("Predicted Mask")
                    axes[samples_shown, 2].axis('off')
                    
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
        
        self.scheduler = None
        if self.config.USE_LR_SCHEDULER:
            scheduler_mode = 'min' if self.config.EARLY_STOPPING_METRIC == 'val_loss' else 'max'
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode=scheduler_mode, 
                factor=config.LR_SCHEDULER_FACTOR,
                patience=config.LR_SCHEDULER_PATIENCE,
                min_lr=config.LR_SCHEDULER_MIN_LR,
                verbose=True
            )
            
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            mode='min' if config.EARLY_STOPPING_METRIC == 'val_loss' else 'max',
            metric_name=f"val_{config.EARLY_STOPPING_METRIC}",
            verbose=True
        )
        
        metric_names = ["loss", "dice", "iou", "sensitivity", "specificity"]
        self.history = {f'{phase}_{m}': [] for phase in ["train", "val"] for m in metric_names}
        self.lr_history = []


    def _set_seed(self):
        torch.manual_seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed(self.config.RANDOM_SEED)
            torch.cuda.manual_seed_all(self.config.RANDOM_SEED) 

    def _get_model(self):
        print(f"Initializing model: {self.config.MODEL_NAME}")
        if self.config.MODEL_NAME == "ResNet34UNetPlusPlus":
            model = ResNet34UNetPlusPlus(
                out_channels=self.config.OUT_CHANNELS,
                decoder_features_start=self.config.DECODER_FEATURES_START,
                deep_supervision=self.config.DEEP_SUPERVISION,
                pretrained_encoder=self.config.ENCODER_PRETRAINED,
                dropout_p=self.config.DROPOUT_P
            )
        else: 
            raise ValueError(f"Unsupported model: {self.config.MODEL_NAME}. This script is for ResNet34UNetPlusPlus.")
        return model

    def _get_optimizer(self):
        if self.config.OPTIMIZER_NAME.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        elif self.config.OPTIMIZER_NAME.lower() == "adamw":
            return optim.AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        else: 
            raise ValueError(f"Unsupported optimizer: {self.config.OPTIMIZER_NAME}")

    def _get_loss_function(self):
        if self.config.LOSS_FUNCTION == "BCEWithLogitsLoss": 
            return nn.BCEWithLogitsLoss()
        elif self.config.LOSS_FUNCTION == "DiceLoss":
            return DiceLoss() 
        elif self.config.LOSS_FUNCTION == "DiceBCELoss":
            return DiceBCELoss() 
        else: 
            raise ValueError(f"Unsupported loss function: {self.config.LOSS_FUNCTION}")

    def _run_epoch(self, dataloader, is_train=True):
        if is_train: self.model.train()
        else: self.model.eval()
        
        total_loss = 0.0
        total_metrics = {k: 0.0 for k in ["dice", "iou", "sensitivity", "specificity"]}
        num_processed_samples = 0
        
        progress_bar_desc = "Training" if is_train else "Validating"
        
        for images, masks in tqdm(dataloader, desc=progress_bar_desc, leave=False):
            batch_size_actual = images.size(0)
            images_dev, masks_dev = images.to(self.device), masks.to(self.device)

            if is_train and self.config.USE_MIXUP:
                images_dev, masks_dev = apply_mixup_batch(images_dev, masks_dev, self.config.MIXUP_ALPHA)

            if is_train: self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_train): 
                with autocast(enabled=self.config.USE_AMP):
                    outputs = self.model(images_dev) 
                    
                    current_batch_loss = 0
                    if self.config.DEEP_SUPERVISION and isinstance(outputs, list):
                        for out_sup in outputs:
                            current_batch_loss += self.criterion(out_sup, masks_dev) 
                        current_batch_loss /= len(outputs) 
                        primary_output_for_metrics = outputs[0] 
                    else:
                        current_batch_loss = self.criterion(outputs, masks_dev)
                        primary_output_for_metrics = outputs
                
                if is_train:
                    self.scaler.scale(current_batch_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            total_loss += current_batch_loss.item() * batch_size_actual
            
            with torch.no_grad(): # Metrics calculation should not require gradients
                batch_metrics = self.metrics_calculator.calculate_metrics(primary_output_for_metrics.detach(), masks_dev)
            for key in total_metrics: 
                total_metrics[key] += batch_metrics[key] * batch_size_actual
            num_processed_samples += batch_size_actual
            
        avg_loss = total_loss / num_processed_samples if num_processed_samples > 0 else 0
        avg_metrics = {key: val / num_processed_samples if num_processed_samples > 0 else 0 for key, val in total_metrics.items()}
        return avg_loss, avg_metrics

    def _save_checkpoint(self, epoch, metric_value, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            f'val_{self.config.EARLY_STOPPING_METRIC}': metric_value,
            'config': self.config 
        }
        if self.config.USE_AMP: 
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        if self.scheduler and self.config.USE_LR_SCHEDULER:
             checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        filename_suffix = "best" if is_best else f"epoch_{epoch+1}"
        save_path = os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_NAME}_{filename_suffix}.pth")
        torch.save(checkpoint, save_path)
        
        metric_display_name = self.config.EARLY_STOPPING_METRIC.replace("val_", "") # Clean name for display
        print_msg = f"Checkpoint saved: {save_path} (Val {metric_display_name}: {metric_value:.4f})"
        if is_best: print_msg += " **Best**"
        print(print_msg)

    def train(self):
        print(f"Starting training for {self.config.MODEL_NAME} on {self.config.DEVICE}")
        print(f"Full configuration: {self.config}")

        full_dataset_no_transform = BladderDataset(self.config.TRAIN_IMG_DIR, self.config.TRAIN_MASK_DIR) 
        
        if len(full_dataset_no_transform) == 0:
            print(f"ERROR: No images found in {self.config.TRAIN_IMG_DIR}. Please check dataset paths.")
            return

        generator = torch.Generator().manual_seed(self.config.RANDOM_SEED)
        train_size = int(self.config.TRAIN_VAL_SPLIT * len(full_dataset_no_transform))
        val_size = len(full_dataset_no_transform) - train_size
        
        if train_size == 0 or val_size == 0:
            print(f"ERROR: Dataset too small for train/val split ({len(full_dataset_no_transform)} samples). Train: {train_size}, Val: {val_size}. Adjust TRAIN_VAL_SPLIT or dataset size.")
            return

        train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset_no_transform)), [train_size, val_size], generator=generator)

        train_dataset_with_transforms = BladderDataset(self.config.TRAIN_IMG_DIR, self.config.TRAIN_MASK_DIR, transform=self.data_transforms.get_train_transforms())
        val_dataset_with_transforms = BladderDataset(self.config.TRAIN_IMG_DIR, self.config.TRAIN_MASK_DIR, transform=self.data_transforms.get_val_transforms())
        
        train_subset = torch.utils.data.Subset(train_dataset_with_transforms, train_indices.indices)
        val_subset = torch.utils.data.Subset(val_dataset_with_transforms, val_indices.indices)
        
        train_loader = DataLoader(train_subset, batch_size=self.config.BATCH_SIZE, shuffle=True, 
                                  num_workers=self.config.NUM_WORKERS, pin_memory=(self.config.DEVICE=='cuda'), drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=self.config.BATCH_SIZE, shuffle=False, 
                                num_workers=self.config.NUM_WORKERS, pin_memory=(self.config.DEVICE=='cuda'))
        
        print(f"Train: {len(train_subset)} samples ({len(train_loader)} batches). Validation: {len(val_subset)} samples ({len(val_loader)} batches).")
        
        final_epoch_completed = -1

        for epoch in range(self.config.NUM_EPOCHS):
            final_epoch_completed = epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)
            print(f"--- Epoch {epoch+1}/{self.config.NUM_EPOCHS} --- LR: {current_lr:.2e} ---")
            
            train_loss, train_metrics = self._run_epoch(train_loader, is_train=True)
            val_loss, val_metrics = self._run_epoch(val_loader, is_train=False)

            self.history['train_loss'].append(train_loss); self.history['val_loss'].append(val_loss)
            for m_key in ["dice", "iou", "sensitivity", "specificity"]:
                self.history[f'train_{m_key}'].append(train_metrics[m_key])
                self.history[f'val_{m_key}'].append(val_metrics[m_key])
            
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Train Metrics: Dice {train_metrics['dice']:.4f}, IoU {train_metrics['iou']:.4f}, Sens {train_metrics['sensitivity']:.4f}, Spec {train_metrics['specificity']:.4f}")
            print(f"  Val Metrics:   Dice {val_metrics['dice']:.4f}, IoU {val_metrics['iou']:.4f}, Sens {val_metrics['sensitivity']:.4f}, Spec {val_metrics['specificity']:.4f}")
            
            self.visualizer.plot_training_curves(self.history, epoch, self.lr_history)
            
            current_val_metric_for_decision = val_metrics[self.config.EARLY_STOPPING_METRIC]
            
            if self.config.USE_LR_SCHEDULER and self.scheduler:
                scheduler_metric_to_use = val_loss if self.early_stopping.mode == 'min' else current_val_metric_for_decision
                self.scheduler.step(scheduler_metric_to_use)

            # Save best model based on EarlyStopping's tracking of best_value and significant improvement logic
            if self.early_stopping.best_value is None or \
               (self.early_stopping.mode == 'max' and current_val_metric_for_decision > self.early_stopping.best_value + self.early_stopping.min_delta) or \
               (self.early_stopping.mode == 'min' and current_val_metric_for_decision < self.early_stopping.best_value - self.early_stopping.min_delta) or \
               (self.early_stopping.mode == 'max' and current_val_metric_for_decision >= self.early_stopping.best_value and self.early_stopping.min_delta == 0.0) or \
               (self.early_stopping.mode == 'min' and current_val_metric_for_decision <= self.early_stopping.best_value and self.early_stopping.min_delta == 0.0) :
                # This condition ensures we save if it's better by min_delta, or just better/equal if min_delta is 0.
                # EarlyStopping call below will update its internal best_value correctly.
                 self._save_checkpoint(epoch, current_val_metric_for_decision, is_best=True)
            elif (epoch + 1) % 10 == 0: 
                self._save_checkpoint(epoch, current_val_metric_for_decision, is_best=False)

            if self.early_stopping(current_val_metric_for_decision): 
                print("Early stopping triggered.")
                break
        
        best_val_metric_str = f"{self.early_stopping.best_value:.4f}" if self.early_stopping.best_value is not None else "N/A"
        print(f"\nTraining finished after {final_epoch_completed + 1} epochs. Best validation {self.config.EARLY_STOPPING_METRIC}: {best_val_metric_str}")
        best_model_path = os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_NAME}_best.pth")
        if os.path.exists(best_model_path):
            print(f"Loading best model for final visualization: {best_model_path}")
            try:
                checkpoint = torch.load(best_model_path, map_location=self.device)
                
                loaded_cfg_from_ckpt = checkpoint.get('config')
                temp_model_to_load = self._get_model() # Get a fresh model instance based on current script's config or loaded_cfg

                if isinstance(loaded_cfg_from_ckpt, ModelConfig):
                    print("Config found in checkpoint. Verifying compatibility...")
                    if loaded_cfg_from_ckpt.MODEL_NAME == self.config.MODEL_NAME and \
                       loaded_cfg_from_ckpt.OUT_CHANNELS == self.config.OUT_CHANNELS and \
                       loaded_cfg_from_ckpt.DECODER_FEATURES_START == self.config.DECODER_FEATURES_START and \
                       loaded_cfg_from_ckpt.DEEP_SUPERVISION == self.config.DEEP_SUPERVISION:
                        print("Saved config is compatible. Loading model with saved architecture.")
                        # Re-init with saved config, ensuring correct architecture before loading state_dict
                        temp_model_to_load = ResNet34UNetPlusPlus( 
                            out_channels=loaded_cfg_from_ckpt.OUT_CHANNELS,
                            decoder_features_start=loaded_cfg_from_ckpt.DECODER_FEATURES_START,
                            deep_supervision=loaded_cfg_from_ckpt.DEEP_SUPERVISION,
                            pretrained_encoder=False, 
                            dropout_p=loaded_cfg_from_ckpt.DROPOUT_P
                        ).to(self.device)
                    else:
                        print("WARNING: Saved model architecture in checkpoint differs from current config. State_dict might not load correctly.")
                else:
                     print("Warning: Config not found or not a ModelConfig instance in checkpoint. Attempting to load state_dict into current model structure.")

                temp_model_to_load.load_state_dict(checkpoint['model_state_dict'])
                self.model = temp_model_to_load 
                
                self.visualizer.visualize_predictions(self.model, val_loader, self.device, num_samples=self.config.NUM_VISUALIZATION_SAMPLES, epoch="best_model_final_eval")
            except Exception as e:
                print(f"Error loading best model or visualizing: {e}. Visualizing with last epoch model instead.")
                self.visualizer.visualize_predictions(self.model, val_loader, self.device, num_samples=self.config.NUM_VISUALIZATION_SAMPLES, epoch=f"epoch_{final_epoch_completed+1}_final_state")
        else:
            print(f"No best model checkpoint found at {best_model_path}. Visualizing model from final epoch {final_epoch_completed+1}.")
            self.visualizer.visualize_predictions(self.model, val_loader, self.device, num_samples=self.config.NUM_VISUALIZATION_SAMPLES, epoch=f"epoch_{final_epoch_completed+1}_final_state")

# %% [markdown]
# ## Main Execution

# %%
def main():
    config = ModelConfig() # Uses defaults set for ResNet34 best score

    # Example of overriding for a specific run:
    # config = ModelConfig(
    #     NUM_EPOCHS = 150,
    #     LEARNING_RATE = 5e-5,
    #     USE_LR_SCHEDULER = True,
    #     LOSS_FUNCTION = "DiceBCELoss",
    #     DROPOUT_P = 0.3
    # )

    trainer = ModelTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
#%%