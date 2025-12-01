# %% [markdown]
# # Nectin-4 膀胱镜图像二分类模型
# 通过迁移学习（EfficientNet-B0）对 Nectin-4 进行二分类 (Low/High)，按患者划分训练/验证集。
# Nectin-4 原始标签 1, 2, 3 将被映射为：1 -> 0 (Low), 2/3 -> 1 (High)

# %%
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support, roc_curve, average_precision_score, precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights # Ensure efficientnet_b0 is used if model_name is \'efficientnet_b0\'
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.utils import make_grid, save_image
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.utils import shuffle as sklearn_shuffle
import logging
import sys # For logger
from datetime import datetime # For logger timestamp
import json # For saving/loading config and history


# 配置matplotlib中文字体
def setup_chinese_font():
    import matplotlib.font_manager as fm
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = 'DejaVu Sans'
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    plt.rcParams['font.sans-serif'] = [selected_font]
    plt.rcParams['axes.unicode_minus'] = False
    print(f"Matplotlib using font: {selected_font}")

setup_chinese_font()

# reproducibility and device
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) # if using multi-GPU
# torch.backends.cudnn.deterministic = True # Can impact performance
# torch.backends.cudnn.benchmark = False   # Can impact performance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 全局常量和Config类定义
# %%

# Data related constants (will be part of Config)
LABEL_COLUMN_NAME = 'Nectin-4'  # Original label column in CSV
MAPPED_LABEL_COLUMN_NAME = 'Nectin-4_mapped' # Mapped label column (0 for Low, 1 for High)
PATIENT_ID_COLUMN = 'PATIENT_ID'
FILE_NAME_COLUMN = 'FILE_NAME'

# Model and training related constants (will be part of Config)
NUM_CLASSES_NECTIN4 = 2 # Binary classification: Low vs High

class Config:
    def __init__(self):
        # 数据相关
        self.label_file = "Nectin4/dataset/label.csv"
        self.image_dir = "Nectin4/dataset/images"
        self.num_classes = NUM_CLASSES_NECTIN4
        self.class_names = ['Nectin4-Low', 'Nectin4-High'] # Mapped: 0 (Low), 1 (High)
        self.label_column_name = LABEL_COLUMN_NAME
        self.mapped_label_column_name = MAPPED_LABEL_COLUMN_NAME
        self.patient_id_column = PATIENT_ID_COLUMN
        self.file_name_column = FILE_NAME_COLUMN
    
    # 模型相关
        self.model_name = "efficientnet_b0" # Can be changed
        self.pretrained = True
        self.dropout_rate = 0.3 # Example dropout rate
    
    # 训练相关
        self.device = device
        self.batch_size = 16
        self.num_epochs = 70 # Example from grade_model
        self.learning_rate = 1e-5 # Example from grade_model
        self.weight_decay = 1e-4
        self.early_stopping_patience = 15
        self.val_size_from_trainval = 0.2 # Proportion for validation set
        self.random_seed = SEED
        self.num_workers = 0 # For DataLoader
        
        # 优化器相关
        self.optimizer_name = 'AdamW' # 'Adam', 'SGD', etc.
        self.scheduler_name = 'ReduceLROnPlateau' # 'StepLR', 'CosineAnnealingLR', etc.
        self.scheduler_patience = 8 # For ReduceLROnPlateau
        self.scheduler_factor = 0.5 # For ReduceLROnPlateau
        self.min_lr = 1e-7 # Minimum learning rate
        self.warmup_epochs = 3 # Number of warmup epochs for LR
        self.gradient_clip_val = 1.0 # Optional gradient clipping
        
        # 损失函数相关 (FocalLoss)
        self.focal_alpha = None # Will be calculated based on class distribution
        self.focal_gamma = 2.0
        
        # 图像处理
        self.image_size = (224, 224) # EfficientNet-B0 typical input size
        self.normalize_mean = (0.485, 0.456, 0.406)
        self.normalize_std = (0.229, 0.224, 0.225)
        self.use_augmentation = True
        self.rotation_degrees = 15
        self.brightness_factor = 0.1
        self.contrast_factor = 0.1
        
        # 保存相关
        self.save_dir = f"checkpoints_nectin4"
        self.model_filename_prefix = f"{self.model_name}_nectin4"

        # Transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(self.rotation_degrees),
            transforms.ColorJitter(brightness=self.brightness_factor, contrast=self.contrast_factor),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
        self.val_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])
    
    def get_optimizer(self, model_parameters):
        if self.optimizer_name.lower() == 'adamw':
            return optim.AdamW(model_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == 'adam':
            return optim.Adam(model_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
    
    def get_scheduler(self, optimizer):
        if self.scheduler_name.lower() == 'reducelronplateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.scheduler_patience, factor=self.scheduler_factor, verbose=True, min_lr=self.min_lr)
        # Add other schedulers if needed
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
            
    def save_config(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and not callable(v) and not isinstance(v, (torch.device, transforms.Compose))}
        config_dict['device'] = str(self.device) # Store device as string
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        logging.info(f"Config saved to {save_path}")
    
    @classmethod
    def load_config(cls, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        loaded_config = cls()
        for key, value in config_dict.items():
            if hasattr(loaded_config, key):
                setattr(loaded_config, key, value)
        loaded_config.device = torch.device(loaded_config.device) # Convert device string back to torch.device
        # Re-init transforms as they are not directly serializable
        loaded_config.train_transform = transforms.Compose([
            transforms.Resize(loaded_config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(loaded_config.rotation_degrees),
            transforms.ColorJitter(brightness=loaded_config.brightness_factor, contrast=loaded_config.contrast_factor),
            transforms.ToTensor(),
            transforms.Normalize(mean=loaded_config.normalize_mean, std=loaded_config.normalize_std)
        ])
        loaded_config.val_transform = transforms.Compose([
            transforms.Resize(loaded_config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=loaded_config.normalize_mean, std=loaded_config.normalize_std)
        ])
        logging.info(f"Config loaded from {config_path}")
        return loaded_config

    def __str__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith('_') and not callable(v) and not isinstance(v, (torch.device, transforms.Compose))])

config = Config() # Global config instance

# %% [markdown]
# ## Focal Loss 定义
# %%
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean', num_classes=config.num_classes, class_weights=None): 
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # alpha can be a list [alpha_class0, alpha_class1] or a single float
        self.num_classes = num_classes
        self.reduction = reduction
        self.class_weights = class_weights # For F.cross_entropy weight argument

        # Alpha handling (adapted for potential list or single float)
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                if self.num_classes == 2: # For binary, a single alpha often refers to the positive class (class 1)
                    # If alpha is for class 1, alpha for class 0 is 1-alpha. Or pass as list.
                    # For simplicity, if single alpha & binary, we make it [1-alpha, alpha] if one wants to interpret it this way.
                    # However, pytorch_grad_cam BaseLoss expects alpha to be a tensor of shape (num_classes,) or (batch_size, num_classes)
                    # If it's a single value, it usually means alpha for all classes.
                    # The original code had a complex interpretation. Let's simplify:
                    # If single float, assume it's a weight for class 1, and class 0 is 1-alpha if this is desired.
                    # More robust: expect a list or a tensor. If single float, apply it to all.
                    print(f"Warning: Single float alpha ({alpha}) for FocalLoss with {num_classes} classes. Interpreting as equal weight or specific list needed.")
                    self.alpha = torch.tensor([alpha] * num_classes) # Apply to all classes equally
                else: # Multi-class
                    self.alpha = torch.tensor([alpha] * num_classes)
            elif isinstance(alpha, list):
                self.alpha = torch.tensor(alpha)
            
            if self.alpha is not None and len(self.alpha) != num_classes:
                print(f"Warning: Alpha list length {len(self.alpha)} does not match num_classes={num_classes}. Alpha will not be applied per class as intended or may error.")
                # Potentially adjust or error out. For now, just warn.
                # self.alpha = None # Disable alpha if mismatched
    
    def forward(self, inputs, targets):
        # Use class_weights in F.cross_entropy if provided
        CE_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (1-pt)**self.gamma * CE_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # Ensure alpha is correctly shaped for gather (targets are 0, 1, ...)
            if self.alpha.ndim == 1 and len(self.alpha) == self.num_classes:
                at = self.alpha.gather(0, targets.data.view(-1))
                F_loss = at * F_loss
            elif self.alpha.numel() == 1: # Scalar alpha applied to all
                F_loss = self.alpha * F_loss
            else:
                print(f"Warning: FocalLoss alpha shape {self.alpha.shape} not compatible for per-class weighting with {self.num_classes} classes. Alpha not applied effectively.")

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# %% [markdown]
# ## 自定义 Dataset for Nectin-4
# %%
class Nectin4Dataset(Dataset): 
    def __init__(self, df, image_dir, transform=None, file_name_col=config.file_name_column, label_col=config.mapped_label_column_name):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.file_name_col = file_name_col
        self.label_col = label_col

        # Pre-check image paths (optional, can be done once after df load)
        # self.df['image_path_check'] = self.df[self.file_name_col].apply(lambda x: os.path.join(self.image_dir, x))
        # self.df = self.df[self.df['image_path_check'].apply(os.path.exists)]
        # if len(self.df) == 0:
        #     print(f"Warning: Nectin4Dataset initialized with DataFrame resulting in 0 valid image paths from column '{self.file_name_col}'.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row[self.file_name_col]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            logging.error(f"Image not found: {img_path}. Skipping or returning placeholder.")
            # Return a placeholder or handle as per requirement
            return torch.zeros((3, config.image_size[0], config.image_size[1])), torch.tensor(-1, dtype=torch.long) # Placeholder
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}. Skipping or returning placeholder.")
            return torch.zeros((3, config.image_size[0], config.image_size[1])), torch.tensor(-1, dtype=torch.long) # Placeholder
            
        if self.transform:
            img = self.transform(img)
        
        label = int(row[self.label_col]) # Assumes label_col contains the mapped 0/1 labels
        return img, label

# %% [markdown]
# ## 读取标签并划分数据集 for Nectin-4
# %%
def load_and_preprocess_nectin4_data(cfg: Config):
    logging.info(f"1. Reading label file: {cfg.label_file}")
    label_df_raw = pd.read_csv(cfg.label_file)
    logging.info(f"   Initial rows loaded: {len(label_df_raw)}")

    # Drop NA from critical columns
    required_cols = [cfg.label_column_name, cfg.file_name_column, cfg.patient_id_column]
    label_df = label_df_raw.dropna(subset=required_cols).copy()
    logging.info(f"2. Rows after dropping NA from key columns ('{ ', '.join(required_cols)}'): {len(label_df)}")

    if len(label_df) == 0:
        logging.error(f"ERROR: All rows were dropped after initial NA check for {cfg.label_column_name}. Please check CSV.")
        sys.exit("Label data processing failed due to NA drop.")
    
    logging.info(f"   Unique values in '{cfg.label_column_name}' before mapping: {label_df[cfg.label_column_name].astype(str).unique()}")

    # Mapping Nectin-4 labels (1 -> 0 for Low, 2/3 -> 1 for High)
    def map_nectin4_to_binary(nectin4_status):
        if pd.isna(nectin4_status):
            return np.nan
        status_str = str(nectin4_status).strip()
        if not status_str:
            return np.nan
        try:
            status_val_float = float(status_str)
            status_val = int(status_val_float)
            if status_val != status_val_float: # Check for decimals
                logging.warning(f"Nectin-4 status '{nectin4_status}' (str: '{status_str}') has non-zero decimal. Mapping to NaN.")
                return np.nan

            if status_val == 1: # Original Low
                return 0 # Mapped Low
            elif status_val in [2, 3]: # Original High
                return 1 # Mapped High
            else:
                logging.warning(f"Unexpected Nectin-4 status '{nectin4_status}' (parsed as {status_val}). Not 1, 2, or 3. Mapping to NaN.")
                return np.nan
        except ValueError:
            logging.warning(f"Could not convert Nectin-4 status '{nectin4_status}' (str: '{status_str}') to number. Mapping to NaN.")
            return np.nan

    label_df[cfg.mapped_label_column_name] = label_df[cfg.label_column_name].apply(map_nectin4_to_binary)
    logging.info(f"3. Rows after applying 'map_nectin4_to_binary' (before dropping NA from '{cfg.mapped_label_column_name}'): {len(label_df)}")
    logging.info(f"   Unique values in '{cfg.mapped_label_column_name}' after mapping (before dropping NA): {label_df[cfg.mapped_label_column_name].unique()}")

    label_df = label_df.dropna(subset=[cfg.mapped_label_column_name]).copy()
    logging.info(f"4. Rows after dropping NA from '{cfg.mapped_label_column_name}': {len(label_df)}")
    
    if len(label_df) == 0:
        logging.error(f"ERROR: All rows were dropped after mapping {cfg.label_column_name} and NA check on mapped column. Check mapping logic and original values.")
        sys.exit("Label data processing failed after mapping.")

    label_df[cfg.mapped_label_column_name] = label_df[cfg.mapped_label_column_name].astype(int)
    logging.info(f"   Final unique mapped values in '{cfg.mapped_label_column_name}': {label_df[cfg.mapped_label_column_name].unique()}")
    
    # Check image file existence
    label_df['image_path_check'] = label_df[cfg.file_name_column].apply(lambda x: os.path.join(cfg.image_dir, x))
    missing_images_df = label_df[~label_df['image_path_check'].apply(os.path.exists)]
    if not missing_images_df.empty:
        logging.warning(f"{len(missing_images_df)} image files listed in CSV do not exist in {cfg.image_dir}. These rows will be dropped.")
        logging.warning(f"Missing files examples: {missing_images_df[cfg.file_name_column].head().tolist()}")
        label_df = label_df[label_df['image_path_check'].apply(os.path.exists)].copy()
        logging.info(f"   Rows after checking image file existence: {len(label_df)}")
    label_df = label_df.drop(columns=['image_path_check'])

    if len(label_df) == 0:
        logging.error(f"ERROR: All rows were dropped after checking for image file existence. Ensure image_dir '{cfg.image_dir}' and file names in CSV are correct.")
        sys.exit("Label data processing failed due to missing image files.")

    # Data splitting (train and val only, similar to grade_model.py)
    df_trainval_full = label_df.copy()
    
    gss_val_split = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size_from_trainval, random_state=cfg.random_seed)

    df_train = pd.DataFrame()
    df_val = pd.DataFrame()

    if len(df_trainval_full[cfg.patient_id_column].unique()) > 1 and len(df_trainval_full) > 1:
        try:
            train_idx, val_idx = next(gss_val_split.split(df_trainval_full, groups=df_trainval_full[cfg.patient_id_column]))
            df_train = df_trainval_full.iloc[train_idx].copy()
            df_val   = df_trainval_full.iloc[val_idx].copy()
            logging.info(f\"Successfully used GroupShuffleSplit for train/validation sets for {cfg.label_column_name}.\")
        except ValueError as e:
            logging.warning(f\"GroupShuffleSplit for {cfg.label_column_name} train/validation failed: {e}. Falling back to random stratified split.\")
            stratify_col = df_trainval_full[cfg.mapped_label_column_name] if df_trainval_full[cfg.mapped_label_column_name].nunique() > 1 else None
            df_train, df_val = train_test_split(df_trainval_full, test_size=cfg.val_size_from_trainval, random_state=cfg.random_seed, stratify=stratify_col)
    elif len(df_trainval_full) > 0:
        logging.warning(f\"Not enough unique patient groups or samples for GroupShuffleSplit for {cfg.label_column_name}. Using random stratified split or assigning all to train.\")
        if len(df_trainval_full) > 1:
            stratify_col = df_trainval_full[cfg.mapped_label_column_name] if df_trainval_full[cfg.mapped_label_column_name].nunique() > 1 else None
            df_train, df_val = train_test_split(df_trainval_full, test_size=cfg.val_size_from_trainval, random_state=cfg.random_seed, stratify=stratify_col)
        else: # Only one sample
            df_train = df_trainval_full.copy()
            # df_val will remain empty
            logging.warning(f\"Only one sample in {cfg.label_column_name} data. Train set has 1 sample, validation set is empty.\")
    else: # Should not happen if previous checks were correct
        logging.error(f\"ERROR: {cfg.label_column_name} data is empty before splitting. Train and Val sets are empty.\")
        sys.exit(\"Data splitting failed.\")

    logging.info(f\"\\nDataset sizes and class distributions ({cfg.label_column_name} - Mapped Labels 0,1):\")
    for name, df_subset in [("Train", df_train), ("Val", df_val)]:
        if not df_subset.empty:
            logging.info(f\"  {name:<8}: {len(df_subset):>4} images, Patients: {df_subset[cfg.patient_id_column].nunique():>2}\")
            distribution_info_series = df_subset[cfg.mapped_label_column_name].value_counts(normalize=True).sort_index()
            distribution_info_str = \'\\n\'.join([f\"    Class {idx} ({cfg.class_names[idx]}): {val:.4f}\" for idx, val in distribution_info_series.items()])
            logging.info(f\"    Class distribution ({cfg.mapped_label_column_name}, normalized):\\n{distribution_info_str}\")
            logging.info(f\"    Unique patients per class ({cfg.mapped_label_column_name}):\")
            if cfg.mapped_label_column_name in df_subset.columns and cfg.patient_id_column in df_subset.columns:
                for class_label_mapped_val in sorted(df_subset[cfg.mapped_label_column_name].unique()):
                    num_patients_in_class = df_subset[df_subset[cfg.mapped_label_column_name] == class_label_mapped_val][cfg.patient_id_column].nunique()
                    logging.info(f\"      Class {class_label_mapped_val} ({cfg.class_names[class_label_mapped_val]}): {num_patients_in_class} patients\")
            else:
                logging.warning(\"      Could not calculate unique patients per class (column missing).\")
        else:
            logging.info(f\"  {name:<8}: Empty\")
    logging.info(\"\\n\")
    return df_train, df_val

# %% [markdown]
# ## 数据增强与 DataLoader for Nectin-4
# %%
def get_nectin4_dataloaders(cfg: Config, df_train: pd.DataFrame, df_val: pd.DataFrame):
    train_ds = Nectin4Dataset(df_train, cfg.image_dir, transform=cfg.train_transform) if not df_train.empty else None
    val_ds = Nectin4Dataset(df_val, cfg.image_dir, transform=cfg.val_transform) if not df_val.empty else None

    train_loader_args = {\'batch_size\': cfg.batch_size, \'num_workers\': cfg.num_workers, \'pin_memory\': True}
    if train_ds:
        counts_train_mapped = df_train[cfg.mapped_label_column_name].value_counts().sort_index()
        if len(counts_train_mapped) > 0 and len(counts_train_mapped) <= cfg.num_classes:
            # Sampler weights for mapped classes (0, 1 for Nectin-4 binary)
            class_sample_weights_values = [0.0] * cfg.num_classes
            for i in range(cfg.num_classes): # Iterate 0, 1
                class_sample_weights_values[i] = 1.0 / counts_train_mapped.get(i, 1e-6) # Use mapped class index i
            
            sample_weights_train = [class_sample_weights_values[label] for label in df_train[cfg.mapped_label_column_name]]
            sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights_train), num_samples=len(sample_weights_train), replacement=True)
            
            weights_str_sampler = ", ".join([f\"Class {i} ({cfg.class_names[i]}): {w:.4f}\" for i, w in enumerate(class_sample_weights_values)])
            logging.info(f\"Sampler weights for {cfg.label_column_name} mapped classes: {weights_str_sampler}\")
            train_loader_args[\'sampler\'] = sampler
            train_loader_args[\'shuffle\'] = False # Sampler handles shuffling
        else:
            logging.warning(f\"Training data for {cfg.label_column_name} has insufficient or unexpected mapped class counts for sampler. Using standard DataLoader with shuffle=True.\")
            train_loader_args[\'shuffle\'] = True
    else: 
        logging.warning(f\"df_train for {cfg.label_column_name} is empty. Train loader will be None.\")
        train_loader_args[\'shuffle\'] = False # Not relevant if loader is None

    train_loader = DataLoader(train_ds, **train_loader_args) if train_ds else None
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True) if val_ds else None
    
    return train_loader, val_loader, train_ds, val_ds


# %% [markdown]
# ## 模型定义 for Nectin-4
# %%
class Nectin4Classifier(nn.Module):
    def __init__(self, cfg: Config):
        super(Nectin4Classifier, self).__init__()
        self.cfg = cfg
        
        if cfg.model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            self.base_model = efficientnet_b0(weights=weights)
            in_features = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(in_features, cfg.num_classes) # Directly replace for binary
        # Add other models here if needed, e.g., resnet34, swin_t
        # elif cfg.model_name == "resnet34":
        #     weights = ResNet34_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        #     self.base_model = resnet34(weights=weights)
        #     in_features = self.base_model.fc.in_features
        #     self.base_model.fc = nn.Linear(in_features, cfg.num_classes)
        else:
            raise ValueError(f\"Unsupported model architecture: {cfg.model_name}\")
            
        # For Grad-CAM, if using the full base_model, need to identify feature layers
        # For EfficientNet, self.base_model.features would be the feature extractor part
        # self.features_for_cam = self.base_model.features # Example for EfficientNet
        # If a more complex head is added later (like in original Trop2), this would change.

    def forward(self, x):
        return self.base_model(x)

    # Helper to get feature extraction part if needed (e.g., for Grad-CAM if not using a hook on base_model directly)
    def get_features_module(self):
        if self.cfg.model_name.startswith(\'efficientnet\'):
            return self.base_model.features
        # Add for other models
        # elif self.cfg.model_name.startswith(\'resnet\'):
        #     return nn.Sequential(*list(self.base_model.children())[:-2]) # Up to before avgpool and fc
        return None


# %% [markdown]
# ## 早停机制
# %%
class EarlyStopping:
    def __init__(self, patience=config.early_stopping_patience, min_delta=0.0001, restore_best_weights=True, mode=\'min\', verbose=True, task_name="Nectin-4"): 
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        self.task_name = task_name # For logging
        
        if mode == \'max\': # For metrics like AUC or Accuracy
            self.monitor_op = lambda current, best: current > best + self.min_delta
            self.best_score = float(\'-inf\')
        else: # For metrics like Loss
            self.monitor_op = lambda current, best: current < best - self.min_delta
            self.best_score = float(\'inf\')

    def __call__(self, score, model):
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy() # Deep copy
            if self.verbose:
                logging.info(f\"EarlyStopping ({self.task_name}): New best score ({self.mode}): {self.best_score:.4f}\") 
        else:
            self.counter += 1
            if self.verbose:
                logging.info(f\"EarlyStopping ({self.task_name}): Counter {self.counter}/{self.patience}. Best score ({self.mode}): {self.best_score:.4f}\") 
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                logging.info(f\"EarlyStopping ({self.task_name}): Patience reached. Stopping training. Best score ({self.mode}): {self.best_score:.4f}\") 
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights) # Restore best weights
                if self.verbose:
                    logging.info(f\"EarlyStopping ({self.task_name}): Restored best model weights.\") 
        return self.early_stop

# %% [markdown]
# ## 训练与验证循环 for Nectin-4
# %%
def train_one_epoch(model, train_loader, criterion, optimizer, device, cfg, epoch_num, scaler=None):
        model.train()
        running_loss = 0.0
    total_samples = 0
    
    pbar_desc = f\"Epoch {epoch_num}/{cfg.num_epochs} [Train {cfg.label_column_name}]\"
    pbar = tqdm(train_loader, desc=pbar_desc, leave=False)
    
    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # LR Warmup
        if epoch_num <= cfg.warmup_epochs and cfg.warmup_epochs > 0:
            # Linear warmup
            lr_scale = min(1., float(epoch_num * len(train_loader) + i + 1) / float(cfg.warmup_epochs * len(train_loader)))
                for param_group in optimizer.param_groups:
                param_group[\'lr\'] = cfg.learning_rate * lr_scale
            
            optimizer.zero_grad()
            
        if scaler: # Mixed precision
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if cfg.gradient_clip_val is not None:
                scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
        else: # Full precision
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if cfg.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
        current_lr_iter = optimizer.param_groups[0][\'lr\']
        pbar.set_postfix({\'loss\': f\'{loss.item():.4f}\', \'lr\': f\'{current_lr_iter:.2e}\'})
        
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return epoch_loss, current_lr_iter # Return last LR of the epoch

def validate_one_epoch(model, val_loader, criterion, device, cfg):
        model.eval()
    running_loss = 0.0
    all_val_labels_list = []
    all_val_probs_list = [] # For Nectin-4 binary, store probs for class 1
    total_samples = 0
    
        with torch.no_grad():
        pbar_desc = f\"[Val {cfg.label_column_name}]\"
        pbar = tqdm(val_loader, desc=pbar_desc, leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
                probs = torch.softmax(outputs, dim=1)
            all_val_labels_list.extend(labels.cpu().numpy())
            all_val_probs_list.append(probs.cpu().numpy()) # Store all probabilities for multi-class if needed
                                                            # For binary, probs[:, 1] will be used for AUC typically
    
    epoch_loss = running_loss / total_samples if total_samples > 0 else float(\'nan\')
    
    all_val_labels_np = np.array(all_val_labels_list)
    all_val_probs_np = np.concatenate(all_val_probs_list, axis=0) if all_val_probs_list else np.array([])

    val_accuracy = 0.0
        val_auc = 0.0

    if total_samples > 0 and all_val_labels_np.size > 0 and all_val_probs_np.size > 0:
        all_val_preds_np = np.argmax(all_val_probs_np, axis=1)
        val_accuracy = np.mean(all_val_labels_np == all_val_preds_np)
        
        if len(np.unique(all_val_labels_np)) >= 2 : # Need at least two classes for AUC
            try: # For Nectin-4 (binary), use probs of the positive class (class 1)
                val_auc = roc_auc_score(all_val_labels_np, all_val_probs_np[:, 1])
            except ValueError as e_auc:
                 logging.warning(f\"Val AUC calculation error for {cfg.label_column_name}: {e_auc}. AUC set to 0.0 or NaN.\")
                 val_auc = float(\'nan\') # Or 0.0
            else:
             logging.warning(f\"{cfg.label_column_name} val set does not have enough distinct classes for AUC. AUC set to 0.0 or NaN.\")
             val_auc = float(\'nan\') # Or 0.0
            else:
        val_accuracy = float(\'nan\')
        val_auc = float(\'nan\')
            
    return epoch_loss, val_accuracy, val_auc, all_val_labels_np, all_val_probs_np


def train_nectin4_model(cfg: Config, model, train_loader, val_loader, criterion, optimizer, scheduler):
    history = {
        \'train_loss\': [], \'val_loss\': [], 
        \'val_accuracy\': [], \'val_auc\': [], 
        \'lr\': []
    }
    best_model_state = None
    
    # Monitor validation loss for ReduceLROnPlateau and EarlyStopping
    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience, mode=\'min\', task_name=cfg.label_column_name, restore_best_weights=True)
    
    use_amp = torch.cuda.is_available() # Use mixed precision if cuda is available
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, cfg.num_epochs + 1):
        train_epoch_loss, current_lr = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device, cfg, epoch, scaler if use_amp else None)
        history[\'train_loss\'].append(train_epoch_loss)
        history[\'lr\'].append(current_lr)
        
        val_epoch_loss, val_accuracy, val_auc, _, _ = validate_one_epoch(model, val_loader, criterion, cfg.device, cfg)
        history[\'val_loss\'].append(val_epoch_loss)
        history[\'val_accuracy\'].append(val_accuracy) # Store as 0-1
        history[\'val_auc\'].append(val_auc)
        
        logging.info(f\"Epoch {epoch}/{cfg.num_epochs}: Train Loss={train_epoch_loss:.4f}, Val Loss={val_epoch_loss:.4f}, Val Acc={val_accuracy:.4f}, Val AUC={val_auc:.4f}, LR={current_lr:.2e}\")

        # Scheduler step (after warmup for some schedulers, ReduceLROnPlateau uses val_loss)
        if epoch > cfg.warmup_epochs or cfg.warmup_epochs == 0: # Allow scheduler from epoch 1 if no warmup
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_epoch_loss) 
            # Add other non-warmup dependent schedulers here if needed for step()
            # else: scheduler.step() # For schedulers that step each epoch regardless of warmup logic in train_one_epoch

        # Early stopping check
        if early_stopping(val_epoch_loss, model):
            logging.info(f\"Early stopping triggered for {cfg.label_column_name}.\")
            break
        
        # Update best_model_state if this epoch is the best so far based on val_loss (as monitored by EarlyStopping)
        if val_epoch_loss == early_stopping.best_score : # If current val_loss is the best score recorded by early_stopping
            best_model_state = {
                \'epoch\': epoch,
                \'model_state_dict\': model.state_dict().copy(), # Get a fresh copy
                \'optimizer_state_dict\': optimizer.state_dict(),
                \'scheduler_state_dict\': scheduler.state_dict() if scheduler else None,
                \'val_loss\': val_epoch_loss,
                \'val_accuracy\': val_accuracy,
                \'val_auc': val_auc
            }
            # Logging for best model save is handled by EarlyStopping if restore_best_weights is True
            # Or, could save a checkpoint file here:
            # torch.save(best_model_state, os.path.join(cfg.save_dir, f\"{cfg.model_filename_prefix}_best_epoch_{epoch}.pth\"))

    # If early stopping restored weights, best_model_state should reflect that.
    # The model object itself is already updated by early_stopping if restore_best_weights=True.
    # We need to ensure best_model_state dict is consistent.
    if early_stopping.early_stop and early_stopping.restore_best_weights:
        logging.info(f\"Final best model weights restored from epoch with val_loss: {early_stopping.best_score:.4f}\")
        # If best_model_state was not updated in the last \'best\' epoch due to how loop terminates,
        # we might need to reconstruct/update it based on early_stopping.best_score and associated model state.
        # However, if the last `val_epoch_loss == early_stopping.best_score` condition ensures `best_model_state`
        # corresponds to `early_stopping.best_weights`, it should be fine.
        # A simpler approach: just save the model.state_dict() from the model object after early stopping.
        if best_model_state is None or best_model_state[\'val_loss\'] > early_stopping.best_score :
             # This case should ideally not happen if best_model_state is updated whenever early_stopping.best_score is.
             # Re-create best_model_state based on the restored model if necessary
             logging.warning("Reconstructing best_model_state after early stopping weight restoration.")
             # Need to find the epoch that corresponds to early_stopping.best_score or mark it.
             # This part can be tricky if history doesn\'t align perfectly.
             # For simplicity, let's assume the model object `model` has the best weights.
             best_model_state = {
                \'epoch\': -1, # Mark as "restored, epoch unknown from this point" or find it
                \'model_state_dict\': model.state_dict(), # Current model state is the best
                \'optimizer_state_dict\': optimizer.state_dict(), # This might not correspond to the best model epoch
                \'scheduler_state_dict\': scheduler.state_dict() if scheduler else None, # Same as optimizer
                \'val_loss\': early_stopping.best_score,
                # Acc and AUC would need to be re-evaluated or fetched from history for that best epoch
                \'val_accuracy\': history[\'val_accuracy\'][history[\'val_loss\'].index(early_stopping.best_score)] if early_stopping.best_score in history[\'val_loss\'] else float(\'nan\'),
                \'val_auc\': history[\'val_auc\'][history[\'val_loss\'].index(early_stopping.best_score)] if early_stopping.best_score in history[\'val_loss\'] else float(\'nan\'),
            }


    elif best_model_state is None and len(history[\'val_loss\']) > 0 : # Training finished, but no improvement or only one epoch
        logging.warning("No improvement in validation loss, or training was too short. Saving last model state.")
        last_epoch = len(history[\'train_loss\'])
        best_model_state = {
            \'epoch\': last_epoch,
            \'model_state_dict\': model.state_dict(),
            \'optimizer_state_dict\': optimizer.state_dict(),
            \'scheduler_state_dict\': scheduler.state_dict() if scheduler else None,
            \'val_loss\': history[\'val_loss\'][-1],
            \'val_accuracy\': history[\'val_accuracy\'][-1],
            \'val_auc\': history[\'val_auc\'][-1]
        }
    
    return history, best_model_state

# %% [markdown]
# ## 绘制训练过程曲线 for Nectin-4
# %%
def plot_training_history_nectin4(history, cfg: Config): 
    epochs_ran = len(history[\'train_loss\'])
    if epochs_ran == 0:
        logging.warning("No training history to plot for Nectin-4.")
        return
        
    epoch_ticks = range(1, epochs_ran + 1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Training and Validation Loss
    axs[0, 0].plot(epoch_ticks, history[\'train_loss\'], color=\'tab:red\', linestyle=\'-\', marker=\'o\', markersize=3, label=\'训练损失\')
    if \'val_loss\' in history and any(not (isinstance(x, float) and np.isnan(x)) for x in history[\'val_loss\']): # Check for non-NaNs
        axs[0, 0].plot(epoch_ticks, history[\'val_loss\'], color=\'tab:orange\', linestyle=\':\', marker=\'x\', markersize=3, label=\'验证损失\')
    axs[0, 0].set_xlabel(\'Epoch\')
    axs[0, 0].set_ylabel(\'Loss\')
    axs[0, 0].legend(loc=\'upper right\')
    axs[0, 0].grid(True, axis=\'y\', linestyle=\'--\', alpha=0.7)
    axs[0, 0].set_title(\'损失函数变化\')

    # Validation Accuracy
    if \'val_accuracy\' in history and any(not (isinstance(x, float) and np.isnan(x)) for x in history[\'val_accuracy\']):
        axs[0, 1].plot(epoch_ticks, history[\'val_accuracy\'], color=\'tab:blue\', linestyle=\'-\', marker=\'s\', markersize=3, label=\'验证准确率 (0-1)\')
    axs[0, 1].set_xlabel(\'Epoch\')
    axs[0, 1].set_ylabel(\'Accuracy\')
    axs[0, 1].legend(loc=\'lower right\')
    axs[0, 1].grid(True, axis=\'y\', linestyle=\'--\', alpha=0.7)
    axs[0, 1].set_ylim(0, 1.05) 
    axs[0, 1].set_title(\'验证准确率\')

    # Validation AUC
    if \'val_auc\' in history and any(not (isinstance(x, float) and np.isnan(x)) for x in history[\'val_auc\']): 
        axs[1, 0].plot(epoch_ticks, history[\'val_auc\'], color=\'tab:purple\', linestyle=\'--\', marker=\'^\', markersize=3, label=\'验证 AUC\')
    axs[1, 0].set_xlabel(\'Epoch\')
    axs[1, 0].set_ylabel(\'AUC\')
    axs[1, 0].legend(loc=\'lower right\')
    axs[1, 0].grid(True, axis=\'y\', linestyle=\'--\', alpha=0.7)
    axs[1, 0].set_ylim(0, 1.05)
    axs[1, 0].set_title(\'验证 AUC\')

    # Learning Rate
    if \'lr\' in history and len(history[\'lr\']) == epochs_ran:
        axs[1, 1].plot(epoch_ticks, history[\'lr\'], color=\'tab:green\', linestyle=\'--\', marker=\'.\', markersize=3, label=\'学习率\')
    elif \'lr\' in history:
        logging.warning(f\"LR history length ({len(history[\'lr\'])}) doesn\'t match epochs_ran ({epochs_ran}) for {cfg.label_column_name}. LR plot might be misaligned.\") 
        # Try to plot what we have
        axs[1,1].plot(range(1, len(history[\'lr\']) + 1), history[\'lr\'], color=\'tab:green\', linestyle=\'--\', marker=\'.\', markersize=3, label=\'学习率 (partial)\')


    axs[1, 1].set_xlabel(\'Epoch\')
    axs[1, 1].set_ylabel(\'Learning Rate\')
    axs[1, 1].legend(loc=\'upper right\')
    axs[1, 1].set_yscale(\'log\') 
    axs[1, 1].grid(True, axis=\'y\', linestyle=\'--\', alpha=0.7)
    axs[1, 1].set_title(\'学习率变化\')
    
    fig.tight_layout()  
    plt.suptitle(f\'{cfg.label_column_name} 二分类训练过程监控\', fontsize=16) 
    fig.subplots_adjust(top=0.92) # Make space for suptitle

    if epochs_ran < 20: # Heuristic for x-ticks
        for ax_row in axs:
            for ax in ax_row:
                ax.set_xticks(epoch_ticks)
                
    save_path = os.path.join(cfg.save_dir, f\"training_history_{cfg.model_filename_prefix}.png\")
    os.makedirs(cfg.save_dir, exist_ok=True) # Ensure save_dir exists
    plt.savefig(save_path) 
    logging.info(f\"Training history plot saved to {save_path}\")
    plt.show()
    plt.close(fig)


# %% [markdown]
# ## 最终评估 (ROC & P-R) for Nectin-4
# %%
def final_evaluation_plots_nectin4(cfg: Config, model, val_loader):
    logging.info(f\"\\nGenerating ROC and P-R curves for {cfg.label_column_name} on the validation set using the best model.\")
    
    model.eval() # Ensure model is in evaluation mode
    
    # Use validate_one_epoch to get necessary outputs
    _, _, _, all_val_labels_final_np, all_val_probs_final_np = validate_one_epoch(model, val_loader, criterion=None, device=cfg.device, cfg=cfg) # Criterion not needed for this eval part

    if all_val_labels_final_np.size == 0 or all_val_probs_final_np.size == 0:
        logging.warning(f\"Validation data for {cfg.label_column_name} is empty or probabilities could not be obtained. Skipping ROC/PR plots.\")
        return

    # Check if there are at least two classes in the validation labels for metric calculation
    if len(np.unique(all_val_labels_final_np)) >= 2:
        try:
            # ROC Curve (using probabilities for the positive class, i.e., class 1)
            val_auc_final = roc_auc_score(all_val_labels_final_np, all_val_probs_final_np[:, 1])
            logging.info(f\"Final Validation AUC for {cfg.label_column_name}: {val_auc_final:.4f}\")
            fpr, tpr, _ = roc_curve(all_val_labels_final_np, all_val_probs_final_np[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, lw=2, label=f\'ROC curve (AUC = {val_auc_final:.3f})\')
            plt.plot([0, 1], [0, 1], color=\'navy\', lw=2, linestyle=\'--\')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel(\'假阳性率 (False Positive Rate)\')
            plt.ylabel(\'真阳性率 (True Positive Rate)\')
            plt.title(f\'验证集ROC曲线 ({cfg.label_column_name})\')
            plt.legend(loc="lower right")
            plt.grid(True)
            roc_save_path = os.path.join(cfg.save_dir, f\'roc_curve_{cfg.model_filename_prefix}_val.png\')
            plt.savefig(roc_save_path)
            logging.info(f\"ROC curve saved to {roc_save_path}\")
            plt.show()
    plt.close()
    
            # Precision-Recall Curve
            val_ap_final = average_precision_score(all_val_labels_final_np, all_val_probs_final_np[:, 1])
            logging.info(f\"Final Validation Average Precision for {cfg.label_column_name}: {val_ap_final:.4f}\")
            precision, recall, _ = precision_recall_curve(all_val_labels_final_np, all_val_probs_final_np[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, lw=2, label=f\'P-R curve (AP = {val_ap_final:.3f})\')
            plt.xlabel(\'召回率 (Recall)\')
            plt.ylabel(\'精确率 (Precision)\')
            plt.title(f\'验证集P-R曲线 ({cfg.label_column_name})\')
            plt.legend(loc="best")
            plt.grid(True)
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            pr_save_path = os.path.join(cfg.save_dir, f\'pr_curve_{cfg.model_filename_prefix}_val.png\')
            plt.savefig(pr_save_path)
            logging.info(f\"P-R curve saved to {pr_save_path}\")
            plt.show()
    plt.close()
    
        except ValueError as e_val_curves:
            logging.error(f\"Final Validation ROC/PR calculation error for {cfg.label_column_name}: {e_val_curves}\")
        else:
        logging.warning(f\"Final Validation ROC/PR not computed for {cfg.label_column_name}: validation set does not contain enough distinct classes (needs at least 2).\")


# %% [markdown]
# ## 特征与标签相关性分析 (Adapted for Nectin-4)
# %% 
def get_embeddings_nectin4(model: Nectin4Classifier, dataloader, device, cfg: Config):
    model.eval()
    embeddings_list = []
    labels_list = []
    
    features_module = model.get_features_module()
    if features_module is None:
        logging.error(\"Cannot get features module from Nectin4Classifier. Skipping embedding extraction.\")
        return np.array([]), np.array([])

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=f\"Extracting embeddings for {cfg.label_column_name}\"):
            imgs = imgs.to(device)
            features = features_module(imgs) # Get features from the identified module
            # EfficientNet typically has an adaptive avg pool after features
            # If base_model itself is the full EfficientNet, its classifier[0] is avgpool
            if hasattr(model.base_model, \'avgpool\'): # Standard for EfficientNet and ResNet
                 pooled_features = model.base_model.avgpool(features)
            elif hasattr(model.base_model, \'classifier\') and isinstance(model.base_model.classifier, nn.Sequential) and isinstance(model.base_model.classifier[0], nn.AdaptiveAvgPool2d):
                 pooled_features = model.base_model.classifier[0](features)
            else: # Fallback to manual adaptive avg pool if structure is unclear
                 logging.warning(\"Could not find standard avgpool layer for embeddings. Using F.adaptive_avg_pool2d.\")
                 pooled_features = F.adaptive_avg_pool2d(features, (1,1))

                 embeddings = torch.flatten(pooled_features, 1)
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.cpu().numpy()) # labels are already 0/1 from Nectin4Dataset
    
    return np.concatenate(embeddings_list) if embeddings_list else np.array([]), \
           np.concatenate(labels_list) if labels_list else np.array([])


def calculate_mutual_information_nectin4(features_or_probs, labels, random_seed=SEED):
    if features_or_probs.ndim == 1: # If it's 1D (e.g. probs for one class), reshape for mutual_info_classif
        features_or_probs = features_or_probs.reshape(-1, 1)
    # mutual_info_classif handles multi-class targets directly if labels are multi-class.
    # For binary Nectin-4, labels are 0/1.
    # If features_or_probs are embeddings (N, D_embed), MI will be (D_embed,).
    # If features_or_probs are probabilities for class 1 (N,1), MI will be (1,).
    mi = mutual_info_classif(features_or_probs, labels, random_state=random_seed)
    return mi

def plot_tsne_visualization_nectin4(embeddings, labels, cfg: Config, title_suffix=\"\"):
    if len(embeddings) == 0:
        logging.warning(f\"Cannot run t-SNE for {cfg.label_column_name}: No embeddings provided.\")
        return
    logging.info(f\"Running t-SNE for {cfg.label_column_name}...\")
    perplexity_val = min(30, len(embeddings)-1 if len(embeddings) > 1 else 1) # Ensure perplexity is valid
    if perplexity_val <=0:
        logging.warning(f\"Perplexity for t-SNE is {perplexity_val}, which is invalid. Skipping t-SNE for {cfg.label_column_name}.\")
        return

    tsne = TSNE(n_components=2, random_state=cfg.random_seed, perplexity=perplexity_val, n_iter=1000, init=\'pca\', learning_rate=\'auto\')
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels) # Should be [0, 1]
    cmap = plt.colormaps.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(unique_labels)))

    for i, label_val in enumerate(unique_labels):
        idx = labels == label_val
        # Use class_names from config for legend
        class_display_name = cfg.class_names[label_val] if label_val < len(cfg.class_names) else f\"Class {label_val}\"
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], color=colors[i], label=class_display_name, alpha=0.7)
    
    plt.title(f\'t-SNE 可视化 ({cfg.label_column_name}{title_suffix})\')
    plt.xlabel(\'t-SNE Component 1\')
    plt.ylabel(\'t-SNE Component 2\')
    plt.legend()
    plt.grid(True)
    tsne_save_path = os.path.join(cfg.save_dir, f\"tsne_visualization_{cfg.model_filename_prefix}{title_suffix.replace(\' \', \'_\').lower()}.png\")
    plt.savefig(tsne_save_path)
    logging.info(f\"t-SNE plot saved to {tsne_save_path}\")
    plt.show()
    plt.close()

def simulate_data_cleaning_test_nectin4(model_probs_class1_np, original_labels_np, cfg: Config, num_samples_to_flip=100):
    """ Uses pre-computed probabilities for class 1."""
    logging.info(f\"\\nSimulating data cleaning test for {cfg.label_column_name} by randomly re-assigning {num_samples_to_flip} labels...\")
    
    if len(original_labels_np) < num_samples_to_flip:
        logging.warning(f\"Not enough samples ({len(original_labels_np)}) to re-assign {num_samples_to_flip} for {cfg.label_column_name}. Skipping simulation.\")
        return float(\'nan\')

    flipped_labels_np = original_labels_np.copy()
    indices_to_flip = np.random.choice(len(flipped_labels_np), num_samples_to_flip, replace=False)
    
    # Flip labels (0 to 1, 1 to 0 for binary)
    flipped_labels_np[indices_to_flip] = 1 - flipped_labels_np[indices_to_flip]

    if len(np.unique(flipped_labels_np)) < 2:
        logging.warning(f\"After re-assigning, less than 2 unique classes in simulated labels for {cfg.label_column_name}. AUC might be ill-defined.\")
        # return float(\'nan\') # Or proceed if roc_auc_score can handle it
        
    try:
        auc_after_cleaning = roc_auc_score(flipped_labels_np, model_probs_class1_np) # Use original model's probabilities for class 1
        logging.info(f\"Macro AUC after simulated cleaning for {cfg.label_column_name} ({num_samples_to_flip} labels re-assigned): {auc_after_cleaning:.4f}\")
        return auc_after_cleaning
    except ValueError as e:
        logging.error(f\"Error calculating Macro AUC after simulated cleaning for {cfg.label_column_name}: {e}\")
        return float(\'nan\')

def perform_permutation_test_nectin4(model_probs_class1_np, original_labels_np, cfg: Config, n_permutations=1000):
    """ Uses pre-computed probabilities for class 1."""
    logging.info(f\"\\nPerforming permutation test for {cfg.label_column_name} (AUC) with {n_permutations} permutations...\")
    
    if len(np.unique(original_labels_np)) < 2: # Binary case check
        logging.warning(f\"Original labels for {cfg.label_column_name} have less than 2 unique classes. Permutation test for AUC might be less reliable or fail.\")
        # return float(\'nan\') # Or proceed cautiously

    try:
        observed_auc = roc_auc_score(original_labels_np, model_probs_class1_np)
    except ValueError as e:
        logging.error(f\"Could not calculate observed AUC for {cfg.label_column_name}: {e}. Permutation test skipped.\")
        return float(\'nan\')
        
    logging.info(f\"Observed AUC for {cfg.label_column_name}: {observed_auc:.4f}\")
    
    permuted_aucs = []
    for i in tqdm(range(n_permutations), desc=f\"Permutation Test {cfg.label_column_name} (AUC)\"):
        permuted_labels = sklearn_shuffle(original_labels_np, random_state=cfg.random_seed + i)
        if len(np.unique(permuted_labels)) < 2: # Check if permutation resulted in single class
             permuted_aucs.append(0.5) # AUC for random chance in binary if one class only (or handle as error)
            continue
        try:
            auc_val = roc_auc_score(permuted_labels, model_probs_class1_np) # Use original model's probabilities for class 1
            permuted_aucs.append(auc_val)
        except ValueError:
             permuted_aucs.append(0.5) # Approx. random chance for binary AUC if error

    permuted_aucs = np.array(permuted_aucs)
    p_value = np.mean(permuted_aucs >= observed_auc)
    
    logging.info(f\"Permutation test for {cfg.label_column_name} (AUC): p-value = {p_value:.4f}\")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(permuted_aucs, bins=30, kde=True, label=\'Permuted AUCs\')
    plt.axvline(observed_auc, color=\'red\', linestyle=\'--\', lw=2, label=f\'Observed AUC ({observed_auc:.3f})\')
    plt.title(f\'Permutation Test Results for {cfg.label_column_name} (AUC)\')
    plt.xlabel(\'AUC Score\')
    plt.ylabel(\'Frequency\')
    plt.legend()
    perm_test_save_path = os.path.join(cfg.save_dir, f\"permutation_test_{cfg.model_filename_prefix}_auc.png\")
    plt.savefig(perm_test_save_path)
    logging.info(f\"Permutation test plot saved to {perm_test_save_path}\")
    plt.show()
    plt.close()
    return p_value

# %% [markdown]
# ## Grad-CAM 可视化 (Adapted for Nectin-4)
# %%
def visualize_grad_cam_nectin4(model: Nectin4Classifier, dataset: Nectin4Dataset, device, cfg: Config, num_images_per_target_class=2, target_classes_to_viz=None):
    
    features_module = model.get_features_module()
    if features_module is None:
        logging.error(\"Cannot get features module for Grad-CAM. Skipping.\")
        return

    # For EfficientNet, a common target is the last convolutional layer in the feature extractor
    # This might be model.get_features_module()[-1] or model.get_features_module()._conv_head
    # This needs to be robust. Let's try to find a suitable layer.
    target_layer_candidate = None
    if hasattr(features_module, \'_conv_head\'): # Common in some EfficientNet impl.
        target_layer_candidate = features_module._conv_head
    elif isinstance(features_module, nn.Sequential) and len(features_module) > 0:
        # Iterate backwards to find the last Conv2D containing module in features_module
        for layer in reversed(features_module):
            if isinstance(layer, nn.Conv2d):
                target_layer_candidate = layer
                break
            elif isinstance(layer, nn.Sequential) and len(layer) > 0: # E.g. last block in EfficientNet
                 # Try to get last conv from this sub-sequential block
                 for sub_layer in reversed(layer):
                     if isinstance(sub_layer, nn.Conv2d):
                         target_layer_candidate = sub_layer
                         break
                     # Look for common wrapper like Conv2dNormActivation
                     elif hasattr(sub_layer, \'0\') and isinstance(sub_layer[0], nn.Conv2d):
                         target_layer_candidate = sub_layer[0]
                         break
                 if target_layer_candidate: break 
    
    if target_layer_candidate is None:
        logging.warning(f\"Could not automatically determine a specific target Conv2D layer for Grad-CAM on {cfg.model_name}. Using the whole features_module. This might not be optimal.\")
        target_layers = [features_module] # Fallback to the whole feature extractor
    else:
        logging.info(f\"Using target layer for Grad-CAM: {target_layer_candidate}\")
        target_layers = [target_layer_candidate]

    cam_obj = GradCAM(model=model, target_layers=target_layers) # Removed use_cuda

    if not dataset or len(dataset) == 0:
        logging.warning(f\"Dataset for {cfg.label_column_name} Grad-CAM is empty.\") 
        return

    if target_classes_to_viz is None:
        target_classes_to_viz = list(range(cfg.num_classes)) # For Nectin-4 binary: [0, 1]
    
    images_shown_count = 0
    num_viz_rows = len(target_classes_to_viz)
    num_viz_cols = num_images_per_target_class 
    
    if num_viz_rows * num_viz_cols == 0:
        logging.warning(f\"No images or target classes specified for {cfg.label_column_name} Grad-CAM.\")
        return

    fig, axes = plt.subplots(num_viz_rows * 2, num_viz_cols, figsize=(num_viz_cols * 3, num_viz_rows * 6))
    # Handle a single image total (1 target class, 1 image)
    if num_viz_rows == 1 and num_viz_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]]).reshape(2,1) # Make it 2D for consistent indexing, ensure it's (2,1) for iteration
    elif num_viz_cols == 1 and num_viz_rows > 1: 
         axes = axes.reshape(num_viz_rows * 2, 1)
    elif num_viz_rows == 1 and num_viz_cols > 1:
         axes = axes.reshape(2, num_viz_cols)
    # No change if already 2D (num_viz_rows > 1 and num_viz_cols > 1)
    
    if num_viz_rows * num_viz_cols == 0 : # Should have been caught earlier
        logging.warning(\"Grad-CAM subplot calculation resulted in zero rows/cols.\")
        plt.close(fig)
        return

    # Select images for each target class to visualize
    images_to_process_indices = []
    for target_cls_idx in target_classes_to_viz: # target_cls_idx is 0 or 1
        # Find images *actually belonging* to this class for more representative CAMs (optional)
        # Or just pick random images from dataset.
        # For now, picking random images from the dataset for each slot.
        
        # Find indices of images in the dataset that actually belong to target_cls_idx
        # This requires dataset.df to have the mapped labels. Nectin4Dataset's __getitem__ uses mapped labels.
        # We need to access the dataframe used by the dataset.
        class_specific_indices = dataset.df[dataset.df[cfg.mapped_label_column_name] == target_cls_idx].index.tolist()

        if not class_specific_indices:
            logging.warning(f\"No images found for class {cfg.class_names[target_cls_idx]} in the dataset for Grad-CAM. Picking random images instead.\")
            # Fallback: pick random images from the whole dataset if no images for this specific class.
            if len(dataset) >= num_images_per_target_class:
                indices_for_this_class_slot = np.random.choice(len(dataset), num_images_per_target_class, replace=False)
            else: # Not enough images in whole dataset
                indices_for_this_class_slot = np.random.choice(len(dataset), num_images_per_target_class, replace=True) if len(dataset) > 0 else []
        elif len(class_specific_indices) < num_images_per_target_class:
            logging.warning(f\"Not enough images for class {cfg.class_names[target_cls_idx]} ({len(class_specific_indices)} found) for Grad-CAM. Using all available and allowing replacement if needed.\")
            indices_for_this_class_slot = np.random.choice(class_specific_indices, num_images_per_target_class, replace=True)
    else:
            indices_for_this_class_slot = np.random.choice(class_specific_indices, num_images_per_target_class, replace=False)
        
        images_to_process_indices.append(indices_for_this_class_slot)


    for r_idx, target_cls_val in enumerate(target_classes_to_viz): # target_cls_val is 0 or 1
        img_indices_for_current_target_cls_slot = images_to_process_indices[r_idx]
        
        if not img_indices_for_current_target_cls_slot.size : # Check if array is empty
             logging.warning(f\"No images selected for target class {cfg.class_names[target_cls_val]} Grad-CAM slot. Skipping.\")
             continue

        for c_idx_local, img_idx_in_dataset in enumerate(img_indices_for_current_target_cls_slot):
            img_tensor, true_label_scalar = dataset[img_idx_in_dataset] 
            true_label_val = true_label_scalar.item() if isinstance(true_label_scalar, torch.Tensor) else int(true_label_scalar)
            
            inv_normalize = transforms.Normalize(
                mean=[-m/s for m, s in zip(cfg.normalize_mean, cfg.normalize_std)],
                std=[1/s for s in cfg.normalize_std]
            )
            rgb_img_denorm = inv_normalize(img_tensor.clone()).permute(1, 2, 0).cpu().numpy()
            rgb_img_denorm = np.clip(rgb_img_denorm, 0, 1)

            input_tensor_unsqueeze = img_tensor.unsqueeze(0).to(device)
            # GradCAM targets are the class indices (0, 1 for Nectin-4 binary)
            cam_targets_for_pytorchcam = [ClassifierOutputTarget(target_cls_val)] 
            
            grayscale_cam_batch = cam_obj(input_tensor=input_tensor_unsqueeze, targets=cam_targets_for_pytorchcam)
            if grayscale_cam_batch is not None and grayscale_cam_batch.shape[0] > 0:
                grayscale_cam = grayscale_cam_batch[0, :] 
            else:
                logging.warning(f\"Grad-CAM returned None or empty for image index {img_idx_in_dataset}, target class {cfg.class_names[target_cls_val]}. Skipping this image.\") 
                # Get current axes to turn off if plot is malformed
                ax_orig = axes[r_idx * 2, c_idx_local] if axes.ndim == 2 else axes[r_idx*2] if num_viz_cols == 1 else axes[0]
                ax_cam  = axes[r_idx * 2 + 1, c_idx_local] if axes.ndim == 2 else axes[r_idx*2+1] if num_viz_cols == 1 else axes[1]
                ax_orig.axis(\'off\'); ax_cam.axis(\'off\')
                continue
            
            cam_image_overlay = show_cam_on_image(rgb_img_denorm, grayscale_cam, use_rgb=True)
            original_img_for_grid_display = (rgb_img_denorm * 255).astype(np.uint8)
            
            title_str = f\"\"\"True: {cfg.class_names[true_label_val]}
CAM for: {cfg.class_names[target_cls_val]}\"\"\" 

            # Handle axes indexing based on subplot layout
            current_ax_orig = None
            current_ax_cam = None
            if num_viz_rows == 1 and num_viz_cols == 1: # Special case: 1 image total
                current_ax_orig = axes[0,0] if axes.ndim ==2 else axes[0]
                current_ax_cam  = axes[1,0] if axes.ndim ==2 else axes[1]
            elif num_viz_cols == 1 : # Single column of images
                 current_ax_orig = axes[r_idx * 2, 0]
                 current_ax_cam  = axes[r_idx * 2 + 1, 0]
            elif num_viz_rows == 1: # Single row of images
                 current_ax_orig = axes[0, c_idx_local]
                 current_ax_cam  = axes[1, c_idx_local]
            else: # Grid
                 current_ax_orig = axes[r_idx * 2, c_idx_local] 
                 current_ax_cam = axes[r_idx * 2 + 1, c_idx_local]

            current_ax_orig.imshow(original_img_for_grid_display)
            current_ax_orig.set_title(title_str, fontsize=8)
            current_ax_orig.axis(\'off\')
            current_ax_cam.imshow(cam_image_overlay)
            current_ax_cam.axis(\'off\')
            images_shown_count +=1

    if images_shown_count == 0:
        logging.warning(f\"No {cfg.label_column_name} CAM images were generated successfully.\") 
        if num_viz_rows * num_viz_cols > 0 : plt.close(fig) # Close figure if it was created but nothing shown
        return

    fig.suptitle(f\"Grad-CAM for {cfg.label_column_name} Model (Targeting Various Classes)\", fontsize=12) 
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    grad_cam_save_path = os.path.join(cfg.save_dir, f\'grad_cam_{cfg.model_filename_prefix}_binary.png\')
    plt.savefig(grad_cam_save_path)
    logging.info(f\"Grad-CAM grid for {cfg.label_column_name} saved to {grad_cam_save_path}\") 
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## 主函数 (Adapted for Nectin-4)
# %%
def setup_logging(cfg: Config):
    """Sets up logging for the training session."""
    log_dir = os.path.join(cfg.save_dir, \'logs\')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime(\'%Y%m%d_%H%M%S\')
    log_file = os.path.join(log_dir, f\'training_{cfg.model_filename_prefix}_{timestamp}.log\')
    
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format=\'%(asctime)s [%(levelname)s] - %(message)s\',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) # Also print to console
        ]
    )
    logging.info(f\"Logging initialized. Log file: {log_file}\")
    logging.info(f\"Using device: {cfg.device}\")
    logging.info(f\"Nectin-4 Configuration:\\n{cfg}\")


def main_nectin4():
    cfg = Config() # Initialize configuration
    setup_logging(cfg) # Setup logger using config
    
    try:
        df_train, df_val = load_and_preprocess_nectin4_data(cfg)

        if df_train.empty:
            logging.error(\"Training dataframe is empty after preprocessing and splitting. Cannot proceed.\")
            return None, None 
        
        train_loader, val_loader, train_ds, val_ds = get_nectin4_dataloaders(cfg, df_train, df_val)

        if not train_loader:
            logging.error(\"Train loader is None. Cannot proceed with training.\")
            return None, None

        logging.info(\"Initializing Nectin-4 model...\")
        model = Nectin4Classifier(cfg).to(cfg.device)
        
        # Calculate class weights for FocalLoss based on training data distribution
        if not df_train.empty and cfg.mapped_label_column_name in df_train.columns:
            counts = df_train[cfg.mapped_label_column_name].value_counts().sort_index()
            if len(counts) == cfg.num_classes : # Expecting 2 classes for binary
                # Inverse frequency weighting for FocalLoss (passed to F.cross_entropy via class_weights)
                # Alpha in FocalLoss itself can provide additional per-class weighting.
                # Here, we calculate weights for the `weight` parameter of cross_entropy.
                # Alpha in FocalLoss class is separate. Setting config.focal_alpha here for clarity.
                
                # Weights for CE component of FocalLoss
                ce_weights_list = [1.0 / counts.get(i, 1e-6) for i in range(cfg.num_classes)]
                total_ce_weight = sum(ce_weights_list)
                normalized_ce_weights = [w / total_ce_weight for w in ce_weights_list]
                class_weights_for_criterion = torch.tensor(normalized_ce_weights, dtype=torch.float).to(cfg.device)
                logging.info(f\"Calculated class weights for FocalLoss (CE component): {class_weights_for_criterion.tolist()}\")

                # Optional: Set config.focal_alpha based on imbalance if desired (e.g., alpha for positive class)
                # For binary: alpha for class 1 (positive) could be counts[0]/(counts[0]+counts[1])
                # and for class 0 (negative) counts[1]/(counts[0]+counts[1]) - or a fixed value like 0.25/0.75.
                # If cfg.focal_alpha is None, the FocalLoss class will handle it (e.g. apply single float alpha to all, or expect list).
                # Let's try to set a list for alpha based on imbalance, e.g. [alpha_for_0, alpha_for_1]
                # alpha_0 = counts.get(1, 1e-6) / (counts.get(0, 1e-6) + counts.get(1, 1e-6)) # Alpha for class 0 related to proportion of class 1
                # alpha_1 = counts.get(0, 1e-6) / (counts.get(0, 1e-6) + counts.get(1, 1e-6)) # Alpha for class 1 related to proportion of class 0
                # cfg.focal_alpha = [alpha_0, alpha_1]
                # logging.info(f\"Calculated FocalLoss alpha param: {cfg.focal_alpha}\")
                # For simplicity, if cfg.focal_alpha is already set (e.g. to 0.25), FocalLoss class will use it.
                # If it's None, and we pass a single float during FocalLoss init, it will be used.
                # For now, rely on cfg.focal_alpha if set, or default single value if not, or pass None.
                # Using the weights in CE is often sufficient.

            else: # Not all classes present or unexpected counts
                logging.warning(f\"Not all {cfg.num_classes} classes present or unexpected counts in Nectin-4 training data ({len(counts)} found). Using None for FocalLoss CE class_weights.\")
                class_weights_for_criterion = None
                # cfg.focal_alpha = None # Reset if calculation is not robust
        else:
            logging.warning(\"Could not calculate class counts for Nectin-4 FocalLoss CE class_weights. Using None.\")
            class_weights_for_criterion = None
            # cfg.focal_alpha = None


        criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, num_classes=cfg.num_classes, class_weights=class_weights_for_criterion)
        optimizer = cfg.get_optimizer(model.parameters())
        scheduler = cfg.get_scheduler(optimizer)
        
        logging.info(\"Starting Nectin-4 model training...\")
        history, best_model_state = train_nectin4_model(cfg, model, train_loader, val_loader, criterion, optimizer, scheduler)
        
        if best_model_state is None:
            logging.error(\"Training did not produce a best_model_state. Exiting.\")
            return None, None # Indicate failure

        # Save training state (config, best model, history)
        # Construct a directory name for this run
        run_timestamp = datetime.now().strftime(\'%Y%m%d_%H%M%S\')
        experiment_save_dir = os.path.join(cfg.save_dir, f\"{cfg.model_filename_prefix}_{run_timestamp}\")
        os.makedirs(experiment_save_dir, exist_ok=True)
        
        cfg.save_config(os.path.join(experiment_save_dir, \'config.json\')) # Save runtime config
        
        if best_model_state and \'model_state_dict\' in best_model_state:
            torch.save(best_model_state, os.path.join(experiment_save_dir, f\"{cfg.model_filename_prefix}_best.pth\"))
            logging.info(f\"Best model state saved to {experiment_save_dir}\")
        else:
            logging.warning(\"No best_model_state dictionary or model_state_dict found to save.\")

        serializable_history = {k: [None if isinstance(x, float) and np.isnan(x) else x for x in v] if isinstance(v, list) else v for k,v in history.items()}
        with open(os.path.join(experiment_save_dir, \'training_history.json\'), \'w\', encoding=\'utf-8\') as f:
            json.dump(serializable_history, f, indent=4, ensure_ascii=False)
        logging.info(f\"Training history saved to {experiment_save_dir}\")

        # Plot training history (already done inside train_nectin4_model, but ensure save dir is used by plot func)
        # Re-plot with final save dir for consistency if plot function doesn't use experiment_save_dir
        # plot_training_history_nectin4(history, cfg) # cfg.save_dir is used by the plot function

        # Load the best model for final evaluation
        if best_model_state and \'model_state_dict\' in best_model_state:
            model.load_state_dict(best_model_state[\'model_state_dict\'])
            logging.info(f\"Loaded best Nectin-4 model (Epoch: {best_model_state.get(\'epoch\', \'N/A\')}, Val Loss: {best_model_state.get(\'val_loss\', \'N/A\'):.4f}) for final evaluation.\")
        else:
            logging.warning(\"No best model state found after training. Evaluation will use the model's last state.\")
        
        # Final evaluation plots (ROC, PR)
        if val_loader:
            final_evaluation_plots_nectin4(cfg, model, val_loader)
        else:
            logging.warning(\"Validation loader is None. Skipping final evaluation plots.\")
        
        # --- Execute Extended Feature-Label Relevance Analysis & Grad-CAM ---
        if val_loader and val_ds and len(val_ds) > 0: # Ensure val_loader and val_ds are valid
            logging.info(f\"\\n--- Starting Extended Nectin-4 Analysis ---\")
            
            # Get final validation probabilities (for class 1) and true labels
            # This re-runs validation pass, could be optimized if validate_one_epoch returned these directly from best epoch
            _, _, _, final_val_labels, final_val_probs_all_classes = validate_one_epoch(model, val_loader, criterion, cfg.device, cfg)
            final_val_probs_class1 = final_val_probs_all_classes[:, 1] if final_val_probs_all_classes.ndim == 2 and final_val_probs_all_classes.shape[1] == 2 else np.array([])


            if final_val_labels.size > 0 and final_val_probs_class1.size == final_val_labels.size:
                val_embeddings, emb_labels = get_embeddings_nectin4(model, val_loader, cfg.device, cfg)

                if val_embeddings.size > 0 and emb_labels.size > 0 and len(val_embeddings) == len(emb_labels):
                    # Mutual Information: probs vs labels
                    mi_scores_probs = calculate_mutual_information_nectin4(final_val_probs_class1, final_val_labels, random_seed=cfg.random_seed)
                    logging.info(f\"Mutual Information (Class 1 Probs vs Labels) for {cfg.label_column_name}: {mi_scores_probs[0]:.4f}\")

                    # Mutual Information: embeddings vs labels
                    mi_scores_embeddings = calculate_mutual_information_nectin4(val_embeddings, emb_labels, random_seed=cfg.random_seed)
                    logging.info(f\"Mean Mutual Information (Embeddings vs Labels) for {cfg.label_column_name}: {np.mean(mi_scores_embeddings):.4f}\")

                    plot_tsne_visualization_nectin4(val_embeddings, emb_labels, cfg)

                    num_samples_to_flip = min(max(1, len(final_val_labels) // 5), 100) 
                    simulate_data_cleaning_test_nectin4(final_val_probs_class1, final_val_labels, cfg, num_samples_to_flip=num_samples_to_flip)
                    
                    perform_permutation_test_nectin4(final_val_probs_class1, final_val_labels, cfg, n_permutations=100) # Reduced for speed
            else:
                    logging.warning(\"Could not extract embeddings or labels for Nectin-4 extended analysis. Skipping some parts.\")
        else:
                logging.warning(\"Could not get final validation probabilities or labels for Nectin-4 extended analysis. Skipping.\")

            logging.info(f\"\\nVisualizing Grad-CAM for Nectin-4 model\")
            visualize_grad_cam_nectin4(model, dataset=val_ds, device=cfg.device, cfg=cfg, 
                                       num_images_per_target_class=2, target_classes_to_viz=[0,1]) # Nectin-4 binary
        else:
            logging.warning(\"Skipping Nectin-4 extended analysis and Grad-CAM: Validation data not available or empty.\")

        return model, history, best_model_state
        
    except Exception as e:
        logging.error(f\"Nectin-4 classifier training/evaluation failed: {str(e)}\", exc_info=True)
        sys.exit(1) # Exit for errors in main execution flow
    
if __name__ == "__main__":
    # try:
    model_result, history_result, best_state_result = main_nectin4()
    if model_result is None:
        logging.error(\"Nectin-4 main function did not complete successfully. Exiting.\")
        # sys.exit(1) # Already handled in main_nectin4
    # except Exception as e:
    #     logging.error(f\"Program execution failed (Nectin-4): {e}\", exc_info=True)
    #     sys.exit(1)
# %%


</rewritten_file> 