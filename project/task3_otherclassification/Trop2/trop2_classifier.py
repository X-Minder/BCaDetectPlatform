# %% [markdown]
# # Trop2分类模型训练与评估
# 这个脚本实现了一个基于EfficientNet-B0的Trop2表达水平分类模型。 (将修改为与grade_model.py对齐的结构)

# %% [markdown]
# ## 导入必要的库

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights # Ensure efficientnet_b0 is used if model_name is 'efficientnet_b0'
import os
import logging # Will be simplified
from PIL import Image
import numpy as np
import random # Add this import
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from torch.cuda.amp import autocast, GradScaler # autocast and GradScaler can be added if mixed precision is desired later
# from dataclasses import dataclass # Config class will be simplified
from typing import List, Tuple, Optional
import sys
from datetime import datetime
import json
import torch.nn.functional as F

# Imports for new analysis (from grade_model.py)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.feature_selection import mutual_info_classif # For MI with continuous features if needed, or use sklearn.metrics.mutual_info_score for discrete
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score # For discrete labels/predictions
from sklearn.manifold import TSNE
from sklearn.utils import shuffle as sklearn_shuffle

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED) # Also set Python's random seed

# 配置matplotlib中文字体
def setup_chinese_font():
    """设置matplotlib中文字体"""
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

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 全局常量和配置

# %%
# Data related constants
LABEL_COLUMN_NAME = 'TROP2'  # Original label column in CSV (e.g., 1, 2, 3)
MAPPED_LABEL_COLUMN_NAME = 'TROP2_mapped' # Mapped label column (e.g., 0, 1, 2)
PATIENT_ID_COLUMN = 'PATIENT_ID'
FILE_NAME_COLUMN = 'FILE_NAME'
IMAGE_DIR = "Trop2/dataset/images" # Path relative to workspace root
LABEL_FILE_PATH = "Trop2/dataset/label.csv" # Path relative to workspace root

# Model and training related constants
NUM_CLASSES = 3  # Trop2 has 3 classes (0, 1, 2 after mapping)
MODEL_NAME = "efficientnet_b0" # Example, ensure this matches desired model
BATCH_SIZE = 16
NUM_EPOCHS = 70 # Example from grade_model
LEARNING_RATE = 1e-5 # Example from grade_model
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3 # Example from Trop2Config
EARLY_STOPPING_PATIENCE = 15 # Example from grade_model
FOCAL_GAMMA = 2.0

# Image processing
IMAGE_SIZE = (224, 224) # Aligning with grade_model's typical EfficientNet-B0 input size
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

class Config:
    def __init__(self):
        # 数据相关
        self.label_file = LABEL_FILE_PATH
        self.image_dir = IMAGE_DIR
        self.num_classes = NUM_CLASSES
        self.class_names = [f"Trop2-{i}" for i in range(NUM_CLASSES)]
        
        # 模型相关
        self.model_name = MODEL_NAME
        self.pretrained = True
        self.dropout_rate = DROPOUT_RATE
        
        # 训练相关
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = BATCH_SIZE
        self.num_epochs = NUM_EPOCHS
        self.learning_rate = LEARNING_RATE
        self.weight_decay = WEIGHT_DECAY
        self.early_stopping_patience = EARLY_STOPPING_PATIENCE
        self.val_size_from_trainval = 0.2
        self.random_seed = 42
        self.num_workers = 0
        
        # 优化器相关
        self.optimizer_name = 'AdamW'
        self.scheduler_name = 'ReduceLROnPlateau'
        self.min_lr = 1e-7
        self.warmup_epochs = 5
        self.gradient_clip_val = 1.0
        
        # 损失函数相关
        self.focal_alpha = None  # 将根据类别分布计算
        self.focal_gamma = FOCAL_GAMMA
        
        # 图像处理相关
        self.image_size = IMAGE_SIZE
        self.normalize_mean = NORMALIZE_MEAN
        self.normalize_std = NORMALIZE_STD
        
        # 保存相关
        self.save_dir = "checkpoints"
        
        # 设置transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(self.normalize_mean, self.normalize_std)
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(self.normalize_mean, self.normalize_std)
        ])
    
    def get_optimizer(self, model_parameters):
        if self.optimizer_name.lower() == 'adamw':
            return optim.AdamW(model_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
    
    def get_scheduler(self, optimizer):
        if self.scheduler_name.lower() == 'reducelronplateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5,
                min_lr=self.min_lr, verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
    
    def save_config(self, save_path):
        """保存配置到JSON文件"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not callable(v) and not isinstance(v, (torch.device, transforms.Compose))}
        config_dict['device'] = str(self.device)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, config_path):
        """从JSON文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = cls()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return config
    
    def __str__(self):
        """返回配置的字符串表示"""
        config_str = "模型配置:\n"
        for k, v in self.__dict__.items():
            if not k.startswith('_') and not callable(v) and not isinstance(v, (torch.device, transforms.Compose)):
                config_str += f"{k}: {v}\n"
        return config_str

# 创建全局配置实例
config = Config()

# %% [markdown]
# ## 定义数据集类

# %%
class Trop2Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        """
        初始化Trop2数据集
        
        Args:
            df (pd.DataFrame): 包含图像文件名和标签的数据框
            image_dir (str): 图像文件目录路径
            transform (callable, optional): 图像转换函数
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        
        # 预检查图像路径
        if not df.empty:
            self.df['image_path'] = self.df[FILE_NAME_COLUMN].apply(lambda x: os.path.join(self.image_dir, x))
            self.df = self.df[self.df['image_path'].apply(os.path.exists)]
            if len(self.df) == 0:
                print("警告: Trop2Dataset初始化后没有有效的图像路径。")

    def __len__(self):
        """返回数据集大小"""
        return len(self.df)

    def __getitem__(self, idx):
        """获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (image, label) 图像张量和标签
        """
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row[FILE_NAME_COLUMN])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"读取图像出错 {img_path}: {str(e)}.")
            # 返回占位图像和标签
            return torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1])), torch.tensor(-1, dtype=torch.long)
            
        if self.transform:
            image = self.transform(image)
            
        label = int(row[MAPPED_LABEL_COLUMN_NAME])
        return image, label

# %% [markdown]
# ## 数据加载与预处理

# %%
print(f"1. Reading label file: {LABEL_FILE_PATH}")
label_df_raw = pd.read_csv(LABEL_FILE_PATH)
print(f"   Initial rows loaded: {len(label_df_raw)}")

# Drop NA from critical columns
label_df = label_df_raw.dropna(subset=[LABEL_COLUMN_NAME, FILE_NAME_COLUMN, PATIENT_ID_COLUMN]).copy()
print(f"2. Rows after dropping NA from key columns ('{LABEL_COLUMN_NAME}', '{FILE_NAME_COLUMN}', '{PATIENT_ID_COLUMN}'): {len(label_df)}")

if len(label_df) == 0:
    print(f"ERROR: All rows were dropped after initial NA check for {LABEL_COLUMN_NAME}. Please check CSV.")
    # sys.exit("Label data processing failed.") # Exit if no data
else:
    print(f"   Unique values in '{LABEL_COLUMN_NAME}' before mapping: {label_df[LABEL_COLUMN_NAME].astype(str).unique()}")

    # Mapping Trop2 labels (1, 2, 3) to (0, 1, 2)
    def map_trop2_to_numeric(trop2_status):
        if pd.isna(trop2_status):
            return np.nan
        status_str = str(trop2_status).strip()
        if not status_str:
            return np.nan
        try:
            status_val_float = float(status_str)
            status_val = int(status_val_float)
            if status_val != status_val_float:
                print(f"Warning: Trop2 status '{trop2_status}' (str: '{status_str}') has non-zero decimal. Mapping to NaN.")
                return np.nan
            if status_val in [1, 2, 3]: # Original labels
                return status_val - 1 # Map to 0, 1, 2
            else:
                print(f"Warning: Unexpected Trop2 status '{trop2_status}' (parsed as {status_val}). Not 1, 2, or 3. Mapping to NaN.")
                return np.nan
        except ValueError:
            print(f"Warning: Could not convert Trop2 status '{trop2_status}' (str: '{status_str}') to number. Mapping to NaN.")
            return np.nan

    label_df[MAPPED_LABEL_COLUMN_NAME] = label_df[LABEL_COLUMN_NAME].apply(map_trop2_to_numeric)
    print(f"3. Rows after applying 'map_trop2_to_numeric' (before dropping NA from '{MAPPED_LABEL_COLUMN_NAME}'): {len(label_df)}")
    print(f"   Unique values in '{MAPPED_LABEL_COLUMN_NAME}' after mapping (before dropping NA): {label_df[MAPPED_LABEL_COLUMN_NAME].unique()}")

    label_df = label_df.dropna(subset=[MAPPED_LABEL_COLUMN_NAME]).copy()
    print(f"4. Rows after dropping NA from '{MAPPED_LABEL_COLUMN_NAME}': {len(label_df)}")
    
    if len(label_df) > 0:
        label_df[MAPPED_LABEL_COLUMN_NAME] = label_df[MAPPED_LABEL_COLUMN_NAME].astype(int)
        print(f"   Final unique mapped values in '{MAPPED_LABEL_COLUMN_NAME}': {label_df[MAPPED_LABEL_COLUMN_NAME].unique()}")
        
        # Check image file existence (moved here for clarity)
        label_df['image_path_check'] = label_df[FILE_NAME_COLUMN].apply(lambda x: os.path.join(IMAGE_DIR, x))
        missing_images_df = label_df[~label_df['image_path_check'].apply(os.path.exists)]
        if not missing_images_df.empty:
            print(f"Warning: {len(missing_images_df)} image files listed in CSV do not exist in {IMAGE_DIR}. These rows will be dropped.")
            print(f"Missing files examples: {missing_images_df[FILE_NAME_COLUMN].head().tolist()}")
            label_df = label_df[label_df['image_path_check'].apply(os.path.exists)].copy()
            print(f"   Rows after checking image file existence: {len(label_df)}")
        label_df = label_df.drop(columns=['image_path_check'])

    else:
        print(f"ERROR: All rows were dropped after mapping {LABEL_COLUMN_NAME}. Check mapping logic and original '{LABEL_COLUMN_NAME}' values.")
        # sys.exit("Label data processing failed.")

if len(label_df) > 0:
    # Data splitting (aligned with grade_model.py: train and val only)
    df_trainval_full = label_df.copy()
    
    gss_val_split = GroupShuffleSplit(n_splits=1, test_size=config.val_size_from_trainval, random_state=SEED)

    if len(df_trainval_full[PATIENT_ID_COLUMN].unique()) > 1 and len(df_trainval_full) > 1:
        try:
            train_idx, val_idx = next(gss_val_split.split(df_trainval_full, groups=df_trainval_full[PATIENT_ID_COLUMN]))
            df_train = df_trainval_full.iloc[train_idx].copy()
            df_val   = df_trainval_full.iloc[val_idx].copy()
            print(f"Successfully used GroupShuffleSplit for train/validation sets for {LABEL_COLUMN_NAME}.")
        except ValueError as e:
            print(f"Warning: GroupShuffleSplit for {LABEL_COLUMN_NAME} train/validation failed: {e}. Falling back to random stratified split.")
            stratify_col = df_trainval_full[MAPPED_LABEL_COLUMN_NAME] if df_trainval_full[MAPPED_LABEL_COLUMN_NAME].nunique() > 1 else None
            df_train, df_val = train_test_split(df_trainval_full, test_size=config.val_size_from_trainval, random_state=SEED, stratify=stratify_col)
    elif len(df_trainval_full) > 0:
        print(f"Warning: Not enough unique patient groups or samples for GroupShuffleSplit for {LABEL_COLUMN_NAME}. Using random stratified split.")
        if len(df_trainval_full) > 1:
            stratify_col = df_trainval_full[MAPPED_LABEL_COLUMN_NAME] if df_trainval_full[MAPPED_LABEL_COLUMN_NAME].nunique() > 1 else None
            df_train, df_val = train_test_split(df_trainval_full, test_size=config.val_size_from_trainval, random_state=SEED, stratify=stratify_col)
        else:
            df_train = df_trainval_full.copy()
            df_val = pd.DataFrame(columns=df_trainval_full.columns)
            print(f"Warning: Only one sample in {LABEL_COLUMN_NAME} data. Train set has 1 sample, validation set is empty.")
    else: # Should not happen if previous checks were correct
        df_train = pd.DataFrame(columns=label_df.columns)
        df_val = pd.DataFrame(columns=label_df.columns)
        print(f"ERROR: {LABEL_COLUMN_NAME} data is empty before splitting. Train and Val sets are empty.")


    print(f"\nDataset sizes and class distributions ({LABEL_COLUMN_NAME} - Mapped Labels 0,1,2):")
    for name, df_subset in [("Train", df_train), ("Val", df_val)]:
        if not df_subset.empty:
            print(f"  {name:<8}: {len(df_subset):>4} images, Patients: {df_subset[PATIENT_ID_COLUMN].nunique():>2}")
            distribution_info_series = df_subset[MAPPED_LABEL_COLUMN_NAME].value_counts(normalize=True).sort_index()
            distribution_info_str = '\n'.join([f"    Class {idx}: {val:.4f}" for idx, val in distribution_info_series.items()])
            print(f"    Class distribution ({MAPPED_LABEL_COLUMN_NAME}, normalized):\n{distribution_info_str}")
            print(f"    Unique patients per class ({MAPPED_LABEL_COLUMN_NAME}):")
            if MAPPED_LABEL_COLUMN_NAME in df_subset.columns and PATIENT_ID_COLUMN in df_subset.columns:
                for class_label_mapped in sorted(df_subset[MAPPED_LABEL_COLUMN_NAME].unique()):
                    num_patients_in_class = df_subset[df_subset[MAPPED_LABEL_COLUMN_NAME] == class_label_mapped][PATIENT_ID_COLUMN].nunique()
                    print(f"      Class {class_label_mapped} (Original {class_label_mapped+1}): {num_patients_in_class} patients")
            else:
                print("      Could not calculate unique patients per class (column missing).")
        else:
            print(f"  {name:<8}: Empty")
    print("\n")
else:
    print(f"Critical Error: {LABEL_COLUMN_NAME} label_df is empty after preprocessing. Cannot proceed.")
    df_train, df_val = pd.DataFrame(), pd.DataFrame()


# DataLoaders
train_ds = Trop2Dataset(df_train, IMAGE_DIR, transform=config.train_transform) if not df_train.empty else None
val_ds = Trop2Dataset(df_val, IMAGE_DIR, transform=config.val_transform) if not df_val.empty else None

train_loader_args_dict = {'batch_size': config.batch_size, 'num_workers': config.num_workers, 'pin_memory': True}
if train_ds:
    counts_train_mapped = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    if len(counts_train_mapped) > 0 and len(counts_train_mapped) <= NUM_CLASSES:
        # Sampler weights for mapped classes (0, 1, 2)
        class_sample_weights_values = [0.0] * NUM_CLASSES
        for i in range(NUM_CLASSES): # Iterate 0, 1, 2
            class_sample_weights_values[i] = 1.0 / counts_train_mapped.get(i, 1e-6) # Use mapped class index i
        
        sample_weights_train = [class_sample_weights_values[label] for label in df_train[MAPPED_LABEL_COLUMN_NAME]]
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights_train), num_samples=len(sample_weights_train), replacement=True)
        
        weights_str_sampler = ", ".join([f"Class {i} (Orig {i+1}): {w:.4f}" for i, w in enumerate(class_sample_weights_values)])
        print(f"Sampler weights for {LABEL_COLUMN_NAME} mapped classes (0,1,2): {weights_str_sampler}")
        train_loader_args_dict['sampler'] = sampler
        train_loader_args_dict['shuffle'] = False # Sampler handles shuffling
    else:
        print(f"Warning: Training data for {LABEL_COLUMN_NAME} has insufficient or unexpected mapped class counts for sampler. Using standard DataLoader with shuffle=True.")
        train_loader_args_dict['shuffle'] = True
else: # train_ds is None or empty
    print(f"Warning: df_train for {LABEL_COLUMN_NAME} is empty. Train loader will be None.")
    train_loader_args_dict['shuffle'] = False # Not relevant if loader is None

train_loader = DataLoader(train_ds, **train_loader_args_dict) if train_ds else None
val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True) if val_ds else None

# %% [markdown]
# ## 定义模型、损失函数和早停机制

# %%
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean', num_classes=NUM_CLASSES, class_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.num_classes = num_classes
        self.class_weights = class_weights
        
        if isinstance(alpha, (float, int)):
            if num_classes > 1 and not isinstance(alpha, list):
                print(f"Warning: Single float alpha ({alpha}) provided for FocalLoss with {num_classes} classes. This will be applied as a weight for each class.")
                self.alpha = torch.tensor([alpha] * num_classes)

        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
            if len(self.alpha) != num_classes:
                print(f"Warning: Alpha list length {len(self.alpha)} does not match num_classes={num_classes}. Adjusting.")
                if len(self.alpha) < num_classes:
                    padding_val = 0.5
                    self.alpha = torch.cat([self.alpha, torch.full((num_classes - len(self.alpha),), padding_val)])
                else:
                    self.alpha = self.alpha[:num_classes]

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_tensor = self.alpha
            if not isinstance(alpha_tensor, torch.Tensor):
                alpha_tensor = torch.tensor(alpha_tensor, device=inputs.device)
            if alpha_tensor.device != inputs.device:
                alpha_tensor = alpha_tensor.to(inputs.device)
            
            if alpha_tensor.ndim > 1 or len(alpha_tensor) != self.num_classes:
                print(f"Error: FocalLoss alpha tensor shape is incorrect. Expected 1D of size {self.num_classes}, got {alpha_tensor.shape}")
                if alpha_tensor.numel() == 1:
                    focal_loss = alpha_tensor * focal_loss
                else:
                    print("FocalLoss: Alpha tensor shape mismatch, may lead to error.")
                    at = alpha_tensor.gather(0, targets.data.view(-1))
                    focal_loss = at * focal_loss
            else:
                at = alpha_tensor.gather(0, targets.data.view(-1))
                focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class Trop2Classifier(nn.Module):
    def __init__(self, model_name_str: str = MODEL_NAME, num_classes_val: int = NUM_CLASSES, dropout_val: float = DROPOUT_RATE, pretrained: bool = True):
        super(Trop2Classifier, self).__init__()
        
        # Load pretrained model based on global MODEL_NAME
        if model_name_str.startswith('efficientnet'):
            if model_name_str == "efficientnet_b0":
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                self.base_model = efficientnet_b0(weights=weights)
                num_features = self.base_model.classifier[1].in_features
                # Replace classifier for base_model (though we build a new one below)
                # self.base_model.classifier[1] = nn.Identity() # Or remove it, if we only use features
            else:
                # Add support for other efficientnet versions if needed
                raise ValueError(f"Unsupported EfficientNet version: {model_name_str}")
            
            self.features = self.base_model.features # Standard EfficientNet features
            self.attention = SpatialAttention() # Keep spatial attention
            
            # Custom classifier head, similar to original Trop2Classifier
            self.custom_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.BatchNorm1d(num_features),
                nn.Dropout(p=dropout_val),
                nn.Linear(num_features, num_features // 2),
                nn.ReLU(),
                nn.BatchNorm1d(num_features // 2),
                nn.Dropout(p=dropout_val / 2), # Original had dropout_rate / 2
                nn.Linear(num_features // 2, num_classes_val)
            )
        else:
            raise ValueError(f"Unsupported model architecture: {model_name_str}")
    
    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.custom_classifier(x)
        return x

class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=0.0001, restore_best_weights=True, mode='min', verbose=True, task_name=LABEL_COLUMN_NAME):
        """
        早停机制初始化
        
        Args:
            patience (int): 容忍的epoch数量
            min_delta (float): 最小改善阈值
            restore_best_weights (bool): 是否恢复最佳权重
            mode (str): 'min' 或 'max'
            verbose (bool): 是否打印信息
            task_name (str): 任务名称，用于日志
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        self.task_name = task_name
        
        if mode == 'max':
            self.monitor_op = lambda current, best: current > best + self.min_delta
            self.best_score = float('-inf')
        else:
            self.monitor_op = lambda current, best: current < best - self.min_delta
            self.best_score = float('inf')

    def __call__(self, score, model):
        """
        调用早停检查
        
        Args:
            score (float): 当前评分
            model (nn.Module): 模型实例
            
        Returns:
            bool: 是否应该停止训练
        """
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                print(f"EarlyStopping ({self.task_name}): 新的最佳分数: {self.best_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping ({self.task_name}): 计数器 {self.counter}/{self.patience}. 最佳分数: {self.best_score:.4f}")
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"EarlyStopping ({self.task_name}): 达到耐心值。停止训练。最佳分数: {self.best_score:.4f}")
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print(f"EarlyStopping ({self.task_name}): 已恢复最佳模型权重。")
        return self.early_stop


# %% [markdown]
# ## 定义评估函数 (to be aligned with grade_model style)

# %%
def evaluate_model(model, data_loader, device, class_names_list=None): # class_names will be [f"{LABEL_COLUMN_NAME}-0", ...]
    model.eval()
    all_preds = []
    all_labels = []
    all_probs_list = [] 
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="评估中"):
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs_list.append(probs.cpu().numpy())
    
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    all_probs_np = np.concatenate(all_probs_list, axis=0)

    accuracy = (all_preds_np == all_labels_np).mean()
    cm = confusion_matrix(all_labels_np, all_preds_np)
    cr = classification_report(all_labels_np, all_preds_np, target_names=class_names_list, digits=4, zero_division=0)
    
    # MODIFIED: Plot all P-R curves on a single figure
    plt.figure(figsize=(10, 8)) 
    for i in range(NUM_CLASSES):
        if i in np.unique(all_labels_np):
            precision, recall, _ = precision_recall_curve(all_labels_np == i, all_probs_np[:, i])
            ap_score = average_precision_score(all_labels_np == i, all_probs_np[:, i])
            plt.plot(recall, precision, lw=2, label=f'{class_names_list[i]} (AP = {ap_score:.3f})')
        else:
            # Optionally, plot a dummy line or just skip for missing classes
            plt.plot([],[], label=f'{class_names_list[i]} (无样本)') 

    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title(f'各类别P-R曲线 ({MODEL_NAME})')
    plt.legend(loc="best")
    plt.grid(True)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(f"pr_curves_combined_{MODEL_NAME}_val.png")
    plt.show()
    plt.close()
    
    # ROC曲线 (Macro and per-class) - remains the same
    plt.figure(figsize=(10, 8))
    # Per-class ROC
    for i in range(NUM_CLASSES):
        if i in np.unique(all_labels_np):
            fpr, tpr, _ = roc_curve(all_labels_np == i, all_probs_np[:, i])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names_list[i]} (AUC = {roc_auc_val:.4f})')
        else:
            plt.plot([], [], label=f'{class_names_list[i]} (无样本)') # Placeholder for legend
            print(f"ROC for class {class_names_list[i]} skipped as it is not in labels.")

    # Macro-average ROC AUC (One-vs-Rest)
    # This requires all_labels_np to be binarized for each class if not already handled by roc_auc_score with multi_class='ovr'
    try:
        macro_roc_auc_ovr = roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr', average='macro')
        print(f"Final Validation Macro ROC AUC (OvR) for {MODEL_NAME}: {macro_roc_auc_ovr:.4f}")
    except ValueError as e_roc:
        macro_roc_auc_ovr = float('nan')
        print(f"Could not calculate Macro ROC AUC for {MODEL_NAME}: {e_roc}")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(f'ROC曲线 (Macro AUC: {macro_roc_auc_ovr:.3f})')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"roc_curve_{MODEL_NAME}_val.png")
    plt.show()
    plt.close()
    
    # 4. 混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_list,
                yticklabels=class_names_list)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()
    
    # 5. 打印详细评估报告
    print("\n=== 模型评估报告 ===")
    print(f"\n总体准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(cr)
    
    # 6. 计算每个类别的特定指标
    for i in range(len(class_names_list)):
        # 对于每个类别，将其视为正类，其他类别视为负类
        true_positive = cm[i, i]  # 对角线上的值
        false_positive = cm[:, i].sum() - true_positive  # 该列之和减去对角线值
        false_negative = cm[i, :].sum() - true_positive  # 该行之和减去对角线值
        true_negative = cm.sum() - (true_positive + false_positive + false_negative)  # 总数减去其他三项
        
        # 计算指标
        sensitivity = true_positive / (true_positive + false_negative)  # 敏感性/召回率
        specificity = true_negative / (true_negative + false_positive)  # 特异性
        ppv = true_positive / (true_positive + false_positive)  # 阳性预测值/精确率
        npv = true_negative / (true_negative + false_negative)  # 阴性预测值
        
        print(f"\n{class_names_list[i]} 详细指标:")
        print(f"敏感性 (Sensitivity/Recall): {sensitivity:.4f}")
        print(f"特异性 (Specificity): {specificity:.4f}")
        print(f"阳性预测值 (PPV/Precision): {ppv:.4f}")
        print(f"阴性预测值 (NPV): {npv:.4f}")
        print(f"真阳性 (TP): {true_positive}")
        print(f"假阳性 (FP): {false_positive}")
        print(f"假阴性 (FN): {false_negative}")
        print(f"真阴性 (TN): {true_negative}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': cr,
        'predictions': all_preds_np,
        'probabilities': all_probs_np, 
        'true_labels': all_labels_np,   
        'macro_roc_auc': macro_roc_auc_ovr 
    }

# %% [markdown]
# ## 定义训练函数

# %%
def plot_training_history(history, config):
    """绘制训练历史
    
    Args:
        history (dict): 包含训练历史数据的字典
        config: 配置对象
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    epochs_ran = len(history['train_loss'])
    epoch_ticks = range(1, epochs_ran + 1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 训练和验证损失
    axs[0, 0].plot(epoch_ticks, history['train_loss'], color='tab:red', linestyle='-', marker='o', markersize=3, label='训练损失')
    if 'val_loss' in history and any(not np.isnan(x) for x in history['val_loss']):
        axs[0, 0].plot(epoch_ticks, history['val_loss'], color='tab:orange', linestyle=':', marker='x', markersize=3, label='验证损失')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
    axs[0, 0].set_title('损失函数变化')

    # 验证准确率
    if 'val_accuracy' in history and any(not np.isnan(x) for x in history['val_accuracy']):
        axs[0, 1].plot(epoch_ticks, history['val_accuracy'], color='tab:blue', linestyle='-', marker='s', markersize=3, label='验证准确率')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend(loc='lower right')
    axs[0, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
    axs[0, 1].set_ylim(0, 1.05)
    axs[0, 1].set_title('验证准确率')

    # 验证Macro AUC
    if 'val_auc' in history and any(not np.isnan(x) for x in history['val_auc']):
        axs[1, 0].plot(epoch_ticks, history['val_auc'], color='tab:purple', linestyle='--', marker='^', markersize=3, label='验证 Macro AUC')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Macro AUC')
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
    axs[1, 0].set_ylim(0, 1.05)
    axs[1, 0].set_title('验证 Macro AUC')

    # 学习率
    if 'lr' in history and len(history['lr']) == epochs_ran:
        axs[1, 1].plot(epoch_ticks, history['lr'], color='tab:green', linestyle='--', marker='.', markersize=3, label='学习率')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Learning Rate')
    axs[1, 1].legend(loc='upper right')
    axs[1, 1].set_yscale('log')
    axs[1, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
    axs[1, 1].set_title('学习率变化')
    
    fig.tight_layout()
    plt.suptitle(f'{MODEL_NAME} 分类训练过程监控', fontsize=16)
    fig.subplots_adjust(top=0.92)
    
    for ax_row in axs:
        for ax in ax_row:
            if epochs_ran < 20:
                ax.set_xticks(epoch_ticks)

    plt.savefig(f"training_history_{MODEL_NAME.lower()}.png")
    plt.show()
    plt.close()

# NEW function based on grade_model.py, adapted for Trop2
def split_data_by_patient_groupshufflesplit_trop2(df, patient_id_column, mapped_label_column_name, seed, val_size_from_trainval=0.2, label_column_name_log="Trop2"):
    df_trainval = df.copy() 
    
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_from_trainval, random_state=seed + 1) 

    if len(df_trainval[patient_id_column].unique()) > 1 and len(df_trainval) > 1:
        try:
            train_idx_inner, val_idx_inner = next(gss_val.split(df_trainval, groups=df_trainval[patient_id_column]))
            df_train = df_trainval.iloc[train_idx_inner].copy()
            df_val   = df_trainval.iloc[val_idx_inner].copy()
            print(f"Successfully used GroupShuffleSplit for train/validation sets for {label_column_name_log} from the full dataset.")
        except ValueError as e:
            print(f"Warning: GroupShuffleSplit for {label_column_name_log} train/validation failed: {e}. Falling back to random split on the full dataset.")
            stratify_col = df_trainval[mapped_label_column_name] if df_trainval[mapped_label_column_name].nunique() > 1 else None
            df_train, df_val = train_test_split(df_trainval, test_size=val_size_from_trainval, random_state=seed + 1, stratify=stratify_col)
    elif len(df_trainval) > 0: 
        print(f"Warning: Not enough unique patient groups or samples in {label_column_name_log} df_trainval for GroupShuffleSplit. Using random split or assigning all to train.")
        if len(df_trainval) > 1:
            stratify_col = df_trainval[mapped_label_column_name] if df_trainval[mapped_label_column_name].nunique() > 1 else None
            df_train, df_val = train_test_split(df_trainval, test_size=val_size_from_trainval, random_state=seed + 1, stratify=stratify_col)
        else: 
            df_train = df_trainval.copy()
            df_val = pd.DataFrame(columns=df_trainval.columns) 
            print(f"Warning: Only one sample in {label_column_name_log} data. Assigning to train set, validation set is empty.")
    else: 
        df_train = pd.DataFrame(columns=df.columns)
        df_val = pd.DataFrame(columns=df.columns)
        print(f"Warning: {label_column_name_log} df_trainval is empty. Train and Val sets are empty.")

    print(f"\nDataset sizes and class distributions ({label_column_name_log}):")
    for name, df_subset in [("Train", df_train), ("Val", df_val)]:
        if not df_subset.empty:
            print(f"  {name:<8}: {len(df_subset):>4} images, Patients: {df_subset[patient_id_column].nunique():>2}")
            distribution_info_series = df_subset[mapped_label_column_name].value_counts(normalize=True).sort_index()
            distribution_info_str = '\n'.join([f"    Class {idx}: {val:.4f}" for idx, val in distribution_info_series.items()])
            print(f"    Class distribution ({mapped_label_column_name}, normalized):\n{distribution_info_str}")
            print(f"    Unique patients per class ({mapped_label_column_name}):")
            if mapped_label_column_name in df_subset.columns and patient_id_column in df_subset.columns:
                for class_label in sorted(df_subset[mapped_label_column_name].unique()):
                    num_patients_in_class = df_subset[df_subset[mapped_label_column_name] == class_label][patient_id_column].nunique()
                    print(f"      Class {class_label}: {num_patients_in_class} patients")
            else:
                print("      Could not calculate unique patients per class (column missing).")
        else:
            print(f"  {name:<8}: Empty")
    print("\n")
    return df_train, df_val


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    scaler = GradScaler()
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [], 'val_accuracy': [],
        'val_auc': [], # For Macro AUC
        'lr': []
    }
    best_model_state = None
    no_improve_epochs = 0
    
    logging.info(f"Starting training for {config.num_epochs} epochs...")
    
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = len(train_loader) * config.warmup_epochs
    step = 0

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            if step < warmup_steps:
                lr = config.learning_rate * (step + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if config.gradient_clip_val is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}'})
            step += 1
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        history['train_loss'].append(epoch_loss)
        
        # Validation phase
        val_epoch_loss_accum = 0.0
        val_correct = 0
        val_total = 0
        all_val_probs_epoch_list = []
        all_val_labels_epoch_list = []

        if val_loader:  # Only validate if val_loader exists
            model.eval()
            print("Validating...")
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Validating"):
                    inputs, labels = inputs.to(config.device), labels.to(config.device)
                    outputs = model(inputs)
                    loss_val_iter = criterion(outputs, labels)
                    probs_val = torch.softmax(outputs, dim=1)
                    val_epoch_loss_accum += loss_val_iter.item() * inputs.size(0)  # Accumulate weighted loss
                    _, predicted_val = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted_val == labels).sum().item()
                    all_val_probs_epoch_list.append(probs_val.cpu().numpy())
                    all_val_labels_epoch_list.extend(labels.cpu().numpy())
                val_epoch_loss = val_epoch_loss_accum / val_total if val_total > 0 else float('nan')
                val_accuracy = (100 * val_correct / val_total) if val_total > 0 else float('nan')
                history['val_loss'].append(val_epoch_loss)
                history['val_accuracy'].append(val_accuracy / 100.0 if not np.isnan(val_accuracy) else float('nan'))  # Store as 0-1 for consistency

                all_val_labels_epoch_np = np.array(all_val_labels_epoch_list)
                all_val_probs_epoch_np = np.concatenate(all_val_probs_epoch_list, axis=0)
                
                val_auc_macro = float('nan')
                if val_total > 0 and len(np.unique(all_val_labels_epoch_np)) > 1:  # Need at least 2 unique classes for AUC
                    try:
                        val_auc_macro = roc_auc_score(all_val_labels_epoch_np, all_val_probs_epoch_np, multi_class='ovr', average='macro')
                    except ValueError as e_auc_val:
                        logging.warning(f"Epoch {epoch+1}, Val Macro AUC calculation error for {MODEL_NAME}: {e_auc_val}. AUC set to NaN.")
                history['val_auc'].append(val_auc_macro)
                
                current_lr_epoch = optimizer.param_groups[0]['lr']
                history['lr'].append(current_lr_epoch)
                
                logging.info(f'Epoch {epoch+1}/{config.num_epochs}: Train Loss={epoch_loss:.4f}, Val Loss={val_epoch_loss:.4f}, Val Acc={val_accuracy:.2f}%, Val Macro AUC={val_auc_macro:.4f}, LR={current_lr_epoch:.2e}')

                if scheduler and epoch >= config.warmup_epochs:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_epoch_loss)  # Monitor val_loss for ReduceLROnPlateau
                    else:
                        scheduler.step()
                        if current_lr_epoch < config.min_lr:
                            logging.info(f"\nLearning rate ({current_lr_epoch:.2e}) below min ({config.min_lr:.2e}), stopping training.")
                            break
            
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_model_state = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'val_loss': best_val_loss,
                        'val_accuracy': val_accuracy / 100.0 if not np.isnan(val_accuracy) else float('nan'),
                        'val_auc': val_auc_macro
                    }
                    logging.info(f"Saved new best model at epoch {epoch+1} with Val Loss: {best_val_loss:.4f}, Val Macro AUC: {val_auc_macro:.4f}")
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                
                if no_improve_epochs >= config.early_stopping_patience:
                    logging.info(f'\nEarly stopping triggered after {epoch+1} epochs due to no improvement in validation loss.')
                    break
        else:  # No val_loader
            history['val_loss'].append(float('nan'))
            history['val_accuracy'].append(float('nan'))
            history['val_auc'].append(float('nan'))
            current_lr_epoch = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr_epoch)
            logging.info(f'Epoch {epoch+1}/{config.num_epochs}: Train Loss={epoch_loss:.4f}, Val metrics=N/A (no val_loader), LR={current_lr_epoch:.2e}')

    plot_training_history(history, config)
    return history, best_model_state

# %% [markdown]
# ## 数据加载和预处理

# %%
def setup_logger(config: Config):
    """设置日志配置"""
    # 创建日志目录
    log_dir = os.path.join(config.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 记录初始信息
    logging.info(f"开始新的训练会话 - {timestamp}")
    logging.info(f"使用设备: {config.device}")
    logging.info(f"配置信息:\n{config}")

def save_training_state(config: Config, model_state: dict, train_history: dict, save_dir: str):
    """
    保存完整的训练状态
    
    Args:
        config: 配置对象
        model_state: 模型状态字典 (best_model_state)
        train_history: 训练历史数据 (history dict from train_model)
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, 'config.json')
    config.save_config(config_path)
    logging.info(f"配置已保存到: {config_path}")
    
    if model_state: # If best_model_state was captured
        model_save_path = os.path.join(save_dir, f"{MODEL_NAME}_best.pth") # More specific name
        torch.save(model_state, model_save_path)
        logging.info(f"最佳模型状态已保存到: {model_save_path}")
    else:
        logging.warning("No best model state was saved (possibly training ended early or no validation).")

    history_path = os.path.join(save_dir, 'training_history.json')
    # Convert NaN to None for JSON serialization if any NaNs are in history
    serializable_history = {k: [None if isinstance(x, float) and np.isnan(x) else x for x in v] if isinstance(v, list) else v for k,v in train_history.items()}
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_history, f, indent=4, ensure_ascii=False)
    logging.info(f"训练历史已保存到: {history_path}")

def load_trained_model(model_dir: str) -> Tuple[Trop2Classifier, Config]:
    """
    加载训练好的模型
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        tuple: (模型对象, 配置对象)
    """
    try:
        # 加载配置
        config_path = os.path.join(model_dir, 'config.json')
        config = Config.load_config(config_path)
        
        # 创建模型
        model = Trop2Classifier(config)
        
        # 加载模型权重
        model_path = os.path.join(model_dir, 'model.pth')
        state_dict = torch.load(model_path, map_location=config.device)
        model.load_state_dict(state_dict['model_state_dict'])
        
        # 将模型设置为评估模式
        model.eval()
        model.to(config.device)
        
        return model, config
        
    except Exception as e:
        logging.error(f"加载模型时发生错误: {str(e)}")
        raise

def predict_batch(model: Trop2Classifier, image_paths: List[str]) -> List[dict]:
    """
    批量预测多张图像
    
    Args:
        model: 模型对象
        image_paths: 图像路径列表
        
    Returns:
        list: 预测结果列表
    """
    results = []
    for image_path in tqdm(image_paths, desc="预测中"):
        try:
            result = model.predict(image_path)
            result['image_path'] = image_path
            results.append(result)
        except Exception as e:
            logging.warning(f"处理图像 {image_path} 时发生错误: {str(e)}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    return results

def analyze_model_performance(model, eval_results, config, save_dir):
    """
    分析模型性能
    
    Args:
        model: 训练好的模型
        eval_results: 评估结果
        config: 配置对象
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 分析预测错误的案例
    cm = eval_results['confusion_matrix']
    true_labels = eval_results['true_labels']
    predictions = eval_results['predictions']
    probabilities = eval_results['probabilities']
    
    # 找出预测错误的样本
    error_indices = np.where(predictions != true_labels)[0]
    
    error_analysis = {
        'total_samples': len(true_labels),
        'error_samples': len(error_indices),
        'error_rate': len(error_indices) / len(true_labels),
        'class_distribution': {
            f'Trop2-{i+1}': int(np.sum(true_labels == i))
            for i in range(config.num_classes)
        },
        'confusion_matrix': cm.tolist(),
        'error_details': []
    }
    
    # 2. 分析每个类别的预测置信度分布
    plt.figure(figsize=(15, 5))
    
    # 正确预测的置信度分布
    plt.subplot(1, 2, 1)
    correct_indices = np.where(predictions == true_labels)[0]
    correct_probs = np.max(probabilities[correct_indices], axis=1)
    plt.hist(correct_probs, bins=20, alpha=0.5, label='正确预测')
    plt.title('正确预测的置信度分布')
    plt.xlabel('置信度')
    plt.ylabel('样本数量')
    plt.legend()
    
    # 错误预测的置信度分布
    plt.subplot(1, 2, 2)
    error_probs = np.max(probabilities[error_indices], axis=1)
    plt.hist(error_probs, bins=20, alpha=0.5, label='错误预测')
    plt.title('错误预测的置信度分布')
    plt.xlabel('置信度')
    plt.ylabel('样本数量')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confidence_distribution.png'))
    plt.close()
    
    # 3. 分析类别间的混淆情况
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm / np.sum(cm, axis=1)[:, None], 
                annot=True, fmt='.2%', cmap='Blues',
                xticklabels=config.class_names,
                yticklabels=config.class_names)
    plt.title('归一化混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(os.path.join(save_dir, 'normalized_confusion_matrix.png'))
    plt.close()
    
    # 4. 计算每个类别的详细指标
    class_metrics = {}
    for i in range(config.num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity)
        
        class_metrics[f'Trop2-{i+1}'] = {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
            'f1_score': float(f1)
        }
    
    # 5. 生成性能分析报告
    report = {
        'error_analysis': error_analysis,
        'class_metrics': class_metrics,
        'overall_accuracy': float(eval_results['accuracy']),
        'model_confidence': {
            'mean_confidence_correct': float(np.mean(correct_probs)),
            'mean_confidence_error': float(np.mean(error_probs)),
            'high_confidence_errors': float(np.sum(error_probs > 0.9)) / len(error_probs)
        }
    }
    
    # 保存分析报告
    with open(os.path.join(save_dir, 'performance_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    # 生成可读性报告
    with open(os.path.join(save_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Trop2分类模型性能分析报告 ===\n\n")
        
        f.write("1. 总体性能指标:\n")
        f.write(f"   - 总体准确率: {report['overall_accuracy']:.4f}\n")
        f.write(f"   - 样本总数: {report['error_analysis']['total_samples']}\n")
        f.write(f"   - 错误样本数: {report['error_analysis']['error_samples']}\n")
        f.write(f"   - 错误率: {report['error_analysis']['error_rate']:.4f}\n\n")
        
        f.write("2. 各类别分布:\n")
        for class_name, count in report['error_analysis']['class_distribution'].items():
            f.write(f"   - {class_name}: {count}样本\n")
        f.write("\n")
        
        f.write("3. 各类别详细指标:\n")
        for class_name, metrics in report['class_metrics'].items():
            f.write(f"\n   {class_name}:\n")
            f.write(f"   - 敏感性: {metrics['sensitivity']:.4f}\n")
            f.write(f"   - 特异性: {metrics['specificity']:.4f}\n")
            f.write(f"   - PPV: {metrics['ppv']:.4f}\n")
            f.write(f"   - NPV: {metrics['npv']:.4f}\n")
            f.write(f"   - F1分数: {metrics['f1_score']:.4f}\n")
        
        f.write("\n4. 模型置信度分析:\n")
        f.write(f"   - 正确预测的平均置信度: {report['model_confidence']['mean_confidence_correct']:.4f}\n")
        f.write(f"   - 错误预测的平均置信度: {report['model_confidence']['mean_confidence_error']:.4f}\n")
        f.write(f"   - 高置信度错误比例: {report['model_confidence']['high_confidence_errors']:.4f}\n")
        
        f.write("\n5. 结论和建议:\n")
        # 添加基于阈值的判断
        if report['overall_accuracy'] < 0.6:
            f.write("   - 模型性能较差，可能无法通过图像准确预测Trop2表达水平\n")
            f.write("   - 建议：\n")
            f.write("     1. 检查数据质量和标注准确性\n")
            f.write("     2. 考虑是否存在图像特征与Trop2表达水平的关联\n")
            f.write("     3. 可能需要结合其他临床指标进行综合判断\n")
        elif report['model_confidence']['high_confidence_errors'] > 0.1:
            f.write("   - 模型存在高置信度错误预测，表明可能存在过拟合\n")
            f.write("   - 建议：\n")
            f.write("     1. 增加数据增强\n")
            f.write("     2. 调整模型复杂度\n")
            f.write("     3. 检查是否存在数据泄露\n")
        else:
            f.write("   - 模型性能尚可，但仍需进一步验证\n")
            f.write("   - 建议：\n")
            f.write("     1. 在更大的独立测试集上验证\n")
            f.write("     2. 进行临床验证\n")
            f.write("     3. 考虑模型集成或多模态方法\n")
    
    return report

def calculate_class_weights(df, num_classes=3, label_col_name='TROP2'): # Added label_col_name
    """计算类别权重"""
    # Ensure labels are 0-indexed for consistency if class_counts expects that
    # However, value_counts on the original 'TROP2' (1,2,3) is fine if we map later.
    # Let's assume df[label_col_name] has the original 1,2,3 labels.
    class_counts = df[label_col_name].value_counts().sort_index()
    
    # Handle cases where some classes might be missing in the df (especially small train_df)
    weights = []
    total_samples = len(df)
    for i in range(1, num_classes + 1): # Original labels are 1, 2, 3
        count = class_counts.get(i, 0) # Get count for original label i
        if count == 0:
            # Assign a default weight if class is missing, or handle as error
            # For FocalLoss, this might mean that class effectively gets low weight / ignored if alpha is also class-specific
            # Or, assign a high weight to prevent ignoring, but this needs care.
            # Let's use a common approach: 1.0, or total_samples / num_classes for a neutral effect for missing classes
            # This part depends on how FocalLoss handles missing classes and its alpha param.
            # For now, simple inverse weighting:
            print(f"Warning: Class {i} has 0 samples in the provided df for weight calculation. Assigning default weight.")
            # A large weight might be problematic if that class truly never appears.
            # Let's use 1.0 as a neutral placeholder, assuming FocalLoss alpha handles balance.
            weights.append(1.0) 
        else:
            weights.append(total_samples / (num_classes * count))
            
    # The FocalLoss implementation provided takes a single alpha or a list.
    # If a single alpha, class_weights are applied in CE.
    # If alpha is a list, it's an additional per-class weighting.
    # This `weights` tensor is for the `weight` param of `F.cross_entropy` inside FocalLoss.
    return torch.FloatTensor(weights)

# --- START: Feature-Label Relevance Analysis Functions (adapted for Trop2) ---

def get_embeddings_trop2(model: Trop2Classifier, dataloader, device, label_column_name_for_log="Trop2"):
    model.eval()
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=f"Extracting embeddings for {label_column_name_for_log}"):
            imgs = imgs.to(device)
            features = model.features(imgs)
            attention_weights = model.attention(features)
            attended_features = features * attention_weights
            # Assuming model.classifier[0] is AdaptiveAvgPool2d
            if hasattr(model.classifier, '0') and isinstance(model.classifier[0], nn.AdaptiveAvgPool2d):
                 pooled_features = model.classifier[0](attended_features) 
                 embeddings = torch.flatten(pooled_features, 1)
            else: 
                 logging.warning(f"Trop2Classifier structure for embedding extraction not as expected. Using attended features globally pooled.")
                 pooled_features = F.adaptive_avg_pool2d(attended_features, (1,1))
                 embeddings = torch.flatten(pooled_features, 1)
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return np.concatenate(embeddings_list) if embeddings_list else np.array([]), \
           np.concatenate(labels_list) if labels_list else np.array([])

def calculate_mutual_information_trop2(features_or_probs, labels):
    if features_or_probs.ndim == 1:
        features_or_probs = features_or_probs.reshape(-1, 1)
    # mutual_info_classif handles multi-class targets directly.
    # If features_or_probs are embeddings (multi-dimensional), it calculates MI for each feature against labels.
    # If features_or_probs are probabilities (N, num_classes), this usage is less standard for MI.
    # For MI with probabilities, usually one would use MI between P(class_i) and true_label_is_class_i (binary per class) or P(predicted_class) vs true_class.
    # Let's assume if it's probabilities, it's (N,1) for a specific class prob vs all labels.
    # If it's embeddings (N, D_embed), MI will be (D_embed,).
    mi = mutual_info_classif(features_or_probs, labels, random_state=config.random_seed)
    return mi

def plot_tsne_visualization_trop2(embeddings, labels, label_column_name="Trop2", title_suffix=""):
    if len(embeddings) == 0:
        logging.warning(f"Cannot run t-SNE for {label_column_name}: No embeddings provided.")
        return
    logging.info(f"Running t-SNE for {label_column_name}...")
    perplexity_val = min(30, len(embeddings)-1 if len(embeddings) > 1 else 1)
    if perplexity_val <=0:
        logging.warning(f"Perplexity for t-SNE is {perplexity_val}, which is invalid. Skipping t-SNE.")
        return

    tsne = TSNE(n_components=2, random_state=config.random_seed, perplexity=perplexity_val, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    cmap = plt.colormaps.get_cmap("viridis") # More modern way than plt.cm.get_cmap
    colors = cmap(np.linspace(0, 1, len(unique_labels)))

    for i, label_val in enumerate(unique_labels):
        idx = labels == label_val
        class_display_name = config.class_names[label_val] if label_val < len(config.class_names) else f"Class {label_val}"
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], color=colors[i], label=class_display_name, alpha=0.7)
    
    plt.title(f't-SNE 可视化 ({label_column_name}{title_suffix})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"tsne_visualization_{label_column_name.lower().replace('-','_')}{title_suffix.replace(' ', '_').lower()}.png")
    plt.show()
    plt.close()

def simulate_data_cleaning_test_trop2(model, original_val_loader, original_labels_np, probabilities_all_classes_np, num_samples_to_flip=100, label_column_name="Trop2"):
    logging.info(f"\nSimulating data cleaning test for {label_column_name} by randomly re-assigning {num_samples_to_flip} labels...")
    
    if len(original_labels_np) < num_samples_to_flip:
        logging.warning(f"Not enough samples ({len(original_labels_np)}) to re-assign {num_samples_to_flip}. Skipping simulation.")
        return float('nan')

    flipped_labels_np = original_labels_np.copy()
    indices_to_flip = np.random.choice(len(flipped_labels_np), num_samples_to_flip, replace=False)
    
    for idx in indices_to_flip:
        original_label = flipped_labels_np[idx]
        possible_new_labels = [l for l in range(config.num_classes) if l != original_label]
        if not possible_new_labels: # Should not happen if num_classes > 1
            logging.warning(f"Cannot flip label {original_label} as no other classes exist.")
            continue
        flipped_labels_np[idx] = np.random.choice(possible_new_labels)
        
    eval_probs_np = probabilities_all_classes_np # Use pre-calculated probabilities (N, num_classes)

    if len(np.unique(flipped_labels_np)) < 2: # Technically, for Macro AUC, we often need all classes present or careful handling
        logging.warning(f"After re-assigning, less than 2 unique classes in simulated labels for {label_column_name}. AUC might be ill-defined or misleading.")
        # return float('nan') # Or proceed if roc_auc_score can handle it
        
    try:
        auc_after_cleaning = roc_auc_score(flipped_labels_np, eval_probs_np, multi_class='ovr', average='macro')
        logging.info(f"Macro AUC after simulated cleaning for {label_column_name} ({num_samples_to_flip} labels re-assigned): {auc_after_cleaning:.4f}")
        return auc_after_cleaning
    except ValueError as e:
        logging.error(f"Error calculating Macro AUC after simulated cleaning for {label_column_name}: {e}")
        return float('nan')

def perform_permutation_test_trop2(model, val_loader, original_labels_np, original_probs_all_classes_np, n_permutations=1000, label_column_name="Trop2"):
    logging.info(f"\nPerforming permutation test for {label_column_name} (Macro AUC) with {n_permutations} permutations...")
    
    if len(np.unique(original_labels_np)) < config.num_classes:
        logging.warning(f"Original labels for {label_column_name} have less than {config.num_classes} unique classes. Permutation test for Macro AUC might be less reliable.")

    try:
        observed_auc = roc_auc_score(original_labels_np, original_probs_all_classes_np, multi_class='ovr', average='macro')
    except ValueError as e:
        logging.error(f"Could not calculate observed Macro AUC for {label_column_name}: {e}. Permutation test skipped.")
        return float('nan')
        
    logging.info(f"Observed Macro AUC for {label_column_name}: {observed_auc:.4f}")
    
    permuted_aucs = []
    for i in tqdm(range(n_permutations), desc=f"Permutation Test {label_column_name} (Macro AUC)"):
        permuted_labels = sklearn_shuffle(original_labels_np, random_state=config.random_seed + i)
        try:
            auc_val = roc_auc_score(permuted_labels, original_probs_all_classes_np, multi_class='ovr', average='macro')
            permuted_aucs.append(auc_val)
        except ValueError: # Handle cases where a permutation might lead to issues (e.g. only one class in permuted_labels for a fold in OvR)
             permuted_aucs.append(0.5 if config.num_classes==2 else 1.0/config.num_classes) # Approx. random chance for multi-class AUC

    permuted_aucs = np.array(permuted_aucs)
    p_value = np.mean(permuted_aucs >= observed_auc)
    
    logging.info(f"Permutation test for {label_column_name} (Macro AUC): p-value = {p_value:.4f}")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(permuted_aucs, bins=30, kde=True, label='Permuted Macro AUCs')
    plt.axvline(observed_auc, color='red', linestyle='--', lw=2, label=f'Observed Macro AUC ({observed_auc:.3f})')
    plt.title(f'Permutation Test Results for {label_column_name} (Macro AUC)')
    plt.xlabel('Macro AUC Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"permutation_test_{label_column_name.lower().replace('-','_')}_macro_auc.png")
    plt.show()
    plt.close()
    return p_value

# --- END: Feature-Label Relevance Analysis Functions ---

# --- START: Grad-CAM Visualization Function (adapted for Trop2) ---
# TROP2_CLASS_NAMES_FOR_GRADCAM already defined in config.class_names (Trop2-0, Trop2-1, Trop2-2)

def visualize_grad_cam_trop2(model: Trop2Classifier, dataset: Trop2Dataset, device, num_images_per_target_class=2, target_classes_to_viz=None, label_column_name="Trop2"):
    # Target layer for EfficientNet-B0 in Trop2Classifier.model.features
    # Based on typical EfficientNet structure, model.features[-1][0] is often the last Conv2dNormActivation
    try:
        target_layer_module = model.features[-1][0]
    except Exception as e_layer:
        logging.warning(f"Could not get default target layer model.features[-1][0] for Grad-CAM: {e_layer}. Trying model.features._conv_head")
        try:
            target_layer_module = model.features._conv_head
        except Exception as e_layer2:
            logging.error(f"Could not find _conv_head either for Grad-CAM: {e_layer2}. Grad-CAM will be skipped.")
            return
    target_layers = [target_layer_module]

    cam_obj = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available() and device.type=="cuda")

    if not dataset or len(dataset) == 0:
        logging.warning(f"Dataset for {label_column_name} Grad-CAM is empty.") 
        return

    if target_classes_to_viz is None:
        target_classes_to_viz = list(range(config.num_classes)) # For Trop2: [0, 1, 2]
    
    images_shown_count = 0
    num_viz_rows = len(target_classes_to_viz)
    num_viz_cols = num_images_per_target_class 
    
    if num_viz_rows * num_viz_cols == 0:
        logging.warning(f"No images or target classes specified for {label_column_name} Grad-CAM.")
        return

    fig, axes = plt.subplots(num_viz_rows * 2, num_viz_cols, figsize=(num_viz_cols * 3, num_viz_rows * 6))
    # Handle a single image total (1 target class, 1 image)
    if num_viz_rows == 1 and num_viz_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]]) # Make it 2D for consistent indexing
    elif num_viz_cols == 1 and num_viz_rows > 1:
        axes = axes.reshape(num_viz_rows * 2, 1)
    elif num_viz_rows == 1 and num_viz_cols > 1:
        axes = axes.reshape(2, num_viz_cols)
    elif num_viz_rows * num_viz_cols == 0:
        logging.warning("Grad-CAM subplot calculation resulted in zero rows/cols. Should have been caught.")
        plt.close(fig)
        return

    # Select images for each target class to visualize
    images_to_process_indices = []
    for target_cls in target_classes_to_viz:
        # Find images actually belonging to this class (optional, but good for representative CAMs)
        # Or just pick random images from dataset. For now, picking random from whole dataset.
        # If we want images *of* that class, we'd need to iterate through dataset.df
        # This simplified version just picks random images from the dataset for each slot.
        if len(dataset) >= num_images_per_target_class:
            indices_for_this_class = np.random.choice(len(dataset), num_images_per_target_class, replace=False)
        else:
            indices_for_this_class = np.random.choice(len(dataset), num_images_per_target_class, replace=True) # Allow replacement if not enough images
        images_to_process_indices.append(indices_for_this_class)

    for r_idx, target_cls in enumerate(target_classes_to_viz):
        img_indices_for_current_target_cls = images_to_process_indices[r_idx]
        for c_idx_local, img_idx_in_dataset in enumerate(img_indices_for_current_target_cls):
            img_tensor, true_label_scalar = dataset[img_idx_in_dataset] 
            true_label = true_label_scalar.item() 
            
            inv_normalize = transforms.Normalize(
                mean=[-m/s for m, s in zip(config.normalize_mean, config.normalize_std)],
                std=[1/s for s in config.normalize_std]
            )
            rgb_img_denorm = inv_normalize(img_tensor.clone()).permute(1, 2, 0).cpu().numpy()
            rgb_img_denorm = np.clip(rgb_img_denorm, 0, 1)

            input_tensor_unsqueeze = img_tensor.unsqueeze(0).to(device)
            cam_targets = [ClassifierOutputTarget(target_cls)]
            
            grayscale_cam_batch = cam_obj(input_tensor=input_tensor_unsqueeze, targets=cam_targets)
            if grayscale_cam_batch is not None and grayscale_cam_batch.shape[0] > 0:
                grayscale_cam = grayscale_cam_batch[0, :] 
            else:
                logging.warning(f"Grad-CAM returned None or empty for image index {img_idx_in_dataset}, target class {target_cls}.") 
                ax_orig = axes[r_idx * 2, c_idx_local] 
                ax_cam  = axes[r_idx * 2 + 1, c_idx_local]
                ax_orig.axis('off'); ax_cam.axis('off')
                continue
            
            cam_image = show_cam_on_image(rgb_img_denorm, grayscale_cam, use_rgb=True)
            original_img_for_grid_display = (rgb_img_denorm * 255).astype(np.uint8)
            
            title_str = f"""True: {config.class_names[true_label]}\nCAM for: {config.class_names[target_cls]}""" 

            ax_orig_current = axes[r_idx * 2, c_idx_local] 
            ax_cam_current = axes[r_idx * 2 + 1, c_idx_local]

            ax_orig_current.imshow(original_img_for_grid_display)
            ax_orig_current.set_title(title_str, fontsize=8)
            ax_orig_current.axis('off')
            ax_cam_current.imshow(cam_image)
            ax_cam_current.axis('off')
            images_shown_count +=1

    if images_shown_count == 0:
        logging.warning(f"No {label_column_name} CAM images were generated.") 
        if num_viz_rows * num_viz_cols > 0 : plt.close(fig) 
        return

    fig.suptitle(f"Grad-CAM for {label_column_name} Model (Targeting Various Classes)", fontsize=12) 
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_filename = f'grad_cam_{label_column_name.lower().replace("-","_")}_multiclass.png' 
    plt.savefig(save_filename)
    logging.info(f"Grad-CAM grid for {label_column_name} saved to {save_filename}") 
    plt.show()
    plt.close(fig)

# --- END: Grad-CAM Visualization Function ---


def main():
    """主函数"""
    try:
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.set_default_tensor_type(torch.FloatTensor)
        setup_logger(config)
        logging.info(f"正在读取标签文件: {config.label_file}")
        df = pd.read_csv(config.label_file)
        logging.info(f"原始数据大小: {len(df)}")
        logging.info(f"CSV列名: {list(df.columns)}")
        df = df.dropna(subset=['TROP2', 'PATIENT_ID', 'FILE_NAME']).copy()
        df = df[df['TROP2'].isin([1, 2, 3])].copy()
        df['TROP2'] = df['TROP2'].astype(int)
        logging.info(f"清洗后数据大小 (valid TROP2 1,2,3 & PATIENT_ID & FILE_NAME): {len(df)}")
        class_dist = df['TROP2'].value_counts(normalize=True).sort_index()
        logging.info(f"\n整体类别分布 (TROP2 original 1,2,3):\n{class_dist}")
        patient_counts = df['PATIENT_ID'].value_counts()
        logging.info("\n每个病人的图像数量统计:")
        logging.info(f"最小值: {patient_counts.min()}")
        logging.info(f"最大值: {patient_counts.max()}")
        logging.info(f"平均值: {patient_counts.mean():.2f}")
        df['TROP2_mapped'] = df['TROP2'] - 1
        train_df, val_df = split_data_by_patient_groupshufflesplit_trop2(
            df, patient_id_column="PATIENT_ID", mapped_label_column_name='TROP2_mapped',
            seed=config.random_seed, val_size_from_trainval=config.val_size_from_trainval,
            label_column_name_log="Trop2"
        )
        if train_df.empty:
            logging.error("Training dataframe is empty after split. Cannot proceed.")
            return None, None, None
        class_weights_for_loss = calculate_class_weights(train_df, num_classes=config.num_classes, label_col_name='TROP2').to(config.device)
        logging.info(f"类别权重 for FocalLoss (based on train_df): {class_weights_for_loss.tolist()}")
        
        # 创建数据集
        train_dataset = Trop2Dataset(train_df, config.image_dir, transform=config.train_transform)
        val_dataset = Trop2Dataset(val_df, config.image_dir, transform=config.val_transform) if not val_df.empty else None
        
        # 设置数据加载器参数
        train_loader_args = {'batch_size': config.batch_size, 'num_workers': config.num_workers, 'pin_memory': True}
        if not train_df.empty:
            temp_train_labels_for_sampler = train_df['TROP2_mapped'].astype(int)
            counts_train = temp_train_labels_for_sampler.value_counts().sort_index()
            if len(counts_train) > 0 and len(counts_train) <= config.num_classes:
                class_sample_weights_values = [0.0] * config.num_classes
                for i in range(config.num_classes):
                    class_sample_weights_values[i] = 1. / counts_train.get(i, 1e-6)
                sample_weights_train = [class_sample_weights_values[label] for label in temp_train_labels_for_sampler]
                sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights_train), num_samples=len(sample_weights_train), replacement=True)
                weights_str = ", ".join([f"Class {i}: {w:.4f}" for i, w in enumerate(class_sample_weights_values)])
                logging.info(f"Sampler weights for Trop2 classes (0,1,2): {weights_str}")
                train_loader_args['sampler'] = sampler
                train_loader_args['shuffle'] = False
            else:
                logging.warning("Training data for Trop2 has insufficient or unexpected class counts for sampler. Using standard DataLoader with shuffle=True.")
                train_loader_args['shuffle'] = True
        else:
            logging.warning("train_df for Trop2 is empty. Using standard DataLoader, shuffle=False.")
            train_loader_args['shuffle'] = False
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, **train_loader_args)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True) if val_dataset else None

        logging.info("初始化模型...")
        model = Trop2Classifier(config).to(config.device)
        model.float()
        
        # Set loss function and optimizer
        criterion = FocalLoss(
            alpha=config.focal_alpha, 
            gamma=config.focal_gamma,
            class_weights=class_weights_for_loss
        )
        optimizer = config.get_optimizer(model.parameters())
        scheduler = config.get_scheduler(optimizer)
        
        # Train model
        logging.info("开始训练模型...")
        history, best_model_state = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config
        )
        
        # Save training state
        current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_save_dir = os.path.join(config.save_dir, f"{MODEL_NAME}_{current_time_str}")
        save_training_state(config, best_model_state, history, experiment_save_dir)
        
        final_eval_results = None
        if best_model_state and val_loader:
            model.load_state_dict(best_model_state['model_state_dict'])
            logging.info(f"加载最佳模型 (Epoch: {best_model_state['epoch']}, Val Loss: {best_model_state['val_loss']:.4f}, Val Macro AUC: {best_model_state.get('val_auc', float('nan')):.4f}) 进行最终评估...")
            final_eval_results = evaluate_model(model, val_loader, config.device, config.class_names)
            logging.info(f"最终评估完成. Accuracy: {final_eval_results['accuracy']:.4f}, Macro AUC: {final_eval_results.get('macro_roc_auc', float('nan')):.4f}")
            eval_results_path = os.path.join(experiment_save_dir, "evaluation_results.json")
            serializable_eval_results = {}
            for k, v in final_eval_results.items():
                if isinstance(v, np.ndarray): 
                    serializable_eval_results[k] = v.tolist()
                elif isinstance(v, pd.DataFrame): 
                    serializable_eval_results[k] = v.to_dict()
                else: 
                    serializable_eval_results[k] = v
            with open(eval_results_path, 'w') as f: 
                json.dump(serializable_eval_results, f, indent=4)
            logging.info(f"Evaluation results saved to {eval_results_path}")

            # --- Execute Feature-Label Relevance Analysis & Grad-CAM ---
            logging.info(f"\n--- Starting Extended Feature-Label Relevance Analysis for Trop2 ---")
            all_val_labels_final_np = final_eval_results['true_labels'] 
            all_val_probs_final_np = final_eval_results['probabilities']  # Shape (N, num_classes)

            val_embeddings, val_true_labels_for_analysis = get_embeddings_trop2(model, val_loader, config.device, label_column_name_for_log="Trop2")

            if val_embeddings.size > 0 and val_true_labels_for_analysis.size > 0 and len(val_embeddings) == len(val_true_labels_for_analysis):
                mi_scores_embeddings = calculate_mutual_information_trop2(val_embeddings, val_true_labels_for_analysis)
                logging.info(f"Mean Mutual Information (Embeddings vs Labels) for Trop2: {np.mean(mi_scores_embeddings):.4f}")

                plot_tsne_visualization_trop2(val_embeddings, val_true_labels_for_analysis, label_column_name="Trop2")

                num_samples_to_flip_trop2 = min(max(1, len(all_val_labels_final_np) // 5), 100) 
                simulate_data_cleaning_test_trop2(model, val_loader, np.array(all_val_labels_final_np), all_val_probs_final_np, 
                                                num_samples_to_flip=num_samples_to_flip_trop2, label_column_name="Trop2")
                
                perform_permutation_test_trop2(model, val_loader, np.array(all_val_labels_final_np), all_val_probs_final_np, 
                                             n_permutations=1000, label_column_name="Trop2")
            else:
                logging.warning("Could not extract embeddings or labels for Trop2 extended analysis. Skipping some parts.")

            logging.info(f"\nVisualizing Grad-CAM for Trop2 model")
            if val_dataset and len(val_dataset) > 0:
                visualize_grad_cam_trop2(model, dataset=val_dataset, device=config.device, 
                                       num_images_per_target_class=2, target_classes_to_viz=[0,1,2], 
                                       label_column_name="Trop2")
            else:
                logging.warning("Skipping Trop2 Grad-CAM: Validation dataset not available or empty.")

        elif not val_loader:
            logging.info("No validation loader provided, skipping final evaluation and extended analysis.")
        else:
            logging.warning("No best model state found, skipping final evaluation and extended analysis.")
        
        return model, final_eval_results, None

    except Exception as e:
        logging.error(f"Trop2 classifier training/evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)
    
if __name__ == "__main__":
    # try: # Original try was here, moved into main() for better error message context
    model_result, eval_summary, perf_report = main()
    if model_result is None : # main() can return None if setup fails (e.g. empty train_df)
        logging.error("Main function did not complete successfully. Exiting.")
        sys.exit(1) 
    # except Exception as e: # Catching outside main might obscure where the error originated within main
    #     logging.error("程序执行失败 (Trop2)", exc_info=True)
    #     sys.exit(1) 
# %%
