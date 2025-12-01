# %% [markdown]
# # HER-2 膀胱镜图像三分类模型
# 通过迁移学习（EfficientNet-B0）对 HER-2 进行三分类，按患者划分训练/验证/测试集，尽量避免同一患者图像落到同一集合。

# %%
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support, roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt # 导入matplotlib
import seaborn as sns # 导入seaborn
import torch.nn.functional as F # 添加 F 功能模块的导入
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.utils import shuffle as sklearn_shuffle

# 配置matplotlib中文字体
def setup_chinese_font():
    """配置matplotlib中文字体，优先使用系统中已有的"""
    import matplotlib.font_manager as fm
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = 'DejaVu Sans' # 默认备用字体
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    plt.rcParams['font.sans-serif'] = [selected_font]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print(f"Matplotlib using font: {selected_font}")

setup_chinese_font()

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Focal Loss 定义 (可根据三分类需求调整或替换为CrossEntropyLoss)
# %%
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean', num_classes=4): # Changed num_classes to 4
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        if isinstance(alpha, (float, int)): 
            # For multi-class, alpha might need to be a list of weights per class
            # Example: if alpha=0.25, it might be interpreted as [0.25, 0.25, 0.25, ...] or specific weights
            # For simplicity, if a single float is given, we'll assume equal weighting unless specified otherwise
             print("Warning: Single float alpha in FocalLoss for multi-class might not be standard. Consider a list of weights.")
             self.alpha = torch.tensor([alpha] * num_classes) 
        if isinstance(alpha, list): self.alpha = torch.tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (1-pt)**self.gamma * CE_loss

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            
            # For multi-class, ensure alpha is correctly indexed by target classes
            at = self.alpha.gather(0, targets.data.view(-1))
            F_loss = at * F_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# %% [markdown]
# ## 自定义 Dataset

# %%
class Her2Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        """
        df: 包含 FILE_NAME, MAPPED_LABEL_COLUMN_NAME, PATIENT_ID (假设HER-2标签列名为 MAPPED_LABEL_COLUMN_NAME)
        image_dir: 图像根路径
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row[FILE_NAME_COLUMN]) # Use global const for file name col
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # 使用 MAPPED_LABEL_COLUMN_NAME 获取标签
        label = int(row[MAPPED_LABEL_COLUMN_NAME]) 
        return img, label

# %% [markdown]
# ## 读取标签并划分数据集

# %%
# 1. 读取 CSV
# !!! 需要用户确认CSV文件名和 HER-2 标签列名 !!!
LABEL_COLUMN_NAME = 'HER-2' # 假设原始HER-2状态列
MAPPED_LABEL_COLUMN_NAME = 'HER-2_mapped_label' # 处理后的多分类标签列 (0, 1, 2, 3)
PATIENT_ID_COLUMN = 'PATIENT_ID'
FILE_NAME_COLUMN = 'FILE_NAME'

label_df = pd.read_csv("dataset/label.csv") # 您可能需要修改这个CSV文件名
print(f"1. Initial rows loaded from CSV: {len(label_df)}")

# 2. 丢弃空值 (基于 HER-2 标签列, 文件名, 患者ID)
label_df = label_df.dropna(subset=[LABEL_COLUMN_NAME, FILE_NAME_COLUMN, PATIENT_ID_COLUMN]).copy()
print(f"2. Rows after dropping NA from key columns ('{LABEL_COLUMN_NAME}', '{FILE_NAME_COLUMN}', '{PATIENT_ID_COLUMN}'): {len(label_df)}")
if len(label_df) == 0:
    print("ERROR: All rows were dropped after initial NA check. Please check your CSV for missing values in key columns.")
else:
    print(f"   Unique values in '{LABEL_COLUMN_NAME}' before mapping: {label_df[LABEL_COLUMN_NAME].astype(str).unique()}")


# 3. 创建多分类标签 (重要: 此处逻辑需要根据实际 HER-2 标签值进行调整)
# 示例：假设 HER-2 原始值为 0.0, 1.0, 2.0, 3.0 (或其他需要映射的值)
# '0.0' -> 类别 0
# '1.0' -> 类别 1
# '2.0' -> 类别 2
# '3.0' -> 类别 3
# !!! 用户需要根据实际情况修改此映射 !!!
def map_her2_to_labels(her2_status):
    s_status = str(her2_status).strip().lower()
    if s_status == '0.0' or s_status == '0':
        return 0
    elif s_status == '1.0' or s_status == '1':
        return 1
    elif s_status == '2.0' or s_status == '2':
        return 2
    elif s_status == '3.0' or s_status == '3':
        return 3
    else:
        print(f"Warning: Unexpected HER-2 status '{her2_status}' (normalized: '{s_status}'). Mapping to NaN.")
        return np.nan

if len(label_df) > 0:
    label_df[MAPPED_LABEL_COLUMN_NAME] = label_df[LABEL_COLUMN_NAME].apply(map_her2_to_labels)
    print(f"3. Rows after applying 'map_her2_to_labels' (before dropping NA from '{MAPPED_LABEL_COLUMN_NAME}'): {len(label_df)}")
    print(f"   Unique values in '{MAPPED_LABEL_COLUMN_NAME}' after mapping (before dropping NA): {label_df[MAPPED_LABEL_COLUMN_NAME].unique()}")

    label_df = label_df.dropna(subset=[MAPPED_LABEL_COLUMN_NAME]).copy() # Drop rows where mapping failed
    print(f"4. Rows after dropping NA from '{MAPPED_LABEL_COLUMN_NAME}': {len(label_df)}")
    if len(label_df) > 0:
        label_df[MAPPED_LABEL_COLUMN_NAME] = label_df[MAPPED_LABEL_COLUMN_NAME].astype(int)
        print(f"   Final unique values in '{MAPPED_LABEL_COLUMN_NAME}': {label_df[MAPPED_LABEL_COLUMN_NAME].unique()}")
    else:
        print(f"ERROR: All rows were dropped after mapping to '{MAPPED_LABEL_COLUMN_NAME}' and dropping NAs. Check 'map_her2_to_labels' logic and original '{LABEL_COLUMN_NAME}' values.")
else:
    print("Skipping mapping and further processing as DataFrame is empty after initial NA check.")


if len(label_df) > 0:
    # Removed GroupShuffleSplit for test set. df_trainval is now the entire preprocessed label_df.
    df_trainval = label_df.copy() 
    # df_test is no longer created.
    
    val_size_from_trainval = 0.2 # Proportion of data to use for validation

    # Splitting df_trainval (which is now the full dataset) into train and validation sets
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_from_trainval, random_state=SEED + 1) 

    if len(df_trainval[PATIENT_ID_COLUMN].unique()) > 1 and len(df_trainval) > 1 : # Need at least 2 groups and more than 1 sample for GroupShuffleSplit
        try:  # <--- 这个 try 语句块需要缩进
        train_idx_inner, val_idx_inner = next(gss_val.split(df_trainval, groups=df_trainval[PATIENT_ID_COLUMN]))
        df_train = df_trainval.iloc[train_idx_inner].copy()
        df_val   = df_trainval.iloc[val_idx_inner].copy()
            print(f"Successfully used GroupShuffleSplit for train/validation sets for {LABEL_COLUMN_NAME} from the full dataset.")
        except ValueError as e: # <--- 这个 except 语句块也需要同步缩进
            print(f"Warning: GroupShuffleSplit for {LABEL_COLUMN_NAME} train/validation failed: {e}. Falling back to random split on the full dataset.")
            # Fallback to train_test_split if GroupShuffleSplit fails (e.g., not enough groups)
            stratify_col = df_trainval[MAPPED_LABEL_COLUMN_NAME] if df_trainval[MAPPED_LABEL_COLUMN_NAME].nunique() > 1 else None
            df_train, df_val = train_test_split(df_trainval, test_size=val_size_from_trainval, random_state=SEED + 1, stratify=stratify_col)
    elif len(df_trainval) > 0: # If not enough unique patients for GroupShuffleSplit, or only one sample, do a simple split or assign all to train
        print(f"Warning: Not enough unique patient groups or samples in {LABEL_COLUMN_NAME} df_trainval for GroupShuffleSplit. Using random split or assigning all to train.")
        if len(df_trainval) > 1 :
            stratify_col = df_trainval[MAPPED_LABEL_COLUMN_NAME] if df_trainval[MAPPED_LABEL_COLUMN_NAME].nunique() > 1 else None
            df_train, df_val = train_test_split(df_trainval, test_size=val_size_from_trainval, random_state=SEED + 1, stratify=stratify_col)
        else: # Only one sample
            df_train = df_trainval.copy()
            df_val = pd.DataFrame(columns=df_trainval.columns) # Empty validation set
            print(f"Warning: Only one sample in {LABEL_COLUMN_NAME} data. Assigning to train set, validation set is empty.")
    else: # df_trainval is empty
        df_train = pd.DataFrame(columns=label_df.columns)
        df_val = pd.DataFrame(columns=label_df.columns)
        print(f"Warning: {LABEL_COLUMN_NAME} df_trainval is empty. Train and Val sets are empty.")
else: # label_df was empty from the start
    df_train = pd.DataFrame(columns=label_df.columns)
    df_val = pd.DataFrame(columns=label_df.columns)
    print(f"Critical Error: {LABEL_COLUMN_NAME} label_df is empty from the start. Cannot proceed.")


print("\nDataset sizes and class distributions:")
for name, df_subset in [("Train", df_train), ("Val", df_val)]: # Removed "Test"
    if not df_subset.empty:
        print(f"  {name:<8}: {len(df_subset):>4} images, Patients: {df_subset[PATIENT_ID_COLUMN].nunique():>2}")
        print(f"    Class distribution ({MAPPED_LABEL_COLUMN_NAME}, normalized):\n{df_subset[MAPPED_LABEL_COLUMN_NAME].value_counts(normalize=True).sort_index()}")
        # New: Print unique patients per class
        print(f"    Unique patients per class ({MAPPED_LABEL_COLUMN_NAME}):")
        if MAPPED_LABEL_COLUMN_NAME in df_subset.columns and PATIENT_ID_COLUMN in df_subset.columns:
            for class_label in sorted(df_subset[MAPPED_LABEL_COLUMN_NAME].unique()):
                num_patients_in_class = df_subset[df_subset[MAPPED_LABEL_COLUMN_NAME] == class_label][PATIENT_ID_COLUMN].nunique()
                print(f"      Class {class_label}: {num_patients_in_class} patients")
        else:
            print("      Could not calculate unique patients per class (column missing).")
    else:
        print(f"  {name:<8}: Empty")
print("\n")

# %% [markdown]
# ## 数据增强与 DataLoader

# %%
IMG_DIR = "dataset/image"  # 图像文件夹

# 变换
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# WeightedRandomSampler for balancing the training set (3 classes)
if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
    counts_train = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    num_classes_train = len(counts_train)
    if num_classes_train > 0:
        class_sample_weights = [1. / counts_train.get(i, 1e-6) for i in range(num_classes_train)] # Use .get for safety
        sample_weights_train = [class_sample_weights[label] for label in df_train[MAPPED_LABEL_COLUMN_NAME]]
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights_train), 
                                        num_samples=len(sample_weights_train), 
                                        replacement=True)
        print(f"Sampler weights for classes: {', '.join([f'Class {i}: {w:.4f}' for i, w in enumerate(class_sample_weights)])}")
        train_loader_args = {'sampler': sampler, 'shuffle': False}
    else:
        print("Warning: Training data is empty or has no class labels for sampler. Using standard DataLoader.")
        train_loader_args = {'shuffle': True} # Default if sampler cannot be created
else:
    print("Warning: df_train is empty or MAPPED_LABEL_COLUMN_NAME is missing. Using standard DataLoader.")
    train_loader_args = {'shuffle': True}


# Dataset & DataLoader
train_ds = Her2Dataset(df_train, IMG_DIR, transform=train_tf)
val_ds   = Her2Dataset(df_val,   IMG_DIR, transform=val_tf)
# test_ds  = Her2Dataset(df_test,  IMG_DIR, transform=val_tf) # Removed test_ds

train_loader = DataLoader(train_ds, batch_size=16, num_workers=0, pin_memory=True, **train_loader_args)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
# test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=0, pin_memory=True) # Removed test_loader

# %% [markdown]
# ## 模型定义与训练设置

# %%
NUM_CLASSES = 4 # For HER-2 four-class classification

print("Using EfficientNet-B0 for HER-2 classification")
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

# 修改分类头以适应四分类
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES) 

model = model.to(device)

# FocalLoss alpha parameter for 4 classes
if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns and len(df_train[MAPPED_LABEL_COLUMN_NAME].unique()) >= NUM_CLASSES-1 :
    # If NUM_CLASSES is 4, we expect at least 3 unique classes to attempt weight calculation, or all 4.
    # The check len(df_train[MAPPED_LABEL_COLUMN_NAME].unique()) == NUM_CLASSES might be too strict if one class is rare in train
    counts = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    focal_loss_alpha_values = []
    # Ensure alpha values are generated for all NUM_CLASSES even if some are not in counts
    for i in range(NUM_CLASSES):
        if i in counts:
            # Basic inverse frequency, can be refined
            focal_loss_alpha_values.append(1.0 / counts[i]) 
        else:
            focal_loss_alpha_values.append(1.0) # Default weight if class not in train set (e.g. 1.0 or other heuristic)
    
    # Normalize alpha weights so they sum to 1 (or a different strategy can be used)
    total_inverse_freq = sum(focal_loss_alpha_values)
    if total_inverse_freq > 0:
        focal_loss_alpha_values = [w / total_inverse_freq for w in focal_loss_alpha_values]
    else: # Should not happen if there's data, but as a fallback
        focal_loss_alpha_values = [1.0/NUM_CLASSES] * NUM_CLASSES

    print(f"Train data counts for HER-2: { {k:v for k,v in counts.items()} }")
    print(f"Calculated FocalLoss alpha values: {focal_loss_alpha_values}")
else: 
    focal_loss_alpha_values = [1.0/NUM_CLASSES] * NUM_CLASSES # Equal weights
    print(f"Warning: Could not calculate class counts for FocalLoss alpha dynamically. Using equal weights: {focal_loss_alpha_values}")

focal_loss_alpha = torch.tensor(focal_loss_alpha_values, dtype=torch.float).to(device)
print(f"Using Focal Loss with alpha: {focal_loss_alpha.tolist()} and gamma=2 for HER-2 ({NUM_CLASSES} classes)")

# 损失与优化器
criterion = FocalLoss(alpha=focal_loss_alpha, gamma=2, num_classes=NUM_CLASSES)
# Alternatively, for multi-class, CrossEntropyLoss is standard:
# criterion = nn.CrossEntropyLoss() # Add class weights here if needed: weight=torch.tensor([w0,w1,w2]).to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4) 

# 学习率调度（可选, 基于val_loss or val_accuracy or macro_val_auc）
# For multi-class, AUC is more complex. val_loss or accuracy might be simpler.
# Let's use val_loss (mode 'min') for scheduler.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True, min_lr=1e-6)

# %% [markdown]
# ## 早停机制

# %%
class EarlyStopping:
    """早停机制，当验证集性能不再提升时停止训练"""
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True, mode='min', verbose=True): # Default to min for loss
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        if mode == 'max':
            self.monitor_op = lambda current, best: current > best + self.min_delta
            self.best_score = float('-inf')
        else: # mode == 'min'
            self.monitor_op = lambda current, best: current < best - self.min_delta
            self.best_score = float('inf')

    def __call__(self, score, model):
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                print(f"EarlyStopping: New best score: {self.best_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: Counter {self.counter}/{self.patience}. Best score: {self.best_score:.4f}")
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"EarlyStopping: Patience reached. Stopping training. Best score: {self.best_score:.4f}")
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print("EarlyStopping: Restored best model weights.")
        return self.early_stop

# %% [markdown]
# ## 训练与验证循环

# %%
# For multi-class AUC, use average='macro' or 'weighted'
# from sklearn.metrics import roc_auc_score (already imported)

NUM_EPOCHS = 50 
# best_val_metric will depend on what we monitor (e.g. loss, accuracy, macro_auc)
best_val_loss = float('inf') 

# Monitor validation loss for early stopping and model saving
early_stopping = EarlyStopping(patience=10, mode='min', verbose=True, min_delta=0.0001) 

history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [], # Added accuracy
    'val_macro_auc': [], # Added macro AUC for multi-class
    'lr': []
}

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train Her2]"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs) # Shape: (batch_size, NUM_CLASSES)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    
    if len(train_ds) > 0:
        train_loss = running_loss / len(train_ds)
    else:
        train_loss = 0.0
        print(f"Warning: Epoch {epoch}, Training dataset is empty. Train loss set to 0.")

    history['train_loss'].append(train_loss)
    
    model.eval()
    all_val_labels = []
    all_val_probs = []  # Store probabilities for each class
    all_val_preds = []  # Store predicted class indices
    val_running_loss = 0.0
    
    if len(val_ds) > 0:
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss_iter = criterion(outputs, labels)
                val_running_loss += val_loss_iter.item() * imgs.size(0)
                
                probs = torch.softmax(outputs, dim=1) # Probabilities for all classes
                preds = torch.argmax(probs, dim=1)    # Predicted class index
                
                all_val_probs.extend(probs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
        
        val_epoch_loss = val_running_loss / len(val_ds)
        history['val_loss'].append(val_epoch_loss)

        all_val_labels_np = np.array(all_val_labels)
        all_val_preds_np = np.array(all_val_preds)
        all_val_probs_np = np.array(all_val_probs)

        # Calculate Accuracy
        val_accuracy = np.mean(all_val_labels_np == all_val_preds_np) if len(all_val_labels_np) > 0 else 0.0
        history['val_accuracy'].append(val_accuracy)

        # Calculate Macro AUC
        # roc_auc_score for multi-class needs probabilities for each class and ovr/ovo strategy
        try:
            if len(np.unique(all_val_labels_np)) == NUM_CLASSES : # Needs all classes present for some averaging methods
                 val_macro_auc = roc_auc_score(all_val_labels_np, all_val_probs_np, multi_class='ovr', average='macro')
            elif len(np.unique(all_val_labels_np)) > 1 : # At least 2 classes
                 val_macro_auc = roc_auc_score(all_val_labels_np, all_val_probs_np, multi_class='ovr', average='macro', labels=np.arange(NUM_CLASSES))
            else: # only one class present
                 val_macro_auc = 0.0
                 print(f"Warning: Epoch {epoch}, Validation set only has one class or too few classes for Macro AUC. AUC set to 0.0")
        except ValueError as e_auc: # Handle cases where AUC cannot be computed
            print(f"Warning: Epoch {epoch}, Could not compute Macro AUC: {e_auc}. AUC set to 0.0")
            val_macro_auc = 0.0
        history['val_macro_auc'].append(val_macro_auc)
        
        scheduler.step(val_epoch_loss) # Step scheduler based on validation loss
        
        print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_epoch_loss:.4f}, Val Acc={val_accuracy:.4f}, Val MacroAUC={val_macro_auc:.4f}, LR={optimizer.param_groups[0]['lr']:.1e}")

        # Save model based on validation loss
        if val_epoch_loss < best_val_loss: 
            best_val_loss = val_epoch_loss
            # No need to check early_stopping.monitor_op if we are just comparing to best_val_loss
            torch.save(model.state_dict(), "best_model_her2.pth")
            print(f"Epoch {epoch}: New best Her2 model saved with Val Loss: {best_val_loss:.4f}")
            
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        # Early stopping also based on validation loss
        if early_stopping(val_epoch_loss, model):
            print("Early stopping triggered based on validation loss.")
            break
    else: # Validation dataset is empty
        history['val_loss'].append(float('nan'))
        history['val_accuracy'].append(float('nan'))
        history['val_macro_auc'].append(float('nan'))
        print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Val Loss=N/A (empty val set), LR={optimizer.param_groups[0]['lr']:.1e}")
        # Potentially save model based on train_loss if no validation is possible
        # but this is generally not recommended.
        # if epoch > early_stopping.patience : # Avoid saving too early if no val set
        #     print("Warning: No validation set. Cannot determine best model or early stop effectively.")

# %% [markdown]
# ## 绘制训练过程曲线

# %%
def plot_training_history_her2(history):
    epochs_ran = len(history['train_loss'])
    epoch_ticks = range(1, epochs_ran + 1)

    fig, ax1 = plt.subplots(figsize=(14, 7))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epoch_ticks, history['train_loss'], color=color, linestyle='-', marker='o', markersize=3, label='训练损失')
    if 'val_loss' in history and any(not np.isnan(x) for x in history['val_loss']):
        ax1.plot(epoch_ticks, history['val_loss'], color=color, linestyle=':', marker='x', markersize=3, label='验证损失')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy / Macro AUC', color=color)  
    if 'val_accuracy' in history and any(not np.isnan(x) for x in history['val_accuracy']):
        ax2.plot(epoch_ticks, history['val_accuracy'], color=color, linestyle='-', marker='s', markersize=3, label='验证准确率')
    if 'val_macro_auc' in history and any(not np.isnan(x) for x in history['val_macro_auc']):
        ax2.plot(epoch_ticks, history['val_macro_auc'], color='tab:purple', linestyle='--', marker='^', markersize=3, label='验证 Macro AUC')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.05) # AUC and Accuracy are between 0 and 1

    ax3 = ax1.twinx() 
    ax3.spines["right"].set_position(("outward", 60)) 
    color = 'tab:green'
    ax3.set_ylabel('Learning Rate', color=color)
    if 'lr' in history:
        ax3.plot(epoch_ticks, history['lr'], color=color, linestyle='--', marker='.', markersize=3, label='学习率')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.legend(loc='lower right')
    ax3.set_yscale('log') 

    fig.tight_layout()  
    plt.title('HER-2 四分类训练过程监控')
    plt.xticks(epoch_ticks) # Ensure all epoch numbers are shown if not too many
    plt.savefig("training_history_her2.png")
    plt.show()

if any(history.values()): # Check if history has any data
    plot_training_history_her2(history)
else:
    print("No training history to plot.")


# %% [markdown]
# ## 验证集最终评估曲线 (ROC & P-R) for HER-2

# %% 
print(f"\nGenerating ROC and P-R curves for {LABEL_COLUMN_NAME} on the validation set using the best model.")

# 定义类别名称 (需要根据您的 HER-2 类别定义进行调整)
HER2_CLASS_NAMES_FOR_PLOT = [f'{LABEL_COLUMN_NAME} Class {i}' for i in range(NUM_CLASSES)]
if len(HER2_CLASS_NAMES_FOR_PLOT) != NUM_CLASSES:
    HER2_CLASS_NAMES_FOR_PLOT = [f"Class {i}" for i in range(NUM_CLASSES)]
    print(f"Warning: Mismatch or issue with HER2_CLASS_NAMES_FOR_PLOT. Using default: {HER2_CLASS_NAMES_FOR_PLOT}")

try:
    model.load_state_dict(torch.load("best_model_her2.pth")) # MODIFIED: Use consistent filename
    model.eval()
    print(f"Loaded best {LABEL_COLUMN_NAME} model for final validation set evaluation.")

    all_val_labels_final = []
    all_val_probs_final = [] # Store probabilities for all classes
    
    if len(val_ds) > 0:
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Final Validation for ROC/PR {LABEL_COLUMN_NAME}"):
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1) 
                all_val_probs_final.extend(probs.cpu().numpy())
                all_val_labels_final.extend(labels.numpy())

        all_val_labels_final_np = np.array(all_val_labels_final)
        all_val_probs_final_np = np.array(all_val_probs_final)

        # ROC Curves (One-vs-Rest for each class)
        plt.figure(figsize=(10, 8))
                for i in range(NUM_CLASSES):
            y_true_binary = (all_val_labels_final_np == i).astype(int)
            if len(np.unique(y_true_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_true_binary, all_val_probs_final_np[:, i])
                    try:
                    auc_val = roc_auc_score(y_true_binary, all_val_probs_final_np[:, i])
                    plt.plot(fpr, tpr, lw=2, label=f'{HER2_CLASS_NAMES_FOR_PLOT[i]} (AUC = {auc_val:.3f})')
                    except ValueError:
                    plt.plot(fpr, tpr, lw=2, label=f'{HER2_CLASS_NAMES_FOR_PLOT[i]} (AUC = N/A)')
                    print(f"Could not calculate AUC for class {i} in OvR ROC on validation set.")
            else:
                print(f"Skipping ROC curve for class {i} on validation set (OvR) due to single class in y_true_binary.")

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (False Positive Rate)')
        plt.ylabel('真阳性率 (True Positive Rate)')
        plt.title(f'验证集ROC曲线 (One-vs-Rest) ({LABEL_COLUMN_NAME})')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'roc_curve_{LABEL_COLUMN_NAME.lower()}_val_ovr.png')
        plt.show()

        # Precision-Recall Curves (One-vs-Rest for each class)
        from sklearn.metrics import precision_recall_curve, average_precision_score
        plt.figure(figsize=(10, 8))
        for i in range(NUM_CLASSES):
            y_true_binary = (all_val_labels_final_np == i).astype(int)
            if len(np.unique(y_true_binary)) > 1:
                precision, recall, _ = precision_recall_curve(y_true_binary, all_val_probs_final_np[:, i])
                try:
                    ap_score = average_precision_score(y_true_binary, all_val_probs_final_np[:, i])
                    plt.plot(recall, precision, lw=2, label=f'{HER2_CLASS_NAMES_FOR_PLOT[i]} (AP = {ap_score:.3f})')
                except ValueError:
                    plt.plot(recall, precision, lw=2, label=f'{HER2_CLASS_NAMES_FOR_PLOT[i]} (AP = N/A)')
                    print(f"Could not calculate Average Precision for class {i} in OvR P-R on validation set.")
            else:
                print(f"Skipping P-R curve for class {i} on validation set (OvR) due to single class in y_true_binary.")

        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title(f'验证集P-R曲线 (One-vs-Rest) ({LABEL_COLUMN_NAME})')
        plt.legend(loc="best") # Changed to best from lower left for potentially better placement
        plt.grid(True)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.savefig(f'pr_curve_{LABEL_COLUMN_NAME.lower()}_val_ovr.png')
        plt.show()
    else:
        print(f"Validation dataset for {LABEL_COLUMN_NAME} is empty. No final ROC/PR curves generated.")

except FileNotFoundError:
    print(f"Error: 'best_model_her2.pth' not found. Was the model trained and saved? Cannot generate final ROC/PR curves for {LABEL_COLUMN_NAME}.") # MODIFIED: Consistent filename in error message
except Exception as e:
    print(f"An error occurred during final {LABEL_COLUMN_NAME} validation set ROC/PR curve generation: {e}")


# %% [markdown]
# ## 特征与标签相关性分析 (互信息, t-SNE, 置换检验)

# %% 
# ==> Add the new analysis functions here:

def get_embeddings_efficientnet(model, dataloader, device):
    model.eval()
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=f"Extracting embeddings for {LABEL_COLUMN_NAME}"):
            imgs = imgs.to(device)
            features = model.features(imgs)
            pooled_features = model.avgpool(features)
            embeddings = torch.flatten(pooled_features, 1)
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(embeddings_list), np.concatenate(labels_list)

def calculate_mutual_information(features, labels):
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    mi = mutual_info_classif(features, labels, random_state=SEED)
    return mi

def plot_tsne_visualization(embeddings, labels, title_suffix=""):
    print(f"Running t-SNE for {LABEL_COLUMN_NAME}...")
    perplexity_value = min(30, len(embeddings) - 1 if len(embeddings) > 1 else 1)
    if perplexity_value <= 0:
        print(f"Warning: Perplexity for t-SNE is {perplexity_value}, which is invalid. Skipping t-SNE for {LABEL_COLUMN_NAME}.")
        return
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity_value, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("viridis", len(unique_labels))
    
    for i, label_val in enumerate(unique_labels):
        idx = labels == label_val
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], color=colors(i), label=f'{LABEL_COLUMN_NAME} Class {label_val}', alpha=0.7)
    
    plt.title(f't-SNE 可视化 ({LABEL_COLUMN_NAME}{title_suffix})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"tsne_visualization_{LABEL_COLUMN_NAME.lower()}{title_suffix.replace(' ', '_').lower()}.png")
    plt.show()

def simulate_data_cleaning_test_multiclass(model, original_val_loader, original_labels_np, num_samples_to_flip=100, num_classes_for_sim=NUM_CLASSES, random_state=SEED):
    print(f"\nSimulating data cleaning test for {LABEL_COLUMN_NAME} by randomly re-assigning {num_samples_to_flip} labels...")
    
    if len(original_labels_np) < num_samples_to_flip:
        print(f"Warning: Not enough samples ({len(original_labels_np)}) to flip {num_samples_to_flip}. Skipping simulation.")
        return float('nan')

    flipped_labels_np = original_labels_np.copy()
    indices_to_flip = np.random.choice(len(flipped_labels_np), num_samples_to_flip, replace=False)
    
    # For multiclass, flip to a different random class
    for idx in indices_to_flip:
        original_class = flipped_labels_np[idx]
        possible_new_classes = [c for c in range(num_classes_for_sim) if c != original_class]
        if not possible_new_classes: # Should not happen if num_classes > 1
            continue 
        flipped_labels_np[idx] = np.random.choice(possible_new_classes)
    
    global all_val_probs_final_np # Using the globally calculated probabilities from validation
    if 'all_val_probs_final_np' not in globals() or all_val_probs_final_np is None or len(all_val_probs_final_np) != len(flipped_labels_np):
        print("Warning: `all_val_probs_final_np` not available or mismatched. Re-evaluating model for data cleaning test.")
        model.eval()
        temp_probs_list = []
        with torch.no_grad():
            for imgs, _ in tqdm(original_val_loader, desc=f"Re-evaluating for data cleaning test {LABEL_COLUMN_NAME}"):
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                temp_probs_list.extend(probs.cpu().numpy())
        eval_probs_np = np.array(temp_probs_list)
    else:
        eval_probs_np = all_val_probs_final_np

    try:
        # Using macro AUC for multiclass evaluation after simulated cleaning
        auc_after_cleaning = roc_auc_score(flipped_labels_np, eval_probs_np, multi_class='ovr', average='macro', labels=np.arange(num_classes_for_sim))
        print(f"Macro AUC after simulated cleaning for {LABEL_COLUMN_NAME} ({num_samples_to_flip} labels re-assigned): {auc_after_cleaning:.4f}")
        return auc_after_cleaning
    except ValueError as e:
        print(f"Error calculating Macro AUC after simulated cleaning for {LABEL_COLUMN_NAME}: {e}")
        return float('nan')

def perform_permutation_test_multiclass(model, val_loader, original_labels_np, original_probs_np, num_classes_for_perm_test=NUM_CLASSES, n_permutations=1000):
    print(f"\nPerforming permutation test for {LABEL_COLUMN_NAME} with {n_permutations} permutations...")
    
    try:
        observed_auc = roc_auc_score(original_labels_np, original_probs_np, multi_class='ovr', average='macro', labels=np.arange(num_classes_for_perm_test))
    except ValueError as e:
        print(f"Could not calculate observed Macro AUC for {LABEL_COLUMN_NAME}: {e}. Permutation test skipped.")
        return float('nan')
        
    print(f"Observed Macro AUC for {LABEL_COLUMN_NAME}: {observed_auc:.4f}")
    
    permuted_aucs = []
    for i in tqdm(range(n_permutations), desc=f"Permutation Test {LABEL_COLUMN_NAME}"):
        permuted_labels = sklearn_shuffle(original_labels_np, random_state=SEED + i)
        try:
            auc = roc_auc_score(permuted_labels, original_probs_np, multi_class='ovr', average='macro', labels=np.arange(num_classes_for_perm_test))
            permuted_aucs.append(auc)
        except ValueError:
             # If a permutation results in too few classes for macro AUC, assign a low/neutral score
             permuted_aucs.append(0.0) # Or 1.0/num_classes as a random chance baseline

    permuted_aucs = np.array(permuted_aucs)
    p_value = np.mean(permuted_aucs >= observed_auc)
    
    print(f"Permutation test for {LABEL_COLUMN_NAME} (Macro AUC): p-value = {p_value:.4f}")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(permuted_aucs, bins=30, kde=True, label='Permuted Macro AUCs')
    plt.axvline(observed_auc, color='red', linestyle='--', lw=2, label=f'Observed Macro AUC ({observed_auc:.3f})')
    plt.title(f'Permutation Test Results for {LABEL_COLUMN_NAME} (Macro AUC)')
    plt.xlabel('Macro AUC Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"permutation_test_{LABEL_COLUMN_NAME.lower()}_macro_auc.png")
    plt.show()
    
    return p_value

# --- Main execution of the new analyses ---
global all_val_labels_final_np, all_val_probs_final_np # Declare as global to be accessible by analysis functions

if 'model' in locals() and 'val_loader' in locals() and len(val_ds) > 0:
    print(f"\n--- Starting Feature-Label Relevance Analysis for {LABEL_COLUMN_NAME} ---")
    
    try:
        model.load_state_dict(torch.load("best_model_her2.pth")) # MODIFIED: Use consistent filename
        model.to(device)
        print(f"Loaded best {LABEL_COLUMN_NAME} model for feature-label analysis.")

        if 'all_val_labels_final_np' not in globals() or 'all_val_probs_final_np' not in globals() or \
            all_val_labels_final_np is None or all_val_probs_final_np is None or \
            len(all_val_labels_final_np) != len(val_ds) or len(all_val_probs_final_np) != len(val_ds):
            
            print("Recalculating final validation labels and probabilities for analysis...")
            temp_all_val_labels_final_list = []
            temp_all_val_probs_final_list = []
            model.eval()
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc=f"Recalculating Validation Data for Analysis {LABEL_COLUMN_NAME}"):
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    probs = torch.softmax(outputs, dim=1)
                    temp_all_val_probs_final_list.extend(probs.cpu().numpy())
                    temp_all_val_labels_final_list.extend(labels.numpy())
            all_val_labels_final_np = np.array(temp_all_val_labels_final_list)
            all_val_probs_final_np = np.array(temp_all_val_probs_final_list)

        val_embeddings, val_true_labels_for_analysis = get_embeddings_efficientnet(model, val_loader, device)
        
        if val_embeddings is not None and len(val_embeddings) > 0 and len(val_true_labels_for_analysis) == len(val_embeddings):
            # Mutual Information: Use probabilities of the true class vs other features, or embeddings directly.
            # For multiclass, MI can be complex. Here, we do a simplified version using embedding features against labels.
            mi_scores_embeddings = calculate_mutual_information(val_embeddings, val_true_labels_for_analysis)
            print(f"Mean Mutual Information (Embeddings vs Labels) for {LABEL_COLUMN_NAME}: {np.mean(mi_scores_embeddings):.4f}")

            plot_tsne_visualization(val_embeddings, val_true_labels_for_analysis)

            num_flip_samples = min(100, len(all_val_labels_final_np) // 2 if len(all_val_labels_final_np) > 0 else 1)
            simulate_data_cleaning_test_multiclass(model, val_loader, all_val_labels_final_np, num_samples_to_flip=num_flip_samples)
            
            if all_val_labels_final_np is not None and all_val_probs_final_np is not None and len(all_val_labels_final_np)>0:
                perform_permutation_test_multiclass(model, val_loader, all_val_labels_final_np, all_val_probs_final_np)
            else:
                print(f"Skipping Permutation Test for {LABEL_COLUMN_NAME} due to missing validation labels or probabilities.")
        else:
            print(f"Could not extract embeddings or labels for {LABEL_COLUMN_NAME} analysis. Skipping.")
            
    except FileNotFoundError:
        print(f"Error: 'best_model_her2.pth' not found. Cannot perform feature-label analysis for {LABEL_COLUMN_NAME}.") # MODIFIED: Consistent filename in error message
    except Exception as e_analysis:
        print(f"An error occurred during {LABEL_COLUMN_NAME} feature-label relevance analysis: {e_analysis}")
else:
    print(f"Skipping Feature-Label Relevance Analysis for {LABEL_COLUMN_NAME}: Model or validation data not available or val_ds is empty.")


# %% [markdown]
# ## Grad-CAM 可视化

# %%
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.utils import make_grid, save_image

# 定义HER2类别名称，用于Grad-CAM标题
HER2_CLASS_NAMES_GRADCAM = [f'{LABEL_COLUMN_NAME} Class {i}' for i in range(NUM_CLASSES)]

def visualize_grad_cam_her2_grid(model, dataset, device, num_images_per_target_class=2, target_classes_to_viz=None):
    target_layer_name = 'features[-1][0]' # 适配 EfficientNet-B0
    try:
            module_path = target_layer_name.split('.')
            current_module = model
            for m_name in module_path:
                if m_name.isdigit(): current_module = current_module[int(m_name)]
                elif m_name.startswith('[') and m_name.endswith(']'): idx = int(m_name[1:-1]); current_module = current_module[idx]
                else: current_module = getattr(current_module, m_name)
            target_layers = [current_module]
    except Exception as e:
        print(f"Error finding target layer '{target_layer_name}' for {LABEL_COLUMN_NAME} Grad-CAM: {e}. Defaulting to model.features[-1].")
        target_layers = [model.features[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    if len(dataset) == 0:
        print(f"Dataset for {LABEL_COLUMN_NAME} Grad-CAM is empty.")
        return

    if target_classes_to_viz is None:
        target_classes_to_viz = list(range(NUM_CLASSES))
    
    num_target_classes = len(target_classes_to_viz)
    if num_target_classes == 0:
        print(f"No target classes specified for {LABEL_COLUMN_NAME} Grad-CAM.")
         return
         
    # 确保每个目标类别都有足够的图像可供选择
    # 我们将为每个目标类别选择 num_images_per_target_class 张不同的图像
    # 总共的图像数量 = num_target_classes * num_images_per_target_class
    total_images_needed = num_target_classes * num_images_per_target_class
    if len(dataset) < total_images_needed:
        print(f"Warning: Requested {total_images_needed} images for Grad-CAM ({num_images_per_target_class} per target class), \ 
                but dataset only has {len(dataset)}. Adjusting num_images_per_target_class.")
        num_images_per_target_class = len(dataset) // num_target_classes
        if num_images_per_target_class == 0 and len(dataset) > 0:
            num_images_per_target_class = 1 # Show at least one image if possible
            total_images_needed = num_target_classes # Recalculate total needed
            if len(dataset) < total_images_needed:
                 print("Dataset too small for even one image per target class. Reducing target classes.")
                 target_classes_to_viz = target_classes_to_viz[:len(dataset)]
                 num_target_classes = len(target_classes_to_viz)
                 if num_target_classes == 0: return
                 total_images_needed = num_target_classes
        elif num_images_per_target_class == 0 and len(dataset) == 0:
            print("No images to display for Grad-CAM as dataset is empty.")
            return
        print(f"Adjusted num_images_per_target_class to {num_images_per_target_class}")

    fig_rows = num_target_classes * 2 # 2 rows per target class (original + CAM)
    fig_cols = num_images_per_target_class

    if fig_rows * fig_cols == 0:
        print(f"Calculated zero images for Grad-CAM grid. Skipping.")
        return
        
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 3, fig_rows * 3))
    # Handle cases where subplots returns a 1D array or single Axes object
    if fig_rows == 1 and fig_cols == 1: axes = np.array([[axes]]) 
    elif fig_rows == 1: axes = axes.reshape(1, fig_cols)
    elif fig_cols == 1: axes = axes.reshape(fig_rows, 1)

    images_shown_count = 0
    # Create a pool of indices to pick from, to avoid reusing images if possible
    available_indices = list(range(len(dataset)))
    random.shuffle(available_indices)

    for r_idx, target_cls in enumerate(target_classes_to_viz):
        if not available_indices or len(available_indices) < num_images_per_target_class:
            print(f"Warning: Not enough unique images for target class {target_cls}. Some images might be repeated or skipped.")
            # Refill or break if strictly no repeats desired and not enough images
            # For this version, we'll try to use what's left, or reuse if we must (though current logic tries to prevent reuse for a single target class viz)
            # The check for total_images_needed and adjustment of num_images_per_target_class should minimize this

        # Select `num_images_per_target_class` distinct images for the current target_cls
        selected_img_indices_for_class = []
        temp_available_indices = available_indices.copy()
        for _ in range(num_images_per_target_class):
            if not temp_available_indices:
                # Fallback: if we run out of unique images, pick randomly from the whole dataset (allowing repeats across classes)
                # This part of the logic might need refinement if strict non-repetition across all visualizations is key
                # and the dataset is very small.
                selected_img_indices_for_class.append(random.choice(range(len(dataset))))
            else:
                selected_img_indices_for_class.append(temp_available_indices.pop(0))
        
        # Update available_indices by removing those that were definitely used if we want to avoid reuse across target_classes
        # This simple pop from main list is not done here to ensure each class gets its attempt at unique images
        # A more robust pool management would be needed for strict global non-repetition with small datasets.

        for c_idx, img_idx_in_dataset in enumerate(selected_img_indices_for_class):
            img_tensor, true_label = dataset[img_idx_in_dataset]
            inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
            rgb_img_denorm = inv_normalize(img_tensor.clone()).permute(1, 2, 0).cpu().numpy()
            rgb_img_denorm = np.clip(rgb_img_denorm, 0, 1) 

            input_tensor_unsqueeze = img_tensor.unsqueeze(0).to(device)
            cam_targets = [ClassifierOutputTarget(target_cls)]
            
            grayscale_cam = cam(input_tensor=input_tensor_unsqueeze, targets=cam_targets)
            if grayscale_cam is None or grayscale_cam.shape[0] == 0:
                print(f"Warning: {LABEL_COLUMN_NAME} Grad-CAM returned None/empty for img_idx {img_idx_in_dataset}, target_cls {target_cls}.")
                axes[r_idx * 2, c_idx].axis('off')
                axes[r_idx * 2 + 1, c_idx].axis('off')
                continue
            grayscale_cam_batch = grayscale_cam[0, :]
            
            cam_image = show_cam_on_image(rgb_img_denorm, grayscale_cam_batch, use_rgb=True)
            cam_image_tensor = transforms.ToTensor()(cam_image) 
            original_img_for_grid = transforms.ToTensor()(Image.fromarray((rgb_img_denorm * 255).astype(np.uint8)))
            
            title_str = f"True: {HER2_CLASS_NAMES_GRADCAM[true_label]}\nCAM for: {HER2_CLASS_NAMES_GRADCAM[target_cls]}"

            ax_orig_current = axes[r_idx * 2, c_idx]
            ax_cam_current = axes[r_idx * 2 + 1, c_idx]

            ax_orig_current.imshow(original_img_for_grid.permute(1,2,0).numpy())
            ax_orig_current.set_title(title_str, fontsize=8)
            ax_orig_current.axis('off')
            ax_cam_current.imshow(cam_image_tensor.permute(1,2,0).numpy())
            ax_cam_current.axis('off')
            images_shown_count +=1

    if images_shown_count == 0:
        print(f"No {LABEL_COLUMN_NAME} CAM images were generated.")
        if fig_rows * fig_cols > 0 : plt.close(fig) 
        return

    fig.suptitle(f"Grad-CAM for {LABEL_COLUMN_NAME} Model (Targeting Various Classes)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_filename = f'grad_cam_{LABEL_COLUMN_NAME.lower()}_multiclass_grid.png'
        plt.savefig(save_filename)
    print(f"Grad-CAM grid for {LABEL_COLUMN_NAME} saved to {save_filename}")
        plt.show()

if 'model' in locals() and 'val_ds' in locals() and len(val_ds) > 0:
    print(f"\nVisualizing Grad-CAM for {LABEL_COLUMN_NAME} model on validation set")
    visualize_grad_cam_her2_grid(model, dataset=val_ds, device=device, num_images_per_target_class=2, target_classes_to_viz=list(range(NUM_CLASSES)))
else:
    print(f"Skipping {LABEL_COLUMN_NAME} Grad-CAM: Model or validation dataset not available or val_ds is empty.")

# %%
print(f"{LABEL_COLUMN_NAME} {NUM_CLASSES}-class model script generation complete.")
print("IMPORTANT: Review and adjust 'LABEL_COLUMN_NAME', 'MAPPED_LABEL_COLUMN_NAME', and the 'map_her2_to_labels' function in the '读取标签并划分数据集' section according to your actual CSV data and HER-2 scoring criteria.")
print("Also, review FocalLoss 'alpha' parameters and class names ('HER2_CLASS_NAMES_FOR_PLOT') for testing.") 