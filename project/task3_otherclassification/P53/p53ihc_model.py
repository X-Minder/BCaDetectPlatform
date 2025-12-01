# %% [markdown]
# # P53-IHC 膀胱镜图像四分类模型
# 通过迁移学习（EfficientNet-B0）对 P53-IHC 进行四分类 (0/1/2/3)，按患者划分训练/验证/测试集。

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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.utils import make_grid, save_image
from sklearn.manifold import TSNE # Added for t-SNE
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score # Added for MI
from sklearn.utils import shuffle as sklearn_shuffle # Added for permutation test


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

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Focal Loss 定义
# %%
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean', num_classes=4): # Adjusted for multi-class (4 classes)
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        
        if isinstance(alpha, (float, int)):
             print(f"Warning: Single float alpha in FocalLoss for {num_classes} classes. Assuming equal weight or specific list needed. Using [alpha]*num_classes.")
             self.alpha = torch.tensor([alpha] * num_classes) # Default for multi-class if single float
        if isinstance(alpha, list): 
            self.alpha = torch.tensor(alpha)
            if len(self.alpha) != num_classes:
                print(f"Warning: Alpha list length {len(self.alpha)} does not match num_classes={num_classes}. Adjusting or erroring might be needed.")
                # Simple truncation/padding for now, or raise error:
                if len(self.alpha) < num_classes:
                    self.alpha = torch.cat([self.alpha, torch.full((num_classes - len(self.alpha),), 1.0/num_classes)]) # Pad with uniform
                else:
                    self.alpha = self.alpha[:num_classes]


        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (1-pt)**self.gamma * CE_loss

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            
            # For multi-class, targets are long, alpha should be indexed by target class
            at = self.alpha.gather(0, targets.data.view(-1))
            F_loss = at * F_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# %% [markdown]
# ## 自定义 Dataset for P53-IHC
# %%
class P53IHCDataset(Dataset): # Renamed from P53MutDataset
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row[FILE_NAME_COLUMN])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row[MAPPED_LABEL_COLUMN_NAME]) # Ensure this column holds 0, 1, 2, 3
        return img, label

# %% [markdown]
# ## 读取标签并划分数据集 for P53-IHC
# %%
LABEL_COLUMN_NAME = 'P53-IHC'  # Target column for P53-IHC status
MAPPED_LABEL_COLUMN_NAME = 'P53-IHC_mapped' # Processed multi-class label (0, 1, 2, or 3)
PATIENT_ID_COLUMN = 'PATIENT_ID'
FILE_NAME_COLUMN = 'FILE_NAME'
NUM_CLASSES = 4 # For P53-IHC multi-class classification

label_df = pd.read_csv("dataset/label.csv")
print(f"1. Initial rows loaded from CSV: {len(label_df)}")

# Drop rows with NA in critical columns (P53-IHC, FILE_NAME, PATIENT_ID)
label_df = label_df.dropna(subset=[LABEL_COLUMN_NAME, FILE_NAME_COLUMN, PATIENT_ID_COLUMN]).copy()
print(f"2. Rows after dropping NA from key columns ('{LABEL_COLUMN_NAME}', '{FILE_NAME_COLUMN}', '{PATIENT_ID_COLUMN}'): {len(label_df)}")

if len(label_df) == 0:
    print("ERROR: All rows were dropped after initial NA check for P53-IHC. Please check CSV.")
else:
    print(f"   Unique values in '{LABEL_COLUMN_NAME}' before mapping: {label_df[LABEL_COLUMN_NAME].astype(str).unique()}")

    def map_p53ihc_to_multiclass(p53ihc_status):
        if pd.isna(p53ihc_status): # 检查是否为 NaN 或 None
            # print(f"调试：P53-IHC 状态 '{p53ihc_status}' 为 NaN/None。映射为 NaN。") # 可选的调试信息
            return np.nan

        status_str = str(p53ihc_status).strip()
        if not status_str:  # 检查去除首尾空格后是否为空字符串
            # print(f"调试：P53-IHC 状态 '{p53ihc_status}' 变为空字符串。映射为 NaN。") # 可选的调试信息
            return np.nan
        
        try:
            # 先尝试转换为浮点数，再转换为整数
            status_val_float = float(status_str)
            status_val = int(status_val_float)
            
            # 检查转换后的整数是否与浮点数相等，确保没有因为截断导致值变化 (例如 "1.5" -> 1)
            if status_val != status_val_float:
                 print(f"警告：P53-IHC 状态 '{p53ihc_status}' (字符串形式：'{status_str}') 包含非零小数部分。映射为 NaN。")
                 return np.nan

            if status_val in [0, 1, 2, 3]:
                return status_val
            else:
                print(f"警告：意外的 P53-IHC 状态 '{p53ihc_status}' (解析为整数：{status_val})。不在 [0,1,2,3] 范围内。映射为 NaN。")
                return np.nan
        except ValueError:
            # 处理无法转换为浮点数的情况
            print(f"警告：无法将 P53-IHC 状态 '{p53ihc_status}' (字符串形式：'{status_str}') 转换为数字。映射为 NaN。")
            return np.nan

    label_df[MAPPED_LABEL_COLUMN_NAME] = label_df[LABEL_COLUMN_NAME].apply(map_p53ihc_to_multiclass)
    print(f"3. Rows after applying 'map_p53ihc_to_multiclass' (before dropping NA from '{MAPPED_LABEL_COLUMN_NAME}'): {len(label_df)}")
    print(f"   Unique values in '{MAPPED_LABEL_COLUMN_NAME}' after mapping (before dropping NA): {label_df[MAPPED_LABEL_COLUMN_NAME].unique()}")

    label_df = label_df.dropna(subset=[MAPPED_LABEL_COLUMN_NAME]).copy()
    print(f"4. Rows after dropping NA from '{MAPPED_LABEL_COLUMN_NAME}': {len(label_df)}")
    
    if len(label_df) > 0:
        label_df[MAPPED_LABEL_COLUMN_NAME] = label_df[MAPPED_LABEL_COLUMN_NAME].astype(int)
        print(f"   Final unique values in '{MAPPED_LABEL_COLUMN_NAME}': {label_df[MAPPED_LABEL_COLUMN_NAME].unique()}")
    else:
        print(f"ERROR: All rows were dropped after mapping P53-IHC. Check mapping logic and original '{LABEL_COLUMN_NAME}' values.")

if len(label_df) > 0:
    # Data splitting: Split into train and validation sets, no separate test set.
    # Aim for roughly 80% train, 20% validation, grouped by patient.
    gss_val_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    
    if len(label_df[PATIENT_ID_COLUMN].unique()) > 1 :
        try:
            train_idx, val_idx = next(gss_val_split.split(label_df, groups=label_df[PATIENT_ID_COLUMN]))
            df_train = label_df.iloc[train_idx].copy()
            df_val   = label_df.iloc[val_idx].copy()
            print("Successfully used GroupShuffleSplit to create train/validation sets for P53-IHC directly from label_df.")
        except ValueError as e:
            print(f"Warning: GroupShuffleSplit for P53-IHC train/validation failed: {e}. Falling back to random split without grouping if possible.")
            if len(label_df) > 1:
                 df_train, df_val = train_test_split(label_df, test_size=0.2, random_state=SEED, stratify=label_df[MAPPED_LABEL_COLUMN_NAME] if label_df[MAPPED_LABEL_COLUMN_NAME].nunique() > 1 else None)
                 print("Used random stratified split for train/validation as fallback for P53-IHC.")
            else: # Very few samples
                df_train = label_df.copy()
                df_val = pd.DataFrame(columns=label_df.columns)
                print("Warning: Not enough data in P53-IHC label_df for robust splitting. Validation set might be empty or very small.")
    elif len(label_df) > 0: # Only one patient group, or no groups identifiable for split
        print("Warning: Only one patient group or insufficient diversity for GroupShuffleSplit for P53-IHC. Attempting random split.")
        if len(label_df) > 1:
            df_train, df_val = train_test_split(label_df, test_size=0.2, random_state=SEED, stratify=label_df[MAPPED_LABEL_COLUMN_NAME] if label_df[MAPPED_LABEL_COLUMN_NAME].nunique() > 1 else None)
            print("Used random stratified split for train/validation for P53-IHC (single group scenario).")
            else:
            df_train = label_df.copy()
            df_val = pd.DataFrame(columns=label_df.columns)
            print("Warning: P53-IHC data is too small for splitting. Validation set is empty.")
    else: # label_df became empty before this point
        df_train = pd.DataFrame(columns=label_df.columns)
        df_val = pd.DataFrame(columns=label_df.columns)
        print("Error: P53-IHC label_df is empty before splitting. Train and Val sets are empty.")


    print("\nDataset sizes and class distributions (P53-IHC):")
    for name, df_subset in [("Train", df_train), ("Val", df_val)]: # Removed "Test"
        if not df_subset.empty:
            print(f"  {name:<8}: {len(df_subset):>4} images, Patients: {df_subset[PATIENT_ID_COLUMN].nunique():>2}")
            # Ensure the f-string for distribution_info is robust
            distribution_info_series = df_subset[MAPPED_LABEL_COLUMN_NAME].value_counts(normalize=True).sort_index()
            distribution_info_str = '\n'.join([f"    Class {idx}: {val:.4f}" for idx, val in distribution_info_series.items()])
            print(f"    Class distribution ({MAPPED_LABEL_COLUMN_NAME}, normalized):\n{distribution_info_str}")
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
else:
    print("Critical Error: P53-IHC label_df is empty after preprocessing. Cannot proceed.")
    df_train, df_val = pd.DataFrame(), pd.DataFrame() # Removed df_test initialization


# %% [markdown]
# ## 数据增强与 DataLoader for P53-IHC
# %%
IMG_DIR = "dataset/image" # Assuming same image directory

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

train_loader_args = {'shuffle': True} 
if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
    counts_train = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    if len(counts_train) >= 1 and len(counts_train) <= NUM_CLASSES : 
        class_sample_weights = [1. / counts_train.get(i, 1e-6) for i in range(NUM_CLASSES)] 
        sample_weights_train = [class_sample_weights[label] for label in df_train[MAPPED_LABEL_COLUMN_NAME]]
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights_train), 
                                        num_samples=len(sample_weights_train), 
                                        replacement=True)
        weights_str = ", ".join([f"Class {i}: {w:.4f}" for i, w in enumerate(class_sample_weights)])
        print(f"Sampler weights for P53-IHC classes: {weights_str}")
        train_loader_args = {'sampler': sampler, 'shuffle': False} # Shuffle is False when sampler is used
    else:
        print("Warning: Training data for P53-IHC has insufficient or unexpected class counts for sampler. Using standard DataLoader.")
else:
    print("Warning: df_train for P53-IHC is empty or mapped label column is missing. Using standard DataLoader.")
    if df_train.empty:
        train_loader_args['shuffle'] = False
        print("   df_train is empty. Forcing shuffle=False for train_loader to prevent error.")

train_ds = P53IHCDataset(df_train, IMG_DIR, transform=train_tf)
val_ds   = P53IHCDataset(df_val,   IMG_DIR, transform=val_tf)
# test_ds  = P53IHCDataset(df_test,  IMG_DIR, transform=val_tf) # Removed test_ds

train_loader = DataLoader(train_ds, batch_size=16, num_workers=0, pin_memory=True, **train_loader_args)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
# test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=0, pin_memory=True) # Removed test_loader

# %% [markdown]
# ## 模型定义与训练设置 for P53-IHC
# %%
print(f"Using EfficientNet-B0 for P53-IHC {NUM_CLASSES}-class classification")
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

# Adjust the classifier for EfficientNet-B0
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)

# FocalLoss alpha parameter for multi-class P53-IHC
focal_loss_alpha_values = [1.0/NUM_CLASSES] * NUM_CLASSES # Default equal weights
if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
    counts = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    if len(counts) == NUM_CLASSES: # All classes present
        # Inverse frequency for weights
        class_weights = [1.0 / counts.get(i, 1e-6) for i in range(NUM_CLASSES)]
        total_weight = sum(class_weights)
        focal_loss_alpha_values = [w / total_weight for w in class_weights] # Normalize to sum to 1 (optional, FocalLoss handles it)

        counts_str = ", ".join([f"Class {i}: {counts.get(i,0)}" for i in range(NUM_CLASSES)])
        print(f"Train data counts for P53-IHC: {counts_str}")
        alpha_str = ", ".join([f"{w:.4f}" for w in focal_loss_alpha_values])
        print(f"Calculated FocalLoss alpha for P53-IHC: [{alpha_str}]")
    else:
        print(f"Warning: Not all {NUM_CLASSES} classes present in P53-IHC training data ({len(counts)} found). Using default FocalLoss alpha.")
else:
    print(f"Warning: Could not calculate class counts for P53-IHC FocalLoss alpha. Using default: {focal_loss_alpha_values}")

focal_loss_alpha = torch.tensor(focal_loss_alpha_values, dtype=torch.float).to(device)
print(f"Using Focal Loss with alpha: {focal_loss_alpha.tolist()} and gamma=2 for P53-IHC")

criterion = FocalLoss(alpha=focal_loss_alpha, gamma=2, num_classes=NUM_CLASSES)
# criterion = nn.CrossEntropyLoss() # Standard alternative

optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True, min_lr=1e-6)

# %% [markdown]
# ## 早停机制 (Same as before)
# %%
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True, mode='min', verbose=True, task_name="Task"): # Added task_name
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        self.task_name = task_name # Store task_name
        
        if mode == 'max':
            self.monitor_op = lambda current, best: current > best + self.min_delta
            self.best_score = float('-inf')
        else:
            self.monitor_op = lambda current, best: current < best - self.min_delta
            self.best_score = float('inf')

    def __call__(self, score, model):
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                print(f"EarlyStopping: New best score for {self.task_name}: {self.best_score:.4f}") # Use task_name
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: Counter {self.counter}/{self.patience} for {self.task_name}. Best score: {self.best_score:.4f}") # Use task_name
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"EarlyStopping: Patience reached for {self.task_name}. Stopping training. Best score: {self.best_score:.4f}") # Use task_name
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print(f"EarlyStopping: Restored best {self.task_name} model weights.") # Use task_name
        return self.early_stop

# %% [markdown]
# ## 训练与验证循环 for P53-IHC
# %%
NUM_EPOCHS = 50
best_val_loss = float('inf') 
# Monitor validation loss for early stopping for P53-IHC
early_stopping = EarlyStopping(patience=10, mode='min', verbose=True, min_delta=0.0001, task_name=LABEL_COLUMN_NAME) # Pass task_name

history = {
    'train_loss': [], 'val_loss': [], 'val_accuracy': [], 
    'val_auc_ovr': [], 'val_auc_ovo': [], # For multi-class AUC (One-vs-Rest, One-vs-One)
    'lr': []
}

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train {LABEL_COLUMN_NAME}]"): # Use LABEL_COLUMN_NAME
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    
    train_loss = running_loss / len(train_ds) if len(train_ds) > 0 else 0.0
    history['train_loss'].append(train_loss)
    
    model.eval()
    all_val_labels = []
    all_val_probs = [] # Store full probability distributions for multi-class AUC
    all_val_preds = []
    val_running_loss = 0.0
    
    if len(val_ds) > 0:
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss_iter = criterion(outputs, labels)
                val_running_loss += val_loss_iter.item() * imgs.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_val_probs.extend(probs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
        
        val_epoch_loss = val_running_loss / len(val_ds)
        history['val_loss'].append(val_epoch_loss)

        all_val_labels_np = np.array(all_val_labels)
        all_val_preds_np = np.array(all_val_preds)
        all_val_probs_np = np.array(all_val_probs)

        val_accuracy = np.mean(all_val_labels_np == all_val_preds_np) if len(all_val_labels_np) > 0 else 0.0
        history['val_accuracy'].append(val_accuracy)

        val_auc_ovr, val_auc_ovo = 0.0, 0.0
        if len(np.unique(all_val_labels_np)) >= 2 and all_val_probs_np.shape[1] == NUM_CLASSES: # Needs at least 2 classes for AUC
            try:
                val_auc_ovr = roc_auc_score(all_val_labels_np, all_val_probs_np, multi_class='ovr', average='macro')
                if len(np.unique(all_val_labels_np)) >= NUM_CLASSES: # OVO needs all classes or will error if a class has no true samples
                     val_auc_ovo = roc_auc_score(all_val_labels_np, all_val_probs_np, multi_class='ovo', average='macro')
                else:
                    print(f"Warning: Epoch {epoch}, {LABEL_COLUMN_NAME} val set does not have all {NUM_CLASSES} classes for OVO AUC. OVO AUC set to 0.0")
            except ValueError as e_auc:
                 print(f"Warning: Epoch {epoch}, AUC calculation error for {LABEL_COLUMN_NAME}: {e_auc}. AUCs set to 0.0")
        elif len(all_val_labels_np) > 0:
             print(f"Warning: Epoch {epoch}, {LABEL_COLUMN_NAME} val set does not have enough distinct classes for AUC. AUCs set to 0.0")
        history['val_auc_ovr'].append(val_auc_ovr)
        history['val_auc_ovo'].append(val_auc_ovo)
        
        scheduler.step(val_epoch_loss)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_epoch_loss:.4f}, Val Acc={val_accuracy:.4f}, Val AUC(OvR)={val_auc_ovr:.4f}, Val AUC(OvO)={val_auc_ovo:.4f}, LR={optimizer.param_groups[0]['lr']:.1e}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), f"best_model_{LABEL_COLUMN_NAME}.pth") # Use LABEL_COLUMN_NAME
            print(f"Epoch {epoch}: New best {LABEL_COLUMN_NAME} model saved with Val Loss: {best_val_loss:.4f}") # Use LABEL_COLUMN_NAME
        
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        if early_stopping(val_epoch_loss, model):
            print(f"Early stopping triggered for {LABEL_COLUMN_NAME} based on validation loss.") # Use LABEL_COLUMN_NAME
            break
    else: # Validation dataset is empty
        history['val_loss'].append(float('nan'))
        history['val_accuracy'].append(float('nan'))
        history['val_auc_ovr'].append(float('nan'))
        history['val_auc_ovo'].append(float('nan'))
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss=N/A (empty val set for {LABEL_COLUMN_NAME}), LR={current_lr:.1e}") # Use LABEL_COLUMN_NAME


# %% [markdown]
# ## 绘制训练过程曲线 for P53-IHC
# %%
def plot_training_history(history, task_name): # Renamed and added task_name
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
    ax2.set_ylabel('Accuracy / AUC', color=color)  
    if 'val_accuracy' in history and any(not np.isnan(x) for x in history['val_accuracy']):
        ax2.plot(epoch_ticks, history['val_accuracy'], color=color, linestyle='-', marker='s', markersize=3, label='验证准确率')
    if 'val_auc_ovr' in history and any(not np.isnan(x) for x in history['val_auc_ovr']):
        ax2.plot(epoch_ticks, history['val_auc_ovr'], color='tab:purple', linestyle='--', marker='^', markersize=3, label='验证 AUC (OvR)')
    if 'val_auc_ovo' in history and any(not np.isnan(x) for x in history['val_auc_ovo']):
        ax2.plot(epoch_ticks, history['val_auc_ovo'], color='tab:cyan', linestyle='-.', marker='d', markersize=3, label='验证 AUC (OvO)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.05) # AUC and Acc are between 0 and 1

    ax3 = ax1.twinx() 
    ax3.spines["right"].set_position(("outward", 60)) 
    color = 'tab:green'
    ax3.set_ylabel('Learning Rate', color=color)
    if 'lr' in history and len(history['lr']) == epochs_ran:
        ax3.plot(epoch_ticks, history['lr'], color=color, linestyle='--', marker='.', markersize=3, label='学习率')
    elif 'lr' in history:
        print(f"Warning: LR history length ({len(history['lr'])}) doesn't match epochs_ran ({epochs_ran}) for {task_name}. LR plot skipped.") # Use task_name

    ax3.tick_params(axis='y', labelcolor=color)
    ax3.legend(loc='lower right')
    ax3.set_yscale('log') 

    fig.tight_layout()  
    plt.title(f'{task_name} 四分类训练过程监控') # Use task_name
    plt.xticks(epoch_ticks)
    plt.savefig(f"training_history_{task_name}.png") # Use task_name
    plt.show()

if any(history.values()):
    plot_training_history(history, LABEL_COLUMN_NAME) # Call with task_name
else:
    print(f"No training history to plot for {LABEL_COLUMN_NAME}.") # Use LABEL_COLUMN_NAME

# %% [markdown]
# ## 测试集评估 for P53-IHC
# %%
CLASS_NAMES = [f'{LABEL_COLUMN_NAME} Class {i}' for i in range(NUM_CLASSES)] # Use LABEL_COLUMN_NAME and NUM_CLASSES

try:
    model.load_state_dict(torch.load(f"best_model_{LABEL_COLUMN_NAME}.pth")) # Load LABEL_COLUMN_NAME model
    model.eval()
    print(f"Loaded best {LABEL_COLUMN_NAME} model for validation.") # Changed to validation

    all_probs_val = [] # Renamed from all_probs_test
    all_labels_val = [] # Renamed from all_labels_test
    all_preds_val = []  # Renamed from all_preds_test
    
    if len(val_ds) > 0: # Changed to val_ds
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Validating {LABEL_COLUMN_NAME}"): # Changed to val_loader and desc
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1) 
                preds = torch.argmax(probs, dim=1)
                
                all_probs_val.extend(probs.cpu().numpy())
                all_labels_val.extend(labels.numpy())
                all_preds_val.extend(preds.cpu().numpy())

        all_labels_val_np = np.array(all_labels_val) # Renamed
        all_probs_val_np = np.array(all_probs_val)   # Renamed
        all_preds_val_np = np.array(all_preds_val)     # Renamed

        print(f"=== Validation Classification Report {LABEL_COLUMN_NAME} ===") # Changed to Validation
        report = classification_report(all_labels_val_np, all_preds_val_np, target_names=CLASS_NAMES, zero_division=0, digits=4) # Use CLASS_NAMES
        cm = confusion_matrix(all_labels_val_np, all_preds_val_np)
        print(report)
        print(f"Confusion Matrix ({LABEL_COLUMN_NAME}, Validation):") # Added Validation
        print(cm)

        def plot_confusion_matrix(cm, class_names, task_name, set_name="Validation"): # Renamed and added task_name, set_name
            plt.figure(figsize=(NUM_CLASSES + 4, NUM_CLASSES + 2)) # Adjust size for more classes
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title(f'{task_name} {set_name} 混淆矩阵') # Use task_name and set_name
            plt.ylabel('实际标签')
            plt.xlabel('预测标签')
            plt.savefig(f"confusion_matrix_{task_name}_{set_name.lower()}.png") # Use task_name and set_name
            plt.show()

        plot_confusion_matrix(cm, CLASS_NAMES, LABEL_COLUMN_NAME, set_name="Validation") # Call with task_name

        if len(np.unique(all_labels_val_np)) >= 2 and all_probs_val_np.shape[1] == NUM_CLASSES: # Use val variables
            try:
                val_auc_ovr_final = roc_auc_score(all_labels_val_np, all_probs_val_np, multi_class='ovr', average='macro') # Renamed
                print(f"Final Validation AUC (OvR, macro) for {LABEL_COLUMN_NAME}: {val_auc_ovr_final:.4f}")
                val_auc_ovo_final = "N/A" # Renamed
                if len(np.unique(all_labels_val_np)) >= NUM_CLASSES:
                    val_auc_ovo_final = roc_auc_score(all_labels_val_np, all_probs_val_np, multi_class='ovo', average='macro')
                    print(f"Final Validation AUC (OvO, macro) for {LABEL_COLUMN_NAME}: {val_auc_ovo_final:.4f}")
                else:
                    print(f"Final Validation AUC (OvO) for {LABEL_COLUMN_NAME} not computed: not all classes present in validation set.")

                # ROC Curve for each class (One-vs-Rest) on Validation set
                plt.figure(figsize=(10, 8))
                for i in range(NUM_CLASSES):
                    fpr, tpr, _ = roc_curve(all_labels_val_np == i, all_probs_val_np[:, i])
                    roc_auc_class = roc_auc_score(all_labels_val_np == i, all_probs_val_np[:, i])
                    plt.plot(fpr, tpr, lw=2, label=f'{CLASS_NAMES[i]} ROC curve (AUC = {roc_auc_class:.3f})')
                
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('假阳性率 (False Positive Rate)')
                plt.ylabel('真阳性率 (True Positive Rate)')
                plt.title(f'验证集ROC曲线 ({LABEL_COLUMN_NAME}, One-vs-Rest)') # Changed to Validation
                plt.legend(loc="lower right")
                plt.grid(True)
                plt.savefig(f"roc_curve_{LABEL_COLUMN_NAME}_validation.png") # Changed to Validation
                plt.show()

                # Precision-Recall Curve for each class (One-vs-Rest) on Validation set
                plt.figure(figsize=(10, 8))
                for i in range(NUM_CLASSES):
                    precision, recall, _ = precision_recall_curve(all_labels_val_np == i, all_probs_val_np[:, i])
                    ap_score_class = average_precision_score(all_labels_val_np == i, all_probs_val_np[:, i])
                    plt.plot(recall, precision, lw=2, label=f'{CLASS_NAMES[i]} P-R curve (AP = {ap_score_class:.3f})')

                plt.xlabel('召回率 (Recall)')
                plt.ylabel('精确率 (Precision)')
                plt.title(f'验证集P-R曲线 ({LABEL_COLUMN_NAME}, One-vs-Rest)') # Changed to Validation
                plt.legend(loc="best") 
                plt.grid(True)
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.savefig(f"pr_curve_{LABEL_COLUMN_NAME}_validation.png") # Changed to Validation
                plt.show()

            except ValueError as e_auc_val: # Renamed
                print(f"Validation AUC/ROC/PR calculation error for {LABEL_COLUMN_NAME}: {e_auc_val}")
        else:
            print(f"Validation AUC/ROC/PR not computed for {LABEL_COLUMN_NAME}: validation set does not contain enough distinct classes or probability shape mismatch.")
    else:
        print(f"Validation dataset for {LABEL_COLUMN_NAME} is empty. No evaluation performed.") # Changed to Validation

except FileNotFoundError:
    print(f"Error: 'best_model_{LABEL_COLUMN_NAME}.pth' not found. Was the {LABEL_COLUMN_NAME} model trained and saved?")
except Exception as e:
    print(f"An error occurred during {LABEL_COLUMN_NAME} validation: {e}") # Changed to validation

# %% [markdown]
# ## Grad-CAM 可视化 for P53-IHC
# %%
# Grad-CAM: Visualize for multiple target classes if desired
def visualize_grad_cam_updated(model, dataset, device, task_name, class_names, num_images=None, target_classes_to_viz=None): # Renamed, added task_name, class_names. num_images default None
    if num_images is None: # Default to NUM_CLASSES images if not specified, one per class ideally
        num_images = NUM_CLASSES

    target_layer_name = 'features[-1]'
    try:
        module_path = target_layer_name.split('.')
        current_module = model
        for m_name in module_path:
            if m_name.isdigit(): current_module = current_module[int(m_name)]
            elif m_name.startswith('[') and m_name.endswith(']'): idx = int(m_name[1:-1]); current_module = current_module[idx]
            else: current_module = getattr(current_module, m_name)
        target_layers = [current_module]
    except Exception as e:
        print(f"Error finding target layer '{target_layer_name}' for P53-IHC Grad-CAM: {e}. Defaulting to model.features[-1].")
        target_layers = [model.features[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    if len(dataset) == 0:
        print(f"Dataset for {task_name} Grad-CAM is empty.") # Use task_name
        return

    if target_classes_to_viz is None:
        target_classes_to_viz = list(range(NUM_CLASSES)) # Default: try to show for all classes
    
    # Calculate images per class, trying to show at least one if possible, up to num_images total
    # If num_images = 4 and target_classes_to_viz = [0,1,2,3], then 1 image per class
    # If num_images = 2 and target_classes_to_viz = [0,1,2,3], still try for 1 per class if that many targets, but total will be capped by len(target_classes_to_viz) if it's less than num_images
    
    actual_num_images_per_target_class_requested = (num_images + len(target_classes_to_viz) - 1) // len(target_classes_to_viz) if len(target_classes_to_viz) > 0 else 0
    actual_num_images_per_target_class_requested = max(1, actual_num_images_per_target_class_requested) # Ensure at least 1 attempt per target class
    
    images_shown_count = 0

    num_viz_rows = len(target_classes_to_viz)
    num_viz_cols = actual_num_images_per_target_class_requested 
    
    if num_viz_rows * num_viz_cols == 0:
        print(f"No images or target classes specified for {task_name} Grad-CAM.") # Use task_name
        return

    fig, axes = plt.subplots(num_viz_rows * 2, num_viz_cols, figsize=(num_viz_cols * 3, num_viz_rows * 6), squeeze=False) # squeeze=False for consistent 2D array

    # Ensure we don't try to pick more unique images than available in dataset per class or in total
    # This part can be complex to ensure unique images per class. 
    # Simple approach: pick num_viz_cols * num_viz_rows random unique images from the dataset if available.
    # Or, pick `actual_num_images_per_target_class_requested` for each target class.
    
    # We will pick `num_viz_cols` distinct images from the dataset
    # These images will be used across all target_classes_to_viz rows
    if len(dataset) < num_viz_cols:
        print(f"Warning: Requested {num_viz_cols} images for Grad-CAM for {task_name}, but dataset only has {len(dataset)}. Using all available.")
        indices_to_use = np.arange(len(dataset))
        num_viz_cols = len(dataset) # Adjust num_viz_cols
        if num_viz_cols == 0:
             print(f"No images in dataset for {task_name} Grad-CAM.")
             plt.close(fig)
             return
        # Re-create axes if num_viz_cols changed and was > 0
        if num_viz_rows > 0 :
             plt.close(fig) # Close the old figure
             fig, axes = plt.subplots(num_viz_rows * 2, num_viz_cols, figsize=(num_viz_cols * 3, num_viz_rows * 6), squeeze=False)

    else:
        indices_to_use = np.random.choice(len(dataset), num_viz_cols, replace=False)


    for r_idx, target_cls in enumerate(target_classes_to_viz):
        if target_cls >= NUM_CLASSES:
            print(f"Warning: Target class {target_cls} is out of bounds for {task_name} ({NUM_CLASSES} classes). Skipping.")
            # Turn off axes for this row if it was created
            for c_col_ax in range(num_viz_cols):
                 axes[r_idx * 2, c_col_ax].axis('off')
                 axes[r_idx * 2 + 1, c_col_ax].axis('off')
            continue

        for c_idx_img_col in range(num_viz_cols): # Iterate through the chosen image columns
            img_idx_in_dataset = indices_to_use[c_idx_img_col]
            img_tensor, true_label = dataset[img_idx_in_dataset] 
            
            inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
            rgb_img_denorm = inv_normalize(img_tensor.clone()).permute(1, 2, 0).cpu().numpy()
            rgb_img_denorm = np.clip(rgb_img_denorm, 0, 1) 

            input_tensor_unsqueeze = img_tensor.unsqueeze(0).to(device)
            cam_targets = [ClassifierOutputTarget(target_cls)]
            
            grayscale_cam = cam(input_tensor=input_tensor_unsqueeze, targets=cam_targets)
            if grayscale_cam is not None and grayscale_cam.shape[0] > 0:
                grayscale_cam_batch = grayscale_cam[0, :]
            else:
                print(f"Warning: {task_name} Grad-CAM returned None or empty for image index {img_idx_in_dataset}, target class {target_cls}.") # Use task_name
                axes[r_idx * 2, c_idx_img_col].axis('off')
                axes[r_idx * 2 + 1, c_idx_img_col].axis('off')
                continue
            
            cam_image = show_cam_on_image(rgb_img_denorm, grayscale_cam_batch, use_rgb=True)
            cam_image_tensor = transforms.ToTensor()(cam_image) 
            original_img_for_grid = transforms.ToTensor()(Image.fromarray((rgb_img_denorm * 255).astype(np.uint8)))
            
            title_str = f"""True: {class_names[true_label]}
CAM for: {class_names[target_cls]}""" # Use class_names

            ax_orig_current = axes[r_idx * 2, c_idx_img_col]
            ax_cam_current = axes[r_idx * 2 + 1, c_idx_img_col]

            ax_orig_current.imshow(original_img_for_grid.permute(1,2,0).numpy())
            ax_orig_current.set_title(title_str, fontsize=8)
            ax_orig_current.axis('off')
            ax_cam_current.imshow(cam_image_tensor.permute(1,2,0).numpy())
            ax_cam_current.axis('off')
            images_shown_count +=1

    if images_shown_count == 0:
        print(f"No {task_name} CAM images were generated. Check dataset or chosen indices/classes.") # Use task_name
        plt.close(fig) # Close empty figure
        return

    fig.suptitle(f"Grad-CAM for {task_name} Model (Targeting Various Classes)", fontsize=12) # Use task_name
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_filename = f"grad_cam_{task_name}_multiclass.png" # Use task_name
    plt.savefig(save_filename)
    print(f"Grad-CAM grid for {task_name} saved to {save_filename}") # Use task_name
    plt.show()


if 'model' in locals() and 'val_ds' in locals() and len(val_ds) > 0: # Changed to val_ds
    print(f"Visualizing Grad-CAM for {LABEL_COLUMN_NAME} model on validation set") # Use LABEL_COLUMN_NAME
    # Visualize CAM for all classes if possible, using up to NUM_CLASSES images in total.
    # target_classes_to_viz can be a subset e.g. [0, 1] if you only want to see for specific classes
    visualize_grad_cam_updated(model, dataset=val_ds, device=device, task_name=LABEL_COLUMN_NAME, class_names=CLASS_NAMES, num_images=NUM_CLASSES, target_classes_to_viz=list(range(NUM_CLASSES))) 
else:
    print(f"Skipping {LABEL_COLUMN_NAME} Grad-CAM: Model or validation dataset not available or val_ds is empty.") # Use LABEL_COLUMN_NAME

# %% [markdown]
# ## 特征分析辅助函数 (for P53-IHC)
# %%
def get_embeddings_efficientnet(model, dataloader, device, task_name):
    model.eval()
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=f"Extracting embeddings for {task_name}"):
            imgs = imgs.to(device)
            # EfficientNet structure: model.features -> model.avgpool -> model.classifier
            x = model.features(imgs)
            x = model.avgpool(x)
            embeddings = torch.flatten(x, 1)
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.numpy()) 
    
    if not embeddings_list: 
        return np.array([]), np.array([])
        
    return np.concatenate(embeddings_list), np.concatenate(labels_list)

def calculate_mutual_information_multiclass_discrete(true_labels, predicted_labels, task_name):
    if len(true_labels) == 0 or len(predicted_labels) == 0:
        print(f"Cannot calculate MI for {task_name}: empty labels or predictions.")
        return float('nan'), float('nan')
    if len(true_labels) != len(predicted_labels):
        print(f"Cannot calculate MI for {task_name}: mismatched lengths of true_labels ({len(true_labels)}) and predicted_labels ({len(predicted_labels)}).")
        return float('nan'), float('nan')
    try:
        mi = mutual_info_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        print(f"Mutual Information (True Labels vs Predicted Labels) for {task_name}: {mi:.4f}")
        print(f"Normalized Mutual Information for {task_name}: {nmi:.4f}")
        return mi, nmi
    except Exception as e:
        print(f"Error calculating MI for {task_name}: {e}")
        return float('nan'), float('nan')

def plot_tsne_visualization_multiclass(embeddings, labels, class_names_list, task_name, num_classes):
    if len(embeddings) == 0:
        print(f"No embeddings to visualize for {task_name} t-SNE.")
        return
    print(f"Running t-SNE for {task_name} ({len(embeddings)} samples)...")
    perplexity_val = min(30, len(embeddings) - 1)
    if perplexity_val <=0:
        print(f"Not enough samples ({len(embeddings)}) for t-SNE. Perplexity would be <=0. Skipping t-SNE for {task_name}.")
        return

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity_val, n_iter=1000, init='pca', learning_rate='auto')
    try:
        embeddings_2d = tsne.fit_transform(embeddings)
    except Exception as e:
        print(f"Error during t-SNE fitting for {task_name}: {e}. Skipping t-SNE plot.")
        return
    
    plt.figure(figsize=(12, 10))
    unique_labels_in_data = np.unique(labels)
    
    colors_map_name = "tab10" if num_classes <= 10 else "viridis"
    colors = plt.cm.get_cmap(colors_map_name, num_classes)

    for i in range(num_classes): 
        if i in unique_labels_in_data:
            idx = labels == i
            if np.sum(idx) > 0: 
                plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], color=colors(i), label=class_names_list[i], alpha=0.7)
    
    plt.title(f't-SNE 可视化 - {task_name} (验证集)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f"tsne_visualization_{task_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_validation.png")
    plt.show()

def simulate_data_cleaning_multiclass(model_probs_val_np, true_labels_val_np, num_classes, task_name, n_samples_to_flip_ratio=0.1):
    num_samples = len(true_labels_val_np)
    if num_samples == 0:
        print(f"No samples for data cleaning simulation for {task_name}.")
        return float('nan')
        
    num_samples_to_flip = int(num_samples * n_samples_to_flip_ratio)
    if num_samples_to_flip == 0:
        print(f"Number of samples to flip is 0 for {task_name} (ratio: {n_samples_to_flip_ratio}, total samples: {num_samples}). Skipping simulation.")
        return float('nan')

    print(f"\nSimulating data cleaning for {task_name} by flipping {num_samples_to_flip} labels (ratio {n_samples_to_flip_ratio:.2f})...")

    flipped_labels_np = true_labels_val_np.copy()
    if num_samples_to_flip > num_samples: # Should not happen if ratio is <=1
        num_samples_to_flip = num_samples
        
    indices_to_flip = np.random.choice(num_samples, num_samples_to_flip, replace=False)
    
    for idx in indices_to_flip:
        original_label = flipped_labels_np[idx]
        possible_new_labels = [l for l in range(num_classes) if l != original_label]
        if not possible_new_labels and num_classes > 1: 
            print(f"Warning: For sample {idx} with label {original_label}, no other labels possible to flip to. This is unexpected for num_classes > 1.")
            continue
        elif not possible_new_labels: # Only one class
             continue
        flipped_labels_np[idx] = np.random.choice(possible_new_labels)
            
    if len(np.unique(flipped_labels_np)) < 2 : 
        print(f"Warning: After flipping, less than 2 unique classes present in simulated labels for {task_name}. Multi-class AUC may be ill-defined or error.")
        
    try:
        auc_after_cleaning = roc_auc_score(flipped_labels_np, model_probs_val_np, multi_class='ovr', average='macro')
        print(f"Macro OvR AUC after simulated cleaning for {task_name} ({num_samples_to_flip} labels flipped): {auc_after_cleaning:.4f}")
        return auc_after_cleaning
    except ValueError as e:
        print(f"Error calculating Macro OvR AUC after simulated cleaning for {task_name}: {e}")
        return float('nan')

def perform_permutation_test_multiclass(model_probs_val_np, true_labels_val_np, num_classes, task_name, n_permutations=1000):
    if len(true_labels_val_np) == 0:
        print(f"No samples for permutation test for {task_name}.")
        return float('nan')
        
    print(f"\nPerforming permutation test for {task_name} with {n_permutations} permutations...")
    
    observed_auc = float('nan')
    if len(np.unique(true_labels_val_np)) < 2: 
        print(f"Warning: Original labels for {task_name} have less than 2 unique classes. Observed AUC might be ill-defined.")
    else:
        try:
            observed_auc = roc_auc_score(true_labels_val_np, model_probs_val_np, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"Could not calculate observed Macro OvR AUC for {task_name}: {e}. Permutation test results might be unreliable.")
            
    if np.isnan(observed_auc):
         print(f"Observed Macro OvR AUC for {task_name} is undefined. Skipping permutation test plot/p-value.")
         return float('nan')

    print(f"Observed Macro OvR AUC for {task_name}: {observed_auc:.4f}")
    
    permuted_aucs = []
    for i in tqdm(range(n_permutations), desc=f"Permutation Test {task_name}"):
        permuted_labels = sklearn_shuffle(true_labels_val_np, random_state=SEED + i)
        current_auc = float('nan')
        if len(np.unique(permuted_labels)) < 2: 
            pass # Keep current_auc as nan
        else:
            try:
                current_auc = roc_auc_score(permuted_labels, model_probs_val_np, multi_class='ovr', average='macro')
            except ValueError:
                 pass # Keep current_auc as nan
        permuted_aucs.append(current_auc)

    permuted_aucs_np = np.array(permuted_aucs)
    permuted_aucs_valid = permuted_aucs_np[~np.isnan(permuted_aucs_np)] 

    if len(permuted_aucs_valid) == 0:
        print(f"No valid permuted AUCs were calculated for {task_name}. Cannot compute p-value.")
        p_value = float('nan')
    else:
        p_value = np.mean(permuted_aucs_valid >= observed_auc)
        print(f"Permutation test for {task_name}: p-value = {p_value:.4f} (based on {len(permuted_aucs_valid)} valid permutations)")
    
    if len(permuted_aucs_valid) > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(permuted_aucs_valid, bins=30, kde=False, label='Permuted Macro OvR AUCs', stat="density") 
        plt.axvline(observed_auc, color='red', linestyle='--', lw=2, label=f'Observed AUC ({observed_auc:.3f})')
        plt.title(f'Permutation Test Results for {task_name} (Validation Set)')
        plt.xlabel('Macro OvR AUC Score')
        plt.ylabel('Density') 
        plt.legend()
        plt.savefig(f"permutation_test_{task_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_validation.png")
        plt.show()
    else:
        print(f"Skipping permutation test plot for {task_name} as no valid permuted AUCs were generated.")
        
    return p_value

# %% [markdown]
# ## 特征与标签相关性分析 (验证集) for P53-IHC
# %%
# --- Main execution of the new analyses for P53-IHC ---
if 'model' in locals() and 'val_loader' in locals() and 'val_ds' in locals() and len(val_ds) > 0 and \
   'all_labels_val_np' in globals() and 'all_preds_val_np' in globals() and 'all_probs_val_np' in globals() and \
   all_labels_val_np is not None and all_preds_val_np is not None and all_probs_val_np is not None and \
   len(all_labels_val_np) == len(val_ds) and 'CLASS_NAMES' in globals() and CLASS_NAMES is not None and 'NUM_CLASSES' in globals() and NUM_CLASSES is not None:

    print(f"\n--- Starting Feature-Label Relevance Analysis for {LABEL_COLUMN_NAME} (Validation Set) ---")
    
    model.eval() 

    val_embeddings, val_labels_from_embedding_extraction = get_embeddings_efficientnet(model, val_loader, device, LABEL_COLUMN_NAME)
    
    if not np.array_equal(val_labels_from_embedding_extraction, all_labels_val_np):
        print(f"Warning: Labels from embedding extraction ({len(val_labels_from_embedding_extraction)}) do not perfectly match global 'all_labels_val_np' ({len(all_labels_val_np)}) for {LABEL_COLUMN_NAME}. Using 'all_labels_val_np' from main validation for consistency in this block.")

    if val_embeddings is not None and len(val_embeddings) > 0 and len(val_labels_from_embedding_extraction) == len(val_embeddings):
        analysis_true_labels = all_labels_val_np

        print("\nCalculating Mutual Information...")
        mi_discrete, nmi_discrete = calculate_mutual_information_multiclass_discrete(
            true_labels=analysis_true_labels, 
            predicted_labels=all_preds_val_np, 
            task_name=LABEL_COLUMN_NAME
        )

        print("\nGenerating t-SNE plot...")
        plot_tsne_visualization_multiclass(
            embeddings=val_embeddings, 
            labels=analysis_true_labels, 
            class_names_list=CLASS_NAMES, 
            task_name=LABEL_COLUMN_NAME,
            num_classes=NUM_CLASSES
        )

        print("\nSimulating data cleaning...")
        auc_after_simulated_cleaning = simulate_data_cleaning_multiclass(
            model_probs_val_np=all_probs_val_np, 
            true_labels_val_np=analysis_true_labels,
            num_classes=NUM_CLASSES,
            task_name=LABEL_COLUMN_NAME,
            n_samples_to_flip_ratio=0.1 
        )
        if not np.isnan(auc_after_simulated_cleaning):
            original_auc_ovr = history.get('val_auc_ovr', [])[-1] if history.get('val_auc_ovr') else float('nan')
            if 'val_auc_ovr_final' in locals():
                 original_auc_ovr = val_auc_ovr_final

            if not np.isnan(original_auc_ovr):
                 print(f"Original Val Macro OvR AUC for {LABEL_COLUMN_NAME}: {original_auc_ovr:.4f}")
            print(f"Val Macro OvR AUC for {LABEL_COLUMN_NAME} after simulated cleaning: {auc_after_simulated_cleaning:.4f}")

        print("\nPerforming permutation test...")
        p_value_permutation = perform_permutation_test_multiclass(
            model_probs_val_np=all_probs_val_np, 
            true_labels_val_np=analysis_true_labels,
            num_classes=NUM_CLASSES,
            task_name=LABEL_COLUMN_NAME,
            n_permutations=1000 
        )
        if not np.isnan(p_value_permutation):
            print(f"Permutation test p-value for {LABEL_COLUMN_NAME} (Validation Macro OvR AUC): {p_value_permutation:.4f}")

    else:
        print(f"Could not extract embeddings or labels for {LABEL_COLUMN_NAME} analysis. Skipping feature-label relevance block.")
        
else:
    print(f"Skipping Feature-Label Relevance Analysis for {LABEL_COLUMN_NAME}: Model, validation data, prior evaluation results (all_labels_val_np etc.), CLASS_NAMES, or NUM_CLASSES not available/defined.")


# %%
print(f"{LABEL_COLUMN_NAME} multi-class classification model script generation complete.")
# ... rest of the script ...
