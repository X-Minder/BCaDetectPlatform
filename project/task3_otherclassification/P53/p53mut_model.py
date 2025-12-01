# %% [markdown]
# # P53-MUT 膀胱镜图像二分类模型
# 通过迁移学习（EfficientNet-B0）对 P53-MUT 进行二分类 (N/Y)，按患者划分训练/验证/测试集。

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
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.utils import shuffle as sklearn_shuffle


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
    def __init__(self, alpha=None, gamma=2., reduction='mean', num_classes=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        
        # Consistent alpha handling from grade_model.py / pdl1_atezo_model.py
        if self.num_classes == 2:
            if isinstance(alpha, list) and len(alpha)==2:
                 self.alpha = torch.tensor(alpha)
            elif isinstance(alpha, (float, int)):
                 # For binary, if a single float is given, it could be ambiguous.
                 # We previously calculated focal_loss_alpha_values as a list [weight_neg, weight_pos]
                 # This path is more for a generic FocalLoss initialization.
                 # To align with grade_model, if it's a single float, it was [alpha, alpha]
                 # For P53, we already calculate a list, so this specific path might not be hit directly
                 # but is here for completeness of the class definition.
                 self.alpha = torch.tensor([alpha, alpha]) 
                 # print(f"Warning: FocalLoss init with single float alpha for binary. Interpreted as [{alpha}, {alpha}].")
        elif isinstance(alpha, (float, int)): 
             self.alpha = torch.tensor([alpha] * num_classes)
             # print(f"Warning: FocalLoss init with single float alpha for {num_classes} classes. Interpreted as equal weight.")
        
        if isinstance(alpha, list): 
            self.alpha = torch.tensor(alpha)
            if len(self.alpha) != num_classes:
                # print(f"Warning: Alpha list length {len(self.alpha)} does not match num_classes={num_classes}. Adjusting.")
                if len(self.alpha) < num_classes:
                    self.alpha = torch.cat([self.alpha, torch.full((num_classes - len(self.alpha),), 1.0/num_classes)])
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
            
            at = self.alpha.gather(0, targets.data.view(-1))
            F_loss = at * F_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# %% [markdown]
# ## 自定义 Dataset for P53-MUT
# %%
class P53MutDataset(Dataset):
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
        label = int(row[MAPPED_LABEL_COLUMN_NAME])
        return img, label

# %% [markdown]
# ## 读取标签并划分数据集 for P53-MUT
# %%
LABEL_COLUMN_NAME = 'P53-MUT'  # Target column for P53-MUT status
MAPPED_LABEL_COLUMN_NAME = 'P53-MUT_binary' # Processed binary label (0 or 1)
PATIENT_ID_COLUMN = 'PATIENT_ID'
FILE_NAME_COLUMN = 'FILE_NAME'

label_df = pd.read_csv("dataset/label.csv")
print(f"1. Initial rows loaded from CSV: {len(label_df)}")

# Drop rows with NA in critical columns (P53-MUT, FILE_NAME, PATIENT_ID)
label_df = label_df.dropna(subset=[LABEL_COLUMN_NAME, FILE_NAME_COLUMN, PATIENT_ID_COLUMN]).copy()
print(f"2. Rows after dropping NA from key columns ('{LABEL_COLUMN_NAME}', '{FILE_NAME_COLUMN}', '{PATIENT_ID_COLUMN}'): {len(label_df)}")

if len(label_df) == 0:
    print("ERROR: All rows were dropped after initial NA check. Please check CSV.")
else:
    print(f"   Unique values in '{LABEL_COLUMN_NAME}' before mapping: {label_df[LABEL_COLUMN_NAME].astype(str).unique()}")

    def map_p53_to_binary(p53_status):
        s_status = str(p53_status).strip().upper() # Standardize to upper
        if s_status == 'N':
            return 0 # Negative / Wild-type
        elif s_status == 'Y':
            return 1 # Positive / Mutant
        else:
            print(f"Warning: Unexpected P53-MUT status '{p53_status}'. Mapping to NaN.")
            return np.nan

    label_df[MAPPED_LABEL_COLUMN_NAME] = label_df[LABEL_COLUMN_NAME].apply(map_p53_to_binary)
    print(f"3. Rows after applying 'map_p53_to_binary' (before dropping NA from '{MAPPED_LABEL_COLUMN_NAME}'): {len(label_df)}")
    print(f"   Unique values in '{MAPPED_LABEL_COLUMN_NAME}' after mapping (before dropping NA): {label_df[MAPPED_LABEL_COLUMN_NAME].unique()}")

    label_df = label_df.dropna(subset=[MAPPED_LABEL_COLUMN_NAME]).copy()
    print(f"4. Rows after dropping NA from '{MAPPED_LABEL_COLUMN_NAME}': {len(label_df)}")
    
    if len(label_df) > 0:
        label_df[MAPPED_LABEL_COLUMN_NAME] = label_df[MAPPED_LABEL_COLUMN_NAME].astype(int)
        print(f"   Final unique values in '{MAPPED_LABEL_COLUMN_NAME}': {label_df[MAPPED_LABEL_COLUMN_NAME].unique()}")
    else:
        print(f"ERROR: All rows were dropped after mapping P53-MUT. Check mapping logic and original values.")

if len(label_df) > 0:
    # Data splitting: Remove dedicated test set, split all into train/val
    df_trainval = label_df.copy()
    # df_test     = label_df.iloc[test_idx].copy() # Removed test set

    val_size_from_trainval = 0.2 # Keep this proportion for validation
    # Use GroupShuffleSplit for val, similar to pdl1_atezo_model.py and grade_model.py logic for val split
    # Fallback to train_test_split if GroupShuffleSplit fails

    if len(df_trainval[PATIENT_ID_COLUMN].unique()) > 1 and len(df_trainval) > 1:
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_from_trainval, random_state=SEED + 1)
        try:
            train_idx_inner, val_idx_inner = next(gss_val.split(df_trainval, groups=df_trainval[PATIENT_ID_COLUMN]))
            df_train = df_trainval.iloc[train_idx_inner].copy()
            df_val   = df_trainval.iloc[val_idx_inner].copy()
            print(f"Successfully used GroupShuffleSplit for train/validation sets for {LABEL_COLUMN_NAME}.")
        except ValueError as e:
            print(f"Warning: GroupShuffleSplit for {LABEL_COLUMN_NAME} train/validation failed: {e}. Falling back to random split on the full dataset.")
            stratify_col = df_trainval[MAPPED_LABEL_COLUMN_NAME] if df_trainval[MAPPED_LABEL_COLUMN_NAME].nunique() > 1 else None
            df_train, df_val = train_test_split(df_trainval, test_size=val_size_from_trainval, random_state=SEED + 1, stratify=stratify_col)
    elif len(df_trainval) > 0:
        print(f"Warning: Not enough unique patient groups or samples in {LABEL_COLUMN_NAME} data for GroupShuffleSplit. Using random split or assigning all to train.")
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
        print(f"Warning: {LABEL_COLUMN_NAME} data is empty. Train and Val sets are empty.")

    print(f"\nDataset sizes and class distributions ({LABEL_COLUMN_NAME}):")
    for name, df_subset in [("Train", df_train), ("Val", df_val)]: # Removed "Test"
        if not df_subset.empty:
            print(f"  {name:<8}: {len(df_subset):>4} images, Patients: {df_subset[PATIENT_ID_COLUMN].nunique():>2}")
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
    print(f"Critical Error: {LABEL_COLUMN_NAME} label_df is empty after preprocessing. Cannot proceed.")
    df_train, df_val = pd.DataFrame(), pd.DataFrame() # Initialize empty to prevent errors if script continues
    # df_test = pd.DataFrame() # Removed df_test initialization here as well


# %% [markdown]
# ## 数据增强与 DataLoader for P53-MUT
# %%
IMG_DIR = "dataset/image"

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

train_loader_args = {'shuffle': True} # Default
if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
    counts_train = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    if len(counts_train) == 2 : # Binary specific
        class_sample_weights = [1. / counts_train.get(i, 1e-6) for i in range(2)] # For class 0 and 1
        sample_weights_train = [class_sample_weights[label] for label in df_train[MAPPED_LABEL_COLUMN_NAME]]
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights_train), 
                                        num_samples=len(sample_weights_train), 
                                        replacement=True)
        print(f"Sampler weights for P53-MUT classes: Class 0: {class_sample_weights[0]:.4f}, Class 1: {class_sample_weights[1]:.4f}")
        train_loader_args = {'sampler': sampler, 'shuffle': False}
    elif len(counts_train) == 1:
        print("Warning: Training data has only one class for P53-MUT. Using standard DataLoader without sampler.")
    else: # Empty or unexpected
        print("Warning: Training data for P53-MUT is empty or has no class labels for sampler. Using standard DataLoader.")
else:
    print("Warning: df_train is empty or MAPPED_LABEL_COLUMN_NAME for P53-MUT is missing. Using standard DataLoader.")

train_ds = P53MutDataset(df_train, IMG_DIR, transform=train_tf)
val_ds   = P53MutDataset(df_val,   IMG_DIR, transform=val_tf)
# test_ds  = P53MutDataset(df_test,  IMG_DIR, transform=val_tf) # Removed test_ds

train_loader = DataLoader(train_ds, batch_size=16, num_workers=0, pin_memory=True, **train_loader_args)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
# test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=0, pin_memory=True) # Removed test_loader

# %% [markdown]
# ## 模型定义与训练设置 for P53-MUT
# %%
NUM_CLASSES = 2 # For P53-MUT binary classification

print(f"Using EfficientNet-B0 for P53-MUT {NUM_CLASSES}-class classification")
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

# FocalLoss alpha parameter for binary P53-MUT
focal_loss_alpha_values = [0.5, 0.5] # Default equal weights for N/Y
if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
    counts = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    if len(counts) == NUM_CLASSES: # Both classes present
        # Inverse frequency for [class_N_weight, class_Y_weight]
        # focal_loss_alpha_values = [(1. / counts[0]), (1. / counts[1])]
        # total_weight = sum(focal_loss_alpha_values)
        # focal_loss_alpha_values = [w / total_weight for w in focal_loss_alpha_values]
        # Simpler: if class 1 (Y - Mutant) is rarer, give it more weight
        # Example: alpha for positive class (Y). Suppose Y is class 1.
        # Let alpha_pos = count_N / (count_N + count_Y), alpha_neg = count_Y / (count_N + count_Y)
        # This makes alpha = [alpha_neg, alpha_pos]
        alpha_pos = counts.get(0, 1e-6) / (counts.get(0,1e-6) + counts.get(1,1e-6)) # Weight for Y is prop of N
        alpha_neg = 1.0 - alpha_pos
        focal_loss_alpha_values = [alpha_neg, alpha_pos]

        print(f"Train data counts for P53-MUT: N(0): {counts.get(0,0)}, Y(1): {counts.get(1,0)}")
        print(f"Calculated FocalLoss alpha for P53-MUT: [N_weight, Y_weight] = {focal_loss_alpha_values}")
    elif len(counts) == 1:
        cls_present = counts.index[0]
        print(f"Warning: Only one class ({cls_present}) present in P53-MUT training data. Using default FocalLoss alpha.")
else:
    print(f"Warning: Could not calculate class counts for P53-MUT FocalLoss alpha. Using default: {focal_loss_alpha_values}")

focal_loss_alpha = torch.tensor(focal_loss_alpha_values, dtype=torch.float).to(device)
print(f"Using Focal Loss with alpha: {focal_loss_alpha.tolist()} and gamma=2 for P53-MUT")

criterion = FocalLoss(alpha=focal_loss_alpha, gamma=2, num_classes=NUM_CLASSES)
# For binary, can also use nn.BCEWithLogitsLoss if model outputs 1 logit and labels are float [0,1]
# criterion = nn.CrossEntropyLoss() # if NUM_CLASSES=2 and output is 2 logits

optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True, min_lr=1e-6)

# %% [markdown]
# ## 早停机制 (Same as HER-2)
# %%
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True, mode='min', verbose=True, task_name="Task"):
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
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                print(f"EarlyStopping ({self.task_name}): New best score: {self.best_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping ({self.task_name}): Counter {self.counter}/{self.patience}. Best score: {self.best_score:.4f}")
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"EarlyStopping ({self.task_name}): Patience reached. Stopping training. Best score: {self.best_score:.4f}")
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print(f"EarlyStopping ({self.task_name}): Restored best model weights.")
        return self.early_stop

# %% [markdown]
# ## 训练与验证循环 for P53-MUT
# %%
NUM_EPOCHS = 50
best_val_loss = float('inf') 
# Initialize EarlyStopping with task_name and mode='min' for loss
early_stopping = EarlyStopping(patience=10, mode='min', verbose=True, min_delta=0.0001, task_name=LABEL_COLUMN_NAME) 

history = {
    'train_loss': [], 'val_loss': [], 'val_accuracy': [], 
    'val_auc': [], 
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
    all_val_probs_pos_class = [] 
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
                
                all_val_probs_pos_class.extend(probs[:, 1].cpu().numpy()) 
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
        
        val_epoch_loss = val_running_loss / len(val_ds)
        history['val_loss'].append(val_epoch_loss)

        all_val_labels_np = np.array(all_val_labels)
        all_val_preds_np = np.array(all_val_preds)
        all_val_probs_pos_class_np = np.array(all_val_probs_pos_class)

        val_accuracy = np.mean(all_val_labels_np == all_val_preds_np) if len(all_val_labels_np) > 0 else 0.0
        history['val_accuracy'].append(val_accuracy)

        val_auc = 0.0
        if len(np.unique(all_val_labels_np)) >= NUM_CLASSES: # Use NUM_CLASSES constant
             try:
             val_auc = roc_auc_score(all_val_labels_np, all_val_probs_pos_class_np)
             except ValueError as e_auc:
                 print(f"Warning: Epoch {epoch}, AUC calculation error for {LABEL_COLUMN_NAME}: {e_auc}. AUC set to 0.0")
        elif len(all_val_labels_np) > 0 :
             print(f"Warning: Epoch {epoch}, Validation set for {LABEL_COLUMN_NAME} does not have enough classes ({np.unique(all_val_labels_np)}) for AUC. AUC set to 0.0")
        history['val_auc'].append(val_auc)
        
        scheduler.step(val_epoch_loss) # Scheduler monitors val_loss
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_epoch_loss:.4f}, Val Acc={val_accuracy:.4f}, Val AUC={val_auc:.4f}, LR={optimizer.param_groups[0]['lr']:.1e}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), f"best_model_{LABEL_COLUMN_NAME.lower()}.pth") # Use LABEL_COLUMN_NAME
            print(f"Epoch {epoch}: New best {LABEL_COLUMN_NAME} model saved with Val Loss: {best_val_loss:.4f}") # Use LABEL_COLUMN_NAME
        
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        if early_stopping(val_epoch_loss, model):
            print(f"Early stopping triggered for {LABEL_COLUMN_NAME} based on validation loss.") # Use LABEL_COLUMN_NAME
            break
    else: 
        history['val_loss'].append(float('nan'))
        history['val_accuracy'].append(float('nan'))
        history['val_auc'].append(float('nan'))
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss=N/A (empty val set for {LABEL_COLUMN_NAME}), LR={current_lr:.1e}")


# %% [markdown]
# ## 绘制训练过程曲线 for P53-MUT
# %%
def plot_training_history_p53(history): # Renamed to be specific, and align with grade_model style
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
    if 'val_auc' in history and any(not np.isnan(x) for x in history['val_auc']): 
        ax2.plot(epoch_ticks, history['val_auc'], color='tab:purple', linestyle='--', marker='^', markersize=3, label='验证 AUC')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.05)

    ax3 = ax1.twinx() 
    ax3.spines["right"].set_position(("outward", 60)) 
    color = 'tab:green'
    ax3.set_ylabel('Learning Rate', color=color)
    if 'lr' in history and len(history['lr']) == epochs_ran: 
        ax3.plot(epoch_ticks, history['lr'], color=color, linestyle='--', marker='.', markersize=3, label='学习率')
    elif 'lr' in history:
        print(f"Warning: LR history length ({len(history['lr'])}) doesn't match epochs_ran ({epochs_ran}) for {LABEL_COLUMN_NAME}. LR plot skipped.")

    ax3.tick_params(axis='y', labelcolor=color)
    ax3.legend(loc='lower right')
    ax3.set_yscale('log') 

    fig.tight_layout()  
    plt.title(f'{LABEL_COLUMN_NAME} 二分类训练过程监控') # Use LABEL_COLUMN_NAME
    plt.xticks(epoch_ticks)
    plt.savefig(f"training_history_{LABEL_COLUMN_NAME.lower()}.png") # Use LABEL_COLUMN_NAME
    plt.show()

if any(history.values()): # Check if history has any data
    plot_training_history_p53(history)
else:
    print(f"No training history to plot for {LABEL_COLUMN_NAME}.")

# %% [markdown]
# ## 测试集评估 for P53-MUT -> 改为验证集最终评估

# %%
# P53_CLASS_NAMES = ['P53_N (Wild-type)', 'P53_Y (Mutant)'] # Already defined or can be dynamic from LABEL_COLUMN_NAME
# Using LABEL_COLUMN_NAME for dynamic class names if needed, or keep specific like P53_CLASS_NAMES

print(f"\nGenerating ROC and P-R curves for {LABEL_COLUMN_NAME} on the validation set using the best model.")
try:
    model.load_state_dict(torch.load(f"best_model_{LABEL_COLUMN_NAME.lower()}.pth")) # Use LABEL_COLUMN_NAME
    model.eval()
    print(f"Loaded best {LABEL_COLUMN_NAME} model for final validation set evaluation.")

    all_val_labels_final = []
    all_val_probs_pos_class_final = [] # Probabilities for the positive class (class 1)
    # all_val_preds_final = [] # Not strictly needed for ROC/PR but can be for classification report
    
    if len(val_ds) > 0:
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Final Validation for ROC/PR {LABEL_COLUMN_NAME}"):
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1) 
                # preds = torch.argmax(probs, dim=1)
                
                all_val_probs_pos_class_final.extend(probs[:, 1].cpu().numpy()) 
                all_val_labels_final.extend(labels.numpy())
                # all_val_preds_final.extend(preds.cpu().numpy())

        all_val_labels_final_np = np.array(all_val_labels_final)
        all_val_probs_pos_class_final_np = np.array(all_val_probs_pos_class_final)
        # all_val_preds_final_np = np.array(all_val_preds_final)

        # Optional: Classification report for validation set at 0.5 threshold
        # print(f"\n=== Validation Classification Report {LABEL_COLUMN_NAME} (Threshold = 0.5) ===")
        # preds_at_0_5_val = (all_val_probs_pos_class_final_np >= 0.5).astype(int)
        # target_names_display_val = [f"{LABEL_COLUMN_NAME} Negative", f"{LABEL_COLUMN_NAME} Positive"]
        # report_0_5_val = classification_report(all_val_labels_final_np, preds_at_0_5_val, target_names=target_names_display_val, zero_division=0)
        # print(report_0_5_val)
        # cm_0_5_val = confusion_matrix(all_val_labels_final_np, preds_at_0_5_val)
        # print("Confusion Matrix (Validation, Threshold = 0.5):")
        # print(cm_0_5_val)
        # plot_confusion_matrix_generic(cm_0_5_val, target_names_display_val, title=f'{LABEL_COLUMN_NAME} Val CM (Thresh 0.5)', filename=f'val_cm_{LABEL_COLUMN_NAME.lower()}_0.5.png')

        if len(np.unique(all_val_labels_final_np)) >= NUM_CLASSES:
            try:
                val_auc_final = roc_auc_score(all_val_labels_final_np, all_val_probs_pos_class_final_np)
                print(f"Final Validation AUC for {LABEL_COLUMN_NAME}: {val_auc_final:.4f}")

            # ROC Curve
                fpr, tpr, _ = roc_curve(all_val_labels_final_np, all_val_probs_pos_class_final_np)
            plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, lw=2, label=f'{LABEL_COLUMN_NAME} ROC curve (AUC = {val_auc_final:.3f})') # Use LABEL_COLUMN_NAME
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率 (False Positive Rate)')
            plt.ylabel('真阳性率 (True Positive Rate)')
                plt.title(f'验证集ROC曲线 ({LABEL_COLUMN_NAME})') # Use LABEL_COLUMN_NAME
            plt.legend(loc="lower right")
            plt.grid(True)
                plt.savefig(f'roc_curve_{LABEL_COLUMN_NAME.lower()}_val.png') # Use LABEL_COLUMN_NAME
            plt.show()

            # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(all_val_labels_final_np, all_val_probs_pos_class_final_np)
                ap_score_final = average_precision_score(all_val_labels_final_np, all_val_probs_pos_class_final_np)
                print(f"Final Validation Average Precision for {LABEL_COLUMN_NAME}: {ap_score_final:.4f}")
            plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, lw=2, label=f'{LABEL_COLUMN_NAME} P-R curve (AP = {ap_score_final:.3f})') # Use LABEL_COLUMN_NAME
                no_skill = len(all_val_labels_final_np[all_val_labels_final_np==1]) / len(all_val_labels_final_np) if len(all_val_labels_final_np) > 0 else 0
                plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label=f'No Skill (AP={no_skill:.3f})')
            plt.xlabel('召回率 (Recall)')
            plt.ylabel('精确率 (Precision)')
                plt.title(f'验证集P-R曲线 ({LABEL_COLUMN_NAME})') # Use LABEL_COLUMN_NAME
                plt.legend(loc="best")
            plt.grid(True)
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
                plt.savefig(f'pr_curve_{LABEL_COLUMN_NAME.lower()}_val.png') # Use LABEL_COLUMN_NAME
            plt.show()
            except ValueError as e_val_curves:
                print(f"Final Validation ROC/PR calculation error for {LABEL_COLUMN_NAME}: {e_val_curves}")
        else:
            print(f"Final Validation ROC/PR not computed for {LABEL_COLUMN_NAME}: validation set does not contain enough distinct classes (needs at least {NUM_CLASSES}). Found: {np.unique(all_val_labels_final_np)}")
    else:
        print(f"Validation dataset for {LABEL_COLUMN_NAME} is empty. No final ROC/PR curves generated.")

except FileNotFoundError:
    print(f"Error: 'best_model_{LABEL_COLUMN_NAME.lower()}.pth' not found. Was the {LABEL_COLUMN_NAME} model trained and saved?")
except Exception as e:
    print(f"An error occurred during final {LABEL_COLUMN_NAME} validation set evaluation: {e}")

# %% [markdown]
# ## Grad-CAM 可视化 for P53-MUT
# %%
# P53_CLASS_NAMES is defined earlier. Keep if specific names are preferred over dynamic ones.
TASK_CLASS_NAMES_P53 = [f'{LABEL_COLUMN_NAME} N (Wild-type)', f'{LABEL_COLUMN_NAME} Y (Mutant)'] # Specific for P53

def visualize_grad_cam_p53(model, dataset, device, num_images=3, target_class_to_viz=1): # Default to viz positive class (Y)
    # More robust target layer finding from pdl1_atezo_model.py
    target_layer = None
    if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
        # Try to find the last Conv2d layer in the EfficientNet features
        # EfficientNet-B0 typically has features[-1][0] as a Conv2dNormActivation or similar
        # Search backwards for a nn.Conv2d or a module containing one (like Conv2dNormActivation)
        for i in range(len(model.features) -1, -1, -1):
            block_or_layer = model.features[i]
            if isinstance(block_or_layer, nn.Conv2d):
                target_layer = block_or_layer
                print(f"Grad-CAM target layer found: model.features[{i}] (direct Conv2d)")
                break
            if isinstance(block_or_layer, nn.Sequential) and len(block_or_layer) > 0:
                # Look inside Sequential blocks, e.g., the last MBConv block
                # Common for EfficientNet: model.features[-1][0] is Conv2dNormActivation
                # Check the last sub-module of this sequential block, or its sub-sub-modules
                # This can be model.features[-1][0] if it is a ConvBNActivation
                # Or the last conv layer within an MBConv block (e.g., model.features[-1].block[-1][2] for some variants)
                # For B0, model.features[-1][0] is often a good candidate
                potential_target = None
                if hasattr(model.features[-1], 'block') and isinstance(model.features[-1].block, nn.Sequential):
                     # Try last element of the last MBConv block's internal sequence, e.g. its last conv layer
                     if len(model.features[-1].block) > 0 and hasattr(model.features[-1].block[-1], ' তাক') and isinstance(model.features[-1].block[-1]. তাক, nn.Conv2d):
                          potential_target = model.features[-1].block[-1]. তাক # Conv from ConvNormActivation
                          print(f"Grad-CAM target layer from MBConv block: model.features[-1].block[-1]. তাক")
                     elif len(model.features[-1].block) > 0 and isinstance(model.features[-1].block[-1], nn.Conv2d):
                          potential_target = model.features[-1].block[-1]
                          print(f"Grad-CAM target layer from MBConv block: model.features[-1].block[-1] (direct Conv2d)")
                if potential_target is None and isinstance(model.features[-1][0], (nn.Conv2d, nn.modules.conv.Conv2d)):
                    potential_target = model.features[-1][0]
                    print(f"Grad-CAM target layer (model.features[-1][0]): {type(potential_target)}")
                elif potential_target is None and hasattr(model.features[-1][0], ' তাক') and isinstance(model.features[-1][0]. তাক, nn.Conv2d):
                    potential_target = model.features[-1][0]. তাক
                    print(f"Grad-CAM target layer (model.features[-1][0]. তাক): {type(potential_target)}")
                
                if potential_target is not None:
                    target_layer = potential_target
                    break
    if target_layer is None: 
        print("Could not automatically determine a suitable Conv2d target layer for Grad-CAM. Defaulting to model.features[-1]. This might not be optimal or correct.")
        target_layers = [model.features[-1]] # Fallback, may not be a conv layer
    else:
        target_layers = [target_layer]

    cam = GradCAM(model=model, target_layers=target_layers)

    if len(dataset) == 0:
        print(f"Dataset for {LABEL_COLUMN_NAME} Grad-CAM is empty.")
        return

    actual_num_images = min(num_images, len(dataset))
    if actual_num_images == 0:
         print(f"No images to visualize for {LABEL_COLUMN_NAME} Grad-CAM.")
         return
         
    # Try to get images from the target class first
    class_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == target_class_to_viz]
    if len(class_indices) < actual_num_images:
        print(f"Warning: Not enough images for class {target_class_to_viz} ({len(class_indices)} found). Supplementing with other images.")
        other_indices = [i for i in range(len(dataset)) if i not in class_indices]
        needed_more = actual_num_images - len(class_indices)
        if len(other_indices) >= needed_more:
            class_indices.extend(np.random.choice(other_indices, needed_more, replace=False))
        else: class_indices.extend(other_indices)
    
    indices_to_use = np.random.choice(class_indices, min(actual_num_images, len(class_indices)), replace=False) if len(class_indices) > 0 else []

    if not list(indices_to_use):
        print(f"No images selected for Grad-CAM for target class {target_class_to_viz}.")
        return
    
    rgb_imgs_list = []
    cam_outputs_list = []
    titles_list = []

    for i, idx in enumerate(indices_to_use):
        img_tensor, true_label = dataset[idx] 
        
        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        rgb_img_denorm = inv_normalize(img_tensor.clone()).permute(1, 2, 0).cpu().numpy()
        rgb_img_denorm = np.clip(rgb_img_denorm, 0, 1) 

        input_tensor_unsqueeze = img_tensor.unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(target_class_to_viz)]
        
        grayscale_cam = cam(input_tensor=input_tensor_unsqueeze, targets=targets)
        if grayscale_cam is not None and grayscale_cam.shape[0] > 0:
            grayscale_cam_batch = grayscale_cam[0, :]
        else:
            print(f"Warning: {LABEL_COLUMN_NAME} Grad-CAM returned None or empty for image index {idx}, target class {target_class_to_viz}.")
            continue
        
        cam_image = show_cam_on_image(rgb_img_denorm, grayscale_cam_batch, use_rgb=True)
        cam_image_tensor = transforms.ToTensor()(cam_image) 
        original_img_for_grid = transforms.ToTensor()(Image.fromarray((rgb_img_denorm * 255).astype(np.uint8)))
        
        rgb_imgs_list.append(original_img_for_grid)
        cam_outputs_list.append(cam_image_tensor)
        titles_list.append(f"True: {TASK_CLASS_NAMES_P53[true_label]}\nCAM for: {TASK_CLASS_NAMES_P53[target_class_to_viz]}")

    if not rgb_imgs_list:
        print(f"No {LABEL_COLUMN_NAME} CAM images generated for target class {target_class_to_viz}.")
        return

    cols = len(rgb_imgs_list)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 4, 8))
    if cols == 1: axes = axes.reshape(2,1) # Ensure axes is 2D for single image

    for j in range(cols):
        axes[0, j].imshow(rgb_imgs_list[j].permute(1,2,0).numpy())
        axes[0, j].set_title(titles_list[j], fontsize=8)
        axes[0, j].axis('off')
        axes[1, j].imshow(cam_outputs_list[j].permute(1,2,0).numpy())
        axes[1, j].axis('off')
    
    fig.suptitle(f"Grad-CAM for Target Class: {TASK_CLASS_NAMES_P53[target_class_to_viz]} ({LABEL_COLUMN_NAME} Model)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_filename = f'grad_cam_{LABEL_COLUMN_NAME.lower()}_target_{target_class_to_viz}.png' # Use LABEL_COLUMN_NAME
    plt.savefig(save_filename)
    print(f"Grad-CAM grid for {LABEL_COLUMN_NAME} (Target Class {target_class_to_viz}) saved to {save_filename}")
    plt.show()

# Use val_ds for Grad-CAM as test_ds is removed.
if 'model' in locals() and 'val_ds' in locals() and len(val_ds) > 0:
    print(f"Visualizing Grad-CAM for {LABEL_COLUMN_NAME} model (targeting positive class Y=1 on validation set)")
    visualize_grad_cam_p53(model, dataset=val_ds, device=device, num_images=3, target_class_to_viz=1) 
    print(f"Visualizing Grad-CAM for {LABEL_COLUMN_NAME} model (targeting negative class N=0 on validation set)")
    visualize_grad_cam_p53(model, dataset=val_ds, device=device, num_images=3, target_class_to_viz=0) 
else:
    print(f"Skipping {LABEL_COLUMN_NAME} Grad-CAM: Model or validation dataset not available or val_ds is empty.")

# %% [markdown]
# ## 特征与标签相关性分析 (互信息, t-SNE, 置换检验) - Adapted for P53-MUT

# %% 
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
    
    if not embeddings_list: 
        return np.array([]), np.array([])
    return np.concatenate(embeddings_list), np.concatenate(labels_list)

def calculate_mutual_information(features, labels):
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if len(features) == 0 or len(labels) == 0 or len(features) != len(labels):
        print("Warning: MI calculation skipped due to empty or mismatched feature/label arrays.")
        return np.array([0.0]) 
    mi = mutual_info_classif(features, labels, random_state=SEED)
    return mi

def plot_tsne_visualization(embeddings, labels, title_suffix=""):
    if len(embeddings) == 0 or len(labels) == 0 or len(embeddings) != len(labels):
        print(f"Skipping t-SNE for {LABEL_COLUMN_NAME}{title_suffix}: Empty or mismatched embeddings/labels.")
        return
    if len(embeddings) < 2 : 
        print(f"Skipping t-SNE for {LABEL_COLUMN_NAME}{title_suffix}: Not enough samples ({len(embeddings)}).")
        return

    print(f"Running t-SNE for {LABEL_COLUMN_NAME}{title_suffix}...")
    perplexity_value = min(30, len(embeddings) - 1) if len(embeddings) > 1 else 5
    if perplexity_value <=0: perplexity_value = 5

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity_value, n_iter=1000, init='pca', learning_rate='auto')
    try:
        embeddings_2d = tsne.fit_transform(embeddings)
    except Exception as e_tsne:
        print(f"Error during t-SNE for {LABEL_COLUMN_NAME}{title_suffix}: {e_tsne}. Skipping t-SNE plot.")
        return
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    # Use a P53-MUT specific color map or default if not many classes
    colors = plt.cm.get_cmap("coolwarm", len(unique_labels) if len(unique_labels) > 0 else 1) 
    
    for i, label_val in enumerate(unique_labels):
        idx = labels == label_val
        # Use P53-MUT specific class names for legend
        display_label = f'{LABEL_COLUMN_NAME} {"N" if label_val == 0 else "Y"}' 
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], color=colors(i), label=display_label, alpha=0.7)
    
    plt.title(f't-SNE 可视化 ({LABEL_COLUMN_NAME}{title_suffix})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    if len(unique_labels) > 0 : plt.legend()
    plt.grid(True)
    plt.savefig(f"tsne_visualization_{LABEL_COLUMN_NAME.lower()}{title_suffix.replace(' ', '_').lower()}.png")
    plt.show()

def simulate_data_cleaning_test(model, original_val_loader, original_labels_np, num_samples_to_flip=50): 
    print(f"\nSimulating data cleaning test for {LABEL_COLUMN_NAME} by flipping up to {num_samples_to_flip} labels...")
    
    actual_samples_to_flip = min(num_samples_to_flip, len(original_labels_np) // 2) 
    if len(original_labels_np) == 0 or actual_samples_to_flip == 0:
        print(f"Warning: Not enough samples ({len(original_labels_np)}) or 0 samples to flip. Skipping simulation.")
        return float('nan')

    flipped_labels_np = original_labels_np.copy()
    indices_to_flip = np.random.choice(len(flipped_labels_np), actual_samples_to_flip, replace=False)
    flipped_labels_np[indices_to_flip] = 1 - flipped_labels_np[indices_to_flip]
    
    global all_val_probs_pos_class_final_np # Use the global var if available from validation eval
    eval_probs_np = None
    if 'all_val_probs_pos_class_final_np' in globals() and all_val_probs_pos_class_final_np is not None and len(all_val_probs_pos_class_final_np) == len(flipped_labels_np):
        eval_probs_np = all_val_probs_pos_class_final_np
    else:
        print("Warning: `all_val_probs_pos_class_final_np` not available or mismatched for data cleaning. Re-evaluating model.")
        model.eval()
        temp_probs_list = []
        if len(val_ds) > 0:
            with torch.no_grad():
                for imgs, _ in tqdm(original_val_loader, desc=f"Re-evaluating for data cleaning test {LABEL_COLUMN_NAME}"):
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    probs = torch.softmax(outputs, dim=1)
                    temp_probs_list.extend(probs[:, 1].cpu().numpy())
            eval_probs_np = np.array(temp_probs_list)
        else:
            print("val_ds is empty, cannot re-evaluate for data cleaning test.")
            return float('nan')

    if eval_probs_np is None or len(eval_probs_np) != len(flipped_labels_np):
        print("Could not obtain model probabilities for data cleaning test. Skipping.")
        return float('nan')

    if len(np.unique(flipped_labels_np)) < 2:
        print(f"Warning: After flipping, only one class present in simulated labels for {LABEL_COLUMN_NAME}. AUC will be undefined.")
        return float('nan')
        
    try:
        auc_after_cleaning = roc_auc_score(flipped_labels_np, eval_probs_np)
        print(f"AUC after simulated cleaning for {LABEL_COLUMN_NAME} ({actual_samples_to_flip} labels flipped): {auc_after_cleaning:.4f}")
        return auc_after_cleaning
    except ValueError as e:
        print(f"Error calculating AUC after simulated cleaning for {LABEL_COLUMN_NAME}: {e}")
        return float('nan')

def perform_permutation_test(model, val_loader, original_labels_np, original_probs_np, n_permutations=1000):
    print(f"\nPerforming permutation test for {LABEL_COLUMN_NAME} with {n_permutations} permutations...")
    
    if len(original_labels_np) == 0 or len(original_probs_np) == 0 or len(original_labels_np) != len(original_probs_np):
        print(f"Skipping permutation test for {LABEL_COLUMN_NAME} due to empty or mismatched labels/probabilities.")
        return float('nan')

    if len(np.unique(original_labels_np)) < 2:
        print(f"Warning: Original labels for {LABEL_COLUMN_NAME} have less than 2 unique classes. Permutation test might be less meaningful.")
    
    try:
        observed_auc = roc_auc_score(original_labels_np, original_probs_np)
    except ValueError:
        print(f"Could not calculate observed AUC for {LABEL_COLUMN_NAME} (likely one class in labels). Permutation test skipped.")
        return float('nan')
        
    print(f"Observed AUC for {LABEL_COLUMN_NAME}: {observed_auc:.4f}")
    
    permuted_aucs = []
    for i in tqdm(range(n_permutations), desc=f"Permutation Test {LABEL_COLUMN_NAME}"):
        permuted_labels = sklearn_shuffle(original_labels_np, random_state=SEED + i)
        if len(np.unique(permuted_labels)) < 2:
            permuted_aucs.append(0.5) 
            continue
        try:
            auc = roc_auc_score(permuted_labels, original_probs_np) 
            permuted_aucs.append(auc)
        except ValueError:
             permuted_aucs.append(0.5) 

    permuted_aucs = np.array(permuted_aucs)
    if len(permuted_aucs) == 0: 
        print("No permuted AUCs generated. Skipping p-value calculation.")
        return float('nan')
    p_value = np.mean(permuted_aucs >= observed_auc)
    
    print(f"Permutation test for {LABEL_COLUMN_NAME}: p-value = {p_value:.4f}")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(permuted_aucs, bins=30, kde=True, label='Permuted AUCs')
    plt.axvline(observed_auc, color='red', linestyle='--', lw=2, label=f'Observed AUC ({observed_auc:.3f})')
    plt.title(f'Permutation Test Results for {LABEL_COLUMN_NAME}')
    plt.xlabel('AUC Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"permutation_test_{LABEL_COLUMN_NAME.lower()}.png")
    plt.show()

    return p_value

# --- Main execution of the new analyses ---
if 'model' in locals() and 'val_loader' in locals() and len(val_ds) > 0:
    print(f"\n--- Starting Feature-Label Relevance Analysis for {LABEL_COLUMN_NAME} ---")
    
    try:
        model.load_state_dict(torch.load(f"best_model_{LABEL_COLUMN_NAME.lower()}.pth"))
        model.to(device) 
        print(f"Loaded best {LABEL_COLUMN_NAME} model for feature-label analysis.")

        # 1. Get Embeddings and True Labels from Validation Set
        # Make sure all_val_labels_final_np and all_val_probs_pos_class_final_np exist from previous validation step
        if 'all_val_labels_final_np' not in globals() or 'all_val_probs_pos_class_final_np' not in globals() or \
            all_val_labels_final_np is None or all_val_probs_pos_class_final_np is None or \
            len(all_val_labels_final_np) != len(val_ds) or len(all_val_probs_pos_class_final_np) != len(val_ds):
            
            print("Recalculating final validation labels and probabilities for analysis...")
            temp_all_val_labels_final_list = []
            temp_all_val_probs_pos_class_final_list = []
            model.eval()
            with torch.no_grad():
                for imgs_an, labels_an in tqdm(val_loader, desc=f"Final Validation for Analysis {LABEL_COLUMN_NAME}"):
                    imgs_an = imgs_an.to(device)
                    outputs_an = model(imgs_an)
                    probs_an = torch.softmax(outputs_an, dim=1)
                    temp_all_val_probs_pos_class_final_list.extend(probs_an[:, 1].cpu().numpy())
                    temp_all_val_labels_final_list.extend(labels_an.numpy())
            all_val_labels_final_np = np.array(temp_all_val_labels_final_list)
            all_val_probs_pos_class_final_np = np.array(temp_all_val_probs_pos_class_final_list)
            if len(all_val_labels_final_np) == 0: 
                 print("Validation set is empty after trying to recalculate labels/probs for analysis. Skipping analysis.")
                 raise ValueError("Empty validation set for analysis") 

        val_embeddings, val_true_labels_for_analysis = get_embeddings_efficientnet(model, val_loader, device) 
        
        if val_embeddings.size > 0 and len(val_embeddings) > 0 and len(val_true_labels_for_analysis) == len(val_embeddings):
            if all_val_probs_pos_class_final_np is not None and len(all_val_probs_pos_class_final_np) == len(val_true_labels_for_analysis):
                mi_scores_probs = calculate_mutual_information(all_val_probs_pos_class_final_np.reshape(-1, 1), val_true_labels_for_analysis)
                print(f"Mutual Information (Class 1 Probs vs Labels) for {LABEL_COLUMN_NAME}: {mi_scores_probs[0]:.4f}")
            else:
                print("Could not calculate MI with probabilities, not available or mismatched length.")

            plot_tsne_visualization(val_embeddings, val_true_labels_for_analysis)
            simulate_data_cleaning_test(model, val_loader, all_val_labels_final_np, num_samples_to_flip=max(1, len(val_ds)//10)) 
            
            if all_val_labels_final_np is not None and all_val_probs_pos_class_final_np is not None and len(all_val_labels_final_np)>0:
                perform_permutation_test(model, val_loader, all_val_labels_final_np, all_val_probs_pos_class_final_np, n_permutations=1000)
            else:
                print(f"Skipping Permutation Test for {LABEL_COLUMN_NAME} due to missing validation labels or probabilities.")
        else:
            print(f"Could not extract embeddings or labels for {LABEL_COLUMN_NAME} analysis (embeddings size: {val_embeddings.size}, len: {len(val_embeddings)}). Skipping relevance analysis.")
            
    except FileNotFoundError:
        print(f"Error: 'best_model_{LABEL_COLUMN_NAME.lower()}.pth' not found. Cannot perform feature-label analysis for {LABEL_COLUMN_NAME}.")
    except ValueError as ve:
        print(f"ValueError during analysis setup for {LABEL_COLUMN_NAME}: {ve}")
    except Exception as e_analysis:
        print(f"An error occurred during {LABEL_COLUMN_NAME} feature-label relevance analysis: {e_analysis}")
else:
    print(f"Skipping Feature-Label Relevance Analysis for {LABEL_COLUMN_NAME}: Model or validation data not available or val_ds is empty.")


# %% [markdown]
# ## Grad-CAM 可视化 for P53-MUT
// ... existing code ...

# %%
print(f"{LABEL_COLUMN_NAME} binary classification model script generation complete.")
print(f"IMPORTANT: Review and adjust 'LABEL_COLUMN_NAME' ('{LABEL_COLUMN_NAME}'), and the 'map_p53_to_binary' function if your labels differ.")
print(f"Also, review FocalLoss 'alpha' parameters for {LABEL_COLUMN_NAME} task.") 