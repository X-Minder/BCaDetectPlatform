# %% [markdown]
# # Ki-67 (30% cut-off) 膀胱镜图像二分类模型
# 通过迁移学习（EfficientNet-B0）对 Ki-67 (30% cut-off) 进行二分类 (0/1)，按患者划分训练/验证/测试集。

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
    def __init__(self, alpha=None, gamma=2., reduction='mean', num_classes=2): # MODIFIED: num_classes for binary
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        
        if self.num_classes == 2: # Specific handling for binary case often means alpha is a single float for positive class
            if isinstance(alpha, list) and len(alpha)==2:
                # User might provide [alpha_neg, alpha_pos], we usually want alpha for the positive class (class 1)
                # Or it's [alpha_for_class_0, alpha_for_class_1]
                # For binary, if alpha is a list of 2, it's typically [alpha_0, alpha_1].
                # The standard Focal Loss formula with alpha often implies alpha for positive, (1-alpha) for negative.
                # PyTorch's BCEWithLogitsLoss pos_weight is similar.
                # For simplicity here, if list of 2, we'll assume it's [alpha_for_class_0, alpha_for_class_1]
                # and it will be handled by the general multi-class logic if not overridden.
                 print(f"FocalLoss: alpha is a list of 2 for binary classification: {alpha}. Ensure this matches expected use.")
                 self.alpha = torch.tensor(alpha)

            elif isinstance(alpha, (float, int)):
                 # For binary, a single alpha often refers to the weight of the positive class.
                 # The loss formula will handle this by applying alpha to class 1 and (1-alpha) to class 0 if using BCE formulation.
                 # Here, we are using cross_entropy, so alpha needs to be a tensor for indexing.
                 # Let's assume the single float is for class 1, and class 0 is 1.0 for now or requires specific setup.
                 # A common practice for binary with CE-like loss is to pass weights for each class.
                 # If a single alpha is given, it's safer to assume it applies to the positive class (1)
                 # and the negative class (0) gets weight 1.0 or (1-alpha).
                 # For this implementation using F.cross_entropy, alpha needs to be a list/tensor of weights.
                 # If a single float is given, let's assume it's for class 1, and class 0 is 1.0.
                 # Or provide as list [alpha_for_0, alpha_for_1]
                 print(f"FocalLoss: single alpha={alpha} for binary. Will be used as weight for class 1 if targets are 0/1. Or provide as a list [alpha_0, alpha_1].")
                 # Defaulting to a list if single float for gather to work, assuming it's for positive class
                 # self.alpha = torch.tensor([1.0, alpha]) # This would make class 0 weight 1.0, class 1 weight alpha
                 # Or, more generally, if alpha is provided, it's a list of weights per class.
                 # If only one value, it's ambiguous without convention.
                 # Let's follow the multi-class path for consistency.
                 self.alpha = torch.tensor([alpha, alpha]) # Or some other convention like [1-alpha, alpha]
                 print(f"Warning: Single float alpha in FocalLoss for {num_classes} classes. Interpreting as equal weight or specific list needed. Using [alpha, alpha].")


        elif isinstance(alpha, (float, int)): # For num_classes > 2
             print(f"Warning: Single float alpha in FocalLoss for {num_classes} classes. Assuming equal weight or specific list needed. Using [alpha]*num_classes.")
             self.alpha = torch.tensor([alpha] * num_classes)
        
        if isinstance(alpha, list): 
            self.alpha = torch.tensor(alpha)
            if len(self.alpha) != num_classes:
                print(f"Warning: Alpha list length {len(self.alpha)} does not match num_classes={num_classes}. Adjusting.")
                if len(self.alpha) < num_classes:
                    self.alpha = torch.cat([self.alpha, torch.full((num_classes - len(self.alpha),), 1.0/num_classes)])
                else:
                    self.alpha = self.alpha[:num_classes]
        
        self.reduction = reduction

    def forward(self, inputs, targets):
        # For binary classification with F.cross_entropy, inputs are [N, 2] and targets are [N] (0 or 1)
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (1-pt)**self.gamma * CE_loss

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            
            at = self.alpha.gather(0, targets.data.view(-1)) # Index alpha with target classes (0 or 1)
            F_loss = at * F_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# %% [markdown]
# ## 自定义 Dataset for Ki-67
# %%
class Ki67Dataset(Dataset): # MODIFIED: Renamed
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
        label = int(row[MAPPED_LABEL_COLUMN_NAME]) # Ensure this column holds 0 or 1
        return img, label

# %% [markdown]
# ## 读取标签并划分数据集 for Ki-67
# %%
LABEL_COLUMN_NAME = 'Ki-67_30cutoff'      # MODIFIED: Target column for Ki-67 status
MAPPED_LABEL_COLUMN_NAME = 'Ki-67_30cutoff_mapped' # MODIFIED: Processed binary label (0 or 1)
PATIENT_ID_COLUMN = 'PATIENT_ID'       # Stays the same
FILE_NAME_COLUMN = 'FILE_NAME'         # Stays the same
NUM_CLASSES = 2                        # MODIFIED: For Ki-67 binary classification

label_df = pd.read_csv("dataset/label.csv")
print(f"1. Initial rows loaded from CSV: {len(label_df)}")

# Drop rows with NA in critical columns (Ki-67_30cut, FILE_NAME, PATIENT_ID)
label_df = label_df.dropna(subset=[LABEL_COLUMN_NAME, FILE_NAME_COLUMN, PATIENT_ID_COLUMN]).copy() # MODIFIED: LABEL_COLUMN_NAME
print(f"2. Rows after dropping NA from key columns ('{LABEL_COLUMN_NAME}', '{FILE_NAME_COLUMN}', '{PATIENT_ID_COLUMN}'): {len(label_df)}")

if len(label_df) == 0:
    print(f"ERROR: All rows were dropped after initial NA check for {LABEL_COLUMN_NAME}. Please check CSV.")
else:
    print(f"   Unique values in '{LABEL_COLUMN_NAME}' before mapping: {label_df[LABEL_COLUMN_NAME].astype(str).unique()}")

    def map_ki67_to_binary(ki67_status): # MODIFIED: New mapping function
        if pd.isna(ki67_status):
            return np.nan

        status_str = str(ki67_status).strip()
        if not status_str:
            return np.nan
        
        try:
            status_val_float = float(status_str)
            status_val = int(status_val_float)
            
            if status_val != status_val_float: # Check if it was truly an integer like "1.0" vs "1.5"
                 print(f"Warning: Ki-67 status '{ki67_status}' (str: '{status_str}') has non-zero decimal. Mapping to NaN.")
                 return np.nan

            if status_val in [0, 1]:
                return status_val
            else: # Values like 2, -1 etc. are not expected for Ki-67_30cut
                print(f"Warning: Unexpected Ki-67 status '{ki67_status}' (parsed as {status_val}). Not 0 or 1. Mapping to NaN.")
                return np.nan
        except ValueError:
            print(f"Warning: Could not convert Ki-67 status '{ki67_status}' (str: '{status_str}') to number. Mapping to NaN.")
            return np.nan

    label_df[MAPPED_LABEL_COLUMN_NAME] = label_df[LABEL_COLUMN_NAME].apply(map_ki67_to_binary) # MODIFIED
    print(f"3. Rows after applying 'map_ki67_to_binary' (before dropping NA from '{MAPPED_LABEL_COLUMN_NAME}'): {len(label_df)}")
    print(f"   Unique values in '{MAPPED_LABEL_COLUMN_NAME}' after mapping (before dropping NA): {label_df[MAPPED_LABEL_COLUMN_NAME].unique()}")

    label_df = label_df.dropna(subset=[MAPPED_LABEL_COLUMN_NAME]).copy()
    print(f"4. Rows after dropping NA from '{MAPPED_LABEL_COLUMN_NAME}': {len(label_df)}")
    
    if len(label_df) > 0:
        label_df[MAPPED_LABEL_COLUMN_NAME] = label_df[MAPPED_LABEL_COLUMN_NAME].astype(int)
        print(f"   Final unique values in '{MAPPED_LABEL_COLUMN_NAME}': {label_df[MAPPED_LABEL_COLUMN_NAME].unique()}")
    else:
        print(f"ERROR: All rows were dropped after mapping {LABEL_COLUMN_NAME}. Check mapping logic and original '{LABEL_COLUMN_NAME}' values.")

if len(label_df) > 0:
    # Removed GroupShuffleSplit for test set. df_trainval is now the entire preprocessed label_df.
    df_trainval = label_df.copy() 
    # df_test is no longer created.
    
    val_size_from_trainval = 0.2 # Proportion of data to use for validation

    # Splitting df_trainval (which is now the full dataset) into train and validation sets
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_from_trainval, random_state=SEED + 1) 

    if len(df_trainval[PATIENT_ID_COLUMN].unique()) > 1 and len(df_trainval) > 1 : # Need at least 2 groups and more than 1 sample for GroupShuffleSplit
        try:
            train_idx_inner, val_idx_inner = next(gss_val.split(df_trainval, groups=df_trainval[PATIENT_ID_COLUMN]))
            df_train = df_trainval.iloc[train_idx_inner].copy()
            df_val   = df_trainval.iloc[val_idx_inner].copy()
            print(f"Successfully used GroupShuffleSplit for train/validation sets for {LABEL_COLUMN_NAME} from the full dataset.")
        except ValueError as e:
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
    df_train, df_val = pd.DataFrame(), pd.DataFrame() # Removed df_test


# %% [markdown]
# ## 数据增强与 DataLoader for Ki-67
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

train_loader_args = {'shuffle': True} 
if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
    counts_train = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    # For binary, expecting counts for class 0 and 1.
    if len(counts_train) >= 1 and len(counts_train) <= NUM_CLASSES: # Check if at least one class is present, up to NUM_CLASSES
        # Ensure weights are calculated for all NUM_CLASSES (0 and 1 for binary)
        class_sample_weights = [1. / counts_train.get(i, 1e-6) for i in range(NUM_CLASSES)] 
        sample_weights_train = [class_sample_weights[label] for label in df_train[MAPPED_LABEL_COLUMN_NAME]]
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights_train), 
                                        num_samples=len(sample_weights_train), 
                                        replacement=True)
        weights_str = ", ".join([f"Class {i}: {w:.4f}" for i, w in enumerate(class_sample_weights)])
        print(f"Sampler weights for {LABEL_COLUMN_NAME} classes: {weights_str}")
        train_loader_args = {'sampler': sampler, 'shuffle': False}
    else:
        print(f"Warning: Training data for {LABEL_COLUMN_NAME} has insufficient or unexpected class counts for sampler. Using standard DataLoader.")
else:
    print(f"Warning: df_train for {LABEL_COLUMN_NAME} is empty or mapped label column is missing. Using standard DataLoader.")
    if df_train.empty:
        train_loader_args['shuffle'] = False
        print("   df_train is empty. Forcing shuffle=False for train_loader to prevent error.")

train_ds = Ki67Dataset(df_train, IMG_DIR, transform=train_tf) # MODIFIED
val_ds   = Ki67Dataset(df_val,   IMG_DIR, transform=val_tf) # MODIFIED
# test_ds  = Ki67Dataset(df_test,  IMG_DIR, transform=val_tf) # MODIFIED # Removed test_ds

train_loader = DataLoader(train_ds, batch_size=16, num_workers=0, pin_memory=True, **train_loader_args)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
# test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=0, pin_memory=True) # Removed test_loader

# %% [markdown]
# ## 模型定义与训练设置 for Ki-67
# %%
print(f"Using EfficientNet-B0 for {LABEL_COLUMN_NAME} {NUM_CLASSES}-class classification") # MODIFIED
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES) # MODIFIED: Output NUM_CLASSES (2 for binary)
model = model.to(device)

# FocalLoss alpha parameter for Ki-67
# For binary, alpha often refers to the weight of the positive class (class 1).
# If using FocalLoss with num_classes=2, alpha can be a list [alpha_class_0, alpha_class_1]
focal_loss_alpha_values = [0.5, 0.5] # Default equal weights for binary
if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
    counts = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    if len(counts) == NUM_CLASSES: # Both classes 0 and 1 present
        class_weights = [1.0 / counts.get(i, 1e-6) for i in range(NUM_CLASSES)]
        total_weight = sum(class_weights)
        focal_loss_alpha_values = [w / total_weight for w in class_weights] 
        counts_str = ", ".join([f"Class {i}: {counts.get(i,0)}" for i in range(NUM_CLASSES)])
        print(f"Train data counts for {LABEL_COLUMN_NAME}: {counts_str}")
        alpha_str = ", ".join([f"{w:.4f}" for w in focal_loss_alpha_values])
        print(f"Calculated FocalLoss alpha for {LABEL_COLUMN_NAME}: [{alpha_str}]")
    elif len(counts) == 1: # Only one class present
        present_class = counts.index[0]
        focal_loss_alpha_values = [0.0, 0.0]
        focal_loss_alpha_values[present_class] = 1.0 # Weight only the present class
        print(f"Warning: Only class {present_class} present in {LABEL_COLUMN_NAME} training data. FocalLoss alpha set to favor this class: {focal_loss_alpha_values}")
    else: # No classes or unexpected number of classes
        print(f"Warning: Not all {NUM_CLASSES} classes present or unexpected counts in {LABEL_COLUMN_NAME} training data ({len(counts)} found). Using default FocalLoss alpha.")
else:
    print(f"Warning: Could not calculate class counts for {LABEL_COLUMN_NAME} FocalLoss alpha. Using default: {focal_loss_alpha_values}")

focal_loss_alpha = torch.tensor(focal_loss_alpha_values, dtype=torch.float).to(device)
print(f"Using Focal Loss with alpha: {focal_loss_alpha.tolist()} and gamma=2 for {LABEL_COLUMN_NAME}")

criterion = FocalLoss(alpha=focal_loss_alpha, gamma=2, num_classes=NUM_CLASSES) # MODIFIED
# For binary, could also use nn.BCEWithLogitsLoss if model outputs 1 logit and labels are float [0,1]
# criterion = nn.CrossEntropyLoss() # Standard alternative (for 2 classes, equivalent to BCE if model outputs 2 logits)

optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4) # lr might need tuning # MODIFIED: Reduced learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True, min_lr=1e-6)

# %% [markdown]
# ## 早停机制
# %%
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True, mode='min', verbose=True, task_name="Ki-67_30cutoff"): # MODIFIED: Added task_name
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        self.task_name = task_name # MODIFIED
        
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
                print(f"EarlyStopping ({self.task_name}): New best score: {self.best_score:.4f}") # MODIFIED
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping ({self.task_name}): Counter {self.counter}/{self.patience}. Best score: {self.best_score:.4f}") # MODIFIED
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"EarlyStopping ({self.task_name}): Patience reached. Stopping training. Best score: {self.best_score:.4f}") # MODIFIED
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print(f"EarlyStopping ({self.task_name}): Restored best model weights.") # MODIFIED
        return self.early_stop

# %% [markdown]
# ## 训练与验证循环 for Ki-67
# %%
NUM_EPOCHS = 50 # Might need adjustment
best_val_loss = float('inf') 
early_stopping = EarlyStopping(patience=10, mode='min', verbose=True, min_delta=0.0001, task_name=LABEL_COLUMN_NAME) # MODIFIED

history = {
    'train_loss': [], 'val_loss': [], 'val_accuracy': [], 
    'val_auc': [], # MODIFIED: For binary, roc_auc_score directly gives AUC
    'lr': []
}

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train {LABEL_COLUMN_NAME}]"): # MODIFIED
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs) # Should be [batch_size, 2] for NUM_CLASSES=2
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    
    train_loss = running_loss / len(train_ds) if len(train_ds) > 0 else 0.0
    history['train_loss'].append(train_loss)
    
    model.eval()
    all_val_labels = []
    all_val_probs_class1 = [] # Store probabilities for the positive class (class 1) for AUC
    all_val_preds = []
    val_running_loss = 0.0
    
    if len(val_ds) > 0:
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss_iter = criterion(outputs, labels)
                val_running_loss += val_loss_iter.item() * imgs.size(0)
                
                probs = torch.softmax(outputs, dim=1) # Get probabilities for both classes
                preds = torch.argmax(probs, dim=1)    # Get predicted class
                
                all_val_probs_class1.extend(probs[:, 1].cpu().numpy()) # Prob for class 1
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
        
        val_epoch_loss = val_running_loss / len(val_ds)
        history['val_loss'].append(val_epoch_loss)

        all_val_labels_np = np.array(all_val_labels)
        all_val_preds_np = np.array(all_val_preds)
        all_val_probs_class1_np = np.array(all_val_probs_class1)

        val_accuracy = np.mean(all_val_labels_np == all_val_preds_np) if len(all_val_labels_np) > 0 else 0.0
        history['val_accuracy'].append(val_accuracy)

        val_auc = 0.0
        if len(np.unique(all_val_labels_np)) >= 2: # Needs at least 2 classes (0 and 1) for AUC
            try:
                val_auc = roc_auc_score(all_val_labels_np, all_val_probs_class1_np)
            except ValueError as e_auc:
                 print(f"Warning: Epoch {epoch}, AUC calculation error for {LABEL_COLUMN_NAME}: {e_auc}. AUC set to 0.0")
        elif len(all_val_labels_np) > 0:
             print(f"Warning: Epoch {epoch}, {LABEL_COLUMN_NAME} val set does not have enough distinct classes for AUC. AUC set to 0.0")
        history['val_auc'].append(val_auc)
        
        scheduler.step(val_epoch_loss)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_epoch_loss:.4f}, Val Acc={val_accuracy:.4f}, Val AUC={val_auc:.4f}, LR={optimizer.param_groups[0]['lr']:.1e}") # MODIFIED

        if val_epoch_loss < best_val_loss: # Save based on validation loss
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), f"best_model_{LABEL_COLUMN_NAME.lower().replace('-', '_')}.pth") # MODIFIED: Filename consistent replace
            print(f"Epoch {epoch}: New best {LABEL_COLUMN_NAME} model saved with Val Loss: {best_val_loss:.4f}") # MODIFIED
        
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        if early_stopping(val_epoch_loss, model):
            print(f"Early stopping triggered for {LABEL_COLUMN_NAME} based on validation loss.") # MODIFIED
            break
    else: 
        history['val_loss'].append(float('nan'))
        history['val_accuracy'].append(float('nan'))
        history['val_auc'].append(float('nan'))
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss=N/A (empty val set for {LABEL_COLUMN_NAME}), LR={current_lr:.1e}") # MODIFIED


# %% [markdown]
# ## 绘制训练过程曲线 for Ki-67
# %%
def plot_training_history_ki67(history): # MODIFIED
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
    if 'val_auc' in history and any(not np.isnan(x) for x in history['val_auc']): # MODIFIED: Use val_auc
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
        print(f"Warning: LR history length ({len(history['lr'])}) doesn\'t match epochs_ran ({epochs_ran}) for {LABEL_COLUMN_NAME}. LR plot skipped.") # MODIFIED

    ax3.tick_params(axis='y', labelcolor=color)
    ax3.legend(loc='lower right')
    ax3.set_yscale('log') 

    fig.tight_layout()  
    plt.title(f'{LABEL_COLUMN_NAME} 二分类训练过程监控') # MODIFIED
    plt.xticks(epoch_ticks)
    plt.savefig(f"training_history_{LABEL_COLUMN_NAME.lower().replace('-', '_')}.png") # MODIFIED: Filename consistent replace
    plt.show()

if any(history.values()):
    plot_training_history_ki67(history) # MODIFIED
else:
    print(f"No training history to plot for {LABEL_COLUMN_NAME}.") # MODIFIED

# %% [markdown]
# ## 验证集最终评估曲线 (ROC & P-R) for Ki-67

# %% 
print(f"\nGenerating ROC and P-R curves for {LABEL_COLUMN_NAME} on the validation set using the best model.")
try:
    # Load the best model saved during training
    model.load_state_dict(torch.load(f"best_model_{LABEL_COLUMN_NAME.lower().replace('-', '_')}.pth")) # MODIFIED: Filename consistent replace
    model.eval() # Set model to evaluation mode
    print(f"Loaded best {LABEL_COLUMN_NAME} model for final validation set evaluation.")

    all_val_labels_final = []
    all_val_probs_class1_final = []
    
    if len(val_ds) > 0: # Ensure validation dataset is not empty
        with torch.no_grad(): # Disable gradient calculations
            for imgs, labels in tqdm(val_loader, desc=f"Final Validation for ROC/PR {LABEL_COLUMN_NAME}"):
                imgs = imgs.to(device)
                # labels are already on CPU from DataLoader or will be moved via .numpy()
                
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1) # Get probabilities for each class
                
                all_val_probs_class1_final.extend(probs[:, 1].cpu().numpy()) # Probabilities for the positive class (class 1)
                all_val_labels_final.extend(labels.numpy()) # True labels

        all_val_labels_final_np = np.array(all_val_labels_final)
        all_val_probs_class1_final_np = np.array(all_val_probs_class1_final)

        # Check if there are at least two classes in the validation labels for metric calculation
        if len(np.unique(all_val_labels_final_np)) >= 2:
            try:
                # ROC Curve
                val_auc_final = roc_auc_score(all_val_labels_final_np, all_val_probs_class1_final_np)
                print(f"Final Validation AUC for {LABEL_COLUMN_NAME}: {val_auc_final:.4f}")
                fpr, tpr, _ = roc_curve(all_val_labels_final_np, all_val_probs_class1_final_np)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {val_auc_final:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal reference line
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('假阳性率 (False Positive Rate)')
                plt.ylabel('真阳性率 (True Positive Rate)')
                plt.title(f'验证集ROC曲线 ({LABEL_COLUMN_NAME})')
                plt.legend(loc="lower right")
                plt.grid(True)
                plt.savefig(f'roc_curve_{LABEL_COLUMN_NAME.lower().replace("-", "_")}_val.png') # MODIFIED: Filename consistent replace
                plt.show()

                # Precision-Recall Curve
                val_ap_final = average_precision_score(all_val_labels_final_np, all_val_probs_class1_final_np)
                print(f"Final Validation Average Precision for {LABEL_COLUMN_NAME}: {val_ap_final:.4f}")
                precision, recall, _ = precision_recall_curve(all_val_labels_final_np, all_val_probs_class1_final_np)
                
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, lw=2, label=f'P-R curve (AP = {val_ap_final:.3f})')
                plt.xlabel('召回率 (Recall)')
                plt.ylabel('精确率 (Precision)')
                plt.title(f'验证集P-R曲线 ({LABEL_COLUMN_NAME})')
                plt.legend(loc="best")
                plt.grid(True)
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.savefig(f'pr_curve_{LABEL_COLUMN_NAME.lower().replace("-", "_")}_val.png') # MODIFIED: Filename consistent replace
                plt.show()

            except ValueError as e_val_curves:
                print(f"Final Validation ROC/PR calculation error for {LABEL_COLUMN_NAME}: {e_val_curves}")
        else:
            print(f"Final Validation ROC/PR not computed for {LABEL_COLUMN_NAME}: validation set does not contain enough distinct classes (needs at least 2).")
    else:
        print(f"Validation dataset for {LABEL_COLUMN_NAME} is empty. No final ROC/PR curves generated.")

except FileNotFoundError:
    print(f"Error: 'best_model_{LABEL_COLUMN_NAME.lower().replace('-', '_')}.pth' not found. Was the model trained and saved? Cannot generate final ROC/PR curves for {LABEL_COLUMN_NAME}.") # MODIFIED: Filename consistent replace
except Exception as e:
    print(f"An error occurred during final {LABEL_COLUMN_NAME} validation set ROC/PR curve generation: {e}")


# %% [markdown]
# ## Grad-CAM 可视化 for Ki-67
# %%
KI67_CLASS_NAMES = [f'{LABEL_COLUMN_NAME} Class 0 (Low)', f'{LABEL_COLUMN_NAME} Class 1 (High)']

def visualize_grad_cam_ki67(model, dataset, device, num_images=4, target_classes_to_viz=None): # MODIFIED, default num_images to 4 for binary
    target_layer_name = 'features[-1][0]' # For EfficientNet-B0, last conv layer in the final block
    try:
        module_path = target_layer_name.split('.')
        current_module = model
        for m_name in module_path:
            if m_name.isdigit(): current_module = current_module[int(m_name)]
            elif m_name.startswith('[') and m_name.endswith(']'): idx = int(m_name[1:-1]); current_module = current_module[idx]
            else: current_module = getattr(current_module, m_name)
        target_layers = [current_module]
    except Exception as e:
        print(f"Error finding target layer '{target_layer_name}' for {LABEL_COLUMN_NAME} Grad-CAM: {e}. Defaulting to model.features[-1].") # MODIFIED
        target_layers = [model.features[-1]] 

    cam = GradCAM(model=model, target_layers=target_layers)

    if len(dataset) == 0:
        print(f"Dataset for {LABEL_COLUMN_NAME} Grad-CAM is empty.") # MODIFIED
        return

    if target_classes_to_viz is None:
        target_classes_to_viz = list(range(NUM_CLASSES)) # Visualize for class 0 and 1
    
    actual_num_images_per_class = (num_images + len(target_classes_to_viz) -1) // len(target_classes_to_viz)
    images_shown_count = 0
    
    num_viz_rows = len(target_classes_to_viz)
    num_viz_cols = actual_num_images_per_class # Show 'actual_num_images_per_class' images for each target class
    
    if num_viz_rows * num_viz_cols == 0:
        print(f"No images or target classes specified for {LABEL_COLUMN_NAME} Grad-CAM.") # MODIFIED
        return

    fig, axes = plt.subplots(num_viz_rows * 2, num_viz_cols, figsize=(num_viz_cols * 3, num_viz_rows * 6))
    if num_viz_rows * num_viz_cols == 1 : # Only one image and one target class
         axes = np.array([[axes[0]],[axes[1]]]).reshape(2,1,1) # Make it 3D for consistent indexing later
    elif num_viz_cols == 1: 
         axes = axes.reshape(num_viz_rows * 2, 1)
    elif num_viz_rows == 1:
         axes = axes.reshape(2, num_viz_cols)

    # Try to pick diverse images if possible, or just random ones.
    # For simplicity, let's pick 'num_viz_cols' random distinct images from the dataset.
    if len(dataset) < num_viz_cols:
        print(f"Warning: Requested {num_viz_cols} images for Grad-CAM, but dataset only has {len(dataset)}. Using all available.")
        indices_to_use = np.arange(len(dataset))
        num_viz_cols = len(dataset) # Adjust num_viz_cols if fewer images available than requested
        if num_viz_rows * num_viz_cols == 0: return # Exit if no images to show
        # Re-create subplots if num_viz_cols changed and it's not 0
        plt.close(fig) # Close previous figure
        fig, axes = plt.subplots(num_viz_rows * 2, num_viz_cols, figsize=(num_viz_cols * 3, num_viz_rows * 6))
        if num_viz_rows * num_viz_cols == 1 : axes = np.array([[axes[0]],[axes[1]]]).reshape(2,1,1)
        elif num_viz_cols == 1: axes = axes.reshape(num_viz_rows * 2, 1)
        elif num_viz_rows == 1: axes = axes.reshape(2, num_viz_cols)

    else:
        indices_to_use = np.random.choice(len(dataset), num_viz_cols, replace=False)


    for r_idx, target_cls in enumerate(target_classes_to_viz):
        for c_idx_local, img_idx_in_dataset in enumerate(indices_to_use): # iterate through selected image indices
            current_c_idx_for_plot = c_idx_local # Use this for subplot indexing

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
                print(f"Warning: {LABEL_COLUMN_NAME} Grad-CAM returned None or empty for image index {img_idx_in_dataset}, target class {target_cls}.") # MODIFIED
                # Handle axes for missing CAM
                if num_viz_cols > 0 and num_viz_rows > 0:
                    ax_orig = axes[r_idx * 2, current_c_idx_for_plot] if num_viz_cols > 1 else (axes[r_idx*2] if num_viz_rows > 1 else axes[0,0,0] if num_viz_rows*num_viz_cols==1 else axes[0] )
                    ax_cam  = axes[r_idx * 2 + 1, current_c_idx_for_plot] if num_viz_cols > 1 else (axes[r_idx*2+1] if num_viz_rows > 1 else axes[1,0,0] if num_viz_rows*num_viz_cols==1 else axes[1])
                    ax_orig.axis('off'); ax_cam.axis('off')
                continue
            
            cam_image = show_cam_on_image(rgb_img_denorm, grayscale_cam_batch, use_rgb=True)
            cam_image_tensor = transforms.ToTensor()(cam_image) 
            original_img_for_grid = transforms.ToTensor()(Image.fromarray((rgb_img_denorm * 255).astype(np.uint8)))
            
            title_str = f"""True: {KI67_CLASS_NAMES[true_label]}
CAM for: {KI67_CLASS_NAMES[target_cls]}""" # MODIFIED

            if num_viz_rows == 1 and num_viz_cols == 1:
                ax_orig_current = axes[0,0,0]; ax_cam_current = axes[1,0,0]
            elif num_viz_rows == 1:
                ax_orig_current = axes[0, current_c_idx_for_plot]; ax_cam_current = axes[1, current_c_idx_for_plot]
            elif num_viz_cols == 1:
                ax_orig_current = axes[r_idx * 2, 0]; ax_cam_current = axes[r_idx * 2 + 1, 0]
            else:
                ax_orig_current = axes[r_idx * 2, current_c_idx_for_plot]; ax_cam_current = axes[r_idx * 2 + 1, current_c_idx_for_plot]

            ax_orig_current.imshow(original_img_for_grid.permute(1,2,0).numpy())
            ax_orig_current.set_title(title_str, fontsize=8)
            ax_orig_current.axis('off')
            ax_cam_current.imshow(cam_image_tensor.permute(1,2,0).numpy())
            ax_cam_current.axis('off')
            images_shown_count +=1

    if images_shown_count == 0:
        print(f"No {LABEL_COLUMN_NAME} CAM images were generated.") # MODIFIED
        if num_viz_rows * num_viz_cols > 0 : plt.close(fig) # Close empty figure if subplots were created
        return

    fig.suptitle(f"Grad-CAM for {LABEL_COLUMN_NAME} Model (Targeting Various Classes)", fontsize=12) # MODIFIED
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_filename = f'grad_cam_{LABEL_COLUMN_NAME.lower().replace('-', '_')}_binary.png' # MODIFIED: Filename consistent replace
    plt.savefig(save_filename)
    print(f"Grad-CAM grid for {LABEL_COLUMN_NAME} saved to {save_filename}") # MODIFIED
    plt.show()


if 'model' in locals() and 'val_ds' in locals() and len(val_ds) > 0: # MODIFIED to use val_ds
    print(f"\nVisualizing Grad-CAM for {LABEL_COLUMN_NAME} model") # MODIFIED
    visualize_grad_cam_ki67(model, dataset=val_ds, device=device, num_images=4, target_classes_to_viz=[0,1]) # MODIFIED: target_classes_to_viz for binary and use val_ds
else:
    print(f"Skipping {LABEL_COLUMN_NAME} Grad-CAM: Model or validation dataset not available or val_ds is empty.") # MODIFIED to check val_ds

# %%
print(f"{LABEL_COLUMN_NAME} binary classification model script generation complete.") # MODIFIED
print(f"IMPORTANT: Review and adjust 'LABEL_COLUMN_NAME' ('{LABEL_COLUMN_NAME}'), and the 'map_ki67_to_binary' function if your Ki-67 labels (0/1) need different parsing.") # MODIFIED
print(f"Also, review FocalLoss 'alpha' parameters and data loading for {LABEL_COLUMN_NAME} task.") # MODIFIED

# %%

# %% [markdown]
# ## 特征与标签相关性分析 (互信息, t-SNE, 置换检验)

# %% 
# ==> Add the new analysis functions here:

def get_embeddings_convnext(model, dataloader, device):
    model.eval()
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=f"Extracting embeddings for {LABEL_COLUMN_NAME}"):
            imgs = imgs.to(device)
            # Get features from the layer before the final classifier
            # For ConvNeXt, model.features extracts features, then adaptive_avg_pool2d, then classifier
            # We will take output of adaptive_avg_pool2d as embeddings
            # For EfficientNet, it is usually model.features -> model.avgpool -> model.classifier
            features = model.features(imgs)
            pooled_features = model.avgpool(features) # (batch_size, num_features, 1, 1)
            embeddings = torch.flatten(pooled_features, 1) # (batch_size, num_features)
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
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(embeddings)-1), n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("viridis", len(unique_labels))
    
    for i, label_val in enumerate(unique_labels):
        idx = labels == label_val
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], color=colors(i), label=f'{LABEL_COLUMN_NAME} {label_val}', alpha=0.7)
    
    plt.title(f't-SNE 可视化 ({LABEL_COLUMN_NAME}{title_suffix})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"tsne_visualization_{LABEL_COLUMN_NAME.lower().replace('-', '_')}{title_suffix.replace(' ', '_').lower()}.png")
    plt.show()

def simulate_data_cleaning_test(model, original_val_loader, original_labels_np, num_samples_to_flip=100, random_state=SEED):
    print(f"\nSimulating data cleaning test for {LABEL_COLUMN_NAME} by flipping {num_samples_to_flip} labels...")
    
    if len(original_labels_np) < num_samples_to_flip:
        print(f"Warning: Not enough samples ({len(original_labels_np)}) to flip {num_samples_to_flip}. Skipping simulation.")
        return float('nan')

    flipped_labels_np = original_labels_np.copy()
    indices_to_flip = np.random.choice(len(flipped_labels_np), num_samples_to_flip, replace=False)
    
    flipped_labels_np[indices_to_flip] = 1 - flipped_labels_np[indices_to_flip]
    
    global all_val_probs_class1_final_np # Ensure we are using the global var if it exists
    if 'all_val_probs_class1_final_np' not in globals() or all_val_probs_class1_final_np is None or len(all_val_probs_class1_final_np) != len(flipped_labels_np):
        print("Warning: `all_val_probs_class1_final_np` not available or mismatched. Re-evaluating model for data cleaning test.")
        model.eval()
        temp_probs_list = []
        with torch.no_grad():
            for imgs, _ in tqdm(original_val_loader, desc=f"Re-evaluating for data cleaning test {LABEL_COLUMN_NAME}"):
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                temp_probs_list.extend(probs[:, 1].cpu().numpy())
        eval_probs_np = np.array(temp_probs_list)
    else:
        eval_probs_np = all_val_probs_class1_final_np

    if len(np.unique(flipped_labels_np)) < 2:
        print(f"Warning: After flipping, only one class present in simulated labels for {LABEL_COLUMN_NAME}. AUC will be undefined.")
        return float('nan')
        
    try:
        auc_after_cleaning = roc_auc_score(flipped_labels_np, eval_probs_np)
        print(f"AUC after simulated cleaning for {LABEL_COLUMN_NAME} ({num_samples_to_flip} labels flipped): {auc_after_cleaning:.4f}")
        return auc_after_cleaning
    except ValueError as e:
        print(f"Error calculating AUC after simulated cleaning for {LABEL_COLUMN_NAME}: {e}")
        return float('nan')

def perform_permutation_test(model, val_loader, original_labels_np, original_probs_np, n_permutations=1000):
    print(f"\nPerforming permutation test for {LABEL_COLUMN_NAME} with {n_permutations} permutations...")
    
    if len(np.unique(original_labels_np)) < 2:
        print(f"Warning: Original labels for {LABEL_COLUMN_NAME} have less than 2 unique classes. Permutation test might not be meaningful.")

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
    p_value = np.mean(permuted_aucs >= observed_auc)
    
    print(f"Permutation test for {LABEL_COLUMN_NAME}: p-value = {p_value:.4f}")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(permuted_aucs, bins=30, kde=True, label='Permuted AUCs')
    plt.axvline(observed_auc, color='red', linestyle='--', lw=2, label=f'Observed AUC ({observed_auc:.3f})')
    plt.title(f'Permutation Test Results for {LABEL_COLUMN_NAME}')
    plt.xlabel('AUC Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"permutation_test_{LABEL_COLUMN_NAME.lower().replace('-', '_')}.png")
    plt.show()
    
    return p_value

# --- Main execution of the new analyses ---
if 'model' in locals() and 'val_loader' in locals() and len(val_ds) > 0:
    print(f"\n--- Starting Feature-Label Relevance Analysis for {LABEL_COLUMN_NAME} ---")
    
    try:
        model.load_state_dict(torch.load(f"best_model_{LABEL_COLUMN_NAME.lower().replace('-', '_')}.pth"))
        model.to(device) 
        print(f"Loaded best {LABEL_COLUMN_NAME} model for feature-label analysis.")

        global all_val_labels_final_np, all_val_probs_class1_final_np # Make sure these are accessible if already computed
        if 'all_val_labels_final_np' not in globals() or 'all_val_probs_class1_final_np' not in globals() or \
            all_val_labels_final_np is None or all_val_probs_class1_final_np is None or \
            len(all_val_labels_final_np) != len(val_ds) or len(all_val_probs_class1_final_np) != len(val_ds):
            
            print("Recalculating final validation labels and probabilities for analysis...")
            all_val_labels_final_list_analysis = []
            all_val_probs_class1_final_list_analysis = []
            model.eval()
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc=f"Final Validation for Analysis {LABEL_COLUMN_NAME}"):
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    probs = torch.softmax(outputs, dim=1)
                    all_val_probs_class1_final_list_analysis.extend(probs[:, 1].cpu().numpy())
                    all_val_labels_final_list_analysis.extend(labels.numpy())
            all_val_labels_final_np = np.array(all_val_labels_final_list_analysis)
            all_val_probs_class1_final_np = np.array(all_val_probs_class1_final_list_analysis)

        val_embeddings, val_true_labels_for_analysis = get_embeddings_convnext(model, val_loader, device) # MODIFIED: Ensure this uses efficientnet logic if needed
        
        if val_embeddings is not None and len(val_embeddings) > 0 and len(val_true_labels_for_analysis) == len(val_embeddings):
            if all_val_probs_class1_final_np is not None and len(all_val_probs_class1_final_np) == len(val_true_labels_for_analysis):
                mi_scores_probs = calculate_mutual_information(all_val_probs_class1_final_np.reshape(-1, 1), val_true_labels_for_analysis)
                print(f"Mutual Information (Class 1 Probs vs Labels) for {LABEL_COLUMN_NAME}: {mi_scores_probs[0]:.4f}")
            else:
                print("Could not calculate MI with probabilities, not available or mismatched length.")

            plot_tsne_visualization(val_embeddings, val_true_labels_for_analysis)

            simulate_data_cleaning_test(model, val_loader, all_val_labels_final_np, num_samples_to_flip=min(100, len(all_val_labels_final_np) // 2 if len(all_val_labels_final_np) > 0 else 1))
            
            if all_val_labels_final_np is not None and all_val_probs_class1_final_np is not None and len(all_val_labels_final_np)>0:
                perform_permutation_test(model, val_loader, all_val_labels_final_np, all_val_probs_class1_final_np, n_permutations=1000)
            else:
                print(f"Skipping Permutation Test for {LABEL_COLUMN_NAME} due to missing validation labels or probabilities.")
        else:
            print(f"Could not extract embeddings or labels for {LABEL_COLUMN_NAME} analysis. Skipping.")
            
    except FileNotFoundError:
        print(f"Error: 'best_model_{LABEL_COLUMN_NAME.lower().replace('-', '_')}.pth' not found. Cannot perform feature-label analysis for {LABEL_COLUMN_NAME}.")
    except Exception as e_analysis:
        print(f"An error occurred during {LABEL_COLUMN_NAME} feature-label relevance analysis: {e_analysis}")
else:
    print(f"Skipping Feature-Label Relevance Analysis for {LABEL_COLUMN_NAME}: Model or validation data not available or val_ds is empty.")


# %%
