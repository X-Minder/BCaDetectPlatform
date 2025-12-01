# %% [markdown]
# # PD-L1(Pemb) 膀胱镜图像二分类模型
# 通过迁移学习（EfficientNet-B0）对 PD-L1(Pemb) 进行二分类，按患者划分训练/验证/测试集，尽量避免同一患者图像落到同一集合。

# %%
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support, average_precision_score, precision_recall_curve
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

# NEW Imports from grade_model.py
from pytorch_grad_cam import GradCAM # Keep for later Grad-CAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget # Keep for later
from pytorch_grad_cam.utils.image import show_cam_on_image # Keep for later
from torchvision.utils import make_grid, save_image # Keep for later Grad-CAM
from sklearn.feature_selection import mutual_info_classif # For feature analysis
from sklearn.manifold import TSNE # For feature analysis
from sklearn.utils import shuffle as sklearn_shuffle # For permutation test
from sklearn.model_selection import train_test_split # For fallback split

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

# Define constants for column names and task (NEW, from grade_model.py structure)
TASK_NAME = "PDL1_Pemb"
LABEL_COLUMN_NAME = 'PD-L1(Pemb)' # Original label column
MAPPED_LABEL_COLUMN_NAME = 'PD-L1(Pemb)_binary' # Target binary label column
PATIENT_ID_COLUMN = 'PATIENT_ID'
FILE_NAME_COLUMN = 'FILE_NAME'
NUM_CLASSES = 2
CLASS_NAMES = ['PD-L1-', 'PD-L1+'] # For reports and plots

# %% [markdown]
# ## 自定义 Dataset

# %%
class PDL1Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None): # df now expected to have MAPPED_LABEL_COLUMN_NAME
        """
        df: 包含 FILE_NAME_COLUMN, MAPPED_LABEL_COLUMN_NAME, PATIENT_ID_COLUMN
        image_dir: 图像根路径
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.file_name_col = FILE_NAME_COLUMN
        self.label_col = MAPPED_LABEL_COLUMN_NAME

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row[self.file_name_col])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row[self.label_col])
        return img, label

# %% [markdown]
# ## 读取标签并划分数据集

# %%
# 1. 读取 CSV
label_df_raw = pd.read_csv("dataset/label.csv")
# 2. 丢弃关键列NA值 (LABEL_COLUMN_NAME is the original PD-L1(Pemb) score column)
label_df = label_df_raw.dropna(subset=[LABEL_COLUMN_NAME, FILE_NAME_COLUMN, PATIENT_ID_COLUMN]).copy()
# 3. 二分类标签 (MAPPED_LABEL_COLUMN_NAME is PD-L1(Pemb)_binary)
# Ensure original label column 'PD-L1(Pemb)' is numeric before comparison
label_df[LABEL_COLUMN_NAME] = pd.to_numeric(label_df[LABEL_COLUMN_NAME], errors='coerce')
label_df = label_df.dropna(subset=[LABEL_COLUMN_NAME]) # Drop if conversion to numeric failed

label_df[MAPPED_LABEL_COLUMN_NAME] = (label_df[LABEL_COLUMN_NAME] > 0).astype(int)

print(f"Initial data: {len(label_df_raw)} rows. After cleaning (NA, valid labels): {len(label_df)} rows.")
print(f"Class distribution in full cleaned dataset for '{MAPPED_LABEL_COLUMN_NAME}':\n{label_df[MAPPED_LABEL_COLUMN_NAME].value_counts(normalize=True)}")


# MODIFIED SPLIT: df_trainval is the full dataset, then split into train and val. Test set is separate.
df_full_data_for_training = label_df.copy()

# First, reserve a test set from the full data, stratified by patient if possible.
gss_test_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
if len(df_full_data_for_training[PATIENT_ID_COLUMN].unique()) > 1 and len(df_full_data_for_training) > 1:
    try:
        trainval_indices, test_indices = next(gss_test_split.split(df_full_data_for_training, groups=df_full_data_for_training[PATIENT_ID_COLUMN]))
        df_trainval_temp = df_full_data_for_training.iloc[trainval_indices].copy()
        df_test_temp     = df_full_data_for_training.iloc[test_indices].copy()

        # Check if df_trainval_temp from GSS has enough unique classes, provided df_full_data_for_training has them
        if (NUM_CLASSES > 1 and 
            not df_trainval_temp.empty and 
            df_full_data_for_training[MAPPED_LABEL_COLUMN_NAME].nunique() >= NUM_CLASSES and 
            df_trainval_temp[MAPPED_LABEL_COLUMN_NAME].nunique() < NUM_CLASSES):
            print(f"Warning: GSS for trainval/test produced a df_trainval with {df_trainval_temp[MAPPED_LABEL_COLUMN_NAME].nunique()} unique class(es) for {TASK_NAME} (expected {NUM_CLASSES} if source has them). Attempting fallback.")
            raise ValueError("Fallback to stratified split for trainval/test: GSS df_trainval has insufficient class diversity.")
        
        df_trainval = df_trainval_temp
        df_test     = df_test_temp
        print("Successfully used GroupShuffleSplit for initial trainval/test split.")

    except ValueError: # Fallback if GroupShuffleSplit fails or forced fallback due to class imbalance in df_trainval_temp
        print(f"Warning: GroupShuffleSplit for trainval/test for {TASK_NAME} failed or fallback triggered. Using random stratified split for trainval/test.")
        stratify_col_tts = df_full_data_for_training[MAPPED_LABEL_COLUMN_NAME] if df_full_data_for_training[MAPPED_LABEL_COLUMN_NAME].nunique() > 1 else None
        df_trainval, df_test = train_test_split(df_full_data_for_training, test_size=0.2, random_state=SEED, stratify=stratify_col_tts)
else: # Not enough groups or samples for GroupShuffleSplit
    print(f"Warning: Not enough unique patient groups for {TASK_NAME} for GroupShuffleSplit (trainval/test). Using random stratified split.")
    stratify_col_tts = df_full_data_for_training[MAPPED_LABEL_COLUMN_NAME] if df_full_data_for_training[MAPPED_LABEL_COLUMN_NAME].nunique() > 1 else None
    df_trainval, df_test = train_test_split(df_full_data_for_training, test_size=0.2, random_state=SEED, stratify=stratify_col_tts)


# Now, split df_trainval into train and validation sets
val_size_from_trainval = 0.2 # e.g., 20% of trainval for validation
gss_val_split = GroupShuffleSplit(n_splits=1, test_size=val_size_from_trainval, random_state=SEED + 1) # Use different seed for this split

if len(df_trainval[PATIENT_ID_COLUMN].unique()) > 1 and len(df_trainval) > 1 :
    try:
        train_indices, val_indices = next(gss_val_split.split(df_trainval, groups=df_trainval[PATIENT_ID_COLUMN]))
        df_train_temp = df_trainval.iloc[train_indices].copy()
        df_val_temp   = df_trainval.iloc[val_indices].copy()

        # Check if validation set from GSS has enough unique classes
        if NUM_CLASSES > 1 and not df_val_temp.empty and df_val_temp[MAPPED_LABEL_COLUMN_NAME].nunique() < NUM_CLASSES:
            print(f"Warning: GroupShuffleSplit for train/validation produced a validation set with {df_val_temp[MAPPED_LABEL_COLUMN_NAME].nunique()} unique class(es) for {TASK_NAME} (expected {NUM_CLASSES}). Attempting fallback to stratified split.")
            raise ValueError("Fallback to stratified split: GSS validation set has insufficient class diversity.")

        df_train = df_train_temp
        df_val   = df_val_temp
        if not df_val.empty:
            print(f"Successfully used GroupShuffleSplit for train/validation split from df_trainval. Validation set for {TASK_NAME} has {df_val[MAPPED_LABEL_COLUMN_NAME].nunique()} unique classes.")
        else:
            print(f"Successfully used GroupShuffleSplit for train/validation split from df_trainval. Validation set for {TASK_NAME} is empty.")

    except ValueError as e_gss_val: # Catches GSS internal errors or our custom raise
        print(f"Info: GroupShuffleSplit for train/validation for {TASK_NAME} triggered fallback: {e_gss_val}. Using random stratified split for train/val from df_trainval.")
        stratify_col_tv = df_trainval[MAPPED_LABEL_COLUMN_NAME] if df_trainval[MAPPED_LABEL_COLUMN_NAME].nunique() > 1 else None
        df_train, df_val = train_test_split(df_trainval, test_size=val_size_from_trainval, random_state=SEED + 1, stratify=stratify_col_tv)
elif len(df_trainval) > 0:
    print(f"Warning: Not enough unique patient groups in df_trainval for {TASK_NAME} for GroupShuffleSplit (train/val). Using random stratified split.")
    stratify_col_tv = df_trainval[MAPPED_LABEL_COLUMN_NAME] if df_trainval[MAPPED_LABEL_COLUMN_NAME].nunique() > 1 else None
    df_train, df_val = train_test_split(df_trainval, test_size=val_size_from_trainval, random_state=SEED + 1, stratify=stratify_col_tv)
else: # df_trainval is empty
    df_train = pd.DataFrame(columns=df_full_data_for_training.columns)
    df_val = pd.DataFrame(columns=df_full_data_for_training.columns)
    print(f"Warning: df_trainval is empty. Train and Val sets for {TASK_NAME} will be empty.")


print(f"\nDataset sizes and class distributions ({TASK_NAME}):")
for name, df_subset in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    if not df_subset.empty:
        print(f"  {name:<8}: {len(df_subset):>4} images, Patients: {df_subset[PATIENT_ID_COLUMN].nunique():>2}")
        distribution_info_series = df_subset[MAPPED_LABEL_COLUMN_NAME].value_counts(normalize=True).sort_index()
        distribution_info_str = '\\n'.join([f"    Class {idx}: {val:.4f}" for idx, val in distribution_info_series.items()])
        print(f"    Class distribution ({MAPPED_LABEL_COLUMN_NAME}, normalized):\\n{distribution_info_str}")
        print(f"    Unique patients per class ({MAPPED_LABEL_COLUMN_NAME}):")
        if MAPPED_LABEL_COLUMN_NAME in df_subset.columns and PATIENT_ID_COLUMN in df_subset.columns:
            for class_label in sorted(df_subset[MAPPED_LABEL_COLUMN_NAME].unique()):
                num_patients_in_class = df_subset[df_subset[MAPPED_LABEL_COLUMN_NAME] == class_label][PATIENT_ID_COLUMN].nunique()
                print(f"      Class {class_label}: {num_patients_in_class} patients")
        else:
            print("      Could not calculate unique patients per class (column missing).")
    else:
        print(f"  {name:<8}: Empty")
print("\\n")

# %% [markdown]
# ## 数据增强与 DataLoader

# %%
IMG_DIR = "dataset/image"  # 图像文件夹 (Using FILE_NAME_COLUMN with this)

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

# WeightedRandomSampler for training set
train_loader_args = {'shuffle': True, 'batch_size':16, 'num_workers':0, 'pin_memory':True}
if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
    counts_train = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    if len(counts_train) >= 1 and len(counts_train) <= NUM_CLASSES: # Should be NUM_CLASSES for binary
        # Calculate weights for each class: 1.0 / count
        # Ensure all NUM_CLASSES are represented in class_sample_weights, even if not in counts_train
        class_weights_values = [1.0 / counts_train.get(i, 1e-6) for i in range(NUM_CLASSES)] # Get count or use a small number to avoid div by zero

        sample_weights_train = [class_weights_values[label] for label in df_train[MAPPED_LABEL_COLUMN_NAME]]
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights_train),
                                num_samples=len(sample_weights_train),
                                replacement=True)
        weights_str = ", ".join([f"Class {i}: {w:.4f}" for i, w in enumerate(class_weights_values)])
        print(f"Sampler weights for {TASK_NAME} classes: {weights_str}")
        train_loader_args = {'sampler': sampler, 'shuffle': False, 'batch_size':16, 'num_workers':0, 'pin_memory':True} # shuffle is False when sampler is used
    else:
        print(f"Warning: Training data for {TASK_NAME} has insufficient or unexpected class counts for sampler. Using standard DataLoader with shuffle.")
else:
    print(f"Warning: df_train for {TASK_NAME} is empty or mapped label column is missing. Using standard DataLoader.")
    if df_train.empty:
        train_loader_args['shuffle'] = False # Avoid error with empty DataLoader
        print("   df_train is empty. Forcing shuffle=False for train_loader.")


# Dataset & DataLoader
train_ds = PDL1Dataset(df_train, IMG_DIR, transform=train_tf)
val_ds   = PDL1Dataset(df_val,   IMG_DIR, transform=val_tf)
test_ds  = PDL1Dataset(df_test,  IMG_DIR, transform=val_tf) # Keep test_ds for final evaluation

train_loader = DataLoader(train_ds, **train_loader_args)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,   num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False,   num_workers=0, pin_memory=True)

# %% [markdown]
# ## 模型定义与训练设置

# %%
# 迁移学习：EfficientNet-B0
print(f"Using EfficientNet-B0 for {TASK_NAME} {NUM_CLASSES}-class classification")
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

# 修改分类头
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES) # NUM_CLASSES is 2

model = model.to(device)

# Calculate class weights for CrossEntropyLoss (NEW)
class_weights_tensor = None
if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
    counts = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    if len(counts) == NUM_CLASSES:
        # Calculate weights: N_total / (N_classes * N_c)
        # Or simpler: inverse of class frequency, then normalize or use as is if loss function handles it.
        # For CrossEntropyLoss, a common approach is inverse frequency.
        total_samples = counts.sum()
        weights_values = []
        for i in range(NUM_CLASSES):
            count_i = counts.get(i, 1e-6) # Get count or small number to avoid div by zero
            if count_i > 1e-7:
                # weights_values.append(total_samples / (NUM_CLASSES * count_i)) # Option 1
                weights_values.append(1.0 / count_i) # Option 2: Inverse count (simpler, often effective)
            else:
                weights_values.append(1.0) # Default weight if class count is zero/tiny
        
        # If using Option 2, can normalize them, e.g., sum to NUM_CLASSES or sum to 1
        # For now, let's use raw inverse counts and see if it works, or normalize to sum to num_classes
        sum_weights = sum(weights_values)
        if sum_weights > 1e-7:
             normalized_weights = [(w / sum_weights) * NUM_CLASSES for w in weights_values] # Normalize to sum to NUM_CLASSES
        else:
             normalized_weights = [1.0] * NUM_CLASSES
        
        class_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float).to(device)
        weights_str = ", ".join([f"Class {i}: {w:.4f}" for i, w in enumerate(normalized_weights)])
        print(f"Calculated class weights for {TASK_NAME} CrossEntropyLoss: [{weights_str}]")

    elif len(counts) == 1:
        present_class = counts.index[0]
        print(f"Warning: Only class {present_class} present in {TASK_NAME} training data. Class weights for CrossEntropyLoss might be default or skewed.")
    else:
        print(f"Warning: Not all {NUM_CLASSES} classes present or unexpected counts in {TASK_NAME} training data ({len(counts)} found). Using default weights for CrossEntropyLoss.")
else:
    print(f"Warning: Could not calculate class counts for {TASK_NAME} CrossEntropyLoss weights. Using default weights.")


# 损失与优化器
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # NEW
print(f"Using nn.CrossEntropyLoss for {TASK_NAME}" + (f" with class_weights: {class_weights_tensor.tolist()}" if class_weights_tensor is not None else " with default weights."))

optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

# 学习率调度 (MODIFIED: monitor val_loss, mode 'min')
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5, verbose=True, min_lr=1e-7) # Increased patience, reduced min_lr

# %% [markdown]
# ## 早停机制

# %%
# MODIFIED EarlyStopping class from grade_model.py (includes task_name)
class EarlyStopping:
    """早停机制，当验证集性能不再提升时停止训练"""
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True, mode='max', verbose=True, task_name="DefaultTask"): # Added task_name
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        self.task_name = task_name # Store task name
        
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
                print(f"EarlyStopping ({self.task_name}): New best score: {self.best_score:.4f}") # Use task_name in log
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping ({self.task_name}): Counter {self.counter}/{self.patience}. Best score: {self.best_score:.4f}") # Use task_name
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"EarlyStopping ({self.task_name}): Patience reached. Stopping training. Best score: {self.best_score:.4f}") # Use task_name
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print(f"EarlyStopping ({self.task_name}): Restored best model weights.") # Use task_name
        return self.early_stop

# %% [markdown]
# ## 训练与验证循环

# %%
# from sklearn.metrics import roc_auc_score # Already imported

NUM_EPOCHS = 70 # Increased epochs
best_val_metric_for_saving = float('inf') # For val_loss

# MODIFIED EarlyStopping: monitor val_loss
early_stopping = EarlyStopping(patience=15, mode='min', verbose=True, min_delta=0.0001, task_name=TASK_NAME) # Increased patience

# 用于存储历史数据的列表 (add val_accuracy)
history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [], # ADDED
    'val_auc': [],
    'lr': []
}
BEST_MODEL_PATH = f"best_model_{TASK_NAME.lower()}.pth" # Consistent model path

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train {TASK_NAME}]"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    train_loss = running_loss / len(train_ds)
    history['train_loss'].append(train_loss)
    
    model.eval()
    all_val_labels = []
    all_val_probs = [] 
    all_val_preds_list = [] # For accuracy
    val_running_loss = 0.0
    
    if len(val_ds) > 0: # Check if val_ds is not empty
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss_iter = criterion(outputs, labels)
                val_running_loss += val_loss_iter.item() * imgs.size(0)
                
                probs_batch = torch.softmax(outputs, dim=1)
                preds_batch = torch.argmax(probs_batch, dim=1)

                all_val_probs.extend(probs_batch[:, 1].cpu().numpy()) # Prob for positive class
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds_list.extend(preds_batch.cpu().numpy()) # Store predictions
            
        val_epoch_loss = val_running_loss / len(val_ds) if len(val_ds) > 0 else float('nan')
        history['val_loss'].append(val_epoch_loss)
    
        all_val_labels_np = np.array(all_val_labels)
        all_val_preds_np = np.array(all_val_preds_list)
        all_val_probs_np = np.array(all_val_probs)

        val_accuracy = np.mean(all_val_labels_np == all_val_preds_np) if len(all_val_labels_np) > 0 else float('nan')
        history['val_accuracy'].append(val_accuracy)

        val_auc = float('nan')
        if len(np.unique(all_val_labels_np)) > 1: # Need at least 2 classes for AUC
            try:
                val_auc = roc_auc_score(all_val_labels_np, all_val_probs_np)
            except ValueError as e_auc_val:
                print(f"Warning: Epoch {epoch}, Val AUC calculation error for {TASK_NAME}: {e_auc_val}. AUC set to NaN.")
        elif len(all_val_labels_np) > 0:
            print(f"Warning: Epoch {epoch}, Validation set for {TASK_NAME} only has one class, AUC set to NaN")
        history['val_auc'].append(val_auc)

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        scheduler.step(val_epoch_loss) # MODIFIED: scheduler steps on val_epoch_loss
        
        print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_epoch_loss:.4f}, Val Acc={val_accuracy:.4f}, Val AUC={val_auc:.4f}, LR={current_lr:.1e}")
        
        # MODIFIED: Save model based on best val_loss
        if val_epoch_loss < best_val_metric_for_saving: 
            best_val_metric_for_saving = val_epoch_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Epoch {epoch}: New best model saved with Val Loss: {best_val_metric_for_saving:.4f} to {BEST_MODEL_PATH}")

        if early_stopping(val_epoch_loss, model): # MODIFIED: early_stopping on val_epoch_loss
            print(f"Early stopping triggered for {TASK_NAME}.")
            break
    else: # val_ds is empty
        history['val_loss'].append(float('nan'))
        history['val_accuracy'].append(float('nan'))
        history['val_auc'].append(float('nan'))
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Val metrics=N/A (empty val_ds for {TASK_NAME}), LR={current_lr:.1e}")
        # If no validation, early stopping won't trigger based on val_loss. Could save last model or best train_loss model.
        # For now, if no val_ds, model saving based on val_loss won't happen.

# %% [markdown]
# ## 绘制训练过程曲线

# %%
# MODIFIED plot_training_history, similar to grade_model.py's version
def plot_training_history(history, task_name):
    epochs_ran = len(history['train_loss'])
    if epochs_ran == 0:
        print(f"No training history to plot for {task_name}.")
        return
        
    epoch_ticks = range(1, epochs_ran + 1)

    fig, ax1 = plt.subplots(figsize=(14, 7)) # Adjusted figure size

    # Plotting Training and Validation Loss
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epoch_ticks, history['train_loss'], color=color, linestyle='-', marker='o', markersize=3, label='训练损失')
    if 'val_loss' in history and any(not np.isnan(x) for x in history['val_loss']):
        ax1.plot(epoch_ticks, history['val_loss'], color=color, linestyle=':', marker='x', markersize=3, label='验证损失')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Plotting Validation Accuracy and AUC on a second y-axis
    ax2 = ax1.twinx()
    color_auc = 'tab:blue'
    color_acc = 'tab:cyan' # Different color for accuracy
    ax2.set_ylabel('Accuracy / AUC', color=color_auc) # Combined label
    
    if 'val_accuracy' in history and any(not np.isnan(x) for x in history['val_accuracy']):
        ax2.plot(epoch_ticks, history['val_accuracy'], color=color_acc, linestyle='-', marker='s', markersize=3, label='验证准确率')
    if 'val_auc' in history and any(not np.isnan(x) for x in history['val_auc']): 
        ax2.plot(epoch_ticks, history['val_auc'], color=color_auc, linestyle='--', marker='^', markersize=3, label='验证 AUC')
    
    ax2.tick_params(axis='y', labelcolor=color_auc)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.05) # AUC and Accuracy are between 0 and 1

    # Plotting Learning Rate on a third y-axis
    ax3 = ax1.twinx() 
    ax3.spines["right"].set_position(("outward", 60)) # Offset the third y-axis
    color_lr = 'tab:green'
    ax3.set_ylabel('Learning Rate', color=color_lr)
    if 'lr' in history and len(history['lr']) == epochs_ran:
        ax3.plot(epoch_ticks, history['lr'], color=color_lr, linestyle='--', marker='.', markersize=3, label='学习率')
    elif 'lr' in history and len(history['lr']) > 0 : # If LR history exists but length mismatch
        print(f"Warning: LR history length ({len(history['lr'])}) doesn't match epochs_ran ({epochs_ran}) for {task_name}. Plotting available LR history.")
        ax3.plot(range(1, len(history['lr'])+1), history['lr'], color=color_lr, linestyle='--', marker='.', markersize=3, label='学习率 (partial)')


    ax3.tick_params(axis='y', labelcolor=color_lr)
    ax3.legend(loc='lower right')
    ax3.set_yscale('log') # Learning rate is often viewed on a log scale

    fig.tight_layout() # Adjust layout to prevent overlap
    plt.title(f'{task_name} 二分类训练过程监控', fontsize=16)
    
    # Ensure all epoch numbers are shown if not too many
    if epochs_ran <= 20: # Heuristic
        ax1.set_xticks(epoch_ticks)
        
    plt.savefig(f"training_history_{task_name.lower()}.png")
    plt.show()
    plt.close(fig)

if any(history.values()): # Check if history dict is not empty
    plot_training_history(history, TASK_NAME)
else:
    print(f"No training history was recorded for {TASK_NAME}, skipping plot.")

# %% [markdown]
# ## 测试集评估

# %% [markdown]
# ### 在验证集上生成最终评估曲线 (ROC & P-R)
# NEW section based on grade_model.py logic

print(f"\n--- Generating ROC and P-R curves for {TASK_NAME} on the VALIDATION set using the best model ---")
if os.path.exists(BEST_MODEL_PATH):
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        model.eval() # Set model to evaluation mode
        print(f"Loaded best {TASK_NAME} model from {BEST_MODEL_PATH} for validation set ROC/PR curve generation.")

        all_val_labels_final_roc_pr = []
        all_val_probs_class1_final_roc_pr = []
        
        if len(val_ds) > 0: # Ensure validation dataset is not empty
            with torch.no_grad(): # Disable gradient calculations
                for imgs, labels in tqdm(val_loader, desc=f"Final Validation ROC/PR for {TASK_NAME}"):
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    probs = torch.softmax(outputs, dim=1) # Get probabilities for each class
                    all_val_probs_class1_final_roc_pr.extend(probs[:, 1].cpu().numpy()) # Probabilities for the positive class (class 1)
                    all_val_labels_final_roc_pr.extend(labels.numpy()) # True labels

            all_val_labels_final_np_roc_pr = np.array(all_val_labels_final_roc_pr)
            all_val_probs_class1_final_np_roc_pr = np.array(all_val_probs_class1_final_roc_pr)

            if len(np.unique(all_val_labels_final_np_roc_pr)) >= 2:
                # ROC Curve for Validation Set
                val_auc_final = roc_auc_score(all_val_labels_final_np_roc_pr, all_val_probs_class1_final_np_roc_pr)
                print(f"Final Validation AUC for {TASK_NAME}: {val_auc_final:.4f}")
                fpr, tpr, _ = roc_curve(all_val_labels_final_np_roc_pr, all_val_probs_class1_final_np_roc_pr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {val_auc_final:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
                plt.xlabel('假阳性率 (False Positive Rate)'); plt.ylabel('真阳性率 (True Positive Rate)')
                plt.title(f'验证集ROC曲线 ({TASK_NAME})'); plt.legend(loc="lower right"); plt.grid(True)
                plt.savefig(f'roc_curve_{TASK_NAME.lower()}_val.png'); plt.show(); plt.close()

                # Precision-Recall Curve for Validation Set
                val_ap_final = average_precision_score(all_val_labels_final_np_roc_pr, all_val_probs_class1_final_np_roc_pr)
                print(f"Final Validation Average Precision for {TASK_NAME}: {val_ap_final:.4f}")
                precision, recall, _ = precision_recall_curve(all_val_labels_final_np_roc_pr, all_val_probs_class1_final_np_roc_pr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, lw=2, label=f'P-R curve (AP = {val_ap_final:.3f})')
                plt.xlabel('召回率 (Recall)'); plt.ylabel('精确率 (Precision)')
                plt.title(f'验证集P-R曲线 ({TASK_NAME})'); plt.legend(loc="best"); plt.grid(True)
                plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
                plt.savefig(f'pr_curve_{TASK_NAME.lower()}_val.png'); plt.show(); plt.close()
            else:
                print(f"Final Validation ROC/PR not computed for {TASK_NAME}: validation set needs at least 2 classes.")
        else:
            print(f"Validation dataset for {TASK_NAME} is empty. No final ROC/PR curves generated for validation set.")
    except Exception as e_val_curves:
        print(f"An error occurred during {TASK_NAME} validation set ROC/PR curve generation: {e_val_curves}")
else:
    print(f"Best model path {BEST_MODEL_PATH} not found. Skipping validation set ROC/PR curve generation.")

# %% [markdown]
# ### 特征与标签相关性分析 (互信息, t-SNE, 置换检验) - 在验证集上
# NEW section based on grade_model.py

print(f"\n--- Starting Feature-Label Relevance Analysis for {TASK_NAME} on VALIDATION set ---")
if os.path.exists(BEST_MODEL_PATH) and len(val_ds) > 0:
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        model.to(device) # Ensure model is on the correct device
        model.eval()
        print(f"Loaded best {TASK_NAME} model from {BEST_MODEL_PATH} for feature-label analysis.")

        # Re-use validation labels and probabilities if already computed for ROC/PR curves
        if 'all_val_labels_final_np_roc_pr' in globals() and \
           'all_val_probs_class1_final_np_roc_pr' in globals() and \
           all_val_labels_final_np_roc_pr is not None and \
           all_val_probs_class1_final_np_roc_pr is not None and \
           len(all_val_labels_final_np_roc_pr) == len(val_ds) and \
           len(all_val_probs_class1_final_np_roc_pr) == len(val_ds):
            print("Re-using previously computed validation labels and probabilities for analysis.")
            analysis_val_labels_np = all_val_labels_final_np_roc_pr
            analysis_val_probs_class1_np = all_val_probs_class1_final_np_roc_pr
        else:
            print("Re-calculating validation labels and probabilities for analysis...")
            analysis_val_labels_list = []
            analysis_val_probs_class1_list = []
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc=f"Recalculating Validation Data for Analysis {TASK_NAME}"):
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    probs = torch.softmax(outputs, dim=1)
                    analysis_val_probs_class1_list.extend(probs[:, 1].cpu().numpy())
                    analysis_val_labels_list.extend(labels.numpy())
            analysis_val_labels_np = np.array(analysis_val_labels_list)
            analysis_val_probs_class1_np = np.array(analysis_val_probs_class1_list)

        # 1. Get Embeddings (Adapting from get_embeddings_convnext for EfficientNet-B0)
        def get_embeddings_pdl1(model, dataloader, device, task_name):
            model.eval()
            embeddings_list = []
            labels_list = []
            # EfficientNet-B0: features -> adaptive_avg_pool -> classifier (dropout, linear)
            # We want the output of the adaptive_avg_pool layer
            activation_hook_output = None
            def hook_fn(module, input, output):
                nonlocal activation_hook_output
                activation_hook_output = output
            
            avg_pool_layer = None
            # torchvision EfficientNet uses model.avgpool before the classifier
            if hasattr(model, 'avgpool') and isinstance(model.avgpool, nn.AdaptiveAvgPool2d):
                avg_pool_layer = model.avgpool
            else:
                print(f"Warning for {task_name} get_embeddings: model.avgpool (AdaptiveAvgPool2d) not found. Will use model.features + manual global avg pool.")

            hook_handle = None
            if avg_pool_layer:
                hook_handle = avg_pool_layer.register_forward_hook(hook_fn)
            
            with torch.no_grad():
                for imgs, labels in tqdm(dataloader, desc=f"Extracting embeddings for {task_name}"):
                    imgs = imgs.to(device)
                    if avg_pool_layer:
                        # Pass through the full model to trigger the hook correctly, up to the hooked layer.
                        # If classifier is complex, just passing through features might not be enough context for some hooks.
                        # However, for avgpool right before classifier, model(imgs) is fine if hook is on avgpool.
                        _ = model(imgs) # Full forward pass to ensure hook on avgpool is triggered
                        pooled_features = activation_hook_output
                        if pooled_features is None: # Should not happen if hook is correct and layer is hit
                            print(f"Error: Activation hook for {task_name} did not capture output. Falling back.")
                            features = model.features(imgs)
                            pooled_features = F.adaptive_avg_pool2d(features, (1,1))
                    else: # Manual feature extraction and pooling if no hook
                        features = model.features(imgs)
                        pooled_features = F.adaptive_avg_pool2d(features, (1,1))

                    embeddings = torch.flatten(pooled_features, 1)
                    embeddings_list.append(embeddings.cpu().numpy())
                    labels_list.append(labels.numpy())
            
            if hook_handle: hook_handle.remove()
            if not embeddings_list: return np.array([]), np.array([])
            return np.concatenate(embeddings_list), np.concatenate(labels_list)

        val_embeddings, val_true_labels_for_analysis = get_embeddings_pdl1(model, val_loader, device, TASK_NAME)
        
        if val_embeddings.size > 0 and val_true_labels_for_analysis.size > 0 and len(val_embeddings) == len(val_true_labels_for_analysis):
            # 2. Calculate Mutual Information
            def calculate_mutual_information_pdl1(features_or_probs, labels, task_name):
                if features_or_probs.ndim == 1: features_or_probs = features_or_probs.reshape(-1, 1)
                if len(features_or_probs) == 0 or len(labels) == 0: return np.array([0.0]) # Handle empty input
                mi = mutual_info_classif(features_or_probs, labels, random_state=SEED)
                return mi

            if analysis_val_probs_class1_np is not None and len(analysis_val_probs_class1_np) == len(val_true_labels_for_analysis):
                mi_scores_probs = calculate_mutual_information_pdl1(analysis_val_probs_class1_np.reshape(-1, 1), val_true_labels_for_analysis, TASK_NAME)
                print(f"Mutual Information (Class 1 Probs vs Labels) for {TASK_NAME}: {mi_scores_probs[0]:.4f}")
            
            # mi_scores_embeddings = calculate_mutual_information_pdl1(val_embeddings, val_true_labels_for_analysis, TASK_NAME)
            # print(f"Mean Mutual Information (Embeddings vs Labels) for {TASK_NAME}: {np.mean(mi_scores_embeddings):.4f}")

            # 3. t-SNE Visualization
            def plot_tsne_visualization_pdl1(embeddings, labels, task_name, title_suffix=""):
                if len(embeddings) == 0: print(f"Cannot run t-SNE for {task_name}: No embeddings."); return
                print(f"Running t-SNE for {task_name}...")
                perplexity_val = min(30.0, float(len(embeddings) - 1) if len(embeddings) > 1 else 1.0)
                if perplexity_val <=0: print(f"Perplexity for t-SNE is {perplexity_val}, skipping t-SNE."); return

                tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity_val, max_iter=1000, init='pca', learning_rate='auto') # Changed n_iter to max_iter
                embeddings_2d = tsne.fit_transform(embeddings)
                
                plt.figure(figsize=(10, 8))
                unique_labels_viz = np.unique(labels)
                # colors = plt.cm.get_cmap("viridis", len(unique_labels_viz)) # Deprecated
                cmap = plt.colormaps.get_cmap("viridis") 
                colors_viz = cmap(np.linspace(0, 1, len(unique_labels_viz)))

                for i, label_val in enumerate(unique_labels_viz):
                    idx = labels == label_val
                    class_display_name = CLASS_NAMES[label_val] if label_val < len(CLASS_NAMES) else f"Class {label_val}"
                    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], color=colors_viz[i], label=class_display_name, alpha=0.7)
                
                plt.title(f't-SNE 可视化 ({task_name}{title_suffix})')
                plt.xlabel('t-SNE Component 1'); plt.ylabel('t-SNE Component 2')
                plt.legend(); plt.grid(True)
                plt.savefig(f"tsne_visualization_{task_name.lower()}{title_suffix.replace(' ', '_').lower()}.png"); plt.show(); plt.close()

            plot_tsne_visualization_pdl1(val_embeddings, val_true_labels_for_analysis, TASK_NAME)

            # 4. Simulate Data Cleaning Test
            def simulate_data_cleaning_test_pdl1(model, original_val_loader, original_labels_np_sim, probabilities_class1_np_sim, task_name, num_samples_to_flip=100):
                print(f"\nSimulating data cleaning test for {task_name} by flipping {num_samples_to_flip} labels...")
                if len(original_labels_np_sim) < num_samples_to_flip: print(f"Warning: Not enough samples for {task_name} to flip. Skipping sim."); return float('nan')
                if len(probabilities_class1_np_sim) != len(original_labels_np_sim): print("Probabilities and labels length mismatch for sim."); return float('nan')

                flipped_labels_np = original_labels_np_sim.copy()
                # Ensure we have at least one of each class if possible to avoid issues with choice, or handle if not.
                if len(np.unique(original_labels_np_sim)) < 2 : # If only one class originally, flipping has no meaning for binary AUC
                    print(f"Warning: Original labels for {task_name} have only one class. Flipping simulation might be misleading.")
                    # We can still proceed, but the interpretation of AUC change is tricky.
                
                indices_to_flip = np.random.choice(len(flipped_labels_np), num_samples_to_flip, replace=False)
                flipped_labels_np[indices_to_flip] = 1 - flipped_labels_np[indices_to_flip]
                
                if len(np.unique(flipped_labels_np)) < 2: print(f"Warning: After flipping, only one class in simulated labels for {task_name}. AUC undefined."); return float('nan')
                try:
                    auc_after_cleaning = roc_auc_score(flipped_labels_np, probabilities_class1_np_sim)
                    print(f"AUC after simulated cleaning for {task_name} ({num_samples_to_flip} labels flipped): {auc_after_cleaning:.4f}")
                    return auc_after_cleaning
                except ValueError as e_sim: print(f"Error in AUC calc after sim cleaning for {task_name}: {e_sim}"); return float('nan')
            
            num_to_flip = min(max(1, len(analysis_val_labels_np) // 10 if len(analysis_val_labels_np) > 0 else 1), 50) 
            if analysis_val_probs_class1_np is not None and analysis_val_labels_np is not None:
                simulate_data_cleaning_test_pdl1(model, val_loader, analysis_val_labels_np, analysis_val_probs_class1_np, TASK_NAME, num_samples_to_flip=num_to_flip)
            else: print("Skipping data cleaning simulation due to missing val probs or labels.")
            
            # 5. Perform Permutation Test
            def perform_permutation_test_pdl1(model, val_loader, original_labels_np_perm, original_probs_np_perm, task_name, n_permutations=1000):
                print(f"\nPerforming permutation test for {task_name} with {n_permutations} permutations...")
                if len(np.unique(original_labels_np_perm)) < 2: print(f"Warning: Original labels for {task_name} have < 2 unique classes. Permutation test may not be meaningful.")
                if len(original_probs_np_perm) != len(original_labels_np_perm): print("Probabilities and labels length mismatch for permutation."); return float('nan')

                try: observed_auc = roc_auc_score(original_labels_np_perm, original_probs_np_perm)
                except ValueError: print(f"Could not calculate observed AUC for {task_name}. Permutation test skipped."); return float('nan')
                print(f"Observed AUC for {task_name}: {observed_auc:.4f}")
                
                permuted_aucs = []
                for i in tqdm(range(n_permutations), desc=f"Permutation Test {task_name}"):
                    permuted_labels = sklearn_shuffle(original_labels_np_perm, random_state=SEED + i)
                    if len(np.unique(permuted_labels)) < 2: permuted_aucs.append(0.5); continue
                    try: permuted_aucs.append(roc_auc_score(permuted_labels, original_probs_np_perm))
                    except ValueError: permuted_aucs.append(0.5)

                permuted_aucs_np = np.array(permuted_aucs)
                p_value = np.mean(permuted_aucs_np >= observed_auc)
                print(f"Permutation test for {task_name}: p-value = {p_value:.4f}")
                
                plt.figure(figsize=(10, 6))
                sns.histplot(permuted_aucs_np, bins=30, kde=True, label='Permuted AUCs')
                plt.axvline(observed_auc, color='red', linestyle='--', lw=2, label=f'Observed AUC ({observed_auc:.3f})')
                plt.title(f'Permutation Test Results for {task_name}'); plt.xlabel('AUC Score'); plt.ylabel('Frequency')
                plt.legend(); plt.savefig(f"permutation_test_{task_name.lower()}.png"); plt.show(); plt.close()
                return p_value

            if analysis_val_labels_np is not None and analysis_val_probs_class1_np is not None and len(analysis_val_labels_np)>0:
                perform_permutation_test_pdl1(model, val_loader, analysis_val_labels_np, analysis_val_probs_class1_np, TASK_NAME, n_permutations=1000)
            else:
                print(f"Skipping Permutation Test for {TASK_NAME} due to missing validation labels or probabilities.")
        else:
            print(f"Could not extract embeddings or labels for {TASK_NAME} analysis. Skipping some parts of feature-label analysis.")
            
    except FileNotFoundError:
        print(f"Error: {BEST_MODEL_PATH} not found. Cannot perform feature-label analysis for {TASK_NAME}.")
    except Exception as e_analysis_main:
        print(f"An error occurred during {TASK_NAME} feature-label relevance analysis main block: {e_analysis_main}")
else:
    print(f"Skipping Feature-Label Relevance Analysis for {TASK_NAME}: Best model path {BEST_MODEL_PATH} not found, or val_ds is empty.")

# %% [markdown]
# ### Grad-CAM 可视化 - 在验证集上
# NEW section based on grade_model.py

print(f"\n--- Visualizing Grad-CAM for {TASK_NAME} model on VALIDATION set ---")

def visualize_grad_cam_pdl1(model, dataset, device, task_name, class_names_viz, num_images=4, target_classes_to_viz=None):
    target_layer_module = None
    # For torchvision EfficientNet-B0, model.features[-1] is the last _MBConvBlock.
    # A good general target within this block could be its final conv layer.
    # Let's try to be more specific if possible, or fallback to model.features[-1]
    if hasattr(model, 'features') and model.features is not None and len(model.features) > 0:
        last_block = model.features[-1] # This is usually an _MBConvBlock
        # Inside _MBConvBlock, `block` is a Sequential. We want a Conv2d from it.
        if hasattr(last_block, 'block') and isinstance(last_block.block, nn.Sequential) and len(last_block.block) > 0:
            # Iterate backwards through the layers in the block to find the last Conv2d
            for layer_in_block in reversed(list(last_block.block)):
                if isinstance(layer_in_block, nn.Conv2d):
                    target_layer_module = layer_in_block
                    break
                # Sometimes Conv2d is wrapped in a Conv2dNormActivation or similar nn.Sequential
                if isinstance(layer_in_block, nn.Sequential) and len(layer_in_block) > 0 and isinstance(layer_in_block[0], nn.Conv2d):
                    target_layer_module = layer_in_block[0]
                    break
        if target_layer_module is None: # If no specific conv found in block, use the whole last block
            target_layer_module = last_block 
    else: # Fallback if features structure is not as expected
        print(f"Warning: Could not access model.features[-1] for {task_name} Grad-CAM. Check model structure.")
        return

    if target_layer_module is None: # Should be caught by return above, but defensive
        print(f"Error: Target layer for Grad-CAM for {task_name} could not be determined. Skipping Grad-CAM.")
        return
        
    print(f"Grad-CAM target layer for {task_name} selected: {type(target_layer_module)}")
    target_layers_for_cam = [target_layer_module]
    cam_obj = GradCAM(model=model, target_layers=target_layers_for_cam)

    if not dataset or len(dataset) == 0: print(f"Dataset for {task_name} Grad-CAM is empty."); return
    if target_classes_to_viz is None: target_classes_to_viz = list(range(NUM_CLASSES))
    
    images_shown_count = 0
    num_viz_rows = len(target_classes_to_viz)
    num_viz_cols = num_images # num_images per target class to try and show
    if num_viz_rows * num_viz_cols == 0: print(f"No images or target classes for {task_name} Grad-CAM."); return

    fig, axes = plt.subplots(num_viz_rows * 2, num_viz_cols, figsize=(num_viz_cols * 3, num_viz_rows * 6), squeeze=False)
    # squeeze=False ensures axes is always 2D array for easier indexing

    total_images_available = len(dataset)
    images_processed_indices = set()

    for r_idx, target_cls_viz in enumerate(target_classes_to_viz):
        img_count_for_this_row = 0
        # Try to find images of the target_cls_viz first, then random if not enough
        candidate_indices_specific_class = [i for i, (_,lbl) in enumerate(dataset) if lbl == target_cls_viz and i not in images_processed_indices]
        random.shuffle(candidate_indices_specific_class) # Shuffle to get different images each run if more than num_viz_cols
        
        candidate_indices_random_fill = [i for i in range(total_images_available) if i not in images_processed_indices and i not in candidate_indices_specific_class]
        random.shuffle(candidate_indices_random_fill)
        
        # Combine specific class images with random fill images
        current_indices_for_row = (candidate_indices_specific_class + candidate_indices_random_fill)[:num_viz_cols]

        for c_idx_local, img_idx_in_dataset in enumerate(current_indices_for_row):
            if img_idx_in_dataset in images_processed_indices or img_count_for_this_row >= num_viz_cols:
                # This check might be redundant if current_indices_for_row is sliced correctly and images_processed_indices is updated
                continue
            images_processed_indices.add(img_idx_in_dataset)
            
            img_tensor, true_label_scalar = dataset[img_idx_in_dataset]
            true_label_viz = true_label_scalar 
            
            inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
            rgb_img_denorm = inv_normalize(img_tensor.clone()).permute(1, 2, 0).cpu().numpy(); rgb_img_denorm = np.clip(rgb_img_denorm, 0, 1)

            input_tensor_unsqueeze = img_tensor.unsqueeze(0).to(device)
            cam_targets_viz = [ClassifierOutputTarget(target_cls_viz)]
            
            grayscale_cam_batch = cam_obj(input_tensor=input_tensor_unsqueeze, targets=cam_targets_viz, aug_smooth=True, eigen_smooth=True)
            if grayscale_cam_batch is None or grayscale_cam_batch.shape[0] == 0: 
                print(f"Grad-CAM returned None/empty for img {img_idx_in_dataset}, target {target_cls_viz}.")
                # Turn off axes for this slot if CAM fails
                axes[r_idx * 2, c_idx_local].axis('off')
                axes[r_idx * 2 + 1, c_idx_local].axis('off')
                continue
            grayscale_cam_img = grayscale_cam_batch[0, :]
            
            cam_image_overlay = show_cam_on_image(rgb_img_denorm, grayscale_cam_img, use_rgb=True)
            original_img_for_display = (rgb_img_denorm * 255).astype(np.uint8)
            
            title_str = f"True: {class_names_viz[true_label_viz]}\nCAM for: {class_names_viz[target_cls_viz]}"

            ax_orig_current = axes[r_idx * 2, c_idx_local]
            ax_cam_current = axes[r_idx * 2 + 1, c_idx_local]

            ax_orig_current.imshow(original_img_for_display); ax_orig_current.set_title(title_str, fontsize=8); ax_orig_current.axis('off')
            ax_cam_current.imshow(cam_image_overlay); ax_cam_current.axis('off')
            images_shown_count +=1
            img_count_for_this_row +=1
        
        # If fewer images were found/processed for this row than num_viz_cols, turn off remaining axes in this row
        for c_idx_remaining in range(img_count_for_this_row, num_viz_cols):
            axes[r_idx * 2, c_idx_remaining].axis('off')
            axes[r_idx * 2 + 1, c_idx_remaining].axis('off')

    if images_shown_count == 0: print(f"No {task_name} CAM images generated."); plt.close(fig); return
    fig.suptitle(f"Grad-CAM for {task_name} Model (Targeting Various Classes)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    save_filename = f'grad_cam_{task_name.lower()}_binary.png' 
    plt.savefig(save_filename); print(f"Grad-CAM grid for {task_name} saved to {save_filename}"); plt.show(); plt.close(fig)

if os.path.exists(BEST_MODEL_PATH) and 'val_ds' in globals() and len(val_ds) > 0:
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.to(device); model.eval()
    visualize_grad_cam_pdl1(model, dataset=val_ds, device=device, task_name=TASK_NAME, class_names_viz=CLASS_NAMES, num_images=4, target_classes_to_viz=[0,1]) 
else:
    print(f"Skipping {TASK_NAME} Grad-CAM: Model or validation dataset not available/empty.") 

# %% [markdown]
# ### 在测试集上进行最终评估
# This section remains for final evaluation on the reserved test_ds
print(f"\n--- Test Set Evaluation for {TASK_NAME} using model from {BEST_MODEL_PATH} ---")
# %%
