# %% [markdown]
# # PD-L1(Nivo) 膀胱镜图像二分类模型
# 通过迁移学习（EfficientNet-B0）对 PD-L1(Nivo) 进行二分类，按患者划分训练/验证/测试集，尽量避免同一患者图像落到同一集合。

# %%
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support, average_precision_score, precision_recall_curve
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.utils import shuffle as sklearn_shuffle # To avoid conflict with other shuffles
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

# Task-specific constants
TASK_NAME = "PDL1_Nivo"
LABEL_COLUMN_NAME = 'PD-L1(Nivo)' # Raw label column from CSV
MAPPED_LABEL_COLUMN_NAME = 'PD-L1(Nivo)_binary' # Processed binary label
PATIENT_ID_COLUMN = 'PATIENT_ID'
FILE_NAME_COLUMN = 'FILE_NAME'
NUM_CLASSES = 2
CLASS_NAMES = ['PD-L1-', 'PD-L1+'] # Used for plotting and reports
BEST_MODEL_PATH = f"best_model_{TASK_NAME.lower()}.pth"

# %% [markdown]
# ## 自定义 Dataset

# %%
class PDL1Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        """
        df: 包含 FILE_NAME, PD-L1(Nivo)_binary, PATIENT_ID
        image_dir: 图像根路径
        """
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
# ## 读取标签并划分数据集

# %%
# 1. 读取 CSV
label_df = pd.read_csv("dataset/label.csv")
# 2. 丢弃空值
label_df = label_df.dropna(subset=[LABEL_COLUMN_NAME, FILE_NAME_COLUMN, PATIENT_ID_COLUMN]).copy()
# 3. 二分类标签
label_df[MAPPED_LABEL_COLUMN_NAME] = (label_df[LABEL_COLUMN_NAME] > 0).astype(int)

# 4. 先划分 Test 集（20%），按 PATIENT_ID 分组
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
trainval_idx, test_idx = next(gss_test.split(label_df, groups=label_df[PATIENT_ID_COLUMN]))
df_trainval = label_df.iloc[trainval_idx].copy()
df_test     = label_df.iloc[test_idx].copy()

# 5. 再从 trainval 中划分 Val 集，尝试按患者是否有少数类样本进行分层
patient_has_positive = df_trainval.groupby(PATIENT_ID_COLUMN)[MAPPED_LABEL_COLUMN_NAME].any()
unique_patients_trainval = df_trainval[PATIENT_ID_COLUMN].unique()
patient_strata_df = pd.DataFrame({
    PATIENT_ID_COLUMN: unique_patients_trainval,
    'has_positive_case': [patient_has_positive.get(pid, False) for pid in unique_patients_trainval]
})
val_size_from_trainval = 0.2

def check_val_set_diversity(val_df, min_classes=2):
    if val_df.empty or MAPPED_LABEL_COLUMN_NAME not in val_df.columns:
        print(f"Warning ({TASK_NAME}): Validation set is empty or missing label column. Cannot check class diversity.")
        return False
    num_unique_classes_val = val_df[MAPPED_LABEL_COLUMN_NAME].nunique()
    if num_unique_classes_val < min_classes:
        print(f"Warning ({TASK_NAME}): Validation set only has {num_unique_classes_val} unique class(es) after split. Expected at least {min_classes}. Distribution:\n{val_df[MAPPED_LABEL_COLUMN_NAME].value_counts()}")
        return False
    return True

try:
    train_patients_ids, val_patients_ids = train_test_split(
        patient_strata_df[PATIENT_ID_COLUMN],
        test_size=val_size_from_trainval,
        stratify=patient_strata_df['has_positive_case'],
        random_state=SEED
    )
    df_train = df_trainval[df_trainval[PATIENT_ID_COLUMN].isin(train_patients_ids)].copy()
    df_val   = df_trainval[df_trainval[PATIENT_ID_COLUMN].isin(val_patients_ids)].copy()
    check_val_set_diversity(df_val) # Check diversity after split
    print("Successfully used stratified split for train/validation sets based on patient minority class presence.")
except ValueError as e:
    print(f"Warning: Stratified split for train/validation failed: {e}. Falling back to GroupShuffleSplit.")
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_from_trainval, random_state=SEED + 1)
    train_idx_inner, val_idx_inner = next(gss_val.split(df_trainval, groups=df_trainval[PATIENT_ID_COLUMN]))
    df_train = df_trainval.iloc[train_idx_inner].copy()
    df_val   = df_trainval.iloc[val_idx_inner].copy()
    check_val_set_diversity(df_val) # Check diversity after fallback

# Final check on df_val before proceeding
if not check_val_set_diversity(df_val):
    print(f"Critical Warning ({TASK_NAME}): Validation set does not have diverse classes. Proceeding, but AUC/other metrics might be unreliable or NaN.")

print(f"\nDataset sizes and class distributions ({TASK_NAME}):")
for name, df_subset in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    if not df_subset.empty:
        print(f"  {name:<8}: {len(df_subset):>4} images, Patients: {df_subset[PATIENT_ID_COLUMN].nunique():>2}")
        print(f"    Class distribution ({MAPPED_LABEL_COLUMN_NAME}, normalized):\n{df_subset[MAPPED_LABEL_COLUMN_NAME].value_counts(normalize=True).sort_index()}")
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

# 重新引入 WeightedRandomSampler 用于平衡训练集
counts_train = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
# 计算每个样本的权重，使得采样时少数类被更频繁选中
# 权重是类频率的倒数
if len(counts_train) == NUM_CLASSES and all(c > 0 for c in counts_train.values):
    class_sample_weights = [1. / counts_train[i] for i in range(NUM_CLASSES)]
    sample_weights_train = [class_sample_weights[label] for label in df_train[MAPPED_LABEL_COLUMN_NAME]]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights_train), 
                                    num_samples=len(sample_weights_train), 
                                    replacement=True)
    print(f"Sampler weights for {TASK_NAME} classes: Class 0: {class_sample_weights[0]:.4f}, Class 1: {class_sample_weights[1]:.4f}")
    train_loader_args = {'sampler': sampler, 'batch_size':16, 'num_workers':0, 'pin_memory':True, 'shuffle': False}
else:
    print(f"Warning: Training data for {TASK_NAME} has insufficient or unexpected class counts for sampler ({counts_train.to_dict()}). Using standard DataLoader with shuffle.")
    sampler = None # Explicitly set sampler to None
    train_loader_args = {'shuffle': True, 'batch_size':16, 'num_workers':0, 'pin_memory':True}

# Dataset & DataLoader
train_ds = PDL1Dataset(df_train, IMG_DIR, transform=train_tf)
val_ds   = PDL1Dataset(df_val,   IMG_DIR, transform=val_tf)
test_ds  = PDL1Dataset(df_test,  IMG_DIR, transform=val_tf)

# 使用 sampler，不再 shuffle
if sampler:
    train_loader = DataLoader(train_ds, **train_loader_args)
else:
    train_loader = DataLoader(train_ds, **train_loader_args) # shuffle will be True if sampler is None
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,   num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False,   num_workers=0, pin_memory=True)

# %% [markdown]
# ## 模型定义与训练设置

# %%
# 迁移学习：EfficientNet-B0
print(f"Using EfficientNet-B0 for {TASK_NAME} ({NUM_CLASSES}-class classification)")
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

# 修改分类头
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES) 

model = model.to(device)

# Calculate class weights for CrossEntropyLoss
class_weights_tensor = None

if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
    counts = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    if len(counts) == NUM_CLASSES and all(c > 0 for c in counts.values):
        weights_values = [1.0 / counts.get(i, 1e-6) for i in range(NUM_CLASSES)]
        sum_weights = sum(weights_values)
        normalized_weights = [(w / sum_weights) * NUM_CLASSES for w in weights_values] if sum_weights > 1e-7 else [1.0] * NUM_CLASSES
        
        class_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float).to(device)
        weights_str = ", ".join([f"Class {i}: {w:.4f}" for i, w in enumerate(normalized_weights)])
        print(f"Train data counts for {TASK_NAME}: {counts.to_dict()}")
        print(f"Calculated class weights for CrossEntropyLoss: [{weights_str}]")
    elif len(counts) < NUM_CLASSES:
        print(f"Warning: Only {len(counts)} classes present in training data for {TASK_NAME} ({counts.to_dict()}). Expected {NUM_CLASSES}. Using default weights for CrossEntropyLoss.")
    else:
        print(f"Warning: Class counts issue in training data for {TASK_NAME} ({counts.to_dict()}). Using default weights for CrossEntropyLoss.")
else:
    print(f"Warning: df_train for {TASK_NAME} is empty or '{MAPPED_LABEL_COLUMN_NAME}' column is missing. Using default weights for CrossEntropyLoss.")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
print(f"Using nn.CrossEntropyLoss for {TASK_NAME}" + (f" with class_weights: {class_weights_tensor.tolist()}" if class_weights_tensor is not None else " with default weights."))

optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4) 
# 学习率调度（可选, 基于val_auc）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True, min_lr=1e-6)

# %% [markdown]
# ## 早停机制

# %%
class EarlyStopping:
    """早停机制，当验证集性能不再提升时停止训练"""
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True, mode='max', verbose=True, task_name="DefaultTask"):
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
# ## 训练与验证循环

# %%
from sklearn.metrics import roc_auc_score 

NUM_EPOCHS = 50 
best_val_auc = 0.0 

early_stopping = EarlyStopping(patience=10, mode='max', verbose=True, min_delta=0.001, task_name=TASK_NAME)

history = {
    'train_loss': [],
    'val_loss': [],
    'val_auc': [],
    'lr': []
}

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
    val_running_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_loss_iter = criterion(outputs, labels)
            val_running_loss += val_loss_iter.item() * imgs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_val_probs.extend(probs.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())
            
    val_epoch_loss = val_running_loss / len(val_ds)
    history['val_loss'].append(val_epoch_loss)
    
    if len(np.unique(all_val_labels)) > 1:
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
    else:
        val_auc = 0.0 
        print(f"Warning: Epoch {epoch}, Validation set for {TASK_NAME} only has one class, AUC set to 0.0")
    history['val_auc'].append(val_auc)

    current_lr = optimizer.param_groups[0]['lr']
    history['lr'].append(current_lr)
    scheduler.step(val_auc) 
    
    print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_epoch_loss:.4f}, Val AUC={val_auc:.4f}, LR={current_lr:.1e}")
    
    if val_auc > best_val_auc: 
        best_val_auc = val_auc
        if early_stopping.monitor_op(val_auc, early_stopping.best_score) or early_stopping.best_score == float('-inf'): 
             torch.save(model.state_dict(), BEST_MODEL_PATH)
             print(f"Epoch {epoch}: New best {TASK_NAME} model saved with Val AUC: {best_val_auc:.4f} to {BEST_MODEL_PATH}")

    if early_stopping(val_auc, model):
        print(f"Early stopping triggered for {TASK_NAME}.")
        break

# %% [markdown]
# ## 绘制训练过程曲线

# %%
def plot_training_history(history, task_name_for_plot):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(history['train_loss'], color=color, linestyle='-', label='训练损失')
    ax1.plot(history['val_loss'], color=color, linestyle=':', label='验证损失')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('AUC / LR', color=color)  
    ax2.plot(history['val_auc'], color=color, linestyle='-', label='验证AUC')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    ax3 = ax1.twinx() 
    ax3.spines["right"].set_position(("outward", 60)) 
    color = 'tab:green'
    ax3.set_ylabel('Learning Rate', color=color)
    ax3.plot(history['lr'], color=color, linestyle='--', label='学习率')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.legend(loc='lower right')
    ax3.set_yscale('log') 

    fig.tight_layout()  
    plt.title(f'{task_name_for_plot} 训练过程监控 (损失, AUC, 学习率)')
    plt.savefig(f"training_history_{task_name_for_plot.lower()}.png")
    plt.show()

plot_training_history(history, TASK_NAME)

# %% [markdown]
# ## 在验证集上进行特征-标签相关性分析

# %%
def get_embeddings_efficientnet(model, dataloader, device):
    model.eval()
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Extracting embeddings for {TASK_NAME} Val"):
            images = images.to(device)
            # EfficientNet's features are extracted before the classifier
            # For efficientnet_b0, model.features outputs the feature maps
            # We then use adaptive average pooling and flatten, similar to how it's done before the classifier
            features = model.features(images)
            pooled_features = model.avgpool(features)
            flattened_features = torch.flatten(pooled_features, 1)
            embeddings_list.append(flattened_features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    
    return np.concatenate(embeddings_list), np.concatenate(labels_list)

def calculate_mutual_information(features, labels):
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    mi = mutual_info_classif(features, labels, discrete_features=False, random_state=SEED)
    return mi

def plot_tsne_visualization(embeddings, labels, title_suffix, filename_suffix, class_names_plot=None):
    if embeddings.shape[0] <= 1:
        print(f"t-SNE ({title_suffix}): Not enough samples ({embeddings.shape[0]}) to perform t-SNE. Skipping.")
        return
    
    perplexity_value = min(30.0, max(5.0, embeddings.shape[0] - 1.0)) # Adjust perplexity based on sample size
    
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity_value, n_iter=1000, init='pca', learning_rate='auto')
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    if class_names_plot and len(class_names_plot) == len(np.unique(labels)):
        for i, class_name in enumerate(class_names_plot):
            plt.scatter(embeddings_tsne[labels == i, 0], embeddings_tsne[labels == i, 1], label=class_name, alpha=0.7)
    else: # Default to numeric labels if class_names_plot is not appropriate
        scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.legend(handles=scatter.legend_elements()[0], labels=np.unique(labels).astype(str))

    plt.title(f't-SNE Visualization of Embeddings ({title_suffix}) for {TASK_NAME}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    filename = f"tsne_visualization_{filename_suffix}_{TASK_NAME.lower()}.png"
    plt.savefig(filename)
    plt.show()
    plt.close()
    print(f"t-SNE plot saved to {filename}")

def simulate_data_cleaning_test(model, dataloader, device, p_flip=0.1):
    model.eval()
    original_labels_list = []
    flipped_labels_list = []
    original_probs_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Simulating data cleaning for {TASK_NAME} Val"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() # Prob of positive class

            original_labels_list.extend(labels.cpu().numpy())
            original_probs_list.extend(probs)

            # Simulate flipping labels
            flipped_labels_batch = labels.cpu().numpy().copy()
            for i in range(len(flipped_labels_batch)):
                if random.random() < p_flip:
                    flipped_labels_batch[i] = 1 - flipped_labels_batch[i] # Flip binary label
            flipped_labels_list.extend(flipped_labels_batch)
            
    original_labels_np = np.array(original_labels_list)
    flipped_labels_np = np.array(flipped_labels_list)
    original_probs_np = np.array(original_probs_list)

    if len(np.unique(original_labels_np)) > 1:
        auc_original = roc_auc_score(original_labels_np, original_probs_np)
    else:
        auc_original = float('nan')
        print(f"Warning (Data Cleaning - Original): Only one class present in original labels for {TASK_NAME}, AUC is NaN.")

    if len(np.unique(flipped_labels_np)) > 1:
        auc_flipped = roc_auc_score(flipped_labels_np, original_probs_np) # Use original_probs with flipped_labels
    else:
        auc_flipped = float('nan')
        print(f"Warning (Data Cleaning - Flipped): Only one class present in flipped labels for {TASK_NAME}, AUC is NaN.")

    print(f"\n--- {TASK_NAME} Data Cleaning Simulation (Validation Set) ---")
    print(f"  AUC with original labels: {auc_original:.4f}")
    print(f"  AUC with {p_flip*100:.1f}% randomly flipped labels: {auc_flipped:.4f}")
    
    mi_original_labels = calculate_mutual_information(original_probs_np.reshape(-1, 1), original_labels_np)
    mi_flipped_labels = calculate_mutual_information(original_probs_np.reshape(-1, 1), flipped_labels_np)
    print(f"  Mutual Information (Probs vs Original Labels): {mi_original_labels[0]:.4f}")
    print(f"  Mutual Information (Probs vs Flipped Labels): {mi_flipped_labels[0]:.4f}")
    print("--- End of Data Cleaning Simulation ---")

def perform_permutation_test(model, dataloader, device, n_permutations=1000):
    model.eval()
    all_original_labels = []
    all_original_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Getting original predictions for {TASK_NAME} Val Permutation Test"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_original_labels.extend(labels.cpu().numpy())
            all_original_probs.extend(probs)
    
    all_original_labels_np = np.array(all_original_labels)
    all_original_probs_np = np.array(all_original_probs)

    if len(np.unique(all_original_labels_np)) <= 1:
        print(f"Warning (Permutation Test): Original labels for {TASK_NAME} Val have only one class. Skipping permutation test.")
        return float('nan'), 1.0 # Observed AUC, p-value

    observed_auc = roc_auc_score(all_original_labels_np, all_original_probs_np)
    
    permuted_aucs = []
    for i in tqdm(range(n_permutations), desc=f"Performing permutations for {TASK_NAME} Val"):
        shuffled_labels = sklearn_shuffle(all_original_labels_np, random_state=SEED + i)
        if len(np.unique(shuffled_labels)) > 1:
             auc = roc_auc_score(shuffled_labels, all_original_probs_np)
             permuted_aucs.append(auc)
        # else: # If a shuffle results in one class, this permutation is not informative for AUC
             # pass # or permuted_aucs.append(0.5) or handle as appropriate

    permuted_aucs_np = np.array(permuted_aucs)
    p_value = np.mean(permuted_aucs_np >= observed_auc) if len(permuted_aucs_np) > 0 else 1.0

    print(f"\n--- {TASK_NAME} Permutation Test (Validation Set) ---")
    print(f"  Observed AUC on Val set: {observed_auc:.4f}")
    print(f"  Number of permutations: {n_permutations} (actual runs: {len(permuted_aucs_np)})")
    print(f"  Mean AUC from permuted labels: {np.mean(permuted_aucs_np):.4f}" if len(permuted_aucs_np) > 0 else "  Mean AUC from permuted labels: N/A")
    print(f"  Permutation p-value: {p_value:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(permuted_aucs_np, bins=50, alpha=0.7, label='Permuted AUCs', color='skyblue', edgecolor='black')
    plt.axvline(observed_auc, color='red', linestyle='--', lw=2, label=f'Observed AUC ({observed_auc:.4f})')
    plt.xlabel('AUC Score')
    plt.ylabel('Frequency')
    plt.title(f'{TASK_NAME} Permutation Test for AUC (Validation Set)\np-value: {p_value:.4f}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    filename = f"permutation_test_auc_{TASK_NAME.lower()}_val.png"
    plt.savefig(filename); plt.show(); plt.close()
    print(f"Permutation test plot saved to {filename}")
    print("--- End of Permutation Test ---")
    return observed_auc, p_value

# Perform analyses on Validation Set after training
if os.path.exists(BEST_MODEL_PATH) and len(val_ds) > 0:
    print(f"\n--- Starting Post-Training Analysis on Validation Set for {TASK_NAME} ---")
    model.load_state_dict(torch.load(BEST_MODEL_PATH)) # Load best model
    model.to(device) # Ensure model is on the correct device
    model.eval()

    # 1. Get Embeddings and True Labels from Validation Set
    val_embeddings, val_labels_for_analysis = get_embeddings_efficientnet(model, val_loader, device)
    
    if val_embeddings.shape[0] > 0 :
        # 2. Calculate Mutual Information between embeddings and labels
        # Average MI across all embedding dimensions for a summary
        # Or calculate for a subset or PCA-reduced version if embeddings are too high-dimensional
        # For simplicity, let's try with first few components or average, 
        # or use model's direct predictions (probabilities) vs labels for a more direct measure of label relevance.

        # Get model predictions (probabilities) for MI with labels
        all_val_probs_for_mi = []
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc=f"Getting predictions for {TASK_NAME} Val MI"):
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() # Prob of positive class
                all_val_probs_for_mi.extend(probs)
        all_val_probs_for_mi_np = np.array(all_val_probs_for_mi)

        if len(val_labels_for_analysis) > 0 and len(all_val_probs_for_mi_np) == len(val_labels_for_analysis):
             if len(np.unique(val_labels_for_analysis)) > 1:
                mi_preds_labels = calculate_mutual_information(all_val_probs_for_mi_np.reshape(-1,1), val_labels_for_analysis)
                print(f"\nMutual Information (Validation Predictions vs Labels) for {TASK_NAME}: {mi_preds_labels[0]:.4f}")
             else:
                print(f"Warning (MI): Only one class present in validation labels for {TASK_NAME}. Skipping MI calculation.")
        
        # 3. Plot t-SNE of Embeddings
        plot_tsne_visualization(val_embeddings, val_labels_for_analysis, 
                                title_suffix="Validation Set", 
                                filename_suffix="val", 
                                class_names_plot=CLASS_NAMES)

        # 4. Simulate Data Cleaning Impact
        simulate_data_cleaning_test(model, val_loader, device, p_flip=0.1)

        # 5. Perform Permutation Test
        perform_permutation_test(model, val_loader, device, n_permutations=1000) # Reduced for speed, can increase
    else:
        print(f"Validation set for {TASK_NAME} is empty or embeddings could not be extracted. Skipping post-training analysis on val set.")
    print(f"--- End of Post-Training Analysis on Validation Set for {TASK_NAME} ---")

else:
    print(f"Skipping Post-Training Analysis on Validation Set for {TASK_NAME}: Best model not found or val_ds is empty.")

# %% [markdown]
# ## Grad-CAM 可视化 (验证集)

# %%
from pytorch_grad_cam import GradCAM # HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad (Removed unused imports for now)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image # deprocess_image, preprocess_image (Removed unused)
from torchvision.utils import make_grid, save_image

def visualize_grad_cam_on_val(model_to_use, target_layer_name, val_dataset, device_to_use, num_images=4, class_names_for_cam=None):
    """
    在验证集上使用 pytorch-grad-cam 库可视化 Grad-CAM。
    model_to_use: 训练好的模型
    target_layer_name: EfficientNet中最后一个卷积层的名称, e.g., 'features[-1][0]'
    val_dataset: 用于选择图像的验证集 Dataset 对象 (val_ds)
    device_to_use: cuda or cpu
    num_images: 要为每个目标类别可视化的图像数量
    class_names_for_cam: 类别名称列表，用于文件名和日志
    """
    if not class_names_for_cam:
        class_names_for_cam = [f"Class_{i}" for i in range(NUM_CLASSES)]

    try:
        # Simplified target layer selection, direct access if 'features[-1][0]' like
        if target_layer_name == 'features[-1][0]':
            target_layers = [model_to_use.features[-1][0]]
        elif target_layer_name == 'features[-1]': # common shorthand
             target_layers = [model_to_use.features[-1][0]]
        else: # Attempt to resolve more complex paths if necessary
            module_path = target_layer_name.split('.')
            current_module = model_to_use
            for m_name in module_path:
                if '[' in m_name and ']' in m_name: # e.g. features[-1][0]
                    base_name = m_name.split('[')[0]
                    index_str = m_name.split('[')[1].replace(']', '')
                    current_module = getattr(current_module, base_name)
                    current_module = current_module[int(index_str)]
                else:
                    current_module = getattr(current_module, m_name)
            target_layers = [current_module]
        print(f"Grad-CAM target layer resolved to: {target_layers}")
    except Exception as e:
        print(f"Error finding Grad-CAM target layer '{target_layer_name}': {e}. Defaulting to model.features[-1][0].")
        try:
            target_layers = [model_to_use.features[-1][0]]
        except Exception as e_default:
            print(f"Error getting default Grad-CAM layer model.features[-1][0]: {e_default}. Skipping Grad-CAM.")
            return

    cam_algorithm = GradCAM 
    cam = cam_algorithm(model=model_to_use, target_layers=target_layers)

    if len(val_dataset) == 0:
        print(f"Grad-CAM ({TASK_NAME}): Validation dataset is empty. Skipping visualization.")
        return
    
    actual_num_images = min(num_images, len(val_dataset))
    if actual_num_images < num_images:
        print(f"Grad-CAM ({TASK_NAME}): Requested {num_images} images, but validation set only has {len(val_dataset)}. Using {actual_num_images}.")

    # Ensure we don't pick more images than available per class if we were to filter
    # For now, pick random images from the val_dataset
    indices = np.random.choice(len(val_dataset), actual_num_images, replace=actual_num_images > len(val_dataset))


    for target_class_idx in range(NUM_CLASSES):
        target_class_name = class_names_for_cam[target_class_idx]
        
        rgb_imgs_list = []
        cam_outputs_list = []
        processed_indices_count = 0

        # We will iterate through the chosen random indices
        # If we wanted to ensure images of a specific true label, we'd need to iterate dataset or filter indices
        
        temp_indices = list(indices) # Use a copy to potentially pop from or iterate safely
        
        # Try to get a diverse set of images, or at least the requested number.
        # This part can be enhanced to pick images OF the target_class_idx if desired.
        # For now, it takes the random sample and generates CAM for that sample for EACH target_class_idx.

        for i, original_idx in enumerate(temp_indices):
            if processed_indices_count >= actual_num_images:
                break # Should not happen if actual_num_images is len(indices)

            img_tensor, true_label = val_dataset[original_idx] 
            
            # Denormalize for visualization
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            rgb_img_denorm = inv_normalize(img_tensor.clone()).permute(1, 2, 0).cpu().numpy()
            rgb_img_denorm = np.clip(rgb_img_denorm, 0, 1) 

            input_tensor_unsqueeze = img_tensor.unsqueeze(0).to(device_to_use)
            
            # Define CAM targets for the current target_class_idx
            cam_targets = [ClassifierOutputTarget(target_class_idx)]
            
            grayscale_cam = cam(input_tensor=input_tensor_unsqueeze, targets=cam_targets)
            # Assuming batch size is 1 for grayscale_cam output here
            grayscale_cam_batch = grayscale_cam[0, :] 
            
            cam_image = show_cam_on_image(rgb_img_denorm, grayscale_cam_batch, use_rgb=True)
            cam_image_tensor = transforms.ToTensor()(cam_image) 

            # Original image for the grid (after denormalization)
            original_img_for_grid = transforms.ToTensor()(Image.fromarray((rgb_img_denorm * 255).astype(np.uint8)))
            
            rgb_imgs_list.append(original_img_for_grid)
            cam_outputs_list.append(cam_image_tensor)
            processed_indices_count += 1

        if not rgb_imgs_list: # If no images were processed for this class
            print(f"Grad-CAM ({TASK_NAME}): No images processed for target class {target_class_name}. Skipping grid saving for this class.")
            continue

        grid_originals = make_grid(rgb_imgs_list, nrow=actual_num_images, normalize=False, pad_value=0.5)
        grid_cams = make_grid(cam_outputs_list, nrow=actual_num_images, normalize=False, pad_value=0.5)
        
        # Stack original and CAM grids vertically
        combined_grid = torch.cat((grid_originals, grid_cams), dim=1) # dim=1 for vertical stack in make_grid output context (height)
        
        filename = f"grad_cam_val_grid_target_{target_class_name.replace(" ", "_")}_{TASK_NAME.lower()}.png"
        save_image(combined_grid, filename)
        print(f"Grad-CAM grid for {TASK_NAME} (Validation Set, Target: {target_class_name}) saved to {filename}")

# Perform Grad-CAM on Validation Set after training
if os.path.exists(BEST_MODEL_PATH) and len(val_ds) > 0:
    print(f"\n--- Starting Grad-CAM Visualization on Validation Set for {TASK_NAME} ---")
    # Load best model if not already loaded (e.g. if running this cell independently)
    # Ensure model is loaded (it should be from analysis section, but good to double check)
    if 'model' not in locals() or not any(p.numel() for p in model.parameters()): # Check if model var exists and has params
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        model.to(device)
        print(f"Grad-CAM: Loaded model from {BEST_MODEL_PATH} for Grad-CAM.")
    model.eval()

    # Specify the target layer for EfficientNet-B0
    # Common last conv block for EfficientNet-B0 is model.features[-1][0]
    grad_cam_target_layer = 'features[-1][0]' 
    
    visualize_grad_cam_on_val(model, 
                              target_layer_name=grad_cam_target_layer, 
                              val_dataset=val_ds, 
                              device_to_use=device, 
                              num_images=4, # Number of images per class target
                              class_names_for_cam=CLASS_NAMES)
    print(f"--- End of Grad-CAM Visualization for {TASK_NAME} ---")
else:
    print(f"Skipping Grad-CAM on Validation Set for {TASK_NAME}: Best model not found or val_ds is empty.")

# %% [markdown]
# ## 在验证集上评估并绘制ROC/PR曲线

# %%
def evaluate_and_plot_curves_on_val(model_to_eval, val_dataloader, device_to_use, task_name_str, class_names_list):
    model_to_eval.eval()
    all_val_labels_eval = []
    all_val_probs_eval = []

    if len(val_dataloader.dataset) == 0:
        print(f"Skipping ROC/PR curve plotting on Val set for {task_name_str}: Validation dataset is empty.")
        return

    with torch.no_grad():
        for images, labels in tqdm(val_dataloader, desc=f"Evaluating {task_name_str} on Val set for ROC/PR"):
            images = images.to(device_to_use)
            outputs = model_to_eval(images)
            # Assuming binary classification, get probabilities for the positive class (class 1)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_val_labels_eval.extend(labels.cpu().numpy())
            all_val_probs_eval.extend(probs)
    
    all_val_labels_np = np.array(all_val_labels_eval)
    all_val_probs_np = np.array(all_val_probs_eval)

    if len(np.unique(all_val_labels_np)) <= 1:
        print(f"Warning ({task_name_str} Val ROC/PR): Only one class present in validation labels. ROC/PR curves may be uninformative or cause errors.")
        # Attempt to plot anyway, but with a note or specific handling if roc_auc_score fails
    
    # ROC Curve
    try:
        fpr, tpr, _ = roc_curve(all_val_labels_np, all_val_probs_np)
        auc_score = roc_auc_score(all_val_labels_np, all_val_probs_np)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_score:.4f})')
        plt.plot([0,1],[0,1],color='navy',lw=2,ls='--'); plt.xlim([0,1]); plt.ylim([0,1.05])
        plt.xlabel('假阳性率'); plt.ylabel('真阳性率'); plt.title(f'验证集ROC曲线 {task_name_str}')
        plt.legend(loc="lower right"); plt.grid(True)
        roc_filename = f'roc_curve_{task_name_str.lower()}_val.png'
        plt.savefig(roc_filename); plt.show(); plt.close()
        print(f"Validation ROC curve for {task_name_str} saved to {roc_filename}")
    except ValueError as e_roc:
        print(f"Error plotting ROC curve for {task_name_str} on Val set (likely due to single class): {e_roc}")

    # PR Curve
    try:
        precision, recall, _ = precision_recall_curve(all_val_labels_np, all_val_probs_np)
        ap_score = average_precision_score(all_val_labels_np, all_val_probs_np)
        plt.figure(figsize=(8,6)); 
        plt.plot(recall, precision, color='blue',lw=2,label=f'P-R (AP = {ap_score:.4f})')
        no_skill = np.sum(all_val_labels_np == 1) / len(all_val_labels_np) if len(all_val_labels_np) > 0 and np.sum(all_val_labels_np == 1) > 0 else 0.0
        plt.plot([0,1],[no_skill,no_skill],color='gray',lw=2,ls='--', label=f'No Skill (AP={no_skill:.4f})')
        plt.xlabel('召回率'); plt.ylabel('精确率'); plt.title(f'验证集P-R曲线 ({task_name_str} - {class_names_list[1] if len(class_names_list)>1 else 'Positive'})')
        plt.legend(loc="best"); plt.grid(True); plt.ylim([0,1.05]); plt.xlim([0,1])
        pr_filename = f'pr_curve_{task_name_str.lower()}_val.png'
        plt.savefig(pr_filename); plt.show(); plt.close()
        print(f"Validation P-R curve for {task_name_str} saved to {pr_filename}")
    except ValueError as e_pr:
        print(f"Error plotting P-R curve for {task_name_str} on Val set (likely due to single class): {e_pr}")

# Call validation curve plotting after training and other analyses
if os.path.exists(BEST_MODEL_PATH) and len(val_ds) > 0:
    print(f"\n--- Evaluating and Plotting ROC/PR Curves on Validation Set for {TASK_NAME} ---")
    if 'model' not in locals() or not any(p.numel() for p in model.parameters()): # Check if model var exists and has params
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        model.to(device)
        print(f"ROC/PR Val: Loaded model from {BEST_MODEL_PATH}.")
    model.eval() # Ensure model is in eval mode
    evaluate_and_plot_curves_on_val(model, val_loader, device, TASK_NAME, CLASS_NAMES)
    print(f"--- Finished ROC/PR Curve Plotting on Validation Set for {TASK_NAME} ---")
else:
    print(f"Skipping ROC/PR curves on Validation Set for {TASK_NAME}: Best model not found or val_ds is empty.")

# %% [markdown]
# ## 测试集评估

# %%
from sklearn.metrics import precision_recall_fscore_support

# 加载最佳模型
if os.path.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    print(f"Loaded best model for {TASK_NAME} from {BEST_MODEL_PATH}")
else:
    print(f"Warning: Best model path {BEST_MODEL_PATH} not found. Using current model state for evaluation.")
    
model.eval()

all_probs_test = [] 
all_labels_test = []
if len(test_ds) > 0:
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Testing {TASK_NAME}"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1] 
            all_probs_test.extend(probs.cpu().numpy())
            all_labels_test.extend(labels.numpy())
else:
    print(f"Test dataset for {TASK_NAME} is empty. Skipping test set evaluation.")

if len(all_labels_test) > 0 and len(all_probs_test) > 0:
    all_labels_test = np.array(all_labels_test)
    all_probs_test = np.array(all_probs_test)
    
    # 1. 使用默认阈值 0.5进行评估
    print(f"\n=== Test Classification Report {TASK_NAME} (Threshold = 0.5) ===")
    preds_at_0_5 = (all_probs_test >= 0.5).astype(int)
    report_0_5 = classification_report(all_labels_test, preds_at_0_5, target_names=CLASS_NAMES, zero_division=0)
    cm_0_5 = confusion_matrix(all_labels_test, preds_at_0_5)
    print(report_0_5)
    print("Confusion Matrix (Threshold = 0.5):")
    print(cm_0_5)
    
    def plot_confusion_matrix_local(cm, class_names_plot, title='混淆矩阵', filename='confusion_matrix.png'): # Renamed to avoid conflict
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_plot, yticklabels=class_names_plot)
        plt.title(title)
        plt.ylabel('实际标签')
        plt.xlabel('预测标签')
        plt.savefig(filename)
        plt.show()
        plt.close() # Close the figure after saving and showing
    
    plot_confusion_matrix_local(cm_0_5, CLASS_NAMES, title=f'{TASK_NAME} 混淆矩阵 (阈值 0.5)', filename=f'confusion_matrix_{TASK_NAME.lower()}_0.5.png')
    
    # 2. 寻找更优的阈值 
    print(f"\n=== Finding Optimal Threshold for {TASK_NAME} ({CLASS_NAMES[1]}) (based on F1-score) ===")
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.5
    best_f1_positive = 0.0
    best_recall_positive = 0.0
    best_precision_positive = 0.0

    for thresh in thresholds:
        current_preds = (all_probs_test >= thresh).astype(int)
        p, r, f1, s = precision_recall_fscore_support(all_labels_test, current_preds, labels=[0, 1], zero_division=0)
        f1_positive = f1[1] 
        if f1_positive > best_f1_positive:
            best_f1_positive = f1_positive
            best_threshold = thresh
            best_recall_positive = r[1]
            best_precision_positive = p[1]
        elif f1_positive == best_f1_positive: 
            if r[1] > best_recall_positive : 
                 best_threshold = thresh
                 best_recall_positive = r[1]
                 best_precision_positive = p[1]

    print(f"Optimal threshold found for {TASK_NAME} {CLASS_NAMES[1]}: {best_threshold:.2f}")
    print(f"  {CLASS_NAMES[1]} F1-score: {best_f1_positive:.4f}")
    print(f"  {CLASS_NAMES[1]} Precision: {best_precision_positive:.4f}")
    print(f"  {CLASS_NAMES[1]} Recall: {best_recall_positive:.4f}")
    
    print(f"\n=== Test Classification Report {TASK_NAME} (Optimal Threshold {best_threshold:.2f}) ===")
    preds_optimal = (all_probs_test >= best_threshold).astype(int)
    report_optimal = classification_report(all_labels_test, preds_optimal, target_names=CLASS_NAMES, zero_division=0)
    cm_optimal = confusion_matrix(all_labels_test, preds_optimal)
    print(report_optimal)
    print(f"Confusion Matrix (Threshold = {best_threshold:.2f}):")
    print(cm_optimal)
    
    plot_confusion_matrix_local(cm_optimal, CLASS_NAMES, title=f'{TASK_NAME} 混淆矩阵 (最佳阈值 {best_threshold:.2f})', filename=f'confusion_matrix_{TASK_NAME.lower()}_optimal.png')
    
    def plot_roc_curve_local(labels, probs, title='ROC曲线', filename='roc_curve.png'): # Renamed
        fpr, tpr, roc_thresholds = roc_curve(labels, probs)
        auc_score = roc_auc_score(labels, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (False Positive Rate)')
        plt.ylabel('真阳性率 (True Positive Rate)')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(filename)
        plt.show()
        plt.close()
    
    plot_roc_curve_local(all_labels_test, all_probs_test, title=f'测试集ROC曲线 {TASK_NAME}', filename=f'roc_curve_{TASK_NAME.lower()}_test.png')
    
    def plot_pr_curve_local(labels, probs, title='P-R曲线', filename='pr_curve.png', positive_class_name_for_plot='Positive'): #Renamed and added positive_class_name
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap_score = average_precision_score(labels, probs)
        no_skill = len(labels[labels==1]) / len(labels) if len(labels) > 0 and np.sum(labels) > 0 else 0.0 # handle case with no positive samples
        plt.plot(recall, precision, color='blue', lw=2, label=f'P-R curve (AP = {ap_score:.4f})')
        plt.plot([0, 1], [no_skill, no_skill], color='gray', lw=2, linestyle='--', label=f'No Skill (AP={no_skill:.4f})')
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title(title)
        plt.legend(loc="best") 
        plt.grid(True)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.savefig(filename)
        plt.show()
        plt.close()
    
    plot_pr_curve_local(all_labels_test, all_probs_test, title=f'测试集P-R曲线 ({TASK_NAME} - {CLASS_NAMES[1]})', filename=f'pr_curve_{TASK_NAME.lower()}_test.png', positive_class_name_for_plot=CLASS_NAMES[1])
else:
    print(f"Grad-CAM: Test dataset for {TASK_NAME} is empty. Skipping Grad-CAM visualization.")

# %%