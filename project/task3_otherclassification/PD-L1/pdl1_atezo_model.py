# %% [markdown]
# # PD-L1(Atezo) 膀胱镜图像二分类模型
# 通过迁移学习（EfficientNet-B0）对 PD-L1(Atezo) 进行二分类，按患者划分训练/验证/测试集，尽量避免同一患者图像落到同一集合。

# %%
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
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
from sklearn.metrics import average_precision_score # 已有 roc_curve, roc_auc_score, precision_recall_curve

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

# 定义列名常量
LABEL_COLUMN_NAME = 'PD-L1(Atezo)'        # 原始标签列名
MAPPED_LABEL_COLUMN_NAME = 'PD-L1(Atezo)_binary' # 处理后的二分类标签列名
PATIENT_ID_COLUMN = 'PATIENT_ID'         # 患者ID列名
FILE_NAME_COLUMN = 'FILE_NAME'           # 文件名列名
NUM_CLASSES = 2                          # 类别数量 (二分类)

# %% [markdown]
# ## Focal Loss 定义
# %%
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean', num_classes=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

        if self.num_classes == 2:
            if isinstance(alpha, list) and len(alpha)==2:
                 self.alpha = torch.tensor(alpha)
            elif isinstance(alpha, (float, int)):
                 self.alpha = torch.tensor([alpha, alpha])
        
        elif isinstance(alpha, (float, int)):
             self.alpha = torch.tensor([alpha] * num_classes)
        
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
            if len(self.alpha) != num_classes:
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
# ## 自定义 Dataset

# %%
class PDL1Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        """
        df: 包含 FILE_NAME, PD-L1(Atezo)_binary, PATIENT_ID
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
label_df = pd.read_csv("dataset/label.csv") # 您可能需要为Atezo任务修改这个CSV文件名或内容
# 2. 丢弃空值
label_df = label_df.dropna(subset=[LABEL_COLUMN_NAME, FILE_NAME_COLUMN, PATIENT_ID_COLUMN]).copy() # MODIFIED HERE, using constants
# 3. 二分类标签
label_df[MAPPED_LABEL_COLUMN_NAME] = (label_df[LABEL_COLUMN_NAME] > 0).astype(int) # MODIFIED HERE, using constants

# 4. 不再划分独立的 Test 集，所有数据用于 Train/Val
df_trainval = label_df.copy()
# df_test     = label_df.iloc[test_idx].copy() # 移除测试集划分

# 5. 再从 trainval 中划分 Val 集，尝试按患者是否有少数类样本进行分层
# 首先，确定哪些患者在 df_trainval 中有类别为 1 的样本
patient_has_positive = df_trainval.groupby(PATIENT_ID_COLUMN)[MAPPED_LABEL_COLUMN_NAME].any() # 使用常量

# 获取 df_trainval 中的所有唯一患者 ID
unique_patients_trainval = df_trainval[PATIENT_ID_COLUMN].unique() # 使用常量

# 创建一个包含患者 ID 和他们是否有正样本的 DataFrame
patient_strata_df = pd.DataFrame({
    PATIENT_ID_COLUMN: unique_patients_trainval, # 使用常量
    'has_positive_case': [patient_has_positive.get(pid, False) for pid in unique_patients_trainval]
})

val_size_from_trainval = 0.2 # 验证集占 trainval 的比例 (与 grade_model 一致)

try:
    # 尝试对患者ID进行分层抽样
    train_patients_ids, val_patients_ids = train_test_split(
        patient_strata_df[PATIENT_ID_COLUMN], # 使用常量
        test_size=val_size_from_trainval,
        stratify=patient_strata_df['has_positive_case'],
        random_state=SEED
    )
    df_train = df_trainval[df_trainval[PATIENT_ID_COLUMN].isin(train_patients_ids)].copy() # 使用常量
    df_val   = df_trainval[df_trainval[PATIENT_ID_COLUMN].isin(val_patients_ids)].copy() # 使用常量
    print("Successfully used stratified split for train/validation sets based on patient minority class presence.")
except ValueError as e:
    # 如果分层失败（例如，某个分层中的样本太少），则回退到 GroupShuffleSplit
    print(f"Warning: Stratified split for train/validation failed: {e}. Falling back to GroupShuffleSplit.")
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_from_trainval, random_state=SEED + 1)
    train_idx_inner, val_idx_inner = next(gss_val.split(df_trainval, groups=df_trainval[PATIENT_ID_COLUMN])) # 使用常量
    df_train = df_trainval.iloc[train_idx_inner].copy()
    df_val   = df_trainval.iloc[val_idx_inner].copy()

# 移除 df_test 的打印
print("Dataset sizes and class distributions:")
for name, df_subset in [("Train", df_train), ("Val", df_val)]: # 移除 Test
    if not df_subset.empty:
        print(f"  {name:<8}: {len(df_subset):>4} images, Patients: {df_subset[PATIENT_ID_COLUMN].nunique():>2}") # 使用常量
        print(f"    Class distribution ({MAPPED_LABEL_COLUMN_NAME}, normalized):{df_subset[MAPPED_LABEL_COLUMN_NAME].value_counts(normalize=True).sort_index()}") # 使用常量
        # 添加每个类别中独立患者数量的打印
        print(f"    Unique patients per class ({MAPPED_LABEL_COLUMN_NAME}):")
        if MAPPED_LABEL_COLUMN_NAME in df_subset.columns and PATIENT_ID_COLUMN in df_subset.columns:
            for class_label in sorted(df_subset[MAPPED_LABEL_COLUMN_NAME].unique()):
                num_patients_in_class = df_subset[df_subset[MAPPED_LABEL_COLUMN_NAME] == class_label][PATIENT_ID_COLUMN].nunique()
                print(f"      Class {class_label}: {num_patients_in_class} patients")
        else:
            print("      Could not calculate unique patients per class (column missing).")
    else:
        print(f"  {name:<8}: Empty")
print("")


# 检查验证集中的类别分布 (这个可以移除或注释掉了，因为上面的循环会打印)
# print("\nValidation set class distribution (PD-L1(Atezo)_binary):")
# if not df_val.empty:
#     print(df_val['PD-L1(Atezo)_binary'].value_counts())
# else:
#     print("Validation set is empty!")
# print("\n")

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
# 使用常量 MAPPED_LABEL_COLUMN_NAME
train_loader_args = {'shuffle': True}
if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
    counts_train = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    if len(counts_train) >= 1 and len(counts_train) <= NUM_CLASSES:
        class_sample_weights = [1. / counts_train.get(i, 1e-6) for i in range(NUM_CLASSES)]
        sample_weights_train = [class_sample_weights[label] for label in df_train[MAPPED_LABEL_COLUMN_NAME]] # 使用常量

sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights_train), 
                                num_samples=len(sample_weights_train), 
                                replacement=True)
        weights_str = ", ".join([f"Class {i}: {w:.4f}" for i, w in enumerate(class_sample_weights)])
        print(f"Sampler weights for {LABEL_COLUMN_NAME} classes: {weights_str}")
        train_loader_args = {'sampler': sampler, 'shuffle': False} # 如果使用sampler，shuffle应为False
    else:
        print(f"Warning: Training data for {LABEL_COLUMN_NAME} has insufficient or unexpected class counts for sampler. Using standard DataLoader with shuffle=True.")
else:
    print(f"Warning: df_train for {LABEL_COLUMN_NAME} is empty or mapped label column is missing. Using standard DataLoader.")
    if df_train.empty:
        train_loader_args['shuffle'] = False # 防止空数据集 DataLoader 出错
        print("   df_train is empty. Forcing shuffle=False for train_loader to prevent error.")


# Dataset & DataLoader
train_ds = PDL1Dataset(df_train, IMG_DIR, transform=train_tf)
val_ds   = PDL1Dataset(df_val,   IMG_DIR, transform=val_tf)
# test_ds  = PDL1Dataset(df_test,  IMG_DIR, transform=val_tf) # 移除 test_ds

# 使用 train_loader_args
train_loader = DataLoader(train_ds, batch_size=16, num_workers=0, pin_memory=True, **train_loader_args)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,   num_workers=0, pin_memory=True)
# test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False,   num_workers=0, pin_memory=True) # 移除 test_loader

# %% [markdown]
# ## 模型定义与训练设置

# %%
# 迁移学习：EfficientNet-B0
print(f"Using EfficientNet-B0 for {LABEL_COLUMN_NAME} {NUM_CLASSES}-class classification") # 使用常量
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

# 修改分类头
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES) # 使用常量 NUM_CLASSES

model = model.to(device)

# 计算类别权重 for FocalLoss alpha parameter
# 与 grade_model.py 类似的方式计算 alpha
focal_loss_alpha_values = [0.5, 0.5] # 默认值
if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
    counts = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index() # 使用常量
    if len(counts) == NUM_CLASSES:
        class_weights_counts = [counts.get(i, 0) for i in range(NUM_CLASSES)] # 原始数量
        # 根据类别不平衡程度调整alpha，例如给少数类更高的alpha权重
        # 这里我们简单地使用 0.25 / 0.75 的策略，假设类别1是少数类且更重要
        # 或者可以根据实际比例的倒数来计算，然后归一化
        # total_samples = sum(class_weights_counts)
        # if total_samples > 0 and class_weights_counts[0] > 0 and class_weights_counts[1] > 0:
        #    alpha_class_0 = class_weights_counts[1] / total_samples # alpha for class 0 is prop of class 1
        #    alpha_class_1 = class_weights_counts[0] / total_samples # alpha for class 1 is prop of class 0
        #    focal_loss_alpha_values = [alpha_class_0, alpha_class_1]
        # 为了与原 Atezo 脚本的意图保持接近，同时借鉴 grade_model 的动态计算
        # 如果类别0样本多，给类别1（Positive）更高的alpha
        if counts.get(0,0) > counts.get(1,0): # 多数类是0
            focal_loss_alpha_values = [0.25, 0.75] # 给少数类（1）更高的alpha
        else: # 多数类是1，或者数量相等
            focal_loss_alpha_values = [0.75, 0.25] # 给少数类（0）更高的alpha

        counts_str = ", ".join([f"Class {i}: {counts.get(i,0)}" for i in range(NUM_CLASSES)])
        print(f"Train data counts for {LABEL_COLUMN_NAME}: {counts_str}")
        alpha_str = ", ".join([f"{w:.4f}" for w in focal_loss_alpha_values])
        print(f"Calculated FocalLoss alpha for {LABEL_COLUMN_NAME}: [{alpha_str}]")
    elif len(counts) == 1:
        present_class = counts.index[0]
        focal_loss_alpha_values = [0.0] * NUM_CLASSES
        focal_loss_alpha_values[present_class] = 1.0
        print(f"Warning: Only class {present_class} present in {LABEL_COLUMN_NAME} training data. FocalLoss alpha set to: {focal_loss_alpha_values}")
    else:
        print(f"Warning: Not all {NUM_CLASSES} classes present or unexpected counts in {LABEL_COLUMN_NAME} training data ({len(counts)} found). Using default FocalLoss alpha: {focal_loss_alpha_values}")
else:
    print(f"Warning: Could not calculate class counts for {LABEL_COLUMN_NAME} FocalLoss alpha. Using default: {focal_loss_alpha_values}")


focal_loss_alpha = torch.tensor(focal_loss_alpha_values, dtype=torch.float).to(device)
print(f"Using Focal Loss with alpha: {focal_loss_alpha.tolist()} and gamma=2 for {LABEL_COLUMN_NAME}") # 使用常量


# 损失与优化器
criterion = FocalLoss(alpha=focal_loss_alpha, gamma=2, num_classes=NUM_CLASSES) # 传递 num_classes
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4) 

# 学习率调度 (基于val_loss，与grade_model一致)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5, verbose=True, min_lr=1e-6) # patience from grade_model

# %% [markdown]
# ## 早停机制

# %%
class EarlyStopping:
    """早停机制，当验证集性能不再提升时停止训练"""
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True, mode='max', verbose=True, task_name="Task"):
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
# best_val_auc = 0.0 # 不再基于 AUC 保存最佳模型
best_val_loss = float('inf') # 基于 val_loss 保存

# EarlyStopping mode改为'min' (监控loss), 并传入 task_name
early_stopping = EarlyStopping(patience=15, mode='min', verbose=True, min_delta=0.0001, task_name=LABEL_COLUMN_NAME) # patience from grade_model

history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [], # 新增: 记录验证准确率
    'val_auc': [],
    'lr': []
}

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train {LABEL_COLUMN_NAME}]"): # 使用 LABEL_COLUMN_NAME
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
    all_val_probs_class1 = [] # 存储类别1的概率，用于AUC
    all_val_preds = [] # 存储预测类别，用于准确率
    val_running_loss = 0.0

    if len(val_ds) > 0: # 确保验证集不为空
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_loss_iter = criterion(outputs, labels)
            val_running_loss += val_loss_iter.item() * imgs.size(0)
                
                probs = torch.softmax(outputs, dim=1) # 获取每个类别的概率
                preds = torch.argmax(probs, dim=1)    # 获取预测的类别
                
                all_val_probs_class1.extend(probs[:, 1].cpu().numpy()) # 假设类别1是阳性类
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
        if len(np.unique(all_val_labels_np)) >= NUM_CLASSES: # 确保有两个类别才计算AUC
            try:
                val_auc = roc_auc_score(all_val_labels_np, all_val_probs_class1_np)
            except ValueError as e_auc:
                 print(f"Warning: Epoch {epoch}, AUC calculation error for {LABEL_COLUMN_NAME}: {e_auc}. AUC set to 0.0")
        elif len(all_val_labels_np) > 0: # 如果只有一个类别，AUC无意义
            print(f"Warning: Epoch {epoch}, {LABEL_COLUMN_NAME} val set only has one class ({np.unique(all_val_labels_np)}), AUC set to 0.0")
    history['val_auc'].append(val_auc)

    current_lr = optimizer.param_groups[0]['lr']
    history['lr'].append(current_lr)
        scheduler.step(val_epoch_loss) # scheduler监控val_loss
        
        print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_epoch_loss:.4f}, Val Acc={val_accuracy:.4f}, Val AUC={val_auc:.4f}, LR={current_lr:.1e}")
        
        # 基于 val_loss 保存最佳模型
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            # if early_stopping.monitor_op(val_epoch_loss, early_stopping.best_score) or early_stopping.best_score == float('inf'): 
            torch.save(model.state_dict(), f"best_model_{LABEL_COLUMN_NAME.lower()}.pth") # 使用 f-string 和 LABEL_COLUMN_NAME
            print(f"Epoch {epoch}: New best {LABEL_COLUMN_NAME} model saved with Val Loss: {best_val_loss:.4f}") # 使用 LABEL_COLUMN_NAME

        if early_stopping(val_epoch_loss, model): # early_stopping 监控 val_loss
            print(f"Early stopping triggered for {LABEL_COLUMN_NAME}.") # 使用 LABEL_COLUMN_NAME
        break
    else: # 处理空验证集的情况
        history['val_loss'].append(float('nan'))
        history['val_accuracy'].append(float('nan'))
        history['val_auc'].append(float('nan'))
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Val Loss=N/A (empty val set for {LABEL_COLUMN_NAME}), LR={current_lr:.1e}")


# %% [markdown]
# ## 绘制训练过程曲线

# %%
def plot_training_history_pdl1(history): # 重命名并修改以匹配 grade_model.py 中的功能
    epochs_ran = len(history['train_loss'])
    epoch_ticks = range(1, epochs_ran + 1)

    fig, ax1 = plt.subplots(figsize=(14, 7)) # 调整图像大小

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
    plt.title(f'{LABEL_COLUMN_NAME} 二分类训练过程监控') # 使用 LABEL_COLUMN_NAME
    plt.xticks(epoch_ticks)
    plt.savefig(f"training_history_{LABEL_COLUMN_NAME.lower()}.png") # 使用 LABEL_COLUMN_NAME
    plt.show()

if any(history.values()): # 检查 history 是否有数据
    plot_training_history_pdl1(history) # 调用新的函数名
else:
    print(f"No training history to plot for {LABEL_COLUMN_NAME}.")


# %% [markdown]
# ## 测试集评估 -> 修改为验证集最终评估

# %% [markdown]
# 原有的测试集评估部分将被移除或修改为在验证集上进行最终评估。
# %% 
# from sklearn.metrics import precision_recall_fscore_support # 这个导入在后面也会用到

# # 加载最佳模型
# model.load_state_dict(torch.load(f"best_model_{LABEL_COLUMN_NAME.lower()}.pth")) # 使用常量
# model.eval()

# all_probs_val = [] # 修改变量名以反映是验证集
# all_labels_val = [] # 修改变量名
# with torch.no_grad():
#     for imgs, labels in tqdm(val_loader, desc=f"Evaluating {LABEL_COLUMN_NAME} on Validation Set"): # 使用验证集加载器
#         imgs = imgs.to(device)
#         outputs = model(imgs)
#         probs = torch.softmax(outputs, dim=1)[:, 1] 
#         all_probs_val.extend(probs.cpu().numpy())
#         all_labels_val.extend(labels.numpy())

# all_labels_val = np.array(all_labels_val)
# all_probs_val = np.array(all_probs_val)

# # 1. 使用默认阈值 0.5进行评估
# print(f"\n=== Validation Classification Report {LABEL_COLUMN_NAME} (Threshold = 0.5) ===") # 修改标题
# preds_at_0_5 = (all_probs_val >= 0.5).astype(int)
# # 确保 target_names 与您的类别对应，例如 ["PD-L1 Negative", "PD-L1 Positive"]
# target_names_display = [f"{LABEL_COLUMN_NAME} Negative", f"{LABEL_COLUMN_NAME} Positive"]
# report_0_5 = classification_report(all_labels_val, preds_at_0_5, target_names=target_names_display) 
# cm_0_5 = confusion_matrix(all_labels_val, preds_at_0_5)
# print(report_0_5)
# print("Confusion Matrix (Threshold = 0.5):")
# print(cm_0_5)

# def plot_confusion_matrix(cm, class_names, title='混淆矩阵', filename='confusion_matrix.png'):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.title(title)
#     plt.ylabel('实际标签')
#     plt.xlabel('预测标签')
#     plt.savefig(filename)
#     plt.show()

# plot_confusion_matrix(cm_0_5, target_names_display, title=f'{LABEL_COLUMN_NAME} 混淆矩阵 (阈值 0.5)', filename=f'confusion_matrix_{LABEL_COLUMN_NAME.lower()}_0.5.png') # 使用常量

# # 2. 寻找更优的阈值 (这部分可以保留，但在验证集上进行)
# print(f"\n=== Finding Optimal Threshold for {LABEL_COLUMN_NAME}+ (based on F1-score) on Validation Set ===") # 修改标题
# thresholds = np.arange(0.05, 0.95, 0.01)
# best_threshold = 0.5
# best_f1_positive = 0.0
# best_recall_positive = 0.0
# best_precision_positive = 0.0

# for thresh in thresholds:
#     current_preds = (all_probs_val >= thresh).astype(int)
#     # 注意 precision_recall_fscore_support 的参数，特别是 labels=[0, 1] 和 pos_label (如果适用)
#     p, r, f1, s = precision_recall_fscore_support(all_labels_val, current_preds, labels=[0, 1], zero_division=0)
#     f1_positive = f1[1] # 假设类别1是阳性类
#     if f1_positive > best_f1_positive:
#         best_f1_positive = f1_positive
#         best_threshold = thresh
#         best_recall_positive = r[1]
#         best_precision_positive = p[1]
#     elif f1_positive == best_f1_positive: 
#         if r[1] > best_recall_positive : 
#              best_threshold = thresh
#              best_recall_positive = r[1]
#              best_precision_positive = p[1]

# print(f"Optimal threshold found for {LABEL_COLUMN_NAME}+: {best_threshold:.2f}") # 使用常量
# print(f"  {LABEL_COLUMN_NAME}+ F1-score: {best_f1_positive:.4f}") # 使用常量
# print(f"  {LABEL_COLUMN_NAME}+ Precision: {best_precision_positive:.4f}") # 使用常量
# print(f"  {LABEL_COLUMN_NAME}+ Recall: {best_recall_positive:.4f}") # 使用常量

# print(f"\n=== Validation Classification Report {LABEL_COLUMN_NAME} (Optimal Threshold) ===") # 修改标题
# preds_optimal = (all_probs_val >= best_threshold).astype(int)
# report_optimal = classification_report(all_labels_val, preds_optimal, target_names=target_names_display)
# cm_optimal = confusion_matrix(all_labels_val, preds_optimal)
# print(report_optimal)
# print(f"Confusion Matrix (Threshold = {best_threshold:.2f}):")
# print(cm_optimal)

# plot_confusion_matrix(cm_optimal, target_names_display, title=f'{LABEL_COLUMN_NAME} 混淆矩阵 (最佳阈值 {best_threshold:.2f})', filename=f'confusion_matrix_{LABEL_COLUMN_NAME.lower()}_optimal.png') # 使用常量

# def plot_roc_curve(labels, probs, title='ROC曲线', filename='roc_curve.png'):
#     fpr, tpr, roc_thresholds = roc_curve(labels, probs)
#     auc_score = roc_auc_score(labels, probs)
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('假阳性率 (False Positive Rate)')
#     plt.ylabel('真阳性率 (True Positive Rate)')
#     plt.title(title)
#     plt.legend(loc="lower right")
#     plt.grid(True)
#     plt.savefig(filename)
#     plt.show()

# plot_roc_curve(all_labels_val, all_probs_val, title=f'验证集ROC曲线 {LABEL_COLUMN_NAME}', filename=f'roc_curve_{LABEL_COLUMN_NAME.lower()}_val.png') # 使用常量

# def plot_pr_curve(labels, probs, title='P-R曲线', filename='pr_curve.png'):
#     from sklearn.metrics import precision_recall_curve, average_precision_score
#     precision, recall, _ = precision_recall_curve(labels, probs)
#     ap_score = average_precision_score(labels, probs)
#     plt.figure(figsize=(8, 6))
#     plt.plot(recall, precision, color='blue', lw=2, label=f'P-R curve (AP = {ap_score:.4f})')
#     no_skill = len(labels[labels==1]) / len(labels) if len(labels[labels==1]) > 0 else 0 # Handle division by zero for no_skill
#     plt.plot([0, 1], [no_skill, no_skill], color='gray', lw=2, linestyle='--', label=f'No Skill (AP={no_skill:.4f})')
#     plt.xlabel('召回率 (Recall)')
#     plt.ylabel('精确率 (Precision)')
#     plt.title(title)
#     plt.legend(loc="lower left") 
#     plt.grid(True)
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.savefig(filename)
#     plt.show()

# plot_pr_curve(all_labels_val, all_probs_val, title=f'验证集P-R曲线 ({LABEL_COLUMN_NAME}+)', filename=f'pr_curve_{LABEL_COLUMN_NAME.lower()}_val.png') # 使用常量


# %% [markdown]
# ## 验证集最终评估曲线 (ROC & P-R) for PD-L1(Atezo)
# 此部分将从 grade_model.py 引入，并替换上面注释掉的评估部分

# %%
print(f"\nGenerating ROC and P-R curves for {LABEL_COLUMN_NAME} on the validation set using the best model.")
try:
    # 加载最佳模型
    model.load_state_dict(torch.load(f"best_model_{LABEL_COLUMN_NAME.lower()}.pth"))
    model.eval() # 设置模型为评估模式
    print(f"Loaded best {LABEL_COLUMN_NAME} model for final validation set evaluation.")

    all_val_labels_final = []
    all_val_probs_class1_final = [] # 存储类别1的概率
    
    if len(val_ds) > 0: # 确保验证数据集不为空
        with torch.no_grad(): # 关闭梯度计算
            for imgs, labels in tqdm(val_loader, desc=f"Final Validation for ROC/PR {LABEL_COLUMN_NAME}"):
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1) # 获取每个类别的概率
                
                all_val_probs_class1_final.extend(probs[:, 1].cpu().numpy()) # 提取类别1的概率
                all_val_labels_final.extend(labels.numpy()) # 真实标签

        all_val_labels_final_np = np.array(all_val_labels_final)
        all_val_probs_class1_final_np = np.array(all_val_probs_class1_final)

        # 检查验证集标签中是否有至少两个类别用于指标计算
        if len(np.unique(all_val_labels_final_np)) >= NUM_CLASSES:
            try:
                from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score # 确保导入
                # ROC 曲线
                val_auc_final = roc_auc_score(all_val_labels_final_np, all_val_probs_class1_final_np)
                print(f"Final Validation AUC for {LABEL_COLUMN_NAME}: {val_auc_final:.4f}")
                fpr, tpr, _ = roc_curve(all_val_labels_final_np, all_val_probs_class1_final_np)
                
    plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {val_auc_final:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # 对角参考线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)')
    plt.ylabel('真阳性率 (True Positive Rate)')
                plt.title(f'验证集ROC曲线 ({LABEL_COLUMN_NAME})')
    plt.legend(loc="lower right")
    plt.grid(True)
                plt.savefig(f'roc_curve_{LABEL_COLUMN_NAME.lower()}_val.png')
    plt.show()

                # P-R 曲线
                val_ap_final = average_precision_score(all_val_labels_final_np, all_val_probs_class1_final_np)
                print(f"Final Validation Average Precision for {LABEL_COLUMN_NAME}: {val_ap_final:.4f}")
                precision, recall, _ = precision_recall_curve(all_val_labels_final_np, all_val_probs_class1_final_np)

    plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, lw=2, label=f'P-R curve (AP = {val_ap_final:.3f})')
                # No skill line for P-R curve
                no_skill = len(all_val_labels_final_np[all_val_labels_final_np==1]) / len(all_val_labels_final_np) if len(all_val_labels_final_np) > 0 else 0
                plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label=f'No Skill (AP={no_skill:.3f})')

    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
                plt.title(f'验证集P-R曲线 ({LABEL_COLUMN_NAME})') # 之前可能是 ({LABEL_COLUMN_NAME}+)，统一一下
                plt.legend(loc="best") # grade_model 用的是 best
    plt.grid(True)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
                plt.savefig(f'pr_curve_{LABEL_COLUMN_NAME.lower()}_val.png')
    plt.show()

            except ValueError as e_val_curves:
                print(f"Final Validation ROC/PR calculation error for {LABEL_COLUMN_NAME}: {e_val_curves}")
        else:
            print(f"Final Validation ROC/PR not computed for {LABEL_COLUMN_NAME}: validation set does not contain enough distinct classes (needs at least {NUM_CLASSES}). Found: {np.unique(all_val_labels_final_np)}")
    else:
        print(f"Validation dataset for {LABEL_COLUMN_NAME} is empty. No final ROC/PR curves generated.")

except FileNotFoundError:
    print(f"Error: 'best_model_{LABEL_COLUMN_NAME.lower()}.pth' not found. Was the model trained and saved? Cannot generate final ROC/PR curves for {LABEL_COLUMN_NAME}.")
except Exception as e:
    print(f"An error occurred during final {LABEL_COLUMN_NAME} validation set ROC/PR curve generation: {e}")


# %% [markdown]
# ## Grad-CAM 可视化

# %%
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from torchvision.utils import make_grid, save_image

def get_embeddings_efficientnet(model, dataloader, device): # 函数名修改为 efficientnet
    model.eval()
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=f"Extracting embeddings for {LABEL_COLUMN_NAME}"):
            imgs = imgs.to(device)
            # For EfficientNet, model.features extracts features, then adaptive_avg_pool2d, then classifier
            # We will take output of adaptive_avg_pool2d as embeddings
            features = model.features(imgs)
            pooled_features = model.avgpool(features) # (batch_size, num_features, 1, 1)
            embeddings = torch.flatten(pooled_features, 1) # (batch_size, num_features)
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.numpy())
    
    if not embeddings_list: # 处理空列表的情况
        return np.array([]), np.array([])
    return np.concatenate(embeddings_list), np.concatenate(labels_list)

def calculate_mutual_information(features, labels):
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if len(features) == 0 or len(labels) == 0 or len(features) != len(labels):
        print("Warning: MI calculation skipped due to empty or mismatched feature/label arrays.")
        return np.array([0.0]) # 返回一个默认值，避免后续出错
    mi = mutual_info_classif(features, labels, random_state=SEED)
    return mi

def plot_tsne_visualization(embeddings, labels, title_suffix=""):
    if len(embeddings) == 0 or len(labels) == 0 or len(embeddings) != len(labels):
        print(f"Skipping t-SNE for {LABEL_COLUMN_NAME}{title_suffix}: Empty or mismatched embeddings/labels.")
        return
    if len(embeddings) < 2 : # t-SNE 需要至少2个样本
        print(f"Skipping t-SNE for {LABEL_COLUMN_NAME}{title_suffix}: Not enough samples ({len(embeddings)}).")
        return

    print(f"Running t-SNE for {LABEL_COLUMN_NAME}{title_suffix}...")
    # 如果样本数量小于等于perplexity，调整perplexity
    perplexity_value = min(30, len(embeddings) - 1) if len(embeddings) > 1 else 5
    if perplexity_value <=0: perplexity_value = 5 # 确保perplexity为正

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity_value, n_iter=1000, init='pca', learning_rate='auto')
    try:
        embeddings_2d = tsne.fit_transform(embeddings)
    except Exception as e_tsne:
        print(f"Error during t-SNE for {LABEL_COLUMN_NAME}{title_suffix}: {e_tsne}. Skipping t-SNE plot.")
        return
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("viridis", len(unique_labels) if len(unique_labels) > 0 else 1)
    
    for i, label_val in enumerate(unique_labels):
        idx = labels == label_val
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], color=colors(i), label=f'{LABEL_COLUMN_NAME} {label_val}', alpha=0.7)
    
    plt.title(f't-SNE 可视化 ({LABEL_COLUMN_NAME}{title_suffix})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    if len(unique_labels) > 0 : plt.legend()
    plt.grid(True)
    plt.savefig(f"tsne_visualization_{LABEL_COLUMN_NAME.lower()}{title_suffix.replace(' ', '_').lower()}.png")
    plt.show()

def simulate_data_cleaning_test(model, original_val_loader, original_labels_np, num_samples_to_flip=50): # Reduced samples to flip
    print(f"\nSimulating data cleaning test for {LABEL_COLUMN_NAME} by flipping up to {num_samples_to_flip} labels...")
    
    actual_samples_to_flip = min(num_samples_to_flip, len(original_labels_np) // 2) # Flip at most half, or num_samples_to_flip
    if len(original_labels_np) == 0 or actual_samples_to_flip == 0:
        print(f"Warning: Not enough samples ({len(original_labels_np)}) or 0 samples to flip. Skipping simulation.")
        return float('nan')

    flipped_labels_np = original_labels_np.copy()
    # Ensure at least one sample from each class if possible, before flipping, to make flipping meaningful
    indices_to_flip = np.random.choice(len(flipped_labels_np), actual_samples_to_flip, replace=False)
    
    flipped_labels_np[indices_to_flip] = 1 - flipped_labels_np[indices_to_flip]
    
    # We need probabilities from the model. Use all_val_probs_class1_final_np if available from previous steps.
    global all_val_probs_class1_final_np
    eval_probs_np = None
    if 'all_val_probs_class1_final_np' in globals() and all_val_probs_class1_final_np is not None and len(all_val_probs_class1_final_np) == len(flipped_labels_np):
        eval_probs_np = all_val_probs_class1_final_np
        else:
        print("Warning: `all_val_probs_class1_final_np` not available or mismatched for data cleaning. Re-evaluating model.")
        model.eval()
        temp_probs_list = []
        # Need val_ds to create a temporary loader if original_val_loader is not directly usable here
        # Or better, ensure original_val_loader is passed correctly and usable
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
            permuted_aucs.append(0.5) # Or handle as NaN, but 0.5 is common for random chance
            continue
        try:
            auc = roc_auc_score(permuted_labels, original_probs_np) # Use original probabilities
            permuted_aucs.append(auc)
        except ValueError:
             permuted_aucs.append(0.5) 

    permuted_aucs = np.array(permuted_aucs)
    if len(permuted_aucs) == 0: # Should not happen if n_permutations > 0
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
        # Ensure all_val_labels_final_np and all_val_probs_class1_final_np are available
        # These should have been computed in the "验证集最终评估曲线" section.
        # If not, recalculate them here.
        if 'all_val_labels_final_np' not in globals() or 'all_val_probs_class1_final_np' not in globals() or \
            all_val_labels_final_np is None or all_val_probs_class1_final_np is None or \
            len(all_val_labels_final_np) != len(val_ds) or len(all_val_probs_class1_final_np) != len(val_ds):
            
            print("Recalculating final validation labels and probabilities for analysis...")
            temp_all_val_labels_final_list = []
            temp_all_val_probs_class1_final_list = []
            model.eval()
            with torch.no_grad():
                for imgs_an, labels_an in tqdm(val_loader, desc=f"Final Validation for Analysis {LABEL_COLUMN_NAME}"):
                    imgs_an = imgs_an.to(device)
                    outputs_an = model(imgs_an)
                    probs_an = torch.softmax(outputs_an, dim=1)
                    temp_all_val_probs_class1_final_list.extend(probs_an[:, 1].cpu().numpy())
                    temp_all_val_labels_final_list.extend(labels_an.numpy())
            all_val_labels_final_np = np.array(temp_all_val_labels_final_list)
            all_val_probs_class1_final_np = np.array(temp_all_val_probs_class1_final_list)
            if len(all_val_labels_final_np) == 0: # Check if still empty
                 print("Validation set is empty after trying to recalculate labels/probs for analysis. Skipping analysis.")
                 raise ValueError("Empty validation set for analysis") # Raise error to skip the rest of analysis block

        val_embeddings, val_true_labels_for_analysis = get_embeddings_efficientnet(model, val_loader, device) # Use updated function name
        
        if val_embeddings.size > 0 and len(val_embeddings) > 0 and len(val_true_labels_for_analysis) == len(val_embeddings):
            # 2. Calculate Mutual Information
            if all_val_probs_class1_final_np is not None and len(all_val_probs_class1_final_np) == len(val_true_labels_for_analysis):
                mi_scores_probs = calculate_mutual_information(all_val_probs_class1_final_np.reshape(-1, 1), val_true_labels_for_analysis)
                print(f"Mutual Information (Class 1 Probs vs Labels) for {LABEL_COLUMN_NAME}: {mi_scores_probs[0]:.4f}")
            else:
                print("Could not calculate MI with probabilities, not available or mismatched length.")

            # 3. t-SNE Visualization
            plot_tsne_visualization(val_embeddings, val_true_labels_for_analysis)

            # 4. Simulate Data Cleaning Test
            simulate_data_cleaning_test(model, val_loader, all_val_labels_final_np, num_samples_to_flip=max(1, len(val_ds)//10)) # Flip 10% or at least 1
            
            # 5. Perform Permutation Test
            if all_val_labels_final_np is not None and all_val_probs_class1_final_np is not None and len(all_val_labels_final_np)>0:
                perform_permutation_test(model, val_loader, all_val_labels_final_np, all_val_probs_class1_final_np, n_permutations=1000)
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
# ## Grad-CAM 可视化 (PD-L1(Atezo) - 更新后版本)

# %%
# 定义类别名称，用于 Grad-CAM 标题
TASK_CLASS_NAMES = [f'{LABEL_COLUMN_NAME} Negative', f'{LABEL_COLUMN_NAME} Positive'] 

def visualize_grad_cam_updated(model, dataset, device, num_images_per_class=4, target_classes_to_viz=None):
    # 适配 EfficientNet-B0 的最后一个卷积层
    # 通常是 model.features的最后一个block中的最后一个卷积层。对于EfficientNet-B0，这通常是 model.features[-1][0]
    # 或者更鲁棒地，可以尝试遍历 model.features 找到最后一个 nn.Conv2d
    target_layer = None
    if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
        for i in range(len(model.features) -1, -1, -1):
            block = model.features[i]
            if isinstance(block, nn.Sequential) and len(block) > 0: # MBConv block
                # Look for the last conv layer in this block, typically in .block[-1] or .block[-2] then a conv
                # A common structure is: MBConv -> block (Sequential) -> _project_conv (Conv2d)
                # Or within the block, the last conv2d of the main path
                if hasattr(block, '_project_conv') and isinstance(block._project_conv, nn.Conv2d):
                    target_layer = block._project_conv
                    print(f"Grad-CAM target layer found: model.features[{i}]._project_conv")
                    break
                # Iterate inside the block if it's a Sequential itself (e.g. the last MBConv block's internal sequence)
                # This part might need more specific introspection for EfficientNet variants
                # For B0, model.features[-1][0] (the Conv2dNormActivation within the last MBConv's expansion) is a common choice
                # Or model.features[-1][-1] if it's a Conv2dNormActivation that is the final conv output of the block.
                # Let's try a common one: last sub-block, last conv-like module in that
                if hasattr(model.features[-1], 'block') and isinstance(model.features[-1].block, nn.Sequential) and len(model.features[-1].block) > 0:
                     # The last conv in the inverted residual block of EfficientNet is often within its .block attribute, e.g., .block[-1][0] or similar
                     # For efficientnet_b0, model.features[-1][0] (if it's Conv2dNormActivation) or model.features[-1].block[-1][0]
                     # Defaulting to model.features[-1][0] which worked before and is a conv layer.
                     if isinstance(model.features[-1][0], (nn.Conv2d, nn.modules.conv.Conv2d)) or \ 
                        (hasattr(model.features[-1][0], ' তাক') and isinstance(model.features[-1][0]. তাক, nn.Conv2d)): # for Conv2dNormActivation
                         target_layer = model.features[-1][0]
                         print(f"Grad-CAM target layer tentatively set to model.features[-1][0]")
                         break
            elif isinstance(block, (nn.Conv2d, nn.modules.conv.Conv2d)): # Fallback: last overall Conv2d in features
                target_layer = block
                print(f"Grad-CAM target layer found by fallback: model.features[{i}]")
                break
    if target_layer is None: # Ultimate fallback
        print("Could not automatically determine a suitable Conv2d target layer for Grad-CAM. Defaulting to model.features[-1]. This might not be optimal.")
        target_layers = [model.features[-1]]
    else:
        target_layers = [target_layer]

    cam = GradCAM(model=model, target_layers=target_layers)

    if len(dataset) == 0:
        print(f"Dataset for {LABEL_COLUMN_NAME} Grad-CAM is empty.") 
        return

    if target_classes_to_viz is None:
        target_classes_to_viz = list(range(NUM_CLASSES))
    if not isinstance(target_classes_to_viz, list): target_classes_to_viz = [target_classes_to_viz]
    
    num_viz_rows = len(target_classes_to_viz)
    num_viz_cols = num_images_per_class 
    
    if num_viz_rows * num_viz_cols == 0:
        print(f"No images or target classes specified for {LABEL_COLUMN_NAME} Grad-CAM.") 
        return

    fig, axes = plt.subplots(num_viz_rows * 2, num_viz_cols, figsize=(num_viz_cols * 3, num_viz_rows * 6))
    # Ensure axes is always a 2D array for consistent indexing, even if 1 image or 1 class
    if num_viz_rows == 1 and num_viz_cols == 1: axes = np.array(axes).reshape(2,1)
    elif num_viz_rows == 1: axes = axes.reshape(2, num_viz_cols)
    elif num_viz_cols == 1: axes = axes.reshape(num_viz_rows * 2, 1)

    images_shown_count = 0
    for r_idx, target_cls in enumerate(target_classes_to_viz):
        # Get indices of images belonging to the target_cls or any class if not enough specific examples
        class_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == target_cls]
        if len(class_indices) < num_images_per_class:
            # If not enough images for this class, supplement with random images
            print(f"Warning: Not enough images for class {target_cls} ({len(class_indices)} found, need {num_images_per_class}). Supplementing with random images.")
            other_indices = [i for i in range(len(dataset)) if i not in class_indices]
            needed_more = num_images_per_class - len(class_indices)
            if len(other_indices) >= needed_more:
                class_indices.extend(np.random.choice(other_indices, needed_more, replace=False))
            else: # Still not enough, take all available
                class_indices.extend(other_indices)
        
        indices_to_use = np.random.choice(class_indices, min(num_images_per_class, len(class_indices)), replace=False) if len(class_indices) > 0 else []

        for c_idx_local, img_idx_in_dataset in enumerate(indices_to_use): 
            current_plot_col = c_idx_local
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
                print(f"Warning: {LABEL_COLUMN_NAME} Grad-CAM returned None or empty for image index {img_idx_in_dataset}, target class {target_cls}.") 
                ax_orig = axes[r_idx * 2, current_plot_col] 
                ax_cam  = axes[r_idx * 2 + 1, current_plot_col]
                ax_orig.axis('off'); ax_cam.axis('off')
                continue
        
        cam_image = show_cam_on_image(rgb_img_denorm, grayscale_cam_batch, use_rgb=True)
        cam_image_tensor = transforms.ToTensor()(cam_image) 
        original_img_for_grid = transforms.ToTensor()(Image.fromarray((rgb_img_denorm * 255).astype(np.uint8)))
        
            title_str = f"True: {TASK_CLASS_NAMES[true_label]}\nCAM for: {TASK_CLASS_NAMES[target_cls]}"

            ax_orig_current = axes[r_idx * 2, current_plot_col] 
            ax_cam_current  = axes[r_idx * 2 + 1, current_plot_col]

            ax_orig_current.imshow(original_img_for_grid.permute(1,2,0).numpy())
            ax_orig_current.set_title(title_str, fontsize=8)
            ax_orig_current.axis('off')
            ax_cam_current.imshow(cam_image_tensor.permute(1,2,0).numpy())
            ax_cam_current.axis('off')
            images_shown_count +=1

    if images_shown_count == 0:
        print(f"No {LABEL_COLUMN_NAME} CAM images were generated.") 
        if num_viz_rows * num_viz_cols > 0 : plt.close(fig) 
        return

    fig.suptitle(f"Grad-CAM for {LABEL_COLUMN_NAME} Model (Targeting Various Classes)", fontsize=12) 
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_filename = f'grad_cam_{LABEL_COLUMN_NAME.lower()}_binary.png' 
    plt.savefig(save_filename)
    print(f"Grad-CAM grid for {LABEL_COLUMN_NAME} saved to {save_filename}") 
    plt.show()


# 调用更新后的 Grad-CAM 可视化
# visualize_grad_cam(model, target_layer_name='features[-1]', dataset=val_ds, device=device, num_images=4, target_class=1)
# visualize_grad_cam(model, target_layer_name='features[-1]', dataset=val_ds, device=device, num_images=4, target_class=0)
if 'model' in locals() and 'val_ds' in locals() and len(val_ds) > 0:
    print(f"\nVisualizing Grad-CAM for {LABEL_COLUMN_NAME} model using updated function") 
    visualize_grad_cam_updated(model, dataset=val_ds, device=device, num_images_per_class=4, target_classes_to_viz=[0,1]) 
else:
    print(f"Skipping {LABEL_COLUMN_NAME} Grad-CAM (updated): Model or validation dataset not available or val_ds is empty.") 

# %% [markdown]
# ## 结束语

# %%
print(f"{LABEL_COLUMN_NAME} binary classification model script generation and modification complete.") 
print(f"IMPORTANT: Review and adjust 'LABEL_COLUMN_NAME' ('{LABEL_COLUMN_NAME}'), and FocalLoss 'alpha' parameters for {LABEL_COLUMN_NAME} task if needed.")

# %%