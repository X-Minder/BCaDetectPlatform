# %% [markdown]
# # Grade 膀胱镜图像二分类模型 (ConvNeXt版本)
# 使用 ConvNeXt-Tiny 对 Grade 进行二分类 (0/1),按患者划分训练/验证集。
# 此版本与 app2.py 中的 ConvNeXtClassifier 完全兼容。

# %%
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support, \
    roc_curve, average_precision_score, precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.utils import shuffle as sklearn_shuffle


# 配置matplotlib中文字体
def setup_chinese_font():
    import matplotlib.font_manager as fm
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', 'WenQuanYi Micro Hei',
                     'Noto Sans CJK SC']
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
# ## ConvNeXt 模型定义 (与 app2.py 完全一致)
# %%
class ConvNeXtClassifier(nn.Module):
    """ConvNeXt 分类器 - 与 app2.py 完全一致"""

    def __init__(self, dropout_rate=0.6, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=0, global_pool='')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.num_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


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
            if isinstance(alpha, list) and len(alpha) == 2:
                self.alpha = torch.tensor(alpha)
            elif isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha, alpha])
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha] * num_classes)

        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
            if len(self.alpha) != num_classes:
                if len(self.alpha) < num_classes:
                    self.alpha = torch.cat(
                        [self.alpha, torch.full((num_classes - len(self.alpha),), 1.0 / num_classes)])
                else:
                    self.alpha = self.alpha[:num_classes]

        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt) ** self.gamma * CE_loss

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
# ## 自定义 Dataset for Grade
# %%
class GradeDataset(Dataset):
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
# ## 读取标签并划分数据集 for Grade
# %%
LABEL_COLUMN_NAME = 'Grade'
MAPPED_LABEL_COLUMN_NAME = 'Grade_mapped'
PATIENT_ID_COLUMN = 'PATIENT_ID'
FILE_NAME_COLUMN = 'FILE_NAME'
NUM_CLASSES = 2

label_df = pd.read_csv("D:\project3\image_dataset\data_task3\label.csv")
print(f"1. Initial rows loaded from CSV: {len(label_df)}")

label_df = label_df.dropna(subset=[LABEL_COLUMN_NAME, FILE_NAME_COLUMN, PATIENT_ID_COLUMN]).copy()
print(f"2. Rows after dropping NA: {len(label_df)}")

if len(label_df) == 0:
    print(f"ERROR: All rows were dropped. Please check CSV.")
else:
    print(f"   Unique values in '{LABEL_COLUMN_NAME}': {label_df[LABEL_COLUMN_NAME].astype(str).unique()}")


    def map_grade_to_binary(grade_status):
        if pd.isna(grade_status):
            return np.nan

        status_str = str(grade_status).strip()
        if not status_str:
            return np.nan

        try:
            status_val_float = float(status_str)
            status_val = int(status_val_float)

            if status_val != status_val_float:
                return np.nan

            if status_val in [0, 1]:
                return status_val
            else:
                return np.nan
        except ValueError:
            return np.nan


    label_df[MAPPED_LABEL_COLUMN_NAME] = label_df[LABEL_COLUMN_NAME].apply(map_grade_to_binary)
    label_df = label_df.dropna(subset=[MAPPED_LABEL_COLUMN_NAME]).copy()
    print(f"3. Rows after mapping: {len(label_df)}")

    if len(label_df) > 0:
        label_df[MAPPED_LABEL_COLUMN_NAME] = label_df[MAPPED_LABEL_COLUMN_NAME].astype(int)
        print(f"   Final unique values: {label_df[MAPPED_LABEL_COLUMN_NAME].unique()}")

if len(label_df) > 0:
    df_trainval = label_df.copy()
    val_size_from_trainval = 0.2

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_from_trainval, random_state=SEED + 1)

    if len(df_trainval[PATIENT_ID_COLUMN].unique()) > 1 and len(df_trainval) > 1:
        try:
            train_idx_inner, val_idx_inner = next(gss_val.split(df_trainval, groups=df_trainval[PATIENT_ID_COLUMN]))
            df_train = df_trainval.iloc[train_idx_inner].copy()
            df_val = df_trainval.iloc[val_idx_inner].copy()
            print(f"Successfully used GroupShuffleSplit for train/validation sets.")
        except ValueError as e:
            stratify_col = df_trainval[MAPPED_LABEL_COLUMN_NAME] if df_trainval[
                                                                        MAPPED_LABEL_COLUMN_NAME].nunique() > 1 else None
            df_train, df_val = train_test_split(df_trainval, test_size=val_size_from_trainval, random_state=SEED + 1,
                                                stratify=stratify_col)
    elif len(df_trainval) > 0:
        if len(df_trainval) > 1:
            stratify_col = df_trainval[MAPPED_LABEL_COLUMN_NAME] if df_trainval[
                                                                        MAPPED_LABEL_COLUMN_NAME].nunique() > 1 else None
            df_train, df_val = train_test_split(df_trainval, test_size=val_size_from_trainval, random_state=SEED + 1,
                                                stratify=stratify_col)
        else:
            df_train = df_trainval.copy()
            df_val = pd.DataFrame(columns=df_trainval.columns)
    else:
        df_train = pd.DataFrame(columns=label_df.columns)
        df_val = pd.DataFrame(columns=label_df.columns)

    print(f"\nDataset sizes and class distributions:")
    for name, df_subset in [("Train", df_train), ("Val", df_val)]:
        if not df_subset.empty:
            print(f"  {name:<8}: {len(df_subset):>4} images, Patients: {df_subset[PATIENT_ID_COLUMN].nunique():>2}")
            distribution_info_series = df_subset[MAPPED_LABEL_COLUMN_NAME].value_counts(normalize=True).sort_index()
            distribution_info_str = '\n'.join(
                [f"    Class {idx}: {val:.4f}" for idx, val in distribution_info_series.items()])
            print(f"    Class distribution:\n{distribution_info_str}")
        else:
            print(f"  {name:<8}: Empty")
else:
    df_train, df_val = pd.DataFrame(), pd.DataFrame()

# %% [markdown]
# ## 数据增强与 DataLoader
# %%
IMG_DIR = "D:\project3\image_dataset\data_task3\image"

# ConvNeXt 推荐的数据增强
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader_args = {'shuffle': True}
if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
    counts_train = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
    if len(counts_train) >= 1 and len(counts_train) <= NUM_CLASSES:
        class_sample_weights = [1. / counts_train.get(i, 1e-6) for i in range(NUM_CLASSES)]
        sample_weights_train = [class_sample_weights[label] for label in df_train[MAPPED_LABEL_COLUMN_NAME]]
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights_train),
                                        num_samples=len(sample_weights_train),
                                        replacement=True)
        train_loader_args = {'sampler': sampler, 'shuffle': False}
        print(f"Using WeightedRandomSampler for class balance")

train_ds = GradeDataset(df_train, IMG_DIR, transform=train_tf)
val_ds = GradeDataset(df_val, IMG_DIR, transform=val_tf)

BATCH_SIZE = 16
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, **train_loader_args)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# %% [markdown]
# ## 模型定义与训练设置
# %%
if __name__ == '__main__':
    print(f"Using ConvNeXt-Tiny for {LABEL_COLUMN_NAME} {NUM_CLASSES}-class classification")
    model = ConvNeXtClassifier(dropout_rate=0.6, num_classes=NUM_CLASSES).to(device)

    # 计算 Focal Loss alpha
    focal_loss_alpha_values = [0.5, 0.5]
    if not df_train.empty and MAPPED_LABEL_COLUMN_NAME in df_train.columns:
        counts = df_train[MAPPED_LABEL_COLUMN_NAME].value_counts().sort_index()
        if len(counts) == NUM_CLASSES:
            class_weights = [1.0 / counts.get(i, 1e-6) for i in range(NUM_CLASSES)]
            total_weight = sum(class_weights)
            focal_loss_alpha_values = [w / total_weight for w in class_weights]
            print(f"Calculated FocalLoss alpha: {focal_loss_alpha_values}")

    focal_loss_alpha = torch.tensor(focal_loss_alpha_values, dtype=torch.float).to(device)
    criterion = FocalLoss(alpha=focal_loss_alpha, gamma=2, num_classes=NUM_CLASSES)

    # 使用 AdamW 优化器,学习率较小
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True,
                                                     min_lr=1e-7)


    # %% [markdown]
    # ## 早停机制
    # %%
    class EarlyStopping:
        def __init__(self, patience=20, min_delta=0.0001, restore_best_weights=True, mode='min', verbose=True):
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
                    print(f"EarlyStopping: New best score: {self.best_score:.4f}")
            else:
                self.counter += 1
                if self.verbose:
                    print(f"EarlyStopping: Counter {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Patience reached. Stopping training.")
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
            return self.early_stop


    # %% [markdown]
    # ## 训练与验证循环
    # %%
    NUM_EPOCHS = 100
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=20, mode='min', verbose=True, min_delta=0.0001)

    history = {
        'train_loss': [], 'val_loss': [], 'val_accuracy': [],
        'val_auc': [], 'lr': []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds) if len(train_ds) > 0 else 0.0
        history['train_loss'].append(train_loss)

        # 验证
        model.eval()
        all_val_labels = []
        all_val_probs_class1 = []
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

                    all_val_probs_class1.extend(probs[:, 1].cpu().numpy())
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
            if len(np.unique(all_val_labels_np)) >= 2:
                try:
                    val_auc = roc_auc_score(all_val_labels_np, all_val_probs_class1_np)
                except ValueError as e_auc:
                    print(f"Warning: AUC calculation error: {e_auc}")
            history['val_auc'].append(val_auc)

            scheduler.step(val_epoch_loss)
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)

            print(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_epoch_loss:.4f}, Val Acc={val_accuracy:.4f}, Val AUC={val_auc:.4f}, LR={current_lr:.1e}")

            # 保存最佳模型
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(model.state_dict(), "grading_model.pth")
                print(f"✅ Epoch {epoch}: New best model saved with Val Loss: {best_val_loss:.4f}")

                # 早停检查
            if early_stopping(val_epoch_loss, model):
                print(f"Early stopping triggered at epoch {epoch}")
                break
        else:
            history['val_loss'].append(float('nan'))
            history['val_accuracy'].append(float('nan'))
            history['val_auc'].append(float('nan'))
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)


    # %% [markdown]
    # ## 绘制训练过程曲线
    # %%
    def plot_training_history(history):
        epochs_ran = len(history['train_loss'])
        epoch_ticks = range(1, epochs_ran + 1)

        fig, ax1 = plt.subplots(figsize=(14, 7))
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(epoch_ticks, history['train_loss'], color=color, linestyle='-', marker='o', markersize=3,
                 label='训练损失')
        if 'val_loss' in history and any(not np.isnan(x) for x in history['val_loss']):
            ax1.plot(epoch_ticks, history['val_loss'], color=color, linestyle=':', marker='x', markersize=3,
                     label='验证损失')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy / AUC', color=color)
        if 'val_accuracy' in history and any(not np.isnan(x) for x in history['val_accuracy']):
            ax2.plot(epoch_ticks, history['val_accuracy'], color=color, linestyle='-', marker='s', markersize=3,
                     label='验证准确率')
        if 'val_auc' in history and any(not np.isnan(x) for x in history['val_auc']):
            ax2.plot(epoch_ticks, history['val_auc'], color='tab:purple', linestyle='--', marker='^', markersize=3,
                     label='验证 AUC')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        ax2.set_ylim(0, 1.05)

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        color = 'tab:green'
        ax3.set_ylabel('Learning Rate', color=color)
        if 'lr' in history and len(history['lr']) == epochs_ran:
            ax3.plot(epoch_ticks, history['lr'], color=color, linestyle='--', marker='.', markersize=3, label='学习率')
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.legend(loc='lower right')
        ax3.set_yscale('log')

        fig.tight_layout()
        plt.title(f'{LABEL_COLUMN_NAME} ConvNeXt 训练过程监控')
        plt.xticks(epoch_ticks)
        plt.savefig(f"training_history_convnext_{LABEL_COLUMN_NAME.lower()}.png")
        plt.show()


    if any(history.values()):
        plot_training_history(history)

        # %% [markdown]
    # ## 验证集最终评估曲线 (ROC & P-R)
    # %%
    print(f"\n生成 ROC 和 P-R 曲线...")
    try:
        model.load_state_dict(torch.load("grading_model.pth"))
        model.eval()
        print(f"已加载最佳模型进行最终评估")

        all_val_labels_final = []
        all_val_probs_class1_final = []

        if len(val_ds) > 0:
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc="最终验证评估"):
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    probs = torch.softmax(outputs, dim=1)

                    all_val_probs_class1_final.extend(probs[:, 1].cpu().numpy())
                    all_val_labels_final.extend(labels.numpy())

            all_val_labels_final_np = np.array(all_val_labels_final)
            all_val_probs_class1_final_np = np.array(all_val_probs_class1_final)

            if len(np.unique(all_val_labels_final_np)) >= 2:
                try:
                    # ROC Curve
                    val_auc_final = roc_auc_score(all_val_labels_final_np, all_val_probs_class1_final_np)
                    print(f"最终验证 AUC: {val_auc_final:.4f}")
                    fpr, tpr, _ = roc_curve(all_val_labels_final_np, all_val_probs_class1_final_np)

                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {val_auc_final:.3f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('假阳性率 (False Positive Rate)')
                    plt.ylabel('真阳性率 (True Positive Rate)')
                    plt.title(f'验证集ROC曲线 (ConvNeXt {LABEL_COLUMN_NAME})')
                    plt.legend(loc="lower right")
                    plt.grid(True)
                    plt.savefig(f'roc_curve_convnext_{LABEL_COLUMN_NAME.lower()}_val.png')
                    plt.show()

                    # Precision-Recall Curve
                    val_ap_final = average_precision_score(all_val_labels_final_np, all_val_probs_class1_final_np)
                    print(f"最终验证 Average Precision: {val_ap_final:.4f}")
                    precision, recall, _ = precision_recall_curve(all_val_labels_final_np,
                                                                  all_val_probs_class1_final_np)

                    plt.figure(figsize=(8, 6))
                    plt.plot(recall, precision, lw=2, label=f'P-R curve (AP = {val_ap_final:.3f})')
                    plt.xlabel('召回率 (Recall)')
                    plt.ylabel('精确率 (Precision)')
                    plt.title(f'验证集P-R曲线 (ConvNeXt {LABEL_COLUMN_NAME})')
                    plt.legend(loc="best")
                    plt.grid(True)
                    plt.ylim([0.0, 1.05])
                    plt.xlim([0.0, 1.0])
                    plt.savefig(f'pr_curve_convnext_{LABEL_COLUMN_NAME.lower()}_val.png')
                    plt.show()

                except ValueError as e_val_curves:
                    print(f"ROC/PR 计算错误: {e_val_curves}")

    except FileNotFoundError:
        print(f"错误: 'grading_model.pth' 未找到")
    except Exception as e:
        print(f"评估过程出错: {e}")

    # %%
    print(f"\n✅ {LABEL_COLUMN_NAME} ConvNeXt 分类模型训练完成!")
    print(f"📁 模型已保存为: grading_model.pth")
    print(f"💡 此模型与 app2.py 中的 ConvNeXtClassifier 完全兼容,存为: grading_model.pth")
print(f"💡 此模型与 app2.py 中的 ConvNeXtClassifier 完全兼容")