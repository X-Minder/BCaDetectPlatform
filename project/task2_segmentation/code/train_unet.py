# %% [markdown]
# # 膀胱镜图像分割训练脚本
# 本脚本实现了基于U-Net的医学图像分割模型训练。
# 主要功能：
# 1. 数据加载和预处理
# 2. 模型训练和验证
# 3. 指标计算和可视化
# 4. 模型保存和早停

# %% [markdown]
# ## 导入必要的库

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，确保在无显示设备的环境下也能保存图片
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2


# %%
@dataclass
class UNetTrainer:
    # 基础配置
    random_seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据集配置
    data_dir: str = "dataset"  # 根数据目录
    train_img_dir: str = "dataset/images"
    train_mask_dir: str = "dataset/masks"
    image_size: int = 256
    batch_size: int = 16
    train_val_split: float = 0.8
    num_workers: int = 1 # 数据加载的工作进程数
    
    # 模型配置
    in_channels: int = 3
    out_channels: int = 1
    
    # 训练配置
    num_epochs: int = 100
    learning_rate: float = 5e-5
    use_amp: bool = True
    
    # 早停配置
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # 保存配置
    save_dir: str = "checkpoints"
    save_prefix: str = "unet"

    def __post_init__(self):
        """初始化后的处理"""
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 确保设备设置正确
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA不可用，切换到CPU")
            self.device = 'cpu'
            
        # 检查数据集
        self._check_dataset()
        
        # 设置随机种子
        self._set_seed()
        
        # 初始化数据增强
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        
        # 初始化训练组件
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        self.early_stopping = None

    def _set_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.random_seed)
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
    
    def _get_train_transforms(self):
        """获取训练数据增强"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    def _get_val_transforms(self):
        """获取验证数据增强"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    def _check_dataset(self):
        """检查数据集完整性"""
        # 检查目录是否存在
        if not os.path.exists(self.train_img_dir):
            raise ValueError(f"图像目录不存在: {self.train_img_dir}")
        if not os.path.exists(self.train_mask_dir):
            raise ValueError(f"掩码目录不存在: {self.train_mask_dir}")
        
        # 获取图像和掩码文件列表
        image_files = set(os.listdir(self.train_img_dir))
        mask_files = set(f.replace('_mask.png', '.jpg') for f in os.listdir(self.train_mask_dir))
        
        # 检查文件对应关系
        missing_masks = image_files - mask_files
        missing_images = mask_files - image_files
        
        if missing_masks:
            print(f"警告: 以下图像缺少对应的掩码:")
            for f in sorted(missing_masks):
                print(f"  - {f}")
        
        if missing_images:
            print(f"警告: 以下掩码缺少对应的图像:")
            for f in sorted(missing_images):
                print(f"  - {f}")
        
        # 打印数据集信息
        valid_pairs = image_files & mask_files
        print(f"\n数据集信息:")
        print(f"- 总图像数量: {len(image_files)}")
        print(f"- 总掩码数量: {len(mask_files)}")
        print(f"- 有效图像-掩码对数量: {len(valid_pairs)}")
        
        # 检查图像格式
        sample_image = os.path.join(self.train_img_dir, next(iter(valid_pairs)))
        with Image.open(sample_image) as img:
            print(f"- 图像格式: {img.format}")
            print(f"- 图像模式: {img.mode}")
            print(f"- 图像大小: {img.size}")
        
        # 检查掩码格式
        sample_mask = os.path.join(self.train_mask_dir, next(iter(valid_pairs)).replace('.jpg', '_mask.png'))
        with Image.open(sample_mask) as mask:
            print(f"- 掩码格式: {mask.format}")
            print(f"- 掩码模式: {mask.mode}")
            print(f"- 掩码大小: {mask.size}")

    def _get_model_save_path(self, epoch: int, metric_value: float) -> str:
        """获取模型保存路径"""
        return os.path.join(
            self.save_dir,
            f"{self.save_prefix}_epoch{epoch}_dice{metric_value:.4f}.pth"
        )
    
    def _get_best_model_path(self) -> str:
        """获取最佳模型保存路径"""
        return os.path.join(self.save_dir, f"{self.save_prefix}_best.pth")
    
    def _save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool = False):
        """保存检查点"""
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': val_metrics,
        }
        
        if is_best:
            save_dict['best_dice'] = val_metrics['dice']
            torch.save(save_dict, self._get_best_model_path())
            print("Saved best model!")
        
        # 保存带指标的检查点
        torch.save(save_dict, self._get_model_save_path(epoch, val_metrics['dice']))

    def _train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        metrics_sum = {'sensitivity': 0, 'specificity': 0, 'dice': 0, 'iou': 0}
        
        for images, masks in tqdm(train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            batch_metrics = calculate_metrics(outputs, masks)
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]
        
        num_batches = len(train_loader)
        avg_metrics = {key: value / num_batches for key, value in metrics_sum.items()}
        return total_loss / num_batches, avg_metrics
    
    def _validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        metrics_sum = {'sensitivity': 0, 'specificity': 0, 'dice': 0, 'iou': 0}
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                batch_metrics = calculate_metrics(outputs, masks)
                for key in metrics_sum:
                    metrics_sum[key] += batch_metrics[key]
        
        num_batches = len(val_loader)
        avg_metrics = {key: value / num_batches for key, value in metrics_sum.items()}
        return total_loss / num_batches, avg_metrics

    def _plot_training_curves(self, train_losses, val_losses, train_metrics_history, val_metrics_history):
        """绘制训练曲线"""
        plt.figure(figsize=(25, 12))
        
        # 1. 损失曲线
        plt.subplot(2, 3, 1)
        plt.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=10)
        
        # 2. Dice系数曲线
        plt.subplot(2, 3, 2)
        train_dice = [m['dice'] for m in train_metrics_history]
        val_dice = [m['dice'] for m in val_metrics_history]
        plt.plot(train_dice, 'b-', label='Train Dice', linewidth=2)
        plt.plot(val_dice, 'r-', label='Val Dice', linewidth=2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Dice Coefficient', fontsize=12)
        plt.title('Dice Curve', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=10)
        
        # 3. IOU曲线
        plt.subplot(2, 3, 3)
        train_iou = [m['iou'] for m in train_metrics_history]
        val_iou = [m['iou'] for m in val_metrics_history]
        plt.plot(train_iou, 'b-', label='Train IoU', linewidth=2)
        plt.plot(val_iou, 'r-', label='Val IoU', linewidth=2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('IoU', fontsize=12)
        plt.title('IoU Curve', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=10)
        
        # 4. Sensitivity曲线
        plt.subplot(2, 3, 4)
        train_sens = [m['sensitivity'] for m in train_metrics_history]
        val_sens = [m['sensitivity'] for m in val_metrics_history]
        plt.plot(train_sens, 'b-', label='Train Sensitivity', linewidth=2)
        plt.plot(val_sens, 'r-', label='Val Sensitivity', linewidth=2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Sensitivity', fontsize=12)
        plt.title('Sensitivity Curve', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=10)

        # 5. Specificity曲线
        plt.subplot(2, 3, 5)
        train_spec = [m['specificity'] for m in train_metrics_history]
        val_spec = [m['specificity'] for m in val_metrics_history]
        plt.plot(train_spec, 'b-', label='Train Specificity', linewidth=2)
        plt.plot(val_spec, 'r-', label='Val Specificity', linewidth=2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Specificity', fontsize=12)
        plt.title('Specificity Curve', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def train(self):
        """训练模型"""
        # 创建数据集和数据加载器
        dataset = BladderDataset(
            image_dir=self.train_img_dir,
            mask_dir=self.train_mask_dir,
            transform=self.train_transform,
        )
        
        # 计算训练集和验证集大小
        train_size = int(self.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        
        # 分割数据集
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.random_seed)  # 确保可重复性
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        print(f"\n数据加载器配置:")
        print(f"- 训练集大小: {len(train_dataset)}")
        print(f"- 验证集大小: {len(val_dataset)}")
        print(f"- 训练批次数: {len(train_loader)}")
        print(f"- 验证批次数: {len(val_loader)}")
        print(f"- 工作进程数: {self.num_workers}")
        print(f"- 设备: {self.device}")
        
        # 初始化模型和训练组件
        self.model = UNet(in_channels=self.in_channels, out_channels=self.out_channels).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            mode='max'
        )
        
        # 训练循环
        best_val_dice = 0
        train_losses = []
        val_losses = []
        train_metrics_history = []
        val_metrics_history = []
        
        for epoch in range(self.num_epochs):
            # 训练和验证
            train_loss, train_metrics = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate(val_loader)
            
            # 记录历史
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)
            
            # 打印信息
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("\nTraining Metrics:")
            for metric, value in train_metrics.items():
                print(f"{metric}: {value:.4f}")
            print("\nValidation Metrics:")
            for metric, value in val_metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # 保存模型
            if val_metrics['dice'] > best_val_dice:
                best_val_dice = val_metrics['dice']
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self._save_checkpoint(epoch, val_metrics)
            
            # 每个epoch后更新并保存训练曲线
            self._plot_training_curves(train_losses, val_losses, train_metrics_history, val_metrics_history)
            
            # 早停检查
            if self.early_stopping(val_metrics['dice']):
                print(f"\nEarly stopping triggered! No improvement in validation Dice score for {self.early_stopping_patience} epochs.")
                print(f"Best validation Dice score: {self.early_stopping.best_value:.4f}")
                break
            
            print("-"*50)
        
        # 训练总结
        print("\nTraining Summary:")
        print(f"Best Validation Dice Score: {best_val_dice:.4f}")
        print(f"Total Epochs Trained: {epoch+1}")
        if self.early_stopping.early_stop:
            print("Training stopped early due to no improvement in validation metrics.")
        else:
            print("Training completed for all epochs.")


# %%
# %%
def calculate_metrics(outputs, masks, threshold=0.5):
    # 对logits应用sigmoid函数
    outputs = torch.sigmoid(outputs)
    
    # 将预测结果二值化
    predictions = (outputs > threshold).float()
    
    # 计算TP, TN, FP, FN
    TP = torch.sum((predictions == 1) & (masks == 1)).float()
    TN = torch.sum((predictions == 0) & (masks == 0)).float()
    FP = torch.sum((predictions == 1) & (masks == 0)).float()
    FN = torch.sum((predictions == 0) & (masks == 1)).float()
    
    # 计算各项指标
    sensitivity = TP / (TP + FN + 1e-7)  # Recall
    specificity = TN / (TN + FP + 1e-7)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-7)
    iou = TP / (TP + FP + FN + 1e-7)
    
    return {
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
        'dice': dice.item(),
        'iou': iou.item()
    }


# %%
def visualize_predictions(model, val_loader, device, num_samples=5):
    """
    可视化模型预测结果
    Args:
        model: 训练好的模型
        val_loader: 验证数据加载器
        device: 设备（CPU/GPU）
        num_samples: 要可视化的样本数量
    """
    model.eval()
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            
            # 获取预测结果
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5
            
            # 转换为numpy数组
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            predictions = predictions.cpu().numpy()
            
            # 对每个批次中的图像进行可视化
            for j in range(images.shape[0]):
                if i * val_loader.batch_size + j >= num_samples:
                    break
                    
                plt.figure(figsize=(15, 5))
                
                # 反归一化图像
                img = images[j].transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                # 显示原始图像
                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title('原始图像')
                plt.axis('off')
                
                # 显示真实掩码
                plt.subplot(1, 3, 2)
                plt.imshow(masks[j].squeeze(), cmap='gray')
                plt.title('真实掩码')
                plt.axis('off')
                
                # 显示预测掩码
                plt.subplot(1, 3, 3)
                plt.imshow(predictions[j].squeeze(), cmap='gray')
                plt.title('预测掩码')
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()

def main():
    # 创建训练器实例并开始训练
    trainer = UNetTrainer()
    trainer.train()
    
    # 打印训练结果保存位置
    print("\n训练完成！")
    print(f"训练曲线已保存至: {os.path.join(trainer.save_dir, 'training_curves.png')}")
    print(f"最佳模型已保存至: {os.path.join(trainer.save_dir, 'unet_best.pth')}")
    
    # 加载最佳模型并进行预测可视化
    print("\n正在生成预测结果可视化...")
    best_model = UNet(in_channels=trainer.in_channels, out_channels=trainer.out_channels)
    best_model.load_state_dict(torch.load(os.path.join(trainer.save_dir, 'unet_best.pth'))['model_state_dict'])
    best_model = best_model.to(trainer.device)
    
    # 创建验证数据集和加载器用于可视化
    val_dataset = BladderDataset(
        image_dir=trainer.train_img_dir,
        mask_dir=trainer.train_mask_dir,
        transform=trainer.val_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=trainer.batch_size,
        shuffle=False,
        num_workers=trainer.num_workers
    )
    
    # 进行预测可视化
    print("\n显示预测结果...")
    visualize_predictions(best_model, val_loader, trainer.device, num_samples=5)

if __name__ == "__main__":
    main() 