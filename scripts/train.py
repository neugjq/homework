# scripts/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import wandb
from pathlib import Path
import sys
# 获取当前脚本的绝对路径
current_file = Path(__file__).resolve()
# 获取项目根目录（假设 `train.py` 在 `scripts/` 下，根目录是它的父目录的父目录）
project_root = current_file.parent.parent
# 将项目根目录添加到 Python 的模块搜索路径
sys.path.append(str(project_root))

from models.yolo_model import YOLOModel
from models.losses import YOLOLoss, reweight_loss
from utils.data_preprocessing import MultiObjectDataset, UnlabeledImageDataset, get_class_counts, BalancedSampler
from utils.augmentation import get_transforms
from utils.semi_supervised import generate_pseudo_labels, contrastive_learning, SupConLoss

# torch.backends.quantized.engine = 'qnnpack'  # 注释掉，避免Windows下报错
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = YOLOModel(
            num_classes=config['num_classes'],
            img_size=config['img_size'],
            anchors=config.get('anchors'),
            use_attention=config.get('use_attention', False),
            attention_type=config.get('attention_type', 'CBAM'),
            backbone_type=config.get('backbone_type', 'efficientnet_v2_s'),
            fpn_type=config.get('fpn_type', 'fpn')
        ).to(self.device)
        
        # 损失函数和优化器
        self.criterion = YOLOLoss(
            num_classes=config['num_classes'],
            anchors=self.model.anchors  # 这里传递 anchors
        )
        #self.criterion = YOLOLoss(config['num_classes'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs']
        )
        
        # 数据加载器
        self.setup_data_loaders()
        
    def setup_data_loaders(self):
        # 训练数据
        train_dataset = MultiObjectDataset(
            images_dir=self.config['train_images'],
            annotations_file=self.config['train_annotations'],
            img_size=self.config['img_size'],
            transform=get_transforms('train', self.config['img_size'])
        )
        # 均衡采样
        if self.config.get('use_balanced_sampler', False):
            class_counts = get_class_counts(train_dataset)
            sampler = BalancedSampler(train_dataset, class_counts, self.config['batch_size'])
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                sampler=sampler,
                num_workers=2,
                collate_fn=self.collate_fn
            )
        else:
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            collate_fn=self.collate_fn
        )
        
        # 验证数据
        val_dataset = MultiObjectDataset(
            images_dir=self.config['val_images'],
            annotations_file=self.config['val_annotations'],
            img_size=self.config['img_size'],
            transform=get_transforms('val', self.config['img_size'])
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            collate_fn=self.collate_fn
        )
        # 未标注数据（半监督）
        if 'unlabeled_images' in self.config:
            unlabeled_dataset = UnlabeledImageDataset(
                images_dir=self.config['unlabeled_images'],
                img_size=self.config['img_size'],
                transform=get_transforms('train', self.config['img_size'])
            )
            self.unlabeled_loader = DataLoader(
                unlabeled_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=2
            )
        else:
            self.unlabeled_loader = None

    # def collate_fn(self, batch):
    #     """
    #     batch: list of dict, 每个 dict = {
    #         'image': Tensor[C,H,W],
    #         'boxes': Tensor[N,4],
    #         'labels': Tensor[N]
    #     }
    #     """
    #     # 1) 拼图像
    #     images = torch.stack([item['image'] for item in batch], dim=0)  # [B,C,H,W]

    #     # 2) 配置参数
    #     img_size = 640
    #     anchors = [
    #         [[10,13], [16,30], [33,23]],      # 小目标
    #         [[30,61], [62,45], [59,119]],     # 中目标  
    #         [[116,90], [156,198], [373,326]]  # 大目标
    #     ]
    #     num_classes = 80
    #     device = images.device
        
    #     # 3) 构建原始目标格式
    #     raw_targets = {
    #         'boxes': [item['boxes'] for item in batch],
    #         'labels': [item['labels'] for item in batch]
    #     }
        
    #     # 4) 调用转换函数
    #     sample_targets = MultiObjectDataset.convert_targets_to_yolo_format(
    #         raw_targets, img_size, anchors, num_classes, device
    #     )
        
    #     return images, sample_targets[0]


    def collate_fn(self, batch):
        """自定义批处理函数"""
        images = torch.stack([item['image'] for item in batch])
        targets = {
            'boxes': [item['boxes'] for item in batch],
            'labels': [item['labels'] for item in batch]
        }
        return images, targets
    
    # def train_epoch(self):
    #     self.model.train()
    #     total_loss = 0
        
    #     pbar = tqdm(self.train_loader, desc='Training')
    #     for batch_idx, (images, targets) in enumerate(pbar):
    #         images = images.to(self.device)
            
    #         self.optimizer.zero_grad()
            
    #         # 前向传播
    #         predictions = self.model(images)
            
    #         # 计算损失（需要实现目标准备函数）
    #         loss_dict = self.criterion(predictions[0], targets)  # 简化：只使用一个尺度
    #         loss = loss_dict['total_loss']
            
    #         # 反向传播
    #         loss.backward()
    #         self.optimizer.step()
            
    #         total_loss += loss.item()
            
    #         pbar.set_postfix({
    #             'Loss': f'{loss.item():.4f}',
    #             'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
    #         })
        
    #     return total_loss / len(self.train_loader)
    # def train_epoch(self):
        # self.model.train()
        # total_loss = 0
        
        # pbar = tqdm(self.train_loader, desc='Training')
        # for batch_idx, (images, targets) in enumerate(pbar):
        #     images = images.to(self.device)
            
        #     # 将targets移到正确的设备上
        #     for i in range(len(targets['boxes'])):
        #         targets['boxes'][i] = targets['boxes'][i].to(self.device)
        #         targets['labels'][i] = targets['labels'][i].to(self.device)
            
        #     self.optimizer.zero_grad()
            
        #     # 前向传播
        #     predictions = self.model(images)
            
        #     # 转换目标格式为YOLO格式
        #     yolo_targets = convert_targets_to_yolo_format(
        #         targets, 
        #         640,  # img_size，需要根据你的实际图像尺寸调整
        #         self.criterion.anchors, 
        #         self.criterion.num_classes,
        #         self.device
        #     )
            
        #     # 计算损失
        #     loss, loss_dict = self.criterion(predictions, yolo_targets)
            
        #     # 反向传播
        #     loss.backward()
        #     self.optimizer.step()
            
        #     total_loss += loss.item()
            
        #     pbar.set_postfix({
        #         'Loss': f'{loss.item():.4f}',
        #         'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        #     })
        
        # return total_loss / len(self.train_loader)
    # 
    def train_epoch(self, use_pseudo_labels=False):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc='Training')
        supcon_loss_fn = SupConLoss().to(self.device) if use_pseudo_labels else None
        teacher_model = None  # 可选teacher-student结构
        pseudo_loss_weight = self.config.get('pseudo_loss_weight', 0.5)
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            for i in range(len(targets['boxes'])):
                targets['boxes'][i] = targets['boxes'][i].to(self.device)
                targets['labels'][i] = targets['labels'][i].to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(images)
            yolo_targets = MultiObjectDataset.convert_targets_to_yolo_format(
                targets, 
                self.config['img_size'],
                self.criterion.anchors,
                self.criterion.num_classes,
                self.device
            )
            for scale_idx in range(len(yolo_targets)):
                for key in yolo_targets[scale_idx]:
                    yolo_targets[scale_idx][key] = yolo_targets[scale_idx][key].to(self.device)
            loss, loss_dict = self.criterion(predictions, yolo_targets)
            if self.config.get('use_loss_reweight', False):
                class_weights = torch.ones(self.config['num_classes']).to(self.device)
                # loss = reweight_loss(loss, targets['labels'], class_weights)
                pass
            # ==== 半监督伪标签训练 ====
            if use_pseudo_labels and self.unlabeled_loader is not None:
                pseudo_data = generate_pseudo_labels(self.model, self.unlabeled_loader, self.device, threshold=0.7, use_teacher=(teacher_model is not None), teacher_model=teacher_model)
                if len(pseudo_data) > 0:
                    pseudo_imgs = torch.stack([item[0] for item in pseudo_data[:images.size(0)]]).to(self.device)
                    pseudo_boxes = [item[1] for item in pseudo_data[:images.size(0)]]
                    pseudo_labels = [item[2] for item in pseudo_data[:images.size(0)]]
                    pseudo_scores = [item[3] for item in pseudo_data[:images.size(0)]]
                    mixed_imgs = torch.cat([images, pseudo_imgs], dim=0)
                    mixed_boxes = targets['boxes'] + pseudo_boxes
                    mixed_labels = targets['labels'] + pseudo_labels
                    mixed_targets = {'boxes': mixed_boxes, 'labels': mixed_labels}
                    mixed_predictions = self.model(mixed_imgs)
                    mixed_yolo_targets = MultiObjectDataset.convert_targets_to_yolo_format(
                        mixed_targets, 
                        self.config['img_size'],
                        self.criterion.anchors,
                        self.criterion.num_classes,
                        self.device
                    )
                    for scale_idx in range(len(mixed_yolo_targets)):
                        for key in mixed_yolo_targets[scale_idx]:
                            mixed_yolo_targets[scale_idx][key] = mixed_yolo_targets[scale_idx][key].to(self.device)
                    loss2, _ = self.criterion(mixed_predictions, mixed_yolo_targets)
                    # 置信度加权（简单实现：乘均值）
                    avg_score = torch.cat(pseudo_scores).mean() if len(pseudo_scores) > 0 else 1.0
                    loss += pseudo_loss_weight * avg_score * loss2
                    # 对比学习损失
                    if supcon_loss_fn is not None:
                        features = self.model.get_features(mixed_imgs)
                        labels = torch.cat(mixed_labels, dim=0)
                        contrastive_loss = supcon_loss_fn(features, labels)
                        loss += 0.1 * contrastive_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                
                predictions = self.model(images)
                loss_dict = self.criterion(predictions[0], targets)
                total_loss += loss_dict['total_loss'].item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, use_pseudo_labels=False):
        best_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # 训练
            train_loss = self.train_epoch(use_pseudo_labels=use_pseudo_labels)
            
            # 验证
            val_loss = self.validate()
            
            # 学习率调度
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, 'best_model.pth')

def main():
    # 加载配置
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化训练器
    trainer = Trainer(config)
    
    # 是否启用半监督伪标签训练
    use_pseudo_labels = True if 'unlabeled_images' in config else False
    # 开始训练
    trainer.train(use_pseudo_labels=use_pseudo_labels)

if __name__ == "__main__":
    main()
