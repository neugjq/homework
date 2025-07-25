# utils/data_preprocessing.py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import os
from tqdm import tqdm
import logging
from utils.augmentation import get_transforms, mosaic_augmentation, mixup_augmentation, cutmix_augmentation
import random
from collections import Counter
from torch.utils.data import Sampler

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiObjectDataset(Dataset):
    def __init__(self, images_dir, annotations_file, img_size=640, transform=None, enable_cache=True, mosaic=False, mixup=False, cutmix=False):
        """
        优化后的COCO格式数据集加载器
        
        参数:
            images_dir: 图像目录路径
            annotations_file: COCO标注文件路径
            img_size: 图像缩放尺寸
            transform: 数据增强变换
            enable_cache: 是否启用内存缓存
            mosaic: 是否启用Mosaic增强
            mixup: 是否启用MixUp增强
            cutmix: 是否启用CutMix增强
        """
        self.images_dir = images_dir
        self.img_size = img_size
        self.transform = transform
        self.enable_cache = enable_cache
        self.cache = {}
        self.mosaic = mosaic
        self.mixup = mixup
        self.cutmix = cutmix
        
        # 验证路径存在
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"图像目录不存在: {images_dir}")
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"标注文件不存在: {annotations_file}")

        # 加载并验证标注
        logger.info(f"加载标注文件: {annotations_file}")
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self._validate_images(self.coco_data['images'])
        self.annotations = self.coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # 构建图像ID映射
        self.img_to_anns = self._build_image_annotations_mapping()
        logger.info(f"数据集初始化完成，共 {len(self.images)} 张图像")

    def _validate_images(self, images):
        """验证图像文件是否存在"""
        valid_images = []
        for img in tqdm(images, desc="验证图像文件"):
            img_path = os.path.join(self.images_dir, img['file_name'])
            if os.path.exists(img_path):
                valid_images.append(img)
            else:
                logger.warning(f"图像文件缺失: {img_path}")
        return valid_images

    def _build_image_annotations_mapping(self):
        """构建图像到标注的映射"""
        img_to_anns = {}
        for ann in tqdm(self.annotations, desc="构建标注映射"):
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        return img_to_anns

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 内存缓存检查
        if self.enable_cache and idx in self.cache:
            return self.cache[idx]

        try:
            img_info = self.images[idx]
            img_path = os.path.join(self.images_dir, img_info['file_name'])
            
            # 加载图像 (优化IO)
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 获取标注
            img_id = img_info['id']
            anns = self.img_to_anns.get(img_id, [])
            boxes, labels = self._process_annotations(anns)
            # 组装为[x1, y1, x2, y2, label]格式
            bboxes = [list(map(float, box)) + [int(label)] for box, label in zip(boxes, labels)]
            img_size = self.img_size
            # ==== 创新增强方法实际调用 ====
            if self.mosaic:
                # 随机采样3张不同图片
                indices = [idx]
                while len(indices) < 4:
                    rand_idx = random.randint(0, len(self.images) - 1)
                    if rand_idx not in indices:
                        indices.append(rand_idx)
                imgs = []
                bboxs = []
                for i in indices:
                    info = self.images[i]
                    path = os.path.join(self.images_dir, info['file_name'])
                    img = cv2.imread(path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    anns_i = self.img_to_anns.get(info['id'], [])
                    boxes_i, labels_i = self._process_annotations(anns_i)
                    bbox_i = [list(map(float, box)) + [int(label)] for box, label in zip(boxes_i, labels_i)]
                    imgs.append(img)
                    bboxs.append(bbox_i)
                image, bboxes = mosaic_augmentation(imgs, bboxs, img_size=img_size)
                boxes = [b[:4] for b in bboxes]
                labels = [b[4] for b in bboxes]
            elif self.mixup:
                # 随机采样1张不同图片
                rand_idx = idx
                while rand_idx == idx:
                    rand_idx = random.randint(0, len(self.images) - 1)
                info2 = self.images[rand_idx]
                path2 = os.path.join(self.images_dir, info2['file_name'])
                img2 = cv2.imread(path2)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                anns2 = self.img_to_anns.get(info2['id'], [])
                boxes2, labels2 = self._process_annotations(anns2)
                bboxes2 = [list(map(float, box)) + [int(label)] for box, label in zip(boxes2, labels2)]
                image, bboxes = mixup_augmentation(image, bboxes, img2, bboxes2, alpha=0.5)
                boxes = [b[:4] for b in bboxes]
                labels = [b[4] for b in bboxes]
            elif self.cutmix:
                # 随机采样1张不同图片
                rand_idx = idx
                while rand_idx == idx:
                    rand_idx = random.randint(0, len(self.images) - 1)
                info2 = self.images[rand_idx]
                path2 = os.path.join(self.images_dir, info2['file_name'])
                img2 = cv2.imread(path2)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                anns2 = self.img_to_anns.get(info2['id'], [])
                boxes2, labels2 = self._process_annotations(anns2)
                bboxes2 = [list(map(float, box)) + [int(label)] for box, label in zip(boxes2, labels2)]
                image, bboxes = cutmix_augmentation(image, bboxes, img2, bboxes2, img_size=img_size)
                boxes = [b[:4] for b in bboxes]
                labels = [b[4] for b in bboxes]
            # ======================
            # 应用数据增强
            if self.transform:
                image, boxes, labels = self._apply_transforms(image, boxes, labels)
            
            # # 转换格式
            # if len(boxes) > 0:
            #     boxes = self.convert_targets_to_yolo_format(boxes, image.shape[:2])
            
            result = {
                'image': image,
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.long),
                'image_id': torch.tensor([img_id])
            }
            
            # 缓存结果
            if self.enable_cache:
                self.cache[idx] = result
                
            return result
            
        except Exception as e:
            logger.error(f"处理样本 {idx} 失败: {str(e)}")
            # 返回空样本或跳过
            return self._get_empty_sample()
    
    coco_id_to_contiguous = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
    46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
    67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
    80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}
    def _process_annotations(self, anns):
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            # COCO类别ID映射 - 使用 self.coco_id_to_contiguous
            if ann['category_id'] in self.coco_id_to_contiguous:
                labels.append(self.coco_id_to_contiguous[ann['category_id']])  # 添加 self.
                boxes.append([x, y, x + w, y + h])
            # else: 跳过不存在的类别
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    # def _process_annotations(self, anns):
    #     """处理标注信息"""
    #     boxes = []
    #     labels = []
    #     for ann in anns:
    #         x, y, w, h = ann['bbox']
    #         boxes.append([x, y, x + w, y + h])  # 转换xywh为xyxy
    #         labels.append(ann['category_id'])
    #     return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _apply_transforms(self, image, boxes, labels):
        """应用数据增强"""
        return self.transform(
            image=image,
            bboxes=boxes,
            class_labels=labels
        )

    def _get_empty_sample(self):
        """返回空样本用于错误处理"""
        return {
            'image': torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32),
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.long),
            'image_id': torch.tensor([-1])
        }

    @staticmethod
    @staticmethod
    def convert_targets_to_yolo_format(raw_targets, img_size, anchors, num_classes, device):
        """
        将原始targets转换为YOLO训练格式，支持多尺度
        raw_targets: {'boxes': [list_of_tensors], 'labels': [list_of_tensors]}
        """
        batch_size = len(raw_targets['boxes'])
        strides = [8, 16, 32]  # 对应80x80, 40x40, 20x20三个尺度
        yolo_targets = []
        
        # 对每个尺度生成targets
        for scale_idx, (anchor_set, stride) in enumerate(zip(anchors, strides)):
            grid_size = img_size // stride
            num_anchors = len(anchor_set)
            
            # 初始化目标张量 - 注意维度顺序调整为 [B, A, H, W, ...]
            target_boxes = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 4).to(device)
            target_conf = torch.zeros(batch_size, num_anchors, grid_size, grid_size).to(device)
            target_classes = torch.zeros(batch_size, num_anchors, grid_size, grid_size).to(device)
            target_mask = torch.zeros(batch_size, num_anchors, grid_size, grid_size).to(device)
            
            for batch_idx in range(batch_size):
                boxes = raw_targets['boxes'][batch_idx]  # [N, 4] tensor
                labels = raw_targets['labels'][batch_idx]  # [N] tensor
                
                if len(boxes) == 0:  # 如果没有目标框，跳过
                    continue
                    
                for box, label in zip(boxes, labels):
                    # boxes 格式是 [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box
                    x_center = (x1 + x2) / 2.0
                    y_center = (y1 + y2) / 2.0
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 转换到网格坐标
                    grid_x = x_center / stride
                    grid_y = y_center / stride
                    
                    grid_x_idx = int(grid_x.item()) if isinstance(grid_x, torch.Tensor) else int(grid_x)
                    grid_y_idx = int(grid_y.item()) if isinstance(grid_y, torch.Tensor) else int(grid_y)
                    
                    # 检查边界
                    if 0 <= grid_x_idx < grid_size and 0 <= grid_y_idx < grid_size:
                        # 计算与所有anchor的IoU，选择最佳anchor
                        best_anchor_idx = 0
                        best_iou = 0
                        
                        for anchor_idx, (anchor_w, anchor_h) in enumerate(anchor_set):
                            # 简化的IoU计算（只考虑宽高比）
                            iou = min(width/anchor_w, anchor_w/width) * min(height/anchor_h, anchor_h/height)
                            if iou > best_iou:
                                best_iou = iou
                                best_anchor_idx = anchor_idx
                        
                        # 计算偏移
                        x_offset = grid_x - grid_x_idx
                        y_offset = grid_y - grid_y_idx
                        
                        anchor_w, anchor_h = anchor_set[best_anchor_idx]
                        
                        # YOLO格式编码
                        target_boxes[batch_idx, best_anchor_idx, grid_y_idx, grid_x_idx, 0] = x_offset
                        target_boxes[batch_idx, best_anchor_idx, grid_y_idx, grid_x_idx, 1] = y_offset
                        target_boxes[batch_idx, best_anchor_idx, grid_y_idx, grid_x_idx, 2] = torch.log(width / (anchor_w * stride) + 1e-8)
                        target_boxes[batch_idx, best_anchor_idx, grid_y_idx, grid_x_idx, 3] = torch.log(height / (anchor_h * stride) + 1e-8)
                        
                        target_conf[batch_idx, best_anchor_idx, grid_y_idx, grid_x_idx] = 1.0
                        target_classes[batch_idx, best_anchor_idx, grid_y_idx, grid_x_idx] = label.item()
                        target_mask[batch_idx, best_anchor_idx, grid_y_idx, grid_x_idx] = 1.0
            
            # 添加到yolo_targets列表
            yolo_targets.append({
                'boxes': target_boxes,
                'conf': target_conf,
                'classes': target_classes,
                'mask': target_mask
            })
        
        return yolo_targets  # 返回3个尺度的目标列表

    # def convert_targets_to_yolo_format(raw_targets, img_size, anchors, num_classes, device):
    #     """
    #     将原始targets转换为YOLO训练格式
    #     raw_targets: {'boxes': [list_of_tensors], 'labels': [list_of_tensors]}
    #     """
    #     yolo_targets = {}
        
    #     # 只处理第一个尺度（因为你只用 predictions[0]）
    #     scale_idx = 0
    #     anchor_set = anchors[scale_idx] if isinstance(anchors[0], list) else anchors
        
    #     # 根据你的模型结构调整stride
    #     stride = 8  # 或者根据你的特征图尺寸计算
    #     grid_size = img_size // stride
    #     batch_size = len(raw_targets['boxes'])
    #     num_anchors = len(anchor_set)
        
    #     # 初始化目标张量
    #     target_boxes = torch.zeros(batch_size, grid_size, grid_size, num_anchors, 4).to(device)
    #     target_conf = torch.zeros(batch_size, grid_size, grid_size, num_anchors).to(device)
    #     target_classes = torch.zeros(batch_size, grid_size, grid_size, num_anchors).to(device)
    #     target_mask = torch.zeros(batch_size, grid_size, grid_size, num_anchors).to(device)
        
    #     for batch_idx in range(batch_size):
    #         boxes = raw_targets['boxes'][batch_idx]  # [N, 4] tensor
    #         labels = raw_targets['labels'][batch_idx]  # [N] tensor
            
    #         if len(boxes) == 0:  # 如果没有目标框，跳过（但初始化的 target 结构依然存在）
    #             continue
                
    #         for box, label in zip(boxes, labels):
    #             # boxes 格式是 [x1, y1, x2, y2]
    #             x1, y1, x2, y2 = box
    #             x_center = (x1 + x2) / 2.0
    #             y_center = (y1 + y2) / 2.0
    #             width = x2 - x1
    #             height = y2 - y1
                
    #             # 转换到网格坐标
    #             grid_x = x_center / stride
    #             grid_y = y_center / stride
                
    #             grid_x_idx = int(grid_x.item()) if isinstance(grid_x, torch.Tensor) else int(grid_x)
    #             grid_y_idx = int(grid_y.item()) if isinstance(grid_y, torch.Tensor) else int(grid_y)
                
    #             # 检查边界
    #             if 0 <= grid_x_idx < grid_size and 0 <= grid_y_idx < grid_size:
    #                 # 计算偏移
    #                 x_offset = grid_x - grid_x_idx
    #                 y_offset = grid_y - grid_y_idx
                    
    #                 # 简化：使用第一个anchor
    #                 best_anchor_idx = 0
                    
    #                 # 确保anchor_set格式正确
    #                 if isinstance(anchor_set[0], (list, tuple)):
    #                     anchor_w, anchor_h = anchor_set[best_anchor_idx]
    #                 else:
    #                     anchor_w, anchor_h = anchor_set[best_anchor_idx], anchor_set[best_anchor_idx]
                    
    #                 # YOLO格式编码
    #                 target_boxes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx, 0] = x_offset
    #                 target_boxes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx, 1] = y_offset
    #                 target_boxes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx, 2] = torch.log(width / (anchor_w * stride) + 1e-8)
    #                 target_boxes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx, 3] = torch.log(height / (anchor_h * stride) + 1e-8)
                    
    #                 target_conf[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx] = 1.0
    #                 target_classes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx] = label.item()
    #                 target_mask[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx] = 1.0
        
    #     # 保证返回值是 list，每个元素是一个 dict，且包含 boxes/conf/classes/mask
    #     return [{
    #         'boxes': target_boxes,
    #         'conf': target_conf,
    #         'classes': target_classes,
    #         'mask': target_mask
    #     }]
    # def convert_targets_to_yolo_format(raw_targets, img_size, anchors, num_classes):
    #     """
    #     将原始targets转换为YOLO训练格式
    #     raw_targets: {'boxes': [list_of_tensors], 'labels': [list_of_tensors]}
    #     boxes格式: [x1, y1, x2, y2] 绝对坐标
    #     """
    #     yolo_targets = {}
        
    #     # YOLO的三个尺度对应的grid size和stride
    #     scales = [
    #         {'grid_size': img_size // 8, 'stride': 8},    # 大特征图，检测小物体
    #         {'grid_size': img_size // 16, 'stride': 16},  # 中特征图，检测中等物体  
    #         {'grid_size': img_size // 32, 'stride': 32},  # 小特征图，检测大物体
    #     ]
        
    #     for scale_idx, (anchor_set, scale_info) in enumerate(zip(anchors, scales)):
    #         grid_size = scale_info['grid_size']
    #         stride = scale_info['stride']
    #         batch_size = len(raw_targets['boxes'])
    #         num_anchors = len(anchor_set)
            
    #         # 初始化该尺度的目标张量
    #         target_boxes = torch.zeros(batch_size, grid_size, grid_size, num_anchors, 4)
    #         target_conf = torch.zeros(batch_size, grid_size, grid_size, num_anchors)
    #         target_classes = torch.zeros(batch_size, grid_size, grid_size, num_anchors)
    #         target_mask = torch.zeros(batch_size, grid_size, grid_size, num_anchors)
            
    #         # 为每个批次样本分配目标
    #         for batch_idx in range(batch_size):
    #             boxes = raw_targets['boxes'][batch_idx]  # [N, 4] tensor
    #             labels = raw_targets['labels'][batch_idx]  # [N] tensor
                
    #             for box, label in zip(boxes, labels):
        #             # 将xyxy格式转换为中心点和宽高
        #             x1, y1, x2, y2 = box
        #             x_center = (x1 + x2) / 2.0
        #             y_center = (y1 + y2) / 2.0
        #             width = x2 - x1
        #             height = y2 - y1
                    
        #             # 转换到当前尺度的网格坐标
        #             grid_x = x_center / stride
        #             grid_y = y_center / stride
                    
        #             # 找到对应的网格位置
        #             grid_x_idx = int(grid_x)
        #             grid_y_idx = int(grid_y)
                    
        #             # 检查边界
        #             if 0 <= grid_x_idx < grid_size and 0 <= grid_y_idx < grid_size:
        #                 # 计算相对于网格的偏移
        #                 x_offset = grid_x - grid_x_idx
        #                 y_offset = grid_y - grid_y_idx
                        
        #                 # 选择最佳匹配的anchor
        #                 best_anchor_idx = 0
        #                 best_iou = 0
                        
        #                 # 计算与每个anchor的IoU来选择最佳anchor
        #                 for anchor_idx, (anchor_w, anchor_h) in enumerate(anchor_set):
        #                     # 简化的IoU计算（基于宽高比）
        #                     anchor_w_scaled = anchor_w * stride
        #                     anchor_h_scaled = anchor_h * stride
                            
        #                     # 计算IoU
        #                     inter_w = min(width, anchor_w_scaled)
        #                     inter_h = min(height, anchor_h_scaled)
        #                     inter_area = inter_w * inter_h
                            
        #                     union_area = width * height + anchor_w_scaled * anchor_h_scaled - inter_area
        #                     iou = inter_area / (union_area + 1e-8)
                            
        #                     if iou > best_iou:
        #                         best_iou = iou
        #                         best_anchor_idx = anchor_idx
                        
        #                 # 只有IoU足够大才分配目标（避免低质量匹配）
        #                 if best_iou > 0.1:  # IoU阈值可以调整
        #                     # 设置目标值 - YOLO格式的边界框编码
        #                     target_boxes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx, 0] = x_offset
        #                     target_boxes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx, 1] = y_offset
        #                     target_boxes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx, 2] = torch.log(width / (anchor_set[best_anchor_idx][0] * stride) + 1e-8)
        #                     target_boxes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx, 3] = torch.log(height / (anchor_set[best_anchor_idx][1] * stride) + 1e-8)
                            
        #                     target_conf[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx] = 1.0
        #                     target_classes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx] = label.item()
        #                     target_mask[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx] = 1.0
            
        #     yolo_targets[scale_idx] = {
        #         'boxes': target_boxes,
        #         'conf': target_conf,  
        #         'classes': target_classes,
        #         'mask': target_mask
        #     }
        
        # return yolo_targets
    # def convert_targets_to_yolo_format(raw_targets, img_size, anchors, num_classes, device):
    #         yolo_targets = {}
            
    #         # 只处理第一个尺度（因为你只用 predictions[0]）
    #         scale_idx = 0
    #         anchor_set = anchors[scale_idx] if isinstance(anchors[0], list) else anchors
            
    #         # 根据你的模型结构调整stride
    #         stride = 8  # 或者根据你的特征图尺寸计算
    #         grid_size = img_size // stride
    #         batch_size = len(raw_targets['boxes'])
    #         num_anchors = len(anchor_set)
            
    #         # 初始化目标张量
    #         target_boxes = torch.zeros(batch_size, grid_size, grid_size, num_anchors, 4).to(device)
    #         target_conf = torch.zeros(batch_size, grid_size, grid_size, num_anchors).to(device)
    #         target_classes = torch.zeros(batch_size, grid_size, grid_size, num_anchors).to(device)
    #         target_mask = torch.zeros(batch_size, grid_size, grid_size, num_anchors).to(device)
            
    #         for batch_idx in range(batch_size):
    #             boxes = raw_targets['boxes'][batch_idx]  # [N, 4] tensor
    #             labels = raw_targets['labels'][batch_idx]  # [N] tensor
                
    #             if len(boxes) == 0:  # 如果没有目标框，跳过
    #                 continue
                    
    #             for box, label in zip(boxes, labels):
    #                 # boxes 格式是 [x1, y1, x2, y2]
    #                 x1, y1, x2, y2 = box
    #                 x_center = (x1 + x2) / 2.0
    #                 y_center = (y1 + y2) / 2.0
    #                 width = x2 - x1
    #                 height = y2 - y1
                    
    #                 # 转换到网格坐标
    #                 grid_x = x_center / stride
    #                 grid_y = y_center / stride
                    
    #                 grid_x_idx = int(grid_x.item()) if isinstance(grid_x, torch.Tensor) else int(grid_x)
    #                 grid_y_idx = int(grid_y.item()) if isinstance(grid_y, torch.Tensor) else int(grid_y)
                    
    #                 # 检查边界
    #                 if 0 <= grid_x_idx < grid_size and 0 <= grid_y_idx < grid_size:
    #                     # 计算偏移
    #                     x_offset = grid_x - grid_x_idx
    #                     y_offset = grid_y - grid_y_idx
                        
    #                     # 简化：使用第一个anchor
    #                     best_anchor_idx = 0
                        
    #                     # 确保anchor_set格式正确
    #                     if isinstance(anchor_set[0], (list, tuple)):
    #                         anchor_w, anchor_h = anchor_set[best_anchor_idx]
    #                     else:
    #                         anchor_w, anchor_h = anchor_set[best_anchor_idx], anchor_set[best_anchor_idx]
                        
    #                     # YOLO格式编码
    #                     target_boxes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx, 0] = x_offset
    #                     target_boxes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx, 1] = y_offset
    #                     target_boxes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx, 2] = torch.log(width / (anchor_w * stride) + 1e-8)
    #                     target_boxes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx, 3] = torch.log(height / (anchor_h * stride) + 1e-8)
                        
    #                     target_conf[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx] = 1.0
    #                     target_classes[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx] = label.item()
    #                     target_mask[batch_idx, grid_y_idx, grid_x_idx, best_anchor_idx] = 1.0
            
    #         yolo_targets[scale_idx] = {
    #             'boxes': target_boxes,
    #             'conf': target_conf,
    #             'classes': target_classes,
    #             'mask': target_mask
    #         }
            
    #         return yolo_targets


class UnlabeledImageDataset(Dataset):
    def __init__(self, images_dir, img_size=640, transform=None):
        self.images_dir = images_dir
        self.img_size = img_size
        self.transform = transform
        self.images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return {'image': image, 'file_name': self.images[idx]}


def get_transforms(phase='train', img_size=640):
    """优化的数据增强管道"""
    if phase == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.CLAHE(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def get_class_counts(dataset):
    """
    统计数据集中每个类别的样本数。
    返回: dict {class_id: count}
    """
    counts = Counter()
    for i in range(len(dataset)):
        item = dataset[i]
        labels = item['labels'] if 'labels' in item else []
        for label in labels:
            counts[int(label)] += 1
    return dict(counts)

from torch.utils.data import Sampler
class BalancedSampler(Sampler):
    """
    类别均衡采样器：保证每个batch中各类别尽量均衡。
    """
    def __init__(self, dataset, class_counts, batch_size):
        self.dataset = dataset
        self.class_counts = class_counts
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        # TODO: 实现采样逻辑
    def __iter__(self):
        # TODO: 返回采样索引序列
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)


def test_dataset():
    """测试数据集"""
    dataset = MultiObjectDataset(
        images_dir="data/coco/train2017",
        annotations_file="data/coco/annotations/instances_train2017.json",
        img_size=640,
        transform=get_transforms('train')
    )
    
    # 测试第一个样本
    sample = dataset[0]
    print("样本结构:", {k: v.shape for k, v in sample.items() if isinstance(v, (torch.Tensor, np.ndarray))})
    
    # 测试数据加载速度
    from time import time
    start = time()
    for i in range(min(100, len(dataset))):
        _ = dataset[i]
    print(f"加载100个样本耗时: {time() - start:.2f}秒")


if __name__ == '__main__':
    test_dataset()