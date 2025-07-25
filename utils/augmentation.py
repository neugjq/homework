import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import numpy as np

def get_transforms(phase='train', img_size=640):
    """获取数据增强变换"""
    if phase == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.RandomCrop(width=int(img_size*0.9), height=int(img_size*0.9), p=0.3),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    else:  # validation/test
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def mosaic_augmentation(images, bboxes, img_size=640):
    """
    Mosaic增强：将4张图片拼接为1张，适用于目标检测。
    images: list of 4 images (numpy arrays)
    bboxes: list of 4 bbox lists (每张图片的标注，格式[x1, y1, x2, y2, label])
    返回: mosaic_image, mosaic_bboxes
    """
    h, w = img_size, img_size
    yc, xc = [int(random.uniform(h * 0.25, h * 0.75)) for _ in range(2)]
    mosaic_img = np.full((h, w, 3), 114, dtype=np.uint8)
    mosaic_bboxes = []
    for i, (img, boxes) in enumerate(zip(images, bboxes)):
        ih, iw = img.shape[:2]
        # 计算放置位置
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = 0, 0, xc, yc
            x1b, y1b, x2b, y2b = max(0, iw - xc), max(0, ih - yc), iw, ih
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, 0, w, yc
            x1b, y1b, x2b, y2b = 0, max(0, ih - yc), min(iw, w - xc), ih
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = 0, yc, xc, h
            x1b, y1b, x2b, y2b = max(0, iw - xc), 0, iw, min(ih, h - yc)
        else:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, w, h
            x1b, y1b, x2b, y2b = 0, 0, min(iw, w - xc), min(ih, h - yc)
        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw, padh = x1a - x1b, y1a - y1b
        # 调整目标框
        for box in boxes:
            x1, y1, x2, y2, label = box
            x1_new = x1 + padw
            y1_new = y1 + padh
            x2_new = x2 + padw
            y2_new = y2 + padh
            # 裁剪到图像范围
            x1_new = np.clip(x1_new, 0, w)
            y1_new = np.clip(y1_new, 0, h)
            x2_new = np.clip(x2_new, 0, w)
            y2_new = np.clip(y2_new, 0, h)
            if x2_new - x1_new > 1 and y2_new - y1_new > 1:
                mosaic_bboxes.append([x1_new, y1_new, x2_new, y2_new, label])
    return mosaic_img, mosaic_bboxes

def mixup_augmentation(image1, bboxes1, image2, bboxes2, alpha=0.5):
    """
    MixUp增强：将两张图片按比例混合，适用于目标检测。
    返回: mixup_image, mixup_bboxes
    """
    lam = np.random.beta(alpha, alpha)
    mixup_img = (image1.astype(np.float32) * lam + image2.astype(np.float32) * (1 - lam)).astype(np.uint8)
    # 合并目标框，标签不变，权重可用于损失加权
    mixup_bboxes = bboxes1 + bboxes2  # 可选：为每个box加上权重lam/(1-lam)
    return mixup_img, mixup_bboxes

def cutmix_augmentation(image1, bboxes1, image2, bboxes2, img_size=640):
    """
    CutMix增强：将一张图片的部分区域替换为另一张图片，适用于目标检测。
    返回: cutmix_image, cutmix_bboxes
    """
    h, w = img_size, img_size
    cutmix_img = image1.copy()
    # 随机生成矩形区域
    lam = np.random.beta(1.0, 1.0)
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)
    # 随机中心点
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    # 替换区域
    cutmix_img[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
    # 合并目标框
    cutmix_bboxes = []
    for box in bboxes1:
        x1b, y1b, x2b, y2b, label = box
        # 保留未被cutmix区域遮挡的目标
        if x2b < x1 or x1b > x2 or y2b < y1 or y1b > y2:
            cutmix_bboxes.append([x1b, y1b, x2b, y2b, label])
        else:
            # 可选：保留部分重叠目标，或直接丢弃
            pass
    for box in bboxes2:
        x1b, y1b, x2b, y2b, label = box
        # 只保留落在cutmix区域内的目标
        if x1b >= x1 and y1b >= y1 and x2b <= x2 and y2b <= y2:
            cutmix_bboxes.append([x1b, y1b, x2b, y2b, label])
    return cutmix_img, cutmix_bboxes