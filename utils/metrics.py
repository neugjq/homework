# utils/metrics.py
import torch
import numpy as np
from sklearn.metrics import average_precision_score

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def calculate_map(predictions, targets, num_classes, iou_threshold=0.5):
    """计算mAP指标"""
    ap_per_class = []
    
    for class_id in range(num_classes):
        # 提取当前类别的预测和真实值
        class_predictions = []
        class_targets = []
        
        for pred, target in zip(predictions, targets):
            pred_mask = pred['labels'] == class_id
            target_mask = target['labels'] == class_id
            
            if pred_mask.sum() > 0:
                class_predictions.append({
                    'boxes': pred['boxes'][pred_mask],
                    'scores': pred['scores'][pred_mask]
                })
            else:
                class_predictions.append({'boxes': [], 'scores': []})
            
            if target_mask.sum() > 0:
                class_targets.append(target['boxes'][target_mask])
            else:
                class_targets.append([])
        
        # 计算AP
        ap = calculate_ap(class_predictions, class_targets, iou_threshold)
        ap_per_class.append(ap)
    
    return np.mean(ap_per_class)

def calculate_ap(predictions, targets, iou_threshold):
    """计算单个类别的AP"""
    # 收集所有预测和分数
    all_predictions = []
    all_scores = []
    
    for pred in predictions:
        if len(pred['boxes']) > 0:
            all_predictions.extend(pred['boxes'])
            all_scores.extend(pred['scores'])
    
    if len(all_predictions) == 0:
        return 0.0
    
    # 按分数排序
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_predictions = [all_predictions[i] for i in sorted_indices]
    
    # 计算精确率和召回率
    tp = 0
    fp = 0
    total_targets = sum(len(target) for target in targets)
    
    precisions = []
    recalls = []
    
    for pred_box in sorted_predictions:
        # 检查是否与任何真实框匹配
        matched = False
        for target_boxes in targets:
            for target_box in target_boxes:
                if calculate_iou(pred_box, target_box) >= iou_threshold:
                    matched = True
                    break
            if matched:
                break
        
        if matched:
            tp += 1
        else:
            fp += 1
        
        precision = tp / (tp + fp)
        recall = tp / total_targets if total_targets > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # 计算AP
    if len(precisions) == 0:
        return 0.0
    
    return average_precision_score([1] * len(precisions), precisions)

