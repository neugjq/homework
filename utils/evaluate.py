# scripts/evaluate.py
import torch
from torch.utils.data import DataLoader
import yaml
from models.yolo_model import YOLOModel
from utils.data_preprocessing import MultiObjectDataset
from utils.augmentation import get_transforms
from utils.metrics import calculate_map

def evaluate_model(config_path, model_path):
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = YOLOModel(
        num_classes=config['num_classes'],
        img_size=config['img_size']
    ).to(device)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载测试数据
    test_dataset = MultiObjectDataset(
        images_dir=config['test_images'],
        annotations_file=config['test_annotations'],
        img_size=config['img_size'],
        transform=get_transforms('test', config['img_size'])
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    
    # 评估
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            
            predictions = model(images)
            # 后处理预测结果（NMS等）
            processed_preds = post_process_predictions(predictions)
            
            all_predictions.extend(processed_preds)
            all_targets.extend(targets)
    
    # 计算指标
    map_score = calculate_map(all_predictions, all_targets, config['num_classes'])
    
    print(f"mAP@0.5: {map_score:.4f}")
    
    return map_score

def post_process_predictions(predictions):
    """后处理预测结果，包括NMS"""
    # 简化版本，实际需要实现完整的后处理
    processed = []
    for pred in predictions:
        # 应用置信度阈值和NMS
        # 这里需要实现完整的后处理逻辑
        processed.append({
            'boxes': [],
            'labels': [],
            'scores': []
        })
    return processed

if __name__ == "__main__":
    evaluate_model('config/config.yaml', 'best_model.pth')
