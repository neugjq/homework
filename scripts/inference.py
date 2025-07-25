# scripts/inference.py
import torch
import cv2
import numpy as np
from models.yolo_model import YOLOModel
import yaml

def load_model(config_path, model_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = YOLOModel(
        num_classes=config['num_classes'],
        img_size=config['img_size']
    ).to(device)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def detect_objects(model, image_path, config):
    # 加载和预处理图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 预处理
    input_image = cv2.resize(image_rgb, (config['img_size'], config['img_size']))
    input_image = input_image.astype(np.float32) / 255.0
    input_image = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0)
    
    device = next(model.parameters()).device
    input_image = input_image.to(device)
    
    # 推理
    with torch.no_grad():
        predictions = model(input_image)
    
    # 后处理
    detections = post_process_detections(predictions, config)
    
    # 在原图上绘制检测结果
    result_image = draw_detections(image, detections)
    
    return result_image, detections

def post_process_detections(predictions, config):
    """后处理检测结果"""
    # 简化版本，实际需要实现完整的NMS和阈值过滤
    detections = []
    return detections

def draw_detections(image, detections):
    """在图像上绘制检测框"""
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        label = detection['label']
        score = detection['score']
        
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'{label}: {score:.2f}', 
                   (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

if __name__ == "__main__":
    model, config = load_model('config/config.yaml', 'best_model.pth')
    result_image, detections = detect_objects(model, 'test_image.jpg', config)
    cv2.imwrite('result.jpg', result_image)
