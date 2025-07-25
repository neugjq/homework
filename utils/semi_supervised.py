import torch
import torch.nn.functional as F
from torch import nn
import torchvision.ops as ops

def generate_pseudo_labels(model, unlabeled_loader, device, threshold=0.7, use_teacher=False, teacher_model=None):
    """
    利用模型对未标注数据生成伪标签，支持NMS和teacher-student结构。
    返回: [(image, pseudo_boxes, pseudo_labels, pseudo_scores), ...]
    """
    model.eval()
    pseudo_labels = []
    net = teacher_model if (use_teacher and teacher_model is not None) else model
    with torch.no_grad():
        for batch in unlabeled_loader:
            images = batch['image'].to(device)
            outputs = net(images)
            preds = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            for i in range(images.size(0)):
                pred = preds[i]
                boxes = pred[..., :4]
                scores = torch.sigmoid(pred[..., 4])
                labels = torch.argmax(pred[..., 5:], dim=-1)
                mask = scores > threshold
                boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
                if boxes.size(0) > 0:
                    keep = ops.nms(boxes, scores, iou_threshold=0.5)
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                pseudo_labels.append((images[i].cpu(), boxes.cpu(), labels.cpu(), scores.cpu()))
    return pseudo_labels

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - torch.eye(labels.shape[0], device=device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        return loss 