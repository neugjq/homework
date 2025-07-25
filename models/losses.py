import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self, num_classes, anchors, img_size=640, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors  # 预定义的 anchor boxes, e.g. [[(w,h),...], [...], [...]]
        self.num_anchors = len(anchors[0])
        self.img_size = img_size
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        # loss 函数都设为 none，后面手动加权、mask
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_loss  = nn.CrossEntropyLoss(reduction='none')
    # def forward(self, predictions, targets):
    #     """
    #     predictions: 模型输出列表 [small, medium, large]  （len = S）
    #     targets:     真实标签列表，要么按尺度（len=S），要么按样本（len=B）
    #     """
    #     # —— 1. 如果 targets 是按样本（len=B）传进来的，就把它转成按尺度（len=S）
    #     if len(targets) != len(predictions):
    #         # 假设每个 targets[i] 都是个 dict，dict 里的每个 value
    #         # 都是一个可以通过 [scale_index] 索引的张量或 list。
    #         B = len(targets)
    #         S = len(predictions)
    #         new_targets = []
    #         for s in range(S):
    #             ts = {}
    #             for k in targets[0].keys():
    #                 # 收集 batch 里每个样本的第 s 个尺度部分，然后 stack
    #                 vals = [t[k][s] for t in targets]                # list of tensors
    #                 ts[k] = torch.stack(vals, dim=0).to(predictions[s].device)
    #             new_targets.append(ts)
    #         targets = new_targets

    #     total_loss = 0
    #     loss_dict = {'coord_loss': 0, 'obj_loss': 0, 'noobj_loss': 0, 'cls_loss': 0}

    #     # —— 2. 主循环
    #     for i, pred in enumerate(predictions):
    #         target = {k: v.to(pred.device) for k, v in targets[i].items()}
    #         tgt_boxes = target['boxes']
    #         tgt_obj   = target['conf']
    #         tgt_cls   = target['classes']
    #         tgt_mask  = target['mask']

    #         # —— 2.1 调试信息
    #         print(f"\nScale {i}:")
    #         print(f"pred shape: {pred.shape}")
    #         print(f"tgt_boxes shape: {tgt_boxes.shape}")
    #         print(f"tgt_mask shape: {tgt_mask.shape}")

    #         # —— 2.2 整形预测
    #         batch_size, H, W = pred.shape[0], pred.shape[1], pred.shape[2]
    #         if pred.dim() == 4:
    #             pred = pred.view(batch_size, H, W, self.num_anchors, -1)
    #         pred_boxes = pred[..., :4]     # [B, H, W, A, 4]
    #         pred_obj   = pred[..., 4]      # [B, H, W, A]
    #         pred_cls   = pred[..., 5:]     # [B, H, W, A, C]

    #         # —— 2.3 对齐 tgt_* 的维度（不用改，这里沿用你原来的代码）
    #         if tgt_boxes.shape != pred_boxes.shape:
    #             # 情况1：tgt [B, A, H, W, 4] → [B, H, W, A, 4]
    #             if tgt_boxes.shape[1] == self.num_anchors:
    #                 tgt_boxes = tgt_boxes.permute(0, 2, 3, 1, 4)
    #                 tgt_obj   = tgt_obj.permute(0, 2, 3, 1)
    #                 tgt_cls   = tgt_cls.permute(0, 2, 3, 1)
    #                 tgt_mask  = tgt_mask.permute(0, 2, 3, 1)
    #             # 情况2：tgt [B, H, W, A, 4] but pred was [B, A, H, W, 4]
    #             elif tgt_boxes.shape[3] == self.num_anchors:
    #                 tgt_boxes = tgt_boxes.permute(0, 3, 1, 2, 4)
    #                 tgt_obj   = tgt_obj.permute(0, 3, 1, 2)
    #                 tgt_cls   = tgt_cls.permute(0, 3, 1, 2)
    #                 tgt_mask  = tgt_mask.permute(0, 3, 1, 2)

    #         # —— 2.4 数值检查
    #         assert torch.all(pred_boxes.isfinite()) and torch.all(tgt_boxes.isfinite())
    #         assert torch.all((tgt_mask == 0) | (tgt_mask == 1))

    #         print(f"After alignment:")
    #         print(f"pred_boxes: {pred_boxes.shape}, tgt_boxes: {tgt_boxes.shape}")
    #         print(f"pred_obj: {pred_obj.shape},   tgt_obj: {tgt_obj.shape}")
    #         print(f"tgt_mask: {tgt_mask.shape}, sum: {tgt_mask.sum().item()}")

    #         # one-hot → 索引
    #         if tgt_cls.dim() == 5 and tgt_cls.shape[-1] == self.num_classes:
    #             tgt_cls = tgt_cls.argmax(dim=-1)

    #         # —— 2.5 coord loss
    #         valid_count = tgt_mask.sum()
    #         if valid_count > 0:
    #             coord_loss = (self.mse_loss(pred_boxes, tgt_boxes) * tgt_mask.unsqueeze(-1)).sum() / valid_count
    #         else:
    #             coord_loss = torch.tensor(0., device=pred.device, requires_grad=True)

    #         # —— 2.6 obj / noobj loss
    #         obj_loss   = (self.bce_loss(pred_obj, tgt_obj) * tgt_mask).sum() / max(1, tgt_mask.sum())
    #         noobj_loss = (self.bce_loss(pred_obj, tgt_obj) * (1 - tgt_mask)).sum() / max(1, (1 - tgt_mask).sum())

    #         # —— 2.7 cls loss
    #         valid_pos = tgt_mask.bool()
    #         if valid_pos.sum() > 0:
    #             pred_cls_valid = pred_cls[valid_pos]            # [N_pos, C]
    #             tgt_cls_valid  = tgt_cls[valid_pos].long()      # [N_pos]
    #             print(">>> cls label range:", tgt_cls_valid.min().item(), "~", tgt_cls_valid.max().item())
    #             cls_loss = self.ce_loss(pred_cls_valid, tgt_cls_valid).mean()
    #         else:
    #             print(">>> 当前 batch 无有效目标类别")
    #             cls_loss = torch.tensor(0., device=pred.device, requires_grad=True)

    #         # —— 2.8 加权求和 & 记录
    #         loss = (self.lambda_coord * coord_loss + obj_loss
    #                 + self.lambda_noobj * noobj_loss + cls_loss)
    #         total_loss += loss

    #         with torch.no_grad():
    #             loss_dict['coord_loss'] += coord_loss.item()
    #             loss_dict['obj_loss']   += obj_loss.item()
    #             loss_dict['noobj_loss'] += noobj_loss.item()
    #             loss_dict['cls_loss']   += cls_loss.item()

    #     loss_dict['total_loss'] = total_loss.item()
    #     return total_loss, loss_dict

    def forward(self, predictions, targets):
        """
        predictions: 模型输出列表 [small, medium, large]
        targets: 真实标签字典列表，每个元素对应一个尺度
        """
        total_loss = 0
        loss_dict = {'coord_loss': 0, 'obj_loss': 0, 'noobj_loss': 0, 'cls_loss': 0}
        
        for i, pred in enumerate(predictions):
            # 获取当前尺度的anchor
            anchor = self.anchors[i]
            
            print("Targets type:", type(targets))
            print("Targets length:", len(targets))

            # 关键修改：使用对应尺度的目标，而不是总是使用 targets[0]
            target = targets[i]  # 改为 targets[i] 而不是 targets[0]
            tgt_boxes = target['boxes']      # 期望: [B, A, H, W, 4] 
            tgt_obj = target['conf']         # 期望: [B, A, H, W]
            tgt_cls = target['classes']      # 期望: [B, A, H, W]
            tgt_mask = target['mask']        # 期望: [B, A, H, W]
            
            # 调试信息
            # print(f"Scale {i}:")
            # print(f"pred shape: {pred.shape}")
            # print(f"tgt_boxes shape: {tgt_boxes.shape}")
            # print(f"tgt_mask shape: {tgt_mask.shape}")
            
            # 预测张量应该是 [B, A, H, W, 85]，分离不同部分
            pred_boxes = pred[..., :4]      # [B, A, H, W, 4]
            pred_obj = pred[..., 4]         # [B, A, H, W]
            pred_cls = pred[..., 5:]        # [B, A, H, W, num_classes]
            
            # 🔥 关键修改：确保维度一致性检查
            # print(f"After extraction:")
            # print(f"pred_boxes shape: {pred_boxes.shape}")
            # print(f"tgt_boxes shape: {tgt_boxes.shape}")
            # print(f"tgt_mask shape: {tgt_mask.shape}")
            
            # 验证维度是否匹配
            if pred_boxes.shape != tgt_boxes.shape:
                raise ValueError(f"尺度{i}维度不匹配: pred={pred_boxes.shape} vs tgt={tgt_boxes.shape}")
            
            # 计算坐标损失 (只对有物体的位置)
            coord_loss = self.mse_loss(pred_boxes, tgt_boxes) * tgt_mask.unsqueeze(-1)
            coord_loss = coord_loss.sum() / max(1, tgt_mask.sum())
            
            # 计算置信度损失
            obj_loss = self.bce_loss(pred_obj, tgt_obj) * tgt_mask
            noobj_loss = self.bce_loss(pred_obj, tgt_obj) * (1 - tgt_mask)
            
            obj_loss = obj_loss.sum() / max(1, tgt_mask.sum())
            noobj_loss = noobj_loss.sum() / max(1, (1 - tgt_mask).sum())
            
            # 计算分类损失 (只对有物体的位置)
            valid_positions = tgt_mask.bool()
            if valid_positions.sum() > 0:
                pred_cls_valid = pred_cls[valid_positions]  # [num_valid_targets, num_classes]
                tgt_cls_valid = tgt_cls[valid_positions].long()  # [num_valid_targets]
                cls_loss = self.ce_loss(pred_cls_valid, tgt_cls_valid).mean()
            else:
                cls_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
            
            # 加权总损失
            scale_loss = (self.lambda_coord * coord_loss + 
                        obj_loss + 
                        self.lambda_noobj * noobj_loss + 
                        cls_loss)
            
            total_loss += scale_loss
            
            # 记录各项损失
            loss_dict['coord_loss'] += coord_loss.item()
            loss_dict['obj_loss'] += obj_loss.item()
            loss_dict['noobj_loss'] += noobj_loss.item()
            loss_dict['cls_loss'] += cls_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict 


        
    # def forward(self, predictions, targets):
    #     """
    #     predictions: 模型输出列表 [small, medium, large]
    #     targets: 真实标签字典列表，每个元素对应一个尺度
    #     """
    #     # 兼容 dict 输入
    #     if isinstance(targets, dict):
    #         # 假设只有一个尺度
    #         targets = [targets]
    #     # 兼容 {0: {...}} 这种情况
    #     elif isinstance(targets, dict) and 0 in targets:
    #         targets = [targets[0]]
    #     total_loss = 0
    #     loss_dict = {'coord_loss': 0, 'obj_loss': 0, 'noobj_loss': 0, 'cls_loss': 0}
        
    #     for i, pred in enumerate(predictions):
    #         # 获取当前尺度的anchor
    #         anchor = self.anchors[i]
            
    #         # 分离预测的不同部分
    #         pred_boxes = pred[..., :4]      # 边界框坐标 (tx, ty, tw, th)
    #         pred_obj = pred[..., 4]        # 置信度
    #         pred_cls = pred[..., 5:]       # 类别概率
            

    #         # 添加调试代码
    #         #print("targets keys:", targets.keys())
    #         print("targets type:", type(targets))
    #         print("targets content:", targets)


    #         # 获取目标值
    #         target = targets[i]
    #         tgt_boxes = target['boxes']
    #         tgt_obj = target['conf']
    #         tgt_cls = target['classes']
    #         tgt_mask = target['mask']      # 指示哪些位置有物体
    #         # 打印shape调试
    #         if pred_boxes.shape != tgt_boxes.shape:
    #             print("Before permute, pred_boxes shape:", pred_boxes.shape)
    #             tgt_boxes = tgt_boxes.permute(0, 3, 1, 2, 4)  # [B, A, H, W, 4] -> [B, H, W, A, 4]
    #             print("After permute, tgt_boxes shape:", tgt_boxes.shape)
    #         # 计算坐标损失 (只对有物体的位置)
    #         coord_loss = self.mse_loss(pred_boxes, tgt_boxes) * tgt_mask.unsqueeze(-1)
    #         coord_loss = coord_loss.sum() / max(1, tgt_mask.sum())
            
    #         # 计算置信度损失
    #         obj_loss = self.bce_loss(pred_obj, tgt_obj) * tgt_mask
    #         noobj_loss = self.bce_loss(pred_obj, tgt_obj) * (1 - tgt_mask)
            
    #         obj_loss = obj_loss.sum() / max(1, tgt_mask.sum())
    #         noobj_loss = noobj_loss.sum() / max(1, (1 - tgt_mask).sum())
            
    #         # 计算分类损失 (只对有物体的位置)
    #         cls_loss = self.ce_loss(
    #             pred_cls.view(-1, self.num_classes),
    #             tgt_cls.view(-1).long()
    #         ) * tgt_mask.view(-1)
    #         cls_loss = cls_loss.sum() / max(1, tgt_mask.sum())
            
    #         # 加权总损失
    #         scale_loss = (self.lambda_coord * coord_loss + 
    #                      obj_loss + 
    #                      self.lambda_noobj * noobj_loss + 
    #                      cls_loss)
            
    #         total_loss += scale_loss
            
    #         # 记录各项损失
    #         loss_dict['coord_loss'] += coord_loss.item()
    #         loss_dict['obj_loss'] += obj_loss.item()
    #         loss_dict['noobj_loss'] += noobj_loss.item()
    #         loss_dict['cls_loss'] += cls_loss.item()
        
    #     loss_dict['total_loss'] = total_loss.item()
    #     return total_loss, loss_dict

def reweight_loss(loss, labels, class_weights):
    """
    损失重加权：根据类别权重调整loss。
    loss: [N]，labels: [N]，class_weights: dict or tensor
    """
    # TODO: 按labels查找class_weights，乘以loss
    # 示例：loss = loss * class_weights[labels]
    return loss