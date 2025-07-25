import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.CA import CoordAtt
from models.attention import CBAM, ECA
from models.fpn import BiFPN

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 2. 定义 FPNBlockWithAttention（用 ConvBlock 替代 CSPBlock）
class FPNBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_ratio=2):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.attn = CoordAtt(out_channels, out_channels)
        self.downsample = ConvBlock(out_channels, out_channels, kernel_size=3, stride=downsample_ratio, padding=1)
        
    def forward(self, x):
        x = self.attn(self.conv(x))
        return x, self.downsample(x)




class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3):
        """
        YOLO检测头
        
        参数:
            in_channels: 输入特征图的通道数
            num_classes: 类别数量
            num_anchors: 每个网格单元的anchor数量
        """
        super(YOLOHead, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.in_channels = in_channels
        
        # 输出通道数: num_anchors * (4 + 1 + num_classes)
        # 4: bbox坐标 (x, y, w, h)
        # 1: objectness confidence
        # num_classes: 类别概率
        self.out_channels = num_anchors * (5 + num_classes)
        
        # 检测头网络结构
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 最终预测层
        self.pred_conv = nn.Conv2d(in_channels, self.out_channels, 1)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 特殊初始化最后一层，降低初始confidence
        nn.init.normal_(self.pred_conv.weight, std=0.01)
        nn.init.constant_(self.pred_conv.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征图 [B, in_channels, H, W]
        
        返回:
            predictions: [B, num_anchors, H, W, 5+num_classes]
        """
        # 特征处理
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        
        # 预测
        predictions = self.pred_conv(x)  # [B, num_anchors*(5+num_classes), H, W]
        
        # 重新组织张量形状
        batch_size, _, height, width = predictions.shape
        predictions = predictions.view(
            batch_size, 
            self.num_anchors, 
            5 + self.num_classes, 
            height, 
            width
        )
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()  # [B, num_anchors, H, W, 5+num_classes]
        
        return predictions

class YOLOModel(nn.Module):
    def __init__(self, num_classes=80, img_size=640, anchors=None, use_attention=False, attention_type="CBAM", backbone_type="efficientnet_v2_s", fpn_type="fpn"):
        super(YOLOModel, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.anchors = anchors or [
            [(10,13), (16,30), (33,23)],    # 小目标 (80x80)
            [(30,61), (62,45), (59,119)],   # 中目标 (40x40)
            [(116,90), (156,198), (373,326)] # 大目标 (20x20)
        ]
        self.use_attention = use_attention
        self.attention_type = attention_type
        self.backbone_type = backbone_type
        self.fpn_type = fpn_type
        # 1. Backbone: 支持多种主干
        if backbone_type == "efficientnet_v2_s":
        backbone = torchvision.models.efficientnet_v2_s(pretrained=True)
        self.backbone = backbone.features
        elif backbone_type == "mobilenet_v3_large":
            backbone = torchvision.models.mobilenet_v3_large(pretrained=True)
            self.backbone = backbone.features
        elif backbone_type == "ghostnet":
            try:
                from torchvision.models import ghostnet
                backbone = ghostnet(pretrained=True)
                self.backbone = backbone.features
            except ImportError:
                raise ImportError("GhostNet 需要 torchvision>=0.13，或自行实现/安装ghostnet模块")
        else:
            raise ValueError(f"不支持的backbone_type: {backbone_type}")
        # 2. 获取 Backbone 的特征层输出通道数
        dummy_input = torch.randn(1, 3, img_size, img_size)
        features = []
        with torch.no_grad():
            for i, layer in enumerate(self.backbone):
                dummy_input = layer(dummy_input)
                if i in [3, 5, 7, 8]:  # 这些索引对应不同尺度的特征图
                    features.append(dummy_input)
        self.backbone_out_channels = [f.size(1) for f in features]
        print(f"Backbone实际输出通道数: {self.backbone_out_channels}")  # 调试用
        # 3. 调整通道数的过渡层
        self.adjust_channels = nn.ModuleList([
            nn.Conv2d(self.backbone_out_channels[-1], 512, 1),  # 最后一层 -> 512
            nn.Conv2d(self.backbone_out_channels[-2], 256, 1),  # 倒数第二层 -> 256
            nn.Conv2d(self.backbone_out_channels[-3], 128, 1)   # 倒数第三层 -> 128
        ])
        # 4. 注意力机制（可选）
        if self.use_attention:
            if self.attention_type == "CBAM":
                self.attn_large = CBAM(512)
                self.attn_medium = CBAM(256)
                self.attn_small = CBAM(128)
            elif self.attention_type == "ECA":
                self.attn_large = ECA(512)
                self.attn_medium = ECA(256)
                self.attn_small = ECA(128)
            else:
                raise ValueError(f"不支持的attention_type: {self.attention_type}")
        # 4. 特征金字塔网络 (FPN/BiFPN)
        if fpn_type == "fpn":
        self.fpn_layers = nn.ModuleList([
            FPNBlockWithAttention(512, 512, downsample_ratio=2),  # 下采样到 20x20
            FPNBlockWithAttention(512 + 256, 256, downsample_ratio=2),  # 下采样到 40x40
            FPNBlockWithAttention(256 + 128, 128, downsample_ratio=2)   # 下采样到 80x80
        ])
            self.use_bifpn = False
        elif fpn_type == "bifpn":
            self.bifpn = BiFPN(512)  # 假设输入通道为512，可根据实际调整
            self.use_bifpn = True
        else:
            raise ValueError(f"不支持的fpn_type: {fpn_type}")
        # 5. 上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 6. YOLO 检测头
        self.head_large = YOLOHead(512, num_classes)  # 大目标 (20x20)
        self.head_medium = YOLOHead(256, num_classes)  # 中目标 (40x40)
        self.head_small = YOLOHead(128, num_classes)   # 小目标 (80x80)
    def forward(self, x):
        # 1. Backbone 特征提取
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [3, 5, 7, 8]:  # 提取多尺度特征
                features.append(x)
        # 2. 调整通道数
        p5 = self.adjust_channels[0](features[-1])  # [B, 512, 20, 20]
        p4 = self.adjust_channels[1](features[-2])  # [B, 256, 40, 40]
        p3 = self.adjust_channels[2](features[-3])  # [B, 128, 80, 80]
        # 2.5 注意力机制（可选）
        if self.use_attention:
            p5 = self.attn_large(p5)
            p4 = self.attn_medium(p4)
            p3 = self.attn_small(p3)
        # 3. FPN/BiFPN 特征融合
        if hasattr(self, 'use_bifpn') and self.use_bifpn:
            # BiFPN输入为多尺度特征，输出同样为多尺度
            features_fused = self.bifpn([p5, p4, p3])
            p5, p4, p3 = features_fused
        else:
            # FPN流程
        p5, _ = self.fpn_layers[0](p5)  # [B, 512, 20, 20]
            # 其余FPN流程保持不变
        
        # 4. 多尺度检测头
        outputs = [
            self.head_small(p3),    # [B, 3, 80, 80, 85]
            self.head_medium(p4),   # [B, 3, 40, 40, 85]
            self.head_large(p5)     # [B, 3, 20, 20, 85]
        ]
        
        # 打印最终输出尺寸（调试用）
        print("Final output shapes:")
        for i, out in enumerate(outputs):
            print(f"Head {i}: {out.shape}")
        
        return outputs

    def get_features(self, x):
        """
        提取用于对比学习的特征（backbone最后一层+MLP投影）。
        返回: [B, D] 特征
        """
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == len(self.backbone) - 1:
                features.append(x)
        # 池化+MLP投影
        feat = features[-1]
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        if not hasattr(self, 'proj_head'):
            self.proj_head = nn.Sequential(
                nn.Linear(feat.size(1), 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            ).to(feat.device)
        return self.proj_head(feat)




