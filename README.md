# Multi-Object Detection 基于YOLO的多目标检测项目

## 项目简介

本项目实现了一个基于YOLO思想的多目标检测系统，支持COCO数据集，集成了注意力机制、特征金字塔、半监督伪标签、类别均衡采样、损失重加权、剪枝量化、对比学习等多种创新与实用模块，适用于学术研究与工程实践。

---

## 环境依赖

请先安装Python 3.8+，推荐使用虚拟环境。

```bash
pip install -r requirements.txt
```

**主要依赖包：**
- torch >= 1.9.0
- torchvision >= 0.10.0
- opencv-python >= 4.5.0
- albumentations >= 1.0.0
- numpy, tqdm, PyYAML, wandb, scikit-learn, matplotlib, seaborn

---

## 数据准备

1. **自动下载COCO 2017数据集**  
   运行如下命令自动下载并解压数据集（如遇网络问题可手动下载）：

   ```bash
   python download_coco.py
   ```

2. **数据目录结构**  
   ```
   data/coco/
     ├── train2017/
     ├── val2017/
     ├── test2017/         # 可选
     └── annotations/
         ├── instances_train2017.json
         ├── instances_val2017.json
         └── instances_test2017.json
   ```

---

## 配置说明

所有训练和推理参数均可在`config/config.yaml`中配置，包括类别数、输入尺寸、数据路径、训练超参数、模型结构等。例如：

```yaml
num_classes: 80
img_size: 640
train_images: "data/coco/train2017"
train_annotations: "data/coco/annotations/instances_train2017.json"
batch_size: 8
epochs: 50
use_attention: false
attention_type: "CBAM"
backbone_type: "efficientnet_v2_s"
fpn_type: "fpn"
```

---

## 训练模型

运行如下命令开始训练：

```bash
python scripts/train.py
```

- 支持配置文件自定义参数
- 支持半监督伪标签训练（需配置`unlabeled_images`路径）
- 支持类别均衡采样、损失重加权、注意力机制、不同主干网络等多种实验

---

## 推理与可视化

推理单张图片并保存结果：

```bash
python scripts/inference.py
```

- 默认读取`config/config.yaml`和`best_model.pth`
- 推理图片路径和结果保存可在脚本中修改

---


`致谢

本项目为机器学习课程实践作业，感谢曹鹏老师的悉心指导！
