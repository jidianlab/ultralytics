---
comments: true
description: 学习如何使用 YOLO26 高效训练目标检测模型，包括设置、增强和硬件利用的综合说明。
keywords: Ultralytics, YOLO26, 模型训练, 深度学习, 目标检测, GPU训练, 数据集增强, 超参数调优, 强化学习
---

# 使用 Ultralytics YOLO 进行模型训练

## 简介

训练深度学习模型涉及向其提供数据并调整其参数，使其能够做出准确的预测。Ultralytics YOLO26 中的训练模式专为有效和高效地训练目标检测模型而设计，充分利用现代硬件功能。

## 为什么选择 Ultralytics YOLO 进行训练？

以下是选择 YOLO26 训练模式的一些重要原因：

- **效率：** 充分利用您的硬件，无论是单 GPU 设置还是跨多个 GPU 扩展。
- **多功能性：** 除了 COCO、VOC 和 ImageNet 等现成数据集外，还可以在自定义数据集上进行训练。
- **用户友好：** 简单而强大的 CLI 和 Python 接口，提供直接的训练体验。
- **超参数灵活性：** 广泛的可自定义超参数，用于微调模型性能。

## 使用示例

在 COCO8 数据集上以图像大小 640 训练 YOLO26n 100 个 epoch。

!!! example "单 GPU 和 CPU 训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.yaml")  # 从 YAML 构建新模型
        model = YOLO("yolo26n.pt")    # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "命令行"

        ```bash
        # 从 YAML 构建新模型并从头开始训练
        yolo detect train data=coco8.yaml model=yolo26n.yaml epochs=100 imgsz=640

        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 训练设置

YOLO 模型的训练设置包括训练过程中使用的各种超参数和配置。这些设置影响模型的性能、速度和准确性。

## 强化学习训练

Ultralytics YOLO 支持强化学习（RL）训练模式，该模式将传统的监督学习与基于奖励的优化相结合。此方法受 GRPO（Group Relative Policy Optimization）算法的启发，可以通过结合 IoU 和类别感知奖励信号来改进模型性能。

### 强化学习参数

| 参数                    | 类型    | 默认值  | 描述                                                                           |
| ----------------------- | ------- | ------- | ------------------------------------------------------------------------------ |
| `rl_enabled`            | `bool`  | `False` | 启用强化学习训练模式                                                           |
| `rl_weight`             | `float` | `0.5`   | RL 损失与监督损失的权重                                                        |
| `iou_weight`            | `float` | `0.7`   | 基于 IoU 的奖励分量权重                                                        |
| `cls_weight`            | `float` | `0.0`   | 基于分类的奖励分量权重                                                         |
| `completeness_weight`   | `float` | `0.3`   | 检测完整性奖励权重                                                             |
| `iou_threshold`         | `float` | `0.5`   | 预测与真实值匹配的 IoU 阈值                                                    |
| `grpo_iterations`       | `int`   | `1`     | 每批次的 GRPO 迭代次数                                                         |
| `grpo_epsilon`          | `float` | `0.2`   | PPO 风格裁剪参数                                                               |
| `grpo_beta`             | `float` | `0.01`  | KL 散度系数                                                                    |
| `use_area_weighting`    | `bool`  | `False` | 启用面积加权奖励                                                               |
| `area_power`            | `float` | `0.5`   | 面积加权计算的幂次                                                             |

!!! example "强化学习训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")

        # 启用 RL 训练
        results = model.train(
            data="coco8.yaml",
            epochs=100,
            rl_enabled=True,
            rl_weight=0.5,
            iou_weight=0.7,
            completeness_weight=0.3,
            use_area_weighting=True  # 更重视较大的目标
        )
        ```

    === "命令行"

        ```bash
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 rl_enabled=True rl_weight=0.5
        ```

### 何时使用 RL 训练

RL 训练在以下场景中特别有效：

- **不平衡数据集**：使用 `class_weights` 强调重要类别
- **尺寸感知检测**：启用 `use_area_weighting` 以关注较大的目标
- **检测完整性**：调整 `completeness_weight` 以惩罚漏检
- **微调**：将 RL 损失与监督损失结合进行模型精细调优

## 高级数据配置

### 标签别名

标签别名允许您将多个数据集标签合并为统一的类别，当组合具有不同标注约定的数据集时非常有用。

在您的 `data.yaml` 中配置：

```yaml
# 将多个标签合并为统一的类别
label_aliases:
  person:
    - human
    - pedestrian
    - people
  vehicle:
    - car
    - truck
    - bus
```

### 按类别数据增强

对特定的目标类别应用不同的增强参数，基于其像素区域。未进行自定义配置的类别使用默认增强设置。

在您的 `data.yaml` 中配置：

```yaml
# 对不同类别应用不同的增强
class_augmentations:
  person:
    hsv_h: 0.02
    hsv_s: 0.8
    hsv_v: 0.5
    region_expand: 0.1  # 10% 扩展（基于比例，< 1.0）
  vehicle:
    hsv_h: 0.01
    hsv_s: 0.5
    region_expand: 10   # 10 像素扩展（基于像素，>= 1.0）
```

!!! note "区域扩展说明"

    - 小于 1.0 的值被视为比例（例如，0.1 = bbox 尺寸的 10%）
    - 大于等于 1.0 的值被视为像素值
    - 扩展在图像边缘和其他目标边界处停止
    - 当区域重叠时，使用具有最小系数的增强

## 数据增强设置

增强技术对于改进 YOLO 模型的稳健性和性能至关重要，它通过向训练数据引入变化，帮助模型更好地泛化到未见过的数据。

## 日志记录

在训练 YOLO26 模型时，跟踪模型性能随时间变化可能很有价值。Ultralytics YOLO 支持三种类型的日志记录器 - Comet、ClearML 和 TensorBoard。

## 常见问题

### 如何使用 Ultralytics YOLO26 训练目标检测模型？

要训练目标检测模型，您可以使用 Python API 或 CLI：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")

        # 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "命令行"

        ```bash
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

### 如何使用强化学习训练以提高模型性能？

启用 RL 训练并调整相关参数：

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.train(
    data="coco8.yaml",
    epochs=100,
    rl_enabled=True,
    rl_weight=0.5,
    iou_weight=0.7,
    completeness_weight=0.3
)
```

### 如何合并来自不同数据集的标签？

在 `data.yaml` 中使用 `label_aliases`：

```yaml
label_aliases:
  person:
    - human
    - pedestrian
```
