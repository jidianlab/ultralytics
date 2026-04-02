---
comments: true
description: 优化您的 Ultralytics YOLO 模型性能，了解正确的设置和超参数。学习训练、验证和预测配置。
keywords: YOLO, 超参数, 配置, 训练, 验证, 预测, 模型设置, Ultralytics, 性能优化, 机器学习, 强化学习
---

# 配置

YOLO 设置和超参数在模型的性能、速度和准确性方面起着关键作用。这些设置和超参数可以在训练、验证和预测等各个阶段影响模型的行为。

Ultralytics 命令使用以下语法：

!!! example "示例"

    === "命令行"

        ```bash
        yolo TASK MODE ARGS
        ```

    === "Python"

        ```python
        from ultralytics import YOLO

        # 从预训练权重文件加载 YOLO 模型
        model = YOLO("yolo26n.pt")

        # 使用自定义参数在指定模式下运行模型
        MODE = "predict"
        ARGS = {"source": "image.jpg", "imgsz": 640}
        getattr(model, MODE)(**ARGS)
        ```

其中：

- `TASK`（可选）是 ([detect](../tasks/detect.md), [segment](../tasks/segment.md), [classify](../tasks/classify.md), [pose](../tasks/pose.md), [obb](../tasks/obb.md)) 之一
- `MODE`（必需）是 ([train](../modes/train.md), [val](../modes/val.md), [predict](../modes/predict.md), [export](../modes/export.md), [track](../modes/track.md), [benchmark](../modes/benchmark.md)) 之一
- `ARGS`（可选）是覆盖默认值的 `arg=value` 对，如 `imgsz=640`

## 训练设置

YOLO 模型的训练设置包括影响模型性能、速度和准确性的各种超参数和配置。关键设置包括批量大小、学习率、动量和权重衰减。优化器、损失函数和数据集组成的选择也会影响训练过程。

## 数据增强设置

数据增强技术对于提高 YOLO 模型的稳健性和性能至关重要，它通过向训练数据引入变化，帮助模型更好地泛化到未见过的数据。

## 强化学习设置

Ultralytics YOLO 支持强化学习（RL）训练模式，该模式将传统的监督学习与基于奖励的优化相结合，灵感来自于 GRPO（Group Relative Policy Optimization）算法。

| 参数                    | 类型    | 默认值  | 描述                                                                                                   |
| ----------------------- | ------- | ------- | ------------------------------------------------------------------------------------------------------ |
| `rl_enabled`            | `bool`  | `False` | 启用强化学习训练模式。启用后，将监督损失与基于 RL 奖励的损失相结合以改进训练。                         |
| `rl_weight`             | `float` | `0.5`   | RL 损失与监督损失的权重（0.0-1.0）。较高的值更重视基于 RL 的学习。                                     |
| `iou_weight`            | `float` | `0.7`   | 基于 IoU 的奖励分量权重（0.0-1.0）。控制 IoU 匹配对奖励计算的影响程度。                               |
| `cls_weight`            | `float` | `0.0`   | 基于分类的奖励分量权重（0.0-1.0）。控制类别预测准确性在奖励中的重要性。                               |
| `completeness_weight`   | `float` | `0.3`   | 检测完整性奖励权重（0.0-1.0）。惩罚漏检和误检。                                                       |
| `iou_threshold`         | `float` | `0.5`   | 奖励计算中预测与真实值匹配的 IoU 阈值。较高的值要求更严格的匹配。                                     |
| `grpo_iterations`       | `int`   | `1`     | 每批次的 GRPO 迭代次数。更多迭代可以改进策略更新但增加计算量。                                        |
| `grpo_epsilon`          | `float` | `0.2`   | GRPO 的 PPO 风格裁剪参数。控制最大策略更新步长以确保训练稳定性。                                      |
| `grpo_beta`             | `float` | `0.01`  | GRPO 的 KL 散度系数。惩罚与参考策略的较大偏离。                                                       |
| `use_area_weighting`    | `bool`  | `False` | 启用面积加权奖励，较大的目标在奖励计算中获得更高的权重。                                              |
| `area_power`            | `float` | `0.5`   | 面积加权计算的幂次。较高的值增加不同大小目标之间的权重差异。                                          |
| `class_weights`         | `dict`  | `None`  | 每个类别的奖励权重字典（例如 `{0: 2.0, 1: 1.0}`）。允许在 RL 训练中强调某些类别。                      |

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
            completeness_weight=0.3
        )
        ```

    === "命令行"

        ```bash
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 rl_enabled=True rl_weight=0.5
        ```

## 标签别名和按类别增强

高级数据配置选项允许您将多个数据集标签合并为统一的类别，并对特定的目标类别应用不同的增强参数。

### 标签别名

标签别名允许您将多个数据集标签合并为统一的类别，当组合具有不同标注约定的数据集时非常有用。

在您的 `data.yaml` 中配置：

```yaml
# 将多个标签合并为统一的类别
label_aliases:
  person:  # 目标类别名称
    - human
    - pedestrian
    - people
  vehicle:
    - car
    - truck
    - bus
```

### 按类别增强

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

!!! note "区域扩展"

    - 小于 1.0 的值被视为比例（例如，0.1 = bbox 尺寸的 10%）
    - 大于等于 1.0 的值被视为像素值
    - 扩展在图像边缘和其他目标边界处停止
    - 当区域重叠时，使用具有最小系数的增强

## 常见问题

### 如何在训练期间改进 YOLO 模型的性能？

通过调整超参数（如批量大小、学习率、动量和权重衰减）来改进性能。调整数据增强设置，选择合适的优化器，并使用早停或混合精度等技术。您还可以启用强化学习训练以获得更好的结果。

### 如何使用强化学习训练 YOLO 模型？

设置 `rl_enabled=True` 并调整 RL 相关参数：

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

使用 `label_aliases` 配置：

```yaml
label_aliases:
  person:
    - human
    - pedestrian
  vehicle:
    - car
    - truck
```
