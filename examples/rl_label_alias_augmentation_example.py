# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Example usage of the new features added to Ultralytics:
1. Reinforcement Learning Training
2. Label Aliasing
3. Per-Class Data Augmentation

This file demonstrates how to configure and use these features.
"""

# Example 1: Label Aliasing Configuration
# Add this to your data.yaml file:
LABEL_ALIAS_CONFIG = """
# data.yaml
train: /path/to/train/images
val: /path/to/val/images
nc: 3  # Number of unified classes after aliasing
names: {0: 'person', 1: 'vehicle', 2: 'animal'}

# Label aliasing configuration
# Maps multiple dataset labels to unified class names
label_aliases:
  person:  # Target class name (unified)
    - human
    - pedestrian
    - people
    - man
    - woman
    - child
  vehicle:
    - car
    - truck
    - bus
    - motorcycle
    - bicycle
  animal:
    - dog
    - cat
    - bird
    - horse
"""

# Example 2: Per-Class Data Augmentation Configuration
# Add this to your data.yaml file:
PER_CLASS_AUG_CONFIG = """
# data.yaml (continued)
# Per-class augmentation configuration
# Different augmentation parameters for different classes
class_augmentations:
  person:  # Apply strong augmentation to person class
    hsv_h: 0.02        # HSV hue augmentation
    hsv_s: 0.8         # HSV saturation augmentation
    hsv_v: 0.5         # HSV value augmentation
    brightness: 0.2    # Brightness adjustment
    contrast: 0.2      # Contrast adjustment
    blur: 3            # Gaussian blur kernel size
    noise: 5           # Gaussian noise std
    region_expand: 0.1 # Expand region by 10% (ratio-based, < 1.0)
  vehicle:
    hsv_h: 0.01
    hsv_s: 0.5
    hsv_v: 0.3
    region_expand: 10  # Expand region by 10 pixels (pixel-based, >= 1.0)
  # 'animal' class uses default augmentation (not specified here)
"""

# Example 3: Reinforcement Learning Training
# Using the RL trainer with IoU and class-aware rewards
RL_TRAINING_EXAMPLE = """
from ultralytics.engine.rl_trainer import (
    DetectionRLTrainer,
    SegmentationRLTrainer,
    PoseRLTrainer,
    OBBRLTrainer,
    ClassificationRLTrainer,
    RewardFunction,
    AreaWeightedReward,
    get_reward_function,
)

# Basic RL training for detection
trainer = DetectionRLTrainer(overrides={
    'model': 'yolo26n.pt',
    'data': 'coco8.yaml',
    'epochs': 100,
    # RL-specific parameters
    'rl_weight': 0.5,           # Weight for RL loss vs supervised loss
    'iou_weight': 0.7,          # Weight for IoU-based reward
    'cls_weight': 0.0,          # Weight for classification reward
    'completeness_weight': 0.3, # Weight for detection completeness reward
    'iou_threshold': 0.5,       # IoU threshold for matching
    # GRPO parameters
    'grpo_iterations': 1,       # Number of GRPO iterations per batch
    'grpo_epsilon': 0.2,        # Clipping parameter
    'grpo_beta': 0.01,          # KL divergence coefficient
    # Optional: class-specific reward weights
    'class_weights': {0: 2.0, 1: 1.0},  # Class 0 is twice as important
    # Optional: use area-weighted rewards
    'use_area_weighting': True,
    'area_power': 0.5,          # Power for area weighting
})

# Custom reward function
reward_func = RewardFunction(
    iou_weight=0.7,
    cls_weight=0.0,
    completeness_weight=0.3,
    class_weights={0: 2.0, 1: 1.0},
    iou_threshold=0.5,
)

# Area-weighted reward function (gives more importance to larger objects)
area_reward = AreaWeightedReward(
    iou_weight=0.7,
    cls_weight=0.0,
    completeness_weight=0.3,
    area_power=0.5,
    normalize_area=True,
)

# Factory function for reward creation
rf = get_reward_function('area_weighted', iou_weight=0.7, area_power=0.5)
"""

# Example 4: Python API Usage
PYTHON_API_EXAMPLE = """
import numpy as np
import torch
from ultralytics.data import LabelAliasManager, PerClassAugmentation

# Label Aliasing
alias_manager = LabelAliasManager(
    aliases={
        'person': ['human', 'pedestrian'],
        'vehicle': ['car', 'truck', 'bus']
    },
    original_names={'human': 0, 'car': 1, 'truck': 2, 'dog': 3}
)

# Get unified class names
print(alias_manager.get_unified_names())  # {0: 'person', 1: 'vehicle', 2: 'dog'}

# Map labels
labels = [{'cls': np.array([[0], [1], [2]])}]  # human, car, truck
updated = alias_manager.update_labels(labels)
print(updated[0]['cls'])  # [[0], [1], [1]] - human->person, car->vehicle, truck->vehicle

# Per-Class Augmentation
config = {
    'class_augmentations': {
        'person': {'hsv_h': 0.05, 'hsv_s': 0.9, 'region_expand': 0.1}
    },
    'hsv_h': 0.015,  # Default values
    'hsv_s': 0.7,
    'hsv_v': 0.4,
}

augmenter = PerClassAugmentation.from_config(config, class_names={0: 'person', 1: 'car'})

# Apply augmentation to image
img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
bboxes = np.array([[100, 100, 300, 300], [400, 400, 600, 600]])  # xyxy format
classes = np.array([0, 1])  # person, car

augmented_img = augmenter.apply(img, bboxes, classes)

# Reward Function Usage
from ultralytics.engine import RewardFunction, compute_iou_matrix

reward_func = RewardFunction(
    iou_weight=0.7,
    completeness_weight=0.3,
    class_weights={0: 2.0, 1: 1.0}
)

pred_boxes = torch.tensor([[100, 100, 300, 300]])
pred_cls = torch.tensor([0])
target_boxes = torch.tensor([[110, 110, 290, 290]])
target_cls = torch.tensor([0])

reward = reward_func.compute_reward(pred_boxes, pred_cls, target_boxes, target_cls)
print(f'Reward: {reward.item():.4f}')
"""


def print_examples():
    """Print all example configurations."""
    print("=" * 80)
    print("EXAMPLE 1: Label Aliasing Configuration (add to data.yaml)")
    print("=" * 80)
    print(LABEL_ALIAS_CONFIG)

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Per-Class Data Augmentation Configuration (add to data.yaml)")
    print("=" * 80)
    print(PER_CLASS_AUG_CONFIG)

    print("\n" + "=" * 80)
    print("EXAMPLE 3: Reinforcement Learning Training")
    print("=" * 80)
    print(RL_TRAINING_EXAMPLE)

    print("\n" + "=" * 80)
    print("EXAMPLE 4: Python API Usage")
    print("=" * 80)
    print(PYTHON_API_EXAMPLE)


if __name__ == "__main__":
    print_examples()
