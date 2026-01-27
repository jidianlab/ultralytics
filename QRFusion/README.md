# QRFusion - Reinforcement Learning Enhanced Ultralytics Framework

<p align="center">
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="400" alt="QRFusion Logo">
</p>

<p align="center">
  <strong>Reinforcement Learning Enhanced Computer Vision Framework</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#examples">Examples</a>
</p>

---

## Overview

QRFusion extends the [Ultralytics](https://github.com/ultralytics/ultralytics) framework with state-of-the-art **Reinforcement Learning (RL)** training capabilities, inspired by [VLM-R1](https://github.com/om-ai-lab/VLM-R1)'s GRPO (Generative Reward Policy Optimization) algorithm.

This enables improved training for:
- 🔍 **Object Detection** (YOLO)
- 🎭 **Instance Segmentation** (YOLO-Seg)
- 🏷️ **Image Classification** (YOLO-Cls)
- 🦴 **Pose Estimation** (YOLO-Pose)
- 📐 **Oriented Bounding Box Detection** (YOLO-OBB)

## Features

### 🎯 GRPO Algorithm
- Generative Reward Policy Optimization for stable RL training
- PPO-style clipping for controlled policy updates
- KL divergence regularization with reference model

### 📊 IoU-Based Rewards
- Direct optimization of detection metrics
- Support for IoU, GIoU, DIoU, and CIoU variants
- Pixel area-based weighting for balanced training

### 🏷️ Class-Aware Rewards
- Per-class reward weighting for imbalanced datasets
- Soft and hard label support
- Configurable class priority

### 🔧 Modular Architecture
- Composite reward functions with configurable weights
- Easy extension for custom rewards
- Registry pattern for reward management

### 🔄 Seamless Integration
- Compatible with existing Ultralytics training pipeline
- No changes required for standard training
- Optional RL enhancement with simple configuration

## Installation

QRFusion is included in this repository. No additional installation is required beyond the standard Ultralytics dependencies:

```bash
# Install ultralytics dependencies
pip install ultralytics

# The QRFusion package is ready to use
```

## Quick Start

### Standard Training (Unchanged)

```python
from QRFusion.ultralytics import YOLO

# Standard training works exactly as before
model = YOLO('yolo26n.pt')
model.train(data='coco8.yaml', epochs=100)
```

### RL-Enhanced Training

```python
from QRFusion.ultralytics import RLDetectionTrainer, RLConfig, GRPOConfig

# Configure RL training
grpo_config = GRPOConfig(
    num_generations=4,
    kl_coeff=0.1,
    reward_weights={
        'iou': 1.0,
        'class_accuracy': 0.5,
        'pixel_area': 0.3
    }
)

rl_config = RLConfig(
    enabled=True,
    algorithm='grpo',
    grpo_config=grpo_config,
    rl_weight=0.3,  # Balance between supervised and RL loss
    warmup_epochs=5  # Supervised warmup before RL
)

# Create RL-enhanced trainer
trainer = RLDetectionTrainer(
    overrides={
        'model': 'yolo26n.pt',
        'data': 'coco8.yaml',
        'epochs': 100,
        'batch': 16
    },
    rl_config=rl_config
)

# Train with RL enhancement
trainer.train()
```

### Using Convenience Functions

```python
from QRFusion.ultralytics import get_rl_trainer, create_rl_config

# Create task-specific config with sensible defaults
config = create_rl_config(
    task='detect',
    enabled=True,
    rl_weight=0.3
)

# Get appropriate trainer
trainer = get_rl_trainer(
    task='detect',
    overrides={'model': 'yolo26n.pt', 'data': 'coco8.yaml'},
    rl_config=config
)

trainer.train()
```

## Documentation

### Configuration Classes

#### GRPOConfig
Core configuration for GRPO algorithm:

```python
GRPOConfig(
    num_generations=4,      # Samples per input for advantage computation
    kl_coeff=0.1,           # KL divergence penalty coefficient
    clip_ratio=0.2,         # PPO clipping ratio
    gamma=0.99,             # Discount factor
    gae_lambda=0.95,        # GAE lambda
    value_loss_coeff=0.5,   # Value function loss weight
    entropy_coeff=0.01,     # Entropy bonus coefficient
    max_grad_norm=0.5,      # Gradient clipping threshold
    normalize_advantage=True,
    use_reference_model=True,
    reward_weights={...},   # Per-reward weights
    temperature=1.0,        # Sampling temperature
    ppo_epochs=4,           # PPO update epochs
    mini_batch_size=8       # Mini-batch size for PPO
)
```

#### RLConfig
High-level RL training configuration:

```python
RLConfig(
    enabled=True,           # Enable RL training
    algorithm='grpo',       # Algorithm: 'grpo', 'ppo', 'a2c'
    grpo_config=GRPOConfig(),
    reward_type='composite', # 'iou', 'class_aware', 'composite', 'segmentation'
    update_frequency=1,     # RL updates per batch
    warmup_epochs=0,        # Supervised warmup epochs
    rl_weight=0.5,          # RL loss weight
    save_rl_checkpoints=True,
    log_rewards=True,
    class_weights=None,     # Per-class reward weights
    area_weights={          # Size-based weights
        'small': 1.5,
        'medium': 1.0,
        'large': 0.8
    }
)
```

### Reward Functions

#### IoUReward
Primary reward for localization quality:

```python
from QRFusion.ultralytics import IoUReward

reward_fn = IoUReward(
    iou_type='ciou',  # 'iou', 'giou', 'diou', 'ciou'
    weight=1.0
)
```

#### ClassAwareReward
Classification accuracy with per-class weighting:

```python
from QRFusion.ultralytics import ClassAwareReward

reward_fn = ClassAwareReward(
    class_weights={0: 1.5, 1: 1.0, 2: 2.0},  # Per-class weights
    use_soft_labels=True
)
```

#### CompositeReward
Combine multiple reward signals:

```python
from QRFusion.ultralytics import CompositeReward, IoUReward, ClassAwareReward

reward_fn = CompositeReward(
    rewards=[
        IoUReward(weight=1.0),
        ClassAwareReward(weight=0.5)
    ],
    normalize=True
)
```

### RL Trainers

| Task | Trainer Class | Default Rewards |
|------|--------------|-----------------|
| Detection | `RLDetectionTrainer` | IoU, Class, Area |
| Segmentation | `RLSegmentationTrainer` | IoU, Mask IoU, Class, Area |
| Classification | `RLClassificationTrainer` | Class Accuracy |
| Pose | `RLPoseTrainer` | IoU, OKS, Class |
| OBB | `RLOBBTrainer` | IoU, Angle, Class |

## Examples

### Object Detection with RL

```python
from QRFusion.ultralytics import RLDetectionTrainer, RLConfig

config = RLConfig(
    enabled=True,
    rl_weight=0.3,
    area_weights={'small': 2.0, 'medium': 1.0, 'large': 0.5}  # Emphasize small objects
)

trainer = RLDetectionTrainer(
    overrides={
        'model': 'yolo26n.pt',
        'data': 'coco.yaml',
        'epochs': 300,
        'batch': 16
    },
    rl_config=config
)
trainer.train()
```

### Segmentation with RL

```python
from QRFusion.ultralytics import RLSegmentationTrainer, RLConfig

config = RLConfig(
    enabled=True,
    reward_type='segmentation',  # Includes mask IoU
    grpo_config=GRPOConfig(
        reward_weights={'iou': 0.8, 'mask_iou': 1.0, 'class_accuracy': 0.5}
    )
)

trainer = RLSegmentationTrainer(
    overrides={
        'model': 'yolo26n-seg.pt',
        'data': 'coco-seg.yaml',
        'epochs': 300
    },
    rl_config=config
)
trainer.train()
```

### Custom Reward Function

```python
from QRFusion.ultralytics import RewardFunction, RewardRegistry
import torch

class CustomReward(RewardFunction):
    def __init__(self, weight=1.0):
        super().__init__(name='custom', weight=weight)

    def __call__(self, predictions, targets, batch=None, **kwargs):
        # Your custom reward logic
        return torch.tensor(1.0)

# Register for easy access
RewardRegistry.register('custom', CustomReward())
```

## Architecture

```
QRFusion/
├── __init__.py                 # Package interface
├── README.md                   # This documentation
└── ultralytics/
    ├── __init__.py             # Ultralytics integration
    └── engine/
        ├── __init__.py
        └── rl/
            ├── __init__.py     # RL module interface
            ├── config.py       # GRPOConfig, RLConfig
            ├── rewards.py      # Reward function classes
            ├── grpo.py         # GRPO algorithm
            └── rl_trainer.py   # RL-enhanced trainers
```

## Algorithm Details

### GRPO (Generative Reward Policy Optimization)

GRPO is a policy optimization algorithm inspired by VLM-R1 that:

1. **Generates** multiple outputs for each input sample
2. **Evaluates** outputs using configurable reward functions
3. **Computes** group-wise advantages for relative ranking
4. **Updates** policy using PPO-style clipped objectives

Key advantages:
- **Stable training** through advantage normalization
- **Sample efficiency** via multiple generations per input
- **Flexibility** in reward function design
- **Regularization** through KL penalty with reference model

### Reward Design Philosophy

Following VLM-R1's approach, rewards are designed to:

1. **Directly optimize metrics** - IoU rewards improve localization
2. **Balance training** - Area weights prevent size bias
3. **Handle imbalance** - Class weights for rare categories
4. **Ensure format compliance** - Format rewards for valid outputs

## Performance Tips

1. **Start with warmup** - Use `warmup_epochs=5` for stable initial training
2. **Tune rl_weight** - Start with 0.2-0.3 and adjust based on results
3. **Use class weights** - For imbalanced datasets, weight rare classes higher
4. **Monitor rewards** - Enable `log_rewards=True` to track training progress
5. **Reference model** - Keep `use_reference_model=True` for stable updates

## Citation

If you use QRFusion in your research, please cite:

```bibtex
@software{qrfusion2024,
    title={QRFusion: Reinforcement Learning Enhanced Ultralytics Framework},
    author={QRFusion Team},
    year={2024},
    url={https://github.com/jidianlab/ultralytics}
}

@software{vlm-r1,
    title={VLM-R1: Solving Visual Understanding with Reinforced VLMs},
    author={OM AI Lab},
    year={2024},
    url={https://github.com/om-ai-lab/VLM-R1}
}
```

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](../LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the excellent base framework
- [VLM-R1](https://github.com/om-ai-lab/VLM-R1) for the GRPO algorithm inspiration
- The open-source community for valuable feedback and contributions
