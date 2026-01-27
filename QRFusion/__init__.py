# QRFusion - Reinforcement Learning Enhanced Ultralytics Framework
# AGPL-3.0 License
"""
QRFusion - Reinforcement Learning Enhanced Computer Vision Framework.

QRFusion extends the Ultralytics framework with state-of-the-art reinforcement learning
training capabilities, inspired by VLM-R1's GRPO algorithm. This enables improved training
for object detection, segmentation, classification, pose estimation, and oriented bounding
box detection tasks.

Features:
    - GRPO (Generative Reward Policy Optimization) algorithm for stable RL training
    - IoU-based reward functions that directly optimize detection metrics
    - Class-aware rewards with per-class weighting for imbalanced datasets
    - Pixel area-based rewards for balanced performance across object sizes
    - Seamless integration with existing Ultralytics training pipeline
    - Support for all Ultralytics model types (YOLO, RT-DETR, etc.)

Quick Start:
    # Standard Ultralytics usage
    >>> from QRFusion.ultralytics import YOLO
    >>> model = YOLO('yolo26n.pt')
    >>> model.train(data='coco8.yaml')

    # RL-enhanced training
    >>> from QRFusion.ultralytics import RLDetectionTrainer, RLConfig
    >>> config = RLConfig(enabled=True, algorithm='grpo', rl_weight=0.3)
    >>> trainer = RLDetectionTrainer(
    ...     overrides={'model': 'yolo26n.pt', 'data': 'coco8.yaml'},
    ...     rl_config=config
    ... )
    >>> trainer.train()

Architecture Overview:
    QRFusion/
    └── ultralytics/
        ├── __init__.py          # Main package interface
        └── engine/
            └── rl/
                ├── __init__.py  # RL module interface
                ├── config.py    # Configuration classes
                ├── rewards.py   # Reward function implementations
                ├── grpo.py      # GRPO algorithm
                └── rl_trainer.py # RL-enhanced trainers

For detailed documentation, see:
    - README.md in the QRFusion directory
    - docstrings in individual modules

License:
    AGPL-3.0 License - https://ultralytics.com/license

References:
    - VLM-R1: https://github.com/om-ai-lab/VLM-R1
    - Ultralytics: https://github.com/ultralytics/ultralytics
"""

__version__ = "1.0.0"
__author__ = "QRFusion Team"

from .ultralytics import (
    # Original Ultralytics models
    YOLO,
    RTDETR,
    SAM,
    FastSAM,
    NAS,
    # RL Configuration
    GRPOConfig,
    RLConfig,
    # GRPO Algorithm
    GRPO,
    GRPOTrainer,
    # Reward Functions
    RewardFunction,
    RewardRegistry,
    IoUReward,
    ClassAwareReward,
    SegmentationIoUReward,
    FormatReward,
    CompositeReward,
    # RL Trainers
    BaseRLTrainer,
    RLDetectionTrainer,
    RLSegmentationTrainer,
    RLClassificationTrainer,
    RLPoseTrainer,
    RLOBBTrainer,
    # Utility functions
    get_rl_trainer,
    create_rl_config,
)

__all__ = [
    # Version
    "__version__",
    # Original Ultralytics models
    "YOLO",
    "RTDETR",
    "SAM",
    "FastSAM",
    "NAS",
    # RL Configuration
    "GRPOConfig",
    "RLConfig",
    # GRPO Algorithm
    "GRPO",
    "GRPOTrainer",
    # Reward Functions
    "RewardFunction",
    "RewardRegistry",
    "IoUReward",
    "ClassAwareReward",
    "SegmentationIoUReward",
    "FormatReward",
    "CompositeReward",
    # RL Trainers
    "BaseRLTrainer",
    "RLDetectionTrainer",
    "RLSegmentationTrainer",
    "RLClassificationTrainer",
    "RLPoseTrainer",
    "RLOBBTrainer",
    # Utility functions
    "get_rl_trainer",
    "create_rl_config",
]
