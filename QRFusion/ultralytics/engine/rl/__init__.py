# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Reinforcement Learning Training Module for Ultralytics.

This module provides reinforcement learning (RL) training capabilities for object detection,
segmentation, classification, pose estimation, and oriented bounding box models. The implementation
is inspired by VLM-R1's GRPO (Generative Reward Policy Optimization) algorithm.

Components:
    - rewards: IoU-based and class-aware reward functions
    - grpo: Generative Reward Policy Optimization algorithm
    - rl_trainer: Base RL trainer class
    - config: RL training configuration

Example:
    >>> from ultralytics.engine.rl import RLDetectionTrainer, GRPOConfig
    >>> config = GRPOConfig(num_generations=4, kl_coeff=0.1)
    >>> trainer = RLDetectionTrainer(overrides={'model': 'yolo26n.pt', 'rl_config': config})
    >>> trainer.train()
"""

from .config import GRPOConfig, RLConfig
from .grpo import GRPO, GRPOTrainer
from .rewards import (
    ClassAwareReward,
    CompositeReward,
    FormatReward,
    IoUReward,
    RewardFunction,
    RewardRegistry,
    SegmentationIoUReward,
)
from .rl_trainer import (
    BaseRLTrainer,
    RLClassificationTrainer,
    RLDetectionTrainer,
    RLOBBTrainer,
    RLPoseTrainer,
    RLSegmentationTrainer,
)

__all__ = [
    # Config
    "GRPOConfig",
    "RLConfig",
    # GRPO Algorithm
    "GRPO",
    "GRPOTrainer",
    # Rewards
    "RewardFunction",
    "RewardRegistry",
    "IoUReward",
    "ClassAwareReward",
    "SegmentationIoUReward",
    "FormatReward",
    "CompositeReward",
    # Trainers
    "BaseRLTrainer",
    "RLDetectionTrainer",
    "RLSegmentationTrainer",
    "RLClassificationTrainer",
    "RLPoseTrainer",
    "RLOBBTrainer",
]
