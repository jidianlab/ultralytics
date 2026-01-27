# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
QRFusion Ultralytics Engine Module.

This module provides the core training, validation, and export functionality
with reinforcement learning enhancements.
"""

from .rl import (
    # Config
    GRPOConfig,
    RLConfig,
    # GRPO
    GRPO,
    GRPOTrainer,
    # Rewards
    RewardFunction,
    RewardRegistry,
    IoUReward,
    ClassAwareReward,
    SegmentationIoUReward,
    FormatReward,
    CompositeReward,
    # Trainers
    BaseRLTrainer,
    RLDetectionTrainer,
    RLSegmentationTrainer,
    RLClassificationTrainer,
    RLPoseTrainer,
    RLOBBTrainer,
)

__all__ = [
    # Config
    "GRPOConfig",
    "RLConfig",
    # GRPO
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
