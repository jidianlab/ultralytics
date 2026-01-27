# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# QRFusion - Reinforcement Learning Enhanced Ultralytics Framework
"""
QRFusion Ultralytics - Reinforcement Learning Enhanced Computer Vision Framework.

This package extends Ultralytics with GRPO-based reinforcement learning training capabilities,
inspired by VLM-R1. It provides RL-enhanced trainers for all supported model types:

- Object Detection (YOLO)
- Instance Segmentation (YOLO-Seg)
- Image Classification (YOLO-Cls)
- Pose Estimation (YOLO-Pose)
- Oriented Bounding Box Detection (YOLO-OBB)

Key Features:
    - GRPO (Generative Reward Policy Optimization) algorithm
    - IoU-based reward functions for localization
    - Class-aware rewards with per-class weighting
    - Pixel area-based rewards for balanced training
    - Seamless integration with existing Ultralytics training pipeline

Example:
    >>> from QRFusion.ultralytics import YOLO
    >>> from QRFusion.ultralytics.engine.rl import RLConfig, RLDetectionTrainer
    >>>
    >>> # Standard training
    >>> model = YOLO('yolo26n.pt')
    >>> model.train(data='coco8.yaml', epochs=100)
    >>>
    >>> # RL-enhanced training
    >>> rl_config = RLConfig(enabled=True, algorithm='grpo')
    >>> trainer = RLDetectionTrainer(
    ...     overrides={'model': 'yolo26n.pt', 'data': 'coco8.yaml'},
    ...     rl_config=rl_config
    ... )
    >>> trainer.train()

Installation:
    The QRFusion package is part of the Ultralytics repository and requires no additional
    installation beyond the standard Ultralytics dependencies.

Documentation:
    For detailed documentation on RL training configuration and usage, see:
    - engine/rl/config.py - Configuration classes
    - engine/rl/rewards.py - Reward function implementations
    - engine/rl/grpo.py - GRPO algorithm implementation
    - engine/rl/rl_trainer.py - RL-enhanced trainer classes

References:
    - VLM-R1: https://github.com/om-ai-lab/VLM-R1
    - GRPO Algorithm: Generative Reward Policy Optimization
    - PPO: Proximal Policy Optimization
"""

__version__ = "1.0.0"
__author__ = "QRFusion Team"
__license__ = "AGPL-3.0"

# Import from original ultralytics for backward compatibility
try:
    from ultralytics import YOLO, RTDETR, SAM, FastSAM, NAS
    from ultralytics import __version__ as ultralytics_version
except ImportError:
    # Allow import even if ultralytics is not installed
    YOLO = None
    RTDETR = None
    SAM = None
    FastSAM = None
    NAS = None
    ultralytics_version = "0.0.0"

# Import RL components
from .engine.rl import (
    # Config
    GRPOConfig,
    RLConfig,
    # GRPO Algorithm
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
    # Original Ultralytics
    "YOLO",
    "RTDETR",
    "SAM",
    "FastSAM",
    "NAS",
    "ultralytics_version",
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


def get_rl_trainer(task: str = "detect", **kwargs):
    """Get the appropriate RL trainer for a given task.

    Args:
        task (str): Task type ('detect', 'segment', 'classify', 'pose', 'obb').
        **kwargs: Additional arguments passed to trainer constructor.

    Returns:
        BaseRLTrainer: RL-enhanced trainer for the specified task.

    Raises:
        ValueError: If task type is not supported.

    Examples:
        >>> trainer = get_rl_trainer('detect', model='yolo26n.pt')
        >>> trainer.train()
    """
    trainers = {
        "detect": RLDetectionTrainer,
        "segment": RLSegmentationTrainer,
        "classify": RLClassificationTrainer,
        "pose": RLPoseTrainer,
        "obb": RLOBBTrainer,
    }

    if task not in trainers:
        raise ValueError(f"Unknown task: {task}. Supported: {list(trainers.keys())}")

    return trainers[task](**kwargs)


def create_rl_config(
    task: str = "detect",
    enabled: bool = True,
    algorithm: str = "grpo",
    **kwargs,
) -> RLConfig:
    """Create an RL configuration for a specific task.

    This is a convenience function to create task-appropriate RL configurations
    with sensible defaults.

    Args:
        task (str): Task type ('detect', 'segment', 'classify', 'pose', 'obb').
        enabled (bool): Whether to enable RL training.
        algorithm (str): RL algorithm to use ('grpo', 'ppo', 'a2c').
        **kwargs: Additional configuration overrides.

    Returns:
        RLConfig: Configured RL configuration object.

    Examples:
        >>> config = create_rl_config('detect', enabled=True, rl_weight=0.3)
        >>> trainer = RLDetectionTrainer(rl_config=config)
    """
    from .engine.rl.config import (
        DETECTION_RL_CONFIG,
        SEGMENTATION_RL_CONFIG,
        CLASSIFICATION_RL_CONFIG,
        POSE_RL_CONFIG,
        OBB_RL_CONFIG,
    )

    default_configs = {
        "detect": DETECTION_RL_CONFIG,
        "segment": SEGMENTATION_RL_CONFIG,
        "classify": CLASSIFICATION_RL_CONFIG,
        "pose": POSE_RL_CONFIG,
        "obb": OBB_RL_CONFIG,
    }

    if task not in default_configs:
        raise ValueError(f"Unknown task: {task}. Supported: {list(default_configs.keys())}")

    # Start with default config for task
    config = default_configs[task]

    # Override with provided values
    config.enabled = enabled
    config.algorithm = algorithm

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.grpo_config, key):
            setattr(config.grpo_config, key, value)

    return config
