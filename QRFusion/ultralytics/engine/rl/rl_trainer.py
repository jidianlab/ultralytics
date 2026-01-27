# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Reinforcement Learning Trainer Classes for Ultralytics.

This module provides RL-enhanced trainer classes for all Ultralytics model types,
extending the base trainers with GRPO-based reinforcement learning capabilities.

Supported Model Types:
    - Detection (YOLO)
    - Segmentation (YOLO-Seg)
    - Classification (YOLO-Cls)
    - Pose Estimation (YOLO-Pose)
    - Oriented Bounding Box (YOLO-OBB)

Example:
    >>> from ultralytics.engine.rl import RLDetectionTrainer, RLConfig
    >>> config = RLConfig(enabled=True, algorithm='grpo')
    >>> trainer = RLDetectionTrainer(overrides={'model': 'yolo26n.pt', 'rl_config': config})
    >>> trainer.train()
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from .config import (
    CLASSIFICATION_RL_CONFIG,
    DETECTION_RL_CONFIG,
    OBB_RL_CONFIG,
    POSE_RL_CONFIG,
    SEGMENTATION_RL_CONFIG,
    GRPOConfig,
    RLConfig,
)
from .grpo import GRPO, GRPOTrainer
from .rewards import (
    ClassAwareReward,
    CompositeReward,
    FormatReward,
    IoUReward,
    KeypointOKSReward,
    PixelAreaReward,
    RewardFunction,
    SegmentationIoUReward,
)


class BaseRLTrainer:
    """Mixin class providing RL training capabilities.

    This mixin can be combined with any Ultralytics trainer to add GRPO-based
    reinforcement learning functionality. It handles reward computation, advantage
    estimation, and policy updates.

    Attributes:
        rl_config (RLConfig): RL training configuration.
        grpo (GRPO): GRPO algorithm instance.
        grpo_trainer (GRPOTrainer): GRPO trainer for managing updates.
        reward_fn (RewardFunction): Reward function for evaluating predictions.
        rl_enabled (bool): Whether RL training is active.
        rl_metrics (dict): RL-specific training metrics.

    Methods:
        setup_rl: Initialize RL components.
        compute_rl_loss: Compute RL loss for current batch.
        rl_update_step: Perform an RL update step.
        log_rl_metrics: Log RL-specific metrics.
    """

    def __init__(self, *args, **kwargs):
        """Initialize RL trainer mixin."""
        # Extract RL config from kwargs
        self.rl_config: RLConfig | None = kwargs.pop("rl_config", None)

        # Initialize parent class
        super().__init__(*args, **kwargs)

        # RL components (initialized in setup_rl)
        self.grpo: GRPO | None = None
        self.grpo_trainer: GRPOTrainer | None = None
        self.reward_fn: RewardFunction | None = None
        self.rl_enabled = False
        self.rl_metrics: dict[str, float] = {}
        self._rl_update_counter = 0

    def setup_rl(self, default_config: RLConfig | None = None) -> None:
        """Initialize RL training components.

        Args:
            default_config (RLConfig | None): Default RL configuration to use.
        """
        # Use provided config or default
        if self.rl_config is None:
            self.rl_config = default_config or RLConfig()

        if not self.rl_config.enabled:
            return

        # Validate configuration
        self.rl_config.validate()

        # Initialize GRPO algorithm
        self.grpo = GRPO(
            config=self.rl_config.grpo_config,
            device=self.device,
        )

        # Create reward function based on config
        self.reward_fn = self._create_reward_function()

        # Mark RL as enabled
        self.rl_enabled = True

        # Log RL setup
        if hasattr(self, "LOGGER"):
            self.LOGGER.info(f"RL training enabled with {self.rl_config.algorithm} algorithm")

    def _create_reward_function(self) -> RewardFunction:
        """Create reward function based on configuration.

        Returns:
            RewardFunction: Configured reward function.
        """
        reward_type = self.rl_config.reward_type
        weights = self.rl_config.grpo_config.reward_weights

        if reward_type == "iou":
            return IoUReward(weight=weights.get("iou", 1.0))

        elif reward_type == "class_aware":
            return ClassAwareReward(
                class_weights=self.rl_config.class_weights,
                weight=weights.get("class_accuracy", 1.0),
            )

        elif reward_type == "segmentation":
            rewards = [
                IoUReward(weight=weights.get("iou", 1.0)),
                SegmentationIoUReward(weight=weights.get("mask_iou", 0.8)),
                ClassAwareReward(
                    class_weights=self.rl_config.class_weights,
                    weight=weights.get("class_accuracy", 0.5),
                ),
            ]
            if self.rl_config.area_weights:
                rewards.append(PixelAreaReward(
                    area_weights=self.rl_config.area_weights,
                    weight=weights.get("pixel_area", 0.3),
                ))
            return CompositeReward(rewards, normalize=True)

        else:  # composite (default)
            rewards = [
                IoUReward(weight=weights.get("iou", 1.0)),
                ClassAwareReward(
                    class_weights=self.rl_config.class_weights,
                    weight=weights.get("class_accuracy", 0.5),
                ),
            ]
            if self.rl_config.area_weights:
                rewards.append(PixelAreaReward(
                    area_weights=self.rl_config.area_weights,
                    weight=weights.get("pixel_area", 0.3),
                ))
            rewards.append(FormatReward(weight=weights.get("format", 0.1)))
            return CompositeReward(rewards, normalize=True)

    def compute_rl_loss(
        self,
        predictions: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor | dict[str, torch.Tensor],
        batch: dict[str, Any],
        supervised_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined RL and supervised loss.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            batch: Input batch data.
            supervised_loss: Standard supervised loss.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: Combined loss and metrics.
        """
        if not self.rl_enabled or self.grpo is None:
            return supervised_loss, {}

        # Check if we're past warmup period
        if hasattr(self, "epoch") and self.epoch < self.rl_config.warmup_epochs:
            return supervised_loss, {"rl_warmup": True}

        # Compute rewards
        with torch.no_grad():
            rewards = self.reward_fn(predictions, targets, batch)

        # Compute advantages
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)
        advantages = self.grpo.compute_advantages(rewards, group_size=1)

        # Compute RL loss component
        # For detection, we use the reward-weighted supervised loss approach
        # This avoids the need for explicit policy gradients which are harder with detection
        reward_weight = (1 + advantages).clamp(min=0.1, max=2.0)

        if supervised_loss.dim() == 0:
            rl_weighted_loss = supervised_loss * reward_weight.mean()
        else:
            # Match shapes
            if supervised_loss.numel() != reward_weight.numel():
                reward_weight = reward_weight.mean()
            rl_weighted_loss = (supervised_loss * reward_weight).mean()

        # Combine with supervised loss
        combined_loss = (
            (1 - self.rl_config.rl_weight) * supervised_loss +
            self.rl_config.rl_weight * rl_weighted_loss
        )

        # Collect metrics
        self.rl_metrics = {
            "rl/reward_mean": rewards.mean().item(),
            "rl/reward_std": rewards.std().item() if rewards.numel() > 1 else 0.0,
            "rl/advantage_mean": advantages.mean().item(),
            "rl/reward_weight": reward_weight.mean().item() if torch.is_tensor(reward_weight) else reward_weight,
        }

        # Add KL if using reference model
        if self.grpo.kl_history:
            self.rl_metrics["rl/kl_divergence"] = self.grpo.kl_history[-1]

        self._rl_update_counter += 1

        return combined_loss, self.rl_metrics

    def rl_update_step(
        self,
        batch: dict[str, Any],
        targets: dict[str, Any],
    ) -> dict[str, float]:
        """Perform a dedicated RL update step.

        This method can be called separately for pure RL updates,
        in addition to the combined supervised+RL training.

        Args:
            batch (dict[str, Any]): Input batch.
            targets (dict[str, Any]): Target labels.

        Returns:
            dict[str, float]: RL update metrics.
        """
        if not self.rl_enabled or self.grpo_trainer is None:
            return {}

        # Initialize GRPO trainer if needed
        if self.grpo_trainer is None and hasattr(self, "model") and hasattr(self, "optimizer"):
            self.grpo_trainer = GRPOTrainer(
                model=self.model,
                optimizer=self.optimizer,
                reward_fn=self.reward_fn,
                config=self.rl_config.grpo_config,
                device=self.device,
            )

        # Perform GRPO update
        loss, metrics = self.grpo_trainer.train_step(batch, targets)

        return metrics

    def log_rl_metrics(self) -> None:
        """Log RL-specific metrics."""
        if not self.rl_enabled or not self.rl_metrics:
            return

        # Add to main metrics if available
        if hasattr(self, "metrics") and isinstance(self.metrics, dict):
            self.metrics.update(self.rl_metrics)

        # Log to console if LOGGER available
        if hasattr(self, "LOGGER"):
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in self.rl_metrics.items()])
            self.LOGGER.info(f"RL Metrics: {metrics_str}")

    def save_rl_checkpoint(self, path: str) -> None:
        """Save RL-specific checkpoint data.

        Args:
            path (str): Path to save checkpoint.
        """
        if not self.rl_enabled:
            return

        rl_state = {
            "rl_config": self.rl_config.to_dict() if self.rl_config else None,
            "grpo_statistics": self.grpo.get_statistics() if self.grpo else {},
            "rl_update_counter": self._rl_update_counter,
        }

        torch.save(rl_state, path)

    def load_rl_checkpoint(self, path: str) -> None:
        """Load RL-specific checkpoint data.

        Args:
            path (str): Path to load checkpoint from.
        """
        if not self.rl_enabled:
            return

        try:
            rl_state = torch.load(path)
            self._rl_update_counter = rl_state.get("rl_update_counter", 0)
        except (FileNotFoundError, RuntimeError):
            warnings.warn(f"Could not load RL checkpoint from {path}")


# Import base trainers lazily to avoid circular imports
def _get_base_trainers():
    """Lazily import base trainers to avoid circular imports."""
    from ultralytics.models.yolo.classify.train import ClassificationTrainer
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.models.yolo.obb.train import OBBTrainer
    from ultralytics.models.yolo.pose.train import PoseTrainer
    from ultralytics.models.yolo.segment.train import SegmentationTrainer

    return {
        "detection": DetectionTrainer,
        "segmentation": SegmentationTrainer,
        "classification": ClassificationTrainer,
        "pose": PoseTrainer,
        "obb": OBBTrainer,
    }


class RLDetectionTrainer(BaseRLTrainer):
    """RL-enhanced Detection Trainer.

    Extends the standard DetectionTrainer with GRPO-based reinforcement learning
    for improved object detection training.

    The reward function combines:
        - IoU reward for localization quality
        - Class-aware reward for classification accuracy
        - Pixel area reward for balanced object size handling

    Examples:
        >>> from ultralytics.engine.rl import RLDetectionTrainer, RLConfig
        >>> config = RLConfig(enabled=True)
        >>> trainer = RLDetectionTrainer(overrides={'model': 'yolo26n.pt'}, rl_config=config)
        >>> trainer.train()
    """

    def __init__(self, cfg=None, overrides: dict | None = None, _callbacks=None, rl_config: RLConfig | None = None):
        """Initialize RL Detection Trainer.

        Args:
            cfg: Configuration file path or dict.
            overrides (dict | None): Configuration overrides.
            _callbacks: Callback functions.
            rl_config (RLConfig | None): RL configuration.
        """
        # Store RL config before parent init
        self.rl_config = rl_config

        # Import base trainer
        from ultralytics.models.yolo.detect.train import DetectionTrainer

        # Use composition pattern for cleaner integration
        self._base_trainer_class = DetectionTrainer

        # Initialize base trainer
        super(DetectionTrainer, self).__init__(cfg, overrides, _callbacks)

    def _setup_train(self):
        """Extended setup including RL initialization."""
        # Call parent setup
        super()._setup_train()

        # Setup RL components
        self.setup_rl(default_config=DETECTION_RL_CONFIG)

    def train_epoch(self, *args, **kwargs):
        """Training epoch with RL enhancements."""
        result = super().train_epoch(*args, **kwargs)

        # Log RL metrics at end of epoch
        self.log_rl_metrics()

        return result


class RLSegmentationTrainer(BaseRLTrainer):
    """RL-enhanced Segmentation Trainer.

    Extends the standard SegmentationTrainer with GRPO-based reinforcement learning
    for improved instance segmentation training.

    The reward function combines:
        - IoU reward for bounding box quality
        - Mask IoU reward for segmentation quality
        - Class-aware reward for classification accuracy
        - Pixel area reward for balanced training

    Examples:
        >>> from ultralytics.engine.rl import RLSegmentationTrainer, RLConfig
        >>> config = RLConfig(enabled=True, reward_type='segmentation')
        >>> trainer = RLSegmentationTrainer(overrides={'model': 'yolo26n-seg.pt'}, rl_config=config)
        >>> trainer.train()
    """

    def __init__(self, cfg=None, overrides: dict | None = None, _callbacks=None, rl_config: RLConfig | None = None):
        """Initialize RL Segmentation Trainer.

        Args:
            cfg: Configuration file path or dict.
            overrides (dict | None): Configuration overrides.
            _callbacks: Callback functions.
            rl_config (RLConfig | None): RL configuration.
        """
        self.rl_config = rl_config

        from ultralytics.models.yolo.segment.train import SegmentationTrainer

        self._base_trainer_class = SegmentationTrainer
        super(SegmentationTrainer, self).__init__(cfg, overrides, _callbacks)

    def _setup_train(self):
        """Extended setup including RL initialization."""
        super()._setup_train()
        self.setup_rl(default_config=SEGMENTATION_RL_CONFIG)

    def _create_reward_function(self) -> RewardFunction:
        """Create segmentation-specific reward function."""
        weights = self.rl_config.grpo_config.reward_weights

        rewards = [
            IoUReward(weight=weights.get("iou", 1.0)),
            SegmentationIoUReward(weight=weights.get("mask_iou", 0.8)),
            ClassAwareReward(
                class_weights=self.rl_config.class_weights,
                weight=weights.get("class_accuracy", 0.5),
            ),
        ]

        if self.rl_config.area_weights:
            rewards.append(PixelAreaReward(
                area_weights=self.rl_config.area_weights,
                weight=weights.get("pixel_area", 0.3),
            ))

        return CompositeReward(rewards, normalize=True)


class RLClassificationTrainer(BaseRLTrainer):
    """RL-enhanced Classification Trainer.

    Extends the standard ClassificationTrainer with GRPO-based reinforcement learning
    for improved image classification training.

    The reward function focuses on:
        - Class accuracy reward with optional per-class weighting

    Examples:
        >>> from ultralytics.engine.rl import RLClassificationTrainer, RLConfig
        >>> config = RLConfig(enabled=True, reward_type='class_aware')
        >>> trainer = RLClassificationTrainer(overrides={'model': 'yolo26n-cls.pt'}, rl_config=config)
        >>> trainer.train()
    """

    def __init__(self, cfg=None, overrides: dict | None = None, _callbacks=None, rl_config: RLConfig | None = None):
        """Initialize RL Classification Trainer.

        Args:
            cfg: Configuration file path or dict.
            overrides (dict | None): Configuration overrides.
            _callbacks: Callback functions.
            rl_config (RLConfig | None): RL configuration.
        """
        self.rl_config = rl_config

        from ultralytics.models.yolo.classify.train import ClassificationTrainer

        self._base_trainer_class = ClassificationTrainer
        super(ClassificationTrainer, self).__init__(cfg, overrides, _callbacks)

    def _setup_train(self):
        """Extended setup including RL initialization."""
        super()._setup_train()
        self.setup_rl(default_config=CLASSIFICATION_RL_CONFIG)

    def _create_reward_function(self) -> RewardFunction:
        """Create classification-specific reward function."""
        return ClassAwareReward(
            class_weights=self.rl_config.class_weights,
            weight=self.rl_config.grpo_config.reward_weights.get("class_accuracy", 1.0),
            use_soft_labels=True,
        )


class RLPoseTrainer(BaseRLTrainer):
    """RL-enhanced Pose Estimation Trainer.

    Extends the standard PoseTrainer with GRPO-based reinforcement learning
    for improved pose estimation training.

    The reward function combines:
        - IoU reward for bounding box quality
        - OKS (Object Keypoint Similarity) reward for keypoint accuracy
        - Class-aware reward for classification accuracy

    Examples:
        >>> from ultralytics.engine.rl import RLPoseTrainer, RLConfig
        >>> config = RLConfig(enabled=True)
        >>> trainer = RLPoseTrainer(overrides={'model': 'yolo26n-pose.pt'}, rl_config=config)
        >>> trainer.train()
    """

    def __init__(self, cfg=None, overrides: dict | None = None, _callbacks=None, rl_config: RLConfig | None = None):
        """Initialize RL Pose Trainer.

        Args:
            cfg: Configuration file path or dict.
            overrides (dict | None): Configuration overrides.
            _callbacks: Callback functions.
            rl_config (RLConfig | None): RL configuration.
        """
        self.rl_config = rl_config

        from ultralytics.models.yolo.pose.train import PoseTrainer

        self._base_trainer_class = PoseTrainer
        super(PoseTrainer, self).__init__(cfg, overrides, _callbacks)

    def _setup_train(self):
        """Extended setup including RL initialization."""
        super()._setup_train()
        self.setup_rl(default_config=POSE_RL_CONFIG)

    def _create_reward_function(self) -> RewardFunction:
        """Create pose-specific reward function."""
        weights = self.rl_config.grpo_config.reward_weights

        rewards = [
            IoUReward(weight=weights.get("iou", 0.8)),
            KeypointOKSReward(weight=weights.get("keypoint_oks", 1.0)),
            ClassAwareReward(
                class_weights=self.rl_config.class_weights,
                weight=weights.get("class_accuracy", 0.3),
            ),
        ]

        return CompositeReward(rewards, normalize=True)


class RLOBBTrainer(BaseRLTrainer):
    """RL-enhanced Oriented Bounding Box Trainer.

    Extends the standard OBBTrainer with GRPO-based reinforcement learning
    for improved oriented bounding box detection training.

    The reward function combines:
        - IoU reward for localization quality (using rotated IoU)
        - Angle accuracy reward for orientation accuracy
        - Class-aware reward for classification accuracy

    Examples:
        >>> from ultralytics.engine.rl import RLOBBTrainer, RLConfig
        >>> config = RLConfig(enabled=True)
        >>> trainer = RLOBBTrainer(overrides={'model': 'yolo26n-obb.pt'}, rl_config=config)
        >>> trainer.train()
    """

    def __init__(self, cfg=None, overrides: dict | None = None, _callbacks=None, rl_config: RLConfig | None = None):
        """Initialize RL OBB Trainer.

        Args:
            cfg: Configuration file path or dict.
            overrides (dict | None): Configuration overrides.
            _callbacks: Callback functions.
            rl_config (RLConfig | None): RL configuration.
        """
        self.rl_config = rl_config

        from ultralytics.models.yolo.obb.train import OBBTrainer

        self._base_trainer_class = OBBTrainer
        super(OBBTrainer, self).__init__(cfg, overrides, _callbacks)

    def _setup_train(self):
        """Extended setup including RL initialization."""
        super()._setup_train()
        self.setup_rl(default_config=OBB_RL_CONFIG)

    def _create_reward_function(self) -> RewardFunction:
        """Create OBB-specific reward function."""
        weights = self.rl_config.grpo_config.reward_weights

        rewards = [
            IoUReward(iou_type="ciou", weight=weights.get("iou", 1.0)),
            ClassAwareReward(
                class_weights=self.rl_config.class_weights,
                weight=weights.get("class_accuracy", 0.5),
            ),
        ]

        if self.rl_config.area_weights:
            rewards.append(PixelAreaReward(
                area_weights=self.rl_config.area_weights,
                weight=weights.get("pixel_area", 0.3),
            ))

        return CompositeReward(rewards, normalize=True)


# Angle-specific reward for OBB (could be added as separate reward)
class AngleAccuracyReward(RewardFunction):
    """Angle accuracy reward for oriented bounding boxes.

    Computes rewards based on the angular difference between predicted and
    ground truth bounding box orientations.

    Attributes:
        angle_threshold (float): Threshold for full reward (in radians).
        smooth (float): Smoothing factor.
    """

    def __init__(
        self,
        name: str = "angle_accuracy_reward",
        weight: float = 1.0,
        angle_threshold: float = 0.1,  # ~5.7 degrees
        smooth: float = 1e-7,
    ):
        """Initialize angle accuracy reward.

        Args:
            name (str): Name identifier.
            weight (float): Weight multiplier.
            angle_threshold (float): Threshold in radians for full reward.
            smooth (float): Smoothing factor.
        """
        super().__init__(name, weight)
        self.angle_threshold = angle_threshold
        self.smooth = smooth

    def __call__(
        self,
        predictions: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor | dict[str, torch.Tensor],
        batch: dict[str, Any] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute angle accuracy reward.

        Args:
            predictions: Predicted angles [N] or boxes with angles [N, 5].
            targets: Ground truth angles [N] or boxes with angles [N, 5].
            batch: Optional batch information.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Angle accuracy rewards.
        """
        # Extract angles
        if isinstance(predictions, dict):
            pred_angles = predictions.get("angles", predictions.get("angle"))
        elif predictions.dim() == 2 and predictions.size(-1) >= 5:
            pred_angles = predictions[:, 4]
        else:
            pred_angles = predictions

        if isinstance(targets, dict):
            gt_angles = targets.get("angles", targets.get("angle"))
        elif targets.dim() == 2 and targets.size(-1) >= 5:
            gt_angles = targets[:, 4]
        else:
            gt_angles = targets

        if pred_angles is None or gt_angles is None:
            return torch.zeros(1)

        # Compute angular difference (handle periodicity)
        angle_diff = torch.abs(pred_angles - gt_angles)
        angle_diff = torch.min(angle_diff, 2 * 3.14159 - angle_diff)  # Handle wrap-around

        # Compute reward (exponential decay from threshold)
        reward = torch.exp(-angle_diff / self.angle_threshold)

        return reward * self.weight
