# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Reinforcement Learning Configuration Module.

This module provides configuration classes for reinforcement learning training in Ultralytics.
The configuration is designed following VLM-R1's GRPO approach with additional features for
object detection specific use cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GRPOConfig:
    """Configuration for Generative Reward Policy Optimization (GRPO) training.

    This class holds all hyperparameters needed for GRPO-based reinforcement learning training,
    inspired by VLM-R1's implementation.

    Attributes:
        num_generations (int): Number of policy generations per training sample.
        kl_coeff (float): KL divergence coefficient for regularization.
        clip_ratio (float): PPO-style clipping ratio for policy updates.
        gamma (float): Discount factor for future rewards.
        gae_lambda (float): GAE lambda for advantage estimation.
        value_loss_coeff (float): Coefficient for value function loss.
        entropy_coeff (float): Coefficient for entropy bonus.
        max_grad_norm (float): Maximum gradient norm for clipping.
        normalize_advantage (bool): Whether to normalize advantages.
        use_reference_model (bool): Whether to use a reference model for KL computation.
        reward_weights (dict): Weights for different reward components.
        temperature (float): Temperature for softmax in policy sampling.
        top_k (int): Top-k sampling parameter (0 to disable).
        top_p (float): Top-p (nucleus) sampling parameter (1.0 to disable).

    Examples:
        >>> config = GRPOConfig(num_generations=4, kl_coeff=0.1)
        >>> print(config.num_generations)
        4
    """

    # Core GRPO parameters
    num_generations: int = 4
    kl_coeff: float = 0.1
    clip_ratio: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Loss coefficients
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5

    # Advantage processing
    normalize_advantage: bool = True
    use_reference_model: bool = True

    # Reward configuration
    reward_weights: dict[str, float] = field(default_factory=lambda: {
        "iou": 1.0,
        "class_accuracy": 0.5,
        "format": 0.1,
        "pixel_area": 0.3,
    })

    # Sampling parameters
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    # Update frequency
    ppo_epochs: int = 4
    mini_batch_size: int = 8

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            dict[str, Any]: Configuration as dictionary.
        """
        return {
            "num_generations": self.num_generations,
            "kl_coeff": self.kl_coeff,
            "clip_ratio": self.clip_ratio,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "value_loss_coeff": self.value_loss_coeff,
            "entropy_coeff": self.entropy_coeff,
            "max_grad_norm": self.max_grad_norm,
            "normalize_advantage": self.normalize_advantage,
            "use_reference_model": self.use_reference_model,
            "reward_weights": self.reward_weights,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "ppo_epochs": self.ppo_epochs,
            "mini_batch_size": self.mini_batch_size,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "GRPOConfig":
        """Create configuration from dictionary.

        Args:
            config_dict (dict[str, Any]): Configuration dictionary.

        Returns:
            GRPOConfig: Configuration object.
        """
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class RLConfig:
    """General configuration for reinforcement learning training.

    This class provides high-level RL training settings that work across different algorithms
    and model types in the Ultralytics framework.

    Attributes:
        enabled (bool): Whether RL training is enabled.
        algorithm (str): RL algorithm to use ('grpo', 'ppo', 'a2c').
        grpo_config (GRPOConfig): GRPO-specific configuration.
        reward_type (str): Type of reward function ('iou', 'class_aware', 'composite').
        update_frequency (int): How often to perform RL updates (in batches).
        warmup_epochs (int): Number of epochs of supervised training before RL.
        rl_weight (float): Weight for RL loss relative to supervised loss.
        save_rl_checkpoints (bool): Whether to save RL-specific checkpoints.
        log_rewards (bool): Whether to log reward statistics.
        class_weights (dict | None): Per-class reward weights.
        area_weights (dict | None): Weights based on object pixel area.

    Examples:
        >>> rl_config = RLConfig(enabled=True, algorithm='grpo')
        >>> print(rl_config.enabled)
        True
    """

    enabled: bool = False
    algorithm: str = "grpo"
    grpo_config: GRPOConfig = field(default_factory=GRPOConfig)
    reward_type: str = "composite"
    update_frequency: int = 1
    warmup_epochs: int = 0
    rl_weight: float = 0.5
    save_rl_checkpoints: bool = True
    log_rewards: bool = True
    class_weights: dict[int, float] | None = None
    area_weights: dict[str, float] | None = field(default_factory=lambda: {
        "small": 1.5,    # Objects with area < 32^2 pixels
        "medium": 1.0,   # Objects with area 32^2 - 96^2 pixels
        "large": 0.8,    # Objects with area > 96^2 pixels
    })

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        if self.algorithm not in {"grpo", "ppo", "a2c"}:
            raise ValueError(f"Invalid RL algorithm: {self.algorithm}. Must be 'grpo', 'ppo', or 'a2c'.")

        if self.reward_type not in {"iou", "class_aware", "composite", "segmentation"}:
            raise ValueError(f"Invalid reward type: {self.reward_type}")

        if not 0.0 <= self.rl_weight <= 1.0:
            raise ValueError(f"rl_weight must be between 0 and 1, got {self.rl_weight}")

        if self.update_frequency < 1:
            raise ValueError(f"update_frequency must be >= 1, got {self.update_frequency}")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            dict[str, Any]: Configuration as dictionary.
        """
        return {
            "enabled": self.enabled,
            "algorithm": self.algorithm,
            "grpo_config": self.grpo_config.to_dict(),
            "reward_type": self.reward_type,
            "update_frequency": self.update_frequency,
            "warmup_epochs": self.warmup_epochs,
            "rl_weight": self.rl_weight,
            "save_rl_checkpoints": self.save_rl_checkpoints,
            "log_rewards": self.log_rewards,
            "class_weights": self.class_weights,
            "area_weights": self.area_weights,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "RLConfig":
        """Create configuration from dictionary.

        Args:
            config_dict (dict[str, Any]): Configuration dictionary.

        Returns:
            RLConfig: Configuration object.
        """
        grpo_dict = config_dict.pop("grpo_config", {})
        grpo_config = GRPOConfig.from_dict(grpo_dict) if grpo_dict else GRPOConfig()
        return cls(grpo_config=grpo_config, **{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# Default configurations for different tasks
DETECTION_RL_CONFIG = RLConfig(
    enabled=True,
    algorithm="grpo",
    reward_type="composite",
    grpo_config=GRPOConfig(
        num_generations=4,
        kl_coeff=0.1,
        reward_weights={"iou": 1.0, "class_accuracy": 0.5, "pixel_area": 0.3},
    ),
)

SEGMENTATION_RL_CONFIG = RLConfig(
    enabled=True,
    algorithm="grpo",
    reward_type="segmentation",
    grpo_config=GRPOConfig(
        num_generations=4,
        kl_coeff=0.1,
        reward_weights={"iou": 1.0, "mask_iou": 0.8, "class_accuracy": 0.5, "pixel_area": 0.3},
    ),
)

CLASSIFICATION_RL_CONFIG = RLConfig(
    enabled=True,
    algorithm="grpo",
    reward_type="class_aware",
    grpo_config=GRPOConfig(
        num_generations=4,
        kl_coeff=0.1,
        reward_weights={"class_accuracy": 1.0},
    ),
)

POSE_RL_CONFIG = RLConfig(
    enabled=True,
    algorithm="grpo",
    reward_type="composite",
    grpo_config=GRPOConfig(
        num_generations=4,
        kl_coeff=0.1,
        reward_weights={"iou": 0.8, "keypoint_oks": 1.0, "class_accuracy": 0.3},
    ),
)

OBB_RL_CONFIG = RLConfig(
    enabled=True,
    algorithm="grpo",
    reward_type="composite",
    grpo_config=GRPOConfig(
        num_generations=4,
        kl_coeff=0.1,
        reward_weights={"iou": 1.0, "angle_accuracy": 0.5, "class_accuracy": 0.5},
    ),
)
