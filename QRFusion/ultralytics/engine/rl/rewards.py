# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Reward Functions for Reinforcement Learning Training.

This module provides reward functions for RL-based training in Ultralytics, inspired by VLM-R1's
approach. The reward functions are designed to work with IoU metrics and object class/area information
for object detection, segmentation, pose estimation, and other vision tasks.

Key Features:
    - IoU-based rewards for localization quality
    - Class-aware rewards for classification accuracy
    - Pixel area-based rewards for balanced training
    - Segmentation-specific mask IoU rewards
    - Pose-specific keypoint OKS rewards
    - Composite rewards combining multiple signals

Example:
    >>> from ultralytics.engine.rl.rewards import IoUReward, ClassAwareReward, CompositeReward
    >>> iou_reward = IoUReward()
    >>> class_reward = ClassAwareReward()
    >>> composite = CompositeReward([iou_reward, class_reward], weights=[1.0, 0.5])
    >>> reward = composite(predictions, targets, batch)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn.functional as F


class RewardFunction(ABC):
    """Abstract base class for reward functions.

    All reward functions should inherit from this class and implement the __call__ method.
    This design follows the VLM-R1 modular reward architecture.

    Attributes:
        name (str): Name identifier for the reward function.
        weight (float): Weight multiplier for this reward in composite rewards.

    Methods:
        __call__: Compute the reward given predictions and targets.
        reset: Reset any internal state (optional).
    """

    def __init__(self, name: str = "base_reward", weight: float = 1.0):
        """Initialize reward function.

        Args:
            name (str): Name identifier for the reward function.
            weight (float): Weight multiplier for this reward.
        """
        self.name = name
        self.weight = weight

    @abstractmethod
    def __call__(
        self,
        predictions: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor | dict[str, torch.Tensor],
        batch: dict[str, Any] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute reward.

        Args:
            predictions: Model predictions (format depends on task).
            targets: Ground truth targets.
            batch: Optional batch information for additional context.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Reward values (higher is better).
        """
        pass

    def reset(self) -> None:
        """Reset any internal state. Override if needed."""
        pass


class RewardRegistry:
    """Registry for reward functions.

    This class provides a centralized registry for reward functions, following VLM-R1's
    task-specific reward registration pattern.

    Attributes:
        _rewards (dict): Dictionary mapping reward names to reward functions.

    Examples:
        >>> registry = RewardRegistry()
        >>> registry.register("iou", IoUReward())
        >>> iou_reward = registry.get("iou")
    """

    _rewards: dict[str, RewardFunction] = {}

    @classmethod
    def register(cls, name: str, reward: RewardFunction) -> None:
        """Register a reward function.

        Args:
            name (str): Name to register the reward under.
            reward (RewardFunction): Reward function instance.
        """
        cls._rewards[name] = reward

    @classmethod
    def get(cls, name: str) -> RewardFunction:
        """Get a registered reward function.

        Args:
            name (str): Name of the reward function.

        Returns:
            RewardFunction: The registered reward function.

        Raises:
            KeyError: If reward is not registered.
        """
        if name not in cls._rewards:
            raise KeyError(f"Reward '{name}' not registered. Available: {list(cls._rewards.keys())}")
        return cls._rewards[name]

    @classmethod
    def list_rewards(cls) -> list[str]:
        """List all registered rewards.

        Returns:
            list[str]: Names of registered rewards.
        """
        return list(cls._rewards.keys())


class IoUReward(RewardFunction):
    """IoU-based reward function for object detection.

    Computes rewards based on Intersection over Union (IoU) between predicted and ground truth
    bounding boxes. This is the primary reward signal for localization quality, following
    VLM-R1's approach.

    Attributes:
        iou_type (str): Type of IoU to compute ('iou', 'giou', 'diou', 'ciou').
        smooth (float): Smoothing factor to avoid division by zero.
        normalize (bool): Whether to normalize rewards to [0, 1].

    Examples:
        >>> reward_fn = IoUReward(iou_type='ciou')
        >>> reward = reward_fn(pred_boxes, gt_boxes)
    """

    def __init__(
        self,
        name: str = "iou_reward",
        weight: float = 1.0,
        iou_type: str = "ciou",
        smooth: float = 1e-7,
        normalize: bool = True,
    ):
        """Initialize IoU reward function.

        Args:
            name (str): Name identifier.
            weight (float): Weight multiplier.
            iou_type (str): Type of IoU computation.
            smooth (float): Smoothing factor.
            normalize (bool): Whether to normalize rewards.
        """
        super().__init__(name, weight)
        self.iou_type = iou_type
        self.smooth = smooth
        self.normalize = normalize

    def __call__(
        self,
        predictions: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor | dict[str, torch.Tensor],
        batch: dict[str, Any] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute IoU-based reward.

        Args:
            predictions: Predicted bounding boxes [N, 4] in xyxy format or dict with 'boxes' key.
            targets: Ground truth boxes [N, 4] in xyxy format or dict with 'boxes' key.
            batch: Optional batch information.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: IoU rewards for each prediction.
        """
        # Handle dict inputs
        if isinstance(predictions, dict):
            pred_boxes = predictions.get("boxes", predictions.get("bboxes"))
        else:
            pred_boxes = predictions

        if isinstance(targets, dict):
            gt_boxes = targets.get("boxes", targets.get("bboxes"))
        else:
            gt_boxes = targets

        if pred_boxes is None or gt_boxes is None:
            return torch.zeros(1, device=predictions.device if torch.is_tensor(predictions) else "cpu")

        # Compute IoU
        iou = self._compute_iou(pred_boxes, gt_boxes)

        # Apply weight
        reward = iou * self.weight

        return reward

    def _compute_iou(
        self,
        box1: torch.Tensor,
        box2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU between two sets of boxes.

        Args:
            box1: Predicted boxes [N, 4] in xyxy format.
            box2: Ground truth boxes [N, 4] in xyxy format.

        Returns:
            torch.Tensor: IoU values [N].
        """
        # Ensure same device
        if box1.device != box2.device:
            box2 = box2.to(box1.device)

        # Get box coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

        # Intersection area
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area + self.smooth

        # Basic IoU
        iou = inter_area / union_area

        if self.iou_type == "iou":
            return iou

        # Enclosing box
        c_x1 = torch.min(b1_x1, b2_x1)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y2 = torch.max(b1_y2, b2_y2)

        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + self.smooth

        if self.iou_type == "giou":
            return iou - (c_area - union_area) / c_area

        # Center distance
        b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
        center_dist_sq = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

        # Diagonal distance
        c_diag_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + self.smooth

        if self.iou_type == "diou":
            return iou - center_dist_sq / c_diag_sq

        # CIoU
        if self.iou_type == "ciou":
            v = (4 / (math.pi ** 2)) * torch.pow(
                torch.atan((b2_x2 - b2_x1) / (b2_y2 - b2_y1 + self.smooth))
                - torch.atan((b1_x2 - b1_x1) / (b1_y2 - b1_y1 + self.smooth)),
                2,
            )
            with torch.no_grad():
                alpha = v / (1 - iou + v + self.smooth)
            return iou - center_dist_sq / c_diag_sq - alpha * v

        return iou


class ClassAwareReward(RewardFunction):
    """Class-aware reward function.

    Computes rewards based on classification accuracy with optional per-class weighting.
    This allows emphasizing certain classes during training, following VLM-R1's approach
    of combining classification signals with localization.

    Attributes:
        class_weights (dict | None): Per-class weights for reward computation.
        smooth (float): Smoothing factor for numerical stability.
        use_soft_labels (bool): Whether to use soft label comparison.

    Examples:
        >>> reward_fn = ClassAwareReward(class_weights={0: 1.5, 1: 1.0, 2: 2.0})
        >>> reward = reward_fn(pred_classes, gt_classes)
    """

    def __init__(
        self,
        name: str = "class_reward",
        weight: float = 1.0,
        class_weights: dict[int, float] | None = None,
        smooth: float = 1e-7,
        use_soft_labels: bool = False,
    ):
        """Initialize class-aware reward function.

        Args:
            name (str): Name identifier.
            weight (float): Weight multiplier.
            class_weights (dict | None): Per-class weight dictionary.
            smooth (float): Smoothing factor.
            use_soft_labels (bool): Whether to use soft labels.
        """
        super().__init__(name, weight)
        self.class_weights = class_weights or {}
        self.smooth = smooth
        self.use_soft_labels = use_soft_labels

    def __call__(
        self,
        predictions: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor | dict[str, torch.Tensor],
        batch: dict[str, Any] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute class-aware reward.

        Args:
            predictions: Predicted class logits [N, C] or class indices [N].
            targets: Ground truth class indices [N] or one-hot [N, C].
            batch: Optional batch information.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Class-aware rewards for each prediction.
        """
        # Handle dict inputs
        if isinstance(predictions, dict):
            pred_cls = predictions.get("cls", predictions.get("classes"))
        else:
            pred_cls = predictions

        if isinstance(targets, dict):
            gt_cls = targets.get("cls", targets.get("classes"))
        else:
            gt_cls = targets

        if pred_cls is None or gt_cls is None:
            return torch.zeros(1, device=predictions.device if torch.is_tensor(predictions) else "cpu")

        # Convert logits to predictions if needed
        if pred_cls.dim() > 1 and pred_cls.size(-1) > 1:
            if self.use_soft_labels:
                pred_probs = F.softmax(pred_cls, dim=-1)
                # Soft accuracy
                if gt_cls.dim() == 1:
                    gt_onehot = F.one_hot(gt_cls.long(), pred_cls.size(-1)).float()
                else:
                    gt_onehot = gt_cls
                reward = (pred_probs * gt_onehot).sum(-1)
            else:
                pred_cls = pred_cls.argmax(-1)
                # Hard accuracy
                reward = (pred_cls == gt_cls.squeeze()).float()
        else:
            # Already class indices
            pred_cls = pred_cls.squeeze()
            gt_cls = gt_cls.squeeze()
            reward = (pred_cls == gt_cls).float()

        # Apply class-specific weights
        if self.class_weights and gt_cls.numel() > 0:
            gt_cls_np = gt_cls.long()
            weights = torch.ones_like(reward)
            for cls_id, cls_weight in self.class_weights.items():
                weights[gt_cls_np == cls_id] = cls_weight
            reward = reward * weights

        return reward * self.weight


class PixelAreaReward(RewardFunction):
    """Pixel area-based reward function.

    Computes rewards that vary based on object pixel area, encouraging balanced
    performance across objects of different sizes. This follows VLM-R1's approach
    of using pixel-level information for reward design.

    Attributes:
        area_thresholds (dict): Thresholds for small/medium/large objects.
        area_weights (dict): Weights for different size categories.

    Examples:
        >>> reward_fn = PixelAreaReward(area_weights={'small': 1.5, 'medium': 1.0, 'large': 0.8})
        >>> reward = reward_fn(predictions, targets, batch)
    """

    def __init__(
        self,
        name: str = "pixel_area_reward",
        weight: float = 1.0,
        area_thresholds: dict[str, float] | None = None,
        area_weights: dict[str, float] | None = None,
    ):
        """Initialize pixel area reward function.

        Args:
            name (str): Name identifier.
            weight (float): Weight multiplier.
            area_thresholds (dict | None): Size category thresholds (in pixels^2).
            area_weights (dict | None): Weights for each size category.
        """
        super().__init__(name, weight)
        self.area_thresholds = area_thresholds or {
            "small": 32 ** 2,    # < 1024 pixels
            "medium": 96 ** 2,   # 1024 - 9216 pixels
        }
        self.area_weights = area_weights or {
            "small": 1.5,
            "medium": 1.0,
            "large": 0.8,
        }

    def __call__(
        self,
        predictions: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor | dict[str, torch.Tensor],
        batch: dict[str, Any] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute pixel area-based reward weights.

        Args:
            predictions: Predicted bounding boxes [N, 4] in xyxy format.
            targets: Ground truth boxes [N, 4] in xyxy format.
            batch: Optional batch information.
            **kwargs: Additional arguments including base_reward.

        Returns:
            torch.Tensor: Area-weighted reward multipliers.
        """
        # Get base reward if provided
        base_reward = kwargs.get("base_reward", torch.ones(1))

        # Handle dict inputs
        if isinstance(targets, dict):
            gt_boxes = targets.get("boxes", targets.get("bboxes"))
        else:
            gt_boxes = targets

        if gt_boxes is None or gt_boxes.numel() == 0:
            return base_reward * self.weight

        # Compute box areas
        if gt_boxes.dim() == 2 and gt_boxes.size(-1) >= 4:
            x1, y1, x2, y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
        else:
            return base_reward * self.weight

        # Categorize by area and apply weights
        weights = torch.ones_like(areas)
        small_mask = areas < self.area_thresholds["small"]
        large_mask = areas > self.area_thresholds["medium"]
        medium_mask = ~small_mask & ~large_mask

        weights[small_mask] = self.area_weights["small"]
        weights[medium_mask] = self.area_weights["medium"]
        weights[large_mask] = self.area_weights["large"]

        # Apply to base reward
        if base_reward.shape == weights.shape:
            return base_reward * weights * self.weight
        else:
            return weights.mean() * self.weight


class SegmentationIoUReward(RewardFunction):
    """Segmentation mask IoU reward function.

    Computes rewards based on mask IoU for instance segmentation tasks.
    This extends the basic IoU reward to work with pixel-level masks.

    Attributes:
        smooth (float): Smoothing factor for numerical stability.
        per_class (bool): Whether to compute per-class IoU.

    Examples:
        >>> reward_fn = SegmentationIoUReward()
        >>> reward = reward_fn(pred_masks, gt_masks)
    """

    def __init__(
        self,
        name: str = "mask_iou_reward",
        weight: float = 1.0,
        smooth: float = 1e-7,
        per_class: bool = False,
    ):
        """Initialize segmentation IoU reward.

        Args:
            name (str): Name identifier.
            weight (float): Weight multiplier.
            smooth (float): Smoothing factor.
            per_class (bool): Whether to compute per-class IoU.
        """
        super().__init__(name, weight)
        self.smooth = smooth
        self.per_class = per_class

    def __call__(
        self,
        predictions: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor | dict[str, torch.Tensor],
        batch: dict[str, Any] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute mask IoU reward.

        Args:
            predictions: Predicted masks [N, H, W] or dict with 'masks' key.
            targets: Ground truth masks [N, H, W] or dict with 'masks' key.
            batch: Optional batch information.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Mask IoU rewards.
        """
        # Handle dict inputs
        if isinstance(predictions, dict):
            pred_masks = predictions.get("masks", predictions.get("mask"))
        else:
            pred_masks = predictions

        if isinstance(targets, dict):
            gt_masks = targets.get("masks", targets.get("mask"))
        else:
            gt_masks = targets

        if pred_masks is None or gt_masks is None:
            return torch.zeros(1, device=predictions.device if torch.is_tensor(predictions) else "cpu")

        # Ensure binary masks
        pred_masks = (pred_masks > 0.5).float()
        gt_masks = (gt_masks > 0.5).float()

        # Compute intersection and union
        intersection = (pred_masks * gt_masks).sum(dim=(-2, -1))
        union = pred_masks.sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1)) - intersection + self.smooth

        # Mask IoU
        mask_iou = intersection / union

        return mask_iou * self.weight


class KeypointOKSReward(RewardFunction):
    """Object Keypoint Similarity (OKS) reward for pose estimation.

    Computes rewards based on OKS metric, which measures the similarity between
    predicted and ground truth keypoints, normalized by object scale.

    Attributes:
        sigmas (torch.Tensor): Per-keypoint standard deviations.
        smooth (float): Smoothing factor.

    Examples:
        >>> reward_fn = KeypointOKSReward()
        >>> reward = reward_fn(pred_keypoints, gt_keypoints)
    """

    # COCO keypoint sigmas
    COCO_SIGMAS = torch.tensor([
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
        0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])

    def __init__(
        self,
        name: str = "keypoint_oks_reward",
        weight: float = 1.0,
        sigmas: torch.Tensor | None = None,
        smooth: float = 1e-7,
    ):
        """Initialize keypoint OKS reward.

        Args:
            name (str): Name identifier.
            weight (float): Weight multiplier.
            sigmas (torch.Tensor | None): Per-keypoint sigmas.
            smooth (float): Smoothing factor.
        """
        super().__init__(name, weight)
        self.sigmas = sigmas if sigmas is not None else self.COCO_SIGMAS
        self.smooth = smooth

    def __call__(
        self,
        predictions: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor | dict[str, torch.Tensor],
        batch: dict[str, Any] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute OKS reward.

        Args:
            predictions: Predicted keypoints [N, K, 2/3] (x, y, [visibility]).
            targets: Ground truth keypoints [N, K, 2/3].
            batch: Optional batch information containing scales.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: OKS rewards.
        """
        # Handle dict inputs
        if isinstance(predictions, dict):
            pred_kpts = predictions.get("keypoints", predictions.get("kpts"))
        else:
            pred_kpts = predictions

        if isinstance(targets, dict):
            gt_kpts = targets.get("keypoints", targets.get("kpts"))
        else:
            gt_kpts = targets

        if pred_kpts is None or gt_kpts is None:
            return torch.zeros(1, device=predictions.device if torch.is_tensor(predictions) else "cpu")

        # Get visibility if available
        if pred_kpts.size(-1) == 3:
            pred_xy = pred_kpts[..., :2]
            visibility = gt_kpts[..., 2] if gt_kpts.size(-1) == 3 else torch.ones(gt_kpts.shape[:-1])
        else:
            pred_xy = pred_kpts
            visibility = torch.ones(gt_kpts.shape[:-1], device=pred_kpts.device)

        gt_xy = gt_kpts[..., :2]

        # Compute squared distances
        dist_sq = ((pred_xy - gt_xy) ** 2).sum(-1)

        # Get scale (area) from batch or estimate
        if batch is not None and "area" in batch:
            scale = batch["area"]
        else:
            # Estimate scale from keypoint spread
            scale = (gt_xy.max(dim=-2).values - gt_xy.min(dim=-2).values).prod(-1) + self.smooth

        # Ensure sigmas on correct device and proper size
        sigmas = self.sigmas.to(pred_kpts.device)
        num_kpts = pred_xy.size(-2)
        if len(sigmas) < num_kpts:
            sigmas = sigmas.repeat(math.ceil(num_kpts / len(sigmas)))[:num_kpts]
        elif len(sigmas) > num_kpts:
            sigmas = sigmas[:num_kpts]

        # Compute OKS
        vars_sq = (2 * sigmas ** 2).unsqueeze(0)
        oks = torch.exp(-dist_sq / (2 * scale.unsqueeze(-1) * vars_sq + self.smooth))

        # Weight by visibility
        oks = (oks * visibility).sum(-1) / (visibility.sum(-1) + self.smooth)

        return oks * self.weight


class FormatReward(RewardFunction):
    """Format compliance reward function.

    Computes rewards based on output format compliance, ensuring predictions
    are in valid ranges and formats. This helps maintain output quality.

    Attributes:
        valid_range (tuple): Valid range for predictions.
        penalty (float): Penalty for invalid format.

    Examples:
        >>> reward_fn = FormatReward(valid_range=(0, 1))
        >>> reward = reward_fn(predictions, targets)
    """

    def __init__(
        self,
        name: str = "format_reward",
        weight: float = 1.0,
        valid_range: tuple[float, float] = (0.0, 1.0),
        penalty: float = -0.1,
    ):
        """Initialize format reward.

        Args:
            name (str): Name identifier.
            weight (float): Weight multiplier.
            valid_range (tuple): Valid value range.
            penalty (float): Penalty for invalid values.
        """
        super().__init__(name, weight)
        self.valid_range = valid_range
        self.penalty = penalty

    def __call__(
        self,
        predictions: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor | dict[str, torch.Tensor],
        batch: dict[str, Any] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute format compliance reward.

        Args:
            predictions: Model predictions.
            targets: Ground truth (unused but kept for interface consistency).
            batch: Optional batch information.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Format compliance rewards.
        """
        if isinstance(predictions, dict):
            # Check all prediction components
            rewards = []
            for key, pred in predictions.items():
                if torch.is_tensor(pred) and pred.numel() > 0:
                    valid = (pred >= self.valid_range[0]) & (pred <= self.valid_range[1])
                    reward = valid.float().mean()
                    rewards.append(reward)
            if rewards:
                return torch.stack(rewards).mean() * self.weight
            return torch.ones(1, device=next(iter(predictions.values())).device) * self.weight
        else:
            valid = (predictions >= self.valid_range[0]) & (predictions <= self.valid_range[1])
            return valid.float().mean() * self.weight


class CompositeReward(RewardFunction):
    """Composite reward combining multiple reward functions.

    This class allows combining multiple reward signals with configurable weights,
    following VLM-R1's modular reward architecture.

    Attributes:
        rewards (list[RewardFunction]): List of reward functions.
        weights (list[float]): Corresponding weights for each reward.
        normalize (bool): Whether to normalize the final reward.

    Examples:
        >>> iou_reward = IoUReward()
        >>> class_reward = ClassAwareReward()
        >>> composite = CompositeReward([iou_reward, class_reward], weights=[1.0, 0.5])
        >>> reward = composite(predictions, targets)
    """

    def __init__(
        self,
        rewards: list[RewardFunction],
        weights: list[float] | None = None,
        name: str = "composite_reward",
        normalize: bool = False,
    ):
        """Initialize composite reward.

        Args:
            rewards (list[RewardFunction]): Reward functions to combine.
            weights (list[float] | None): Weights for each reward (uses reward.weight if None).
            name (str): Name identifier.
            normalize (bool): Whether to normalize output.
        """
        super().__init__(name, 1.0)
        self.rewards = rewards
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [r.weight for r in rewards]
        self.normalize = normalize

    def __call__(
        self,
        predictions: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor | dict[str, torch.Tensor],
        batch: dict[str, Any] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute composite reward.

        Args:
            predictions: Model predictions.
            targets: Ground truth.
            batch: Optional batch information.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Combined reward.
        """
        total_reward = None
        total_weight = 0.0

        for reward_fn, weight in zip(self.rewards, self.weights):
            try:
                reward = reward_fn(predictions, targets, batch, **kwargs)
                if total_reward is None:
                    total_reward = reward * weight
                else:
                    # Handle shape broadcasting
                    if reward.shape != total_reward.shape:
                        if reward.numel() == 1:
                            reward = reward.expand_as(total_reward)
                        elif total_reward.numel() == 1:
                            total_reward = total_reward.expand_as(reward)
                    total_reward = total_reward + reward * weight
                total_weight += weight
            except (RuntimeError, KeyError):
                # Skip rewards that fail (e.g., missing keys)
                continue

        if total_reward is None:
            return torch.zeros(1)

        if self.normalize and total_weight > 0:
            total_reward = total_reward / total_weight

        return total_reward

    def reset(self) -> None:
        """Reset all component reward functions."""
        for reward in self.rewards:
            reward.reset()


# Register default rewards
def _register_default_rewards():
    """Register default reward functions in the global registry."""
    RewardRegistry.register("iou", IoUReward())
    RewardRegistry.register("class_aware", ClassAwareReward())
    RewardRegistry.register("pixel_area", PixelAreaReward())
    RewardRegistry.register("mask_iou", SegmentationIoUReward())
    RewardRegistry.register("keypoint_oks", KeypointOKSReward())
    RewardRegistry.register("format", FormatReward())


# Auto-register on import
_register_default_rewards()
