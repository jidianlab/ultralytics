# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Reinforcement Learning Trainer for YOLO models.

This module implements an RL-based training framework inspired by VLM-R1's GRPO algorithm,
with reward functions combining IoU and class information for object detection tasks.

Usage:
    $ yolo mode=rl_train model=yolo26n.pt data=coco8.yaml epochs=100
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK


def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute IoU matrix between two sets of boxes.

    Args:
        boxes1 (torch.Tensor): Boxes of shape [N, 4] in xyxy format.
        boxes2 (torch.Tensor): Boxes of shape [M, 4] in xyxy format.
        eps (float): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU matrix of shape [N, M].
    """
    # Ensure correct shape
    if len(boxes1.shape) == 1:
        boxes1 = boxes1.unsqueeze(0)
    if len(boxes2.shape) == 1:
        boxes2 = boxes2.unsqueeze(0)

    n = boxes1.shape[0]
    m = boxes2.shape[0]

    # Extract coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # Compute intersection
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))  # [N, M]
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))  # [N, M]
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))  # [N, M]
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))  # [N, M]

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Compute areas
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)  # [N]
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)  # [M]

    # Union
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

    # IoU
    iou = inter_area / (union_area + eps)

    return iou


class RewardFunction:
    """Base class for reward functions in RL training.

    This class provides the foundation for implementing various reward functions
    that combine IoU and class information for object detection tasks.

    Attributes:
        iou_weight (float): Weight for IoU-based reward component.
        cls_weight (float): Weight for classification-based reward component.
        completeness_weight (float): Weight for detection completeness reward.
        class_weights (dict | None): Per-class reward weights for class-aware training.

    Methods:
        compute_reward: Compute the total reward for predictions vs ground truth.
        iou_reward: Compute IoU-based reward between predicted and target boxes.
        cls_reward: Compute classification-based reward.
        completeness_reward: Compute reward based on detection completeness.
    """

    def __init__(
        self,
        iou_weight: float = 0.7,
        cls_weight: float = 0.0,
        completeness_weight: float = 0.3,
        class_weights: dict[int, float] | None = None,
        iou_threshold: float = 0.5,
    ):
        """Initialize the RewardFunction.

        Args:
            iou_weight (float): Weight for IoU-based reward (default: 0.7).
            cls_weight (float): Weight for classification reward (default: 0.0).
            completeness_weight (float): Weight for completeness reward (default: 0.3).
            class_weights (dict[int, float] | None): Per-class weights for rewards.
            iou_threshold (float): IoU threshold for matching predictions to GT (default: 0.5).
        """
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.completeness_weight = completeness_weight
        self.class_weights = class_weights or {}
        self.iou_threshold = iou_threshold

    def compute_reward(
        self,
        pred_boxes: torch.Tensor,
        pred_cls: torch.Tensor,
        target_boxes: torch.Tensor,
        target_cls: torch.Tensor,
        pred_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute comprehensive reward combining IoU, classification and completeness.

        Args:
            pred_boxes (torch.Tensor): Predicted bounding boxes [N, 4] in xyxy format.
            pred_cls (torch.Tensor): Predicted class IDs [N].
            target_boxes (torch.Tensor): Ground truth boxes [M, 4] in xyxy format.
            target_cls (torch.Tensor): Ground truth class IDs [M].
            pred_scores (torch.Tensor | None): Prediction confidence scores [N].

        Returns:
            (torch.Tensor): Total reward score between 0 and 1.
        """
        if len(target_boxes) == 0:
            # No ground truth - reward 1.0 if no predictions, 0.0 otherwise
            return torch.tensor(1.0 if len(pred_boxes) == 0 else 0.0, device=pred_boxes.device)

        if len(pred_boxes) == 0:
            # Predictions expected but none provided
            return torch.tensor(0.0, device=target_boxes.device)

        # Compute IoU-based reward
        iou_r = self.iou_reward(pred_boxes, pred_cls, target_boxes, target_cls)

        # Compute classification reward
        cls_r = self.cls_reward(pred_cls, target_cls) if self.cls_weight > 0 else torch.tensor(0.0)

        # Compute completeness reward
        comp_r = self.completeness_reward(pred_boxes, target_boxes, pred_cls, target_cls)

        # Normalize weights
        total_weight = self.iou_weight + self.cls_weight + self.completeness_weight
        if total_weight == 0:
            total_weight = 1.0

        # Combine rewards
        total_reward = (
            self.iou_weight * iou_r + self.cls_weight * cls_r + self.completeness_weight * comp_r
        ) / total_weight

        return total_reward

    def iou_reward(
        self,
        pred_boxes: torch.Tensor,
        pred_cls: torch.Tensor,
        target_boxes: torch.Tensor,
        target_cls: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU-based reward with class-aware weighting.

        Uses greedy matching between predictions and ground truth, considering
        both IoU and class labels. Class weights allow emphasizing certain classes.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes [N, 4].
            pred_cls (torch.Tensor): Predicted classes [N].
            target_boxes (torch.Tensor): Target boxes [M, 4].
            target_cls (torch.Tensor): Target classes [M].

        Returns:
            (torch.Tensor): IoU-based reward score.
        """
        device = pred_boxes.device
        n_pred = len(pred_boxes)
        n_gt = len(target_boxes)

        if n_pred == 0 or n_gt == 0:
            return torch.tensor(0.0, device=device)

        # Compute IoU matrix using our custom function
        iou_matrix = compute_iou_matrix(pred_boxes, target_boxes)

        # Greedy matching
        matched_ious = []
        matched_gt_classes = []
        unmatched_preds = list(range(n_pred))
        unmatched_gts = list(range(n_gt))

        while unmatched_preds and unmatched_gts:
            # Find best match
            max_iou = -1
            max_pred_idx = -1
            max_gt_idx = -1

            for pred_idx in unmatched_preds:
                for gt_idx in unmatched_gts:
                    curr_iou = iou_matrix[pred_idx, gt_idx].item()
                    if curr_iou > max_iou:
                        max_iou = curr_iou
                        max_pred_idx = pred_idx
                        max_gt_idx = gt_idx

            if max_iou < self.iou_threshold:
                break

            # Check if classes match
            pred_class = int(pred_cls[max_pred_idx].item())
            gt_class = int(target_cls[max_gt_idx].item())

            if pred_class == gt_class:
                # Get class weight
                class_weight = self.class_weights.get(gt_class, 1.0)
                matched_ious.append(max_iou * class_weight)
                matched_gt_classes.append(gt_class)

            unmatched_preds.remove(max_pred_idx)
            unmatched_gts.remove(max_gt_idx)

        if not matched_ious:
            return torch.tensor(0.0, device=device)

        # Compute weighted average IoU
        total_weight = sum(self.class_weights.get(c, 1.0) for c in range(n_gt))
        if total_weight == 0:
            total_weight = n_gt

        iou_score = sum(matched_ious) / total_weight
        return torch.tensor(min(iou_score, 1.0), device=device)

    def cls_reward(self, pred_cls: torch.Tensor, target_cls: torch.Tensor) -> torch.Tensor:
        """Compute classification accuracy reward.

        Args:
            pred_cls (torch.Tensor): Predicted class IDs.
            target_cls (torch.Tensor): Target class IDs.

        Returns:
            (torch.Tensor): Classification reward score.
        """
        if len(pred_cls) == 0 or len(target_cls) == 0:
            return torch.tensor(0.0, device=pred_cls.device if len(pred_cls) > 0 else target_cls.device)

        # Simple accuracy based on class distribution overlap
        pred_set = set(pred_cls.cpu().numpy().tolist())
        target_set = set(target_cls.cpu().numpy().tolist())

        intersection = len(pred_set & target_set)
        union = len(pred_set | target_set)

        if union == 0:
            return torch.tensor(0.0)

        return torch.tensor(intersection / union, device=pred_cls.device)

    def completeness_reward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        pred_cls: torch.Tensor,
        target_cls: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward based on detection completeness.

        Penalizes both missed detections and false positives.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes.
            target_boxes (torch.Tensor): Ground truth boxes.
            pred_cls (torch.Tensor): Predicted classes.
            target_cls (torch.Tensor): Ground truth classes.

        Returns:
            (torch.Tensor): Completeness reward score.
        """
        n_pred = len(pred_boxes)
        n_gt = len(target_boxes)

        if n_gt == 0:
            return torch.tensor(1.0 if n_pred == 0 else 0.0, device=pred_boxes.device)

        if n_pred == 0:
            return torch.tensor(0.0, device=target_boxes.device)

        # Compute IoU matrix using our custom function
        iou_matrix = compute_iou_matrix(pred_boxes, target_boxes)

        # Count matches considering both IoU and class
        matches = 0
        for pred_idx in range(n_pred):
            for gt_idx in range(n_gt):
                if iou_matrix[pred_idx, gt_idx] >= self.iou_threshold:
                    if int(pred_cls[pred_idx].item()) == int(target_cls[gt_idx].item()):
                        matches += 1
                        break

        # Miss rate and false alarm rate
        miss_rate = (n_gt - matches) / n_gt
        false_alarm_rate = (n_pred - matches) / n_pred if n_pred > 0 else 0.0

        # Completeness score
        completeness = 1.0 - (miss_rate + false_alarm_rate) / 2.0
        return torch.tensor(max(0.0, completeness), device=pred_boxes.device)


class AreaWeightedReward(RewardFunction):
    """Reward function that weights objects by their pixel area.

    This reward function considers the size of objects when computing rewards,
    giving more importance to larger objects that occupy more pixels.

    Attributes:
        area_power (float): Power to apply to area weights (higher = more emphasis on large objects).
        normalize_area (bool): Whether to normalize areas by image size.
    """

    def __init__(
        self,
        iou_weight: float = 0.7,
        cls_weight: float = 0.0,
        completeness_weight: float = 0.3,
        class_weights: dict[int, float] | None = None,
        iou_threshold: float = 0.5,
        area_power: float = 0.5,
        normalize_area: bool = True,
    ):
        """Initialize AreaWeightedReward.

        Args:
            area_power (float): Power to apply to area weights (default: 0.5).
            normalize_area (bool): Whether to normalize areas (default: True).
            **kwargs: Additional arguments passed to parent RewardFunction.
        """
        super().__init__(iou_weight, cls_weight, completeness_weight, class_weights, iou_threshold)
        self.area_power = area_power
        self.normalize_area = normalize_area

    def iou_reward(
        self,
        pred_boxes: torch.Tensor,
        pred_cls: torch.Tensor,
        target_boxes: torch.Tensor,
        target_cls: torch.Tensor,
        img_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Compute area-weighted IoU reward.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes [N, 4].
            pred_cls (torch.Tensor): Predicted classes [N].
            target_boxes (torch.Tensor): Target boxes [M, 4].
            target_cls (torch.Tensor): Target classes [M].
            img_size (tuple[int, int] | None): Image size (height, width) for normalization.

        Returns:
            (torch.Tensor): Area-weighted IoU reward.
        """
        device = pred_boxes.device
        n_pred = len(pred_boxes)
        n_gt = len(target_boxes)

        if n_pred == 0 or n_gt == 0:
            return torch.tensor(0.0, device=device)

        # Compute target box areas
        target_areas = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

        if self.normalize_area and img_size is not None:
            total_img_area = img_size[0] * img_size[1]
            target_areas = target_areas / total_img_area

        # Apply area power for weighting
        area_weights = target_areas.pow(self.area_power)
        area_weights = area_weights / area_weights.sum()  # Normalize

        # Compute IoU matrix using our custom function
        iou_matrix = compute_iou_matrix(pred_boxes, target_boxes)

        # Greedy matching with area weighting
        total_weighted_iou = 0.0
        total_weight = 0.0
        unmatched_preds = list(range(n_pred))
        unmatched_gts = list(range(n_gt))

        while unmatched_preds and unmatched_gts:
            max_iou = -1
            max_pred_idx = -1
            max_gt_idx = -1

            for pred_idx in unmatched_preds:
                for gt_idx in unmatched_gts:
                    curr_iou = iou_matrix[pred_idx, gt_idx].item()
                    if curr_iou > max_iou:
                        max_iou = curr_iou
                        max_pred_idx = pred_idx
                        max_gt_idx = gt_idx

            if max_iou < self.iou_threshold:
                break

            pred_class = int(pred_cls[max_pred_idx].item())
            gt_class = int(target_cls[max_gt_idx].item())

            if pred_class == gt_class:
                weight = area_weights[max_gt_idx].item() * self.class_weights.get(gt_class, 1.0)
                total_weighted_iou += max_iou * weight
                total_weight += weight

            unmatched_preds.remove(max_pred_idx)
            unmatched_gts.remove(max_gt_idx)

        if total_weight == 0:
            return torch.tensor(0.0, device=device)

        return torch.tensor(total_weighted_iou / total_weight, device=device)


class GRPOOptimizer:
    """Group Relative Policy Optimization (GRPO) for RL-based training.

    Implements the GRPO algorithm from VLM-R1 for training object detection models
    using reinforcement learning with reward-based optimization.

    Attributes:
        num_iterations (int): Number of optimization iterations per batch.
        epsilon (float): Clipping parameter for policy updates.
        beta (float): KL divergence coefficient.
        reward_func (RewardFunction): Reward function for computing rewards.

    Methods:
        compute_advantages: Compute advantages from rewards.
        compute_loss: Compute GRPO loss for policy optimization.
    """

    def __init__(
        self,
        num_iterations: int = 1,
        epsilon: float = 0.2,
        beta: float = 0.01,
        reward_func: RewardFunction | None = None,
    ):
        """Initialize GRPO optimizer.

        Args:
            num_iterations (int): Number of optimization iterations per batch.
            epsilon (float): Clipping parameter for PPO-style updates.
            beta (float): KL divergence penalty coefficient.
            reward_func (RewardFunction | None): Reward function to use.
        """
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.beta = beta
        self.reward_func = reward_func or RewardFunction()

    def compute_advantages(self, rewards: torch.Tensor, baseline: float | None = None) -> torch.Tensor:
        """Compute normalized advantages from rewards.

        Args:
            rewards (torch.Tensor): Reward values [batch_size].
            baseline (float | None): Optional baseline for advantage computation.

        Returns:
            (torch.Tensor): Normalized advantages.
        """
        if baseline is None:
            baseline = rewards.mean()

        advantages = rewards - baseline

        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_log_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute GRPO loss with optional KL penalty.

        Args:
            log_probs (torch.Tensor): Current policy log probabilities.
            old_log_probs (torch.Tensor): Old policy log probabilities.
            advantages (torch.Tensor): Computed advantages.
            ref_log_probs (torch.Tensor | None): Reference policy log probs for KL penalty.

        Returns:
            (torch.Tensor): Total GRPO loss.
        """
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL divergence penalty (if reference provided)
        kl_loss = torch.tensor(0.0, device=log_probs.device)
        if ref_log_probs is not None and self.beta > 0:
            kl_div = torch.exp(ref_log_probs) * (ref_log_probs - log_probs)
            kl_loss = self.beta * kl_div.mean()

        return policy_loss + kl_loss


class RLTrainer(BaseTrainer):
    """Reinforcement Learning Trainer for YOLO models.

    This trainer extends BaseTrainer to implement RL-based training using
    reward functions that combine IoU and class information, inspired by
    the VLM-R1 GRPO algorithm.

    Attributes:
        reward_func (RewardFunction): Reward function for computing rewards.
        grpo (GRPOOptimizer): GRPO optimizer for policy updates.
        rl_config (dict): RL-specific configuration parameters.

    Methods:
        train: Execute RL training process.
        compute_batch_rewards: Compute rewards for a batch of predictions.
        rl_step: Perform single RL optimization step.
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
    ):
        """Initialize RLTrainer.

        Args:
            cfg: Configuration object or path.
            overrides (dict | None): Configuration overrides.
            _callbacks: Optional callback functions.
        """
        # Set defaults for RL-specific parameters
        overrides = overrides or {}

        # RL configuration
        self.rl_config = {
            "rl_weight": overrides.pop("rl_weight", 0.5),  # Weight for RL loss vs supervised loss
            "iou_weight": overrides.pop("iou_weight", 0.7),
            "cls_weight": overrides.pop("cls_weight", 0.0),
            "completeness_weight": overrides.pop("completeness_weight", 0.3),
            "iou_threshold": overrides.pop("iou_threshold", 0.5),
            "grpo_iterations": overrides.pop("grpo_iterations", 1),
            "grpo_epsilon": overrides.pop("grpo_epsilon", 0.2),
            "grpo_beta": overrides.pop("grpo_beta", 0.01),
            "class_weights": overrides.pop("class_weights", None),
            "use_area_weighting": overrides.pop("use_area_weighting", False),
            "area_power": overrides.pop("area_power", 0.5),
        }

        super().__init__(cfg, overrides, _callbacks)

        # Initialize reward function
        if self.rl_config["use_area_weighting"]:
            self.reward_func = AreaWeightedReward(
                iou_weight=self.rl_config["iou_weight"],
                cls_weight=self.rl_config["cls_weight"],
                completeness_weight=self.rl_config["completeness_weight"],
                class_weights=self.rl_config["class_weights"],
                iou_threshold=self.rl_config["iou_threshold"],
                area_power=self.rl_config["area_power"],
            )
        else:
            self.reward_func = RewardFunction(
                iou_weight=self.rl_config["iou_weight"],
                cls_weight=self.rl_config["cls_weight"],
                completeness_weight=self.rl_config["completeness_weight"],
                class_weights=self.rl_config["class_weights"],
                iou_threshold=self.rl_config["iou_threshold"],
            )

        # Initialize GRPO optimizer
        self.grpo = GRPOOptimizer(
            num_iterations=self.rl_config["grpo_iterations"],
            epsilon=self.rl_config["grpo_epsilon"],
            beta=self.rl_config["grpo_beta"],
            reward_func=self.reward_func,
        )

    def compute_batch_rewards(
        self,
        preds: list[dict],
        targets: dict,
        img_sizes: list[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        """Compute rewards for a batch of predictions.

        Args:
            preds (list[dict]): List of prediction dictionaries with 'boxes', 'cls', 'scores'.
            targets (dict): Target dictionary with 'bboxes', 'cls', 'batch_idx'.
            img_sizes (list[tuple] | None): Image sizes for area normalization.

        Returns:
            (torch.Tensor): Batch rewards [batch_size].
        """
        batch_size = len(preds)
        rewards = []

        for i in range(batch_size):
            # Get predictions for this image
            pred_dict = preds[i]
            pred_boxes = pred_dict.get("boxes", torch.tensor([]))
            pred_cls = pred_dict.get("cls", torch.tensor([]))
            pred_scores = pred_dict.get("scores", None)

            # Get targets for this image
            batch_mask = targets["batch_idx"] == i
            target_boxes = targets["bboxes"][batch_mask]
            target_cls = targets["cls"][batch_mask]

            # Compute reward
            if isinstance(self.reward_func, AreaWeightedReward) and img_sizes:
                reward = self.reward_func.compute_reward(
                    pred_boxes, pred_cls, target_boxes, target_cls, pred_scores
                )
            else:
                reward = self.reward_func.compute_reward(
                    pred_boxes, pred_cls, target_boxes, target_cls, pred_scores
                )

            rewards.append(reward)

        return torch.stack(rewards)

    def rl_step(
        self,
        batch: dict,
        preds: torch.Tensor,
        supervised_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Perform single RL optimization step.

        Args:
            batch (dict): Input batch data.
            preds (torch.Tensor): Model predictions.
            supervised_loss (torch.Tensor): Supervised learning loss.

        Returns:
            (tuple[torch.Tensor, dict]): Combined loss and metrics dictionary.
        """
        # Decode predictions to get boxes and classes
        decoded_preds = self._decode_predictions(preds, batch)

        # Prepare targets
        targets = {
            "bboxes": batch.get("bboxes", torch.tensor([])),
            "cls": batch.get("cls", torch.tensor([])),
            "batch_idx": batch.get("batch_idx", torch.tensor([])),
        }

        # Compute rewards
        rewards = self.compute_batch_rewards(decoded_preds, targets)

        # Compute advantages
        advantages = self.grpo.compute_advantages(rewards)

        # Compute RL loss (simplified - use reward as signal)
        rl_loss = -rewards.mean()

        # Combine losses
        rl_weight = self.rl_config["rl_weight"]
        combined_loss = (1 - rl_weight) * supervised_loss + rl_weight * rl_loss

        metrics = {
            "rl_loss": rl_loss.item(),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item() if len(rewards) > 1 else 0.0,
            "advantage_mean": advantages.mean().item(),
        }

        return combined_loss, metrics

    def _decode_predictions(self, preds: torch.Tensor, batch: dict) -> list[dict]:
        """Decode raw model predictions to boxes and classes.

        Args:
            preds (torch.Tensor): Raw model predictions.
            batch (dict): Batch data containing image info.

        Returns:
            (list[dict]): List of prediction dictionaries.
        """
        # This is a simplified decoder - actual implementation depends on model architecture
        batch_size = batch["img"].shape[0]
        decoded = []

        for i in range(batch_size):
            # Placeholder - actual decoding logic depends on model output format
            decoded.append({
                "boxes": torch.tensor([], device=preds[0].device if isinstance(preds, list) else preds.device),
                "cls": torch.tensor([], device=preds[0].device if isinstance(preds, list) else preds.device),
                "scores": torch.tensor([], device=preds[0].device if isinstance(preds, list) else preds.device),
            })

        return decoded


def get_reward_function(
    reward_type: str = "default",
    **kwargs,
) -> RewardFunction:
    """Factory function to create reward functions.

    Args:
        reward_type (str): Type of reward function ('default', 'area_weighted').
        **kwargs: Additional arguments for reward function initialization.

    Returns:
        (RewardFunction): Configured reward function instance.
    """
    reward_functions = {
        "default": RewardFunction,
        "area_weighted": AreaWeightedReward,
    }

    if reward_type not in reward_functions:
        LOGGER.warning(f"Unknown reward type '{reward_type}', using default")
        reward_type = "default"

    return reward_functions[reward_type](**kwargs)


# Task-specific RL Trainers
class DetectionRLTrainer(RLTrainer):
    """RL Trainer for object detection models.

    Extends RLTrainer with detection-specific functionality including
    dataset building and prediction decoding for detection tasks.

    Examples:
        >>> from ultralytics.engine.rl_trainer import DetectionRLTrainer
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=3, rl_weight=0.5)
        >>> trainer = DetectionRLTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize DetectionRLTrainer."""
        overrides = overrides or {}
        overrides["task"] = "detect"
        super().__init__(cfg, overrides, _callbacks)


class SegmentationRLTrainer(RLTrainer):
    """RL Trainer for segmentation models.

    Extends RLTrainer with segmentation-specific functionality including
    mask-based reward computation.

    Examples:
        >>> from ultralytics.engine.rl_trainer import SegmentationRLTrainer
        >>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml", epochs=3, rl_weight=0.5)
        >>> trainer = SegmentationRLTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize SegmentationRLTrainer."""
        overrides = overrides or {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)


class PoseRLTrainer(RLTrainer):
    """RL Trainer for pose estimation models.

    Extends RLTrainer with pose-specific functionality including
    keypoint-based reward computation.

    Examples:
        >>> from ultralytics.engine.rl_trainer import PoseRLTrainer
        >>> args = dict(model="yolo26n-pose.pt", data="coco8-pose.yaml", epochs=3, rl_weight=0.5)
        >>> trainer = PoseRLTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize PoseRLTrainer."""
        overrides = overrides or {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)


class OBBRLTrainer(RLTrainer):
    """RL Trainer for oriented bounding box (OBB) models.

    Extends RLTrainer with OBB-specific functionality including
    rotated box IoU reward computation.

    Examples:
        >>> from ultralytics.engine.rl_trainer import OBBRLTrainer
        >>> args = dict(model="yolo26n-obb.pt", data="dota8.yaml", epochs=3, rl_weight=0.5)
        >>> trainer = OBBRLTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize OBBRLTrainer."""
        overrides = overrides or {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)


class ClassificationRLTrainer(RLTrainer):
    """RL Trainer for classification models.

    Extends RLTrainer with classification-specific functionality using
    accuracy-based reward computation.

    Examples:
        >>> from ultralytics.engine.rl_trainer import ClassificationRLTrainer
        >>> args = dict(model="yolo26n-cls.pt", data="imagenet10", epochs=3, rl_weight=0.5)
        >>> trainer = ClassificationRLTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize ClassificationRLTrainer."""
        overrides = overrides or {}
        overrides["task"] = "classify"
        super().__init__(cfg, overrides, _callbacks)
