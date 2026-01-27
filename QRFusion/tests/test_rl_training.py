# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Unit tests for QRFusion RL training functionality.

These tests verify the core RL components including:
- Configuration classes (GRPOConfig, RLConfig)
- Reward functions (IoUReward, ClassAwareReward, etc.)
- GRPO algorithm
- RL trainer integration
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

# Add QRFusion to path for imports
# This handles both local development and CI environments
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from QRFusion.ultralytics.engine.rl.config import GRPOConfig, RLConfig
from QRFusion.ultralytics.engine.rl.rewards import (
    IoUReward,
    ClassAwareReward,
    PixelAreaReward,
    SegmentationIoUReward,
    KeypointOKSReward,
    FormatReward,
    CompositeReward,
    RewardRegistry,
)
from QRFusion.ultralytics.engine.rl.grpo import GRPO, GRPOTrainer


class TestGRPOConfig:
    """Test cases for GRPOConfig class."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = GRPOConfig()

        assert config.num_generations == 4
        assert config.kl_coeff == 0.1
        assert config.clip_ratio == 0.2
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.normalize_advantage is True
        assert config.use_reference_model is True

    def test_custom_initialization(self):
        """Test custom configuration values."""
        config = GRPOConfig(
            num_generations=8,
            kl_coeff=0.2,
            clip_ratio=0.3,
            reward_weights={'iou': 2.0, 'class_accuracy': 1.0}
        )

        assert config.num_generations == 8
        assert config.kl_coeff == 0.2
        assert config.reward_weights['iou'] == 2.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = GRPOConfig(num_generations=8)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['num_generations'] == 8
        assert 'reward_weights' in config_dict

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'num_generations': 6,
            'kl_coeff': 0.15,
            'clip_ratio': 0.25,
        }
        config = GRPOConfig.from_dict(config_dict)

        assert config.num_generations == 6
        assert config.kl_coeff == 0.15


class TestRLConfig:
    """Test cases for RLConfig class."""

    def test_default_initialization(self):
        """Test default RL configuration."""
        config = RLConfig()

        assert config.enabled is False
        assert config.algorithm == 'grpo'
        assert config.reward_type == 'composite'
        assert config.rl_weight == 0.5

    def test_validation_valid_config(self):
        """Test validation with valid configuration."""
        config = RLConfig(enabled=True, algorithm='grpo', rl_weight=0.3)
        config.validate()  # Should not raise

    def test_validation_invalid_algorithm(self):
        """Test validation with invalid algorithm."""
        config = RLConfig(algorithm='invalid')

        with pytest.raises(ValueError, match="Invalid RL algorithm"):
            config.validate()

    def test_validation_invalid_rl_weight(self):
        """Test validation with invalid rl_weight."""
        config = RLConfig(rl_weight=1.5)

        with pytest.raises(ValueError, match="rl_weight must be between 0 and 1"):
            config.validate()


class TestIoUReward:
    """Test cases for IoUReward class."""

    def test_initialization(self):
        """Test IoU reward initialization."""
        reward = IoUReward(iou_type='ciou', weight=1.5)

        assert reward.name == 'iou_reward'
        assert reward.weight == 1.5
        assert reward.iou_type == 'ciou'

    def test_iou_computation(self):
        """Test basic IoU computation."""
        reward = IoUReward(iou_type='iou')

        # Perfect overlap
        box1 = torch.tensor([[0, 0, 10, 10]])
        box2 = torch.tensor([[0, 0, 10, 10]])

        result = reward(box1, box2)
        assert result.item() == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        """Test IoU with no overlap."""
        reward = IoUReward(iou_type='iou')

        box1 = torch.tensor([[0, 0, 10, 10]])
        box2 = torch.tensor([[20, 20, 30, 30]])

        result = reward(box1, box2)
        assert result.item() == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self):
        """Test IoU with partial overlap."""
        reward = IoUReward(iou_type='iou')

        box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        box2 = torch.tensor([[5.0, 5.0, 15.0, 15.0]])

        result = reward(box1, box2)
        # Intersection: 5x5=25, Union: 100+100-25=175
        expected_iou = 25.0 / 175.0
        assert result.item() == pytest.approx(expected_iou, abs=0.01)

    def test_dict_input(self):
        """Test IoU reward with dictionary input."""
        reward = IoUReward()

        predictions = {'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]])}
        targets = {'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]])}

        result = reward(predictions, targets)
        assert result.item() == pytest.approx(1.0, abs=0.01)


class TestClassAwareReward:
    """Test cases for ClassAwareReward class."""

    def test_initialization(self):
        """Test class-aware reward initialization."""
        reward = ClassAwareReward(class_weights={0: 1.5, 1: 1.0})

        assert reward.name == 'class_reward'
        assert reward.class_weights[0] == 1.5

    def test_correct_classification(self):
        """Test reward for correct classification."""
        reward = ClassAwareReward()

        pred_cls = torch.tensor([0, 1, 2])
        gt_cls = torch.tensor([0, 1, 2])

        result = reward(pred_cls, gt_cls)
        assert result.mean().item() == pytest.approx(1.0, abs=0.01)

    def test_incorrect_classification(self):
        """Test reward for incorrect classification."""
        reward = ClassAwareReward()

        pred_cls = torch.tensor([1, 2, 0])
        gt_cls = torch.tensor([0, 1, 2])

        result = reward(pred_cls, gt_cls)
        assert result.mean().item() == pytest.approx(0.0, abs=0.01)

    def test_class_weights(self):
        """Test class-specific weighting."""
        reward = ClassAwareReward(class_weights={0: 2.0, 1: 1.0})

        pred_cls = torch.tensor([0, 1])
        gt_cls = torch.tensor([0, 1])

        result = reward(pred_cls, gt_cls)
        # Both correct, but class 0 has weight 2.0, class 1 has weight 1.0
        expected = (2.0 + 1.0) / 2  # Average
        assert result.mean().item() == pytest.approx(1.5, abs=0.01)


class TestPixelAreaReward:
    """Test cases for PixelAreaReward class."""

    def test_initialization(self):
        """Test pixel area reward initialization."""
        reward = PixelAreaReward(
            area_weights={'small': 2.0, 'medium': 1.0, 'large': 0.5}
        )

        assert reward.area_weights['small'] == 2.0

    def test_small_object_weighting(self):
        """Test higher weight for small objects."""
        reward = PixelAreaReward(
            area_thresholds={'small': 1000, 'medium': 5000},
            area_weights={'small': 2.0, 'medium': 1.0, 'large': 0.5}
        )

        # Small box (area = 100)
        targets = {'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]])}

        result = reward(targets, targets)
        # Should get small weight
        assert result.item() == pytest.approx(2.0, abs=0.01)


class TestSegmentationIoUReward:
    """Test cases for SegmentationIoUReward class."""

    def test_initialization(self):
        """Test segmentation IoU reward initialization."""
        reward = SegmentationIoUReward(weight=0.8)

        assert reward.name == 'mask_iou_reward'
        assert reward.weight == 0.8

    def test_perfect_mask_overlap(self):
        """Test mask IoU with perfect overlap."""
        reward = SegmentationIoUReward()

        mask = torch.ones(1, 32, 32)
        result = reward(mask, mask)

        assert result.item() == pytest.approx(1.0, abs=0.01)

    def test_no_mask_overlap(self):
        """Test mask IoU with no overlap."""
        reward = SegmentationIoUReward()

        pred_mask = torch.zeros(1, 32, 32)
        pred_mask[0, :16, :16] = 1.0

        gt_mask = torch.zeros(1, 32, 32)
        gt_mask[0, 16:, 16:] = 1.0

        result = reward(pred_mask, gt_mask)
        assert result.item() == pytest.approx(0.0, abs=0.01)


class TestCompositeReward:
    """Test cases for CompositeReward class."""

    def test_initialization(self):
        """Test composite reward initialization."""
        rewards = [IoUReward(), ClassAwareReward()]
        composite = CompositeReward(rewards, weights=[1.0, 0.5])

        assert len(composite.rewards) == 2
        assert composite.weights == [1.0, 0.5]

    def test_combined_reward(self):
        """Test combined reward computation."""
        iou_reward = IoUReward(weight=1.0)
        class_reward = ClassAwareReward(weight=1.0)

        composite = CompositeReward([iou_reward, class_reward], weights=[1.0, 1.0])

        predictions = {
            'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            'cls': torch.tensor([0])
        }
        targets = {
            'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            'cls': torch.tensor([0])
        }

        result = composite(predictions, targets)
        # Both rewards should be 1.0, combined = 2.0
        assert result.item() == pytest.approx(2.0, abs=0.1)


class TestRewardRegistry:
    """Test cases for RewardRegistry class."""

    def test_register_and_get(self):
        """Test registering and retrieving rewards."""
        test_reward = IoUReward(name='test_iou')
        RewardRegistry.register('test_iou', test_reward)

        retrieved = RewardRegistry.get('test_iou')
        assert retrieved.name == 'test_iou'

    def test_get_nonexistent(self):
        """Test getting non-existent reward."""
        with pytest.raises(KeyError):
            RewardRegistry.get('nonexistent_reward')

    def test_list_rewards(self):
        """Test listing registered rewards."""
        rewards = RewardRegistry.list_rewards()

        assert isinstance(rewards, list)
        assert 'iou' in rewards
        assert 'class_aware' in rewards


class TestGRPO:
    """Test cases for GRPO algorithm class."""

    def test_initialization(self):
        """Test GRPO initialization."""
        config = GRPOConfig()
        grpo = GRPO(config=config, device='cpu')

        assert grpo.config == config
        assert grpo.reference_model is None

    def test_compute_advantages(self):
        """Test advantage computation."""
        config = GRPOConfig(num_generations=4, normalize_advantage=True)
        grpo = GRPO(config=config, device='cpu')

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        advantages = grpo.compute_advantages(rewards, group_size=4)

        # Advantages should be normalized
        assert advantages.mean().abs() < 0.1
        assert len(advantages) == 4

    def test_compute_loss(self):
        """Test loss computation."""
        config = GRPOConfig()
        grpo = GRPO(config=config, device='cpu')

        log_probs = torch.tensor([-0.5, -0.3, -0.7, -0.2])
        old_log_probs = torch.tensor([-0.5, -0.3, -0.7, -0.2])
        advantages = torch.tensor([0.5, -0.3, 0.8, -0.5])

        loss, metrics = grpo.compute_loss(log_probs, old_log_probs, advantages)

        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)
        assert 'policy_loss' in metrics

    def test_compute_kl_divergence(self):
        """Test KL divergence computation."""
        config = GRPOConfig()
        grpo = GRPO(config=config, device='cpu')

        # Same distribution
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        kl = grpo.compute_kl_divergence(logits, logits)

        assert kl.item() == pytest.approx(0.0, abs=0.01)

    def test_get_statistics(self):
        """Test statistics retrieval."""
        config = GRPOConfig()
        grpo = GRPO(config=config, device='cpu')

        # Add some history
        grpo.kl_history.append(0.1)
        grpo.reward_history.append(0.8)

        stats = grpo.get_statistics()
        assert 'kl_mean' in stats
        assert 'reward_mean' in stats


class TestFormatReward:
    """Test cases for FormatReward class."""

    def test_valid_format(self):
        """Test reward for valid format."""
        reward = FormatReward(valid_range=(0.0, 1.0))

        predictions = torch.tensor([[0.5, 0.3, 0.8]])
        result = reward(predictions, None)

        assert result.item() == pytest.approx(1.0, abs=0.01)

    def test_invalid_format(self):
        """Test reward for invalid format."""
        reward = FormatReward(valid_range=(0.0, 1.0))

        predictions = torch.tensor([[1.5, -0.3, 0.8]])  # 2 out of 3 invalid
        result = reward(predictions, None)

        # 1 out of 3 valid
        assert result.item() == pytest.approx(1.0 / 3, abs=0.1)


class TestKeypointOKSReward:
    """Test cases for KeypointOKSReward class."""

    def test_initialization(self):
        """Test OKS reward initialization."""
        reward = KeypointOKSReward(weight=1.0)

        assert reward.name == 'keypoint_oks_reward'
        assert reward.sigmas is not None

    def test_perfect_keypoints(self):
        """Test OKS with perfect keypoint predictions."""
        reward = KeypointOKSReward()

        # Keypoints in format [N, K, 2]
        keypoints = torch.tensor([[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]])

        result = reward(keypoints, keypoints, batch={'area': torch.tensor([100.0])})

        assert result.item() == pytest.approx(1.0, abs=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
