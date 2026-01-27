# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generative Reward Policy Optimization (GRPO) Algorithm.

This module implements the GRPO algorithm for reinforcement learning training in Ultralytics,
inspired by VLM-R1. GRPO is a policy optimization method that uses generated samples and
reward-based feedback to improve model performance.

Key Features:
    - Group-wise advantage computation
    - KL divergence regularization with reference model
    - PPO-style clipping for stable updates
    - Support for multiple reward signals

Example:
    >>> from ultralytics.engine.rl.grpo import GRPO, GRPOTrainer
    >>> from ultralytics.engine.rl.config import GRPOConfig
    >>> config = GRPOConfig(num_generations=4, kl_coeff=0.1)
    >>> grpo = GRPO(config)
    >>> loss = grpo.compute_loss(predictions, rewards, old_log_probs)
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GRPOConfig


class GRPO:
    """Generative Reward Policy Optimization algorithm.

    GRPO optimizes policies by generating multiple completions, evaluating them with reward
    functions, computing group-wise advantages, and updating the policy to favor high-reward
    generations while maintaining proximity to a reference policy.

    Attributes:
        config (GRPOConfig): Configuration for GRPO training.
        reference_model (nn.Module | None): Reference model for KL computation.
        device (torch.device): Computation device.

    Methods:
        compute_advantages: Compute group-wise advantages from rewards.
        compute_loss: Compute the GRPO policy loss.
        compute_kl_divergence: Compute KL divergence from reference policy.
        update_reference_model: Update the reference model.

    Examples:
        >>> config = GRPOConfig(num_generations=4, kl_coeff=0.1)
        >>> grpo = GRPO(config)
        >>> advantages = grpo.compute_advantages(rewards)
    """

    def __init__(
        self,
        config: GRPOConfig | None = None,
        reference_model: nn.Module | None = None,
        device: torch.device | str = "cuda",
    ):
        """Initialize GRPO algorithm.

        Args:
            config (GRPOConfig | None): GRPO configuration.
            reference_model (nn.Module | None): Reference model for KL penalty.
            device (torch.device | str): Computation device.
        """
        self.config = config or GRPOConfig()
        self.reference_model = reference_model
        self.device = torch.device(device) if isinstance(device, str) else device

        # Statistics tracking
        self.kl_history = []
        self.reward_history = []
        self.advantage_history = []

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int | None = None,
    ) -> torch.Tensor:
        """Compute group-wise advantages from rewards.

        Following VLM-R1's approach, advantages are computed by normalizing rewards
        within each group of generations for the same input sample.

        Args:
            rewards (torch.Tensor): Reward values [N] or [B, G] where G is group size.
            group_size (int | None): Number of generations per sample. Inferred if None.

        Returns:
            torch.Tensor: Normalized advantages, same shape as rewards.
        """
        if group_size is None:
            group_size = self.config.num_generations

        # Reshape to [B, G] if flat
        if rewards.dim() == 1:
            if rewards.numel() % group_size != 0:
                # Pad or truncate
                n_groups = math.ceil(rewards.numel() / group_size)
                padded = torch.zeros(n_groups * group_size, device=rewards.device)
                padded[:rewards.numel()] = rewards
                rewards = padded
            rewards = rewards.view(-1, group_size)

        # Compute group statistics
        group_mean = rewards.mean(dim=-1, keepdim=True)
        group_std = rewards.std(dim=-1, keepdim=True) + 1e-8

        # Normalize within groups
        advantages = (rewards - group_mean) / group_std

        # Optionally normalize globally
        if self.config.normalize_advantage:
            global_mean = advantages.mean()
            global_std = advantages.std() + 1e-8
            advantages = (advantages - global_mean) / global_std

        # Track for logging
        self.advantage_history.append(advantages.mean().item())

        return advantages.flatten()

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        entropy: torch.Tensor | None = None,
        kl_penalty: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute GRPO policy loss with PPO-style clipping.

        Args:
            log_probs (torch.Tensor): Current policy log probabilities.
            old_log_probs (torch.Tensor): Old policy log probabilities.
            advantages (torch.Tensor): Computed advantages.
            entropy (torch.Tensor | None): Policy entropy for bonus.
            kl_penalty (torch.Tensor | None): KL divergence penalty.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: Loss tensor and metrics dict.
        """
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # PPO clipping
        clip_ratio = self.config.clip_ratio
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

        # Policy loss (negative because we want to maximize)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Total loss
        total_loss = policy_loss

        # Add KL penalty if provided
        if kl_penalty is not None:
            kl_loss = self.config.kl_coeff * kl_penalty.mean()
            total_loss = total_loss + kl_loss
        else:
            kl_loss = torch.tensor(0.0)

        # Add entropy bonus if provided
        if entropy is not None:
            entropy_bonus = -self.config.entropy_coeff * entropy.mean()
            total_loss = total_loss + entropy_bonus
        else:
            entropy_bonus = torch.tensor(0.0)

        # Compute metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item() if torch.is_tensor(kl_loss) else kl_loss,
            "entropy": entropy.mean().item() if entropy is not None else 0.0,
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
            "clip_fraction": ((ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)).float().mean().item(),
        }

        return total_loss, metrics

    def compute_kl_divergence(
        self,
        current_logits: torch.Tensor,
        reference_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between current and reference policies.

        Args:
            current_logits (torch.Tensor): Logits from current policy.
            reference_logits (torch.Tensor): Logits from reference policy.

        Returns:
            torch.Tensor: KL divergence values.
        """
        current_probs = F.softmax(current_logits, dim=-1)
        reference_probs = F.softmax(reference_logits, dim=-1)

        # KL(p || q) = sum(p * log(p/q))
        kl = (current_probs * (current_probs.log() - reference_probs.log())).sum(-1)

        # Track for logging
        self.kl_history.append(kl.mean().item())

        return kl

    def compute_reference_kl(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute KL divergence using reference model.

        Args:
            model (nn.Module): Current model.
            inputs (dict[str, torch.Tensor]): Model inputs.

        Returns:
            torch.Tensor: KL divergence penalty.
        """
        if self.reference_model is None:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            reference_outputs = self.reference_model(inputs["img"])

        current_outputs = model(inputs["img"])

        # Compute KL for detection outputs
        if isinstance(current_outputs, (tuple, list)):
            current_logits = current_outputs[0] if len(current_outputs) > 0 else current_outputs
            reference_logits = reference_outputs[0] if len(reference_outputs) > 0 else reference_outputs
        else:
            current_logits = current_outputs
            reference_logits = reference_outputs

        return self.compute_kl_divergence(current_logits, reference_logits)

    def update_reference_model(self, model: nn.Module) -> None:
        """Update reference model with current model weights.

        Args:
            model (nn.Module): Current model to copy from.
        """
        if self.config.use_reference_model:
            self.reference_model = deepcopy(model)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False

    def get_statistics(self) -> dict[str, float]:
        """Get training statistics.

        Returns:
            dict[str, float]: Training statistics.
        """
        stats = {}
        if self.kl_history:
            stats["kl_mean"] = sum(self.kl_history[-100:]) / len(self.kl_history[-100:])
        if self.reward_history:
            stats["reward_mean"] = sum(self.reward_history[-100:]) / len(self.reward_history[-100:])
        if self.advantage_history:
            stats["advantage_mean"] = sum(self.advantage_history[-100:]) / len(self.advantage_history[-100:])
        return stats

    def reset_statistics(self) -> None:
        """Reset tracking statistics."""
        self.kl_history.clear()
        self.reward_history.clear()
        self.advantage_history.clear()


class GRPOTrainer:
    """GRPO Trainer for managing the training loop.

    This class provides a high-level interface for GRPO-based training, handling
    generation, reward computation, and policy updates.

    Attributes:
        grpo (GRPO): GRPO algorithm instance.
        reward_fn (callable): Reward function for evaluating generations.
        model (nn.Module): Model being trained.
        optimizer (torch.optim.Optimizer): Optimizer for model updates.

    Methods:
        train_step: Perform a single GRPO training step.
        generate: Generate multiple completions for inputs.
        evaluate_rewards: Compute rewards for generations.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        reward_fn: Any,
        config: GRPOConfig | None = None,
        device: torch.device | str = "cuda",
    ):
        """Initialize GRPO trainer.

        Args:
            model (nn.Module): Model to train.
            optimizer (torch.optim.Optimizer): Optimizer.
            reward_fn: Reward function.
            config (GRPOConfig | None): GRPO configuration.
            device (torch.device | str): Computation device.
        """
        self.model = model
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.device = torch.device(device) if isinstance(device, str) else device
        self.config = config or GRPOConfig()
        self.grpo = GRPO(config=self.config, device=self.device)

        # Initialize reference model
        if self.config.use_reference_model:
            self.grpo.update_reference_model(model)

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Perform a single GRPO training step.

        Args:
            batch (dict[str, torch.Tensor]): Input batch.
            targets (dict[str, torch.Tensor]): Target labels.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: Loss and metrics.
        """
        self.model.train()

        # Generate multiple outputs
        generations = self.generate(batch)

        # Compute rewards for each generation
        rewards = self.evaluate_rewards(generations, targets)

        # Compute advantages
        advantages = self.grpo.compute_advantages(rewards)

        # Compute log probabilities
        log_probs = self._compute_log_probs(generations)
        old_log_probs = log_probs.detach()

        # Compute KL penalty if using reference model
        kl_penalty = None
        if self.config.use_reference_model and self.grpo.reference_model is not None:
            kl_penalty = self.grpo.compute_reference_kl(self.model, batch)

        # Compute policy loss
        loss, metrics = self.grpo.compute_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            kl_penalty=kl_penalty,
        )

        # Update model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        self.optimizer.step()

        # Update metrics
        metrics["reward_mean"] = rewards.mean().item()
        metrics["reward_std"] = rewards.std().item()

        # Track reward history
        self.grpo.reward_history.append(rewards.mean().item())

        return loss, metrics

    def generate(
        self,
        batch: dict[str, torch.Tensor],
    ) -> list[dict[str, torch.Tensor]]:
        """Generate multiple outputs for the input batch.

        Args:
            batch (dict[str, torch.Tensor]): Input batch.

        Returns:
            list[dict[str, torch.Tensor]]: List of generation outputs.
        """
        generations = []
        num_gen = self.config.num_generations

        # Use different temperatures for diversity
        temperatures = [self.config.temperature * (1.0 + 0.1 * i) for i in range(num_gen)]

        with torch.no_grad():
            for temp in temperatures:
                # Forward pass with temperature scaling
                output = self.model(batch["img"])

                # Apply temperature if applicable
                if isinstance(output, (tuple, list)):
                    # Detection model output
                    scaled_output = []
                    for o in output:
                        if torch.is_tensor(o) and o.dim() > 1:
                            scaled_output.append(o / temp)
                        else:
                            scaled_output.append(o)
                    generations.append({"output": scaled_output, "temperature": temp})
                else:
                    generations.append({"output": output / temp, "temperature": temp})

        return generations

    def evaluate_rewards(
        self,
        generations: list[dict[str, torch.Tensor]],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute rewards for generated outputs.

        Args:
            generations (list[dict[str, torch.Tensor]]): Generated outputs.
            targets (dict[str, torch.Tensor]): Ground truth targets.

        Returns:
            torch.Tensor: Reward values for each generation.
        """
        rewards = []

        for gen in generations:
            # Extract predictions from generation
            predictions = gen["output"]

            # Compute reward
            reward = self.reward_fn(predictions, targets)

            if torch.is_tensor(reward):
                rewards.append(reward.mean())
            else:
                rewards.append(torch.tensor(reward, device=self.device))

        return torch.stack(rewards)

    def _compute_log_probs(
        self,
        generations: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute log probabilities for generations.

        Args:
            generations (list[dict[str, torch.Tensor]]): Generated outputs.

        Returns:
            torch.Tensor: Log probabilities.
        """
        log_probs = []

        for gen in generations:
            output = gen["output"]

            # Handle different output types
            if isinstance(output, (tuple, list)):
                # Use first output (typically class logits)
                logits = output[0] if len(output) > 0 else None
                if logits is not None and torch.is_tensor(logits):
                    log_prob = F.log_softmax(logits, dim=-1).max(-1).values.mean()
                else:
                    log_prob = torch.tensor(0.0, device=self.device)
            elif torch.is_tensor(output):
                if output.dim() > 1:
                    log_prob = F.log_softmax(output, dim=-1).max(-1).values.mean()
                else:
                    log_prob = output.mean()
            else:
                log_prob = torch.tensor(0.0, device=self.device)

            log_probs.append(log_prob)

        return torch.stack(log_probs)

    def update_reference(self) -> None:
        """Update the reference model with current weights."""
        self.grpo.update_reference_model(self.model)

    def get_statistics(self) -> dict[str, float]:
        """Get training statistics.

        Returns:
            dict[str, float]: Training statistics.
        """
        return self.grpo.get_statistics()
