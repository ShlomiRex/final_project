"""
Learning Rate Schedulers.

Provides learning rate scheduling strategies for training.
"""

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def get_lr_scheduler(
    config: dict,
    optimizer: Optimizer,
    num_training_steps: int,
) -> LRScheduler:
    """
    Get a learning rate scheduler based on configuration.
    
    Args:
        config: Scheduler configuration
        optimizer: Optimizer to schedule
        num_training_steps: Total training steps
    
    Returns:
        LRScheduler instance
    """
    scheduler_type = config.get("type", "cosine")
    warmup_steps = config.get("warmup_steps", 0)
    
    if scheduler_type == "constant":
        return ConstantWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
        )
    
    elif scheduler_type == "linear":
        return LinearWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    elif scheduler_type == "cosine":
        return CosineWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=config.get("min_lr_ratio", 0.0),
        )
    
    elif scheduler_type == "cosine_with_restarts":
        return CosineWithWarmupAndRestarts(
            optimizer,
            warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=config.get("num_cycles", 1),
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class ConstantWithWarmup(LRScheduler):
    """Constant learning rate with linear warmup."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            return [
                base_lr * self._step_count / max(1, self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        return self.base_lrs


class LinearWithWarmup(LRScheduler):
    """Linear decay with linear warmup."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        num_training_steps: int = 1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            return [
                base_lr * self._step_count / max(1, self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        
        progress = (self._step_count - self.warmup_steps) / max(
            1, self.num_training_steps - self.warmup_steps
        )
        
        return [
            base_lr * max(0.0, 1.0 - progress)
            for base_lr in self.base_lrs
        ]


class CosineWithWarmup(LRScheduler):
    """Cosine decay with linear warmup."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        num_training_steps: int = 1,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            return [
                base_lr * self._step_count / max(1, self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        
        progress = (self._step_count - self.warmup_steps) / max(
            1, self.num_training_steps - self.warmup_steps
        )
        
        return [
            base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * 
                      (1 + math.cos(math.pi * progress)) / 2)
            for base_lr in self.base_lrs
        ]


class CosineWithWarmupAndRestarts(LRScheduler):
    """Cosine decay with warmup and hard restarts."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        num_training_steps: int = 1,
        num_cycles: int = 1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            return [
                base_lr * self._step_count / max(1, self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        
        progress = (self._step_count - self.warmup_steps) / max(
            1, self.num_training_steps - self.warmup_steps
        )
        
        return [
            base_lr * (1 + math.cos(math.pi * ((progress * self.num_cycles) % 1.0))) / 2
            for base_lr in self.base_lrs
        ]
