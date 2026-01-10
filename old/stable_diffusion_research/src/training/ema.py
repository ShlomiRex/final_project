"""
Exponential Moving Average (EMA) for model weights.

EMA provides a smoothed version of model weights that often
performs better than the raw trained weights.
"""

import copy
from typing import Dict, Iterable, Optional, Union

import torch
import torch.nn as nn


class EMAModel:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of model parameters that is updated
    with exponential moving average of the training parameters.
    
    Usage:
        ema = EMAModel(model.parameters())
        
        # During training
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            ema.step(model.parameters())
        
        # For evaluation, use EMA weights
        ema.copy_to(model.parameters())
        model.eval()
        # ... evaluate ...
        
        # Restore original weights
        ema.restore(model.parameters())
    """
    
    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        update_every: int = 1,
        use_ema_warmup: bool = False,
        inv_gamma: float = 1.0,
        power: float = 2/3,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            parameters: Model parameters to track
            decay: EMA decay rate
            min_decay: Minimum decay rate (for warmup)
            update_after_step: Start updating EMA after this many steps
            update_every: Update EMA every N steps
            use_ema_warmup: Use EMA decay warmup
            inv_gamma: Inverse gamma for warmup schedule
            power: Power for warmup schedule
            device: Device to store shadow parameters
        """
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        
        self.optimization_step = 0
        self.cur_decay_value = 0.0
        
        # Store shadow parameters
        parameters = list(parameters)
        self.shadow_params = [
            p.clone().detach().to(device or p.device)
            for p in parameters
        ]
        
        # Store original parameters for restore
        self.collected_params = None
    
    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the current step.
        
        Args:
            optimization_step: Current optimization step
        
        Returns:
            Decay factor
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        
        if step <= 0:
            return 0.0
        
        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** (-self.power)
        else:
            cur_decay_value = self.decay
        
        cur_decay_value = max(self.min_decay, min(cur_decay_value, self.decay))
        
        return cur_decay_value
    
    @torch.no_grad()
    def step(self, parameters: Iterable[nn.Parameter]):
        """
        Update shadow parameters with EMA.
        
        Args:
            parameters: Current model parameters
        """
        self.optimization_step += 1
        
        # Check if we should update
        if self.optimization_step <= self.update_after_step:
            return
        
        if self.optimization_step % self.update_every != 0:
            return
        
        # Compute decay
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        
        # Update shadow parameters
        parameters = list(parameters)
        
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_((1 - decay) * (s_param - param.data))
            else:
                s_param.copy_(param.data)
    
    @torch.no_grad()
    def copy_to(self, parameters: Iterable[nn.Parameter]):
        """
        Copy shadow parameters to model parameters.
        
        Call this before evaluation to use EMA weights.
        
        Args:
            parameters: Model parameters to copy to
        """
        parameters = list(parameters)
        
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)
    
    @torch.no_grad()
    def store(self, parameters: Iterable[nn.Parameter]):
        """
        Store current model parameters for later restore.
        
        Args:
            parameters: Model parameters to store
        """
        self.collected_params = [
            param.clone().detach()
            for param in parameters
        ]
    
    @torch.no_grad()
    def restore(self, parameters: Iterable[nn.Parameter]):
        """
        Restore stored model parameters.
        
        Call this after evaluation to restore training weights.
        
        Args:
            parameters: Model parameters to restore
        """
        if self.collected_params is None:
            raise RuntimeError("No stored parameters to restore. Call store() first.")
        
        parameters = list(parameters)
        
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
        
        self.collected_params = None
    
    def state_dict(self) -> Dict:
        """Return state dict for serialization."""
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "cur_decay_value": self.cur_decay_value,
            "shadow_params": self.shadow_params,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dict."""
        self.decay = state_dict.get("decay", self.decay)
        self.min_decay = state_dict.get("min_decay", self.min_decay)
        self.optimization_step = state_dict.get("optimization_step", 0)
        self.cur_decay_value = state_dict.get("cur_decay_value", 0.0)
        self.shadow_params = state_dict.get("shadow_params", self.shadow_params)
    
    def to(self, device: torch.device):
        """Move shadow parameters to device."""
        self.shadow_params = [p.to(device) for p in self.shadow_params]
        return self


def create_ema_model(
    model: nn.Module,
    decay: float = 0.9999,
    device: Optional[torch.device] = None,
) -> EMAModel:
    """
    Create an EMA model from a model.
    
    Args:
        model: Model to create EMA from
        decay: EMA decay rate
        device: Device for shadow parameters
    
    Returns:
        EMAModel instance
    """
    return EMAModel(
        parameters=model.parameters(),
        decay=decay,
        device=device,
    )
