"""
Distributed training utilities.

Provides helpers for multi-GPU training with Accelerate.
"""

from typing import Optional

import torch


def setup_accelerator(
    config: dict,
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "bf16",
    log_with: Optional[str] = None,
):
    """
    Set up Accelerate for distributed training.
    
    Args:
        config: Training configuration
        gradient_accumulation_steps: Number of gradient accumulation steps
        mixed_precision: Mixed precision mode ("no", "fp16", "bf16")
        log_with: Logging backend ("mlflow", "wandb", "tensorboard", None)
    
    Returns:
        Accelerator instance
    """
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    
    # Get training config
    training_config = config.get("training", {})
    seed = training_config.get("seed", 42)
    
    # Create accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=log_with,
    )
    
    # Set seed for reproducibility
    set_seed(seed)
    
    return accelerator


def get_world_size() -> int:
    """Get the number of processes in distributed training."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def get_rank() -> int:
    """Get the rank of the current process."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def wait_for_everyone():
    """Synchronize all processes."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
    
    Returns:
        Concatenated tensor from all processes
    """
    if not torch.distributed.is_initialized():
        return tensor
    
    world_size = get_world_size()
    
    if world_size == 1:
        return tensor
    
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    
    return torch.cat(tensor_list, dim=0)


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce tensor by mean across all processes.
    
    Args:
        tensor: Tensor to reduce
    
    Returns:
        Reduced tensor
    """
    if not torch.distributed.is_initialized():
        return tensor
    
    world_size = get_world_size()
    
    if world_size == 1:
        return tensor
    
    tensor = tensor.clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / world_size
    
    return tensor


def print_rank_0(message: str, *args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(message, *args, **kwargs)


class DistributedDataParallelKwargs:
    """
    Kwargs for DDP setup.
    """
    
    def __init__(
        self,
        find_unused_parameters: bool = False,
        broadcast_buffers: bool = True,
    ):
        self.find_unused_parameters = find_unused_parameters
        self.broadcast_buffers = broadcast_buffers


def get_effective_batch_size(
    batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int = 1,
) -> int:
    """
    Calculate effective batch size for distributed training.
    
    Args:
        batch_size: Per-GPU batch size
        gradient_accumulation_steps: Gradient accumulation steps
        world_size: Number of GPUs
    
    Returns:
        Effective batch size
    """
    return batch_size * gradient_accumulation_steps * world_size


def prepare_model_for_distributed(
    model: torch.nn.Module,
    accelerator,
):
    """
    Prepare a model for distributed training.
    
    Args:
        model: PyTorch model
        accelerator: Accelerate accelerator
    
    Returns:
        Prepared model
    """
    return accelerator.prepare_model(model)
