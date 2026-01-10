"""
Checkpoint Management.

Handles saving and loading of training checkpoints.
Features:
- Save/load full training state
- Keep only last N checkpoints
- Separate EMA checkpoint saving
- Auto-resume from latest
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages training checkpoints.
    
    Saves and loads:
    - Model weights
    - Optimizer state
    - LR scheduler state
    - EMA weights (optionally separate)
    - Training step/epoch
    - Configuration
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_every_n_steps: int = 5000,
        keep_last_n: int = 5,
        save_ema_separately: bool = True,
        resume_from_latest: bool = True,
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every_n_steps: Save every N training steps
            keep_last_n: Keep only last N checkpoints
            save_ema_separately: Save EMA weights in separate file
            resume_from_latest: Auto-resume from latest checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_every_n_steps = save_every_n_steps
        self.keep_last_n = keep_last_n
        self.save_ema_separately = save_ema_separately
        self.resume_from_latest = resume_from_latest
        
        # Track last saved step to avoid re-saving on resume
        self.last_saved_step = -1
    
    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        ema: Optional[Any] = None,
        config: Optional[dict] = None,
        extra_state: Optional[dict] = None,
        accelerator=None,
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            step: Current training step
            model: Model to save
            optimizer: Optimizer to save
            lr_scheduler: LR scheduler to save
            ema: EMA model (optional)
            config: Training configuration (optional)
            extra_state: Additional state to save (optional)
            accelerator: Accelerate accelerator (for unwrapping models)
        
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint directory for this step
        step_dir = self.checkpoint_dir / f"checkpoint-{step:08d}"
        step_dir.mkdir(exist_ok=True)
        
        # Unwrap model if using accelerator
        if accelerator is not None:
            model = accelerator.unwrap_model(model)
        
        # Save model
        model_path = step_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save optimizer
        optimizer_path = step_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        
        # Save scheduler
        scheduler_path = step_dir / "scheduler.pt"
        torch.save(lr_scheduler.state_dict(), scheduler_path)
        
        # Save EMA
        if ema is not None:
            ema_path = step_dir / "ema.pt"
            if hasattr(ema, 'state_dict'):
                torch.save(ema.state_dict(), ema_path)
            else:
                torch.save({'shadow_params': ema.shadow_params}, ema_path)
        
        # Save training state
        state = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }
        if extra_state:
            state.update(extra_state)
        
        state_path = step_dir / "state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        
        # Save config
        if config is not None:
            config_path = step_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        
        # Update latest pointer
        latest_path = self.checkpoint_dir / "latest"
        with open(latest_path, "w") as f:
            f.write(str(step))
        
        # Track this as the last saved step
        self.last_saved_step = step
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return step_dir
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        ema: Optional[Any] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        device: Optional[torch.device] = None,
    ) -> int:
        """
        Load a checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            lr_scheduler: LR scheduler to load state into (optional)
            ema: EMA model to load state into (optional)
            checkpoint_path: Specific checkpoint to load (uses latest if None)
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            device: Device to load tensors to
        
        Returns:
            Training step of loaded checkpoint
        """
        # Find checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                return 0  # No checkpoint found, start from scratch
        
        checkpoint_path = Path(checkpoint_path)
        
        # Load model
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=device or "cpu")
            model.load_state_dict(state_dict)
        
        # Load optimizer
        if load_optimizer and optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pt"
            if optimizer_path.exists():
                state_dict = torch.load(optimizer_path, map_location=device or "cpu")
                optimizer.load_state_dict(state_dict)
        
        # Load scheduler
        if load_scheduler and lr_scheduler is not None:
            scheduler_path = checkpoint_path / "scheduler.pt"
            if scheduler_path.exists():
                state_dict = torch.load(scheduler_path, map_location=device or "cpu")
                lr_scheduler.load_state_dict(state_dict)
        
        # Load EMA
        if ema is not None:
            ema_path = checkpoint_path / "ema.pt"
            if ema_path.exists():
                state_dict = torch.load(ema_path, map_location=device or "cpu")
                if hasattr(ema, 'load_state_dict'):
                    ema.load_state_dict(state_dict)
                else:
                    ema.shadow_params = state_dict['shadow_params']
        
        # Load training state
        state_path = checkpoint_path / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            step = state.get("step", 0)
            # Track this as the last saved step to avoid re-saving
            self.last_saved_step = step
            return step
        
        return 0
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the latest checkpoint."""
        latest_path = self.checkpoint_dir / "latest"
        
        if not latest_path.exists():
            return None
        
        with open(latest_path) as f:
            step = int(f.read().strip())
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step:08d}"
        
        if checkpoint_path.exists():
            return checkpoint_path
        
        return None
    
    def get_all_checkpoints(self) -> list:
        """Get all checkpoint directories, sorted by step."""
        checkpoints = []
        
        for path in self.checkpoint_dir.glob("checkpoint-*"):
            if path.is_dir():
                try:
                    step = int(path.name.split("-")[1])
                    checkpoints.append((step, path))
                except (ValueError, IndexError):
                    pass
        
        checkpoints.sort(key=lambda x: x[0])
        return [path for _, path in checkpoints]
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = self.get_all_checkpoints()
        
        if len(checkpoints) > self.keep_last_n:
            for old_checkpoint in checkpoints[:-self.keep_last_n]:
                shutil.rmtree(old_checkpoint)
    
    def should_save(self, step: int) -> bool:
        """Check if a checkpoint should be saved at this step."""
        should_save_now = step > 0 and step % self.save_every_n_steps == 0
        # Don't re-save if we just saved this step
        return should_save_now and step != self.last_saved_step
    
    def exists(self) -> bool:
        """Check if any checkpoint exists."""
        return self.get_latest_checkpoint() is not None


def save_model_for_inference(
    model: nn.Module,
    save_path: Union[str, Path],
    half_precision: bool = False,
):
    """
    Save model weights only, optimized for inference.
    
    Args:
        model: Model to save
        save_path: Path to save model
        half_precision: Save in FP16 for smaller file size
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    state_dict = model.state_dict()
    
    if half_precision:
        state_dict = {k: v.half() for k, v in state_dict.items()}
    
    torch.save(state_dict, save_path)
