"""
MLflow Logging Utilities.

Provides a wrapper for MLflow tracking with:
- Experiment management
- Metric logging
- Image artifact logging
- Model checkpointing
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image


class MLflowLogger:
    """
    Wrapper for MLflow logging.
    
    Provides a consistent interface for:
    - Logging training metrics
    - Logging evaluation metrics
    - Logging sample images
    - Logging configuration
    """
    
    def __init__(
        self,
        config: dict,
    ):
        """
        Args:
            config: MLflow configuration dict with keys:
                - enabled: Whether to enable MLflow logging
                - tracking_uri: MLflow tracking server URI
                - experiment_name: Experiment name
                - run_name: Run name (optional, auto-generated if not provided)
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        
        if not self.enabled:
            return
        
        import mlflow
        
        # Set tracking URI
        tracking_uri = config.get("tracking_uri", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        experiment_name = config.get("experiment_name", "stable-diffusion")
        mlflow.set_experiment(experiment_name)
        
        # Start run
        run_name = config.get("run_name")
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add Slurm job ID if available
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            run_name = f"{run_name}_job{slurm_job_id}"
        
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id
        
        self.mlflow = mlflow
    
    def log_config(self, config: dict):
        """
        Log training configuration as parameters.
        
        Args:
            config: Configuration dictionary (will be flattened)
        """
        if not self.enabled:
            return
        
        from .config import config_to_flat_dict
        
        flat_config = config_to_flat_dict(config)
        
        # MLflow has a 500 parameter limit, so we may need to truncate
        for key, value in flat_config.items():
            try:
                # Convert to string if needed
                if isinstance(value, (list, tuple)):
                    value = str(value)
                
                self.mlflow.log_param(key, value)
            except Exception as e:
                print(f"Failed to log param {key}: {e}")
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
    ):
        """
        Log a single metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
        """
        if not self.enabled:
            return
        
        self.mlflow.log_metric(name, value, step=step)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Training step
        """
        if not self.enabled:
            return
        
        self.mlflow.log_metrics(metrics, step=step)
    
    def log_images(
        self,
        images: Union[Dict[str, Image.Image], List[Image.Image]],
        step: int,
        prefix: str = "samples",
        labels: Optional[List[str]] = None,
    ):
        """
        Log images as artifacts.
        
        Args:
            images: Dictionary mapping names to images, or list of images
            step: Training step
            prefix: Artifact directory prefix
            labels: Optional list of text labels for images (creates labeled grid)
        """
        if not self.enabled:
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            if isinstance(images, dict):
                # Save individual images
                for name, image in images.items():
                    # Sanitize name for filename
                    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
                    safe_name = safe_name[:50]  # Truncate
                    
                    image_path = tmpdir / f"{safe_name}.png"
                    image.save(image_path)
                
                # Create grid with labels if provided
                image_list = list(images.values())
                if labels is not None and len(labels) == len(image_list):
                    from ..evaluation.sample_generator import create_labeled_image_grid
                    grid = create_labeled_image_grid(image_list, labels)
                else:
                    # Extract labels from keys or use "Text: ''" for no text
                    grid_labels = []
                    for key in images.keys():
                        # Try to extract meaningful text from key
                        # Format is usually "00_some_text" so we take everything after first underscore
                        parts = key.split("_", 1)
                        if len(parts) > 1 and parts[1]:
                            grid_labels.append(parts[1].replace("_", " "))
                        else:
                            grid_labels.append("Text: ''")
                    
                    from ..evaluation.sample_generator import create_labeled_image_grid
                    grid = create_labeled_image_grid(image_list, grid_labels)
                
                grid.save(tmpdir / "grid.png")
            
            else:
                # List of images
                for i, image in enumerate(images):
                    image_path = tmpdir / f"sample_{i:03d}.png"
                    image.save(image_path)
                
                # Create grid with labels if provided
                if labels is not None and len(labels) == len(images):
                    from ..evaluation.sample_generator import create_labeled_image_grid
                    grid = create_labeled_image_grid(images, labels)
                else:
                    # No labels provided, use "Text: ''"
                    grid_labels = ["Text: ''" for _ in images]
                    from ..evaluation.sample_generator import create_labeled_image_grid
                    grid = create_labeled_image_grid(images, grid_labels)
                
                grid.save(tmpdir / "grid.png")
            
            # Log artifacts
            artifact_path = f"{prefix}/step_{step:08d}"
            self.mlflow.log_artifacts(str(tmpdir), artifact_path=artifact_path)
    
    def log_image_grid(
        self,
        grid: Image.Image,
        step: int,
        name: str = "samples",
    ):
        """
        Log a single image grid.
        
        Args:
            grid: Grid image
            step: Training step
            name: Artifact name
        """
        if not self.enabled:
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            grid_path = Path(tmpdir) / f"{name}_step_{step:08d}.png"
            grid.save(grid_path)
            
            self.mlflow.log_artifact(str(grid_path), artifact_path="samples")
    
    def log_model(
        self,
        model,
        artifact_path: str = "model",
    ):
        """
        Log a PyTorch model.
        
        Args:
            model: PyTorch model
            artifact_path: Artifact path
        """
        if not self.enabled:
            return
        
        self.mlflow.pytorch.log_model(model, artifact_path)
    
    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ):
        """
        Log a local file or directory as an artifact.
        
        Args:
            local_path: Local path to file or directory
            artifact_path: Destination path in artifacts
        """
        if not self.enabled:
            return
        
        self.mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
    
    def set_tags(self, tags: Dict[str, str]):
        """
        Set run tags.
        
        Args:
            tags: Dictionary of tag names to values
        """
        if not self.enabled:
            return
        
        self.mlflow.set_tags(tags)
    
    def end_run(self):
        """End the current MLflow run."""
        if not self.enabled:
            return
        
        self.mlflow.end_run()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
