"""
MLflow logging utilities.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Any
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# MLflow experiment names
EXPERIMENTS = {
    "vqvae_training": "vqvae-training",
    "pretrained_vqvae": "latent-gpt-pretrained-vqvae",
    "custom_vqvae": "latent-gpt-custom-vqvae",
}


def setup_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[dict] = None,
) -> str:
    """
    Setup MLflow tracking for an experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI (default: from env or localhost)
        run_name: Optional run name (default: timestamp-based)
        tags: Optional tags to set on the run
        
    Returns:
        Run ID
        
    Example:
        >>> run_id = setup_mlflow(
        ...     experiment_name="latent-gpt-pretrained-vqvae",
        ...     tags={"resolution": "256", "phase": "1"}
        ... )
    """
    # Set tracking URI
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Generate run name if not provided
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start run
    run = mlflow.start_run(run_name=run_name)
    
    # Set tags
    if tags:
        mlflow.set_tags(tags)
    
    return run.info.run_id


def log_config(config: Any):
    """
    Log configuration to MLflow.
    
    Args:
        config: Configuration object with to_dict() method or dict
    """
    if hasattr(config, "to_dict"):
        params = config.to_dict()
    elif isinstance(config, dict):
        params = config
    else:
        params = vars(config)
    
    mlflow.log_params(params)


def log_metrics(metrics: dict, step: Optional[int] = None):
    """
    Log metrics to MLflow.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number
    """
    mlflow.log_metrics(metrics, step=step)


def log_model(model, artifact_path: str = "model"):
    """
    Log PyTorch model to MLflow.
    
    Args:
        model: PyTorch model
        artifact_path: Path in artifacts to store model
    """
    mlflow.pytorch.log_model(model, artifact_path)


def log_artifact(local_path: str | Path, artifact_path: Optional[str] = None):
    """
    Log artifact file to MLflow.
    
    Args:
        local_path: Local path to file
        artifact_path: Optional destination path in artifacts
    """
    mlflow.log_artifact(str(local_path), artifact_path)


def log_image(image, artifact_name: str, step: Optional[int] = None):
    """
    Log image to MLflow.
    
    Args:
        image: PIL Image or numpy array
        artifact_name: Name for the artifact
        step: Optional step number (appended to name)
    """
    import tempfile
    from PIL import Image
    import numpy as np
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    # Add step to name if provided
    if step is not None:
        name_parts = artifact_name.rsplit(".", 1)
        if len(name_parts) == 2:
            artifact_name = f"{name_parts[0]}_step{step}.{name_parts[1]}"
        else:
            artifact_name = f"{artifact_name}_step{step}"
    
    # Save and log
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f.name)
        mlflow.log_artifact(f.name, "images")
        os.unlink(f.name)


def end_run():
    """End the current MLflow run."""
    mlflow.end_run()


def get_experiment_runs(experiment_name: str) -> list:
    """
    Get all runs for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        List of run info dictionaries
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        return []
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    
    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        }
        for run in runs
    ]
