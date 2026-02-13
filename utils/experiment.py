"""
Experiment management utilities

Handle experiment directory creation, config persistence, and metric saving.
"""

from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

def setup_experiment_dir(output_dir: Path, exp_name: str) -> Path:
    """
    Set up experiment output directory structure.
    
    Layout:
    outputs/
    └── {exp_name}/
        ├── checkpoints/          # model checkpoints
        ├── logs/                 # training logs
        ├── visualizations/       # figure outputs
        ├── predictions/          # prediction CSVs
        ├── metrics/              # JSON metric dumps
        └── config.json           # experiment configuration
    
    Args:
        output_dir: root output directory
        exp_name: experiment name
        
    Returns:
        exp_dir: experiment root directory
    """
    exp_dir = output_dir / exp_name
    
    # Create standard subdirectories
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (exp_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (exp_dir / "metrics").mkdir(parents=True, exist_ok=True)
    
    return exp_dir

def save_experiment_config(exp_dir: Path, args: Any, config_dict: Dict) -> None:
    """
    Save experiment configuration to config.json.
    
    Args:
        exp_dir: experiment directory
        args: argparse.Namespace with CLI args
        config_dict: configuration dictionary
    """
    config_info = {
        'experiment_name': args.exp_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'command_line_args': vars(args),
        'model_config': config_dict,
    }
    
    config_file = exp_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config_info, f, indent=2)
    
    print(f"✓ Experiment configuration saved to: {config_file}")

def save_metrics_json(metrics_dir: Path, metrics: Dict, epoch: int) -> None:
    """
    Save metrics for a given epoch as JSON.
    
    Args:
        metrics_dir: directory in which to save metrics
        metrics: metrics dictionary
        epoch: epoch index
    """
    # Per-epoch metrics
    metrics_file = metrics_dir / f"metrics_epoch_{epoch}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Also store latest metrics for quick inspection
    latest_metrics_file = metrics_dir / "metrics_latest.json"
    with open(latest_metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics_file

def get_experiment_info(exp_dir: Path) -> Dict:
    """
    Load experiment configuration information.
    
    Args:
        exp_dir: experiment directory
        
    Returns:
        config_dict: configuration dictionary (or empty if missing)
    """
    config_file = exp_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        return {}

def list_experiment_epochs(exp_dir: Path) -> list:
    """
    List all epochs for which metrics exist.
    
    Args:
        exp_dir: experiment directory
        
    Returns:
        epochs: sorted list of epoch indices
    """
    metrics_dir = exp_dir / "metrics"
    if not metrics_dir.exists():
        return []
    
    epochs = []
    for file in metrics_dir.glob("metrics_epoch_*.json"):
        epoch_num = int(file.stem.split('_')[-1])
        epochs.append(epoch_num)
    
    return sorted(epochs)

if __name__ == "__main__":
    # Simple manual test
    from pathlib import Path
    import argparse
    
    print("Testing experiment utilities...")
    
    # Create temporary experiment
    test_dir = Path("/tmp/test_exp")
    exp_dir = setup_experiment_dir(test_dir, "test_experiment")
    print(f"✓ Experiment directory: {exp_dir}")
    
    # Simulate CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='test_experiment')
    parser.add_argument('--multi_gpu', action='store_true')
    args = parser.parse_args(['--exp_name', 'test_experiment'])
    
    config_dict = {
        'batch_size': 2,
        'learning_rate': 1e-4,
    }
    
    save_experiment_config(exp_dir, args, config_dict)
    print("✓ Experiment configuration saved.")
    
    # Save metrics
    metrics = {
        'real_rmse': 5.5,
        'real_mae': 4.2,
        'norm_rmse': 0.3,
        'norm_mae': 0.2,
    }
    save_metrics_json(exp_dir / "metrics", metrics, epoch=1)
    print("✓ Metrics saved.")
    
    # Load configuration
    config = get_experiment_info(exp_dir)
    print(f"✓ Loaded experiment name: {config['experiment_name']}")
    
    # List epochs
    epochs = list_experiment_epochs(exp_dir)
    print(f"✓ Epochs: {epochs}")
    
    print("\n✓ Experiment utilities test finished.")