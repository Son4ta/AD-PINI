# Logging utilities: thin wrappers around Python logging and optional WandB
import logging
from pathlib import Path
from typing import Dict, Optional
import json
import os

class Logger:
    """
    Unified logging helper.
    
    Features:
    1. Local file logging
    2. Optional WandB online logging
    3. Simple metric history tracking
    """
    
    def __init__(self, log_dir: Path, experiment_name: str, use_wandb: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        
        # Local log file
        self.log_file = self.log_dir / f"{experiment_name}.log"
        
        # Configure Python logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize WandB if requested
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.logger.info("WandB enabled.")
            except ImportError:
                self.logger.warning("WandB not installed; disabling WandB logging.")
                self.use_wandb = False
        
        # Metric history
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
    
    def init_wandb(self, config: Dict, project_name: str = "AD-PINI"):
        """
        Initialize WandB project.
        
        Args:
            config: configuration dictionary
            project_name: WandB project name
        """
        if self.use_wandb:
            self.wandb.init(
                project=project_name,
                name=self.experiment_name,
                config=config
            )
            self.logger.info(f"WandB project initialized: {project_name}")
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """
        Log scalar metrics.
        
        Args:
            metrics: metric dictionary {'loss': 0.5, 'accuracy': 0.9, ...}
            step: global step index (optional)
        """
        # Local log
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}" if step else metrics_str)
        
        # WandB logging
        if self.use_wandb:
            self.wandb.log(metrics, step=step)
        
        # Accumulate history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                 additional_metrics: Optional[Dict] = None):
        """
        Log per-epoch summary.
        
        Args:
            epoch: epoch index
            train_loss: training loss
            val_loss: validation loss
            additional_metrics: any extra metrics to log
        """
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.log_metrics(metrics)
    
    def save_metrics(self):
        """Persist metric history to a JSON file."""
        metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        self.logger.info(f"Metric history saved to: {metrics_file}")
    
    def info(self, message: str):
        """Log an info-level message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log a warning-level message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log an error-level message."""
        self.logger.error(message)
    
    def close(self):
        """Flush metrics, close WandB (if any), and finalize logging."""
        self.save_metrics()
        if self.use_wandb:
            self.wandb.finish()
        self.logger.info("Logger closed.")


def setup_logger(log_path: str, name: str = "carbon_pinp") -> logging.Logger:
    """
    Configure a standard Python logger with file and console handlers.
    
    Args:
        log_path: log file path
        name: logger name
        
    Returns:
        logger: configured logger instance
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Register handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

if __name__ == "__main__":
    # Simple self-test
    logger = Logger(
        log_dir=Path("./logs"),
        experiment_name="test_run",
        use_wandb=False
    )
    
    logger.info("Logger smoke test started.")
    logger.log_metrics({'loss': 0.5, 'accuracy': 0.9}, step=1)
    logger.log_epoch(epoch=1, train_loss=0.5, val_loss=0.6)
    logger.close()
    
    print("Logger test finished.")