# AD-PINI v4 WandB logger
# Provides comprehensive experiment tracking and visualization utilities

import os
import torch
import wandb
from typing import Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class WandbLogger:
    """
    WandB experiment logger tailored for AD-PINI v4.
    """
    
    def __init__(self, 
                 config, 
                 project_name: str = "AD-PINI-v4",
                 experiment_name: Optional[str] = None,
                 entity: Optional[str] = None,
                 tags: Optional[list] = None):
        """
        Initialize WandB logger.
        
        Args:
            config: configuration object
            project_name: WandB project name
            experiment_name: experiment name
            entity: WandB username / entity
            tags: experiment tags
        """
        self.config = config
        self.enabled = getattr(config, 'USE_WANDB', False)
        
        if not self.enabled:
            print("📊 WandB logging disabled by config.")
            return
            
        # Determine experiment name
        if experiment_name is None:
            experiment_name = f"carbon_pinp_v4_{datetime.now().strftime('%m%d_%H%M')}"
        
        # Prepare config dict
        wandb_config = self._prepare_config_dict(config)
        
        # Initialize wandb
        try:
            self.run = wandb.init(
                project=project_name,
                name=experiment_name,
                entity=entity,
                config=wandb_config,
                tags=tags or getattr(config, 'WANDB_TAGS', []),
                notes=getattr(config, 'WANDB_NOTES', ''),
                reinit=True
            )
            print(f"✅ WandB initialized - project: {project_name}, experiment: {experiment_name}")
            
            # Optionally log basic model info
            if getattr(config, 'WANDB_LOG_MODEL_TOPOLOGY', True):
                self._log_model_info()
                
        except Exception as e:
            print(f"⚠️  Failed to initialize WandB: {e}")
            self.enabled = False
            self.run = None
    
    def _prepare_config_dict(self, config) -> Dict[str, Any]:
        """Prepare configuration dictionary to pass to WandB."""
        config_dict = {}
        
        # Model parameters
        config_dict.update({
            'model_architecture': 'CarbonNetV4',
            'history_length': config.HISTORY_LENGTH,
            'unet_features': config.UNET_FEATURES,
            'corrector_features': config.CORRECTOR_FEATURES,
            'corrector_freeze_epochs': config.CORRECTOR_FREEZE_EPOCHS,
        })
        
        # Training hyperparameters
        config_dict.update({
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'gradient_clip': config.GRADIENT_CLIP,
            'warmup_epochs': config.WARMUP_EPOCHS,
        })
        
        # Loss weights
        config_dict.update({
            'lambda_state': config.LAMBDA_STATE,
            'lambda_task': config.LAMBDA_TASK,
            'lambda_sparsity': config.LAMBDA_SPARSITY,
            'lambda_physics': getattr(config, 'LAMBDA_PHYSICS', 0.01),
            'lambda_constraint': getattr(config, 'LAMBDA_CONSTRAINT', 0.05),
        })
        
        # Data configuration
        config_dict.update({
            'normalization_mode': config.NORMALIZATION_MODE,
            'downsample_factor': config.DOWNSAMPLE_FACTOR,
            'target_size': f"{config.TARGET_LAT}x{config.TARGET_LON}",
        })
        
        return config_dict
    
    def _log_model_info(self):
        """Log high-level model info into WandB config."""
        if not self.enabled:
            return
            
        try:
            # Here we can add architecture diagrams or parameter statistics
            model_info = {
                'model_type': 'Physics-Informed Neural Network',
                'framework': 'PyTorch',
                'architecture': 'Anomaly Decomposition + Physics Inversion',
                'total_parameters': 'TBD',  # to be updated in training script
            }
            self.run.config.update(model_info)
        except Exception as e:
            print(f"⚠️  Failed to log model info: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None, prefix: str = ""):
        """Log scalar metrics, with optional prefix."""
        if not self.enabled:
            return
            
        try:
            # Apply prefix if requested
            if prefix:
                metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            
            self.run.log(metrics, step=step)
        except Exception as e:
            print(f"⚠️  Failed to log metrics: {e}")
    
    def log_loss_components(self, loss_dict: Dict[str, float], epoch: int, phase: str = "train"):
        """Log per-component losses and their relative ratios."""
        if not self.enabled:
            return
            
        try:
            # Assemble metrics
            loss_metrics = {}
            
            # Raw loss values
            for key, value in loss_dict.items():
                if isinstance(value, (int, float)):
                    loss_metrics[f"loss/{phase}_{key}"] = value
            
            # Ratios for analyzing relative contributions
            total_loss = loss_dict.get('total_loss', 1.0)
            if total_loss > 0:
                for key, value in loss_dict.items():
                    if key != 'total_loss' and isinstance(value, (int, float)):
                        loss_metrics[f"loss_ratio/{phase}_{key}_ratio"] = value / total_loss
            
            self.log_metrics(loss_metrics, step=epoch)
            
        except Exception as e:
            print(f"⚠️  Failed to log loss components: {e}")
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, epoch: int):
        """Log learning rate(s) from optimizer parameter groups."""
        if not self.enabled:
            return
            
        try:
            # Main learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log all parameter-group LRs
            lr_metrics = {'lr/main': current_lr}
            for i, param_group in enumerate(optimizer.param_groups):
                lr_metrics[f'lr/group_{i}'] = param_group['lr']
            
            self.log_metrics(lr_metrics, step=epoch)
            
        except Exception as e:
            print(f"⚠️  Failed to log learning rate: {e}")
    
    def log_gradients(self, model: torch.nn.Module, epoch: int):
        """Log gradient histograms for selected modules."""
        if not self.enabled or not getattr(self.config, 'WANDB_LOG_GRADIENTS', True):
            return
            
        try:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Log gradients only for main modules
                    if any(module in name for module in ['anomaly_net', 'corrector', 'physics']):
                        self.run.log({
                            f"gradients/{name.replace('.', '_')}_grad": wandb.Histogram(param.grad.detach().cpu().numpy())
                        }, step=epoch)
        except Exception as e:
            print(f"⚠️  Failed to log gradients: {e}")
    
    def log_model_parameters(self, model: torch.nn.Module, epoch: int = 0):
        """Log parameter counts and per-module parameter statistics."""
        if not self.enabled:
            return
            
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            param_stats = {
                'model/total_parameters': total_params,
                'model/trainable_parameters': trainable_params,
                'model/frozen_parameters': total_params - trainable_params,
                'model/parameter_ratio': trainable_params / total_params if total_params > 0 else 0
            }
            
            # Per-module parameter counts
            for name, module in model.named_children():
                module_params = sum(p.numel() for p in module.parameters())
                param_stats[f'model/{name}_parameters'] = module_params
            
            self.log_metrics(param_stats, step=epoch)
            
            # Update config with total parameter count
            self.run.config.update({'total_parameters': total_params})
            
        except Exception as e:
            print(f"⚠️  Failed to log model parameter statistics: {e}")
    
    def log_epoch_summary(self, epoch: int, train_loss: float, val_loss: float, 
                         train_metrics: Dict[str, float] = None, 
                         val_metrics: Dict[str, float] = None):
        """Log a concise epoch summary."""
        if not self.enabled:
            return
            
        try:
            summary_metrics = {
                'epoch': epoch,
                'loss/train': train_loss,
                'loss/val': val_loss,
                'loss/val_train_ratio': val_loss / train_loss if train_loss > 0 else 0
            }
            
            # Add training metrics
            if train_metrics:
                for key, value in train_metrics.items():
                    summary_metrics[f'train/{key}'] = value
            
            # Add validation metrics
            if val_metrics:
                for key, value in val_metrics.items():
                    summary_metrics[f'val/{key}'] = value
            
            self.log_metrics(summary_metrics, step=epoch)
            
        except Exception as e:
            print(f"⚠️  Failed to log epoch summary: {e}")
    
    def watch_model(self, model: torch.nn.Module):
        """Attach model to WandB for automatic gradient/parameter logging."""
        if not self.enabled:
            return
            
        try:
            self.run.watch(model, log='all', log_freq=getattr(self.config, 'WANDB_LOG_FREQ', 100))
        except Exception as e:
            print(f"⚠️  Failed to set up model watching: {e}")
    
    def finish(self):
        """Finish WandB run."""
        if self.enabled and self.run is not None:
            try:
                self.run.finish()
                print("✅ WandB run closed.")
            except Exception as e:
                print(f"⚠️  Failed to close WandB run: {e}")
    
    def log_system_metrics(self):
        """Log basic system metrics (currently GPU utilization)."""
        if not self.enabled:
            return
            
        try:
            # GPU information
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                max_memory = torch.cuda.max_memory_allocated() / 1024**3   # GB
                
                system_metrics = {
                    'system/gpu_count': gpu_count,
                    'system/gpu_memory_current_gb': current_memory,
                    'system/gpu_memory_max_gb': max_memory,
                    'system/gpu_memory_utilization': current_memory / (torch.cuda.get_device_properties(0).total_memory / 1024**3)
                }
                
                self.log_metrics(system_metrics)
                
        except Exception as e:
            print(f"⚠️  Failed to log system metrics: {e}")

def create_wandb_logger(config, 
                       experiment_name: Optional[str] = None,
                       **kwargs) -> WandbLogger:
    """
    Convenience wrapper to construct a WandbLogger.
    
    Args:
        config: configuration object
        experiment_name: experiment name override
        **kwargs: additional args forwarded to WandbLogger
        
    Returns:
        WandbLogger instance
    """
    # Merge tags from config and kwargs
    config_tags = getattr(config, 'WANDB_TAGS', [])
    kwarg_tags = kwargs.pop('tags', [])
    combined_tags = config_tags + kwarg_tags
    
    return WandbLogger(
        config=config,
        project_name=getattr(config, 'WANDB_PROJECT', 'AD-PINI-v4'),
        experiment_name=experiment_name,
        entity=getattr(config, 'WANDB_ENTITY', None),
        tags=combined_tags,
        **kwargs
    )