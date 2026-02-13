#!/usr/bin/env python3
# AD-PINI v4 main training script - escalator principle anomaly prediction training

import os
import sys
import argparse
import json
import time
from datetime import datetime
import warnings

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# WandB integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Install with 'pip install wandb'")
    WANDB_AVAILABLE = False

import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Project imports
from configs.config_v4 import config
from data.v4.preprocessing_v4 import DataPreprocessorV4
from data.v4.dataset_v4 import create_data_loaders
from models.v4.carbon_net_v4 import CarbonNetV4
from models.v4.loss_v4 import CombinedLossV4
from utils.v4.visualization_v4 import ScientificVisualizerV4
from utils.logger import setup_logger

warnings.filterwarnings('ignore')

def set_random_seeds(seed: int = 42, use_deterministic: bool = True, cuda_deterministic: bool = True):
    """
    Set all random seeds to ensure experiment reproducibility.
    
    Args:
        seed: random seed
        use_deterministic: whether to use deterministic algorithms
        cuda_deterministic: whether to enforce CUDA determinism
    """
    # Python standard library RNG
    import random
    random.seed(seed)
    
    # NumPy RNG
    np.random.seed(seed)
    
    # PyTorch RNG
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
    
    # PyTorch determinism settings
    if use_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    if use_deterministic and cuda_deterministic and torch.cuda.is_available():
        # Ensure deterministic CUDA operations
        torch.use_deterministic_algorithms(True)
        
    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA determinism
    
    print(f"🌱 Random seeds set: {seed} (deterministic: {use_deterministic})")

def setup_distributed():
    """Set up distributed training if requested."""
    distributed = False
    local_rank = 0
    
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        distributed = True
        print(f"Distributed training initialized. Local rank: {local_rank}")
    
    return distributed, local_rank

def init_wandb(config, exp_name: str, use_wandb: bool = False, distributed: bool = False):
    """Initialize WandB experiment tracking."""
    if not use_wandb or not WANDB_AVAILABLE:
        return None
    
    # Initialize wandb only on main process
    if distributed and dist.get_rank() != 0:
        return None
    
    try:
        # WandB configuration
        wandb_config = {
            # Model architecture
            'model_type': 'AD-PINI-v4',
            'model_parameters': 5470443,  # number of parameters after scaling up
            'unet_features': config.UNET_FEATURES,
            'history_length': config.HISTORY_LENGTH,
            'input_channels': config.INPUT_CHANNELS,
            'output_channels': config.OUTPUT_CHANNELS,
            
            # Training configuration
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'gradient_clip': config.GRADIENT_CLIP,
            
            # Loss weights
            'lambda_state': config.LAMBDA_STATE,
            'lambda_task': config.LAMBDA_TASK,
            'lambda_sparsity': config.LAMBDA_SPARSITY,
            
            # Data configuration
            'downsample_factor': config.DOWNSAMPLE_FACTOR,
            'target_lat': config.TARGET_LAT,
            'target_lon': config.TARGET_LON,
            'time_steps': config.TIME_STEPS,
            
            # Optimization strategy
            'warmup_epochs': config.WARMUP_EPOCHS,
            'warmup_lr': config.WARMUP_LR,
            'scheduler_patience': config.SCHEDULER_PATIENCE,
            'scheduler_factor': config.SCHEDULER_FACTOR
        }
        
        # Initialize WandB run
        run = wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=exp_name,
            config=wandb_config,
            tags=config.WANDB_TAGS,
            notes=config.WANDB_NOTES,
            mode='online'  # change to 'offline' to run without network
        )
        
        print(f"✅ WandB initialized: {wandb.run.url}")
        return run
        
    except Exception as e:
        print(f"❌ WandB initialization failed: {e}")
        print("Continue training without logging to WandB...")
        return None


def cleanup_distributed():
    """Clean up distributed training state."""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_experiment_dir(config, exp_name: str) -> tuple:
    """Create experiment directory structure."""
    if exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"carbon_pinp_v4_{timestamp}"
    
    exp_dir = os.path.join(config.OUTPUT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Subdirectories
    subdirs = ['checkpoints', 'logs', 'visualizations', 'metrics']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    return exp_dir, exp_name

def save_config(config, exp_dir: str):
    """Save configuration to JSON."""
    config_dict = {
        attr: getattr(config, attr) for attr in dir(config) 
        if not attr.startswith('_') and not callable(getattr(config, attr))
    }
    
    # Handle non-serializable objects
    for key, value in config_dict.items():
        if isinstance(value, torch.device):
            config_dict[key] = str(value)
        elif hasattr(value, '__dict__'):
            config_dict[key] = str(value)
    
    config_path = os.path.join(exp_dir, 'config_v4.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_model_and_optimizer(config, device, distributed=False):
    """Create model, optimizer, scheduler and criterion."""
    # Create model
    model = CarbonNetV4(config)
    model = model.to(device)
    
    if distributed:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    
    # Optimizer - use AdamW for better regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning-rate scheduler - simplified version
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.MIN_LR,
        verbose=True
    )
    
    # Loss function
    criterion = CombinedLossV4(config)
    
    return model, optimizer, scheduler, criterion

def print_eval_results(val_loss, val_metrics):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("📊 Model evaluation results")
    print("=" * 60)
    print(f"Validation loss:     {val_loss:.6f}")
    
    # Print all key metrics
    metrics_to_show = ['mse', 'mae', 'r2', 'correlation', 'rmse']
    for metric in metrics_to_show:
        if metric in val_metrics:
            value = val_metrics[metric]
            if metric == 'mse':
                print(f"MSE (mean squared error):      {value:.6f}")
            elif metric == 'mae':
                print(f"MAE (mean absolute error):     {value:.6f}")
            elif metric == 'rmse':
                print(f"RMSE (root mean squared error): {value:.6f}")
            elif metric == 'r2':
                print(f"R² (coefficient of determination): {value:.6f}")
            elif metric == 'correlation':
                print(f"Correlation:                    {value:.6f}")
    print("=" * 60)

def validate_model(model, val_loader, criterion, device):
    """Run validation loop and return mean loss and metrics."""
    model.eval()
    val_losses = []
    val_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Compute loss
            targets = {
                'delta_gt': batch['delta_gt'],
                'pco2_target': batch['pco2_target']
            }
            
            loss, _ = criterion(outputs, targets)
            metrics = criterion.compute_metrics(outputs, targets)
            
            val_losses.append(loss.item())
            val_metrics.append(metrics)
            
            # Optionally limit number of validation batches
            if batch_idx >= 50:  # use first 50 batches
                break
    
    # Average validation results
    avg_val_loss = np.mean(val_losses)
    avg_metrics = {}
    for key in val_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in val_metrics])
    
    return avg_val_loss, avg_metrics

def save_checkpoint(model, optimizer, scheduler, epoch, exp_dir, exp_name, is_best=False, metrics=None):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config.__dict__,
        'metrics': metrics  # attach performance metrics
    }
    
    # Regular checkpoint
    checkpoint_path = os.path.join(exp_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Best model
    if is_best:
        best_path = os.path.join(exp_dir, 'checkpoints', f'{exp_name}_best.pth')
        torch.save(checkpoint, best_path)
        
        # Save metadata for best model
        best_info = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': time.time(),
            'validation_loss': metrics.get('val_loss') if metrics else None
        }
        
        best_info_path = os.path.join(exp_dir, 'checkpoints', f'{exp_name}_best_info.json')
        with open(best_info_path, 'w') as f:
            json.dump(best_info, f, indent=2)

def generate_visualizations(model, val_loader, visualizer, epoch, viz_base_dir, device, metrics, preprocessor=None, logger=None):
    """
    Generate scientific visualizations.
    
    Args:
        model: trained model
        val_loader: validation data loader
        visualizer: visualization helper
        epoch: current epoch
        viz_base_dir: base visualization directory
        device: compute device
        metrics: validation metrics dictionary (from training loop)
        preprocessor: data preprocessor
        logger: logger instance
    """
    model.eval()
    
    with torch.no_grad():
        # Take one validation batch
        batch = next(iter(val_loader))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Model prediction
        outputs = model(batch)
        
        # Convert to numpy
        outputs_np = {}
        targets_np = {}
        
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                outputs_np[key] = value.detach().cpu().numpy()
        
        targets_np = {
            'pco2_target': batch['pco2_target'].detach().cpu().numpy(),
            'delta_gt': batch['delta_gt'].detach().cpu().numpy()
        }
        
        # Build lat/lon coordinates
        H, W = outputs_np['pco2_final'].shape[1:]
        lat = np.linspace(-89.875, 89.875, H)  # based on 0.25° grid
        lon = np.linspace(-179.875, 179.875, W)
        
        # Note: metrics are passed as argument and computed in training loop
        
        # Create visualization directory - with epoch subdirectory under given base dir
        viz_dir = os.path.join(viz_base_dir, f'epoch_{epoch}')
        
        # Use scientific visualization system for paper-quality figures
        try:
            visualizer.generate_scientific_visualizations(
                epoch, outputs_np, targets_np, lat, lon, metrics, viz_dir, preprocessor
            )
            
            if logger:
                logger.info(f"Scientific visualizations generated (5 paper-quality figures): {viz_dir}")
            else:
                print(f"Scientific visualizations generated (5 paper-quality figures): {viz_dir}")
                
        except Exception as e:
            if logger:
                logger.warning(f"Visualization generation failed: {e}")
            else:
                print(f"Visualization generation failed: {e}")

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer, logger, wandb_run=None):
    """Train model for one epoch."""
    model.train()
    
    # Set epoch for freeze-control logic
    criterion.set_epoch(epoch)
    if hasattr(model, 'set_epoch'):
        model.set_epoch(epoch)
    elif hasattr(model, 'module') and hasattr(model.module, 'set_epoch'):
        model.module.set_epoch(epoch)
    
    total_loss = 0
    total_batches = len(train_loader)
    log_interval = max(1, total_batches // 10)  # log roughly every 10%
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        
        # Compute loss
        targets = {
            'delta_gt': batch['delta_gt'],
            'pco2_target': batch['pco2_target']
        }
        
        loss, loss_dict = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
        if batch_idx % log_interval == 0:
            progress = 100.0 * batch_idx / total_batches
            # Detailed loss components
            frozen_status = "🧊FROZEN" if loss_dict.get('is_corrector_frozen', False) else "🔥ACTIVE"
            loss_components = f"State: {loss_dict.get('state_loss', 0):.2f}, " \
                             f"Task: {loss_dict.get('task_loss', 0):.2f}, " \
                             f"Sparsity: {loss_dict.get('sparsity_loss', 0):.2f}, " \
                             f"Physics: {loss_dict.get('physics_loss', 0):.2f}"
            
            logger.info(f'Epoch {epoch} [{batch_idx}/{total_batches} ({progress:.1f}%)] {frozen_status}\t'
                       f'Total Loss: {loss.item():.2f} | {loss_components}')
            
            # TensorBoard logging
            if writer:
                global_step = epoch * total_batches + batch_idx
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                for key, value in loss_dict.items():
                    writer.add_scalar(f'Train/{key}', value, global_step)
            
            # WandB logging at reduced frequency
            if wandb_run and batch_idx % (log_interval * 2) == 0:
                global_step = epoch * total_batches + batch_idx
                wandb_train_logs = {
                    'train/step_loss': loss.item(),
                    'train/step': global_step,
                }
                # Add detailed per-component losses
                for key, value in loss_dict.items():
                    if isinstance(value, (int, float)) and not key.startswith('is_'):
                        wandb_train_logs[f'train/{key}'] = value
                
                wandb.log(wandb_train_logs, step=global_step)
    
    avg_loss = total_loss / total_batches
    return avg_loss

def main():
    """Main entry point for AD-PINI v4 training."""
    parser = argparse.ArgumentParser(description='AD-PINI v4 Training')
    parser.add_argument('--exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--force_recompute', action='store_true', help='force recomputing preprocessed data')
    parser.add_argument('--use_wandb', action='store_true', help='enable WandB logging')
    parser.add_argument('--debug', action='store_true', help='debug mode (train only a few epochs)')
    parser.add_argument('--no_vis', action='store_true', help='disable visualizations to speed up training')
    parser.add_argument('--eval_only', action='store_true', help='evaluation-only mode: run validation on resume checkpoint without training')
    
    args = parser.parse_args()
    
    # Adjust config according to CLI flags
    if args.no_vis:
        config.ENABLE_VISUALIZATION = False
    
    # Ablation experiments: read modifications from environment variable
    ablation_config_str = os.environ.get('ABLATION_CONFIG')
    if ablation_config_str:
        try:
            ablation_modifications = json.loads(ablation_config_str)
            print(f"🔬 Ablation mode: applying config modifications {ablation_modifications}")
            
            for key, value in ablation_modifications.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    print(f"   ✓ set {key} = {value}")
                else:
                    print(f"   ⚠️  attribute not found in config: {key}")
                    
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse ablation config: {e}")
    
    # Set random seeds (must be done before other initialization)
    set_random_seeds(
        seed=config.RANDOM_SEED,
        use_deterministic=config.USE_DETERMINISTIC,
        cuda_deterministic=config.CUDA_DETERMINISTIC
    )
    
    # Set up distributed training
    distributed, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create experiment directory
    exp_dir, exp_name = create_experiment_dir(config, args.exp_name)
    
    # Set up logger
    log_path = os.path.join(exp_dir, 'logs', f'{exp_name}.log')
    logger = setup_logger(log_path)
    
    logger.info("=" * 60)
    logger.info("AD-PINI v4 training started")
    logger.info("=" * 60)
    logger.info(f"Experiment name: {exp_name}")
    logger.info(f"Experiment dir:  {exp_dir}")
    logger.info(f"Device:          {device}")
    logger.info(f"Distributed:     {distributed}")
    logger.info(f"Config:\n{config}")
    
        # Initialize WandB (if enabled)
    wandb_run = init_wandb(config, exp_name, 
                          use_wandb=args.use_wandb or config.USE_WANDB, 
                          distributed=distributed)
    
    # Save config
    save_config(config, exp_dir)
    
    # Data preprocessing
    logger.info("Start data preprocessing...")
    preprocessor = DataPreprocessorV4(config)
    _ = preprocessor.process_all(force_recompute=args.force_recompute)
    logger.info("Data preprocessing finished.")
    
    # Build data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    logger.info(f"Number of training batches:   {len(train_loader)}")
    logger.info(f"Number of validation batches: {len(val_loader)}")
    
    # Get normalization statistics
    train_dataset = train_loader.dataset
    norm_stats = train_dataset.get_normalization_stats()
    if norm_stats:
        logger.info("Obtained normalization statistics; will be passed to the model.")
    
    # Load model
    logger.info("Initializing model...")
    model, optimizer, scheduler, criterion = load_model_and_optimizer(
        config, device, distributed
    )
    
    model_info = model.module.get_model_size() if hasattr(model, 'module') else model.get_model_size()
    logger.info(f"Model parameter statistics: {model_info}")
    
    # Set normalization statistics on the model
    if norm_stats:
        if hasattr(model, 'set_normalization_stats'):
            model.set_normalization_stats(norm_stats)
        elif hasattr(model, 'module') and hasattr(model.module, 'set_normalization_stats'):
            model.module.set_normalization_stats(norm_stats)
        logger.info("Normalization statistics set on model.")
    
    # Save experiment metadata (simplified)
    try:
        config_save_path = os.path.join(exp_dir, 'config', 'experiment_config.json')
        os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
        
        experiment_metadata = {
            'experiment_name': exp_name,
            'random_seed': config.RANDOM_SEED,
            'model_config': {
                'history_length': config.HISTORY_LENGTH,
                'unet_features': config.UNET_FEATURES,
                'input_channels': config.INPUT_CHANNELS,
                'output_channels': config.OUTPUT_CHANNELS,
            },
            'training_config': {
                'epochs': config.EPOCHS,
                'learning_rate': config.LEARNING_RATE,
                'batch_size': config.BATCH_SIZE,
                'use_deterministic': config.USE_DETERMINISTIC,
            },
            'args': vars(args)
        }
        
        with open(config_save_path, 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        
        logger.info(f"Experiment metadata saved to: {config_save_path}")
    except Exception as e:
        logger.warning(f"Failed to save experiment metadata: {e}")
    
    # Create scientific visualizer (only when visualization is enabled to save memory)
    if config.ENABLE_VISUALIZATION:
        visualizer = ScientificVisualizerV4(config)
        logger.info("✅ Visualization system enabled")
    else:
        visualizer = None
        logger.info("⚡ Visualization system disabled - maximize training speed")
    
    # TensorBoard
    writer = None
    if SummaryWriter is not None and local_rank == 0:
        try:
            writer = SummaryWriter(os.path.join(exp_dir, 'logs', 'tensorboard'))
        except Exception:
            writer = None
            logger.warning("Failed to create TensorBoard writer; skipping TensorBoard logging.")
    
    # Training loop
    logger.info("Starting training loop...")
    best_val_loss = float('inf')
    best_val_mse = float('inf')  # track best MSE
    start_epoch = 1
    
    # Early stopping
    early_stopping_counter = 0
    best_epoch = 0
    training_history = {'train_loss': [], 'val_loss': [], 'val_mse': []}
    
    # Resume from checkpoint if requested
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        current_epoch = checkpoint['epoch']
        start_epoch = current_epoch + 1
        if args.eval_only:
            logger.info("🔍 Evaluation-only mode: skip training and run validation.")
            logger.info(f"Using checkpoint: {args.resume} (epoch {current_epoch})")
            
            logger.info(f"Syncing model and loss state to epoch {current_epoch}...")
            
            # 1. Sync internal model state (freeze/unfreeze, scheduling, etc.)
            if hasattr(model, 'module'):
                model.module.set_epoch(current_epoch)
            else:
                model.set_epoch(current_epoch)
                
            # 2. Sync loss/metrics state (for dynamic weights)
            if hasattr(criterion, 'set_epoch'):
                criterion.set_epoch(current_epoch)
            
            # Run validation on full validation set
            val_loss, val_metrics = validate_model(
                model, val_loader, criterion, device
            )
            
            # Print results
            print_eval_results(val_loss, val_metrics)
            logger.info("✅ Evaluation finished.")
            
            # Cleanup and exit
            if writer:
                writer.close()
            if distributed:
                cleanup_distributed()
            if wandb_run:
                wandb.finish()
            
            logger.info("Exiting evaluation-only mode.")
            return
    
    try:
        for epoch in range(start_epoch, config.EPOCHS + 1):
            epoch_start_time = time.time()
            
            # Training
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, 
                device, epoch, writer, logger, wandb_run
            )
            
            # Track training loss
            training_history['train_loss'].append(train_loss)
            
            # WandB logging
            if wandb_run:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                }, step=epoch)
            
            # Manual learning-rate warmup
            if epoch <= config.WARMUP_EPOCHS:
                lr_scale = epoch / config.WARMUP_EPOCHS
                warmup_lr = config.WARMUP_LR + (config.LEARNING_RATE - config.WARMUP_LR) * lr_scale
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Validation
            if epoch % config.EVAL_EVERY == 0:
                val_loss, val_metrics = validate_model(
                    model, val_loader, criterion, device
                )
                training_history['val_loss'].append(val_loss)
                
                # WandB validation logging
                if wandb_run:
                    wandb_val_logs = {
                        'val/loss': val_loss,
                        **{f'val/{k}': v for k, v in val_metrics.items()}
                    }
                    wandb.log(wandb_val_logs, step=epoch)
                
                # LR scheduling (ReduceLROnPlateau after warmup)
                if epoch > config.WARMUP_EPOCHS:
                    scheduler.step(val_loss)
                
                # Log validation results
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch} validation results (LR: {current_lr:.2e}):")
                logger.info(f"  train_loss: {train_loss:.6f}")
                logger.info(f"  val_loss:   {val_loss:.6f}")
                for key, value in val_metrics.items():
                    logger.info(f"  {key}: {value:.6f}")
                
                # TensorBoard logging
                if writer:
                    writer.add_scalar('Train/Loss', train_loss, epoch)
                    writer.add_scalar('Validation/Loss', val_loss, epoch)
                    writer.add_scalar('Learning_Rate', current_lr, epoch)
                    for key, value in val_metrics.items():
                        writer.add_scalar(f'Validation/{key}', value, epoch)
                
                # Early stopping and best model check - using MSE as primary metric
                is_best = False
                current_mse = val_metrics.get('mse', float('inf'))
                
                # Update training history
                training_history['train_loss'].append(train_loss)
                training_history['val_loss'].append(val_loss)
                training_history['val_mse'].append(current_mse)
                
                # Use MSE to determine best model
                if current_mse < (best_val_mse - config.EARLY_STOPPING_MIN_DELTA):
                    best_val_loss = val_loss
                    best_val_mse = current_mse
                    best_epoch = epoch
                    early_stopping_counter = 0
                    is_best = True
                    logger.info(f"🎉 New best model! MSE: {current_mse:.6f}, val_loss: {val_loss:.6f}")
                    
                    # Generate detailed visualization for best model (if enabled)
                    if local_rank == 0:
                        if config.ENABLE_VISUALIZATION:
                            logger.info(f"Generating detailed visualizations for best model (epoch {epoch})...")
                            try:
                                best_viz_dir = os.path.join(exp_dir, 'visualizations', 'best_model')
                                generate_visualizations(
                                    model, val_loader, visualizer, epoch, best_viz_dir, device, val_metrics, preprocessor, logger
                                )
                                logger.info(f"Best-model visualizations saved to: {best_viz_dir}")
                            except Exception as e:
                                logger.warning(f"Best-model visualization generation failed: {e}")
                        else:
                            logger.info(f"Skip best-model (epoch {epoch}) visualization (ENABLE_VISUALIZATION=False)")
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE and config.EARLY_STOPPING:
                        logger.info(f"⏹️  Early stopping triggered! Best epoch: {best_epoch}")
                        logger.info(f"   Best MSE: {best_val_mse:.6f}, best val_loss: {best_val_loss:.6f}")
                        logger.info(f"   Epochs since last improvement: {early_stopping_counter}")
                        break
                
            else:
                is_best = False
            
            # Save checkpoint
            if epoch % config.SAVE_EVERY == 0 or is_best:
                if local_rank == 0:  # only save from main process
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, 
                        exp_dir, exp_name, is_best
                    )
            
            # Generate visualizations (config-controlled)
            if config.ENABLE_VISUALIZATION and epoch % config.VISUALIZE_EVERY == 0 and local_rank == 0:
                logger.info(f"Generating visualizations for epoch {epoch}...")
                try:
                    viz_base_dir = os.path.join(exp_dir, 'visualizations')
                    generate_visualizations(
                        model, val_loader, visualizer, epoch, viz_base_dir, device, val_metrics, preprocessor, logger
                    )
                except Exception as e:
                    logger.warning(f"Visualization generation failed: {e}")
            elif not config.ENABLE_VISUALIZATION and epoch % config.VISUALIZE_EVERY == 0 and local_rank == 0:
                logger.info(f"Skip visualizations for epoch {epoch} (ENABLE_VISUALIZATION=False)")
            
            # Timing statistics
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s, "
                       f"train_loss: {train_loss:.6f}")
            
            # In debug mode, stop after a few epochs
            if args.debug and epoch >= 3:
                logger.info("Debug mode enabled; stopping early after 3 epochs.")
                break
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    finally:
        # Cleanup
        if writer:
            writer.close()
        
        if distributed:
            cleanup_distributed()
        
        logger.info("Training finished.")
        
        # Finish WandB run
        if wandb_run:
            wandb.finish()
            
        logger.info("=" * 60)

if __name__ == "__main__":
    main()