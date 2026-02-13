"""
Training utility functions for AD-PINI model.

This module contains refactored training functions to improve code organization
and reduce duplication between single-GPU and multi-GPU training.
"""
import os
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.tensor_utils import safe_to_device
from models.constants import PhysicsConstants


def setup_experiment(args, config):
    """
    Setup experiment directory and configuration.
    
    Args:
        args: Command line arguments
        config: Configuration object
        
    Returns:
        Experiment directory path
    """
    from utils.experiment import setup_experiment_dir, save_experiment_config
    
    exp_dir = setup_experiment_dir(config.OUTPUT_DIR, args.exp_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Save experiment configuration
    config_dict = {
        'version': 'v2',
        'data_shape': config.TARGET_DIMENSIONS,
        'input_var': config.PRIMARY_INPUT_VARIABLE,
        'active_vars': config.AUXILIARY_VARIABLES,
        'target_var': config.TARGET_VARIABLE,
        'input_length': config.INPUT_LENGTH,
        'output_length': config.OUTPUT_LENGTH,
        'predict_length': config.PREDICT_LENGTH,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'num_epochs': config.NUM_EPOCHS,
        'base_channels': config.BASE_CHANNELS,
        'num_input_channels': config.NUM_INPUT_CHANNELS,
        'num_physics_heads': config.NUM_PHYSICS_HEADS,
        'lambda_param': config.LAMBDA_PARAM,
        'lambda_state': config.LAMBDA_STATE,
        'lambda_data': config.LAMBDA_DATA,
        'lambda_physics': config.LAMBDA_PHYSICS,
    }
    save_experiment_config(exp_dir, args, config_dict)
    
    return exp_dir, config_dict


def create_datasets(data, config, use_distributed=False, rank=0, world_size=1):
    """
    Create train and validation datasets with proper splitting.
    
    Args:
        data: Preprocessed data dictionary
        config: Configuration object
        use_distributed: Whether to use distributed sampling
        rank: Process rank (for distributed training)
        world_size: Total number of processes
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from data.dataset import CarbonDataset
    
    # Split data
    train_size = int(len(data['X']) * 0.8)
    indices = np.arange(len(data['X']))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create datasets
    train_dataset = CarbonDataset(
        X=data['X'][train_indices],
        Y_target=data['Y_target'][train_indices],
        Y_delta_sst=data['Y_delta_sst'][train_indices],
        Y_delta_dic=data['Y_delta_dic'][train_indices],
        Y_beta=data['Y_beta'][train_indices],
        mask=data['mask'],
        stats=data['stats']
    )
    
    val_dataset = CarbonDataset(
        X=data['X'][val_indices],
        Y_target=data['Y_target'][val_indices],
        Y_delta_sst=data['Y_delta_sst'][val_indices],
        Y_delta_dic=data['Y_delta_dic'][val_indices],
        Y_beta=data['Y_beta'][val_indices],
        mask=data['mask'],
        stats=data['stats']
    )
    
    # Create samplers and data loaders
    train_sampler = None
    if use_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    if rank == 0:  # Only print from main process
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def initialize_model_and_optimizer(config, data_stats, device):
    """
    Initialize model, loss function, and optimizer.
    
    Args:
        config: Configuration object
        data_stats: Data normalization statistics
        device: Device to place model on
        
    Returns:
        Tuple of (model, criterion, optimizer, scheduler)
    """
    from models.carbon_net import CarbonPINP
    from models.loss import CarbonLoss
    
    # Create model
    model = CarbonPINP(config, data_stats).to(device)
    
    # Create loss function
    criterion = CarbonLoss(
        lambda_param=config.LAMBDA_PARAM,
        lambda_state=config.LAMBDA_STATE,
        lambda_data=config.LAMBDA_DATA,
        lambda_physics=config.LAMBDA_PHYSICS
    )
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    return model, criterion, optimizer, scheduler


def train_single_epoch(model, dataloader, criterion, optimizer, device, epoch, logger=None, rank=0):
    """
    Train model for one epoch with improved error handling.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        logger: Logger instance (optional)
        rank: Process rank
        
    Returns:
        Tuple of average losses (total, param, state, data, physics)
    """
    model.train()
    
    total_loss = 0.0
    total_loss_param = 0.0
    total_loss_state = 0.0
    total_loss_data = 0.0
    total_loss_physics = 0.0
    num_batches = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        try:
            # Unpack batch data
            X, Y_target, Y_delta_sst, Y_delta_dic, Y_beta, mask = batch_data
            
            # Move data to device with error handling
            X = safe_to_device(X, device)
            Y_target = safe_to_device(Y_target, device)
            Y_delta_sst = safe_to_device(Y_delta_sst, device)
            Y_delta_dic = safe_to_device(Y_delta_dic, device)
            Y_beta = safe_to_device(Y_beta, device)
            mask = safe_to_device(mask, device)
            
            # Forward pass
            physics_params, pCO2_phy, pCO2_final = model(X, mask)
            
            # Calculate loss
            loss, loss_dict = criterion(
                physics_params, pCO2_phy, pCO2_final,
                Y_delta_sst, Y_delta_dic, Y_beta, Y_target, mask
            )
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=PhysicsConstants.GRADIENT_CLIP_MAX_NORM
            )
            
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict['total']
            total_loss_param += loss_dict['param']
            total_loss_state += loss_dict['state']
            total_loss_data += loss_dict['data']
            total_loss_physics += loss_dict['physics']
            num_batches += 1
            
            # Log progress
            if rank == 0 and batch_idx % 10 == 0 and logger:
                logger.info(
                    f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss_dict['total']:.4f} "
                    f"(Param: {loss_dict['param']:.4f}, "
                    f"State: {loss_dict['state']:.4f}, "
                    f"Data: {loss_dict['data']:.4f}, "
                    f"Physics: {loss_dict['physics']:.4f})"
                )
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                if rank == 0:
                    print(f"Warning: OOM at batch {batch_idx}, skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    # Calculate average losses
    if num_batches == 0:
        raise RuntimeError("No batches processed successfully")
    
    avg_losses = (
        total_loss / num_batches,
        total_loss_param / num_batches,
        total_loss_state / num_batches,
        total_loss_data / num_batches,
        total_loss_physics / num_batches
    )
    
    return avg_losses


def validate_model(model, dataloader, criterion, device, rank=0):
    """
    Validate model with improved error handling.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use
        rank: Process rank
        
    Returns:
        Average validation loss
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            try:
                # Unpack and move data
                X, Y_target, Y_delta_sst, Y_delta_dic, Y_beta, mask = batch_data
                
                X = safe_to_device(X, device)
                Y_target = safe_to_device(Y_target, device)
                Y_delta_sst = safe_to_device(Y_delta_sst, device)
                Y_delta_dic = safe_to_device(Y_delta_dic, device)
                Y_beta = safe_to_device(Y_beta, device)
                mask = safe_to_device(mask, device)
                
                # Forward pass
                physics_params, pCO2_phy, pCO2_final = model(X, mask)
                
                # Calculate loss
                loss, loss_dict = criterion(
                    physics_params, pCO2_phy, pCO2_final,
                    Y_delta_sst, Y_delta_dic, Y_beta, Y_target, mask
                )
                
                total_loss += loss_dict['total']
                num_batches += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if rank == 0:
                        print("Warning: OOM during validation, skipping batch...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    if num_batches == 0:
        raise RuntimeError("No validation batches processed successfully")
    
    return total_loss / num_batches


def setup_distributed_training(rank, world_size):
    """
    Setup distributed training environment.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed_training():
    """Clean up distributed training environment."""
    dist.destroy_process_group()