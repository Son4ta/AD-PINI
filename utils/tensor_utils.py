"""
Utility functions for tensor operations and transformations.
"""
import torch
import numpy as np


def expand_mask_to_match(mask: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Expand mask to match target tensor dimensions.
    
    Args:
        mask: Mask tensor with shape (H, W) or (B, H, W)
        target_tensor: Target tensor to match dimensions
        
    Returns:
        Expanded mask tensor matching target dimensions
    """
    target_shape = target_tensor.shape
    
    if mask.dim() == 2:  # (H, W) -> (B, C, H, W, T)
        expanded = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    elif mask.dim() == 3:  # (B, H, W) -> (B, C, H, W, T)
        expanded = mask.unsqueeze(1).unsqueeze(-1)
    else:
        expanded = mask
    
    # Expand to match all dimensions
    while expanded.dim() < len(target_shape):
        expanded = expanded.unsqueeze(-1)
    
    # Repeat to match target shape
    repeat_dims = []
    for i in range(len(target_shape)):
        if i < expanded.dim():
            repeat_dims.append(target_shape[i] // expanded.shape[i])
        else:
            repeat_dims.append(1)
    
    return expanded.repeat(*repeat_dims)


def safe_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Safely move tensor to device with error handling.
    
    Args:
        tensor: Input tensor
        device: Target device
        
    Returns:
        Tensor moved to device
        
    Raises:
        RuntimeError: If device transfer fails
    """
    try:
        return tensor.to(device)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise RuntimeError(f"GPU out of memory when moving tensor of shape {tensor.shape} to {device}")
        else:
            raise RuntimeError(f"Failed to move tensor to {device}: {e}")


def validate_tensor_shape(tensor: torch.Tensor, expected_shape: tuple, name: str) -> None:
    """
    Validate tensor shape matches expected shape.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape tuple
        name: Tensor name for error messages
        
    Raises:
        ValueError: If shape doesn't match
    """
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} shape {tensor.shape} != expected {expected_shape}")


def clip_and_validate(tensor: torch.Tensor, min_val: float, max_val: float, name: str) -> torch.Tensor:
    """
    Clip tensor values and validate they're in expected range.
    
    Args:
        tensor: Input tensor
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Tensor name for logging
        
    Returns:
        Clipped tensor
    """
    clipped = torch.clamp(tensor, min_val, max_val)
    
    # Log if significant clipping occurred
    clipped_count = (tensor != clipped).sum().item()
    if clipped_count > 0:
        total_elements = tensor.numel()
        print(f"Warning: Clipped {clipped_count}/{total_elements} values in {name}")
    
    return clipped