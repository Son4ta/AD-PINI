# utils/denormalize.py
"""
Denormalization manager

Centralizes normalization / denormalization operations for all variables.
"""

import numpy as np
import torch
from typing import Dict, Union

class Denormalizer:
    """
    Denormalization manager - v4-specific version.
    
    Features:
    1. Central management of normalization statistics for all variables
    2. Unified normalization / denormalization API
    3. Supports both NumPy arrays and PyTorch tensors
    4. Supports two normalization modes: "zscore" and "minmax"
    """
    
    def __init__(self, stats: Dict[str, Dict[str, float]], mode: str = "zscore", norm_range: tuple = (-1, 1)):
        """
        Args:
            stats: normalization statistics
                   zscore mode: {'var_name': {'mean': float, 'std': float}}
                   minmax mode: {'var_name': {'min': float, 'max': float}}
            mode: normalization mode, "zscore" or "minmax"
            norm_range: target range for MinMax normalization, default (-1, 1)
        """
        self.stats = stats
        self.mode = mode
        self.norm_range = norm_range
        
        # Pre-create tensors for PyTorch use (optional performance optimization)
        self.torch_stats = {}
        for var_name, stat in stats.items():
            self.torch_stats[var_name] = {
                'mean': torch.tensor(stat['mean']),
                'std': torch.tensor(stat['std'])
            }
    
    def normalize(self, 
                  data: Union[np.ndarray, torch.Tensor], 
                  var_name: str) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize data using either z-score or MinMax, depending on mode.
        
        Args:
            data: raw values
            var_name: variable name key in stats
            
        Returns:
            normalized data (same type as input)
        """
        if var_name not in self.stats:
            raise ValueError(f"Statistics for variable {var_name} not found.")
        
        if self.mode == "zscore":
            return self._zscore_normalize(data, var_name)
        elif self.mode == "minmax":
            return self._minmax_normalize(data, var_name)
        else:
            raise ValueError(f"Unsupported normalization mode: {self.mode}")
    
    def _zscore_normalize(self, data, var_name):
        """Z-score normalization: (x - mean) / std."""
        mean = self.stats[var_name]['mean']
        std = self.stats[var_name]['std']
        
        if isinstance(data, torch.Tensor):
            mean_t = torch.tensor(mean, device=data.device, dtype=data.dtype)
            std_t = torch.tensor(std, device=data.device, dtype=data.dtype)
            return (data - mean_t) / (std_t + 1e-8)
        else:
            return (data - mean) / (std + 1e-8)
    
    def _minmax_normalize(self, data, var_name):
        """MinMax normalization: map into configured range, default [-1, 1]."""
        min_val = self.stats[var_name]['min']
        max_val = self.stats[var_name]['max']
        min_range, max_range = self.norm_range
        
        if isinstance(data, torch.Tensor):
            min_t = torch.tensor(min_val, device=data.device, dtype=data.dtype)
            max_t = torch.tensor(max_val, device=data.device, dtype=data.dtype)
            # Map to [0, 1] then to target range
            normalized = (data - min_t) / (max_t - min_t + 1e-8)
            return normalized * (max_range - min_range) + min_range
        else:
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            return normalized * (max_range - min_range) + min_range
    
    def denormalize(self, 
                    data_norm: Union[np.ndarray, torch.Tensor], 
                    var_name: str) -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalize data, supporting both z-score and MinMax modes.
        
        Args:
            data_norm: normalized data
            var_name: variable name key in stats
            
        Returns:
            denormalized data (same type as input)
        """
        if var_name not in self.stats:
            raise ValueError(f"Statistics for variable {var_name} not found.")
        
        if self.mode == "zscore":
            return self._zscore_denormalize(data_norm, var_name)
        elif self.mode == "minmax":
            return self._minmax_denormalize(data_norm, var_name)
        else:
            raise ValueError(f"Unsupported normalization mode: {self.mode}")
    
    def _zscore_denormalize(self, data_norm, var_name):
        """Z-score denormalization: x * std + mean."""
        mean = self.stats[var_name]['mean']
        std = self.stats[var_name]['std']
        
        if isinstance(data_norm, torch.Tensor):
            mean_t = torch.tensor(mean, device=data_norm.device, dtype=data_norm.dtype)
            std_t = torch.tensor(std, device=data_norm.device, dtype=data_norm.dtype)
            return data_norm * std_t + mean_t
        else:
            return data_norm * std + mean
    
    def _minmax_denormalize(self, data_norm, var_name):
        """MinMax denormalization: map back from target range into original range."""
        min_val = self.stats[var_name]['min']
        max_val = self.stats[var_name]['max']
        min_range, max_range = self.norm_range
        
        if isinstance(data_norm, torch.Tensor):
            min_t = torch.tensor(min_val, device=data_norm.device, dtype=data_norm.dtype)
            max_t = torch.tensor(max_val, device=data_norm.device, dtype=data_norm.dtype)
            # Map from target range to [0, 1], then to original range
            normalized_01 = (data_norm - min_range) / (max_range - min_range)
            return normalized_01 * (max_t - min_t) + min_t
        else:
            normalized_01 = (data_norm - min_range) / (max_range - min_range)
            return normalized_01 * (max_val - min_val) + min_val
    
    def get_stat(self, var_name: str, stat_type: str) -> float:
        """
        Get a single statistic value.
        
        Args:
            var_name: variable name
            stat_type: 'mean' or 'std'
            
        Returns:
            scalar statistic value
        """
        return self.stats[var_name][stat_type]


if __name__ == "__main__":
    # Simple self-test
    stats = {
        'sst': {'mean': 18.0, 'std': 8.0},
        'dic': {'mean': 2050.0, 'std': 50.0},
        'alk': {'mean': 2300.0, 'std': 50.0},
        'pco2': {'mean': 380.0, 'std': 20.0}
    }
    
    denorm = Denormalizer(stats)
    
    # NumPy test
    data = np.array([18.0, 26.0, 10.0])  # SST values
    data_norm = denorm.normalize(data, 'sst')
    print(f"Normalize (NumPy): {data} -> {data_norm}")
    data_recovered = denorm.denormalize(data_norm, 'sst')
    print(f"Denormalize (NumPy): {data_norm} -> {data_recovered}")
    
    # PyTorch test
    data_torch = torch.tensor([380.0, 400.0, 360.0])
    data_norm_torch = denorm.normalize(data_torch, 'pco2')
    print(f"Normalize (PyTorch): {data_torch} -> {data_norm_torch}")
    
    print("✓ Denormalizer test passed.")