# AD-PINI v4 dataset - anomaly prediction data loader

import torch
import torch.utils.data as data
import xarray as xr
import numpy as np
from typing import Dict, Tuple, Optional
import logging

class CarbonAnomalyDataset(data.Dataset):
    """
    AD-PINI v4 anomaly prediction dataset.
    Based on the escalator principle, loads historical pCO2 anomaly sequences to predict future anomaly increments.
    """
    
    def __init__(self, 
                 data_path: str,
                 history_length: int = 6,
                 prediction_horizon: int = 1,
                 train: bool = True,
                 train_split: float = 0.8,
                 normalize: bool = True):
        """
        Initialize dataset.
        
        Args:
            data_path: path to preprocessed data
            history_length: length of historical sequence
            prediction_horizon: number of steps ahead to predict
            train: whether this is the training subset
            train_split: fraction of data used for training
            normalize: whether to normalize variables
        """
        self.data_path = data_path
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.train = train
        self.train_split = train_split
        self.normalize = normalize
        
        # Load dataset
        self.dataset = xr.open_dataset(data_path)
        
        # Create train/validation split
        self._create_split()
        
        # Normalization
        if self.normalize:
            self._compute_normalization_stats()
            
        # Build valid sample indices
        self._create_sample_indices()
        
        logging.info(
            f"Dataset initialized - mode: {'Train' if train else 'Test'}, "
            f"num_samples: {len(self.valid_indices)}"
        )
    
    def _create_split(self):
        """Create temporal train/validation split."""
        n_times = len(self.dataset.time)
        split_idx = int(n_times * self.train_split)
        
        if self.train:
            self.time_slice = slice(0, split_idx)
        else:
            self.time_slice = slice(split_idx, n_times)
            
        # Subset data according to split
        self.data_subset = self.dataset.isel(time=self.time_slice)
        
    def _compute_normalization_stats(self):
        """Compute normalization statistics (training set only)."""
        if self.train:
            # Compute statistics in training mode
            train_data = self.data_subset.pco2_anom_input.values
            valid_mask = ~np.isnan(train_data)
            
            self.norm_stats = {
                'pco2_anom_mean': np.nanmean(train_data),
                'pco2_anom_std': np.nanstd(train_data),
                
                # Increment statistics
                'delta_sst_mean': np.nanmean(self.data_subset.delta_sst_gt.values),
                'delta_sst_std': np.nanstd(self.data_subset.delta_sst_gt.values),
                'delta_dic_mean': np.nanmean(self.data_subset.delta_dic_gt.values), 
                'delta_dic_std': np.nanstd(self.data_subset.delta_dic_gt.values),
                
                # Climatological increment statistics
                'delta_sst_clim_mean': np.nanmean(self.data_subset.delta_sst_clim.values),
                'delta_sst_clim_std': np.nanstd(self.data_subset.delta_sst_clim.values),
                'delta_dic_clim_mean': np.nanmean(self.data_subset.delta_dic_clim.values),
                'delta_dic_clim_std': np.nanstd(self.data_subset.delta_dic_clim.values),
                
                # pCO2 statistics
                'pco2_mean': np.nanmean(self.data_subset.pco2_target.values),
                'pco2_std': np.nanstd(self.data_subset.pco2_target.values),
                'pco2_current_mean': np.nanmean(self.data_subset.pco2_current.values),
                'pco2_current_std': np.nanstd(self.data_subset.pco2_current.values),
                
                # Physical variable statistics
                'dic_current_mean': np.nanmean(self.data_subset.dic_current.values),
                'dic_current_std': np.nanstd(self.data_subset.dic_current.values),
                'alk_current_mean': np.nanmean(self.data_subset.alk_current.values),
                'alk_current_std': np.nanstd(self.data_subset.alk_current.values),
                
                # Sensitivity factor statistics
                's_thermal_mean': np.nanmean(self.data_subset.s_thermal.values),
                's_thermal_std': np.nanstd(self.data_subset.s_thermal.values),
                's_nonther_mean': np.nanmean(self.data_subset.s_nonther.values),
                's_nonther_std': np.nanstd(self.data_subset.s_nonther.values)
            }
            
            # Save statistics to file
            self._save_norm_stats()
            
        else:
            # In evaluation mode, load statistics from file
            self._load_norm_stats()
    
    def _save_norm_stats(self):
        """Save normalization statistics to disk."""
        stats_path = self.data_path.replace('.nc', '_norm_stats.npz')
        np.savez(stats_path, **self.norm_stats)
        
    def _load_norm_stats(self):
        """Load normalization statistics from disk."""
        stats_path = self.data_path.replace('.nc', '_norm_stats.npz')
        with np.load(stats_path) as data:
            self.norm_stats = {key: data[key].item() for key in data.keys()}
    
    def _create_sample_indices(self):
        """Create indices for valid samples."""
        n_times = len(self.data_subset.time)
        
        # Require sufficient historical sequence length
        start_idx = self.history_length - 1
        end_idx = n_times - self.prediction_horizon
        
        self.valid_indices = []
        
        for t in range(start_idx, end_idx):
            # Check whether historical sequence is valid (not too many NaNs)
            hist_start = t - self.history_length + 1
            hist_end = t + 1
            
            # Historical data
            hist_data = self.data_subset.pco2_anom_input.isel(
                time=slice(hist_start, hist_end)
            ).values
            
            # Target data
            target_t = t + self.prediction_horizon
            if target_t >= len(self.data_subset.time):
                continue  # skip out-of-range samples
            target_data = self.data_subset.delta_sst_gt.isel(time=target_t).values
            
            # Validity check: require sufficient valid data (ocean vs land NaNs)
            hist_valid_count = (~np.isnan(hist_data)).sum()
            target_valid_count = (~np.isnan(target_data)).sum()
            
            # Consider sample valid if it has enough ocean pixels (at least 1000 valid cells)
            if hist_valid_count > 1000 and target_valid_count > 1000:
                self.valid_indices.append(t)
    
    def _normalize_data(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Normalize data using precomputed statistics."""
        if not self.normalize:
            return data
            
        # Map variable names to stat keys
        var_mapping = {
            'pco2_anom': ('pco2_anom_mean', 'pco2_anom_std'),
            'delta_sst': ('delta_sst_mean', 'delta_sst_std'),
            'delta_dic': ('delta_dic_mean', 'delta_dic_std'),
            'delta_sst_clim': ('delta_sst_clim_mean', 'delta_sst_clim_std'),
            'delta_dic_clim': ('delta_dic_clim_mean', 'delta_dic_clim_std'),
            'pco2': ('pco2_mean', 'pco2_std'),
            'pco2_current': ('pco2_current_mean', 'pco2_current_std'),
            'dic_current': ('dic_current_mean', 'dic_current_std'),
            'alk_current': ('alk_current_mean', 'alk_current_std'),
            's_thermal': ('s_thermal_mean', 's_thermal_std'),
            's_nonther': ('s_nonther_mean', 's_nonther_std')
        }
        
        if var_name in var_mapping:
            mean_key, std_key = var_mapping[var_name]
            mean = self.norm_stats[mean_key]
            std = self.norm_stats[std_key]
            return (data - mean) / (std + 1e-8)  # avoid division by zero
        else:
            return data  # do not normalize unknown variables
    
    def _denormalize_data(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Denormalize data back to physical space."""
        if not self.normalize:
            return data
            
        # Use the same mapping logic
        var_mapping = {
            'pco2_anom': ('pco2_anom_mean', 'pco2_anom_std'),
            'delta_sst': ('delta_sst_mean', 'delta_sst_std'),
            'delta_dic': ('delta_dic_mean', 'delta_dic_std'),
            'delta_sst_clim': ('delta_sst_clim_mean', 'delta_sst_clim_std'),
            'delta_dic_clim': ('delta_dic_clim_mean', 'delta_dic_clim_std'),
            'pco2': ('pco2_mean', 'pco2_std'),
            'pco2_current': ('pco2_current_mean', 'pco2_current_std'),
            'dic_current': ('dic_current_mean', 'dic_current_std'),
            'alk_current': ('alk_current_mean', 'alk_current_std'),
            's_thermal': ('s_thermal_mean', 's_thermal_std'),
            's_nonther': ('s_nonther_mean', 's_nonther_std')
        }
        
        if var_name in var_mapping:
            mean_key, std_key = var_mapping[var_name]
            mean = self.norm_stats[mean_key]
            std = self.norm_stats[std_key]
            return data * (std + 1e-8) + mean
        else:
            return data
    
    def __len__(self) -> int:
        """Dataset size (number of valid samples)."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: sample index
            
        Returns:
            sample: dict containing model inputs and targets
        """
        t = self.valid_indices[idx]
        
        # === Inputs ===
        # Historical anomaly sequence [history_length, H, W]
        hist_start = t - self.history_length + 1
        hist_end = t + 1
        
        pco2_anom_hist = self.data_subset.pco2_anom_input.isel(
            time=slice(hist_start, hist_end)
        ).values
        
        # Escalator increments (known forcings) [H, W]
        target_t = t + self.prediction_horizon
        delta_sst_clim = self.data_subset.delta_sst_clim.isel(time=target_t).values
        delta_dic_clim = self.data_subset.delta_dic_clim.isel(time=target_t).values
        
        # Current physical state [H, W]
        pco2_current = self.data_subset.pco2_current.isel(time=t).values
        dic_current = self.data_subset.dic_current.isel(time=t).values
        alk_current = self.data_subset.alk_current.isel(time=t).values
        
        # Sensitivity factors [H, W]
        s_thermal = self.data_subset.s_thermal.isel(time=t).values
        s_nonther = self.data_subset.s_nonther.isel(time=t).values
        
        # === Targets ===
        # Full increments (supervision targets) [H, W]
        delta_sst_gt = self.data_subset.delta_sst_gt.isel(time=target_t).values
        delta_dic_gt = self.data_subset.delta_dic_gt.isel(time=target_t).values
        
        # Final target pCO2 [H, W]
        pco2_target = self.data_subset.pco2_target.isel(time=target_t).values
        
        # === Normalization ===
        # Normalize model inputs
        pco2_anom_hist_norm = self._normalize_data(pco2_anom_hist, 'pco2_anom')
        delta_sst_clim_norm = self._normalize_data(delta_sst_clim, 'delta_sst_clim')
        delta_dic_clim_norm = self._normalize_data(delta_dic_clim, 'delta_dic_clim')
        pco2_current_norm = self._normalize_data(pco2_current, 'pco2_current')
        dic_current_norm = self._normalize_data(dic_current, 'dic_current')
        alk_current_norm = self._normalize_data(alk_current, 'alk_current')
        s_thermal_norm = self._normalize_data(s_thermal, 's_thermal')
        s_nonther_norm = self._normalize_data(s_nonther, 's_nonther')
        
        # Normalize supervision targets
        delta_sst_gt_norm = self._normalize_data(delta_sst_gt, 'delta_sst')
        delta_dic_gt_norm = self._normalize_data(delta_dic_gt, 'delta_dic')
        pco2_target_norm = self._normalize_data(pco2_target, 'pco2')
        
        # === Save original NaN mask (fix for NPY masks all being 1) ===
        # Save mask of valid target values before cleaning NaNs
        pco2_target_valid_mask = ~np.isnan(pco2_target_norm)  # valid target mask
        
        # === Handle NaN values ===
        # Fill with 0 in normalized space (corresponding to mean)
        pco2_anom_hist_norm = np.nan_to_num(pco2_anom_hist_norm, 0.0)
        delta_sst_clim_norm = np.nan_to_num(delta_sst_clim_norm, 0.0)
        delta_dic_clim_norm = np.nan_to_num(delta_dic_clim_norm, 0.0)
        pco2_current_norm = np.nan_to_num(pco2_current_norm, 0.0)
        dic_current_norm = np.nan_to_num(dic_current_norm, 0.0)
        alk_current_norm = np.nan_to_num(alk_current_norm, 0.0)
        s_thermal_norm = np.nan_to_num(s_thermal_norm, 0.0)
        s_nonther_norm = np.nan_to_num(s_nonther_norm, 0.0)
        delta_sst_gt_norm = np.nan_to_num(delta_sst_gt_norm, 0.0)
        delta_dic_gt_norm = np.nan_to_num(delta_dic_gt_norm, 0.0)
        pco2_target_norm = np.nan_to_num(pco2_target_norm, 0.0)
        
        # === Convert to tensors ===
        sample = {
            # Network inputs (all normalized)
            'pco2_anom_hist': torch.FloatTensor(pco2_anom_hist_norm),  # [T, H, W]
            'delta_clim': torch.FloatTensor(np.stack([delta_sst_clim_norm, delta_dic_clim_norm], axis=0)),  # [2, H, W]
            
            # Physical state (all normalized)
            'pco2_current': torch.FloatTensor(pco2_current_norm),  # [H, W]
            'dic_current': torch.FloatTensor(dic_current_norm),    # [H, W]
            'alk_current': torch.FloatTensor(alk_current_norm),    # [H, W]
            's_thermal': torch.FloatTensor(s_thermal_norm),        # [H, W]
            's_nonther': torch.FloatTensor(s_nonther_norm),        # [H, W]
            
            # Supervision targets (all normalized)
            'delta_gt': torch.FloatTensor(np.stack([delta_sst_gt_norm, delta_dic_gt_norm], axis=0)),  # [2, H, W]
            'pco2_target': torch.FloatTensor(pco2_target_norm),  # [H, W]
            
            # Valid data mask (fix for NPY mask issue)
            'valid_mask': torch.FloatTensor(pco2_target_valid_mask.astype(np.float32)),  # [H, W], 1=ocean, 0=land/invalid
            
            # Meta info (fix validation timestamp issues)
            # Compute absolute time index, accounting for train/validation split
            'time_index': torch.LongTensor([t + (self.time_slice.start or 0)]),
            'sample_id': torch.LongTensor([idx])
        }
        
        return sample
    
    def get_normalization_stats(self) -> Dict[str, float]:
        """Return normalization statistics."""
        if hasattr(self, 'norm_stats'):
            return self.norm_stats
        else:
            return {}
    
    def denormalize_prediction(self, pred: torch.Tensor, var_name: str) -> torch.Tensor:
        """Denormalize model predictions."""
        pred_np = pred.detach().cpu().numpy()
        denorm_np = self._denormalize_data(pred_np, var_name)
        return torch.from_numpy(denorm_np).to(pred.device)
    
    def get_future_step_data(self, base_time_idx: int, future_step: int) -> Dict[str, torch.Tensor]:
        """
        Get data at a future time step (for autoregressive prediction).
        
        Args:
            base_time_idx: base time index (dataset time index)
            future_step: how many steps ahead (1 for next time step)
            
        Returns:
            future_data: dictionary containing future-step data
        """
        target_time_idx = base_time_idx + future_step
        
        # Bounds check
        n_times = len(self.data_subset.time)
        if target_time_idx >= n_times:
            raise ValueError(f"Target time index {target_time_idx} out of range [0, {n_times-1}]")
        
        # Climatological forcings at future time step
        delta_sst_clim = self.data_subset.delta_sst_clim.isel(time=target_time_idx).values
        delta_dic_clim = self.data_subset.delta_dic_clim.isel(time=target_time_idx).values
        
        # Targets at future time step
        pco2_target = self.data_subset.pco2_target.isel(time=target_time_idx).values
        delta_sst_gt = self.data_subset.delta_sst_gt.isel(time=target_time_idx).values
        delta_dic_gt = self.data_subset.delta_dic_gt.isel(time=target_time_idx).values
        
        # Physical state at future time step (current state)
        pco2_current = self.data_subset.pco2_current.isel(time=target_time_idx).values
        dic_current = self.data_subset.dic_current.isel(time=target_time_idx).values
        alk_current = self.data_subset.alk_current.isel(time=target_time_idx).values
        
        # Sensitivity factors (assumed to vary slowly in time)
        s_thermal = self.data_subset.s_thermal.isel(time=target_time_idx).values
        s_nonther = self.data_subset.s_nonther.isel(time=target_time_idx).values
        
        # Normalize
        delta_sst_clim_norm = self._normalize_data(delta_sst_clim, 'delta_sst_clim')
        delta_dic_clim_norm = self._normalize_data(delta_dic_clim, 'delta_dic_clim')
        pco2_target_norm = self._normalize_data(pco2_target, 'pco2')
        delta_sst_gt_norm = self._normalize_data(delta_sst_gt, 'delta_sst')
        delta_dic_gt_norm = self._normalize_data(delta_dic_gt, 'delta_dic')
        pco2_current_norm = self._normalize_data(pco2_current, 'pco2_current')
        dic_current_norm = self._normalize_data(dic_current, 'dic_current')
        alk_current_norm = self._normalize_data(alk_current, 'alk_current')
        s_thermal_norm = self._normalize_data(s_thermal, 's_thermal')
        s_nonther_norm = self._normalize_data(s_nonther, 's_nonther')
        
        # === Save original NaN mask (for NPY saving in autoregressive prediction) ===
        pco2_target_valid_mask = ~np.isnan(pco2_target_norm)  # valid target mask
        
        # Handle NaNs
        delta_sst_clim_norm = np.nan_to_num(delta_sst_clim_norm, 0.0)
        delta_dic_clim_norm = np.nan_to_num(delta_dic_clim_norm, 0.0)
        pco2_target_norm = np.nan_to_num(pco2_target_norm, 0.0)
        delta_sst_gt_norm = np.nan_to_num(delta_sst_gt_norm, 0.0)
        delta_dic_gt_norm = np.nan_to_num(delta_dic_gt_norm, 0.0)
        pco2_current_norm = np.nan_to_num(pco2_current_norm, 0.0)
        dic_current_norm = np.nan_to_num(dic_current_norm, 0.0)
        alk_current_norm = np.nan_to_num(alk_current_norm, 0.0)
        s_thermal_norm = np.nan_to_num(s_thermal_norm, 0.0)
        s_nonther_norm = np.nan_to_num(s_nonther_norm, 0.0)
        
        # Convert to tensors and pack dictionary
        future_data = {
            # Climatological forcings
            'delta_clim': torch.FloatTensor(np.stack([delta_sst_clim_norm, delta_dic_clim_norm], axis=0)),  # [2, H, W]
            
            # Targets
            'pco2_target': torch.FloatTensor(pco2_target_norm),  # [H, W]
            'delta_gt': torch.FloatTensor(np.stack([delta_sst_gt_norm, delta_dic_gt_norm], axis=0)),  # [2, H, W]
            
            # Physical state
            'pco2_current': torch.FloatTensor(pco2_current_norm),  # [H, W]
            'dic_current': torch.FloatTensor(dic_current_norm),    # [H, W]
            'alk_current': torch.FloatTensor(alk_current_norm),    # [H, W]
            's_thermal': torch.FloatTensor(s_thermal_norm),        # [H, W]
            's_nonther': torch.FloatTensor(s_nonther_norm),        # [H, W]
            
            # Valid data mask (for NPY saving in autoregressive prediction)
            'valid_mask': torch.FloatTensor(pco2_target_valid_mask.astype(np.float32)),  # [H, W], 1=ocean, 0=land/invalid
            
            # Meta info (fix validation timestamp issues)
            'time_index': torch.LongTensor([target_time_idx + (self.time_slice.start or 0)]),
            'future_step': torch.LongTensor([future_step])
        }
        
        return future_data




def create_data_loaders(config, 
                       batch_size: Optional[int] = None,
                       num_workers: Optional[int] = None) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        config: configuration object
        batch_size: mini-batch size
        num_workers: number of worker processes
        
    Returns:
        train_loader: training data loader
        val_loader: validation data loader
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    
    # Build datasets
    train_dataset = CarbonAnomalyDataset(
        data_path=config.PROCESSED_DATA_PATH,
        history_length=config.HISTORY_LENGTH,
        train=True,
        train_split=0.8,  # 80% for training
        normalize=True
    )
    
    val_dataset = CarbonAnomalyDataset(
        data_path=config.PROCESSED_DATA_PATH,
        history_length=config.HISTORY_LENGTH,
        train=False,
        train_split=0.8,  # 20% for validation
        normalize=True
    )
    
    # Build data loaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # ensure consistent batch size
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader


def main():
    """Simple dataset test."""
    from configs.config_v4 import config
    
    # Create dataset
    dataset = CarbonAnomalyDataset(
        data_path=config.PROCESSED_DATA_PATH,
        history_length=config.HISTORY_LENGTH,
        train=True
    )
    
    # Inspect one sample
    sample = dataset[0]
    
    print("Sample tensor shapes:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # Test data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    print(f"\nData loader info:")
    print(f"  num train batches: {len(train_loader)}")
    print(f"  num val batches:   {len(val_loader)}")
    
    # Inspect one batch
    batch = next(iter(train_loader))
    print(f"\nBatch tensor shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")


if __name__ == "__main__":
    main()