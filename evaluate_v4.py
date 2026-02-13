#!/usr/bin/env python3
# AD-PINI v4 Model Evaluation Script - Supports Standard Evaluation and Autoregressive Prediction

import os
import sys
import argparse
import time
from typing import Dict, Optional, Tuple, List
import warnings
from datetime import datetime, timedelta

import torch
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Project imports - Reuse components from train_v4.py
from configs.config_v4 import config
from data.v4.preprocessing_v4 import DataPreprocessorV4
from data.v4.dataset_v4 import create_data_loaders
from models.v4.carbon_net_v4 import CarbonNetV4
from models.v4.loss_v4 import CombinedLossV4

warnings.filterwarnings('ignore')

@dataclass
class EvaluationConfig:
    """Evaluation Configuration Class"""
    checkpoint_path: str
    num_batches: Optional[int] = None
    force_recompute: bool = False
    device: Optional[str] = None
    output_path: Optional[str] = None
    # New: Autoregressive prediction configuration
    future_steps: int = 1  # N=1 for normal evaluation, N>1 for autoregressive prediction
    # New: NPY data saving configuration
    save_npy: bool = False  # Whether to save NPY prediction results
    npy_output_dir: Optional[str] = None  # NPY file save directory
    spatial_alignment: bool = True  # Whether to force spatial alignment
    # New: Extended model output saving configuration
    save_extended_outputs: bool = True  # Whether to save extended model outputs channels


class NPYDataSaver:
    """
    NPY Data Saver - Responsible for collecting, processing, and saving full validation prediction results
    
    Output Format:
    - Filename: YYYYMM.npy
    - Data Shape: (N_channels, T_out, 713, 1440)
    
    Base Channel Definitions (N_channels=5):
      - Channel 0: Pred_Raw (Predicted physical value, denormalized)
      - Channel 1: GT_Raw (Ground Truth physical value)
      - Channel 2: Pred_Norm (Predicted normalized value, direct model output)
      - Channel 3: GT_Norm (Ground Truth normalized value)
      - Channel 4: Valid_Mask (Valid point mask)
    
    Extended Channel Definitions (N_channels=23, when save_extended_outputs=True):
      - Channel 5: pco2_contrib_therm (Thermodynamic contribution component)
      - Channel 6: pco2_contrib_bio (Non-thermodynamic contribution component)
      - Channel 7: R_therm (Thermodynamic contribution μatm)
      - Channel 8: R_bio (Non-thermodynamic contribution μatm)
      - Channel 9: delta_sst_anom (SST Anomaly)
      - Channel 10: delta_dic_anom (DIC Anomaly)
      - Channel 11: pco2_anom_reconstructed (Reconstructed pCO2 Anomaly)
      - Channel 12: S_T (Calibrated Thermodynamic Sensitivity)
      - Channel 13: S_NT (Calibrated Non-thermodynamic Sensitivity)
      - Channel 14: delta_sst_total (Total SST Delta)
      - Channel 15: delta_dic_total (Total DIC Delta)
      - Channel 16: pco2_physics (Physics calculation result)
      - Channel 17: delta_thermal (Thermodynamic Delta)
      - Channel 18: delta_nonther (Non-thermodynamic Delta)
      - Channel 19: correction (Correction term)
      - Channel 20: pco2_current (Current pCO2 state)
      - Channel 21: s_thermal_orig (Original Thermodynamic Sensitivity)
      - Channel 22: s_nonther_orig (Original Non-thermodynamic Sensitivity)
    """
    
    def __init__(self, 
                 output_dir: str, 
                 target_height: int = 713, 
                 target_width: int = 1440,
                 spatial_alignment: bool = True,
                 save_extended_outputs: bool = True):
        """
        Initialize NPY Data Saver
        
        Args:
            output_dir: Output directory path
            target_height: Target spatial resolution height
            target_width: Target spatial resolution width
            spatial_alignment: Whether to enable forced spatial alignment
            save_extended_outputs: Whether to save extended model outputs channels
        """
        self.output_dir = output_dir
        self.target_height = target_height
        self.target_width = target_width
        self.spatial_alignment = spatial_alignment
        self.save_extended_outputs = save_extended_outputs
        self.norm_stats = None  # Normalization stats, used for denormalization
        
        # Calculate number of channels
        self.base_channels = 5
        self.extended_channels = 18
        self.total_channels = self.base_channels + (self.extended_channels if save_extended_outputs else 0)
        
        # Create output directory
        self.npy_data_dir = os.path.join(output_dir, "npy_data_PICABU")
        os.makedirs(self.npy_data_dir, exist_ok=True)
        
        print(f"📁 NPY Data Saver Initialized")
        print(f"   Output Directory: {self.npy_data_dir}")
        print(f"   Target Resolution: {target_height}×{target_width}")
        print(f"   Spatial Alignment: {'Enabled' if spatial_alignment else 'Disabled'}")
        print(f"   Extended Saving: {'Enabled' if save_extended_outputs else 'Disabled'}")
        print(f"   Total Channels: {self.total_channels} ({self.base_channels} Base + {self.extended_channels if save_extended_outputs else 0} Extended)")
    
    def set_normalization_stats(self, norm_stats: Dict):
        """Set normalization statistics for denormalization"""
        self.norm_stats = norm_stats
        print(f"✅ Normalization statistics set (contains {len(norm_stats)} items)")
    
    def _spatial_align(self, data: torch.Tensor) -> torch.Tensor:
        """
        Spatial Alignment: Interpolate to target resolution
        
        Args:
            data: Input data [B, H, W] or [B, C, H, W]
            
        Returns:
            aligned_data: Aligned data [B, target_H, target_W] or [B, C, target_H, target_W]
        """
        if not self.spatial_alignment:
            return data
            
        original_shape = data.shape
        current_height, current_width = original_shape[-2], original_shape[-1]
        
        # If already target resolution, return directly
        if current_height == self.target_height and current_width == self.target_width:
            return data
        
        # Prepare interpolation
        if data.dim() == 3:  # [B, H, W]
            data = data.unsqueeze(1)  # [B, 1, H, W]
            squeeze_needed = True
        else:  # [B, C, H, W]
            squeeze_needed = False
        
        # Bilinear interpolation to target resolution
        aligned_data = F.interpolate(
            data,
            size=(self.target_height, self.target_width),
            mode='bilinear',
            align_corners=False
        )
        
        if squeeze_needed:
            aligned_data = aligned_data.squeeze(1)  # [B, H, W]
            
        return aligned_data
    
    def _denormalize_pco2(self, normalized_data: torch.Tensor, variable_name: str = 'pco2') -> torch.Tensor:
        """
        Denormalize pCO2 data
        
        Args:
            normalized_data: Normalized data
            variable_name: Variable name to look up corresponding statistics
            
        Returns:
            denormalized_data: Denormalized data (Physical unit: uatm)
        """
        if self.norm_stats is None:
            print("⚠️ Warning: Normalization statistics not set, cannot denormalize")
            return normalized_data
        
        # Select statistics based on variable name
        mean_key = f'{variable_name}_mean'
        std_key = f'{variable_name}_std'
        
        if mean_key not in self.norm_stats or std_key not in self.norm_stats:
            print(f"⚠️ Warning: Missing normalization statistics for {variable_name}, using pco2 statistics")
            mean_key = 'pco2_mean'
            std_key = 'pco2_std'
        
        mean = self.norm_stats[mean_key]
        std = self.norm_stats[std_key]
        
        # Z-score denormalization: value = normalized_value * std + mean
        denormalized = normalized_data * std + mean
        return denormalized
    
    def _generate_valid_mask(self, data: torch.Tensor, batch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate valid mask - Fixes issue where mask is all 1s
        
        Args:
            data: Input data [B, H, W]
            batch_mask: Original valid mask from dataset [B, H, W] (Optional)
            
        Returns:
            mask: Valid mask [B, H, W] (1=Valid, 0=Invalid)
        """
        # Prioritize using the original mask provided by the dataset
        if batch_mask is not None:
            return batch_mask.float()
        
        # Fallback: Generate mask based on NaN values (for backward compatibility)
        mask = ~torch.isnan(data)
        return mask.float()
    
    def _extract_sample_timestamp(self, batch_data: Dict, sample_index: int) -> str:
        """
        Extract timestamp for a specific sample from batch data and convert to YYYYMM format
        
        Args:
            batch_data: Batch data dictionary
            sample_index: Index of the sample in the batch
            
        Returns:
            timestamp: Timestamp in YYYYMM format
        """
        # Scheme 1: Get from batch's time_info
        if 'time_info' in batch_data:
            time_info = batch_data['time_info']
            if isinstance(time_info, (list, tuple)) and len(time_info) > sample_index:
                time_obj = time_info[sample_index]
                if hasattr(time_obj, 'strftime'):
                    return time_obj.strftime('%Y%m')
                elif isinstance(time_obj, str):
                    try:
                        parsed_time = datetime.strptime(time_obj[:7], '%Y-%m')
                        return parsed_time.strftime('%Y%m')
                    except:
                        pass
        
        # Scheme 2: Get from time-related tensor (if dataset provides time index)
        if 'time_index' in batch_data:
            time_indices = batch_data['time_index']
            if isinstance(time_indices, torch.Tensor) and len(time_indices) > sample_index:
                time_idx = int(time_indices[sample_index].item())
                # Time index is now absolute, starting from 1993-01, incrementing monthly
                base_year, base_month = 1993, 1
                total_months = time_idx
                year = base_year + (base_month + total_months - 1) // 12
                month = ((base_month + total_months - 1) % 12) + 1
                return f"{year:04d}{month:02d}"
        
        # Scheme 3: Use global counter + base time (fallback)
        # This scheme assumes samples are arranged in chronological order
        # Adjust according to the actual time range of the dataset
        if not hasattr(self, '_sample_counter'):
            self._sample_counter = 0
            self._base_time = datetime(1993, 1, 1)  # Adjust based on actual data start time
        
        # Each sample corresponds to one month
        time_delta = timedelta(days=30 * (self._sample_counter + sample_index))
        sample_time = self._base_time + time_delta
        self._sample_counter += 1  # Update counter for next batch call
        
        return sample_time.strftime('%Y%m')
    
    def _extract_timestamp_from_batch(self, batch_data: Dict) -> str:
        """
        Extract timestamp from batch data and convert to YYYYMM format (Keep compatibility)
        """
        return self._extract_sample_timestamp(batch_data, 0)
    
    def _process_model_outputs(self, 
                             model_outputs: Dict[str, torch.Tensor], 
                             sample_index: int) -> Dict[str, torch.Tensor]:
        """
        Process model outputs, extracting all channel data for a single sample
        
        Args:
            model_outputs: Model output dictionary (from forward method in carbon_net_v4.py)
            sample_index: Sample index
            
        Returns:
            processed_outputs: Processed single sample output dictionary [H, W] format
        """
        if not self.save_extended_outputs:
            return {}
        
        processed = {}
        
        # Define outputs to process (exclude sensitivity_refine_debug)
        output_mapping = {
            # pco2_contributions needs special handling, split into two channels
            'pco2_contributions': ['pco2_contrib_therm', 'pco2_contrib_bio'],
            # Other single channel outputs
            'R_therm': 'R_therm',
            'R_bio': 'R_bio',
            'delta_sst_anom': 'delta_sst_anom',
            'delta_dic_anom': 'delta_dic_anom',
            'pco2_anom_reconstructed': 'pco2_anom_reconstructed',
            'S_T': 'S_T',
            'S_NT': 'S_NT',
            'delta_sst_total': 'delta_sst_total',
            'delta_dic_total': 'delta_dic_total',
            'pco2_physics': 'pco2_physics',
            'delta_thermal': 'delta_thermal',
            'delta_nonther': 'delta_nonther',
            'correction': 'correction',
            'pco2_current': 'pco2_current',
            's_thermal': 's_thermal_orig',
            's_nonther': 's_nonther_orig'
        }
        
        for output_key, channel_name in output_mapping.items():
            if output_key in model_outputs:
                output_tensor = model_outputs[output_key]
                
                if output_key == 'pco2_contributions':
                    # pco2_contributions is [B, 2, H, W], split into two channels
                    if output_tensor.shape[1] >= 2:
                        processed[channel_name[0]] = output_tensor[sample_index, 0]  # Thermodynamic contribution [H, W]
                        processed[channel_name[1]] = output_tensor[sample_index, 1]  # Non-thermodynamic contribution [H, W]
                else:
                    # Single channel outputs, shape [B, H, W] -> [H, W]
                    if output_tensor.dim() == 3:  # [B, H, W]
                        processed[channel_name] = output_tensor[sample_index]
                    elif output_tensor.dim() == 2:  # [H, W] - Single sample case
                        processed[channel_name] = output_tensor
        
        return processed
    
    def save_single_sample(self, 
                          prediction: torch.Tensor,
                          target: torch.Tensor, 
                          prediction_norm: torch.Tensor,
                          target_norm: torch.Tensor,
                          sample_timestamp: str,
                          sample_index: Optional[int] = None,
                          batch_mask: Optional[torch.Tensor] = None,
                          model_outputs: Optional[Dict[str, torch.Tensor]] = None) -> str:
        """
        Save prediction results for a single sample as NPY format
        
        Args:
            prediction: Denormalized prediction value [H, W] (single sample)
            target: Denormalized target value [H, W] (single sample)
            prediction_norm: Normalized prediction value [H, W] (single sample)
            target_norm: Normalized target value [H, W] (single sample)
            sample_timestamp: Sample timestamp in YYYYMM format
            sample_index: Sample index (for debugging)
            batch_mask: Original valid mask from dataset [H, W] (Fixes mask all 1s issue)
            model_outputs: Complete model output dictionary (Optional, for extended channel saving)
            
        Returns:
            saved_file_path: Saved file path
        """
        # Ensure input is single sample [H, W]
        if prediction.dim() != 2:
            raise ValueError(f"Expected single sample data [H, W], but got {prediction.shape}")
        
        # 1. Spatial alignment - Add batch dimension for processing
        prediction = prediction.unsqueeze(0)  # [1, H, W] 
        target = target.unsqueeze(0)
        prediction_norm = prediction_norm.unsqueeze(0)
        target_norm = target_norm.unsqueeze(0)
        
        if self.spatial_alignment:
            prediction = self._spatial_align(prediction)
            target = self._spatial_align(target)
            prediction_norm = self._spatial_align(prediction_norm)
            target_norm = self._spatial_align(target_norm)
        
        # 2. Generate valid mask (Use original mask provided by dataset)
        if batch_mask is not None:
            # Add batch dimension for spatial alignment
            batch_mask_with_batch = batch_mask.unsqueeze(0)  # [1, H, W]
            if self.spatial_alignment:
                batch_mask_with_batch = self._spatial_align(batch_mask_with_batch)
            valid_mask = batch_mask_with_batch  # [1, H, W]
        else:
            # Fallback: Generate mask based on ground truth
            valid_mask = self._generate_valid_mask(target)  # Generate mask based on GT
            if self.spatial_alignment:
                valid_mask = self._spatial_align(valid_mask)
        
        # Remove batch dimension to return to single sample [H, W]
        prediction = prediction.squeeze(0)
        target = target.squeeze(0) 
        prediction_norm = prediction_norm.squeeze(0)
        target_norm = target_norm.squeeze(0)
        valid_mask = valid_mask.squeeze(0)
        
        # 3. Build channel data structure
        # Base 5 channels: Add time dimension to each channel [H, W] -> [1, H, W]
        base_channels = []
        base_channels.append(prediction.unsqueeze(0))      # Channel 0: Predicted physical value
        base_channels.append(target.unsqueeze(0))          # Channel 1: GT physical value
        base_channels.append(prediction_norm.unsqueeze(0)) # Channel 2: Predicted normalized value
        base_channels.append(target_norm.unsqueeze(0))     # Channel 3: GT normalized value
        base_channels.append(valid_mask.unsqueeze(0))      # Channel 4: Valid mask
        
        # Extended channels (if enabled)
        extended_channels = []
        if self.save_extended_outputs and model_outputs is not None and sample_index is not None:
            # Process model outputs
            processed_outputs = self._process_model_outputs(model_outputs, sample_index)
            
            # Add extended channels in order (consistent with definition in docstring)
            channel_order = [
                'pco2_contrib_therm', 'pco2_contrib_bio', 'R_therm', 'R_bio',
                'delta_sst_anom', 'delta_dic_anom', 'pco2_anom_reconstructed',
                'S_T', 'S_NT', 'delta_sst_total', 'delta_dic_total',
                'pco2_physics', 'delta_thermal', 'delta_nonther', 'correction',
                'pco2_current', 's_thermal_orig', 's_nonther_orig'
            ]
            
            for channel_name in channel_order:
                if channel_name in processed_outputs:
                    # Spatial alignment processing
                    channel_data = processed_outputs[channel_name].unsqueeze(0)  # [1, H, W]
                    if self.spatial_alignment:
                        channel_data = self._spatial_align(channel_data)
                    extended_channels.append(channel_data.squeeze(0))
                else:
                    # Fill missing channels with zeros
                    zero_channel = torch.zeros_like(prediction)
                    extended_channels.append(zero_channel)
        
        # Combine all channels
        all_channels = base_channels + [ch.unsqueeze(0) for ch in extended_channels]
        
        # Stack as multi-channel data: [N_channels, 1, H, W]
        multi_channel_data = torch.stack(all_channels, dim=0)
        
        # 4. Convert to numpy and move to CPU
        data_np = multi_channel_data.detach().cpu().numpy()
        
        # 5. Generate filename
        filename = f"{sample_timestamp}.npy"
        
        # 6. Save file
        file_path = os.path.join(self.npy_data_dir, filename)
        np.save(file_path, data_np)
        
        print(f"💾 Saved Sample NPY: {filename}")
        print(f"   Data Shape: {data_np.shape} ({self.total_channels} channels x 1 timestep x H x W)")
        if sample_index is not None:
            print(f"   Sample Index: {sample_index}")
        if self.save_extended_outputs and len(extended_channels) > 0:
            print(f"   Extended Channels: {len(extended_channels)}")
        
        return file_path
    
    def save_samples_from_validation(self,
                                   val_predictions: List[torch.Tensor],
                                   val_targets: List[torch.Tensor],
                                   val_predictions_norm: List[torch.Tensor], 
                                   val_targets_norm: List[torch.Tensor],
                                   val_batch_data: List[Dict],
                                   val_model_outputs: Optional[List[Dict[str, torch.Tensor]]] = None) -> List[str]:
        """
        Save all data collected during validation sample by sample
        
        Args:
            val_predictions: All prediction results during validation [List of [B, H, W]]
            val_targets: All target results during validation [List of [B, H, W]]
            val_predictions_norm: All normalized prediction results during validation [List of [B, H, W]]
            val_targets_norm: All normalized target results during validation [List of [B, H, W]]
            val_batch_data: All batch data during validation
            val_model_outputs: Model output data during validation [List of Dict] (Optional, for extended channels)
            
        Returns:
            saved_files: List of saved file paths
        """
        saved_files = []
        total_samples = 0
        
        # Calculate total number of samples
        for pred_batch in val_predictions:
            total_samples += pred_batch.shape[0]
        
        print(f"\n📦 Start saving validation data sample by sample")
        print(f"   Total Batches: {len(val_predictions)}")
        print(f"   Total Samples: {total_samples}")
        
        global_sample_idx = 0
        
        # Add model_outputs handling for zip
        zip_data = [val_predictions, val_targets, val_predictions_norm, val_targets_norm, val_batch_data]
        if val_model_outputs is not None:
            zip_data.append(val_model_outputs)
        else:
            zip_data.append([None] * len(val_predictions))
        
        for batch_idx, batch_items in enumerate(zip(*zip_data)):
            pred_batch, target_batch, pred_norm_batch, target_norm_batch, batch_data = batch_items[:5]
            batch_model_outputs = batch_items[5] if len(batch_items) > 5 else None
            
            try:
                batch_size = pred_batch.shape[0]
                
                # Save each sample in the current batch individually
                for sample_idx in range(batch_size):
                    # Extract single sample [H, W]
                    pred_sample = pred_batch[sample_idx]
                    target_sample = target_batch[sample_idx]
                    pred_norm_sample = pred_norm_batch[sample_idx]
                    target_norm_sample = target_norm_batch[sample_idx]
                    
                    # Extract sample timestamp
                    sample_timestamp = self._extract_sample_timestamp(batch_data, sample_idx)
                    
                    # Extract sample valid mask (if available in batch_data)
                    sample_mask = None
                    if 'valid_mask' in batch_data:
                        sample_mask = batch_data['valid_mask'][sample_idx]  # [H, W]
                    
                    # Save single sample
                    saved_file = self.save_single_sample(
                        pred_sample,
                        target_sample,
                        pred_norm_sample,
                        target_norm_sample,
                        sample_timestamp,
                        sample_index=sample_idx,  # Use sample_idx within the batch
                        batch_mask=sample_mask,  # Pass original mask
                        model_outputs=batch_model_outputs  # Pass model outputs
                    )
                    saved_files.append(saved_file)
                    
                    global_sample_idx += 1
                    
                    # Progress display (every 10%)
                    if global_sample_idx % max(1, total_samples // 10) == 0:
                        progress = 100.0 * global_sample_idx / total_samples
                        print(f"   Sample Saving Progress: {global_sample_idx}/{total_samples} ({progress:.1f}%)")
                    
            except Exception as e:
                print(f"⚠️ Error saving batch {batch_idx}: {e}")
                continue
        
        print(f"✅ Sample saving completed, saved {len(saved_files)} sample files")
        return saved_files
    
    # Keep old method name for compatibility (marked as deprecated)
    def batch_save_from_validation(self, *args, **kwargs):
        """Deprecated: Use save_samples_from_validation instead"""
        print("⚠️ batch_save_from_validation is deprecated, use save_samples_from_validation")
        return self.save_samples_from_validation(*args, **kwargs)


class ModelLoader:
    """Model Loader - Responsible for loading model weights and initialization"""
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
        """Load checkpoint file"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        
        print(f"⏳ Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check checkpoint content
        required_keys = ['model_state_dict', 'epoch']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint file missing required keys: {missing_keys}")
        
        print(f"✅ Checkpoint loaded successfully (Train Epoch: {checkpoint['epoch']})")
        return checkpoint
    
    @staticmethod
    def initialize_model(config, device: torch.device, checkpoint_path: str) -> Tuple[CarbonNetV4, int]:
        """Initialize model and load weights"""
        # Load checkpoint
        checkpoint = ModelLoader.load_checkpoint(checkpoint_path, device)
        
        # If config exists in checkpoint, use it
        if 'config' in checkpoint:
            print("⚙️ Using configuration saved in checkpoint...")
            checkpoint_config = checkpoint['config']
            
            # Update current config to match checkpoint
            for key, value in checkpoint_config.items():
                if hasattr(config, key):
                    original_value = getattr(config, key)
                    setattr(config, key, value)
                    if str(original_value) != str(value):
                        print(f"  Config update: {key}: {original_value} -> {value}")
            
            # Reset random seeds (use setting from checkpoint)
            if hasattr(config, 'RANDOM_SEED'):
                print(f"🌱 Resetting random seed to checkpoint value: {config.RANDOM_SEED}")
                Utils.set_random_seeds(
                    seed=config.RANDOM_SEED,
                    use_deterministic=getattr(config, 'USE_DETERMINISTIC', True),
                    cuda_deterministic=getattr(config, 'CUDA_DETERMINISTIC', True)
                )
        
        # Create model
        model = CarbonNetV4(config)
        model = model.to(device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        # Get model info
        model_info = model.get_model_size()
        print(f"📊 Model Parameter Stats: {model_info}")
        
        # Return model and epoch info
        epoch = checkpoint.get('epoch', 0)
        return model, epoch

class Utils:
    """Utility Class - Responsible for auxiliary functions"""
    
    @staticmethod
    def set_random_seeds(seed: int = 42, use_deterministic: bool = True, cuda_deterministic: bool = True):
        """Set all random seeds to ensure experiment reproducibility"""
        # Python standard library random seed
        import random
        random.seed(seed)
        
        # NumPy random seed
        np.random.seed(seed)
        
        # PyTorch random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        # PyTorch deterministic settings
        if use_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        if use_deterministic and cuda_deterministic and torch.cuda.is_available():
            # Ensure deterministic CUDA operations
            torch.use_deterministic_algorithms(True)
            
        # Set environment variables
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA deterministic calculation
        
        print(f"🌱 Random seeds set: {seed} (Deterministic: {use_deterministic})")
    
    @staticmethod
    def print_results(results: Dict):
        """Print evaluation results"""
        print("\n" + "=" * 60)
        if results.get('is_autoregressive', False):
            print(f"📈 Autoregressive Prediction Results (Future {results.get('future_steps', 1)} frames)")
        else:
            print("📊 Standard Evaluation Results")
        print("=" * 60)
        
        if 'validation_loss' in results:
            print(f"Validation Loss:     {results['validation_loss']:.6f}")
        
        # Print all performance metrics
        metrics_to_show = ['mse', 'mae', 'r2', 'correlation', 'rmse']
        for metric in metrics_to_show:
            if metric in results:
                value = results[metric]
                if metric == 'mse':
                    print(f"MSE (Mean Squared Error):       {value:.6f}")
                elif metric == 'mae': 
                    print(f"MAE (Mean Absolute Error):      {value:.6f}")
                elif metric == 'rmse':
                    print(f"RMSE (Root Mean Squared Error): {value:.6f}")
                elif metric == 'r2':
                    print(f"R² (Coefficient of Determ.):    {value:.6f}")
                elif metric == 'correlation':
                    print(f"Correlation Coefficient:        {value:.6f}")
        
        # If autoregressive mode, show per-frame metrics summary
        if results.get('is_autoregressive', False) and 'per_frame_metrics' in results:
            per_frame_metrics = results['per_frame_metrics']
            print(f"\n📊 Per-frame Metrics Summary:")
            print(f"{'Frame':<6} {'Loss':<12} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'Corr':<12}")
            print("-" * 80)
            for step, frame_metrics in enumerate(per_frame_metrics):
                print(f"{step+1:<6} {frame_metrics.get('loss', 0):<12.6f} "
                      f"{frame_metrics.get('mse', 0):<12.6f} {frame_metrics.get('mae', 0):<12.6f} "
                      f"{frame_metrics.get('rmse', 0):<12.6f} {frame_metrics.get('r2', 0):<12.6f} "
                      f"{frame_metrics.get('correlation', 0):<12.6f}")
        
        print(f"\nValidation Batches:   {results.get('num_validation_batches', 'N/A')}")
        print(f"Computing Device:     {results.get('device', 'N/A')}")
        print(f"Evaluation Time:      {results.get('evaluation_time', 0):.2f} seconds")
        
        if results.get('is_autoregressive', False):
            print(f"Prediction Mode:      Autoregressive (Future {results.get('future_steps', 1)} frames)")
        else:
            print(f"Prediction Mode:      Standard Evaluation")
        print("=" * 60)

class ModelEvaluator:
    """Model Evaluator - Responsible for standard evaluation and autoregressive prediction"""
    
    def __init__(self, 
                 model: CarbonNetV4, 
                 criterion: CombinedLossV4, 
                 device: torch.device,
                 npy_saver: Optional[NPYDataSaver] = None,
                 dataset = None):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.npy_saver = npy_saver  # NPY Data Saver
        self.dataset = dataset  # Dataset reference, used to get future timestep data
    
    def validate_standard(self, val_loader, num_batches: Optional[int] = None) -> Tuple[float, Dict]:
        """Standard Validation - Fix: Save on the fly to prevent OOM"""
        self.model.eval()
        val_losses = []
        val_metrics = []
        
        max_batches = num_batches if num_batches is not None else 50
        
        print(f"🔍 Starting Standard Model Validation...")
        if self.npy_saver:
            print(f"📦 Enabling NPY instant save mode (to prevent OOM)")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                targets = {
                    'delta_gt': batch['delta_gt'],
                    'pco2_target': batch['pco2_target']
                }
                
                loss, _ = self.criterion(outputs, targets)
                metrics = self.criterion.compute_metrics(outputs, targets)
                
                val_losses.append(loss.item())
                val_metrics.append(metrics)
                
                # === [Core Modification] Save NPY data instantly, do not hoard in memory ===
                if self.npy_saver is not None:
                    # 1. Prepare data for current batch
                    pred_norm = outputs['pco2_final']
                    target_norm = targets['pco2_target']
                    pred_raw = self.npy_saver._denormalize_pco2(pred_norm, 'pco2_target')
                    target_raw = self.npy_saver._denormalize_pco2(target_norm, 'pco2_target')
                    
                    # 2. Prepare CPU data
                    batch_cpu = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    outputs_cpu = {}
                    for k, v in outputs.items():
                        if k != 'sensitivity_refine_debug':
                             outputs_cpu[k] = v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    
                    # 3. Call save immediately (only pass list of current batch)
                    try:
                        self.npy_saver.save_samples_from_validation(
                            [pred_raw.detach().cpu()],
                            [target_raw.detach().cpu()],
                            [pred_norm.detach().cpu()],
                            [target_norm.detach().cpu()],
                            [batch_cpu],
                            val_model_outputs=[outputs_cpu]
                        )
                    except Exception as e:
                        print(f"❌ NPY Save Failed (Batch {batch_idx}): {e}")
                    
                    # 4. [Important] Manually delete references to aid GC memory recovery
                    del pred_raw, target_raw, pred_norm, target_norm, batch_cpu, outputs_cpu
                    torch.cuda.empty_cache() # Slightly clean up GPU memory fragmentation

                # Limit validation batch count
                if batch_idx >= max_batches:
                    break
                
                if (batch_idx + 1) % max(1, max_batches // 10) == 0:
                    print(f"   Validation Progress: {batch_idx + 1}/{max_batches + 1}")
        
        avg_val_loss = np.mean(val_losses)
        avg_metrics = {}
        for key in val_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in val_metrics])
        
        return avg_val_loss, avg_metrics

    def validate_all_data(self, train_loader, val_loader, num_batches: Optional[int] = None) -> Tuple[float, Dict]:
        """Full Dataset Validation - Fix: Save on the fly to prevent OOM"""
        self.model.eval()
        all_losses = []
        all_metrics = []
        
        max_batches_per_loader = num_batches
        
        print(f"🔍 Starting Full Dataset Evaluation (Instant Save Mode)...")
        
        def process_loader(loader, name):
            count = 0
            for batch_idx, batch in enumerate(loader):
                if max_batches_per_loader is not None and batch_idx >= max_batches_per_loader:
                    break
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(batch)
                
                targets = {'delta_gt': batch['delta_gt'], 'pco2_target': batch['pco2_target']}
                loss, _ = self.criterion(outputs, targets)
                metrics = self.criterion.compute_metrics(outputs, targets)
                
                all_losses.append(loss.item())
                all_metrics.append(metrics)
                
                # === [Core Modification] Instant Save ===
                if self.npy_saver is not None:
                    pred_norm = outputs['pco2_final']
                    target_norm = targets['pco2_target']
                    pred_raw = self.npy_saver._denormalize_pco2(pred_norm, 'pco2_target')
                    target_raw = self.npy_saver._denormalize_pco2(target_norm, 'pco2_target')
                    
                    batch_cpu = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    outputs_cpu = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) 
                                  for k, v in outputs.items() if k != 'sensitivity_refine_debug'}
                    
                    self.npy_saver.save_samples_from_validation(
                        [pred_raw.detach().cpu()], [target_raw.detach().cpu()],
                        [pred_norm.detach().cpu()], [target_norm.detach().cpu()],
                        [batch_cpu], val_model_outputs=[outputs_cpu]
                    )
                    # Clean up memory
                    del pred_raw, target_raw, batch_cpu, outputs_cpu
                
                count += 1
                if count % 10 == 0:
                    print(f"   {name} Processing Progress: {count} batches")
            return count

        with torch.no_grad():
            print(f"\n📚 Processing Training Set Data...")
            train_count = process_loader(train_loader, "Training Set")
            
            print(f"\n📊 Processing Validation Set Data...")
            val_count = process_loader(val_loader, "Validation Set")
        
        print(f"\n✅ Full Dataset Processing Complete: Train {train_count} + Val {val_count}")
        
        avg_loss = np.mean(all_losses) if all_losses else 0.0
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_loss, avg_metrics
    
    def validate_autoregressive(self, val_loader, future_steps: int, num_batches: Optional[int] = None) -> Tuple[float, Dict]:
        """Autoregressive Validation - Multi-step Prediction"""
        if future_steps <= 1:
            raise ValueError("Autoregressive prediction requires future_steps > 1")
            
        self.model.eval()
        # Collect predictions and targets by frame: frame_metrics[step] = [predictions, targets]
        frame_predictions = [[] for _ in range(future_steps)]
        frame_targets = [[] for _ in range(future_steps)]
        
        max_batches = num_batches if num_batches is not None else 50  # Autoregressive is computationally heavy, reduce batches
        
        print(f"🔮 Starting Autoregressive Prediction Validation (Future {future_steps} frames)...")
        print(f"   Validation Batch Limit: {max_batches} (Autoregressive Mode)")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                    
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Autoregressive Prediction
                predictions, targets = self._autoregressive_predict(batch, future_steps)
                
                # Collect predictions and targets by frame
                for step in range(future_steps):
                    frame_predictions[step].append(predictions[step])
                    frame_targets[step].append(targets[step])
                
                # Progress display
                if (batch_idx + 1) % max(1, max_batches // 5) == 0:
                    progress = 100.0 * (batch_idx + 1) / max_batches
                    print(f"   Autoregressive Progress: {batch_idx + 1}/{max_batches} ({progress:.1f}%)")
        
        # Calculate per-frame metrics
        print(f"\n📊 Per-frame Performance Metrics:")
        print("-" * 80)
        frame_metrics_list = []
        for step in range(future_steps):
            # Concatenate all predictions and targets for this frame
            step_predictions = torch.cat(frame_predictions[step], dim=0)
            step_targets = torch.cat(frame_targets[step], dim=0)
            
            # Calculate metrics for this frame
            step_metrics = self._compute_simple_metrics(step_predictions, step_targets)
            step_loss = torch.nn.functional.mse_loss(step_predictions, step_targets).item()
            step_metrics['loss'] = step_loss
            frame_metrics_list.append(step_metrics)
            
            # Print metrics for this frame
            print(f"Frame {step + 1}/{future_steps}:")
            print(f"  Loss (MSE):     {step_loss:.6f}")
            print(f"  MSE:            {step_metrics['mse']:.6f}")
            print(f"  MAE:            {step_metrics['mae']:.6f}")
            print(f"  RMSE:           {step_metrics['rmse']:.6f}")
            print(f"  R²:             {step_metrics['r2']:.6f}")
            print(f"  Correlation:    {step_metrics['correlation']:.6f}")
            print("-" * 80)
        
        # Calculate overall metrics (merge all frames)
        all_predictions = torch.cat([torch.cat(frame_predictions[step], dim=0) for step in range(future_steps)], dim=0)
        all_targets = torch.cat([torch.cat(frame_targets[step], dim=0) for step in range(future_steps)], dim=0)
        
        # Use simplified metric calculation in autoregressive mode (avoid complex dependency on criterion)
        metrics = self._compute_simple_metrics(all_predictions, all_targets)
        
        # Calculate average loss (MSE)
        avg_loss = torch.nn.functional.mse_loss(all_predictions, all_targets).item()
        
        # Add per-frame metrics to return results
        metrics['per_frame_metrics'] = frame_metrics_list
        
        return avg_loss, metrics
    
    def _autoregressive_predict(self, batch: Dict, future_steps: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Execute Autoregressive Prediction"""
        predictions = []
        targets = []
        
        # Prefetch data for all future time steps (Resolve missing climate forcing and static target value issues)
        future_data_list = self._prefetch_future_data(batch, future_steps)
        
        # Initial input
        current_input = batch.copy()
        
        for step in range(future_steps):
            # Forward pass to get current prediction
            outputs = self.model(current_input)
            pred = outputs['pco2_final']  # shape: [B, H, W] - Final predicted pCO2
            
            # Use correct target value for the corresponding time step
            step_target = future_data_list[step]['pco2_target'] if future_data_list else batch['pco2_target']
            
            # Save prediction and target
            predictions.append(pred)
            targets.append(step_target)
            
            # Update input for next prediction step
            if step < future_steps - 1:  # No need to update input for the last step
                next_future_data = future_data_list[step + 1] if future_data_list else None
                current_input = self._update_input_with_prediction(current_input, pred, next_future_data)
        
        return predictions, targets
    
    def _prefetch_future_data(self, batch: Dict, future_steps: int) -> Optional[List[Dict]]:
        """
        Prefetch future time step data (Resolve missing climate forcing and static target value issues)
        
        Args:
            batch: Current batch data
            future_steps: Prediction steps
            
        Returns:
            future_data_list: List of future time step data, returns None if unavailable
        """
        if self.dataset is None:
            print("⚠️ Warning: Dataset reference not set, cannot fetch future time step data")
            return None
        
        try:
            future_data_list = []
            
            # Get time index of current batch
            if 'time_index' not in batch:
                print("⚠️ Warning: Time index missing in batch data, cannot fetch future data")
                return None
            
            # Each sample in the batch
            batch_size = batch['time_index'].shape[0]
            current_time_indices = batch['time_index'].cpu().numpy()
            
            # Prefetch data for each prediction step
            for step in range(1, future_steps + 1):  # step=1 means next time step
                step_data_batch = []
                
                # Process each sample in the batch
                for sample_idx in range(batch_size):
                    current_time_idx = current_time_indices[sample_idx].item()
                    
                    try:
                        # Get data for this sample at future step 'step'
                        future_data = self.dataset.get_future_step_data(current_time_idx, step)
                        
                        # Move to device
                        future_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                     for k, v in future_data.items()}
                        step_data_batch.append(future_data)
                        
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to fetch future step {step} data for sample {sample_idx}: {e}")
                        return None
                
                # Merge batch data
                if step_data_batch:
                    batch_future_data = self._merge_batch_future_data(step_data_batch)
                    future_data_list.append(batch_future_data)
                else:
                    return None
            
            return future_data_list
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to prefetch future data: {e}")
            return None
    
    def _merge_batch_future_data(self, step_data_batch: List[Dict]) -> Dict:
        """
        Merge future data for all samples in a batch
        
        Args:
            step_data_batch: List of future data for each sample in the batch
            
        Returns:
            batch_data: Merged batch data
        """
        batch_data = {}
        
        for key in step_data_batch[0].keys():
            if isinstance(step_data_batch[0][key], torch.Tensor):
                # Stack all sample data into a batch
                tensors = [sample_data[key] for sample_data in step_data_batch]
                if tensors[0].dim() == 0:  # Scalar data
                    batch_data[key] = torch.stack(tensors)
                else:  # Multi-dimensional data, stack on dim 0
                    batch_data[key] = torch.stack(tensors, dim=0)
            else:
                # Non-tensor data, collect as list
                batch_data[key] = [sample_data[key] for sample_data in step_data_batch]
        
        return batch_data
    
    def _compute_simple_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        Compute simplified regression metrics for autoregressive prediction evaluation
        
        Args:
            predictions: Predicted values [N, H, W]
            targets: Target values [N, H, W]
            
        Returns:
            metrics: Metrics dictionary
        """
        # Flatten data for metric calculation
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Remove NaN values
        valid_mask = ~(torch.isnan(pred_flat) | torch.isnan(target_flat))
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        if len(pred_valid) == 0:
            return {'mse': float('nan'), 'mae': float('nan'), 'r2': float('nan'), 'correlation': float('nan'), 'rmse': float('nan')}
        
        # Compute basic metrics
        mse = torch.nn.functional.mse_loss(pred_valid, target_valid).item()
        mae = torch.nn.functional.l1_loss(pred_valid, target_valid).item()
        rmse = np.sqrt(mse)
        
        # Compute R² and Correlation Coefficient
        if len(pred_valid) > 1:
            # Convert to numpy for calculation
            pred_np = pred_valid.cpu().numpy()
            target_np = target_valid.cpu().numpy()
            
            # R² Score
            ss_res = np.sum((target_np - pred_np) ** 2)
            ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
            
            # Pearson Correlation Coefficient
            correlation = np.corrcoef(pred_np, target_np)[0, 1] if len(pred_np) > 1 else float('nan')
        else:
            r2 = float('nan')
            correlation = float('nan')
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'correlation': correlation
        }
    
    def _update_input_with_prediction(self, batch: Dict, prediction: torch.Tensor, future_data: Optional[Dict] = None) -> Dict:
        """
        Update input batch with prediction values - Fix climate forcing missing and anomaly calculation errors
        
        Args:
            batch: Current input batch
            prediction: pCO2 prediction values [B, H, W]
            future_data: Future time step data (contains correct delta_clim and target values)
            
        Returns:
            updated_batch: Updated input batch
        """
        updated_batch = {}
        for k, v in batch.items():
            updated_batch[k] = v.clone() if isinstance(v, torch.Tensor) else v
        
        # === 1. Fix missing climatology forcing issue ===
        # Use correct climate forcing data from future time step
        if future_data is not None and 'delta_clim' in future_data:
            updated_batch['delta_clim'] = future_data['delta_clim']
            
            # Update other physical states as well (if available in future_data)
            for key in ['dic_current', 'alk_current', 's_thermal', 's_nonther']:
                if key in future_data:
                    updated_batch[key] = future_data[key]
        
        # === 2. Fix anomaly calculation logic error ===
        # Utilize physical relationship: current = climatology + anomaly
        # Therefore: climatology = current - anomaly
        pco2_anom_hist = updated_batch['pco2_anom_hist']  # [B, T, H, W]
        pco2_current = updated_batch['pco2_current']      # [B, H, W]
        
        # Get latest anomaly value (last time step of history sequence)
        latest_anom = pco2_anom_hist[:, -1, :, :]  # [B, H, W]
        
        # Calculate current climatology: climatology = current - anomaly
        climatology = pco2_current - latest_anom  # [B, H, W]
        
        # Correctly calculate new anomaly: anomaly = prediction - climatology
        new_anom = prediction - climatology  # [B, H, W]
        
        # Roll history anomaly sequence: remove oldest, add newest
        new_hist = torch.cat([
            pco2_anom_hist[:, 1:, :, :],  # Remove first time step [B, T-1, H, W]
            new_anom.unsqueeze(1)         # Add new anomaly [B, 1, H, W]
        ], dim=1)  # [B, T, H, W]
        
        updated_batch['pco2_anom_hist'] = new_hist
        
        # === 3. Update current pCO2 state ===
        updated_batch['pco2_current'] = prediction
        
        # Debug output (Optional)
        if hasattr(self, '_debug_update') and self._debug_update:
            print(f"🔍 Update Debug:")
            print(f"   Climatology Range: [{climatology.min().item():.3f}, {climatology.max().item():.3f}]")
            print(f"   New Anomaly Range: [{new_anom.min().item():.3f}, {new_anom.max().item():.3f}]")
            print(f"   Prediction Range: [{prediction.min().item():.3f}, {prediction.max().item():.3f}]")
            if future_data is not None:
                print(f"   Updated Climate Forcing: {list(future_data.keys())}")
        
        return updated_batch


def evaluate_model(eval_config: EvaluationConfig, 
                   config_override=None) -> Dict:
    """
    Evaluate model performance - Supports standard evaluation and autoregressive prediction
    
    Args:
        eval_config: Evaluation configuration object
        config_override: Configuration override (default uses config_v4)
        
    Returns:
        metrics: Performance metrics dictionary
    """
    start_time = time.time()
    
    # Use configuration
    cfg = config_override if config_override else config
    
    # Set random seeds to ensure reproducibility
    Utils.set_random_seeds(
        seed=cfg.RANDOM_SEED,
        use_deterministic=cfg.USE_DETERMINISTIC,
        cuda_deterministic=cfg.CUDA_DETERMINISTIC
    )
    
    # Device setup
    if eval_config.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(eval_config.device)
    
    # Determine evaluation mode
    is_autoregressive = eval_config.future_steps > 1
    
    print("=" * 60)
    print(f"AD-PINI v4 Model Evaluation - {'Autoregressive Mode' if is_autoregressive else 'Standard Mode'}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {eval_config.checkpoint_path}")
    if is_autoregressive:
        print(f"Prediction Mode: Autoregressive (Future {eval_config.future_steps} frames)")
    else:
        print(f"Prediction Mode: Standard Evaluation")
    print(f"Config: {cfg}")
    
    # 1. Data Preprocessing
    print("\n📊 Data Preprocessing...")
    preprocessor = DataPreprocessorV4(cfg)
    _ = preprocessor.process_all(force_recompute=eval_config.force_recompute)
    print("✅ Data Preprocessing Complete")
    
    # 2. Create Data Loaders
    print("\n📥 Creating Data Loaders...")
    train_loader, val_loader = create_data_loaders(cfg)
    print(f"✅ Validation Batches: {len(val_loader)}")
    
    # 3. Get Normalization Statistics
    train_dataset = train_loader.dataset
    norm_stats = train_dataset.get_normalization_stats()
    if norm_stats:
        print("✅ Normalization statistics acquired")
    
    # 4. Initialize Model
    print("\n🤖 Initializing Model...")
    model, current_epoch = ModelLoader.initialize_model(cfg, device, eval_config.checkpoint_path)
    
    # Set model normalization statistics
    if norm_stats:
        if hasattr(model, 'set_normalization_stats'):
            model.set_normalization_stats(norm_stats)
        print("✅ Model normalization statistics set")
    
    # 5. Initialize NPY Data Saver (if enabled)
    npy_saver = None
    if eval_config.save_npy:
        # Determine output directory
        output_dir = eval_config.npy_output_dir or cfg.OUTPUT_DIR
        
        # Create NPY Data Saver
        npy_saver = NPYDataSaver(
            output_dir=output_dir,
            target_height=cfg.TARGET_LAT,
            target_width=cfg.TARGET_LON,
            spatial_alignment=eval_config.spatial_alignment,
            save_extended_outputs=eval_config.save_extended_outputs  # Pass extended output save config
        )
        
        # Set normalization statistics
        if norm_stats:
            npy_saver.set_normalization_stats(norm_stats)
    
    # 6. Initialize Loss Function and Evaluator
    criterion = CombinedLossV4(cfg)
    val_dataset = val_loader.dataset  # Get validation dataset reference for autoregressive prediction
    evaluator = ModelEvaluator(model, criterion, device, npy_saver, val_dataset)
    
    # 7. Synchronize Epoch State
    print(f"🔄 Synchronizing model state to Epoch {current_epoch}...")
    
    # Synchronize internal model state (control freeze/unfreeze/gating etc.)
    if hasattr(model, 'module'):
        model.module.set_epoch(current_epoch)
    else:
        model.set_epoch(current_epoch)
    
    # Synchronize Loss/Metrics state (if dynamic weighting exists)
    if hasattr(criterion, 'set_epoch'):
        criterion.set_epoch(current_epoch)
    print(f"✅ Synchronized to Epoch {current_epoch}")
    
    # 8. Model Evaluation - Select evaluation method based on mode
    print(f"\n🔍 Starting Model {'Autoregressive' if is_autoregressive else 'Standard'} Evaluation...")
    
    if is_autoregressive:
        # Autoregressive Prediction
        val_loss, val_metrics = evaluator.validate_autoregressive(
            val_loader, eval_config.future_steps, eval_config.num_batches
        )
    else:
        # Check if NPY data needs saving (including training samples)
        if eval_config.save_npy and npy_saver is not None:
            # Full dataset evaluation mode - Process both training and validation sets
            val_loss, val_metrics = evaluator.validate_all_data(
                train_loader, val_loader, eval_config.num_batches
            )
        else:
            # Standard validation set evaluation
            val_loss, val_metrics = evaluator.validate_standard(
                val_loader, eval_config.num_batches
            )
    
    # 9. Organize Results
    # Calculate actual processed batches
    if not is_autoregressive and eval_config.save_npy and npy_saver is not None:
        # Full dataset mode: Includes training and validation batches
        train_batches = min(eval_config.num_batches or len(train_loader), len(train_loader))
        val_batches = min(eval_config.num_batches or len(val_loader), len(val_loader))
        total_batches = train_batches + val_batches
        batch_info = f"Train {train_batches} + Val {val_batches} = {total_batches}"
    else:
        # Standard mode: Validation batches only
        total_batches = min(eval_config.num_batches or len(val_loader), len(val_loader))
        batch_info = f"Val {total_batches}"
    
    results = {
        'validation_loss': val_loss,
        **val_metrics,
        'checkpoint_path': eval_config.checkpoint_path,
        'num_validation_batches': total_batches,
        'batch_info': batch_info,  # New: Detailed batch info
        'device': str(device),
        'evaluation_time': time.time() - start_time,
        'is_autoregressive': is_autoregressive,
        'future_steps': eval_config.future_steps,
        'includes_train_data': not is_autoregressive and eval_config.save_npy and npy_saver is not None  # New: Whether training data is included
    }
    
    return results


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='AD-PINI v4 Model Evaluation - Supports Standard Evaluation and Autoregressive Prediction')
    
    # Basic Arguments  /data/AD-PINI/outputs/v4/carbon_pinp_v4_20260111_114330/checkpoints/carbon_pinp_v4_20260111_114330_best.pth
    parser.add_argument('--checkpoint_path', 
                        default="/data/AD-PINI/outputs/v4/carbon_pinp_v4_20260113_145044/checkpoints/checkpoint_epoch_8.pth", 
                        type=str, help='Weight file path (.pth)')
    parser.add_argument('--num_batches', type=int, default=None, 
                        help='Limit number of validation batches (Default: Standard 50, Autoregressive 20)')
    parser.add_argument('--force_recompute', action='store_true', 
                        help='Force recompute preprocessing data')
    parser.add_argument('--device', type=str, default=None, 
                        help='Specify computing device (cuda/cpu, Default: Auto)')
    parser.add_argument('--output', type=str, default=None, 
                        help='Save results to JSON file')
    
    # New: Autoregressive Prediction Arguments
    parser.add_argument('--future_steps', type=int, default=1, 
                        help='Future prediction steps (N=1 for normal eval, N>1 for autoregressive, Default: 1)')
    
    # New: NPY Data Saving Arguments
    parser.add_argument('--save_npy', action='store_true', 
                        help='Enable NPY prediction result saving')
    parser.add_argument('--npy_output_dir', type=str, default=None,
                        help='NPY file save directory (Default: Use output directory in config)')
    parser.add_argument('--no_spatial_alignment', action='store_true',
                        help='Disable forced spatial alignment (Default: Enable alignment to 713x1440)')
    parser.add_argument('--no_extended_outputs', action='store_true',
                        help='Disable extended model output saving (Default: Enable 23-channel extended saving)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.future_steps < 1:
        print(f"❌ Error: future_steps must be >= 1, current value: {args.future_steps}")
        sys.exit(1)
    
    try:
        # Create evaluation config
        eval_config = EvaluationConfig(
            checkpoint_path=args.checkpoint_path,
            num_batches=args.num_batches,
            force_recompute=args.force_recompute,
            device=args.device,
            output_path=args.output,
            future_steps=args.future_steps,
            save_npy=args.save_npy,
            npy_output_dir=args.npy_output_dir,
            spatial_alignment=not args.no_spatial_alignment,  # Default enabled, unless no_spatial_alignment is set
            save_extended_outputs=not args.no_extended_outputs  # Default enabled, unless no_extended_outputs is set
        )
        
        # Evaluate model
        results = evaluate_model(eval_config)
        
        # Print results
        Utils.print_results(results)
        
        # Save results (Optional)
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                # Convert numpy types to Python native types for JSON serialization
                json_results = {}
                for k, v in results.items():
                    if isinstance(v, np.floating):
                        json_results[k] = float(v)
                    elif isinstance(v, np.integer):
                        json_results[k] = int(v)
                    else:
                        json_results[k] = v
                        
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to: {args.output}")
            
    except Exception as e:
        print(f"❌ Evaluation Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()