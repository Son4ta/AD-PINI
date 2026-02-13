# AD-PINI v4 model architecture - anomaly prediction network based on escalator principle

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
import logging

class DoubleConv(nn.Module):
    """U-Net double convolution block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """U-Net downsampling block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """U-Net upsampling block"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch between encoder and decoder feature maps
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ConvLSTMCell(nn.Module):
    """Single-layer ConvLSTM cell"""
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool):
        """
        Initialize ConvLSTM cell.
        
        Args:
            input_dim: number of input channels
            hidden_dim: number of hidden state channels
            kernel_size: convolution kernel size
            bias: whether to use bias
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Core idea: use convolution instead of fully-connected layers
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,  # 4 gates (input, forget, cell, output)
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias)

    def forward(self, input_tensor: torch.Tensor, cur_state: tuple):
        """
        Forward pass of ConvLSTM cell.
        
        Args:
            input_tensor: current input at time t, shape [B, input_dim, H, W]
            cur_state: current state (h_cur, c_cur)
            
        Returns:
            h_next: next hidden state [B, hidden_dim, H, W]
            c_next: next cell state [B, hidden_dim, H, W]
        """
        h_cur, c_cur = cur_state
        
        # Concatenate current input and previous hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)    # input gate
        f = torch.sigmoid(cc_f)    # forget gate  
        o = torch.sigmoid(cc_o)    # output gate
        g = torch.tanh(cc_g)       # candidate values

        c_next = f * c_cur + i * g           # update cell state
        h_next = o * torch.tanh(c_next)      # update hidden state

        return h_next, c_next


class TemporalEncoder(nn.Module):
    """ConvLSTM-based temporal encoder that preserves full spatial resolution"""
    
    def __init__(self, history_length: int, spatial_channels: int = 64):
        """
        Initialize ConvLSTM temporal encoder.
        
        Args:
            history_length: length of historical time series
            spatial_channels: number of output spatial feature channels
        """
        super(TemporalEncoder, self).__init__()
        
        self.history_length = history_length
        self.spatial_channels = spatial_channels
        
        # ConvLSTM cell: input_dim=1 (one pCO2 anomaly map per time step)
        self.conv_lstm = ConvLSTMCell(
            input_dim=1, 
            hidden_dim=spatial_channels, 
            kernel_size=3, 
            bias=True
        )
        
        print(f"TemporalEncoder initialized: ConvLSTM architecture, "
              f"history_length={history_length}, spatial_channels={spatial_channels}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ConvLSTM forward pass, preserving full spatial resolution.
        
        Args:
            x: historical anomaly sequence [B, T, H, W]
            
        Returns:
            temporal_features: temporal encoding features [B, spatial_channels, H, W]
        """
        B, T, H, W = x.shape
        
        # Initialize ConvLSTM states with zeros
        h = torch.zeros(B, self.spatial_channels, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.spatial_channels, H, W, device=x.device, dtype=x.dtype)
        
        # Iterate over time steps
        for t in range(T):
            # Take current frame and add channel dimension: [B, H, W] -> [B, 1, H, W]
            x_t = x[:, t].unsqueeze(1) 
            
            # ConvLSTM state update
            h, c = self.conv_lstm(x_t, (h, c))
            
        # Return hidden state at last time step, preserving spatial resolution
        return h  # [B, spatial_channels, H, W]


class AnomalyUNet(nn.Module):
    """Anomaly decomposition U-Net - predicts pCO2 contribution components (thermal and non-thermal)"""
    
    def __init__(self, 
                 history_length: int,
                 features: list = [32, 64, 128, 256, 512],
                 bilinear: bool = True,
                 single_channel_mode: bool = False):
        super(AnomalyUNet, self).__init__()
        
        self.history_length = history_length
        self.features = features
        self.single_channel_mode = single_channel_mode  # S2 ablation: single-channel mode
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(history_length, features[0])
        
        # U-Net encoder
        self.inc = DoubleConv(features[0], features[0])  # temporal features as input
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[4] // factor)
        
        # U-Net decoder
        self.up1 = Up(features[4], features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        
        # Output head - predict pCO2 contribution components
        # S2 ablation: optional single- or dual-channel output
        output_channels = 1 if self.single_channel_mode else 2
        self.outc = nn.Conv2d(features[0], output_channels, 1)  # 1 channel (total pCO2 anomaly) or 2 channels (R_therm, R_bio)
    
    def forward(self, pco2_anom_hist: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pco2_anom_hist: historical anomaly sequence [B, T, H, W]
            
        Returns:
            pco2_contributions: predicted pCO2 contribution components
                - dual-channel mode: [B, 2, H, W] (R_therm, R_bio)
                - single-channel mode: [B, 1, H, W] -> converted to [B, 2, H, W] for interface compatibility
        """
        # Temporal encoding
        x = self.temporal_encoder(pco2_anom_hist)  # [B, C, H, W]
        
        # U-Net encoding
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # U-Net decoding
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output pCO2 contribution components
        pco2_contributions = self.outc(x)  # [B, 1, H, W] or [B, 2, H, W]
        
        # S2 ablation: in single-channel mode, expand to 2 channels for compatibility
        if self.single_channel_mode:
            # Assume thermal contribution accounts for 40% and non-thermal for 60% of total anomaly (empirical ratio)
            total_anom = pco2_contributions[:, 0]  # [B, H, W]
            R_therm = total_anom * 0.4
            R_bio = total_anom * 0.6
            pco2_contributions = torch.stack([R_therm, R_bio], dim=1)  # [B, 2, H, W]
        
        return pco2_contributions


class DifferentiablePhysicsLayer(nn.Module):
    """Differentiable physics layer implementing Taylor expansion-based pCO2 prediction"""
    
    def __init__(self):
        super(DifferentiablePhysicsLayer, self).__init__()
        
    def forward(self, 
                delta_sst_total: torch.Tensor,
                delta_dic_total: torch.Tensor,
                s_thermal: torch.Tensor,
                s_nonther: torch.Tensor,
                pco2_current: torch.Tensor) -> torch.Tensor:
        """
        Physics computation layer.
        
        Args:
            delta_sst_total: total SST change [B, H, W]
            delta_dic_total: total DIC change [B, H, W]
            s_thermal: thermal sensitivity [B, H, W]
            s_nonther: non-thermal sensitivity [B, H, W]
            pco2_current: current pCO2 [B, H, W]
            
        Returns:
            pco2_physics: physics-based pCO2 prediction [B, H, W]
        """
        # Taylor expansion:
        # ΔpCO2 = S_T * ΔSST + S_NT * ΔDIC
        delta_pco2_thermal = s_thermal * delta_sst_total      # thermal term
        delta_pco2_nonther = s_nonther * delta_dic_total      # non-thermal term
        
        delta_pco2_total = delta_pco2_thermal + delta_pco2_nonther
        
        # Predict next-step pCO2
        pco2_physics = pco2_current + delta_pco2_total
        
        return pco2_physics, delta_pco2_thermal, delta_pco2_nonther


class ResidualCorrector(nn.Module):
    """Residual correction network based on U-Net architecture with dynamic depth"""
    
    def __init__(self, features: list = [32, 64, 128]):
        super(ResidualCorrector, self).__init__()
        
        # Inputs: physics prediction + current state + delta (optional)
        self.input_channels = 3  # pco2_physics, pco2_current, delta (optional)
        self.features = features
        self.depth = len(features)
        
        # ===== Encoder - dynamically constructed =====
        # Input layer: map inputs to first feature level
        self.inc = DoubleConv(self.input_channels, features[0])
        
        # Dynamically construct downsampling layers
        self.down_layers = nn.ModuleList()
        for i in range(self.depth - 1):
            self.down_layers.append(Down(features[i], features[i + 1]))
        
        # ===== Decoder - dynamically constructed =====
        # Dynamically construct upsampling layers (reverse order)
        self.up_layers = nn.ModuleList()
        for i in range(self.depth - 1):
            # Start from bottom: features[-1] + features[-2] -> features[-2]
            in_channels = features[-(i+1)] + features[-(i+2)]  # channels for skip connection
            out_channels = features[-(i+2)]
            self.up_layers.append(Up(in_channels, out_channels, bilinear=True))
        
        # Output layer: features[0] -> 1 (correction term)
        self.outc = nn.Conv2d(features[0], 1, kernel_size=1)
        
        print(f"ResidualCorrector initialized: features={features}, "
              f"{self.depth}-layer U-Net architecture with dynamic depth")
    
    def forward(self, 
                pco2_physics: torch.Tensor,
                pco2_current: torch.Tensor,
                delta: torch.Tensor = None,) -> torch.Tensor:
        """
        Dynamic U-Net forward pass to compute residual correction.
        
        Args:
            pco2_physics: physics-based prediction [B, H, W]
            pco2_current: current pCO2 [B, H, W]
            delta: delta term [B, H, W], optional; if None it is not concatenated
            
        Returns:
            correction: correction term [B, H, W] - unconstrained, supervised by loss
        """
        # Concatenate input features: use 2 or 3 channels depending on whether delta is provided
        if delta is not None:
            x = torch.stack([pco2_physics, pco2_current, delta], dim=1)  # [B, 3, H, W]
        else:
            x = torch.stack([pco2_physics, pco2_current], dim=1)  # [B, 2, H, W]
        
        # ===== Encoder path (dynamic depth) =====
        x = self.inc(x)  # [B, features[0], H, W]
        
        # Store features for skip connections
        skip_connections = [x]
        
        # Dynamic downsampling
        for down_layer in self.down_layers:
            x = down_layer(x)
            skip_connections.append(x)
        
        # ===== Decoder path (dynamic depth + skip connections) =====
        # Start from bottom feature map and upsample
        for i, up_layer in enumerate(self.up_layers):
            skip_feature = skip_connections[-(i+2)]  # corresponding skip feature
            x = up_layer(x, skip_feature)
        
        # Output layer
        correction = self.outc(x)  # [B, 1, H, W]
        correction = correction.squeeze(1)  # [B, H, W]
        
        return correction
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'ResidualCorrector',
            'architecture': f'{self.depth}-Layer U-Net (dynamic depth design)',
            'features': self.features,
            'input_channels': self.input_channels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'depth': self.depth,
            'components': ['DoubleConv', 'Down', 'Up']
        }


class SensitivityRefiner(nn.Module):
    """Sensitivity refiner - adaptive adjustment based on physical state"""
    
    def __init__(self, hidden_dim=32, correction_range=0.15):
        """
        Initialize sensitivity refiner.
        
        Args:
            hidden_dim: hidden layer dimension
            correction_range: correction range (±correction_range)
        """
        super(SensitivityRefiner, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.correction_range = correction_range
        
        # Feature fusion network
        # Inputs: [pco2_current, R_therm, R_bio, S_T_raw, S_NT_raw] = 5 channels
        self.feature_conv = nn.Sequential(
            nn.Conv2d(5, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(inplace=True)
        )
        
        # Spatial attention mechanism - identify regions that require refinement
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_dim//2, 1, kernel_size=1),
            nn.Sigmoid()  # attention weights in [0, 1]
        )
        
        # Predictor for refinement factors
        self.refiner = nn.Sequential(
            nn.Conv2d(hidden_dim//2, 2, kernel_size=1),  # output 2 channels [alpha_T, alpha_NT]
            nn.Tanh()  # constrain correction factors to [-1, 1]
        )
        
        print(f"SensitivityRefiner initialized: hidden_dim={hidden_dim}, "
              f"correction_range=±{correction_range:.1%}")
        
    def forward(self, pco2_current, R_therm, R_bio, S_T_orig, S_NT_orig):
        """
        Forward pass: refine sensitivity parameters based on physical state.
        
        Args:
            pco2_current: current pCO2 [B, H, W]
            R_therm: thermal contribution [B, H, W] 
            R_bio: biogeochemical contribution [B, H, W]
            S_T_orig: original thermal sensitivity [B, H, W]
            S_NT_orig: original non-thermal sensitivity [B, H, W]
            
        Returns:
            S_T_refined: refined thermal sensitivity [B, H, W]
            S_NT_refined: refined non-thermal sensitivity [B, H, W]
            debug_info: debug information dictionary
        """
        # Feature concatenation: [B, 5, H, W]
        features = torch.stack([pco2_current, R_therm, R_bio, S_T_orig, S_NT_orig], dim=1)
        
        # Feature extraction
        feat = self.feature_conv(features)  # [B, hidden_dim//2, H, W]
        
        # Spatial attention weights - identify key regions to refine
        attention = self.attention(feat)    # [B, 1, H, W]
        
        # Predict refinement factors
        correction = self.refiner(feat)     # [B, 2, H, W]
        correction = correction * self.correction_range  # scale to actual correction range
        
        # Extract per-channel correction coefficients
        alpha_T = correction[:, 0] * attention.squeeze(1)   # [B, H, W] thermal refinement coefficient
        alpha_NT = correction[:, 1] * attention.squeeze(1)  # [B, H, W] non-thermal refinement coefficient
        
        # Residual refinement: S_new = S_old × (1 + α)
        S_T_refined = S_T_orig * (1 + alpha_T)
        S_NT_refined = S_NT_orig * (1 + alpha_NT)
        
        # Physical constraints - keep refined parameters within reasonable ranges
        S_T_refined = torch.clamp(S_T_refined, min=0.1, max=50.0)      # S_T must be positive
        S_NT_refined = torch.clamp(S_NT_refined, min=-10.0, max=10.0)   # reasonable range for S_NT
        
        # Debug information
        debug_info = {
            'attention_weights': attention,           # attention weight distribution
            'correction_factors': correction,         # raw correction factors
            'alpha_T_mean': alpha_T.abs().mean().item(),   # mean magnitude of T refinement
            'alpha_NT_mean': alpha_NT.abs().mean().item(), # mean magnitude of NT refinement
            'refinement_active_ratio': (attention > 0.1).float().mean().item()  # ratio of actively refined regions
        }
        
        return S_T_refined, S_NT_refined, debug_info


class CarbonNetV4(nn.Module):
    """AD-PINI v4 full model - anomaly prediction framework based on escalator principle"""
    
    def __init__(self, config):
        super(CarbonNetV4, self).__init__()
        
        self.config = config
        
        # Set random seed to ensure reproducible parameter initialization
        if hasattr(config, 'RANDOM_SEED'):
            torch.manual_seed(config.RANDOM_SEED)
        
        # Anomaly prediction network
        self.anomaly_net = AnomalyUNet(
            history_length=config.HISTORY_LENGTH,
            features=config.UNET_FEATURES,
            single_channel_mode=config.SINGLE_CHANNEL_MODE  # S2 ablation
        )
        
        # Differentiable physics layer
        self.physics_layer = DifferentiablePhysicsLayer()
        
        # Residual correction network
        self.corrector = ResidualCorrector(config.CORRECTOR_FEATURES)
        
        # M3 ablation: Black-Box MLP regressor
        if config.BLACK_BOX_MODE:
            self.black_box_regressor = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1)
            )
            logging.info("Black-Box mode enabled: skip physics layer, directly regress Latent→pCO2")
        else:
            self.black_box_regressor = None
        
        # Sensitivity refiner (optional)
        if config.ENABLE_SENSITIVITY_REFINING:
            self.sensitivity_refiner = SensitivityRefiner(
                hidden_dim=config.SENSITIVITY_REFINER_DIM,
                correction_range=config.SENSITIVITY_REFINER_CORRECTION_RANGE
            )
            logging.info(
                f"Sensitivity refiner enabled: hidden_dim={config.SENSITIVITY_REFINER_DIM}, "
                f"correction_range=±{config.SENSITIVITY_REFINER_CORRECTION_RANGE:.1%}"
            )
        else:
            self.sensitivity_refiner = None
            logging.info("Sensitivity refiner disabled")
        
        # Freeze control for corrector
        self.corrector_freeze_epochs = getattr(config, 'CORRECTOR_FREEZE_EPOCHS', 50)
        self.current_epoch = 0
        
        # Normalization statistics (to be provided by data loader)
        self.norm_stats = None
        
        logging.info("AD-PINI v4 model initialization complete")
    
    def set_epoch(self, epoch: int):
        """Set current epoch and control Corrector freezing state."""
        self.current_epoch = epoch
        is_frozen = epoch <= self.corrector_freeze_epochs
        
        # Freeze/unfreeze Corrector parameters
        for param in self.corrector.parameters():
            param.requires_grad = not is_frozen
        
        if is_frozen:
            logging.info(f"Epoch {epoch}: Corrector frozen, training only U-Net part")
        elif epoch == self.corrector_freeze_epochs + 1:
            logging.info(f"Epoch {epoch}: Corrector unfrozen, start full-model training")
    
    def set_normalization_stats(self, norm_stats: Dict[str, float]):
        """Set normalization statistics."""
        self.norm_stats = norm_stats
    
    def _denormalize(self, data: torch.Tensor, var_name: str) -> torch.Tensor:
        """Denormalize data back to original physical space - supports zscore and minmax modes."""
        if self.norm_stats is None:
            return data
        
        # Import config to get normalization mode
        from configs.config_v4 import config
        norm_mode = config.NORMALIZATION_MODE
        
        # Select statistic keys based on mode
        if norm_mode == "zscore":
            mean_key, std_key = f'{var_name}_mean', f'{var_name}_std'
            if mean_key in self.norm_stats and std_key in self.norm_stats:
                mean = torch.tensor(self.norm_stats[mean_key], device=data.device, dtype=data.dtype)
                std = torch.tensor(self.norm_stats[std_key], device=data.device, dtype=data.dtype)
                return data * std + mean
            else:
                return data
                
        elif norm_mode == "minmax":
            min_key, max_key = f'{var_name}_min', f'{var_name}_max'
            if min_key in self.norm_stats and max_key in self.norm_stats:
                min_val = torch.tensor(self.norm_stats[min_key], device=data.device, dtype=data.dtype)
                max_val = torch.tensor(self.norm_stats[max_key], device=data.device, dtype=data.dtype)
                min_range, max_range = config.NORMALIZATION_RANGE
                
                # MinMax denormalization
                normalized_01 = (data - min_range) / (max_range - min_range)
                return normalized_01 * (max_val - min_val) + min_val
            else:
                return data
        else:
            return data
    
    def _normalize(self, data: torch.Tensor, var_name: str) -> torch.Tensor:
        """Normalize data into normalized space - supports zscore and minmax modes."""
        if self.norm_stats is None:
            return data
        
        # Import config to get normalization mode
        from configs.config_v4 import config
        norm_mode = config.NORMALIZATION_MODE
        
        # Select statistic keys based on mode
        if norm_mode == "zscore":
            mean_key, std_key = f'{var_name}_mean', f'{var_name}_std'
            if mean_key in self.norm_stats and std_key in self.norm_stats:
                mean = torch.tensor(self.norm_stats[mean_key], device=data.device, dtype=data.dtype)
                std = torch.tensor(self.norm_stats[std_key], device=data.device, dtype=data.dtype)
                return (data - mean) / (std + config.NORMALIZATION_EPSILON)
            else:
                return data
                
        elif norm_mode == "minmax":
            min_key, max_key = f'{var_name}_min', f'{var_name}_max'
            if min_key in self.norm_stats and max_key in self.norm_stats:
                min_val = torch.tensor(self.norm_stats[min_key], device=data.device, dtype=data.dtype)
                max_val = torch.tensor(self.norm_stats[max_key], device=data.device, dtype=data.dtype)
                min_range, max_range = config.NORMALIZATION_RANGE
                
                # MinMax normalization
                normalized = (data - min_val) / (max_val - min_val + config.NORMALIZATION_EPSILON)
                return normalized * (max_range - min_range) + min_range
            else:
                return data
        else:
            return data
    
    def _normalize_delta(self, delta_data: torch.Tensor, var_name: str) -> torch.Tensor:
        """Normalize increment data - use only standard deviation, without mean shift.
        
        For Z-Score: Δx_norm = Δx / σ (no mean subtraction, since delta is already a change).
        For MinMax: Δx_norm = Δx / (max - min) (no offset added).
        """
        if self.norm_stats is None:
            return delta_data
        
        # Import config to get normalization mode
        from configs.config_v4 import config
        norm_mode = config.NORMALIZATION_MODE
        
        if norm_mode == "zscore":
            std_key = f'{var_name}_std'
            if std_key in self.norm_stats:
                std = torch.tensor(self.norm_stats[std_key], device=delta_data.device, dtype=delta_data.dtype)
                return delta_data / (std + config.NORMALIZATION_EPSILON)
            else:
                return delta_data
                
        elif norm_mode == "minmax":
            # For minmax mode, use value range as scaling factor
            min_key, max_key = f'{var_name}_min', f'{var_name}_max'
            if min_key in self.norm_stats and max_key in self.norm_stats:
                min_val = torch.tensor(self.norm_stats[min_key], device=delta_data.device, dtype=delta_data.dtype)
                max_val = torch.tensor(self.norm_stats[max_key], device=delta_data.device, dtype=delta_data.dtype)
                scale = max_val - min_val + config.NORMALIZATION_EPSILON
                return delta_data / scale
            else:
                return delta_data
        else:
            return delta_data
    
    def _denormalize_delta(self, delta_data: torch.Tensor, var_name: str) -> torch.Tensor:
        """Denormalize increment data - use only standard deviation, without mean shift.
        
        For Z-Score: Δx = Δx_norm * σ (no mean added, since delta is already a change).
        For MinMax: Δx = Δx_norm * (max - min) (no offset added).
        """
        if self.norm_stats is None:
            return delta_data
        
        # Import config to get normalization mode
        from configs.config_v4 import config
        norm_mode = config.NORMALIZATION_MODE
        
        if norm_mode == "zscore":
            std_key = f'{var_name}_std'
            if std_key in self.norm_stats:
                std = torch.tensor(self.norm_stats[std_key], device=delta_data.device, dtype=delta_data.dtype)
                return delta_data * (std + config.NORMALIZATION_EPSILON)
            else:
                return delta_data
                
        elif norm_mode == "minmax":
            # For minmax mode, use value range as scaling factor
            min_key, max_key = f'{var_name}_min', f'{var_name}_max'
            if min_key in self.norm_stats and max_key in self.norm_stats:
                min_val = torch.tensor(self.norm_stats[min_key], device=delta_data.device, dtype=delta_data.dtype)
                max_val = torch.tensor(self.norm_stats[max_key], device=delta_data.device, dtype=delta_data.dtype)
                scale = max_val - min_val + config.NORMALIZATION_EPSILON
                return delta_data * scale
            else:
                return delta_data
        else:
            return delta_data
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            batch: input batch data
            
        Returns:
            outputs: dictionary of model outputs
        """
        # Extract inputs
        pco2_anom_hist = batch['pco2_anom_hist']    # [B, T, H, W]
        delta_clim = batch['delta_clim']            # [B, 2, H, W] (SST, DIC)
        pco2_current = batch['pco2_current']        # [B, H, W]
        s_thermal = batch['s_thermal']              # [B, H, W]
        s_nonther = batch['s_nonther']              # [B, H, W]
        
        # === 1. Anomaly decomposition network (Anomaly Decomposition Net) ===
        # M3 ablation: in Black-Box mode directly regress final pCO2 and skip physics computation
        if self.config.BLACK_BOX_MODE:
            # Directly regress pCO2 from historical sequence, skipping physical decomposition
            pco2_contributions = self.anomaly_net(pco2_anom_hist)  # [B, 2, H, W]
            pco2_final = self.black_box_regressor(pco2_contributions).squeeze(1)  # [B, H, W]
            
            # To keep output format consistent, fill with dummy values
            R_therm = torch.zeros_like(pco2_final)
            R_bio = torch.zeros_like(pco2_final)
            correction = torch.zeros_like(pco2_final)
            pco2_physics = pco2_final.clone()
            delta_thermal = torch.zeros_like(pco2_final)
            delta_nonther = torch.zeros_like(pco2_final)
            delta_sst_anom = torch.zeros_like(pco2_final)
            delta_dic_anom = torch.zeros_like(pco2_final)
            pco2_anom_reconstructed = torch.zeros_like(pco2_final)
            S_T = torch.zeros_like(pco2_final)
            S_NT = torch.zeros_like(pco2_final)
            delta_sst_total = torch.zeros_like(pco2_final)
            delta_dic_total = torch.zeros_like(pco2_final)
            refine_debug = None
            
        else:
            # Standard pipeline: predict pCO2 contributions rather than physical variables directly
            pco2_contributions = self.anomaly_net(pco2_anom_hist)  # [B, 2, H, W]
            R_therm = self._denormalize_delta(pco2_contributions[:, 0], 'pco2')    # [B, H, W] thermal contribution (μatm)
            R_bio = self._denormalize_delta(pco2_contributions[:, 1], 'pco2')     # [B, H, W] non-thermal contribution (μatm)
            
            # === 2. Physics inversion ===
            S_T = self._denormalize(s_thermal, 's_thermal')
            # Compute non-thermal sensitivity
            S_NT = self._denormalize(s_nonther, 's_nonther')
        
            # === 2.1 Sensitivity refinement ===
            refine_debug = None
            if self.sensitivity_refiner is not None:
                # Refine sensitivity parameters based on current physical state
                S_T, S_NT, refine_debug = self.sensitivity_refiner(
                    pco2_current, R_therm, R_bio, S_T, S_NT
                )
        
            # Physics inversion: infer physical anomalies from pCO2 contributions
            # Add numerical stability protection
            R_therm_safe = torch.clamp(R_therm, min=-50.0, max=50.0)  # limit range of thermal contribution
            R_bio_safe = torch.clamp(R_bio, min=-100.0, max=100.0)    # limit range of non-thermal contribution
        
            delta_sst_anom = R_therm_safe / (S_T + 1e-8)  # [B, H, W] inferred SST anomaly (°C)
            delta_dic_anom = R_bio_safe / (S_NT + 1e-8)   # [B, H, W] inferred DIC anomaly (μmol/kg)
        
            # Apply physical constraints to inversion results
            delta_sst_anom = torch.clamp(delta_sst_anom, min=-6.9, max=6.7)     # SST anomalies
            delta_dic_anom = torch.clamp(delta_dic_anom, min=-251.2, max=259.0) # DIC anomalies
        
            delta_sst_anom_norm = self._normalize_delta(delta_sst_anom, 'delta_sst')
            delta_dic_anom_norm = self._normalize_delta(delta_dic_anom, 'delta_dic')

            # Physics constraint: pCO2 anomaly decomposition
            pco2_anom_reconstructed = R_therm + R_bio  # [B, H, W] reconstructed pCO2 anomaly
        
            # === 3. Increment synthesis (escalator principle) ===
            # Total increment = climatological increment (given) + anomaly increment (inverted)
            delta_sst_clim = delta_clim[:, 0]  # [B, H, W]
            delta_dic_clim = delta_clim[:, 1]  # [B, H, W]
        
            delta_sst_total = delta_sst_clim + delta_sst_anom_norm  # normalized space
            delta_dic_total = delta_dic_clim + delta_dic_anom_norm  # normalized space
        
            # === 3. Physics computation ===
            # Physics calculations should be done in physical space, so denormalize first
            delta_sst_total_phys = self._denormalize_delta(delta_sst_total, 'delta_sst')
            delta_dic_total_phys = self._denormalize_delta(delta_dic_total, 'delta_dic')
            s_thermal_phys = self._denormalize(s_thermal, 's_thermal')
            s_nonther_phys = self._denormalize(s_nonther, 's_nonther')
            pco2_current_phys = self._denormalize(pco2_current, 'pco2_current')
        
            # Clip denormalized data to avoid extreme values
            delta_sst_total_phys = torch.clamp(delta_sst_total_phys, -6.9, 6.7)  # limit SST change
            delta_dic_total_phys = torch.clamp(delta_dic_total_phys, -251.2, 259.0)  # limit DIC change
            s_thermal_phys = torch.clamp(s_thermal_phys, 4.0, 26.2)  # limit thermal sensitivity
            s_nonther_phys = torch.clamp(s_nonther_phys, 0.2, 23.0)  # limit non-thermal sensitivity
            pco2_current_phys = torch.clamp(pco2_current_phys, 94.0, 619.0)  # limit pCO2 to reasonable range
        
            pco2_physics_phys, delta_thermal_phys, delta_nonther_phys = self.physics_layer(
                delta_sst_total_phys, delta_dic_total_phys, s_thermal_phys, s_nonther_phys, pco2_current_phys
            )
        
            # Clip physics outputs
            pco2_physics_phys = torch.clamp(pco2_physics_phys, 50.0, 750.0)  # limit final pCO2
            # delta_thermal_phys = torch.clamp(delta_thermal_phys, -20.0, 20.0)
            # delta_nonther_phys = torch.clamp(delta_nonther_phys, -50.0, 50.0)
        
            # Convert physics outputs back to normalized space
            pco2_physics = self._normalize(pco2_physics_phys, 'pco2')
        
            # Use correct normalization for pCO2 increments - only std, no mean shift
            delta_thermal = self._normalize_delta(delta_thermal_phys, 'pco2')
            delta_nonther = self._normalize_delta(delta_nonther_phys, 'pco2')
        
            delta_pco2_physics = delta_thermal + delta_nonther
        
            # === 4. Residual correction ===
            # M2 ablation: Physics-Only mode with correction=0 to test Taylor truncation error
            if self.config.PHYSICS_ONLY_MODE or self.current_epoch <= self.corrector_freeze_epochs:
                correction = torch.zeros_like(pco2_physics)
                pco2_final = pco2_physics
            else:
                correction = self.corrector(pco2_physics, pco2_current, delta_pco2_physics)
                pco2_final = pco2_physics + correction
        
        # === Outputs ===
        outputs = {            
            # Network predictions - pCO2 contribution components
            'pco2_contributions': pco2_contributions,  # [B, 2, H, W]
            'R_therm': R_therm,                        # [B, H, W] thermal contribution (μatm)
            'R_bio': R_bio,                            # [B, H, W] non-thermal contribution (μatm)
            
            # Physics inversion results
            'delta_sst_anom': delta_sst_anom,          # [B, H, W]
            'delta_dic_anom': delta_dic_anom,          # [B, H, W]
            
            # Physics constraints
            'pco2_anom_reconstructed': pco2_anom_reconstructed,  # [B, H, W] reconstructed pCO2 anomaly
            'S_T': S_T,                                # [B, H, W] thermal sensitivity
            'S_NT': S_NT,                              # [B, H, W] non-thermal sensitivity
            
            # Combined results
            'delta_sst_total': delta_sst_total,        # [B, H, W]
            'delta_dic_total': delta_dic_total,        # [B, H, W]
            
            # Physics computation outputs
            'pco2_physics': pco2_physics,              # [B, H, W]
            'delta_thermal': delta_thermal,            # [B, H, W]
            'delta_nonther': delta_nonther,            # [B, H, W]
            
            # Final results
            'correction': correction,                  # [B, H, W]
            'pco2_final': pco2_final,                  # [B, H, W]
            
            # Intermediate variables (for analysis)
            'pco2_current': pco2_current,              # [B, H, W]
            's_thermal': s_thermal,                    # [B, H, W]
            's_nonther': s_nonther,                    # [B, H, W]
            
            # Sensitivity refinement debug info (optional)
            'sensitivity_refine_debug': refine_debug,  # Dict or None
        }
        
        return outputs
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'anomaly_net_params': sum(p.numel() for p in self.anomaly_net.parameters()),
            'corrector_params': sum(p.numel() for p in self.corrector.parameters()),
        }


def main():
    """Simple model test."""
    from configs.config_v4 import config
    
    # Create model
    model = CarbonNetV4(config)
    
    # Model statistics
    model_info = model.get_model_size()
    print("Model information:")
    for key, value in model_info.items():
        print(f"  {key}: {value:,}")
    
    # Create test data
    batch_size = 2
    H, W = 713, 1440
    T = config.HISTORY_LENGTH
    
    test_batch = {
        'pco2_anom_hist': torch.randn(batch_size, T, H, W),
        'delta_clim': torch.randn(batch_size, 2, H, W),
        'pco2_current': torch.randn(batch_size, H, W) + 400,
        's_thermal': torch.randn(batch_size, H, W) * 0.1 + 0.04,
        's_nonther': torch.randn(batch_size, H, W) * 5 + 10,
    }
    
    # Forward pass test
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(test_batch)
    
    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\nModel test finished.")


if __name__ == "__main__":
    main()