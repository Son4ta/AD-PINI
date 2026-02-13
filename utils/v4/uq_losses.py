# AD-PINI v4 UQ loss module
# High-cohesion design: UQ losses implemented and managed in an isolated module

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, Union
import logging
import numpy as np

from configs.uq_config import uq_config

class UQLoss(nn.Module):
    """Unified UQ loss implementing GNLL and related regularization terms."""
    
    def __init__(self, 
                 gnll_weight: float = 1.0,
                 var_reg_weight: float = 0.01,
                 kl_weight: float = 0.0):
        super().__init__()
        
        self.gnll_weight = gnll_weight
        self.var_reg_weight = var_reg_weight  
        self.kl_weight = kl_weight
        
        # Use PyTorch's built-in Gaussian negative log-likelihood loss
        # reduction='none' so we can aggregate manually
        self.gnll = nn.GaussianNLLLoss(reduction='none', eps=uq_config.min_variance)
        
        logging.info(f"UQLoss initialized: GNLL_weight={gnll_weight}, var_reg_weight={var_reg_weight}")
    
    def forward(self, 
                pred_dist: Dict[str, torch.Tensor], 
                target: torch.Tensor,
                target_var: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute UQ loss.
        
        Args:
            pred_dist: predictive distribution {'mu': Tensor, 'var': Tensor, 'std': Tensor}
            target: observations [B, H, W] or [B, C, H, W]
            target_var: observational uncertainty (optional) [B, H, W] or [B, C, H, W]
            mask: valid-data mask (optional) [B, H, W] or [B, C, H, W]
            
        Returns:
            total_loss: scalar total loss
            loss_components: dictionary of individual loss terms
        """
        pred_mu = pred_dist['mu']      # [B, C, H, W] or [B, H, W]
        pred_var = pred_dist['var']    # [B, C, H, W] or [B, H, W]
        
        # Ensure shapes are compatible
        if pred_mu.dim() == 3 and target.dim() == 3:
            # [B, H, W] case
            pass
        elif pred_mu.dim() == 4 and target.dim() == 3:
            # pred is [B, C, H, W], target is [B, H, W]; assume C=1
            if pred_mu.shape[1] == 1:
                pred_mu = pred_mu.squeeze(1)
                pred_var = pred_var.squeeze(1)
            else:
                raise ValueError(f"Prediction/target shape mismatch: pred={pred_mu.shape}, target={target.shape}")
        elif pred_mu.dim() == 3 and target.dim() == 4:
            # pred is [B, H, W], target is [B, C, H, W]; assume C=1
            if target.shape[1] == 1:
                target = target.squeeze(1)
                if target_var is not None:
                    target_var = target_var.squeeze(1)
            else:
                raise ValueError(f"Prediction/target shape mismatch: pred={pred_mu.shape}, target={target.shape}")
        
        # === 1. Total variance ===
        # total variance = epistemic (prediction) + aleatoric (observation)
        if target_var is not None:
            total_var = pred_var + target_var
        else:
            total_var = pred_var
        
        # Numerical stability
        total_var = torch.clamp(total_var, min=uq_config.min_variance, max=uq_config.max_variance)
        
        # === 2. GNLL loss ===
        # GNLL = 0.5 * log(2π * σ²) + (y - μ)² / (2σ²)
        # PyTorch's GaussianNLLLoss implements this formula
        gnll_loss = self.gnll(pred_mu, target, total_var)  # [B, H, W]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.bool()
            gnll_loss = gnll_loss * mask
            valid_pixels = mask.sum()
            if valid_pixels > 0:
                gnll_loss = gnll_loss.sum() / valid_pixels
            else:
                gnll_loss = torch.tensor(0.0, device=gnll_loss.device)
        else:
            gnll_loss = gnll_loss.mean()
        
        # === 3. Variance regularization ===
        # Discourage trivially large variances
        var_reg = torch.mean(pred_var)
        
        # === 4. Optional KL regularization ===
        # Constrain predictive distribution to stay close to prior (e.g. standard normal)
        kl_loss = torch.tensor(0.0, device=pred_mu.device)
        if self.kl_weight > 0:
            # KL divergence: KL(N(μ,σ²) || N(0,1)) = 0.5 * (μ² + σ² - 1 - log(σ²))
            kl_loss = 0.5 * torch.mean(pred_mu**2 + pred_var - 1 - torch.log(pred_var))
        
        # === 5. Total loss ===
        total_loss = (self.gnll_weight * gnll_loss + 
                     self.var_reg_weight * var_reg +
                     self.kl_weight * kl_loss)
        
        # Expose components for logging
        loss_components = {
            'gnll': gnll_loss.detach(),
            'var_reg': var_reg.detach(), 
            'kl': kl_loss.detach(),
            'total': total_loss.detach()
        }
        
        return total_loss, loss_components

class UQLossManager:
    """Manager that unifies deterministic and probabilistic (UQ) loss computation."""
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.is_uq_enabled = uq_config.enable_uq
        
        # Initialize underlying loss
        if self.is_uq_enabled:
            self.uq_loss = UQLoss(
                gnll_weight=uq_config.gnll_weight,
                var_reg_weight=uq_config.var_reg_weight,
                kl_weight=uq_config.kl_weight
            )
            logging.info("✅ UQ loss enabled.")
        else:
            self.mse_loss = nn.MSELoss()
            self.l1_loss = nn.L1Loss()
            logging.info("🔒 Using deterministic losses (MSE/L1).")
        
        # Loss weights inherited from base config
        self.lambda_state = getattr(base_config, 'LAMBDA_STATE', 1.0)
        self.lambda_task = getattr(base_config, 'LAMBDA_TASK', 1.0)
        self.lambda_sparsity = getattr(base_config, 'LAMBDA_SPARSITY', 0.1)
    
    def compute_loss(self, 
                    outputs: Dict[str, Any], 
                    targets: Dict[str, torch.Tensor],
                    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss according to whether UQ is enabled.
        
        Args:
            outputs: model outputs
            targets: targets
            mask: valid-data mask
            
        Returns:
            total_loss: scalar total loss
            loss_dict: dictionary with detailed components
        """
        if self.is_uq_enabled:
            return self._compute_uq_loss(outputs, targets, mask)
        else:
            return self._compute_deterministic_loss(outputs, targets, mask)
    
    def _compute_uq_loss(self, 
                        outputs: Dict[str, Any], 
                        targets: Dict[str, torch.Tensor],
                        mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full UQ-style loss computation."""
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=next(iter(targets.values())).device)
        
        # === 1. State supervision loss (SST + DIC anomaly increments) ===
        state_loss = torch.tensor(0.0, device=total_loss.device)
        
        if 'sst_anom_dist' in outputs and 'delta_sst_anom' in targets and targets['delta_sst_anom'] is not None:
            sst_loss, sst_components = self.uq_loss(
                outputs['sst_anom_dist'], targets['delta_sst_anom'], 
                targets.get('delta_sst_anom_var'), mask
            )
            state_loss = state_loss + sst_loss
            loss_dict.update({f'sst_{k}': v for k, v in sst_components.items()})
        
        if 'dic_anom_dist' in outputs and 'delta_dic_anom' in targets and targets['delta_dic_anom'] is not None:
            dic_loss, dic_components = self.uq_loss(
                outputs['dic_anom_dist'], targets['delta_dic_anom'],
                targets.get('delta_dic_anom_var'), mask
            )
            state_loss = state_loss + dic_loss
            loss_dict.update({f'dic_{k}': v for k, v in dic_components.items()})
        
        # === 2. Task loss (final pCO2) ===
        task_loss = torch.tensor(0.0, device=total_loss.device)
        
        if 'final_dist' in outputs and 'pco2_next' in targets and targets['pco2_next'] is not None:
            task_loss, task_components = self.uq_loss(
                outputs['final_dist'], targets['pco2_next'],
                targets.get('pco2_next_var'), mask
            )
            loss_dict.update({f'task_{k}': v for k, v in task_components.items()})
        
        # === 3. Sparsity loss (optional corrector regularization) ===
        sparsity_loss = torch.tensor(0.0, device=total_loss.device)
        if 'correction_dist' in outputs:
            # Encourage corrections with mean near zero and moderate variance
            correction_mu = outputs['correction_dist']['mu']
            sparsity_loss = torch.mean(correction_mu ** 2)
        
        # === 4. Combine total loss ===
        total_loss = (self.lambda_state * state_loss + 
                     self.lambda_task * task_loss +
                     self.lambda_sparsity * sparsity_loss)
        
        # Update dictionary
        loss_dict.update({
            'state_loss': state_loss.detach(),
            'task_loss': task_loss.detach(), 
            'sparsity_loss': sparsity_loss.detach(),
            'total_loss': total_loss.detach()
        })
        
        return total_loss, loss_dict
    
    def _compute_deterministic_loss(self, 
                                   outputs: Dict[str, torch.Tensor], 
                                   targets: Dict[str, torch.Tensor],
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Deterministic loss computation (backwards-compatible with original v4)."""
        loss_dict = {}
        
        # === 1. State supervision loss ===
        state_loss = torch.tensor(0.0, device=next(iter(targets.values())).device)
        
        if 'delta_sst_anom' in outputs and 'delta_sst_anom' in targets:
            sst_loss = self._apply_mask(
                self.mse_loss(outputs['delta_sst_anom'], targets['delta_sst_anom']), mask
            )
            state_loss = state_loss + sst_loss
            loss_dict['sst_mse'] = sst_loss.detach()
        
        if 'delta_dic_anom' in outputs and 'delta_dic_anom' in targets:
            dic_loss = self._apply_mask(
                self.mse_loss(outputs['delta_dic_anom'], targets['delta_dic_anom']), mask
            )
            state_loss = state_loss + dic_loss
            loss_dict['dic_mse'] = dic_loss.detach()
        
        # === 2. Task loss ===
        task_loss = torch.tensor(0.0, device=state_loss.device)
        
        if 'pco2_final' in outputs and 'pco2_next' in targets:
            task_loss = self._apply_mask(
                self.mse_loss(outputs['pco2_final'], targets['pco2_next']), mask
            )
            loss_dict['task_mse'] = task_loss.detach()
        
        # === 3. Sparsity loss ===
        sparsity_loss = torch.tensor(0.0, device=state_loss.device)
        if 'correction' in outputs:
            sparsity_loss = torch.mean(outputs['correction'] ** 2)
            loss_dict['sparsity_l2'] = sparsity_loss.detach()
        
        # === 4. Total loss ===
        total_loss = (self.lambda_state * state_loss +
                     self.lambda_task * task_loss + 
                     self.lambda_sparsity * sparsity_loss)
        
        loss_dict.update({
            'state_loss': state_loss.detach(),
            'task_loss': task_loss.detach(),
            'sparsity_loss': sparsity_loss.detach(), 
            'total_loss': total_loss.detach()
        })
        
        return total_loss, loss_dict
    
    def _apply_mask(self, loss: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply mask to per-pixel loss and average over valid pixels."""
        if mask is not None:
            mask = mask.bool()
            valid_pixels = mask.sum()
            if valid_pixels > 0:
                if loss.dim() > 0:  # if loss is not already scalar
                    loss = (loss * mask).sum() / valid_pixels
            else:
                loss = torch.tensor(0.0, device=loss.device)
        return loss

class AdaptiveLossWeighting:
    """Adaptive loss weighting to dynamically balance different components."""
    
    def __init__(self, 
                 initial_weights: Dict[str, float],
                 adaptation_rate: float = 0.01,
                 target_ratios: Optional[Dict[str, float]] = None):
        """
        Initialize adaptive weighting.
        
        Args:
            initial_weights: initial weights for loss components
            adaptation_rate: rate at which weights adapt
            target_ratios: desired relative magnitudes for each component (optional)
        """
        self.weights = initial_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.target_ratios = target_ratios or {}
        
        # Loss history to compute moving averages
        self.loss_history = {key: [] for key in initial_weights.keys()}
        self.history_length = 100
    
    def update_weights(self, loss_components: Dict[str, torch.Tensor]):
        """Update weights based on recent loss components."""
        # Update loss history
        for key, loss in loss_components.items():
            if key in self.loss_history:
                self.loss_history[key].append(float(loss))
                if len(self.loss_history[key]) > self.history_length:
                    self.loss_history[key].pop(0)
        
        # Adjust weights according to target ratios
        if len(self.loss_history['gnll']) > 10:  # require sufficient history
            for key in self.weights.keys():
                if key in self.target_ratios:
                    current_avg = np.mean(self.loss_history[key][-10:])
                    target_ratio = self.target_ratios[key]
                    
                    # Simple proportional control
                    if current_avg > 0:
                        adjustment = target_ratio / current_avg
                        self.weights[key] *= (1 + self.adaptation_rate * (adjustment - 1))
                        self.weights[key] = np.clip(self.weights[key], 0.001, 10.0)
    
    def get_weights(self) -> Dict[str, float]:
        """Return current component weights."""
        return self.weights.copy()

# =========================
# Convenience helpers
# =========================
def create_loss_manager(base_config, adaptive: bool = False) -> UQLossManager:
    """
    Convenience constructor for UQLossManager.
    
    Args:
        base_config: base configuration object
        adaptive: whether to enable adaptive weighting
        
    Returns:
        instance of UQLossManager
    """
    loss_manager = UQLossManager(base_config)
    
    if adaptive and uq_config.enable_uq:
        # Attach adaptive weighting
        initial_weights = {
            'gnll': uq_config.gnll_weight,
            'var_reg': uq_config.var_reg_weight
        }
        loss_manager.adaptive_weighting = AdaptiveLossWeighting(initial_weights)
        logging.info("✅ Adaptive loss weighting enabled.")
    
    return loss_manager

def compute_nll_score(pred_dist: Dict[str, torch.Tensor], 
                     target: torch.Tensor) -> torch.Tensor:
    """
    Compute average NLL score (for evaluation).
    
    Args:
        pred_dist: predictive distribution
        target: ground truth
        
    Returns:
        mean NLL value
    """
    pred_mu = pred_dist['mu']
    pred_var = pred_dist['var']
    
    # NLL = 0.5 * log(2π * σ²) + (y - μ)² / (2σ²)
    nll = 0.5 * torch.log(2 * np.pi * pred_var) + (target - pred_mu)**2 / (2 * pred_var)
    
    return nll.mean()

# =========================
# Simple test harness
# =========================
if __name__ == "__main__":
    print("=== Testing UQ loss functions ===")
    
    # Create test data
    B, H, W = 2, 8, 8
    
    # Simulated predictive distribution
    pred_dist = {
        'mu': torch.randn(B, H, W),
        'var': torch.rand(B, H, W) * 0.5 + 0.1,  # strictly positive
        'std': torch.sqrt(torch.rand(B, H, W) * 0.5 + 0.1)
    }
    
    # Simulated targets
    target = torch.randn(B, H, W)
    target_var = torch.rand(B, H, W) * 0.1 + 0.05
    
    # Test UQ loss
    print("\n1. UQ loss test")
    uq_loss = UQLoss()
    loss, components = uq_loss(pred_dist, target, target_var)
    print(f"Total loss: {loss:.4f}")
    print(f"Components: {components}")
    
    # Test loss manager
    print("\n2. Loss manager (UQ mode)")
    from configs.config_v4 import config
    
    # UQ mode
    uq_config.enable_uq = True
    uq_manager = UQLossManager(config)
    
    # Simulated model outputs
    outputs = {
        'final_dist': pred_dist,
        'sst_anom_dist': pred_dist,
        'dic_anom_dist': pred_dist,
        'correction_dist': pred_dist
    }
    
    # Simulated targets
    targets = {
        'pco2_next': target,
        'delta_sst_anom': target,
        'delta_dic_anom': target,
        'pco2_next_var': target_var
    }
    
    total_loss, loss_dict = uq_manager.compute_loss(outputs, targets)
    print(f"Total UQ loss: {total_loss:.4f}")
    print(f"Detailed losses: {loss_dict}")
    
    # Deterministic mode
    print("\n3. Deterministic mode test")
    uq_config.enable_uq = False
    det_manager = UQLossManager(config)
    
    det_outputs = {
        'pco2_final': pred_dist['mu'],
        'delta_sst_anom': pred_dist['mu'],
        'delta_dic_anom': pred_dist['mu'],
        'correction': pred_dist['mu']
    }
    
    det_targets = {
        'pco2_next': target,
        'delta_sst_anom': target, 
        'delta_dic_anom': target
    }
    
    det_loss, det_dict = det_manager.compute_loss(det_outputs, det_targets)
    print(f"Total deterministic loss: {det_loss:.4f}")
    print(f"Detailed deterministic losses: {det_dict}")
    
    print("\n✅ UQ loss function tests finished.")