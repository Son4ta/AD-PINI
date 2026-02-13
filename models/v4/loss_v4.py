# AD-PINI v4 loss functions - escalator principle loss computation

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

class StateSupervisionLoss(nn.Module):
    """State supervision loss - strongly constrains physical variable decomposition."""
    
    def __init__(self, weight: float = 10.0):
        super(StateSupervisionLoss, self).__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, 
                delta_sst_pred: torch.Tensor,
                delta_dic_pred: torch.Tensor,
                delta_sst_gt: torch.Tensor,
                delta_dic_gt: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute state supervision loss.
        
        Args:
            delta_sst_pred: predicted SST increment [B, H, W]
            delta_dic_pred: predicted DIC increment [B, H, W]
            delta_sst_gt: ground-truth SST increment [B, H, W]
            delta_dic_gt: ground-truth DIC increment [B, H, W]
            mask: valid-region mask [B, H, W]
            
        Returns:
            loss: state supervision loss
        """
        # Compute MSE loss
        sst_loss = self.mse_loss(delta_sst_pred, delta_sst_gt)
        dic_loss = self.mse_loss(delta_dic_pred, delta_dic_gt)
        
        # Apply mask if provided
        if mask is not None:
            sst_loss = sst_loss * mask
            dic_loss = dic_loss * mask
            
            # Average over valid pixels
            valid_pixels = mask.sum()
            if valid_pixels > 0:
                sst_loss = sst_loss.sum() / valid_pixels
                dic_loss = dic_loss.sum() / valid_pixels
            else:
                sst_loss = sst_loss.mean()
                dic_loss = dic_loss.mean()
        else:
            sst_loss = sst_loss.mean()
            dic_loss = dic_loss.mean()
        
        total_loss = self.weight * (sst_loss + dic_loss)
        
        return total_loss, {'sst_loss': sst_loss.item(), 'dic_loss': dic_loss.item()}


class TaskLoss(nn.Module):
    """Task loss - accuracy of final pCO2 prediction."""
    
    def __init__(self, weight: float = 1.0, physics_weight: float = 0.5):
        super(TaskLoss, self).__init__()
        self.weight = weight
        self.physics_weight = physics_weight
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self,
                pco2_final: torch.Tensor,
                pco2_physics: torch.Tensor,
                pco2_target: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute task loss.
        
        Args:
            pco2_final: final predicted pCO2 [B, H, W]
            pco2_physics: physics-based predicted pCO2 [B, H, W]
            pco2_target: target pCO2 [B, H, W]
            mask: valid-region mask [B, H, W]
            
        Returns:
            loss: task loss
        """
        # Loss for final prediction
        final_loss = self.mse_loss(pco2_final, pco2_target)
        
        # Physics-based prediction loss
        physics_loss = self.mse_loss(pco2_physics, pco2_target)
        
        # Apply mask if provided
        if mask is not None:
            final_loss = final_loss * mask
            physics_loss = physics_loss * mask
            
            valid_pixels = mask.sum()
            if valid_pixels > 0:
                final_loss = final_loss.sum() / valid_pixels
                physics_loss = physics_loss.sum() / valid_pixels
            else:
                final_loss = final_loss.mean()
                physics_loss = physics_loss.mean()
        else:
            final_loss = final_loss.mean()
            physics_loss = physics_loss.mean()
        
        # Combined loss
        total_loss = self.weight * (final_loss + self.physics_weight * physics_loss)
        
        return total_loss, {
            'final_pco2_loss': final_loss.item(),
            'physics_pco2_loss': physics_loss.item()
        }


class SparsityLoss(nn.Module):
    """Sparsity loss for correction term - encourages Occam's razor."""
    
    def __init__(self, weight: float = 0.1):
        super(SparsityLoss, self).__init__()
        self.weight = weight
    
    def forward(self, correction: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute sparsity loss for correction term.
        
        Args:
            correction: correction term [B, H, W]
            mask: valid-region mask [B, H, W]
            
        Returns:
            loss: sparsity loss
        """
        # L1 regularization - encourage correction to be as small as possible
        l1_loss = torch.abs(correction)
        
        if mask is not None:
            l1_loss = l1_loss * mask
            valid_pixels = mask.sum()
            if valid_pixels > 0:
                l1_loss = l1_loss.sum() / valid_pixels
            else:
                l1_loss = l1_loss.mean()
        else:
            l1_loss = l1_loss.mean()
        
        total_loss = self.weight * l1_loss
        
        return total_loss, {'correction_l1': l1_loss.item()}


class PhysicsConsistencyLoss(nn.Module):
    """Physics consistency loss - ensures physical computations are reasonable."""
    
    def __init__(self, weight: float = 0.01):
        super(PhysicsConsistencyLoss, self).__init__()
        self.weight = weight
    
    def forward(self,
                delta_thermal: torch.Tensor,
                delta_nonther: torch.Tensor,
                delta_sst_total: torch.Tensor,
                delta_dic_total: torch.Tensor,
                s_thermal: torch.Tensor,
                s_nonther: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute physics consistency loss.
        
        Args:
            delta_thermal: thermal term [B, H, W]
            delta_nonther: non-thermal term [B, H, W]
            delta_sst_total: total SST change [B, H, W]
            delta_dic_total: total DIC change [B, H, W]
            s_thermal: thermal sensitivity [B, H, W]
            s_nonther: non-thermal sensitivity [B, H, W]
            mask: valid-region mask [B, H, W]
            
        Returns:
            loss: physics consistency loss
        """
        # Verify physical formula:
        # delta_thermal should equal s_thermal * delta_sst_total
        expected_thermal = s_thermal * delta_sst_total
        thermal_consistency = torch.abs(delta_thermal - expected_thermal)
        
        # delta_nonther should equal s_nonther * delta_dic_total
        expected_nonther = s_nonther * delta_dic_total
        nonther_consistency = torch.abs(delta_nonther - expected_nonther)
        
        total_consistency = thermal_consistency + nonther_consistency
        
        if mask is not None:
            total_consistency = total_consistency * mask
            valid_pixels = mask.sum()
            if valid_pixels > 0:
                total_consistency = total_consistency.sum() / valid_pixels
            else:
                total_consistency = total_consistency.mean()
        else:
            total_consistency = total_consistency.mean()
        
        total_loss = self.weight * total_consistency
        
        return total_loss, {
            'thermal_consistency': thermal_consistency.mean().item(),
            'nonther_consistency': nonther_consistency.mean().item()
        }


class CombinedLossV4(nn.Module):
    """Combined loss function for AD-PINI v4."""
    
    def __init__(self, config):
        super(CombinedLossV4, self).__init__()
        
        self.config = config
        
        # Loss components
        self.state_loss = StateSupervisionLoss(config.LAMBDA_STATE)
        self.task_loss = TaskLoss(config.LAMBDA_TASK)
        self.sparsity_loss = SparsityLoss(config.LAMBDA_SPARSITY)
        self.physics_loss = PhysicsConsistencyLoss(0.0)  # small weight by default
        
        # Corrector freeze settings
        self.corrector_freeze_epochs = getattr(config, 'CORRECTOR_FREEZE_EPOCHS', 50)
        self.current_epoch = 0
        
    def set_epoch(self, epoch: int):
        """Set current epoch for dynamic weight adjustment."""
        self.current_epoch = epoch
    
    def _compute_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Compute valid-region mask."""
        # Build mask based on target values (exclude outliers)
        mask = torch.isfinite(target) & (torch.abs(target) < 1000)  # exclude extreme values
        return mask.float()
    
    def _get_dynamic_weights(self) -> Dict[str, float]:
        """Dynamic weight adjustment implementing Corrector freezing logic."""
        # Corrector freeze phase: only train U-Net in the first N epochs
        is_corrector_frozen = self.current_epoch <= self.corrector_freeze_epochs
        
        if is_corrector_frozen:
            # During freeze: only compute state loss (SST/DIC increments) and physics consistency loss
            weights = {
                'state_weight': self.config.LAMBDA_STATE,
                'task_weight': 0.0,        # disable final pCO2 loss
                'sparsity_weight': 0.0,    # disable correction sparsity loss
                'physics_weight': 0.0      # keep physics consistency
            }
        else:
            # Normal training phase: all losses enabled
            weights = {
                'state_weight': self.config.LAMBDA_STATE,
                'task_weight': self.config.LAMBDA_TASK,
                'sparsity_weight': self.config.LAMBDA_SPARSITY,
                'physics_weight': 0.0
            }
        
        return weights, is_corrector_frozen
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.
        
        Args:
            outputs: model output dictionary
            targets: target data dictionary
            
        Returns:
            total_loss: total loss
            loss_dict: detailed loss components
        """
        # Compute masks
        pco2_mask = self._compute_mask(targets['pco2_target'])
        delta_mask = self._compute_mask(targets['delta_gt'][:, 0])  # based on SST increment
        
        # Get dynamic weights
        weights, is_corrector_frozen = self._get_dynamic_weights()
        
        # === 1. State supervision loss ===
        delta_gt_sst = targets['delta_gt'][:, 0]  # [B, H, W]
        delta_gt_dic = targets['delta_gt'][:, 1]  # [B, H, W]
        
        state_loss_val, state_details = self.state_loss(
            outputs['delta_sst_total'],
            outputs['delta_dic_total'],
            delta_gt_sst,
            delta_gt_dic,
            delta_mask
        )
        
        # === 2. Task loss ===
        task_loss_val, task_details = self.task_loss(
            outputs['pco2_final'],
            outputs['pco2_physics'],
            targets['pco2_target'],
            pco2_mask
        )
        
        # === 3. Sparsity loss ===
        sparsity_loss_val, sparsity_details = self.sparsity_loss(
            outputs['correction'],
            pco2_mask
        )
        
        # === 4. Physics consistency loss ===
        physics_loss_val, physics_details = self.physics_loss(
            outputs['delta_thermal'],
            outputs['delta_nonther'],
            outputs['delta_sst_total'],
            outputs['delta_dic_total'],
            outputs['s_thermal'],
            outputs['s_nonther'],
            delta_mask
        )
        
        # === Total loss - dynamically combined based on freeze state ===
        total_loss = (weights['state_weight'] * state_loss_val +
                     weights['task_weight'] * task_loss_val + 
                     weights['sparsity_weight'] * sparsity_loss_val + 
                     weights['physics_weight'] * physics_loss_val)
        
        # === Loss details ===
        loss_dict = {
            'total_loss': total_loss.item(),
            'state_loss': state_loss_val.item(),
            'task_loss': task_loss_val.item(),
            'sparsity_loss': sparsity_loss_val.item(),
            'physics_loss': physics_loss_val.item(),
            **state_details,
            **task_details,
            **sparsity_details,
            **physics_details,
            'valid_pixels_ratio': pco2_mask.mean().item(),
            'is_corrector_frozen': is_corrector_frozen,
            'corrector_freeze_epochs': self.corrector_freeze_epochs
        }
        
        return total_loss, loss_dict
    
    def compute_metrics(self, 
                       outputs: Dict[str, torch.Tensor], 
                       targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            outputs: model outputs
            targets: target data
            
        Returns:
            metrics: dictionary of evaluation metrics
        """
        with torch.no_grad():
            # Compute mask
            mask = self._compute_mask(targets['pco2_target'])
            
            # Extract valid predictions and targets
            pred = outputs['pco2_final'][mask > 0]
            gt = targets['pco2_target'][mask > 0]
            
            if len(pred) == 0:
                return {'mse': float('inf'), 'mae': float('inf'), 'r2': 0.0}
            
            # MSE
            mse = F.mse_loss(pred, gt).item()
            
            # MAE
            mae = F.l1_loss(pred, gt).item()
            
            # R²
            ss_res = torch.sum((pred - gt) ** 2)
            ss_tot = torch.sum((gt - torch.mean(gt)) ** 2)
            r2 = 1 - (ss_res / ss_tot).item() if ss_tot > 0 else 0.0
            
            # Correlation coefficient
            pred_centered = pred - torch.mean(pred)
            gt_centered = gt - torch.mean(gt)
            correlation = torch.sum(pred_centered * gt_centered) / (
                torch.sqrt(torch.sum(pred_centered ** 2)) * torch.sqrt(torch.sum(gt_centered ** 2))
            )
            correlation = correlation.item() if torch.isfinite(correlation) else 0.0
            
            # CRPS (Continuous Ranked Probability Score)
            # For deterministic predictions, CRPS = MAE
            crps = mae
            
            # Performance of physical decomposition
            delta_pred_sst = outputs['delta_sst_total'][mask > 0]
            delta_gt_sst = targets['delta_gt'][:, 0][mask > 0]
            delta_pred_dic = outputs['delta_dic_total'][mask > 0]
            delta_gt_dic = targets['delta_gt'][:, 1][mask > 0]
            
            sst_mse = F.mse_loss(delta_pred_sst, delta_gt_sst).item() if len(delta_pred_sst) > 0 else float('inf')
            dic_mse = F.mse_loss(delta_pred_dic, delta_gt_dic).item() if len(delta_pred_dic) > 0 else float('inf')
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'correlation': correlation,
                'crps': crps,
                'delta_sst_mse': sst_mse,
                'delta_dic_mse': dic_mse,
                'correction_magnitude': torch.abs(outputs['correction']).mean().item()
            }
            
            return metrics


def main():
    """Simple test for loss functions."""
    from configs.config_v4 import config
    
    # Create loss function
    criterion = CombinedLossV4(config)
    
    # Create test data
    batch_size = 2
    H, W = 713, 1440
    
    outputs = {
        'delta_sst_total': torch.randn(batch_size, H, W),
        'delta_dic_total': torch.randn(batch_size, H, W),
        'pco2_final': torch.randn(batch_size, H, W) + 400,
        'pco2_physics': torch.randn(batch_size, H, W) + 400,
        'correction': torch.randn(batch_size, H, W) * 0.1,
        'delta_thermal': torch.randn(batch_size, H, W),
        'delta_nonther': torch.randn(batch_size, H, W),
        's_thermal': torch.ones(batch_size, H, W) * 0.04,
        's_nonther': torch.ones(batch_size, H, W) * 10,
    }
    
    targets = {
        'delta_gt': torch.randn(batch_size, 2, H, W),
        'pco2_target': torch.randn(batch_size, H, W) + 400,
    }
    
    # Compute loss
    total_loss, loss_dict = criterion(outputs, targets)
    
    print("Loss computation results:")
    print(f"Total loss: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Compute metrics
    metrics = criterion.compute_metrics(outputs, targets)
    print("\nEvaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()