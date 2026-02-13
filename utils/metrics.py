"""
Performance metric computation module.

Provides a complete set of metrics in both normalized and physical spaces.
"""

import numpy as np
from typing import Dict, Tuple

def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, 
                   metric_prefix: str = "") -> Dict[str, float]:
    """
    Compute metrics over valid region defined by mask.
    
    Args:
        pred: predictions (B, 1, H, W) numpy array
        gt: ground truth (B, 1, H, W) numpy array
        mask: (H, W) numpy array with values 0 or 1
        metric_prefix: optional prefix for metric names (e.g. "norm_" for normalized space)
        
    Returns:
        metrics_dict: dictionary of scalar metrics
    """
    # Ensure mask is boolean
    valid_mask = mask > 0.5
    
    # Collect valid-region values across batch
    B = pred.shape[0]
    
    all_pred = []
    all_gt = []
    
    for b in range(B):
        # Apply mask for each batch sample
        pred_valid = pred[b, 0][valid_mask]  # (N_valid,)
        gt_valid = gt[b, 0][valid_mask]      # (N_valid,)
        
        all_pred.append(pred_valid)
        all_gt.append(gt_valid)
    
    # Concatenate across batch dimension
    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)
    
    # If no valid pixels, return zeros
    if len(all_pred) == 0:
        return {
            f'{metric_prefix}mse': 0.0,
            f'{metric_prefix}rmse': 0.0,
            f'{metric_prefix}mae': 0.0,
            f'{metric_prefix}mape': 0.0,
            f'{metric_prefix}r2': 0.0,
            f'{metric_prefix}correlation': 0.0,
            f'{metric_prefix}bias': 0.0,
            f'{metric_prefix}n_samples': 0
        }
    
    # Compute metrics
    # 1. MSE (Mean Squared Error)
    mse = np.mean((all_pred - all_gt) ** 2)
    
    # 2. RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # 3. MAE (Mean Absolute Error)
    mae = np.mean(np.abs(all_pred - all_gt))
    
    # 4. MAPE (Mean Absolute Percentage Error), only where gt != 0
    nonzero_mask = np.abs(all_gt) > 1e-6
    if np.sum(nonzero_mask) > 0:
        mape = np.mean(np.abs((all_gt[nonzero_mask] - all_pred[nonzero_mask]) / all_gt[nonzero_mask])) * 100
    else:
        mape = 0.0
    
    # 5. R² (coefficient of determination)
    ss_res = np.sum((all_gt - all_pred) ** 2)
    ss_tot = np.sum((all_gt - np.mean(all_gt)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # 6. Pearson correlation coefficient
    correlation = np.corrcoef(all_pred, all_gt)[0, 1]
    
    # 7. Bias (mean error)
    bias = np.mean(all_pred - all_gt)
    
    # 8. Sample count
    n_samples = len(all_pred)
    
    metrics = {
        f'{metric_prefix}mse': float(mse),
        f'{metric_prefix}rmse': float(rmse),
        f'{metric_prefix}mae': float(mae),
        f'{metric_prefix}mape': float(mape),
        f'{metric_prefix}r2': float(r2),
        f'{metric_prefix}correlation': float(correlation),
        f'{metric_prefix}bias': float(bias),
        f'{metric_prefix}n_samples': int(n_samples)
    }
    
    return metrics

def compute_dual_space_metrics(pred_norm: np.ndarray, gt_norm: np.ndarray,
                               pred_denorm: np.ndarray, gt_denorm: np.ndarray,
                               mask: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute metrics in both normalized and real (denormalized) spaces.
    
    Args:
        pred_norm: normalized predictions (B, 1, H, W)
        gt_norm: normalized ground truth (B, 1, H, W)
        pred_denorm: physical predictions (B, 1, H, W)
        gt_denorm: physical ground truth (B, 1, H, W)
        mask: (H, W) valid-region mask
        
    Returns:
        (normalized_metrics, real_metrics): two metric dictionaries
    """
    # Normalized-space metrics
    normalized_metrics = compute_metrics(
        pred_norm, gt_norm, mask, 
        metric_prefix="norm_"
    )
    
    # Real-space metrics
    real_metrics = compute_metrics(
        pred_denorm, gt_denorm, mask,
        metric_prefix="real_"
    )
    
    return normalized_metrics, real_metrics

def format_metrics_for_display(metrics: Dict[str, float], 
                               title: str = "Performance Metrics",
                               unit: str = "") -> str:
    """
    Format metrics dictionary into a human-readable console string.
    
    Args:
        metrics: metrics dictionary
        title: section title
        unit: physical unit label (e.g. "µatm")
        
    Returns:
        formatted_string: nicely formatted text
    """
    lines = []
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)
    
    # Determine namespace prefix
    if any(k.startswith('norm_') for k in metrics.keys()):
        prefix = "norm_"
        display_prefix = "Normalized Space"
    elif any(k.startswith('real_') for k in metrics.keys()):
        prefix = "real_"
        display_prefix = "Real Space"
    else:
        prefix = ""
        display_prefix = ""
    
    if display_prefix:
        lines.append(f"Space: {display_prefix}")
    
    # Format each metric
    n_samples = metrics.get(f'{prefix}n_samples', 0)
    lines.append(f"  Samples:      {n_samples:,}")
    
    mse = metrics.get(f'{prefix}mse', 0.0)
    lines.append(f"  MSE:          {mse:.4f} {unit}²" if unit else f"  MSE:          {mse:.4f}")
    
    rmse = metrics.get(f'{prefix}rmse', 0.0)
    lines.append(f"  RMSE:         {rmse:.4f} {unit}" if unit else f"  RMSE:         {rmse:.4f}")
    
    mae = metrics.get(f'{prefix}mae', 0.0)
    lines.append(f"  MAE:          {mae:.4f} {unit}" if unit else f"  MAE:          {mae:.4f}")
    
    mape = metrics.get(f'{prefix}mape', 0.0)
    lines.append(f"  MAPE:         {mape:.2f} %")
    
    r2 = metrics.get(f'{prefix}r2', 0.0)
    lines.append(f"  R²:           {r2:.4f}")
    
    corr = metrics.get(f'{prefix}correlation', 0.0)
    lines.append(f"  Correlation:  {corr:.4f}")
    
    bias = metrics.get(f'{prefix}bias', 0.0)
    lines.append(f"  Bias:         {bias:.4f} {unit}" if unit else f"  Bias:         {bias:.4f}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)

def format_dual_metrics_for_display(norm_metrics: Dict[str, float],
                                    real_metrics: Dict[str, float],
                                    unit: str = "µatm") -> str:
    """
    Format normalized vs real-space metrics side by side for comparison.
    
    Args:
        norm_metrics: normalized-space metrics
        real_metrics: real-space metrics
        unit: physical unit label for real space
        
    Returns:
        formatted_string: formatted comparison table
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("Performance Metrics Comparison (Normalized vs Real Space)")
    lines.append("=" * 80)
    
    # Table header
    lines.append(f"{'Metric':<20} {'Normalized':<20} {'Real':<20} {'Unit':<15}")
    lines.append("-" * 80)
    
    # Sample count
    norm_n = norm_metrics.get('norm_n_samples', 0)
    real_n = real_metrics.get('real_n_samples', 0)
    lines.append(f"{'Samples':<20} {norm_n:<20,} {real_n:<20,} {'count':<15}")
    
    # MSE
    norm_mse = norm_metrics.get('norm_mse', 0.0)
    real_mse = real_metrics.get('real_mse', 0.0)
    lines.append(f"{'MSE':<20} {norm_mse:<20.4f} {real_mse:<20.4f} {'/ ' + unit + '²':<15}")
    
    # RMSE
    norm_rmse = norm_metrics.get('norm_rmse', 0.0)
    real_rmse = real_metrics.get('real_rmse', 0.0)
    lines.append(f"{'RMSE':<20} {norm_rmse:<20.4f} {real_rmse:<20.4f} {'/ ' + unit:<15}")
    
    # MAE
    norm_mae = norm_metrics.get('norm_mae', 0.0)
    real_mae = real_metrics.get('real_mae', 0.0)
    lines.append(f"{'MAE':<20} {norm_mae:<20.4f} {real_mae:<20.4f} {'/ ' + unit:<15}")
    
    # MAPE
    norm_mape = norm_metrics.get('norm_mape', 0.0)
    real_mape = real_metrics.get('real_mape', 0.0)
    lines.append(f"{'MAPE':<20} {norm_mape:<20.2f} {real_mape:<20.2f} {'%':<15}")
    
    # R²
    norm_r2 = norm_metrics.get('norm_r2', 0.0)
    real_r2 = real_metrics.get('real_r2', 0.0)
    lines.append(f"{'R²':<20} {norm_r2:<20.4f} {real_r2:<20.4f} {'-':<15}")
    
    # Correlation
    norm_corr = norm_metrics.get('norm_correlation', 0.0)
    real_corr = real_metrics.get('real_correlation', 0.0)
    lines.append(f"{'Correlation':<20} {norm_corr:<20.4f} {real_corr:<20.4f} {'-':<15}")
    
    # Bias
    norm_bias = norm_metrics.get('norm_bias', 0.0)
    real_bias = real_metrics.get('real_bias', 0.0)
    lines.append(f"{'Bias':<20} {norm_bias:<20.4f} {real_bias:<20.4f} {'/ ' + unit:<15}")
    
    lines.append("=" * 80 + "\n")
    
    return "\n".join(lines)

if __name__ == "__main__":
    # Simple self-test
    print("Testing metrics module...")
    
    # Synthetic data
    B, H, W = 2, 100, 100
    pred_norm = np.random.randn(B, 1, H, W)
    gt_norm = pred_norm + np.random.randn(B, 1, H, W) * 0.1
    
    pred_real = pred_norm * 20 + 380
    gt_real = gt_norm * 20 + 380
    
    mask = np.random.rand(H, W) > 0.3
    
    # Compute metrics in both spaces
    norm_metrics, real_metrics = compute_dual_space_metrics(
        pred_norm, gt_norm, pred_real, gt_real, mask
    )
    
    # Print comparison
    print(format_dual_metrics_for_display(norm_metrics, real_metrics, unit="µatm"))
    
    print("✓ Metrics test finished.")