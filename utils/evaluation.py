"""
Model evaluation and result saving module (v2).

Handles visualization, CSV export, and metric computation.
v2 changes: adapted to new data structures and multi-timestep outputs.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Optional

from utils.metrics import compute_dual_space_metrics, format_dual_metrics_for_display
from utils.experiment import save_metrics_json
from utils.visualize import Visualizer

def denormalize_data(data_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Denormalize data using x = x_norm * std + mean.
    
    Args:
        data_norm: normalized data
        mean: mean value
        std: standard deviation
        
    Returns:
        denormalized data
    """
    return data_norm * std + mean

def save_visualization_and_predictions(
    model,
    dataloader,
    device,
    stats: Dict,
    save_dir: Path,
    epoch: int,
    mask_cpu: np.ndarray,
    logger=None
) -> Dict[str, float]:
    """
    v2 API: save visualization figures, prediction CSVs, and metrics in both normalized and real spaces.
    
    v2 changes:
    1. Adapt to new 6-element batch unpacking.
    2. Handle temporal dimension T_out.
    3. Save physical parameters (ΔSST, ΔDIC, β) into CSV.
    
    Args:
        model: trained model
        dataloader: evaluation dataloader
        device: torch device
        stats: normalization statistics
        save_dir: experiment root directory
        epoch: current epoch
        mask_cpu: mask on CPU (H, W)
        logger: optional logger
    
    Returns:
        combined_metrics: merged metrics dictionary (normalized and real spaces)
    """
    model.eval()
    
    # Create output directories
    vis_dir = save_dir / "visualizations" / f"epoch_{epoch}"
    csv_dir = save_dir / "predictions" / f"epoch_{epoch}"
    metrics_dir = save_dir / "metrics"
    
    vis_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualizer
    visualizer = Visualizer(save_dir=vis_dir)
    
    # Normalization statistics
    pco2_mean = stats['pco2']['mean']
    pco2_std = stats['pco2']['std']
    sst_mean = stats['sst']['mean']
    sst_std = stats['sst']['std']
    dic_mean = stats['dic']['mean']
    dic_std = stats['dic']['std']
    alk_mean = stats['alk']['mean']
    alk_std = stats['alk']['std']
    beta_mean = stats['beta']['mean']  # v2: additional parameter
    beta_std = stats['beta']['std']    # v2: additional parameter
    
    # Collect predictions (normalized and denormalized)
    all_pco2_pred_norm = []
    all_pco2_gt_norm = []
    all_pco2_pred_denorm = []
    all_pco2_gt_denorm = []
    csv_data = []
    
    if logger:
        logger.info("Collecting predictions (v2 evaluation)...")
    
    with torch.no_grad():
        for batch_idx, (X, Y_target, Y_delta_sst, Y_delta_dic, Y_beta, mask) in enumerate(dataloader):
            # v2: unpack 6-element batch
            X = X.to(device)
            Y_target = Y_target.to(device)
            Y_delta_sst = Y_delta_sst.to(device)
            Y_delta_dic = Y_delta_dic.to(device)
            Y_beta = Y_beta.to(device)
            mask = mask.to(device)
            
            # Forward pass
            physics_params, pCO2_phy, pCO2_final = model(X, mask)
            # physics_params: (B, 3, H, W, T_out)
            # pCO2_phy: (B, 1, H, W, T_out)
            # pCO2_final: (B, 1, H, W, T_out)
            
            # 移动到CPU并处理时间维度
            # 取第一个时间步（T_out当前=1）
            pCO2_final_norm = pCO2_final[:, :, :, :, 0].cpu().numpy()  # (B, 1, H, W)
            Y_target_norm = Y_target[:, :, :, :, 0].cpu().numpy()      # (B, 1, H, W)
            
            # Physical parameters (for CSV export)
            delta_sst_pred_norm = physics_params[:, 0:1, :, :, 0].cpu().numpy()  # (B, 1, H, W)
            delta_dic_pred_norm = physics_params[:, 1:2, :, :, 0].cpu().numpy()
            beta_pred_norm = physics_params[:, 2:3, :, :, 0].cpu().numpy()
            
            # Ground-truth physical parameters
            Y_delta_sst_norm = Y_delta_sst[:, :, :, :, 0].cpu().numpy()
            Y_delta_dic_norm = Y_delta_dic[:, :, :, :, 0].cpu().numpy()
            Y_beta_norm = Y_beta[:, :, :, :, 0].cpu().numpy()
            
            # Denormalize pCO2
            pCO2_final_denorm = denormalize_data(pCO2_final_norm, pco2_mean, pco2_std)
            Y_target_denorm = denormalize_data(Y_target_norm, pco2_mean, pco2_std)
            
            # Accumulate for metric computation
            all_pco2_pred_norm.append(pCO2_final_norm)
            all_pco2_gt_norm.append(Y_target_norm)
            all_pco2_pred_denorm.append(pCO2_final_denorm)
            all_pco2_gt_denorm.append(Y_target_denorm)
            
            # === Visualize first sample of first batch ===
            if batch_idx == 0:
                sample_idx = 0
                pred_map = pCO2_final_denorm[sample_idx, 0]  # (H, W)
                gt_map = Y_target_denorm[sample_idx, 0]      # (H, W)
                
                visualizer.plot_comparison(
                    pred=pred_map,
                    gt=gt_map,
                    mask=mask_cpu,
                    title_prefix=f"Epoch {epoch}",
                    save_name=f"comparison_sample_{sample_idx}.png"
                )
            
            # === Save CSV data (first batch only) ===
            if batch_idx == 0:
                batch_size = pCO2_final_norm.shape[0]
                
                for b in range(batch_size):
                    # Valid region on CPU mask
                    valid_mask = mask_cpu > 0.5
                    
            # pCO2 (normalized and real)
                    pco2_pred_norm = pCO2_final_norm[b, 0][valid_mask]
                    pco2_gt_norm = Y_target_norm[b, 0][valid_mask]
                    pco2_pred_real = pCO2_final_denorm[b, 0][valid_mask]
                    pco2_gt_real = Y_target_denorm[b, 0][valid_mask]
                    
                    # v2 physical parameters (predictions, normalized and real)
                    delta_sst_pred_norm_val = delta_sst_pred_norm[b, 0][valid_mask]
                    delta_dic_pred_norm_val = delta_dic_pred_norm[b, 0][valid_mask]
                    beta_pred_norm_val = beta_pred_norm[b, 0][valid_mask]
                    
                    delta_sst_pred_real = denormalize_data(delta_sst_pred_norm_val, sst_mean, sst_std)
                    delta_dic_pred_real = denormalize_data(delta_dic_pred_norm_val, dic_mean, dic_std)
                    beta_pred_real = denormalize_data(beta_pred_norm_val, beta_mean, beta_std)
                    
                    # v2 physical parameters (targets, normalized and real)
                    delta_sst_gt_norm_val = Y_delta_sst_norm[b, 0][valid_mask]
                    delta_dic_gt_norm_val = Y_delta_dic_norm[b, 0][valid_mask]
                    beta_gt_norm_val = Y_beta_norm[b, 0][valid_mask]
                    
                    delta_sst_gt_real = denormalize_data(delta_sst_gt_norm_val, sst_mean, sst_std)
                    delta_dic_gt_real = denormalize_data(delta_dic_gt_norm_val, dic_mean, dic_std)
                    beta_gt_real = denormalize_data(beta_gt_norm_val, beta_mean, beta_std)
                    
                    # Build coordinate arrays
                    H, W = mask_cpu.shape
                    lat_coords, lon_coords = np.meshgrid(
                        np.linspace(90, -90, H),
                        np.linspace(-180, 180, W),
                        indexing='ij'
                    )
                    
                    lats = lat_coords[valid_mask]
                    lons = lon_coords[valid_mask]
                    
                    # v2: assemble extended CSV row (with physical parameters)
                    for i in range(len(pco2_pred_real)):
                        csv_data.append({
                            'sample_id': b,
                            'latitude': lats[i],
                            'longitude': lons[i],
                            
                            # pCO2 - 真实空间
                            'pCO2_pred_real': pco2_pred_real[i],
                            'pCO2_gt_real': pco2_gt_real[i],
                            'pCO2_error_real': pco2_pred_real[i] - pco2_gt_real[i],
                            
                            # pCO2 - 归一化空间
                            'pCO2_pred_norm': pco2_pred_norm[i],
                            'pCO2_gt_norm': pco2_gt_norm[i],
                            'pCO2_error_norm': pco2_pred_norm[i] - pco2_gt_norm[i],
                            
                            # v2新增：物理参数 - 真实空间
                            'delta_SST_pred_real': delta_sst_pred_real[i],
                            'delta_SST_gt_real': delta_sst_gt_real[i],
                            'delta_SST_error_real': delta_sst_pred_real[i] - delta_sst_gt_real[i],
                            
                            'delta_DIC_pred_real': delta_dic_pred_real[i],
                            'delta_DIC_gt_real': delta_dic_gt_real[i],
                            'delta_DIC_error_real': delta_dic_pred_real[i] - delta_dic_gt_real[i],
                            
                            'beta_pred_real': beta_pred_real[i],
                            'beta_gt_real': beta_gt_real[i],
                            'beta_error_real': beta_pred_real[i] - beta_gt_real[i],
                            
                            # v2新增：物理参数 - 归一化空间
                            'delta_SST_pred_norm': delta_sst_pred_norm_val[i],
                            'delta_SST_gt_norm': delta_sst_gt_norm_val[i],
                            'delta_SST_error_norm': delta_sst_pred_norm_val[i] - delta_sst_gt_norm_val[i],
                            
                            'delta_DIC_pred_norm': delta_dic_pred_norm_val[i],
                            'delta_DIC_gt_norm': delta_dic_gt_norm_val[i],
                            'delta_DIC_error_norm': delta_dic_pred_norm_val[i] - delta_dic_gt_norm_val[i],
                            
                            'beta_pred_norm': beta_pred_norm_val[i],
                            'beta_gt_norm': beta_gt_norm_val[i],
                            'beta_error_norm': beta_pred_norm_val[i] - beta_gt_norm_val[i],
                        })
    
    # === Compute metrics in normalized and real spaces ===
    if logger:
        logger.info("Computing metrics in normalized and real spaces...")
    
    # Concatenate batch lists
    all_pco2_pred_norm = np.concatenate(all_pco2_pred_norm, axis=0)
    all_pco2_gt_norm = np.concatenate(all_pco2_gt_norm, axis=0)
    all_pco2_pred_denorm = np.concatenate(all_pco2_pred_denorm, axis=0)
    all_pco2_gt_denorm = np.concatenate(all_pco2_gt_denorm, axis=0)
    
    # Compute dual-space metrics
    norm_metrics, real_metrics = compute_dual_space_metrics(
        all_pco2_pred_norm, all_pco2_gt_norm,
        all_pco2_pred_denorm, all_pco2_gt_denorm,
        mask_cpu
    )
    
    # Merge into a single dict
    combined_metrics = {**norm_metrics, **real_metrics}
    
    # === Save CSV ===
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = csv_dir / "predictions_batch0.csv"
        df.to_csv(csv_path, index=False)
        
        if logger:
            logger.info(f"✓ CSV saved to: {csv_path}")
            logger.info(f"  - rows: {len(df)}")
            logger.info(f"  - columns: {len(df.columns)} (v2 extended: includes physical parameters)")
    
    # === Save metrics JSON ===
    metrics_file = save_metrics_json(metrics_dir, combined_metrics, epoch)
    
    if logger:
        logger.info(f"✓ Visualizations saved under: {vis_dir}")
        logger.info(f"✓ Metrics saved to: {metrics_file}")
        
        # Pretty-print metrics comparison
        display_text = format_dual_metrics_for_display(
            norm_metrics, real_metrics, unit="µatm"
        )
        logger.info(display_text)
    
    return combined_metrics
    
if __name__ == "__main__":
    print("evaluation module (v2) defined; run in training environment for full test.")