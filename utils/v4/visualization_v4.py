# AD-PINI v4 scientific visualization module - publication-quality figures
"""
Research-oriented visualization system focused on model evaluation and physical interpretation:
1. prediction_vs_truth.png - spatial comparison of prediction vs ground truth (2x2 layout)
2. residual_correction_analysis.png - residual correction network analysis  
3. global_error_map.png - global error distribution map
4. driver_dominance_map.png - dominance map of physical drivers
5. model_performance_summary.png - comprehensive model performance summary
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import torch
import xarray as xr
from typing import Dict, Optional, Tuple, List, Union
import warnings
import logging
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import matplotlib.ticker as mticker
from datetime import datetime
import os
from scipy.stats import pearsonr
from matplotlib.gridspec import GridSpec

# Suppress noisy matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)

class ScientificVisualizerV4:
    """Publication-quality visualizer for AD-PINI v4."""
    
    def __init__(self, config, dpi: int = 300, figsize: Tuple[float, float] = (12, 8)):
        """
        Initialize scientific visualizer.
        
        Args:
            config: configuration object
            dpi: figure DPI, 300+ recommended for publications
            figsize: base figure size
        """
        self.config = config
        self.dpi = dpi
        self.figsize = figsize
        
        # Set matplotlib parameters for publication quality
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',  # serif fonts common in publications
            'font.serif': ['Times New Roman', 'Computer Modern Roman'],
            'axes.linewidth': 1.0,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
            'text.usetex': False,  # set True if LaTeX is available and desired
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True
        })
        
        # Scientific colormaps
        self.setup_scientific_colormaps()
        
        logging.info("AD-PINI v4 scientific visualizer initialized (publication-ready).")
    
    def setup_scientific_colormaps(self):
        """Configure scientific colormaps suitable for publications."""
        # Error colormap - diverging (blue–white–red)
        error_colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#ffffff', 
                       '#fdbf6f', '#ff7f00', '#e31a1c', '#b10026']
        self.error_cmap = LinearSegmentedColormap.from_list('scientific_error', error_colors, N=256)
        
        # Dominance colormap - colorblind-friendly
        dominance_colors = ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
                           '#ffffff', '#fdbf6f', '#ff7f00', '#e08214', '#bd0026', '#67001f']
        self.dominance_cmap = LinearSegmentedColormap.from_list('dominance', dominance_colors, N=256)
        
        # Magnitude colormap - sequential
        magnitude_colors = ['#ffffff', '#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', 
                           '#df65b0', '#e7298a', '#ce1256', '#91003f']
        self.magnitude_cmap = LinearSegmentedColormap.from_list('magnitude', magnitude_colors, N=256)
        
        # Prediction quality colormap
        quality_colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
        self.quality_cmap = LinearSegmentedColormap.from_list('quality', quality_colors, N=256)
    
    def create_geo_projection(self, central_longitude: float = 0.0) -> ccrs.PlateCarree:
        """Create PlateCarree geographic projection."""
        return ccrs.PlateCarree(central_longitude=central_longitude)
    
    def add_publication_map_features(self, ax, detailed: bool = True):
        """Add coastline, borders, land, and gridlines with publication styling."""
        # Coastlines and borders
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='#333333', alpha=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='#666666', alpha=0.6)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 11}
        gl.ylabel_style = {'size': 11}
        gl.top_labels = False
        gl.right_labels = False
    
    def generate_scientific_visualizations(self, epoch: int, outputs: Dict, targets: Dict, 
                                         lat: np.ndarray, lon: np.ndarray, 
                                         metrics: Dict, save_dir: str,
                                         preprocessor=None) -> None:
        """
        Generate the full set of publication-quality figures.
        
        Args:
            epoch: training epoch index
            outputs: model output dictionary
            targets: target dictionary  
            lat, lon: latitude/longitude coordinate arrays
            metrics: evaluation metrics
            save_dir: directory for saving figures
            preprocessor: data preprocessor (optional)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Persist validation metrics in parallel
        self._save_validation_metrics(epoch, metrics, save_dir)
        
        # Work with the first sample in batch
        pco2_pred = outputs['pco2_final'][0]  # [H, W]
        pco2_target = targets['pco2_target'][0]  # [H, W] 
        delta_thermal = outputs['delta_thermal'][0]  # [H, W]
        delta_nonther = outputs['delta_nonther'][0]  # [H, W]
        
        # Optional residual correction data
        corrector_output = None
        physics_output = None
        if 'corrector_output' in outputs:
            corrector_output = outputs['corrector_output'][0]
        if 'physics_output' in outputs:
            physics_output = outputs['physics_output'][0]
        elif 'pco2_physics' in outputs:
            physics_output = outputs['pco2_physics'][0]
        
        # 1. Prediction vs truth comparison (2x2 layout)
        self.plot_prediction_vs_truth(pco2_pred, pco2_target, lat, lon, epoch, 
                                     metrics, save_dir)
        
        # 2. Residual correction network analysis
        if corrector_output is not None and physics_output is not None:
            self.plot_residual_correction_analysis(physics_output, corrector_output, 
                                                  pco2_pred, pco2_target, lat, lon, 
                                                  epoch, save_dir)
        
        # 3. Global error map
        self.plot_global_error_map(pco2_pred, pco2_target, lat, lon, epoch, 
                                  metrics, save_dir)
        
        # 4. Physical driver dominance map
        self.plot_driver_dominance_map(delta_thermal, delta_nonther, lat, lon, 
                                      epoch, save_dir)
        
        # 5. Comprehensive performance summary
        self.plot_model_performance_summary(pco2_pred, pco2_target, delta_thermal,
                                           delta_nonther, lat, metrics, epoch, save_dir)
    
    def plot_prediction_vs_truth(self, pco2_pred: np.ndarray, pco2_target: np.ndarray,
                                lat: np.ndarray, lon: np.ndarray, epoch: int,
                                metrics: Dict, save_dir: str) -> None:
        """1. Prediction vs truth spatial comparison (2x2 subplot layout)."""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Global color scale across prediction and target
        vmin = np.nanmin([np.nanmin(pco2_pred), np.nanmin(pco2_target)])
        vmax = np.nanmax([np.nanmax(pco2_pred), np.nanmax(pco2_target)])
        
        LON, LAT = np.meshgrid(lon, lat)
        
        # Panel (a): prediction
        ax1 = fig.add_subplot(gs[0, 0], projection=self.create_geo_projection())
        im1 = ax1.pcolormesh(LON, LAT, pco2_pred, transform=ccrs.PlateCarree(),
                            cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
        self.add_publication_map_features(ax1)
        ax1.set_global()
        ax1.set_title('(a) Model Prediction', fontsize=14, fontweight='bold')
        
        # Panel (b): ground truth
        ax2 = fig.add_subplot(gs[0, 1], projection=self.create_geo_projection())
        im2 = ax2.pcolormesh(LON, LAT, pco2_target, transform=ccrs.PlateCarree(),
                            cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')
        self.add_publication_map_features(ax2)
        ax2.set_global()
        ax2.set_title('(b) Observations (Ground Truth)', fontsize=14, fontweight='bold')
        
        # Panel (c): prediction error
        error = pco2_pred - pco2_target
        error_max = np.nanpercentile(np.abs(error), 95)
        ax3 = fig.add_subplot(gs[1, 0], projection=self.create_geo_projection())
        im3 = ax3.pcolormesh(LON, LAT, error, transform=ccrs.PlateCarree(),
                            cmap=self.error_cmap, vmin=-error_max, vmax=error_max, shading='auto')
        self.add_publication_map_features(ax3)
        ax3.set_global()
        ax3.set_title('(c) Prediction Error (Pred - Obs)', fontsize=14, fontweight='bold')
        
        # Panel (d): scatter comparison
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Random subsampling for scatter to avoid overplotting
        valid_mask = np.isfinite(pco2_pred) & np.isfinite(pco2_target)
        if np.sum(valid_mask) > 10000:  # randomly subsample if too many points
            indices = np.where(valid_mask)
            sample_idx = np.random.choice(len(indices[0]), 10000, replace=False)
            pred_sample = pco2_pred[indices[0][sample_idx], indices[1][sample_idx]]
            true_sample = pco2_target[indices[0][sample_idx], indices[1][sample_idx]]
        else:
            pred_sample = pco2_pred[valid_mask]
            true_sample = pco2_target[valid_mask]
        
        # Scatter plot
        ax4.scatter(true_sample, pred_sample, alpha=0.3, s=1, color='navy')
        
        # 1:1 line
        lims = [np.min([ax4.get_xlim(), ax4.get_ylim()]),
                np.max([ax4.get_xlim(), ax4.get_ylim()])]
        ax4.plot(lims, lims, 'r--', alpha=0.8, linewidth=2, label='1:1 line')
        
        ax4.set_xlabel('Observed pCO₂ (μatm)', fontsize=12)
        ax4.set_ylabel('Predicted pCO₂ (μatm)', fontsize=12)
        ax4.set_title('(d) Model Performance', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Annotate with key statistics
        rmse = metrics.get('rmse', np.sqrt(np.nanmean(error**2)))
        r2 = metrics.get('r2', 0.0)
        correlation = metrics.get('correlation', 0.0)
        
        stats_text = f'RMSE: {rmse:.2f} μatm\\nR²: {r2:.3f}\\nCorr: {correlation:.3f}'
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Shared colorbars
        cbar1 = fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', 
                            pad=0.05, shrink=0.8, aspect=30)
        cbar1.set_label('pCO₂ (μatm)', fontsize=12)
        
        cbar3 = fig.colorbar(im3, ax=ax3, orientation='horizontal', 
                            pad=0.05, shrink=0.8, aspect=20)
        cbar3.set_label('Prediction Error (μatm)', fontsize=12)
        
        # Global title
        fig.suptitle(f'Model Prediction vs Ground Truth Comparison (Epoch {epoch})', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.savefig(os.path.join(save_dir, 'prediction_vs_truth.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
    
    def plot_residual_correction_analysis(self, physics_output: np.ndarray, 
                                        corrector_output: np.ndarray,
                                        final_pred: np.ndarray, truth: np.ndarray,
                                        lat: np.ndarray, lon: np.ndarray, 
                                        epoch: int, save_dir: str) -> None:
        """2. Residual correction network analysis."""
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)
        
        LON, LAT = np.meshgrid(lon, lat)
        
        # Panel (a): physics-based output
        ax1 = fig.add_subplot(gs[0, 0], projection=self.create_geo_projection())
        physics_range = np.nanpercentile(np.abs(physics_output), [2, 98])
        im1 = ax1.pcolormesh(LON, LAT, physics_output, transform=ccrs.PlateCarree(),
                            cmap='RdYlBu_r', vmin=physics_range[0], vmax=physics_range[1], 
                            shading='auto')
        self.add_publication_map_features(ax1)
        ax1.set_global()
        ax1.set_title('(a) Physics Model Output', fontsize=14, fontweight='bold')
        
        # Panel (b): residual correction field
        ax2 = fig.add_subplot(gs[0, 1], projection=self.create_geo_projection())
        corrector_range = np.nanpercentile(np.abs(corrector_output), [2, 98])
        im2 = ax2.pcolormesh(LON, LAT, corrector_output, transform=ccrs.PlateCarree(),
                            cmap=self.error_cmap, vmin=-corrector_range[1], vmax=corrector_range[1], 
                            shading='auto')
        self.add_publication_map_features(ax2)
        ax2.set_global()
        ax2.set_title('(b) Residual Correction', fontsize=14, fontweight='bold')
        
        # Panel (c): final prediction
        ax3 = fig.add_subplot(gs[0, 2], projection=self.create_geo_projection())
        final_range = np.nanpercentile([np.nanmin(final_pred), np.nanmax(final_pred)], [2, 98])
        im3 = ax3.pcolormesh(LON, LAT, final_pred, transform=ccrs.PlateCarree(),
                            cmap='RdYlBu_r', vmin=final_range[0], vmax=final_range[1], 
                            shading='auto')
        self.add_publication_map_features(ax3)
        ax3.set_global()
        ax3.set_title('(c) Final Prediction', fontsize=14, fontweight='bold')
        
        # Panel (d): distribution of correction magnitudes
        ax4 = fig.add_subplot(gs[1, 0])
        correction_magnitude = np.abs(corrector_output)
        ax4.hist(correction_magnitude[np.isfinite(correction_magnitude)].flatten(), 
                bins=50, alpha=0.7, color='steelblue', density=True)
        ax4.set_xlabel('Correction Magnitude (|ε|)', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.set_title('(d) Correction Magnitude Distribution', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Panel (e): physics vs final error comparison
        ax5 = fig.add_subplot(gs[1, 1])
        physics_error = np.abs(physics_output - truth)
        final_error = np.abs(final_pred - truth)
        
        # Random subsampling
        valid_mask = (np.isfinite(physics_error) & np.isfinite(final_error))
        if np.sum(valid_mask) > 5000:
            indices = np.where(valid_mask)
            sample_idx = np.random.choice(len(indices[0]), 5000, replace=False)
            phys_err_sample = physics_error[indices[0][sample_idx], indices[1][sample_idx]]
            final_err_sample = final_error[indices[0][sample_idx], indices[1][sample_idx]]
        else:
            phys_err_sample = physics_error[valid_mask]
            final_err_sample = final_error[valid_mask]
        
        ax5.scatter(phys_err_sample, final_err_sample, alpha=0.3, s=1, color='green')
        ax5.plot([0, np.max(phys_err_sample)], [0, np.max(phys_err_sample)], 
                'r--', alpha=0.8, linewidth=2, label='No improvement line')
        ax5.set_xlabel('Physics Model Error', fontsize=12)
        ax5.set_ylabel('Final Model Error', fontsize=12)
        ax5.set_title('(e) Error Reduction Analysis', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Panel (f): summary of correction effectiveness
        ax6 = fig.add_subplot(gs[1, 2])
        
        improvement_ratio = (physics_error - final_error) / (physics_error + 1e-8)
        improvement_valid = improvement_ratio[np.isfinite(improvement_ratio)]
        
        # Quantify regions improved, degraded, unchanged
        improved_mask = improvement_valid > 0
        degraded_mask = improvement_valid < 0
        
        categories = ['Improved', 'Degraded', 'No Change']
        counts = [np.sum(improved_mask), np.sum(degraded_mask), 
                 np.sum(np.abs(improvement_valid) < 0.01)]
        colors = ['green', 'red', 'gray']
        
        ax6.bar(categories, counts, color=colors, alpha=0.7)
        ax6.set_ylabel('Number of Grid Points', fontsize=12)
        ax6.set_title('(f) Correction Effect Summary', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Percentage labels
        total = sum(counts)
        for i, count in enumerate(counts):
            ax6.text(i, count + total*0.01, f'{count/total*100:.1f}%', 
                    ha='center', fontsize=11, fontweight='bold')
        
        # Colorbars for the three maps
        cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.6, aspect=15)
        cbar1.set_label('pCO₂ (μatm)', fontsize=10)
        
        cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.6, aspect=15)
        cbar2.set_label('Correction (μatm)', fontsize=10)
        
        cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.6, aspect=15)
        cbar3.set_label('pCO₂ (μatm)', fontsize=10)
        
        # Global title
        fig.suptitle(f'Residual Correction Network Analysis (Epoch {epoch})', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.savefig(os.path.join(save_dir, 'residual_correction_analysis.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
    
    def plot_global_error_map(self, pco2_pred: np.ndarray, pco2_target: np.ndarray,
                             lat: np.ndarray, lon: np.ndarray, epoch: int,
                             metrics: Dict, save_dir: str) -> None:
        """3. Global error map showing spatial distribution of prediction error."""
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, projection=self.create_geo_projection())
        
        # Error field
        error = pco2_pred - pco2_target
        valid_mask = np.isfinite(error)
        error_masked = np.where(valid_mask, error, np.nan)
        
        # Reasonable error range (robust to outliers)
        vmax = np.nanpercentile(np.abs(error_masked), 95)
        vmin = -vmax
        
        # Draw error map
        LON, LAT = np.meshgrid(lon, lat)
        im = ax.pcolormesh(LON, LAT, error_masked, transform=ccrs.PlateCarree(),
                          cmap=self.error_cmap, vmin=vmin, vmax=vmax, shading='auto')
        
        # Map features and colorbar
        self.add_publication_map_features(ax)
        ax.set_global()
        
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, aspect=40)
        cbar.set_label('Prediction Error (μatm)', fontsize=14, fontweight='bold')
        
        # Summary statistics
        rmse = metrics.get('rmse', np.sqrt(np.nanmean(error**2)))
        mae = metrics.get('mae', np.nanmean(np.abs(error)))
        r2 = metrics.get('r2', 0.0)
        bias = np.nanmean(error)
        
        # Text box with key statistics
        stats_text = (f'RMSE: {rmse:.2f} μatm\\n'
                     f'MAE: {mae:.2f} μatm\\n'
                     f'Bias: {bias:.2f} μatm\\n'
                     f'R²: {r2:.3f}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
        # Title
        ax.set_title(f'Global Prediction Error Distribution (Epoch {epoch})', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.savefig(os.path.join(save_dir, 'global_error_map.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
    
    def plot_driver_dominance_map(self, delta_thermal: np.ndarray, delta_nonther: np.ndarray,
                                 lat: np.ndarray, lon: np.ndarray, epoch: int, 
                                 save_dir: str) -> None:
        """4. Physical driver dominance map for validating underlying mechanisms."""
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, projection=self.create_geo_projection())
        
        # Dominance ratio
        total_contrib = np.abs(delta_thermal) + np.abs(delta_nonther)
        dominance_ratio = np.where(total_contrib > 1e-10, 
                                  np.abs(delta_thermal) / total_contrib, 
                                  0.5)
        
        # Dominance map
        LON, LAT = np.meshgrid(lon, lat)
        im = ax.pcolormesh(LON, LAT, dominance_ratio, transform=ccrs.PlateCarree(),
                          cmap=self.dominance_cmap, vmin=0, vmax=1, shading='auto')
        
        # Map features
        self.add_publication_map_features(ax)
        ax.set_global()
        
        ax.plot([-180, 180], [23.5, 23.5], color='orange', linestyle='-', alpha=0.8, 
                transform=ccrs.PlateCarree(), linewidth=3, label='Tropics')
        ax.plot([-180, 180], [-23.5, -23.5], color='orange', linestyle='-', alpha=0.8, 
                transform=ccrs.PlateCarree(), linewidth=3)
        
        # Annotate key latitude bands
        ax.text(0, 0, 'Tropical Zone\\n(Expected: Thermal)', 
               transform=ccrs.PlateCarree(), ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        ax.text(0, 50, 'High Latitudes\\n(Expected: Non-thermal)', 
               transform=ccrs.PlateCarree(), ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, aspect=40)
        cbar.set_label('Thermal Dominance Ratio (0=Non-thermal, 1=Thermal)', fontsize=14, fontweight='bold')
        
        # Title
        ax.set_title(f'Physical Driver Dominance Map (Epoch {epoch})', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.savefig(os.path.join(save_dir, 'driver_dominance_map.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
    
    def plot_model_performance_summary(self, pco2_pred: np.ndarray, pco2_target: np.ndarray,
                                     delta_thermal: np.ndarray, delta_nonther: np.ndarray,
                                     lat: np.ndarray, metrics: Dict, epoch: int, save_dir: str) -> None:
        """5. Comprehensive model performance summary figure."""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Basic statistics and sampling
        error = pco2_pred - pco2_target
        valid_mask = np.isfinite(error) & np.isfinite(pco2_pred) & np.isfinite(pco2_target)
        
        if np.sum(valid_mask) > 5000:  # subsample for scatter plot
            indices = np.where(valid_mask)
            sample_idx = np.random.choice(len(indices[0]), 5000, replace=False)
            pred_sample = pco2_pred[indices[0][sample_idx], indices[1][sample_idx]]
            true_sample = pco2_target[indices[0][sample_idx], indices[1][sample_idx]]
            error_sample = error[indices[0][sample_idx], indices[1][sample_idx]]
        else:
            pred_sample = pco2_pred[valid_mask]
            true_sample = pco2_target[valid_mask]
            error_sample = error[valid_mask]
        
        # Panel (a): error distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(error_sample, bins=50, alpha=0.7, color='steelblue', density=True)
        ax1.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Perfect prediction')
        ax1.set_xlabel('Prediction Error (μatm)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('(a) Error Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel (b): residual vs predicted
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(pred_sample, error_sample, alpha=0.3, s=1, color='green')
        ax2.axhline(0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax2.set_xlabel('Predicted pCO₂ (μatm)', fontsize=12)
        ax2.set_ylabel('Prediction Error (μatm)', fontsize=12)
        ax2.set_title('(b) Residual vs Predicted', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Panel (c): Q-Q plot for residual normality
        ax3 = fig.add_subplot(gs[0, 2])
        from scipy import stats
        stats.probplot(error_sample, dist="norm", plot=ax3)
        ax3.set_title('(c) Error Normality (Q-Q Plot)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Panel (d): performance metrics summary
        ax4 = fig.add_subplot(gs[1, 0])
        metrics_names = ['RMSE', 'MAE', 'R²', 'Correlation']
        rmse = metrics.get('rmse', np.sqrt(np.nanmean(error**2)))
        mae = metrics.get('mae', np.nanmean(np.abs(error)))
        r2 = metrics.get('r2', 0.0)
        corr = metrics.get('correlation', pearsonr(pred_sample, true_sample)[0])
        
        metrics_values = [rmse, mae, r2, corr]
        colors = ['red' if v < 0.5 else 'orange' if v < 0.8 else 'green' for v in metrics_values]
        # Custom thresholds for RMSE and MAE color-coding
        colors[0] = 'red' if rmse > 20 else 'orange' if rmse > 10 else 'green'
        colors[1] = 'red' if mae > 15 else 'orange' if mae > 8 else 'green'
        
        bars = ax4.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax4.set_ylabel('Metric Value', fontsize=12)
        ax4.set_title('(d) Performance Metrics', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Metric labels on bars
        for bar, value in zip(bars, metrics_values):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Panel (e): relative thermal vs non-thermal contributions
        ax5 = fig.add_subplot(gs[1, 1])
        thermal_mean = np.nanmean(np.abs(delta_thermal))
        nonther_mean = np.nanmean(np.abs(delta_nonther))
        
        contributions = [thermal_mean, nonther_mean]
        labels = ['Thermal', 'Non-thermal']
        colors = ['red', 'blue']
        
        wedges, texts, autotexts = ax5.pie(contributions, labels=labels, colors=colors, autopct='%1.1f%%')
        for wedge in wedges:
            wedge.set_alpha(0.7)
        ax5.set_title('(e) Average Contribution', fontsize=14, fontweight='bold')
        
        # Panel (f): regional performance
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Correlations by latitude band
        regions = {
            'Tropics': (lat >= -23.5) & (lat <= 23.5),
            'North': lat > 23.5,
            'South': lat < -23.5
        }
        
        region_corrs = []
        region_names = []
        
        for region_name, lat_mask in regions.items():
            region_pred = pco2_pred[lat_mask, :]
            region_true = pco2_target[lat_mask, :]
            region_valid = np.isfinite(region_pred) & np.isfinite(region_true)
            
            if np.sum(region_valid) > 100:
                corr, _ = pearsonr(region_pred[region_valid], region_true[region_valid])
                region_corrs.append(corr)
                region_names.append(region_name)
        
        ax6.bar(region_names, region_corrs, color=['orange', 'lightblue', 'lightgreen'], alpha=0.7)
        ax6.set_ylabel('Correlation', fontsize=12)
        ax6.set_title('(f) Regional Performance', fontsize=14, fontweight='bold')
        ax6.set_ylim([0, 1])
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Global title
        fig.suptitle(f'Model Performance Comprehensive Summary (Epoch {epoch})', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.savefig(os.path.join(save_dir, 'model_performance_summary.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
    
    def _save_validation_metrics(self, epoch: int, metrics: Dict, save_dir: str) -> None:
        """
        Save validation metrics to JSON files (detailed and summary).
        
        Args:
            epoch: epoch index
            metrics: metrics dictionary
            save_dir: directory to write JSON files into
        """
        import json
        from datetime import datetime
        
        # Detailed metrics record
        detailed_metrics = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'validation_metrics': {
                # Core performance metrics
                'mse': metrics.get('mse', 0.0),
                'mae': metrics.get('mae', 0.0), 
                'rmse': metrics.get('rmse', 0.0),
                'r2': metrics.get('r2', 0.0),
                'correlation': metrics.get('correlation', 0.0),
                'crps': metrics.get('crps', 0.0),
                
                # Physical decomposition quality
                'delta_sst_mse': metrics.get('delta_sst_mse', 0.0),
                'delta_dic_mse': metrics.get('delta_dic_mse', 0.0),
                
                # Correction network behavior
                'correction_magnitude': metrics.get('correction_magnitude', 0.0),
            },
            'performance_summary': {
                'prediction_quality': 'Excellent' if metrics.get('r2', 0) > 0.9 else 
                                    ('Good' if metrics.get('r2', 0) > 0.8 else 
                                     ('Fair' if metrics.get('r2', 0) > 0.6 else 'Poor')),
                'rmse_normalized': metrics.get('rmse', 0.0) / (metrics.get('mae', 1.0) + 1e-8),
                'crps_performance': 'Low Error' if metrics.get('crps', 100) < 5.0 else 
                                  ('Medium Error' if metrics.get('crps', 100) < 15.0 else 'High Error')
            }
        }
        
        # Save detailed record
        metrics_file = os.path.join(save_dir, f'validation_metrics_epoch_{epoch}.json')
        with open(metrics_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2, ensure_ascii=False)
        
        # Also save a compact summary for quick inspection
        summary_file = os.path.join(save_dir, 'latest_metrics_summary.json')
        summary = {
            'epoch': epoch,
            'mse': metrics.get('mse', 0.0),
            'mae': metrics.get('mae', 0.0),
            'r2': metrics.get('r2', 0.0),
            'crps': metrics.get('crps', 0.0),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Validation metrics saved to: {metrics_file}")
        logging.info(
            f"Key metrics - MSE: {metrics.get('mse', 0.0):.4f}, "
            f"R²: {metrics.get('r2', 0.0):.3f}, CRPS: {metrics.get('crps', 0.0):.4f}"
        )