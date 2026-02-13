# Visualization utilities: draw global ocean maps and training curves
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

class Visualizer:
    """
    Visualization helper for AD-PINI.
    
    Features:
    1. Plot global ocean pCO2 maps
    2. Plot training curves
    3. Plot prediction vs ground-truth comparison maps
    """
    
    def __init__(self, save_dir="./outputs/figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_global_map(self, data, mask, title="Global Ocean pCO2", 
                       vmin=None, vmax=None, cmap='RdYlBu_r', 
                       save_name=None):
        """
        Plot a global ocean map.
        
        Args:
            data: (H, W) - 2D data array
            mask: (H, W) - valid-region mask
            title: plot title
            vmin, vmax: color scale bounds
            cmap: colormap name
            save_name: optional filename to save
        """
        H, W = data.shape
        
        # Apply mask: set land to NaN
        data_masked = np.where(mask > 0.5, data, np.nan)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot; origin='upper' ensures proper orientation
        im = ax.imshow(data_masked, 
                      cmap=cmap, 
                      vmin=vmin, 
                      vmax=vmax,
                      aspect='auto',
                      origin='upper',  # prevent upside-down map
                      extent=[-180, 180, -90, 90])  # lon/lat extent
        
        # Axes labels
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('pCO2 [µatm]', fontsize=12)
        
        plt.tight_layout()
        
        # Save to disk if requested
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    def plot_comparison(self, pred, gt, mask, title_prefix="", save_name=None):
        """
        Plot prediction vs ground truth and error maps.
        
        Args:
            pred: (H, W) - prediction values
            gt: (H, W) - ground-truth values
            mask: (H, W) - valid-region mask
            title_prefix: optional title prefix
            save_name: optional filename to save
        """
        # Apply mask
        pred_masked = np.where(mask > 0.5, pred, np.nan)
        gt_masked = np.where(mask > 0.5, gt, np.nan)
        
        # Error field
        error = pred - gt
        error_masked = np.where(mask > 0.5, error, np.nan)
        
        # Shared color scale for pred/gt
        vmin = np.nanmin(gt_masked)
        vmax = np.nanmax(gt_masked)
        
        # Subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Prediction
        im1 = axes[0].imshow(pred_masked, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, 
                            origin='upper', extent=[-180, 180, -90, 90])
        axes[0].set_title(f'{title_prefix} Prediction', fontsize=12)
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # Ground truth
        im2 = axes[1].imshow(gt_masked, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, 
                            origin='upper', extent=[-180, 180, -90, 90])
        axes[1].set_title(f'{title_prefix} Ground Truth', fontsize=12)
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # Error
        im3 = axes[2].imshow(error_masked, cmap='seismic', vmin=-50, vmax=50, 
                            origin='upper', extent=[-180, 180, -90, 90])
        axes[2].set_title(f'{title_prefix} Error (Pred - GT)', fontsize=12)
        axes[2].set_xlabel('Longitude')
        axes[2].set_ylabel('Latitude')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        
        # Save to disk if requested
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison figure saved to: {save_path}")
        
        return fig
    
    def plot_training_curves(self, train_losses, val_losses, save_name="training_curves.png"):
        """
        Plot training and validation loss curves.
        
        Args:
            train_losses: list of training losses
            val_losses: list of validation losses
            save_name: output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
        
        return fig

if __name__ == "__main__":
    # Simple smoke test
    viz = Visualizer()
    
    # Synthetic data
    H, W = 713, 1440
    data = np.random.randn(H, W) * 20 + 380  # mock pCO2
    mask = np.random.rand(H, W) > 0.3        # mock mask
    
    # Plot map
    viz.plot_global_map(data, mask, title="Test Map", 
                       vmin=350, vmax=420, save_name="test_map.png")
    
    print("Visualization test finished.")