# AD-PINI v4 WandB dashboard utilities
# Provide custom charts and advanced visualization helpers

import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import torch

class WandbDashboard:
    """
    WandB dashboard helper for creating custom charts and professional summaries.
    """
    
    def __init__(self, wandb_logger):
        """
        Initialize dashboard.
        
        Args:
            wandb_logger: WandB logger instance
        """
        self.wandb_logger = wandb_logger
        self.enabled = wandb_logger.enabled if wandb_logger else False
    
    def create_loss_comparison_chart(self, 
                                   train_losses: List[float], 
                                   val_losses: List[float], 
                                   epoch: int):
        """Create training vs validation loss line chart."""
        if not self.enabled:
            return
            
        try:
            # Build loss comparison table
            data = []
            for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                data.append([i+1, train_loss, "Train"])
                data.append([i+1, val_loss, "Validation"])
            
            table = wandb.Table(data=data, columns=["Epoch", "Loss", "Split"])
            
            self.wandb_logger.run.log({
                "charts/loss_comparison": wandb.plot.line(
                    table, "Epoch", "Loss", groupby="Split",
                    title="Training vs Validation Loss"
                )
            }, step=epoch)
            
        except Exception as e:
            print(f"⚠️  Failed to create loss comparison chart: {e}")
    
    def create_lr_schedule_chart(self, lr_history: List[float], epoch: int):
        """Create learning-rate schedule chart."""
        if not self.enabled:
            return
            
        try:
            data = [[i+1, lr] for i, lr in enumerate(lr_history)]
            table = wandb.Table(data=data, columns=["Epoch", "Learning_Rate"])
            
            self.wandb_logger.run.log({
                "charts/learning_rate_schedule": wandb.plot.line(
                    table, "Epoch", "Learning_Rate",
                    title="Learning Rate Schedule"
                )
            }, step=epoch)
            
        except Exception as e:
            print(f"⚠️  Failed to create learning-rate chart: {e}")
    
    def create_uq_metrics_radar(self, uq_metrics: Dict[str, float], epoch: int):
        """Create radar-style overview for UQ metrics (implemented as bar chart)."""
        if not self.enabled or not uq_metrics:
            return
            
        try:
            # Select key UQ metrics
            key_metrics = ['PICP_95', 'MPIW_95', 'NLL', 'CRPS', 'UQ_Score']
            values = []
            labels = []
            
            for metric in key_metrics:
                if metric in uq_metrics:
                    values.append(uq_metrics[metric])
                    labels.append(metric)
            
            if len(values) >= 3:  # need at least 3 metrics
                # Build radar data
                radar_data = []
                for i, (label, value) in enumerate(zip(labels, values)):
                    radar_data.append([label, value])
                
                table = wandb.Table(data=radar_data, columns=["Metric", "Value"])
                
                self.wandb_logger.run.log({
                    "charts/uq_metrics_radar": wandb.plot.bar(
                        table, "Metric", "Value",
                        title="UQ Metrics Overview"
                    )
                }, step=epoch)
                
        except Exception as e:
            print(f"⚠️  Failed to create UQ radar chart: {e}")
    
    def create_loss_components_pie(self, loss_components: Dict[str, float], epoch: int, phase: str = "train"):
        """Create pie chart of loss components."""
        if not self.enabled:
            return
            
        try:
            # Filter out total_loss and keep positive components only
            components = {k: v for k, v in loss_components.items() 
                         if k != 'total_loss' and v > 0}
            
            if len(components) > 1:
                data = [[k, v] for k, v in components.items()]
                table = wandb.Table(data=data, columns=["Component", "Value"])
                
                self.wandb_logger.run.log({
                    f"charts/{phase}_loss_components": wandb.plot.pie(
                        table, values="Value", names="Component",
                        title=f"{phase.title()} Loss Components Distribution"
                    )
                }, step=epoch)
                
        except Exception as e:
            print(f"⚠️  Failed to create loss components pie chart: {e}")
    
    def create_model_performance_summary(self, 
                                       metrics_history: List[Dict[str, float]], 
                                       epoch: int):
        """Create model performance summary table."""
        if not self.enabled or not metrics_history:
            return
            
        try:
            # Compute performance trend between last two epochs
            if len(metrics_history) >= 2:
                latest = metrics_history[-1]
                previous = metrics_history[-2] if len(metrics_history) > 1 else latest
                
                performance_data = []
                for metric_name in latest.keys():
                    if metric_name in previous:
                        current = latest[metric_name]
                        prev = previous[metric_name]
                        change = current - prev
                        change_pct = (change / prev * 100) if prev != 0 else 0
                        
                        performance_data.append([
                            metric_name, 
                            f"{current:.4f}", 
                            f"{change:+.4f}", 
                            f"{change_pct:+.2f}%"
                        ])
                
                if performance_data:
                    table = wandb.Table(
                        data=performance_data, 
                        columns=["Metric", "Current", "Change", "Change %"]
                    )
                    
                    self.wandb_logger.run.log({
                        "tables/performance_summary": table
                    }, step=epoch)
                    
        except Exception as e:
            print(f"⚠️  Failed to create performance summary table: {e}")
    
    def log_custom_metrics_groups(self, epoch: int, **metric_groups):
        """Log arbitrary grouped metrics to WandB."""
        if not self.enabled:
            return
            
        try:
            for group_name, metrics in metric_groups.items():
                if isinstance(metrics, dict):
                    grouped_metrics = {f"{group_name}/{k}": v for k, v in metrics.items()}
                    self.wandb_logger.log_metrics(grouped_metrics, step=epoch)
                    
        except Exception as e:
            print(f"⚠️  Failed to log custom metric groups: {e}")
    
    def create_training_progress_heatmap(self, 
                                       loss_history: List[List[float]], 
                                       epoch: int,
                                       components: List[str]):
        """Create heatmap of recent loss component evolution."""
        if not self.enabled or not loss_history:
            return
            
        try:
            # Build heatmap data
            if len(loss_history) > 10:  # require some history
                # Use last few epochs
                recent_history = loss_history[-10:]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Convert to numpy array
                data_matrix = np.array(recent_history).T
                
                # Create heatmap
                sns.heatmap(data_matrix, 
                           xticklabels=range(len(recent_history)),
                           yticklabels=components,
                           annot=True, fmt='.3f', 
                           cmap='YlOrRd', ax=ax)
                
                ax.set_title('Recent Loss Components Evolution')
                ax.set_xlabel('Epoch (Recent)')
                ax.set_ylabel('Loss Components')
                
                # Log figure
                self.wandb_logger.run.log({
                    "charts/loss_heatmap": wandb.Image(fig)
                }, step=epoch)
                
                plt.close(fig)
                
        except Exception as e:
            print(f"⚠️  Failed to create training progress heatmap: {e}")

def setup_wandb_dashboard(wandb_logger) -> WandbDashboard:
    """
    Convenience helper to construct a WandbDashboard.
    
    Args:
        wandb_logger: WandB logger instance
        
    Returns:
        WandbDashboard instance
    """
    return WandbDashboard(wandb_logger)

# Predefined chart configurations
CHART_CONFIGS = {
    "loss_charts": {
        "train_val_comparison": True,
        "components_pie": True,
        "trend_heatmap": False,  # computationally expensive; disabled by default
    },
    "metrics_charts": {
        "uq_radar": True,
        "performance_table": True,
        "trend_analysis": True,
    },
    "system_charts": {
        "lr_schedule": True,
        "gradient_flow": False,  # computationally expensive; disabled by default
        "memory_usage": False,   # requires extra monitoring; disabled by default
    }
}

class WandbExperimentManager:
    """Experiment manager providing high-level tracking utilities on top of WandB."""
    
    def __init__(self, wandb_logger, dashboard):
        self.wandb_logger = wandb_logger
        self.dashboard = dashboard
        self.enabled = wandb_logger.enabled if wandb_logger else False
        
        # History buffers
        self.loss_history = []
        self.lr_history = []
        self.metrics_history = []
        
    def log_comprehensive_epoch(self, 
                              epoch: int, 
                              train_loss: float, 
                              val_loss: float,
                              train_components: Dict[str, float],
                              val_components: Dict[str, float],
                              uq_metrics: Dict[str, float] = None,
                              optimizer = None):
        """Comprehensively log all relevant information for a single epoch."""
        try:
            # Always update in-memory histories (even if WandB is disabled)
            self.loss_history.append([train_loss, val_loss])
            
            if optimizer:
                current_lr = optimizer.param_groups[0]['lr']
                self.lr_history.append(current_lr)
            
            if uq_metrics:
                self.metrics_history.append(uq_metrics.copy())
            
            # WandB-specific logging only if enabled
            if self.enabled:
                # Base loss logging
                self.wandb_logger.log_loss_components(
                    {'total_loss': train_loss, **train_components}, epoch, 'train'
                )
                self.wandb_logger.log_loss_components(
                    {'total_loss': val_loss, **val_components}, epoch, 'val'
                )
                
                if optimizer:
                    self.wandb_logger.log_learning_rate(optimizer, epoch)
            
            # Advanced charts (only if WandB enabled)
            if self.enabled and epoch % 5 == 0:  # update charts every 5 epochs
                # Loss comparison
                train_losses = [h[0] for h in self.loss_history]
                val_losses = [h[1] for h in self.loss_history]
                self.dashboard.create_loss_comparison_chart(train_losses, val_losses, epoch)
                
                # LR schedule
                if self.lr_history:
                    self.dashboard.create_lr_schedule_chart(self.lr_history, epoch)
                
                # Loss component pies
                self.dashboard.create_loss_components_pie(train_components, epoch, 'train')
                self.dashboard.create_loss_components_pie(val_components, epoch, 'val')
                
                # UQ metrics chart
                if uq_metrics:
                    self.dashboard.create_uq_metrics_radar(uq_metrics, epoch)
                
                # Performance summary table
                if self.metrics_history:
                    self.dashboard.create_model_performance_summary(self.metrics_history, epoch)
            
            # Custom metric groups (only if WandB enabled)
            if self.enabled:
                metric_groups = {
                    'loss': {'train': train_loss, 'val': val_loss, 'gap': val_loss - train_loss},
                    'components_train': train_components,
                    'components_val': val_components,
                }
                
                if uq_metrics:
                    metric_groups['uq'] = uq_metrics
                    
                self.dashboard.log_custom_metrics_groups(epoch, **metric_groups)
            
        except Exception as e:
            print(f"⚠️  Failed to log comprehensive epoch data: {e}")
    
    def generate_final_report(self):
        """Generate a final experiment report and summary charts."""
        if not self.enabled:
            return
            
        try:
            # Build final summary table and charts
            if self.loss_history:
                train_losses = [h[0] for h in self.loss_history]
                val_losses = [h[1] for h in self.loss_history]
                
                final_stats = {
                    'best_train_loss': min(train_losses),
                    'best_val_loss': min(val_losses),
                    'final_train_loss': train_losses[-1],
                    'final_val_loss': val_losses[-1],
                    'convergence_stability': np.std(val_losses[-10:]) if len(val_losses) >= 10 else 0,
                    'total_epochs_trained': len(self.loss_history)
                }
                
                self.wandb_logger.log_metrics(final_stats, prefix="final_report")
                
                # Final comparison chart
                self.dashboard.create_loss_comparison_chart(train_losses, val_losses, len(self.loss_history))
                
                print("✅ Final experiment report generated.")
                
        except Exception as e:
            print(f"⚠️  Failed to generate final report: {e}")