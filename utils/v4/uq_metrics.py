# AD-PINI v4 UQ evaluation metrics module
# High-cohesion design: centralizes UQ-specific evaluation metrics

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy import stats
from scipy.special import erf
import warnings

from configs.uq_config import uq_config

class UQMetrics:
    """Calculator for UQ evaluation metrics such as NLL, PICP, MPIW, ACE, and CRPS."""
    
    def __init__(self):
        self.confidence_levels = uq_config.confidence_levels
        self.eval_metrics = uq_config.eval_metrics
        
    def compute_all_metrics(self, 
                           pred_dist: Dict[str, torch.Tensor], 
                           targets: torch.Tensor,
                           mask: Optional[torch.Tensor] = None,
                           return_raw: bool = False) -> Dict[str, float]:
        """
        Compute all configured UQ metrics.
        
        Args:
            pred_dist: predictive distribution {'mu': Tensor, 'var': Tensor, 'std': Tensor}
            targets: ground truth [B, H, W] or [B, C, H, W]
            mask: valid-data mask (optional)
            return_raw: whether to return raw (unaggregated) values (currently unused)
            
        Returns:
            Dictionary of metric values.
        """
        # Convert to NumPy arrays and flatten
        pred_mu, pred_std, targets_flat = self._prepare_data(pred_dist, targets, mask)
        
        results = {}
        
        # Basic accuracy metrics (always computed)
        accuracy_metrics = self._compute_accuracy_metrics(pred_mu, targets_flat)
        results.update(accuracy_metrics)
        
        # UQ-specific metrics
        for metric in self.eval_metrics:
            if metric == "NLL":
                results['NLL'] = self._compute_nll(pred_mu, pred_std, targets_flat)
            elif metric == "PICP":
                for conf in self.confidence_levels:
                    picp = self._compute_picp(targets_flat, pred_mu, pred_std, conf)
                    results[f'PICP_{int(conf*100)}'] = picp
            elif metric == "MPIW":
                for conf in self.confidence_levels:
                    mpiw = self._compute_mpiw(pred_std, conf)
                    results[f'MPIW_{int(conf*100)}'] = mpiw
            elif metric == "ACE":
                ace = self._compute_ace(targets_flat, pred_mu, pred_std)
                results['ACE'] = ace
            elif metric == "CRPS":
                crps = self._compute_crps(targets_flat, pred_mu, pred_std)
                results['CRPS'] = crps
        
        # 计算UQ质量评分
        uq_score = self._compute_uq_quality_score(results)
        results['UQ_Score'] = uq_score
        
        return results
    
    def _prepare_data(self, 
                     pred_dist: Dict[str, torch.Tensor], 
                     targets: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare flattened NumPy arrays for metric computation."""
        pred_mu = pred_dist['mu'].detach().cpu().numpy()
        pred_std = pred_dist['std'].detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        # Apply mask
        if mask is not None:
            mask = mask.detach().cpu().numpy().astype(bool)
            pred_mu = pred_mu[mask]
            pred_std = pred_std[mask]
            targets = targets[mask]
        else:
            pred_mu = pred_mu.flatten()
            pred_std = pred_std.flatten()
            targets = targets.flatten()
        
        # Remove NaNs and invalid values
        valid_mask = (~np.isnan(pred_mu)) & (~np.isnan(pred_std)) & (~np.isnan(targets)) & (pred_std > 0)
        pred_mu = pred_mu[valid_mask]
        pred_std = pred_std[valid_mask]
        targets = targets[valid_mask]
        
        return pred_mu, pred_std, targets
    
    def _compute_accuracy_metrics(self, pred_mu: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Standard accuracy metrics (MSE, MAE, RMSE, R², correlation)."""
        if len(pred_mu) == 0:
            return {'MSE': float('nan'), 'MAE': float('nan'), 'RMSE': float('nan'), 
                   'R2': float('nan'), 'Correlation': float('nan')}
        
        # Core metrics
        mse = np.mean((pred_mu - targets) ** 2)
        mae = np.mean(np.abs(pred_mu - targets))
        rmse = np.sqrt(mse)
        
        # R² and correlation coefficient
        try:
            ss_res = np.sum((targets - pred_mu) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            
            correlation = np.corrcoef(pred_mu, targets)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except:
            r2 = float('nan')
            correlation = float('nan')
        
        return {
            'MSE': float(mse),
            'MAE': float(mae), 
            'RMSE': float(rmse),
            'R2': float(r2),
            'Correlation': float(correlation)
        }
    
    def _compute_nll(self, pred_mu: np.ndarray, pred_std: np.ndarray, targets: np.ndarray) -> float:
        """Negative Log Likelihood - evaluates quality of predictive distribution fit."""
        if len(pred_mu) == 0:
            return float('nan')
        
        # NLL = 0.5 * log(2π * σ²) + (y - μ)² / (2σ²)
        log_term = 0.5 * np.log(2 * np.pi * pred_std**2)
        square_term = (targets - pred_mu)**2 / (2 * pred_std**2)
        nll = log_term + square_term
        
        return float(np.mean(nll))
    
    def _compute_picp(self, 
                     targets: np.ndarray, 
                     pred_mu: np.ndarray, 
                     pred_std: np.ndarray, 
                     confidence: float) -> float:
        """
        Prediction Interval Coverage Probability
        Ideal value should be close to the nominal confidence (e.g. 0.95 for 95% interval).
        """
        if len(targets) == 0:
            return float('nan')
        
        z_score = stats.norm.ppf((1 + confidence) / 2)
        lower = pred_mu - z_score * pred_std
        upper = pred_mu + z_score * pred_std
        
        coverage = (targets >= lower) & (targets <= upper)
        picp = np.mean(coverage)
        
        return float(picp)
    
    def _compute_mpiw(self, pred_std: np.ndarray, confidence: float) -> float:
        """
        Mean Prediction Interval Width
        For a given PICP, smaller values indicate higher confidence (narrower intervals).
        """
        if len(pred_std) == 0:
            return float('nan')
        
        z_score = stats.norm.ppf((1 + confidence) / 2)
        width = 2 * z_score * pred_std
        mpiw = np.mean(width)
        
        return float(mpiw)
    
    def _compute_ace(self, 
                    targets: np.ndarray, 
                    pred_mu: np.ndarray, 
                    pred_std: np.ndarray,
                    n_bins: int = 10) -> float:
        """
        Average Calibration Error
        Evaluates calibration quality of predictive distribution; ideal value is 0.
        """
        if len(targets) == 0:
            return float('nan')
        
        try:
            # Standardized residuals
            std_residuals = (targets - pred_mu) / pred_std
            
            # Bin by confidence levels
            conf_levels = np.linspace(0.1, 0.9, n_bins)
            calibration_errors = []
            
            for conf in conf_levels:
                z_score = stats.norm.ppf((1 + conf) / 2)
                
                # Expected coverage
                expected_coverage = conf
                
                # Empirical coverage
                actual_coverage = np.mean(np.abs(std_residuals) <= z_score)
                
                # Calibration error
                calibration_error = np.abs(actual_coverage - expected_coverage)
                calibration_errors.append(calibration_error)
            
            ace = np.mean(calibration_errors)
            return float(ace)
            
        except Exception as e:
            logging.warning(f"Failed to compute ACE: {e}")
            return float('nan')
    
    def _compute_crps(self, 
                     targets: np.ndarray, 
                     pred_mu: np.ndarray, 
                     pred_std: np.ndarray) -> float:
        """
        Continuous Ranked Probability Score
        For Gaussian distributions, has a closed-form expression; smaller is better.
        """
        if len(targets) == 0:
            return float('nan')
        
        try:
            # Standardized target values
            std_targets = (targets - pred_mu) / pred_std
            
            # Closed-form CRPS for Gaussian:
            # CRPS = σ * [z * (2*Φ(z) - 1) + 2*φ(z) - 1/√π]
            # where z = (y - μ)/σ, Φ is CDF, φ is PDF
            
            phi_z = stats.norm.pdf(std_targets)  # PDF
            Phi_z = stats.norm.cdf(std_targets)  # CDF
            
            crps_normalized = std_targets * (2 * Phi_z - 1) + 2 * phi_z - 1/np.sqrt(np.pi)
            crps = pred_std * crps_normalized
            
            return float(np.mean(crps))
            
        except Exception as e:
            logging.warning(f"Failed to compute CRPS: {e}")
            return float('nan')
    
    def _compute_uq_quality_score(self, metrics: Dict[str, float]) -> float:
        """
        Aggregate UQ quality score combining accuracy, calibration, and coverage.
        """
        try:
            score = 0.0
            weight_sum = 0.0
            
            # Accuracy (40% weight)
            if 'R2' in metrics and not np.isnan(metrics['R2']):
                score += 0.4 * max(0, metrics['R2'])  # R²越高越好
                weight_sum += 0.4
            
            # Calibration (30% weight)
            if 'ACE' in metrics and not np.isnan(metrics['ACE']):
                # ACE smaller is better; map to 0–1 score
                ace_score = max(0, 1 - metrics['ACE'] * 10)  # 假设ACE>0.1就很差
                score += 0.3 * ace_score
                weight_sum += 0.3
            
            # Coverage quality (30% weight)
            picp_scores = []
            for conf in self.confidence_levels:
                picp_key = f'PICP_{int(conf*100)}'
                if picp_key in metrics and not np.isnan(metrics[picp_key]):
                    # PICP should be close to nominal confidence
                    picp_error = abs(metrics[picp_key] - conf)
                    picp_score = max(0, 1 - picp_error * 5)  # 误差>0.2就很差
                    picp_scores.append(picp_score)
            
            if picp_scores:
                score += 0.3 * np.mean(picp_scores)
                weight_sum += 0.3
            
            # Normalize
            if weight_sum > 0:
                score = score / weight_sum
            else:
                score = 0.0
            
            return float(score)
            
        except Exception as e:
            logging.warning(f"Failed to compute UQ quality score: {e}")
            return 0.0

class UQAnalyzer:
    """Analyzer offering deeper diagnostics of predictive uncertainty."""
    
    def __init__(self):
        self.metrics_calculator = UQMetrics()
    
    def analyze_uncertainty_components(self, 
                                     pred_dist: Dict[str, torch.Tensor],
                                     targets: torch.Tensor) -> Dict[str, float]:
        """Analyze basic statistics of predicted uncertainty and its relation to errors."""
        pred_mu = pred_dist['mu'].detach().cpu().numpy().flatten()
        pred_var = pred_dist['var'].detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()
        
        # Remove invalid entries
        valid_mask = (~np.isnan(pred_mu)) & (~np.isnan(pred_var)) & (~np.isnan(targets)) & (pred_var > 0)
        pred_mu = pred_mu[valid_mask]
        pred_var = pred_var[valid_mask]
        targets = targets[valid_mask]
        
        if len(pred_mu) == 0:
            return {}
        
        # Uncertainty statistics
        total_uncertainty = np.mean(pred_var)
        uncertainty_std = np.std(pred_var)
        min_uncertainty = np.min(pred_var)
        max_uncertainty = np.max(pred_var)
        
        # Relation between errors and uncertainty
        errors = np.abs(targets - pred_mu)
        error_uncertainty_corr = np.corrcoef(errors, np.sqrt(pred_var))[0, 1]
        
        return {
            'total_uncertainty': float(total_uncertainty),
            'uncertainty_std': float(uncertainty_std),
            'min_uncertainty': float(min_uncertainty),
            'max_uncertainty': float(max_uncertainty),
            'error_uncertainty_correlation': float(error_uncertainty_corr) if not np.isnan(error_uncertainty_corr) else 0.0
        }
    
    def reliability_analysis(self, 
                           pred_dist: Dict[str, torch.Tensor],
                           targets: torch.Tensor) -> Dict[str, Union[float, List[float]]]:
        """Reliability analysis to assess whether predictive uncertainty is trustworthy."""
        pred_mu, pred_std, targets_flat = self.metrics_calculator._prepare_data(pred_dist, targets, None)
        
        if len(pred_mu) == 0:
            return {}
        
        # QQ-style analysis
        residuals = targets_flat - pred_mu
        standardized_residuals = residuals / pred_std
        
        # Summary of standardized residual distribution
        residual_mean = np.mean(standardized_residuals)
        residual_std = np.std(standardized_residuals)
        
        # Kolmogorov–Smirnov test vs standard normal
        ks_stat, ks_pvalue = stats.kstest(standardized_residuals, 'norm')
        
        # Skewness and kurtosis
        skewness = stats.skew(standardized_residuals)
        kurtosis = stats.kurtosis(standardized_residuals)
        
        return {
            'residual_mean': float(residual_mean),
            'residual_std': float(residual_std),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis)
        }
    
    def generate_uncertainty_report(self, 
                                  pred_dist: Dict[str, torch.Tensor],
                                  targets: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Generate a complete uncertainty analysis report."""
        # Basic metrics
        basic_metrics = self.metrics_calculator.compute_all_metrics(pred_dist, targets, mask)
        
        # Uncertainty statistics
        uncertainty_analysis = self.analyze_uncertainty_components(pred_dist, targets)
        
        # Reliability
        reliability = self.reliability_analysis(pred_dist, targets)
        
        # Aggregate report
        report = {
            'basic_metrics': basic_metrics,
            'uncertainty_analysis': uncertainty_analysis,
            'reliability_analysis': reliability,
            'overall_assessment': self._assess_uq_quality(basic_metrics, uncertainty_analysis, reliability)
        }
        
        return report
    
    def _assess_uq_quality(self, 
                          basic_metrics: Dict,
                          uncertainty_analysis: Dict, 
                          reliability: Dict) -> Dict[str, str]:
        """Qualitative assessment of UQ quality with brief textual recommendations."""
        assessment = {}
        
        # Calibration quality
        if 'ACE' in basic_metrics:
            ace = basic_metrics['ACE']
            if ace < 0.05:
                assessment['calibration'] = "Excellent: calibration quality is very good."
            elif ace < 0.1:
                assessment['calibration'] = "Good: calibration is mostly accurate."
            else:
                assessment['calibration'] = "Needs improvement: calibration shows noticeable bias."
        
        # Coverage
        picp_95 = basic_metrics.get('PICP_95', None)
        if picp_95 is not None:
            if 0.92 <= picp_95 <= 0.98:
                assessment['coverage'] = "Excellent: 95% prediction interval coverage is accurate."
            elif 0.85 <= picp_95 <= 1.0:
                assessment['coverage'] = "Good: 95% prediction interval coverage is broadly acceptable."
            else:
                assessment['coverage'] = f"Needs improvement: 95% coverage deviates significantly ({picp_95:.3f})."
        
        # Reliability
        if 'ks_pvalue' in reliability:
            if reliability['ks_pvalue'] > 0.05:
                assessment['reliability'] = "Excellent: residuals are close to normal."
            else:
                assessment['reliability'] = "Needs improvement: residuals deviate from normality."
        
        return assessment

# =========================
# Convenience functions
# =========================
def quick_uq_eval(pred_dist: Dict[str, torch.Tensor], 
                  targets: torch.Tensor,
                  confidence: float = 0.95) -> Dict[str, float]:
    """Quick UQ evaluation returning core metrics at a single confidence level."""
    metrics = UQMetrics()
    
    # Compute only core metrics
    temp_config = uq_config.eval_metrics
    uq_config.eval_metrics = ['NLL', 'PICP', 'MPIW']
    uq_config.confidence_levels = [confidence]
    
    result = metrics.compute_all_metrics(pred_dist, targets)
    
    # Restore config
    uq_config.eval_metrics = temp_config
    
    return result

def compare_uq_models(model_results: Dict[str, Dict[str, torch.Tensor]], 
                     targets: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """Compare multiple UQ models on the same targets."""
    comparison = {}
    
    for model_name, pred_dist in model_results.items():
        metrics = UQMetrics()
        comparison[model_name] = metrics.compute_all_metrics(pred_dist, targets)
    
    return comparison

# =========================
# Simple test harness
# =========================
if __name__ == "__main__":
    print("=== Testing UQ metrics ===")
    
    # Create synthetic test data
    n_samples = 1000
    
    # Simulated predictive distribution
    true_mean = np.random.randn(n_samples)
    pred_mu = true_mean + np.random.randn(n_samples) * 0.1  # small bias
    pred_std = np.random.uniform(0.1, 0.5, n_samples)      # varying uncertainty
    
    # Sample targets from true distribution
    targets = true_mean + np.random.randn(n_samples) * 0.2
    
    # Wrap as torch tensors
    pred_dist = {
        'mu': torch.tensor(pred_mu, dtype=torch.float32),
        'var': torch.tensor(pred_std**2, dtype=torch.float32),
        'std': torch.tensor(pred_std, dtype=torch.float32)
    }
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    # Test metrics computation
    print("\n1. Base metric computation")
    metrics = UQMetrics()
    results = metrics.compute_all_metrics(pred_dist, targets_tensor)
    
    print("Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    # Test analyzer
    print("\n2. UQ analyzer")
    analyzer = UQAnalyzer()
    
    # Uncertainty component analysis
    uncertainty_analysis = analyzer.analyze_uncertainty_components(pred_dist, targets_tensor)
    print("Uncertainty analysis:")
    for key, value in uncertainty_analysis.items():
        print(f"  {key}: {value:.4f}")
    
    # Reliability analysis
    reliability = analyzer.reliability_analysis(pred_dist, targets_tensor)
    print("Reliability analysis:")
    for key, value in reliability.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Full report
    print("\n3. Full report")
    report = analyzer.generate_uncertainty_report(pred_dist, targets_tensor)
    
    print("Overall assessment:")
    for key, assessment in report['overall_assessment'].items():
        print(f"  {key}: {assessment}")
    
    # Quick evaluation
    print("\n4. Quick evaluation")
    quick_results = quick_uq_eval(pred_dist, targets_tensor)
    print("Quick evaluation results:")
    for key, value in quick_results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ UQ metrics test finished.")