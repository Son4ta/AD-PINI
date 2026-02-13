# AD-PINI v4 data preprocessing - implementation of the escalator principle

import os
import numpy as np
import xarray as xr
import pandas as pd
from typing import Tuple, Dict, Optional
import warnings
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import logging

class ClimateDecomposer:
    """Climatology decomposer implementing the escalator principle."""
    
    def __init__(self, order: int = 2, harmonic_terms: int = 2, simple_climatology: bool = False):
        """
        Initialize climatology decomposer.
        
        Args:
            order: polynomial trend order
            harmonic_terms: number of harmonic terms
            simple_climatology: S4 ablation; if True use simple monthly means instead of harmonic fit
        """
        self.order = order
        self.harmonic_terms = harmonic_terms
        self.simple_climatology = simple_climatology  # S4 ablation flag
        self.fitted_params = {}  # store fitted parameters
        
    def _build_design_matrix(self, time_array: np.ndarray) -> np.ndarray:
        """
        Build design matrix X for climatology fitting (numerically stable version).
        
        X(t) = a1 + a2*t + a3*t² + Σ[ak*sin(2kπt/T) + bk*cos(2kπt/T)]
        
        Args:
            time_array: time indices (monthly index 0-371)
            
        Returns:
            design_matrix: [T, n_features] design matrix
        """
        n_times = len(time_array)
        n_features = 1 + self.order + 2 * self.harmonic_terms  # intercept + trend + harmonics
        
        X = np.zeros((n_times, n_features), dtype=np.float64)  # force double precision
        
        # Standardize time into [-1, 1] to avoid large values
        t_mean = time_array.mean()
        t_std = time_array.std()
        t_norm = (time_array - t_mean) / (t_std + 1e-8)  # avoid division by zero
        
        # Constant term
        X[:, 0] = 1.0
        
        # Polynomial trend terms - using normalized time
        for i in range(1, self.order + 1):
            X[:, i] = t_norm ** i
            
        # Harmonic terms (seasonal cycles) - using original month index
        feature_idx = self.order + 1
        for k in range(1, self.harmonic_terms + 1):
            omega = 2 * np.pi * k / 12  # monthly frequency
            X[:, feature_idx] = np.sin(omega * time_array)
            X[:, feature_idx + 1] = np.cos(omega * time_array)
            feature_idx += 2
        
        # Save normalization parameters for prediction
        if not hasattr(self, 'time_norm_params'):
            self.time_norm_params = {'mean': t_mean, 'std': t_std}
            
        return X
    
    def fit_simple_monthly_climatology(self, data: xr.DataArray, train_mask: np.ndarray) -> Dict:
        """
        S4 ablation: fit simple monthly-mean climatology (instead of harmonic fit).
        
        Args:
            data: input data [time, lat, lon]
            train_mask: training-time boolean mask
            
        Returns:
            fit_results: dictionary of fitting results
        """
        logging.info("S4 ablation: using simple monthly-mean climatology...")
        
        # Extract training data
        train_data = data[train_mask]
        
        # Month index for training times
        if hasattr(train_data, 'time'):
            months = train_data.time.dt.month.values
        else:
            # Assume data starts from 1993-01 and each time step is one month
            train_indices = np.where(train_mask)[0]
            months = (train_indices % 12) + 1  # month 1-12
        
        # Compute mean for each month
        monthly_means = np.full((12,) + train_data.shape[1:], np.nan, dtype=np.float32)
        
        for month in range(1, 13):
            month_mask = (months == month)
            if month_mask.sum() > 0:
                monthly_means[month-1] = np.nanmean(train_data[month_mask], axis=0)
        
        # Store monthly means
        self.fitted_params['monthly_means'] = monthly_means
        self.fitted_params['method'] = 'simple_monthly'
        
        return {
            'method': 'simple_monthly',
            'monthly_means': monthly_means,
            'n_parameters': 12,  # one mean for each of the 12 months
            'r2_mean': np.nan,   # R² not applicable for simple monthly means
            'valid_pixels': np.isfinite(monthly_means).any(axis=0).sum()
        }
    
    def fit_climatology(self, data: xr.DataArray, train_mask: np.ndarray) -> Dict:
        """
        Fit climatology (training set only, to avoid data leakage) - memory-efficient version.
        
        Args:
            data: input data [time, lat, lon]
            train_mask: training-time boolean mask
            
        Returns:
            fit_results: dictionary of fitting results
        """
        # S4 ablation: choose simple monthly mean or harmonic fit
        if self.simple_climatology:
            return self.fit_simple_monthly_climatology(data, train_mask)
        
        logging.info("Start climatology fitting (harmonic mode)...")
        
        # Training time indices
        train_times = np.where(train_mask)[0]
        X_train = self._build_design_matrix(train_times)
        
        # Initialize parameter storage - use float32
        lat_size, lon_size = data.shape[1], data.shape[2]
        n_features = X_train.shape[1]
        
        params = np.full((lat_size, lon_size, n_features), np.nan, dtype=np.float32)
        r2_scores = np.full((lat_size, lon_size), np.nan, dtype=np.float32)
        
        # Process in blocks to save memory
        block_size = 50  # process 50 latitude rows per block
        valid_pixels = 0
        
        for i_start in range(0, lat_size, block_size):
            i_end = min(i_start + block_size, lat_size)
            
            # Log progress intermittently
            if i_start % (block_size * 10) == 0:
                logging.info(f"Fitting climatology progress: {i_start}/{lat_size} latitude rows")
            
            # Extract current block
            block_data = data.isel(time=train_times, latitude=slice(i_start, i_end)).values
            
            # Fit per pixel
            for i_rel in range(i_end - i_start):
                i = i_start + i_rel
                for j in range(lon_size):
                    pixel_data = block_data[:, i_rel, j]
                    
                    # Skip pixels with NaNs
                    if np.isnan(pixel_data).any():
                        continue
                        
                    try:
                        # Least-squares fit (force float64 for numerical stability)
                        X_train_f64 = X_train.astype(np.float64)
                        pixel_data_f64 = pixel_data.astype(np.float64)
                        coeffs, residuals, rank, s = np.linalg.lstsq(X_train_f64, pixel_data_f64, rcond=None)
                        params[i, j, :] = coeffs.astype(np.float32)
                        
                        # Compute R²
                        y_pred = X_train @ coeffs
                        ss_res = np.sum((pixel_data - y_pred) ** 2)
                        ss_tot = np.sum((pixel_data - np.mean(pixel_data)) ** 2)
                        r2_scores[i, j] = (1 - (ss_res / ss_tot)) if ss_tot > 0 else 0
                        
                        valid_pixels += 1
                        
                    except np.linalg.LinAlgError:
                        continue
        
        # Save fitted parameters
        self.fitted_params = {
            'params': params,
            'r2_scores': r2_scores,
            'n_features': n_features,
            'valid_pixels': valid_pixels
        }
        
        logging.info(f"Climatology fitting complete, valid pixels: {valid_pixels}")
        
        return self.fitted_params
    
    def predict_climatology(self, time_indices: np.ndarray) -> np.ndarray:
        """
        Predict climatology time series - memory-efficient version.
        
        Args:
            time_indices: time indices to predict
            
        Returns:
            climatology: predicted climatology [time, lat, lon]
        """
        if not self.fitted_params:
            raise ValueError("Must call fit_climatology() before predict_climatology().")
        
        # S4 ablation: simple monthly-mean prediction
        if self.fitted_params.get('method') == 'simple_monthly':
            return self._predict_simple_monthly(time_indices)
        
        # Standard harmonic prediction
        X = self._build_design_matrix(time_indices)
        params = self.fitted_params['params']
        
        n_times = len(time_indices)
        lat_size, lon_size = params.shape[0], params.shape[1]
        
        # Use float32 to save memory
        climatology = np.full((n_times, lat_size, lon_size), np.nan, dtype=np.float32)
        
        # Process in blocks
        block_size = 50
        for i_start in range(0, lat_size, block_size):
            i_end = min(i_start + block_size, lat_size)
            
            if i_start % (block_size * 10) == 0:
                logging.info(f"Climatology prediction progress: {i_start}/{lat_size} latitude rows")
            
            for i in range(i_start, i_end):
                for j in range(lon_size):
                    if not np.isnan(params[i, j, 0]):  # check for valid parameters
                        climatology[:, i, j] = (X @ params[i, j, :]).astype(np.float32)
                    
        return climatology
    
    def _predict_simple_monthly(self, time_indices: np.ndarray) -> np.ndarray:
        """
        S4 ablation: predict climatology using simple monthly-mean values.
        
        Args:
            time_indices: time indices to predict
            
        Returns:
            climatology: predicted climatology [time, lat, lon]
        """
        monthly_means = self.fitted_params['monthly_means']
        n_times = len(time_indices)
        lat_size, lon_size = monthly_means.shape[1], monthly_means.shape[2]
        
        climatology = np.full((n_times, lat_size, lon_size), np.nan, dtype=np.float32)
        
        for i, time_idx in enumerate(time_indices):
            # Determine month index (assume start at 1993-01)
            month = (time_idx % 12)  # month index 0-11
            climatology[i] = monthly_means[month]
        
        return climatology


class SensitivityCalculator:
    """Calculator for physical sensitivity factors."""
    
    @staticmethod
    def calculate_revelle_factor(alk: np.ndarray, dic: np.ndarray) -> np.ndarray:
        """
        Compute Revelle buffer factor.
        
        γ_DIC ≈ (3·Alk·DIC - 2·DIC²) / ((2·DIC - Alk)(Alk - DIC))
        
        Args:
            alk: total alkalinity [μmol/kg]
            dic: dissolved inorganic carbon [μmol/kg]
            
        Returns:
            revelle_factor: Revelle factor
        """
        # Avoid division by zero
        numerator = 3 * alk * dic - 2 * dic**2
        denominator = (2 * dic - alk) * (alk - dic)
        
        # Apply safety thresholds
        valid_mask = (np.abs(denominator) > 1e-6) & np.isfinite(numerator) & np.isfinite(denominator)
        
        revelle = np.full_like(alk, np.nan)
        revelle[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        # Physical constraints (Revelle factors typically ~5–20)
        revelle = np.clip(revelle, 1, 50)
        
        return revelle
    
    @staticmethod
    def calculate_sensitivities(pco2: np.ndarray, dic: np.ndarray, alk: np.ndarray, 
                              gamma_t: float = 0.0423) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute thermal and non-thermal sensitivity factors.
        
        S_T = γ_T · pCO2
        S_NT = γ_DIC · pCO2 / DIC
        
        Args:
            pco2: sea-surface pCO2 [μatm]
            dic: dissolved inorganic carbon [μmol/kg]
            alk: total alkalinity [μmol/kg]
            gamma_t: thermal sensitivity coefficient
            
        Returns:
            s_thermal: thermal sensitivity
            s_nonther: non-thermal sensitivity
        """
        # Thermal sensitivity
        s_thermal = gamma_t * pco2
        
        # Non-thermal sensitivity
        revelle = SensitivityCalculator.calculate_revelle_factor(alk, dic)
        
        # Avoid division by zero
        valid_mask = (dic > 0) & np.isfinite(revelle)
        s_nonther = np.full_like(pco2, np.nan)
        s_nonther[valid_mask] = revelle[valid_mask] * pco2[valid_mask] / dic[valid_mask]
        
        return s_thermal, s_nonther


class DataPreprocessorV4:
    """Data preprocessor for AD-PINI v4."""
    
    def __init__(self, config):
        """
        Initialize preprocessor.
        
        Args:
            config: configuration object
        """
        self.config = config
        self.decomposer = ClimateDecomposer(
            order=config.CLIMATOLOGY_ORDER,
            harmonic_terms=config.HARMONIC_TERMS,
            simple_climatology=config.SIMPLE_CLIMATOLOGY  # S4 ablation flag
        )
        self.calc = SensitivityCalculator()
        
        # Set random seed (if provided in config)
        if hasattr(config, 'RANDOM_SEED'):
            np.random.seed(config.RANDOM_SEED)
        
        # Logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_raw_data(self) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Load raw NetCDF data.
        
        Returns:
            chem_ds: chemical dataset
            sst_ds: temperature dataset
        """
        self.logger.info("Loading raw datasets...")
        
        # Load chemical data
        chem_ds = xr.open_dataset(self.config.CHEM_DATA_PATH)
        
        # Load SST data
        sst_ds = xr.open_dataset(self.config.SST_DATA_PATH)
        
        # Remove SST depth dimension if present
        if 'depth' in sst_ds.dims:
            sst_ds = sst_ds.squeeze('depth')
            
        self.logger.info(f"Chemical dataset shape (spco2): {chem_ds.spco2.shape}")
        self.logger.info(f"SST dataset shape (thetao_mean): {sst_ds.thetao_mean.shape}")
        
        return chem_ds, sst_ds
    
    def align_datasets(self, chem_ds: xr.Dataset, sst_ds: xr.Dataset) -> xr.Dataset:
        """
        Align datasets in time and space, with interpolation (memory-efficient).
        
        Args:
            chem_ds: chemical dataset
            sst_ds: temperature dataset
            
        Returns:
            aligned_ds: aligned dataset
        """
        self.logger.info("Aligning datasets...")
        
        # Align time axis - take intersection
        common_times = pd.Index(chem_ds.time.values).intersection(pd.Index(sst_ds.time.values))
        
        chem_aligned = chem_ds.sel(time=common_times)
        sst_aligned = sst_ds.sel(time=common_times)
        
        # Spatial downsampling (if enabled)
        if hasattr(self.config, 'DOWNSAMPLE_FACTOR') and self.config.DOWNSAMPLE_FACTOR > 1:
            factor = self.config.DOWNSAMPLE_FACTOR
            self.logger.info(f"Applying spatial downsampling factor: {factor}")
            
            # Downsample chemical data
            chem_aligned = chem_aligned.isel(
                latitude=slice(None, None, factor),
                longitude=slice(None, None, factor)
            )
            
            # Downsample SST, then regrid
            sst_aligned = sst_aligned.isel(
                latitude=slice(None, None, factor),
                longitude=slice(None, None, factor)
            )
        
        # Fix spatial alignment: correct grid offset
        # SST(-180.000° start) vs chemistry(-179.875° start) = 0.125° half-grid offset
        
        # Interpolate SST onto chemical grid
        sst_regridded = sst_aligned.interp(
            latitude=chem_aligned.latitude,
            longitude=chem_aligned.longitude,  # explicitly regrid longitude
            method='linear'
        )
        
        logging.info("Grid alignment details:")
        logging.info(f"  Original SST lon range:  {float(sst_aligned.longitude.min()):.6f} - {float(sst_aligned.longitude.max()):.6f}")
        logging.info(f"  Target chem lon range:   {float(chem_aligned.longitude.min()):.6f} - {float(chem_aligned.longitude.max()):.6f}")
        logging.info(f"  Regridded SST lon range: {float(sst_regridded.longitude.min()):.6f} - {float(sst_regridded.longitude.max()):.6f}")
        
        # Merge to a single dataset
        aligned_ds = xr.Dataset({
            'pco2': chem_aligned.spco2,
            'dic': chem_aligned.tco2, 
            'alk': chem_aligned.talk,
            'sst': sst_regridded.thetao_mean
        })
        
        # Handle fill values
        aligned_ds = aligned_ds.where(aligned_ds != self.config.FILL_VALUE, np.nan)
        
        # Cast to float32 to save memory
        if hasattr(self.config, 'USE_FLOAT32') and self.config.USE_FLOAT32:
            for var in aligned_ds.data_vars:
                aligned_ds[var] = aligned_ds[var].astype(np.float32)
        
        self.logger.info(f"Aligned dataset shape: {aligned_ds.pco2.shape}")
        self.logger.info(f"Estimated memory usage: {aligned_ds.nbytes / 1024**3:.2f} GB")
        
        return aligned_ds
    
    def decompose_to_anomalies(self, aligned_ds: xr.Dataset) -> xr.Dataset:
        """
        Decompose variables into climatology and anomalies.
        
        Args:
            aligned_ds: aligned dataset
            
        Returns:
            decomposed_ds: dataset with climatology and anomaly fields
        """
        self.logger.info("Performing climatology–anomaly decomposition...")
        
        # Training mask (first 80% of time used to fit climatology)
        n_times = len(aligned_ds.time)
        train_size = int(n_times * self.config.TRAIN_SPLIT)
        train_mask = np.zeros(n_times, dtype=bool)
        train_mask[:train_size] = True
        
        # Time index array
        time_indices = np.arange(n_times)
        
        # Decompose variables
        variables = ['pco2', 'sst', 'dic']
        decomposed_data = {}
        
        for var in variables:
            self.logger.info(f"Decomposing variable: {var}")
            
            data = aligned_ds[var]
            
            # Fit climatology on training set
            self.decomposer.fit_climatology(data, train_mask)
            
            # Predict climatology across full time series
            climatology = self.decomposer.predict_climatology(time_indices)
            
            # Anomalies
            anomaly = data.values - climatology
            
            # Store results
            decomposed_data[f'{var}_clim'] = (['time', 'latitude', 'longitude'], climatology)
            decomposed_data[f'{var}_anom'] = (['time', 'latitude', 'longitude'], anomaly)
            decomposed_data[var] = data  # keep original data field
        
        # Keep alk (for sensitivity calculation)
        decomposed_data['alk'] = aligned_ds.alk
        
        # Build new dataset
        decomposed_ds = xr.Dataset(
            decomposed_data,
            coords=aligned_ds.coords
        )
        
        return decomposed_ds
    
    def absolute_learning_transform(self, aligned_ds: xr.Dataset) -> xr.Dataset:
        """
        S3 ablation: transform into absolute-learning mode using raw values instead of anomalies.
        
        Args:
            aligned_ds: aligned dataset
            
        Returns:
            transformed_ds: transformed dataset (interface-compatible)
        """
        self.logger.info("Running S3 ablation: absolute-learning mode transform...")
        
        # Use absolute values but keep variable naming for interface compatibility:
        # treat original values as "anomalies" and set climatologies to zero.
        result_ds = aligned_ds.copy(deep=True)
        
        # Create dummy climatology (all zeros) and increments (all zeros)
        result_ds['pco2_clim'] = xr.zeros_like(aligned_ds['pco2'])
        result_ds['sst_clim'] = xr.zeros_like(aligned_ds['sst'])
        result_ds['dic_clim'] = xr.zeros_like(aligned_ds['dic'])
        
        # "Anomalies" are actually raw values here
        result_ds['pco2_anom'] = aligned_ds['pco2']
        result_ds['sst_anom'] = aligned_ds['sst'] 
        result_ds['dic_anom'] = aligned_ds['dic']
        
        # Dummy climatological increments (all zeros)
        result_ds['delta_sst_clim'] = xr.zeros_like(aligned_ds['sst'])
        result_ds['delta_dic_clim'] = xr.zeros_like(aligned_ds['dic'])
        
        # Drop original variables to avoid confusion
        for var in ['pco2', 'sst', 'dic']:
            if var in result_ds:
                result_ds = result_ds.drop_vars(var)
        
        self.logger.info(f"S3 absolute-learning transform complete: {result_ds.dims}")
        return result_ds
    
    def calculate_targets_and_sensitivities(self, decomposed_ds: xr.Dataset) -> xr.Dataset:
        """
        Compute training targets and sensitivity factors (memory-efficient).
        
        Args:
            decomposed_ds: dataset after climatology/anomaly decomposition
            
        Returns:
            final_ds: final dataset containing all training variables
        """
        self.logger.info("Computing training targets and sensitivities...")
        
        # Use less memory by looping over time steps
        n_times = len(decomposed_ds.time) - 1  # one fewer because of differencing
        n_lat, n_lon = decomposed_ds.pco2.shape[1:]
        
        # Preallocate arrays in float32
        delta_sst_total = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        delta_dic_total = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        delta_sst_clim = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        delta_dic_clim = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        s_thermal = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        s_nonther = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        
        self.logger.info(f"Processing {n_times} time steps...")
        
        # Loop over time steps
        for t in range(n_times):
            if t % 50 == 0:
                self.logger.info(f"Processing time step {t}/{n_times}")
            
            # Compute increments
            delta_sst_total[t] = (decomposed_ds.sst.values[t+1] - decomposed_ds.sst.values[t]).astype(np.float32)
            delta_dic_total[t] = (decomposed_ds.dic.values[t+1] - decomposed_ds.dic.values[t]).astype(np.float32)
            delta_sst_clim[t] = (decomposed_ds.sst_clim.values[t+1] - decomposed_ds.sst_clim.values[t]).astype(np.float32)
            delta_dic_clim[t] = (decomposed_ds.dic_clim.values[t+1] - decomposed_ds.dic_clim.values[t]).astype(np.float32)
            
            # Compute sensitivity factors
            s_t, s_nt = self.calc.calculate_sensitivities(
                decomposed_ds.pco2.values[t],  # state at time t
                decomposed_ds.dic.values[t],
                decomposed_ds.alk.values[t],
                self.config.GAMMA_T
            )
            s_thermal[t] = s_t.astype(np.float32)
            s_nonther[t] = s_nt.astype(np.float32)
        
        # Build final dataset (time dimension reduced by one because of differencing)
        time_coords = decomposed_ds.time.values[:-1]
        
        self.logger.info("Building final dataset...")
        
        final_data = {
            # Network inputs (historical anomalies) - cast to float32
            'pco2_anom_input': (['time', 'latitude', 'longitude'], 
                               decomposed_ds.pco2_anom.values[:-1].astype(np.float32)),
            
            # Escalator increments (known forcings)
            'delta_sst_clim': (['time', 'latitude', 'longitude'], delta_sst_clim),
            'delta_dic_clim': (['time', 'latitude', 'longitude'], delta_dic_clim),
            
            # Supervision targets
            'delta_sst_gt': (['time', 'latitude', 'longitude'], delta_sst_total),
            'delta_dic_gt': (['time', 'latitude', 'longitude'], delta_dic_total),
            'pco2_target': (['time', 'latitude', 'longitude'], 
                           decomposed_ds.pco2.values[1:].astype(np.float32)),  # pCO2 at t+1
            
            # Physical state (current time)
            'pco2_current': (['time', 'latitude', 'longitude'], 
                            decomposed_ds.pco2.values[:-1].astype(np.float32)),
            'dic_current': (['time', 'latitude', 'longitude'], 
                           decomposed_ds.dic.values[:-1].astype(np.float32)),
            'alk_current': (['time', 'latitude', 'longitude'], 
                           decomposed_ds.alk.values[:-1].astype(np.float32)),
            
            # Sensitivity factors
            's_thermal': (['time', 'latitude', 'longitude'], s_thermal),
            's_nonther': (['time', 'latitude', 'longitude'], s_nonther),
        }
        
        final_ds = xr.Dataset(
            final_data,
            coords={
                'time': time_coords,
                'latitude': decomposed_ds.latitude,
                'longitude': decomposed_ds.longitude
            }
        )
        
        return final_ds
    
    def save_processed_data(self, final_ds: xr.Dataset) -> None:
        """
        Save processed dataset to disk.
        
        Args:
            final_ds: final processed dataset
        """
        self.logger.info(f"Saving processed data to: {self.config.PROCESSED_DATA_PATH}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config.PROCESSED_DATA_PATH), exist_ok=True)
        
        # Save as NetCDF
        final_ds.to_netcdf(self.config.PROCESSED_DATA_PATH)
        
        # Save metadata as JSON sidecar
        metadata = {
            'processing_date': pd.Timestamp.now().isoformat(),
            'data_shape': str(final_ds.pco2_anom_input.shape),
            'time_range': [str(final_ds.time.min().values), str(final_ds.time.max().values)],
            'spatial_coverage': {
                'lat_range': [float(final_ds.latitude.min()), float(final_ds.latitude.max())],
                'lon_range': [float(final_ds.longitude.min()), float(final_ds.longitude.max())]
            },
            'climatology_params': {
                'order': self.decomposer.order,
                'harmonic_terms': self.decomposer.harmonic_terms,
                'train_split': self.config.TRAIN_SPLIT
            }
        }
        
        import json
        with open(self.config.PROCESSED_DATA_PATH.replace('.nc', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def process_all(self, force_recompute: bool = False) -> xr.Dataset:
        """
        Run full preprocessing pipeline.
        
        Args:
            force_recompute: whether to force recomputation even if cache exists
            
        Returns:
            final_ds: final processed dataset
        """
        # Check cache
        if os.path.exists(self.config.PROCESSED_DATA_PATH) and not force_recompute:
            # Check whether file is valid (non-empty and readable)
            if os.path.getsize(self.config.PROCESSED_DATA_PATH) > 0:
                try:
                    self.logger.info("Loading existing processed dataset...")
                    return xr.open_dataset(self.config.PROCESSED_DATA_PATH)
                except Exception as e:
                    self.logger.warning(f"Cached file invalid, regenerating: {e}")
                    os.remove(self.config.PROCESSED_DATA_PATH)
            else:
                self.logger.warning("Cached file empty, regenerating.")
                os.remove(self.config.PROCESSED_DATA_PATH)
        
        # Run full preprocessing pipeline
        self.logger.info("Starting full preprocessing pipeline...")
        
        # 1. Load raw data
        chem_ds, sst_ds = self.load_raw_data()
        
        # 2. Align datasets
        aligned_ds = self.align_datasets(chem_ds, sst_ds)
        
        # 3. Climatology–anomaly decomposition or S3 ablation (absolute-learning mode)
        if self.config.ABSOLUTE_LEARNING_MODE:
            # S3 ablation: use absolute values instead of anomalies, skip climatology fitting
            decomposed_ds = self.absolute_learning_transform(aligned_ds)
            self.logger.info("S3 ablation: absolute-learning mode enabled; skipping anomaly decomposition.")
        else:
            # Standard pipeline: climatology–anomaly decomposition
            decomposed_ds = self.decompose_to_anomalies(aligned_ds)
        
        # 4. Compute targets and sensitivities
        final_ds = self.calculate_targets_and_sensitivities(decomposed_ds)
        
        # 5. Save to disk
        self.save_processed_data(final_ds)
        
        self.logger.info("Data preprocessing finished.")
        
        return final_ds


def main():
    """Simple smoke test for preprocessor."""
    from configs.config_v4 import config
    
    processor = DataPreprocessorV4(config)
    final_ds = processor.process_all(force_recompute=True)
    
    print(f"Preprocessing complete. Data shape: {final_ds.pco2_anom_input.shape}")
    print(f"Variables: {list(final_ds.data_vars.keys())}")


if __name__ == "__main__":
    main()