# AD-PINI v4 configuration file - anomaly prediction framework based on escalator principle

import os
import torch

class ConfigV4:
    def __init__(self):
        # =========================
        # Data path configuration (Data Paths)
        # =========================
        self.BASE_DATA_DIR = "/data/PINP/data"
        self.CHEM_DATA_PATH = os.path.join(self.BASE_DATA_DIR, "pco2_dic_alk_ph", "pco2_dic_alk_ph.nc")
        self.SST_DATA_PATH = os.path.join(self.BASE_DATA_DIR, "sst", "sst.nc")
        self.CHL_DATA_PATH = os.path.join(self.BASE_DATA_DIR, "chl", "chl.nc")  # reserved interface
        
        # Cache paths
        self.CACHE_DIR = "/data/AD-PINI/data/v4/cache"
        self.PROCESSED_DATA_PATH = os.path.join(self.CACHE_DIR, "processed_data_v4.nc")
        
        # Output paths
        self.OUTPUT_DIR = "/data/AD-PINI/outputs/v4"
        self.CHECKPOINT_DIR = os.path.join(self.OUTPUT_DIR, "checkpoints")
        self.LOG_DIR = os.path.join(self.OUTPUT_DIR, "logs")
        self.VIZ_DIR = os.path.join(self.OUTPUT_DIR, "visualizations")
        
        # =========================
        # Data processing configuration (Data Processing)
        # =========================
        self.FILL_VALUE = 9.96921e+36  # NetCDF missing value indicator
        self.TIME_START = "1993-01"     # Data start time
        self.TIME_END = "2023-12"       # Data end time
        self.TARGET_LAT = 713           # Target latitude grid size
        self.TARGET_LON = 1440          # Target longitude grid size
        self.TIME_STEPS = 372           # Total number of time steps
        
        # Memory optimization options
        self.DOWNSAMPLE_FACTOR = 2      # Downsampling factor (1=no downsampling, 2=1/4, 4=1/16) - fix resolution issues
        self.USE_FLOAT32 = True         # Use float32 instead of float64
        
        # =========================
        # Normalization configuration (Normalization Configuration)
        # =========================
        self.NORMALIZATION_MODE = "zscore"     # Normalization mode: "zscore" | "minmax"
        self.NORMALIZATION_RANGE = (-1, 1)    # Target range for MinMax normalization
        self.NORMALIZATION_EPSILON = 1e-8     # Numerical stability epsilon
        
        # Climatology fitting parameters
        self.CLIMATOLOGY_ORDER = 2      # Polynomial trend order (a1 + a2*t + a3*t^2)
        self.HARMONIC_TERMS = 4         # Number of harmonic terms (seasonal cycle)
        self.TRAIN_SPLIT = 0.8          # Train split ratio (for climatology fitting only)
        
        # =========================
        # Model architecture configuration (Model Architecture)
        # =========================
        # Anomaly prediction network (Anomaly U-Net) - enlarged 1.5x to improve capacity
        self.HISTORY_LENGTH = 12        # Length of historical time series (more temporal information)
        self.INPUT_CHANNELS = 1         # Number of input channels (pCO2 anomaly)
        self.OUTPUT_CHANNELS = 2        # Number of output channels (δSST_anom, δDIC_anom)
        self.UNET_FEATURES = [32, 64, 128, 256, 512]  # Feature dimensions
        
        # Physics layer parameters
        self.GAMMA_T = 0.0423          # Thermodynamic sensitivity constant (%/°C)
        
        # Residual correction network - adjusted accordingly
        self.CORRECTOR_FEATURES = [32, 64, 128]  # Feature dimensions of the correction network
        self.CORRECTOR_FREEZE_EPOCHS = 0       # Number of epochs to freeze the corrector
        
        # =========================
        # Sensitivity refiner configuration (Sensitivity Refiner Configuration)
        # =========================
        self.ENABLE_SENSITIVITY_REFINING = False    # Whether to enable sensitivity refining
        self.SENSITIVITY_REFINER_DIM = 32         # Hidden dimension of the refiner
        self.SENSITIVITY_REFINER_CORRECTION_RANGE = 0.15  # Correction range ±15%
        
        # =========================
        # Training configuration (Training Configuration)  
        # =========================
        self.EPOCHS = 100                # Fewer epochs after optimization due to faster convergence
        self.BATCH_SIZE = 2           # Conservative batch size to avoid OOM
        
        # Learning rate strategy for larger models
        self.LEARNING_RATE = 1e-5       # Lower learning rate for stability with large model
        self.WARMUP_EPOCHS = 5          # Warmup epochs (shorter warmup)
        self.WARMUP_LR = 1e-6          # More conservative warmup learning rate
        self.WEIGHT_DECAY = 5e-5       # Stronger regularization to prevent overfitting in large model
        self.GRADIENT_CLIP = 1.0       # Smaller gradient clipping threshold for large model
        '''
        
        '''
        # Re-balanced loss weights
        self.LAMBDA_STATE = 1.0         # State supervision loss weight for SST+DIC increments (ΔSST, ΔDIC)
        self.LAMBDA_TASK = 1.0          # Task loss weight for final pCO2 prediction and physics-layer pCO2
        self.LAMBDA_SPARSITY = 0.1      # Sparsity correction loss weight (L1 on residual correction term)
        self.LAMBDA_PHYSICS = 0.0       # Physics constraint loss weight (pCO2 anomaly decomposition)
        self.LAMBDA_CONSTRAINT = 0.0   # Anomaly decomposition constraint weight (R_therm + R_bio ≈ pCO2_anom)
        
        # Optimized learning rate scheduling strategy
        self.SCHEDULER_PATIENCE = 30    # Reasonable patience (reduces waiting time)
        self.SCHEDULER_FACTOR = 0.7     # Smaller LR decay factor  
        self.MIN_LR = 1e-7             # Lower minimum learning rate
        
        # =========================
        # Multi-GPU configuration (Multi-GPU Configuration)
        # =========================
        self.USE_MULTI_GPU = True
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_WORKERS = 4           # More DataLoader workers to better utilize CPU
        
        # Early stopping strategy
        self.EARLY_STOPPING = True      # Enable early stopping
        self.EARLY_STOPPING_PATIENCE = 75  # Early stopping patience
        self.EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum improvement threshold (more sensitive)
        
        # =========================
        # Experiment configuration (Experiment Configuration)
        # =========================
        self.EXPERIMENT_NAME = "carbon_pinp_v4_scaled"
        self.USE_WANDB = True           # WandB logging switch - enable experiment tracking
        self.WANDB_PROJECT = "AD-PINI-v4"
        self.WANDB_ENTITY = None        # Your wandb username, None to use default
        self.WANDB_TAGS = ["scaled-model", "1.25x-features", "optimized-loss", "v4", "physics-constrained"]
        self.WANDB_NOTES = "Scaled model - 5.47M parameters, physics constraints, anomaly decomposition architecture"
        
        # WandB logging configuration
        self.WANDB_LOG_FREQ = 10        # Log every N batches
        self.WANDB_LOG_GRADIENTS = True # Whether to log gradient histograms
        self.WANDB_LOG_LEARNING_RATE = True  # Whether to log learning rate
        self.WANDB_LOG_MODEL_TOPOLOGY = True # Whether to log model structure
        
        # Saving and evaluation
        self.SAVE_EVERY = 25            # Save checkpoints every N epochs
        self.EVAL_EVERY = 1             # Evaluate every epoch
        self.VISUALIZE_EVERY = 10       # Core visualization frequency (6 figures)
        
        # =========================
        # Visualization configuration (Visualization Configuration)
        # =========================
        self.ENABLE_VISUALIZATION = True  # Global visualization switch - disabling speeds up training
        self.VIZ_DPI = 300              # DPI for high-quality scientific figures
        self.VIZ_FIGSIZE = (12, 8)      # Figure size
        self.VIZ_CMAP_THERMAL = 'Reds'  # Colormap for thermodynamics-dominated regions
        self.VIZ_CMAP_NONTHER = 'Blues' # Colormap for non-thermodynamics-dominated regions
        
        # =========================
        # Random seed configuration (Random Seed Configuration)
        # =========================
        self.RANDOM_SEED = 721           # Main random seed to ensure reproducibility
        self.USE_DETERMINISTIC = False   # Avoid conflict with adaptive_avg_pool2d
        self.CUDA_DETERMINISTIC = True  # Deterministic CUDA computation
        
        # =========================
        # Ablation study configuration (Ablation Study Configuration)
        # =========================
        # Model Group ablations
        self.PHYSICS_ONLY_MODE = False      # M2: physics layer only, no residual correction (tests Taylor truncation error)
        self.BLACK_BOX_MODE = False         # M3: pure black-box regression (tests physics-layer inductive bias)
        
        # Strategy Group ablations  
        self.SINGLE_CHANNEL_MODE = False    # S2: single-channel output, no thermal/non-thermal decomposition
        self.ABSOLUTE_LEARNING_MODE = False # S3: learn absolute values instead of anomalies
        self.SIMPLE_CLIMATOLOGY = False     # S4: use simple monthly means instead of harmonic fit
        
        # Loss Group ablations (implemented via adjusting weights)
        # C2: w/o Physics Consistency - set physics consistency-related weights to 0
        # C3: w/o State Loss - set LAMBDA_STATE = 0  
        # C4: w/o Sparsity Loss - set LAMBDA_SPARSITY = 0
        
        # =========================
        # Debug configuration (Debug Configuration) 
        # =========================
        self.DEBUG_MODE = False
        self.FORCE_RECOMPUTE = False    # Force recomputing preprocessed data
        self.VERBOSE = True
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directory structure"""
        dirs = [self.CACHE_DIR, self.OUTPUT_DIR, self.CHECKPOINT_DIR, 
                self.LOG_DIR, self.VIZ_DIR]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_device_info(self):
        """Get device information"""
        if torch.cuda.is_available():
            return {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'memory_total': torch.cuda.get_device_properties(0).total_memory,
                'is_multi_gpu': torch.cuda.device_count() > 1
            }
        return {'device_count': 0, 'current_device': 'cpu'}
    
    def __str__(self):
        """String representation of configuration"""
        return f"""
AD-PINI v4 Configuration:
============================
Model Type: Anomaly Prediction with Escalator Principle
Data Path: {self.BASE_DATA_DIR}
Device: {self.DEVICE}
Multi-GPU: {self.USE_MULTI_GPU and torch.cuda.device_count() > 1}
History Length: {self.HISTORY_LENGTH}
Batch Size: {self.BATCH_SIZE}
Learning Rate: {self.LEARNING_RATE}
Visualization: {'Enabled' if self.ENABLE_VISUALIZATION else 'Disabled (Speed Mode)'}
============================
        """

# Global configuration instance
config = ConfigV4()