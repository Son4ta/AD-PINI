# AD-PINI: Anomaly-Decoupled Physics-Informed Neural Interpreter for Global Carbon Flux Prediction

**[ICML 2026 Under Review]**

This repository contains the official PyTorch implementation of the paper **"Anomaly-Decoupled Physics-Informed Neural Interpreter for Global Carbon Flux Prediction"**.

**AD-PINI** is a physics-informed deep learning framework designed to address the **ill-posedness** of inverse problems in ocean carbon flux prediction. By reformulating the learning problem in **Anomaly Space** and incorporating discretized physical evolution equations, AD-PINI achieves high spatiotemporal forecasting accuracy while decoupling and interpreting the underlying **Thermodynamic (SST)** and **Biogeochemical (DIC)** drivers.

---

## 🌟 Motivation

![Motivation](moti.jpg)
*Figure 1: Comparison of Learning Paradigms. (a) Standard Deep Learning acts as a "black-box" operator. (b) Standard PINNs face the ill-posedness problem where multiple latent physical states can yield the same observation, leading to signal aliasing. (c) AD-PINI (Ours) achieves a physically unique and interpretable solution through anomaly decoupling and explicit discrete physical constraints.*

Existing physics-informed approaches often embed governing equations directly in the raw variable space. This strategy frequently overlooks the ill-posed nature of the inverse problem, resulting in non-unique solutions and signal aliasing between physical components. AD-PINI introduces an **Anomaly-Decoupled** strategy that explicitly models the evolution increments ($\Delta$) of physical variables. This ensures physical consistency and significantly improves robustness for long-term forecasting and extreme climate events.

---

## 🚀 Methodology

![Architecture](main.pdf)
*Figure 2: Schematic Architecture of AD-PINI. The model consists of three core components: (1) Anomaly-Decoupled Neural Interpreter, (2) Discrete PDEs Prediction Network, and (3) Residual Correction Network.*

### Core Components (see `models/v4/carbon_net_v4.py`)

1.  **Anomaly-Decoupled Neural Interpreter**:
    * **Input**: Historical $pCO_2$ anomaly sequences.
    * **Function**: Uses a U-Net backbone (`AnomalyUNet`) to invert latent physical driving increments—**Thermodynamic** ($\Delta SST$) and **Non-thermodynamic/Biogeochemical** ($\Delta DIC$).
    * **Principle**: Strips away the climatological background to focus on dynamic increments, maximizing the signal-to-noise ratio.

2.  **Discrete PDEs Prediction Network**:
    * **Implementation**: `DifferentiablePhysicsLayer`.
    * **Function**: Based on Taylor expansion-derived discretized governing equations. It combines the current state, inferred increments ($\Delta SST, \Delta DIC$), and sensitivity coefficients ($\gamma_s, \gamma_d$) to explicitly calculate the physical prediction for the next time step.
    * **Constraints**: Enforces adherence to Henry's Law and carbonate chemistry equilibrium.

3.  **Residual Correction Network**:
    * **Implementation**: `ResidualCorrector`.
    * **Function**: Learns truncation errors from the physical approximation and unmodeled processes (e.g., minor alkalinity variations). Sparsity regularization is applied to prevent the network from bypassing the physical module.

---

## 🛠️ Installation & Requirements

This project is built on **Python 3.8+** and **PyTorch**.

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/anonymous/AD-PINI.git](https://github.com/anonymous/AD-PINI.git)
    cd AD-PINI
    ```

2.  **Install Dependencies**:
    ```bash
    pip install torch torchvision numpy scipy wandb matplotlib
    ```

3.  **Key Libraries**:
    * `torch >= 1.9.0`
    * `numpy`
    * `wandb` (Optional, for experiment tracking)

---

## 📂 Project Structure

```text
AD-PINI/
├── configs/
│   └── config_v4.py            # Global configuration (Hyperparameters, Paths, Constants)
├── data/
│   └── v4/
│       ├── dataset_v4.py       # Data loaders (Spatiotemporal sequence handling)
│       └── preprocessing_v4.py # Data preprocessing (Climatology removal, Normalization)
├── models/
│   └── v4/
│       ├── carbon_net_v4.py    # Core AD-PINI model architecture
│       ├── loss_v4.py          # Loss functions (State, Task, Physics, Sparsity)
│       └── __init__.py
├── utils/
│   ├── v4/
│   │   └── visualization_v4.py # Scientific visualization tools
│   ├── logger.py               # Logging utility
│   └── training.py             # Training loop helpers
├── train_v4.py                 # Main training script
├── evaluate_v4.py              # Evaluation script
├── moti.jpg                    # Motivation Figure
├── main.pdf                    # Architecture Figure
└── README.md
```
## 📈 Results

AD-PINI achieves State-of-the-Art (SOTA) performance on **Global Carbon Flux Forecasting** tasks (3/6/12-month horizons) while offering superior mechanism interpretability.

| Model              | 3-Month MSE ↓ | 6-Month MSE ↓ | 12-Month MSE ↓ | Interpretability |
|-------------------|--------------|--------------|---------------|------------------|
| FNO               | 0.4559       | 0.6910       | 0.6278        | Low              |
| ConvLSTM          | 0.1777       | 0.1863       | 0.2849        | Low              |
| EarthFormer       | 0.2215       | 0.2226       | 0.2208        | Low              |
| **AD-PINI (Ours)**| **0.0490**   | **0.0541**   | **0.0868**    | **High**         |

> **Note:** Please refer to the paper for detailed experimental results, error analysis, and visualizations.

---

## 🛡️ License

This project is released under the MIT License.

---

**Anonymous Authors**  
Submission to the 43rd International Conference on Machine Learning (ICML 2026).
