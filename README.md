# KRR Model for Predicting Leaching Efficiency

This project uses a GPU-accelerated Kernel Ridge Regression (KRR) model to analyse and predict the leaching efficiency of various metals under different experimental conditions. It includes a suite of Python scripts for model training, validation, feature analysis, and prediction.

## Features

* **Model Training & Validation:** Trains a KRR model using Leave-One-Out Cross-Validation (LOOCV) and evaluates its performance.
* **Feature Importance:** Uses permutation importance to determine the influence of each experimental parameter on the model's predictions.
* **Learning Curve Analysis:** Generates learning curves to diagnose model performance (bias vs. variance) as the training set size increases.
* **Prediction on New Data:** Predicts leaching efficiencies for a comprehensive grid of unseen experimental parameters.
* **Model Validation:** Compares model predictions against new, real-world experimental results to validate its accuracy.
* **Automated Environment Setup:** Includes a `Makefile` for one-step creation and setup of the required Conda environment.

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

* **Conda:** An open-source package and environment management system. (e.g., [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda).
* **NVIDIA GPU:** A CUDA-compatible NVIDIA graphics card.
* **NVIDIA Drivers:** The appropriate drivers for your GPU.
* **CUDA Toolkit (System-Level):** As discovered during our troubleshooting, the project requires the CUDA toolkit to be installed at the system level for stability. On Debian/Ubuntu systems, this can be done with:
    ```bash
    wget [https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb)
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-3 # Or your required version
    ```

---

## Installation

This project uses a `Makefile` to automate the setup process.

1.  **Clone the repository** or ensure all the project files (`.py` scripts, `Makefile`, `requirements.txt`) are in the same directory.
2.  **Open your terminal** in the project directory.
3.  **Run the installation command:**
    ```bash
    make install
    ```
    This command will:
    * Create a new Conda environment named `m_leeching`.
    * Install all the required Python packages and CUDA libraries (`cupy`, `pandas`, `scikit-learn`, etc.) using the most robust Conda-based method.
4.  **Activate the environment** to use the scripts:
    ```bash
    conda activate m_leeching
    ```

---
