import sys
import re

import pandas as pd
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, learning_curve

# Set global parameters for the plot using rcParams
plt.rcParams.update({
    'font.size': 18,
    'font.weight': 'normal',
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'grid.linewidth': 0.5,
    'grid.linestyle': '--'
})

# --- Re-used functions for consistency ---

def get_dataframe(filepath: str) -> pd.DataFrame:
    """
    Loads and preprocesses the experimental data from an Excel file.
    """
    df = pd.read_excel(filepath, header=1)
    
    selected_columns = ['Sample name', 'solvent 1', 'solvent 2', 'ratio',
                        'Temperature [degC]', 'time [min/hours]', 'Co', 'Ni', 'Mn', 'Li', 'Al']
    df = df[selected_columns].copy()
    df.dropna(subset=['Sample name'], inplace=True)

    exclusion_patterns = ['CAM', 'LS-CRM-SP', 'LS-AS1', '-F1', '-F2', '-F3', 'LS-AS00']
    regex_pattern = '|'.join(exclusion_patterns)
    df = df[~df['Sample name'].str.contains(regex_pattern, case=False, na=False)]
    df.reset_index(drop=True, inplace=True)

    for col in ['solvent 1', 'solvent 2', 'time [min/hours]']:
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]']
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    id_vars = ['Sample name'] + feature_columns
    value_vars = ['Co', 'Ni', 'Mn', 'Li', 'Al']
    df_melted = df.melt(id_vars=id_vars,
                          value_vars=value_vars,
                          var_name='Metal',
                          value_name='Leaching Efficiency')

    encoder = OneHotEncoder(sparse_output=False)
    metal_encoded = encoder.fit_transform(df_melted[['Metal']])
    metal_encoded_df = pd.DataFrame(metal_encoded, columns=encoder.get_feature_names_out(['Metal']))

    df_final = pd.concat([df_melted.reset_index(drop=True), metal_encoded_df], axis=1)
    
    return df_final


def laplacian_kernel_gpu(x1: cp.ndarray, x2: cp.ndarray, sigma: float) -> cp.ndarray:
    """
    Computes the Laplacian kernel matrix on the GPU using CuPy.
    """
    l1_dists = cp.sum(cp.abs(x1[:, cp.newaxis, :] - x2[cp.newaxis, :, :]), axis=2)
    gamma = 1.0 / sigma
    kernel_matrix = cp.exp(-gamma * l1_dists)
    cp.cuda.Stream.null.synchronize()
    return kernel_matrix

# --- KRR Wrapper for Scikit-learn Compatibility ---

class KRRWrapper(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper for the custom GPU-accelerated KRR model.
    """
    def __init__(self, sigma=None, lambda_reg=None):
        self.sigma = sigma
        self.lambda_reg = lambda_reg
        self.alpha_ = None
        self.X_train_gpu_ = None

    def fit(self, X, y):
        """
        Fits the KRR model. Finds hyperparameters if not provided, then trains.
        """
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)

        if self.sigma is None or self.lambda_reg is None:
            self.sigma, self.lambda_reg = self._get_hyperparams(X_gpu, y_gpu)

        self.X_train_gpu_ = X_gpu
        K_train = laplacian_kernel_gpu(self.X_train_gpu_, self.X_train_gpu_, self.sigma)
        C = K_train + cp.eye(K_train.shape[0]) * self.lambda_reg
        self.alpha_ = cp.linalg.solve(C, y_gpu)
        return self

    def predict(self, X):
        """
        Makes predictions with the fitted model.
        """
        if self.alpha_ is None:
            raise RuntimeError("You must call fit before predict!")
        
        X_test_gpu = cp.asarray(X)
        K_test = laplacian_kernel_gpu(self.X_train_gpu_, X_test_gpu, self.sigma)
        predictions_gpu = cp.dot(K_test.T, self.alpha_)
        return predictions_gpu.get()

    def _get_hyperparams(self, X_gpu: cp.ndarray, y_gpu: cp.ndarray) -> tuple[float, float]:
        """
        Internal method for hyperparameter search.
        """
        # This is a simplified search for the learning curve generation.
        # A full grid search for each point on the curve would be very slow.
        # We find the best parameters once on the full dataset.
        np.random.seed(667)
        sigmas = [0.1 * 2**i for i in range(15)]
        lambdas = [1e-3, 1e-5, 1e-7, 1e-9]
        best_mae = float('inf')
        best_sigma, best_lambda = None, None
        kf = KFold(n_splits=3, shuffle=True, random_state=667)
        
        for sigma in sigmas:
            for lambda_reg in lambdas:
                current_mae = 0.0
                for train_idx, val_idx in kf.split(X_gpu):
                    X_train, X_val = X_gpu[train_idx], X_gpu[val_idx]
                    y_train, y_val = y_gpu[train_idx], y_gpu[val_idx]
                    K_train = laplacian_kernel_gpu(X_train, X_train, sigma)
                    C = K_train + cp.eye(len(X_train)) * lambda_reg
                    alpha = cp.linalg.solve(C, y_train)
                    K_val = laplacian_kernel_gpu(X_train, X_val, sigma)
                    predictions_gpu = cp.dot(K_val.T, alpha)
                    error = cp.mean(cp.abs(predictions_gpu - y_val))
                    current_mae += error.get()
                avg_mae = current_mae / kf.get_n_splits()
                if avg_mae < best_mae:
                    best_mae, best_sigma, best_lambda = avg_mae, sigma, lambda_reg
        return best_sigma, best_lambda

# --- New function for plotting the learning curve ---

def plot_learning_curve(train_sizes, train_scores, test_scores):
    """
    Creates and saves a plot of the learning curve using MAE on a log-log scale.

    Parameters:
    - train_sizes (array): The number of training examples used.
    - train_scores (array): The negative MAE scores on the training sets.
    - test_scores (array): The negative MAE scores on the validation sets.
    """
    # Convert negative MAE to positive MAE and scale to percentage
    train_scores_mean = -np.mean(train_scores, axis=1) * 100
    train_scores_std = np.std(train_scores, axis=1) * 100
    test_scores_mean = -np.mean(test_scores, axis=1) * 100
    test_scores_std = np.std(test_scores, axis=1) * 100

    fig, ax = plt.subplots(figsize=(5, 7)) # Reduced width
    
    # Plot the standard deviation as a shaded area
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="C0")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="C1")
    
    # Plot the mean scores with distinct styles
    ax.plot(train_sizes, train_scores_mean, 'o-', color="C0", label="Training MAE")
    ax.plot(train_sizes, test_scores_mean, 's--', color="C1", label="Cross-validation MAE")
    
    ax.axhline(y=22.51, color='C3', linestyle=':', label='Null Model MAE')

    # Set plot scales and labels
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$\it{N}$") # Italic N for x-axis label
    ax.set_ylabel("MAE [%]")

    # Set custom ticks and formatters
    xticks = [20, 40, 80, 120]
    yticks = [0.1, 0.5, 1, 2, 4, 8, 16, 32] # Added 0.5
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    
    ax.legend(loc="best")
    ax.grid(True, which="both", ls="--") # Grid for both major and minor ticks
    
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=300)
    print("\nLearning curve plot saved to 'learning_curve.png'")
    # The following line is commented out to prevent errors in non-interactive environments.
    # plt.show()


def main():
    """
    Main function to execute the learning curve generation.
    """
    if len(sys.argv) < 2:
        print("Usage: python learning_curve_script.py <path_to_excel_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    # --- 1. Load and Prepare Data ---
    print("Loading and preparing data...")
    df = get_dataframe(filepath)

    base_feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]']
    metal_feature_columns = [col for col in df.columns if col.startswith('Metal_')]
    feature_columns = base_feature_columns + metal_feature_columns
    
    target_column = 'Leaching Efficiency'

    X = df[feature_columns].to_numpy()
    y = df[target_column].to_numpy()

    # --- 2. Generate Learning Curve Data ---
    # First, find the best hyperparameters on the full dataset
    print("Finding optimal hyperparameters for the model...")
    temp_model = KRRWrapper()
    best_sigma, best_lambda = temp_model._get_hyperparams(cp.asarray(X), cp.asarray(y))
    print(f"\nUsing fixed hyperparameters for learning curve: sigma={best_sigma}, lambda={best_lambda}")

    # Create a new model instance with the optimal fixed parameters
    estimator = KRRWrapper(sigma=best_sigma, lambda_reg=best_lambda)

    # Define cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=667)
    
    # Define the specific training set sizes to use for the curve
    train_sizes = np.array([20, 40, 80, 120])

    print("\nGenerating learning curve data (using MAE)...")
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=train_sizes, scoring="neg_mean_absolute_error"
    )

    # --- 3. Plot the Learning Curve ---
    plot_learning_curve(train_sizes_abs, train_scores, test_scores)


if __name__ == '__main__':
    main()

