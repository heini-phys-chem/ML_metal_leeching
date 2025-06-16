import sys
import re

import pandas as pd
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance

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

# --- Re-used functions from the main script for consistency ---

def get_dataframe(filepath: str) -> pd.DataFrame:
    """
    Loads and preprocesses the experimental data from an Excel file.

    This function reads a specified Excel file, cleans the data by filtering
    out irrelevant samples, converts columns to their proper numeric types,
    scales numerical features to a [0, 1] range, and transforms the
    dataframe into a long format suitable for machine learning, including
    one-hot encoding of the metal types.

    Parameters:
    - filepath (str): The path to the input Excel file.

    Returns:
    - pd.DataFrame: A fully preprocessed and cleaned dataframe ready for model training.
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

# --- New KRR Wrapper for Scikit-learn Compatibility ---

class KRRWrapper(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper for the custom GPU-accelerated KRR model.
    
    This class encapsulates the hyperparameter search, training, and prediction logic,
    allowing the custom model to be used with standard scikit-learn tools like
    permutation_importance.
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
            print("Finding optimal hyperparameters...")
            self.sigma, self.lambda_reg = self._get_hyperparams(X_gpu, y_gpu)
            print(f"\nOptimal hyperparameters found: sigma={self.sigma}, lambda={self.lambda_reg}")

        # Train the final model
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
        np.random.seed(667)
        sigmas = [0.01 * 2**i for i in range(20)]
        lambdas = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]
        best_mae = float('inf')
        best_sigma, best_lambda = None, None
        kf = KFold(n_splits=3, shuffle=True, random_state=667)
        
        for sigma in tqdm(sigmas, desc="Sigma Search"):
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


def plot_feature_importance(result, feature_names):
    """
    Creates and saves a bar chart of feature importances with standard deviations.

    Parameters:
    - result: The result object from sklearn.inspection.permutation_importance.
    - feature_names (list): The list of feature names.
    """
    sorted_idx = result.importances_mean.argsort()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(np.array(feature_names)[sorted_idx], result.importances_mean[sorted_idx], 
            xerr=result.importances_std[sorted_idx], align='center', color='skyblue',
            capsize=5)
    ax.set_xlabel('Permutation Importance (Drop in R² Score)')
    ax.set_title('Feature Importance Analysis (10 repeats)')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("\nFeature importance plot saved to 'feature_importance.png'")
    # The following line is commented out to prevent errors in non-interactive environments.
    # plt.show()


def main():
    """
    Main function to execute the feature importance analysis.
    """
    if len(sys.argv) < 2:
        print("Usage: python feature_importance_script.py <path_to_excel_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    # --- 1. Load and Prepare Data ---
    df = get_dataframe(filepath)

    base_feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]']
    metal_feature_columns = [col for col in df.columns if col.startswith('Metal_')]
    feature_columns = base_feature_columns + metal_feature_columns
    
    target_column = 'Leaching Efficiency'

    X = df[feature_columns].to_numpy()
    y = df[target_column].to_numpy()

    # --- 2. Train Model and Calculate Importance ---
    model = KRRWrapper()
    
    # The fit method will automatically find hyperparameters and train the model
    model.fit(X, y)

    print("\nCalculating permutation importance with scikit-learn...")
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=667, n_jobs=-1
    )

    # --- 3. Visualise Results ---
    sorted_importances = result.importances_mean.argsort()[::-1]

    print("\n--- Feature Importance Results ---")
    for i in sorted_importances:
        print(f"{feature_columns[i]:<25}"
              f"Importance: {result.importances_mean[i]:.4f} "
              f" +/- {result.importances_std[i]:.4f}")
    
    plot_feature_importance(result, feature_columns)


if __name__ == '__main__':
    main()

