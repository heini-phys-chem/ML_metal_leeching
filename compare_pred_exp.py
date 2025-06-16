import sys
import re

import pandas as pd
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold

# Set global parameters for the plot
plt.rcParams.update({
    'font.size': 16, 'font.weight': 'normal', 'axes.labelsize': 14,
    'axes.titlesize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 12, 'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
    'grid.linewidth': 0.5, 'grid.linestyle': '--'
})

# --- Data Loading and KRR Model Components ---

def preprocess_data(df: pd.DataFrame, scaler=None, encoder=None):
    """
    Preprocesses raw data into a format suitable for the KRR model.

    This function cleans the data, handles time unit conversions, and applies
    scaling and one-hot encoding. If a scaler/encoder is provided, it uses
    them to transform the data; otherwise, it fits new ones.
    """
    all_metals = ['Co', 'Ni', 'Mn', 'Li', 'Al']
    base_params = ['Sample name', 'solvent 1', 'solvent 2', 'ratio',
                   'Temperature [degC]', 'time [min/hours]']
    
    existing_metals = [m for m in all_metals if m in df.columns]
    df = df[base_params + existing_metals].copy()
    df.dropna(subset=['Sample name'], inplace=True)

    def convert_time_to_minutes(time_val):
        time_str = str(time_val).lower().strip()
        if 'h' in time_str:
            return float(re.sub(r'[^0-9.]', '', time_str)) * 60
        return time_val

    df['time [min/hours]'] = df['time [min/hours]'].apply(convert_time_to_minutes)
    
    param_cols = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]']
    for col in param_cols:
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=param_cols, inplace=True)

    if scaler is None:
        scaler = MinMaxScaler()
        df[param_cols] = scaler.fit_transform(df[param_cols])
    else:
        df[param_cols] = scaler.transform(df[param_cols])

    df_melted = df.melt(id_vars=['Sample name'] + param_cols, value_vars=existing_metals,
                        var_name='Metal', value_name='Leaching Efficiency')
    
    if not df_melted.empty and df_melted['Leaching Efficiency'].max() > 1.0:
        df_melted['Leaching Efficiency'] /= 100.0

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        metal_encoded = encoder.fit_transform(df_melted[['Metal']])
    else:
        metal_encoded = encoder.transform(df_melted[['Metal']])
        
    metal_cols = encoder.get_feature_names_out(['Metal'])
    metal_encoded_df = pd.DataFrame(metal_encoded, columns=metal_cols)
    
    df_final = pd.concat([df_melted.reset_index(drop=True), metal_encoded_df], axis=1)
    
    final_feature_cols = param_cols + metal_cols.tolist()
    
    return df_final, scaler, encoder, final_feature_cols


def laplacian_kernel_gpu(x1: cp.ndarray, x2: cp.ndarray, sigma: float) -> cp.ndarray:
    l1_dists = cp.sum(cp.abs(x1[:, cp.newaxis, :] - x2[cp.newaxis, :, :]), axis=2)
    gamma = 1.0 / sigma
    kernel_matrix = cp.exp(-gamma * l1_dists)
    cp.cuda.Stream.null.synchronize()
    return kernel_matrix


class KRRWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, sigma=None, lambda_reg=None):
        self.sigma = sigma
        self.lambda_reg = lambda_reg
        self.alpha_ = None
        self.X_train_gpu_ = None

    def fit(self, X, y):
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
        if self.alpha_ is None: raise RuntimeError("Model not fitted!")
        X_test_gpu = cp.asarray(X)
        K_test = laplacian_kernel_gpu(self.X_train_gpu_, X_test_gpu, self.sigma)
        return cp.dot(K_test.T, self.alpha_).get()

    def _get_hyperparams(self, X_gpu, y_gpu):
        np.random.seed(667)
        sigmas, lambdas = [0.1 * 2**i for i in range(15)], [1e-3, 1e-5, 1e-7, 1e-9]
        best_mae, best_sigma, best_lambda = float('inf'), None, None
        kf = KFold(n_splits=3, shuffle=True, random_state=667)
        for sigma in tqdm(sigmas, desc="Hyperparameter Search"):
            for lambda_reg in lambdas:
                mae = 0.0
                for train_idx, val_idx in kf.split(X_gpu):
                    X_t, X_v, y_t, y_v = X_gpu[train_idx], X_gpu[val_idx], y_gpu[train_idx], y_gpu[val_idx]
                    K = laplacian_kernel_gpu(X_t, X_t, sigma) + cp.eye(len(X_t)) * lambda_reg
                    alpha = cp.linalg.solve(K, y_t)
                    K_v = laplacian_kernel_gpu(X_t, X_v, sigma)
                    preds = cp.dot(K_v.T, alpha)
                    mae += cp.mean(cp.abs(preds - y_v)).get()
                if (mae / 3) < best_mae:
                    best_mae, best_sigma, best_lambda = (mae / 3), sigma, lambda_reg
        return best_sigma, best_lambda

# --- Main Application Logic ---

def main():
    """
    Trains a model on existing data, then predicts on new experimental
    data and compares the predicted vs. actual results.
    """
    if len(sys.argv) != 3:
        print("Usage: python comparison_script.py <path_to_training_xlsx> <path_to_new_experimental_xlsx>")
        sys.exit(1)

    training_filepath = sys.argv[1]
    new_data_filepath = sys.argv[2]

    # --- 1. Load Training Data and Get Preprocessing Tools ---
    print("Loading and processing training data...")
    df_train, scaler, encoder, feature_columns = preprocess_data(pd.read_excel(training_filepath, header=1))
    X_train = df_train[feature_columns].to_numpy()
    y_train = df_train['Leaching Efficiency'].to_numpy()
    print(f"Training data has {len(df_train)} samples.")
    
    # --- 2. Train Final Model ---
    print("\nTraining final KRR model on all training data...")
    model = KRRWrapper()
    model.fit(X_train, y_train)
    print(f"Model training complete. Optimal params: sigma={model.sigma}, lambda={model.lambda_reg}")

    # --- 3. Load and Process New Experimental Data ---
    print("\nLoading and processing new experimental data for validation...")
    df_actual_processed, _, _, _ = preprocess_data(
        pd.read_excel(new_data_filepath, header=1),
        scaler=scaler,  # Use the SAME scaler from training
        encoder=encoder  # Use the SAME encoder from training
    )
    X_actual = df_actual_processed[feature_columns].to_numpy()
    
    if len(df_actual_processed) == 0:
        print("Error: No valid data rows found in the new experimental file after processing.")
        sys.exit(1)
        
    print(f"Found {len(df_actual_processed)} validation samples.")

    # --- 4. Predict on New Data ---
    print("Making predictions on new data...")
    y_predicted = model.predict(X_actual)
    df_actual_processed['Predicted Efficiency'] = np.clip(y_predicted, 0, 1)

    # --- 5. Generate Comparison Plots and Print Metrics ---
    metals_to_compare = sorted(df_actual_processed['Metal'].unique())
    n_metals = len(metals_to_compare)
    n_cols = 3
    n_rows = (n_metals + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
    axes = axes.flatten()

    print("\n--- Validation Metrics by Metal ---")
    for i, metal in enumerate(metals_to_compare):
        ax = axes[i]
        temp_df = df_actual_processed[df_actual_processed['Metal'] == metal]
        
        y_actual_metal = temp_df['Leaching Efficiency']
        y_pred_metal = temp_df['Predicted Efficiency']

        r2, mae = r2_score(y_actual_metal, y_pred_metal), mean_absolute_error(y_actual_metal, y_pred_metal)
        
        # Print metrics to terminal
        print(f"  - {metal:<5s} | R²: {r2:6.3f} | MAE: {mae*100:5.2f}%")
        
        # Plotting
        ax.scatter(y_actual_metal * 100, y_pred_metal * 100, alpha=0.7, edgecolors='k', label=f'Data Points ({len(temp_df)})')
        ax.plot([0, 100], [0, 100], 'r--', label='Ideal')
        
        ax.set_title(f'{metal}\n(R² = {r2:.2f}, MAE = {mae*100:.2f}%)')
        ax.set_xlabel('Actual Efficiency [%]')
        ax.set_ylabel('Predicted Efficiency [%]')
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        ax.legend()

    # --- 6. Calculate and Print Overall MAE ---
    y_actual_all = df_actual_processed['Leaching Efficiency']
    y_pred_all = df_actual_processed['Predicted Efficiency']
    overall_mae = mean_absolute_error(y_actual_all, y_pred_all)
    print(f"\n--- Overall Model Performance ---")
    print(f"  Overall MAE on {len(df_actual_processed)} validation points: {overall_mae*100:.2f}%")
    
    # --- 7. Finalize Plot ---
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig("prediction_vs_actual_comparison.png", dpi=300)
    print("\n✅ Comparison plot saved to 'prediction_vs_actual_comparison.png'")

if __name__ == '__main__':
    main()

