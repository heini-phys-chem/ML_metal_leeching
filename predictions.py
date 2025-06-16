import sys
import re
import itertools

import pandas as pd
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold

# --- Re-used components from previous scripts ---

def get_dataframe(filepath: str) -> tuple[pd.DataFrame, MinMaxScaler, OneHotEncoder, list]:
    """
    Loads and preprocesses data, now also returning the fitted scaler and encoder.

    This function is adapted to return the tools used for preprocessing,
    which are essential for correctly transforming new prediction data.

    Parameters:
    - filepath (str): The path to the input Excel file.

    Returns:
    - tuple containing:
        - pd.DataFrame: The fully preprocessed and cleaned dataframe.
        - MinMaxScaler: The scaler fitted on the training data.
        - OneHotEncoder: The encoder fitted on the metal types.
        - list: The final list of feature columns used for training.
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

    base_feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]']
    scaler = MinMaxScaler()
    df[base_feature_columns] = scaler.fit_transform(df[base_feature_columns])

    id_vars = ['Sample name'] + base_feature_columns
    value_vars = ['Co', 'Ni', 'Mn', 'Li', 'Al']
    df_melted = df.melt(id_vars=id_vars,
                          value_vars=value_vars,
                          var_name='Metal',
                          value_name='Leaching Efficiency')

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    metal_encoded = encoder.fit_transform(df_melted[['Metal']])
    metal_encoded_df = pd.DataFrame(metal_encoded, columns=encoder.get_feature_names_out(['Metal']))

    df_final = pd.concat([df_melted.reset_index(drop=True), metal_encoded_df], axis=1)
    
    final_feature_columns = base_feature_columns + metal_encoded_df.columns.tolist()

    return df_final, scaler, encoder, final_feature_columns


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
        if self.alpha_ is None:
            raise RuntimeError("You must call fit before predict!")
        
        X_test_gpu = cp.asarray(X)
        K_test = laplacian_kernel_gpu(self.X_train_gpu_, X_test_gpu, self.sigma)
        predictions_gpu = cp.dot(K_test.T, self.alpha_)
        return predictions_gpu.get()

    def _get_hyperparams(self, X_gpu: cp.ndarray, y_gpu: cp.ndarray) -> tuple[float, float]:
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


def main():
    """
    Main function to train a model and predict leaching efficiencies
    for a grid of unknown experimental parameters.
    """
    if len(sys.argv) < 2:
        print("Usage: python prediction_script.py <path_to_training_data_excel_file>")
        sys.exit(1)

    training_filepath = sys.argv[1]

    # --- 1. Load Training Data and Train Final Model ---
    print("Loading training data and fitting final model...")
    df_train, scaler, encoder, feature_columns = get_dataframe(training_filepath)
    
    X_train = df_train[feature_columns].to_numpy()
    y_train = df_train['Leaching Efficiency'].to_numpy()
    
    model = KRRWrapper()
    model.fit(X_train, y_train)
    print("Model training complete.")
    print(f"Using hyperparameters: sigma={model.sigma}, lambda={model.lambda_reg}")

    # --- 2. Define the Grid of New Parameters for Prediction ---
    print("\nGenerating grid of new experimental parameters...")
    # Convert all time units to minutes for consistency
    time_minutes = [60, 120, 240, 300, 360, 12*60, 18*60, 24*60]
    temps = [40, 55, 65, 75, 85, 95]
    solvent1s = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    ratios = [10, 15, 20, 25, 30]
    solvent2s = [0.0, 0.02, 0.04, 0.06, 0.08, 0.15]
    
    # Generate all combinations
    param_grid = list(itertools.product(time_minutes, temps, solvent1s, ratios, solvent2s))
    
    # Create a DataFrame from the grid
    df_pred = pd.DataFrame(param_grid, columns=['time [min/hours]', 'Temperature [degC]', 'solvent 1', 'ratio', 'solvent 2'])
    # Reorder columns to match the training scaler
    df_pred = df_pred[['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]']]

    # --- 3. Prepare the Full Prediction Dataset ---
    metals_to_predict = ['Co', 'Ni', 'Mn', 'Li', 'Al']
    full_pred_list = []

    for metal in metals_to_predict:
        df_metal_pred = df_pred.copy()
        df_metal_pred['Metal'] = metal
        full_pred_list.append(df_metal_pred)

    df_full_pred = pd.concat(full_pred_list, ignore_index=True)

    # --- 4. Scale and Encode Prediction Data ---
    # Apply the *same* scaler that was fitted on the training data
    base_feature_cols = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]']
    df_full_pred[base_feature_cols] = scaler.transform(df_full_pred[base_feature_cols])

    # Apply the *same* one-hot encoder from training
    metal_encoded_pred = encoder.transform(df_full_pred[['Metal']])
    metal_encoded_pred_df = pd.DataFrame(metal_encoded_pred, columns=encoder.get_feature_names_out(['Metal']))
    
    df_full_pred = pd.concat([df_full_pred, metal_encoded_pred_df], axis=1)

    # Ensure final feature columns match training
    X_pred = df_full_pred[feature_columns].to_numpy()

    # --- 5. Make Predictions ---
    print(f"Making predictions for {len(X_pred)} combinations...")
    predictions = model.predict(X_pred)
    
    # Clip predictions to be within a logical range [0, 1]
    predictions_clipped = np.clip(predictions, 0, 1)

    # --- 6. Format and Save Results ---
    # Create the result dataframe in the original "un-melted" format
    result_df = pd.DataFrame(param_grid, columns=['time [min/hours]', 'Temperature [degC]', 'solvent 1', 'ratio', 'solvent 2'])
    result_df_long = pd.concat([result_df] * len(metals_to_predict), ignore_index=True)
    result_df_long['Metal'] = df_full_pred['Metal']
    result_df_long['Predicted Leaching Efficiency'] = predictions_clipped

    # Pivot the table to get the desired wide format
    print("Formatting results into a wide table...")
    final_output_df = result_df_long.pivot_table(
        index=['time [min/hours]', 'Temperature [degC]', 'solvent 1', 'ratio', 'solvent 2'],
        columns='Metal',
        values='Predicted Leaching Efficiency'
    ).reset_index()

    # Save to CSV
    output_filename = "predicted_leaching_efficiencies.csv"
    final_output_df.to_csv(output_filename, index=False)
    print(f"\n✅ Predictions saved successfully to '{output_filename}'")


if __name__ == '__main__':
    main()


