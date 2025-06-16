import sys
import re
from copy import deepcopy

import pandas as pd
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
from adjustText import adjust_text

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# Set global parameters for the plot using rcParams
plt.rcParams.update({
    'font.size': 22,
    'font.weight': 'bold',
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'grid.linewidth': 0.5,
    'grid.linestyle': '--'
})


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

    # --- Data Selection and Cleaning ---
    selected_columns = ['Sample name', 'solvent 1', 'solvent 2', 'ratio',
                        'Temperature [degC]', 'time [min/hours]', 'Co', 'Ni', 'Mn', 'Li']
    df = df[selected_columns].copy()
    df.dropna(subset=['Sample name'], inplace=True)

    # Consolidate filtering into a single regex for efficiency
    exclusion_patterns = ['CAM', 'LS-CRM-SP', 'LS-AS1', '-F1', '-F2', '-F3', 'LS-AS00']
    regex_pattern = '|'.join(exclusion_patterns)
    df = df[~df['Sample name'].str.contains(regex_pattern, case=False, na=False)]
    df.reset_index(drop=True, inplace=True)

    # --- Type Conversion ---
    # Clean and convert columns to numeric, coercing errors to NaN
    for col in ['solvent 1', 'solvent 2', 'time [min/hours]']:
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows that may have become NaN during conversion
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Feature Scaling ---
    feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]']
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # --- Data Transformation ---
    # Melt the dataframe to have one row per metal per sample
    id_vars = ['Sample name'] + feature_columns
    value_vars = ['Co', 'Ni', 'Mn', 'Li']
    df_melted = df.melt(id_vars=id_vars,
                          value_vars=value_vars,
                          var_name='Metal',
                          value_name='Leaching Efficiency')

    # --- Feature Engineering (One-Hot Encoding) ---
    encoder = OneHotEncoder(sparse_output=False)
    metal_encoded = encoder.fit_transform(df_melted[['Metal']])
    metal_encoded_df = pd.DataFrame(metal_encoded, columns=encoder.get_feature_names_out(['Metal']))

    # Combine one-hot encoded columns back with the melted dataframe
    df_final = pd.concat([df_melted.reset_index(drop=True), metal_encoded_df], axis=1)

    return df_final


def laplacian_kernel_gpu(x1: cp.ndarray, x2: cp.ndarray, sigma: float) -> cp.ndarray:
    """
    Computes the Laplacian kernel matrix on the GPU using CuPy.

    Parameters:
    - x1 (cp.ndarray): A CuPy array of shape (n_samples1, n_features).
    - x2 (cp.ndarray): A CuPy array of shape (n_samples2, n_features).
    - sigma (float): The bandwidth parameter of the kernel.

    Returns:
    - cp.ndarray: The resulting kernel matrix of shape (n_samples1, n_samples2).
    """
    # Broadcasting is used for an efficient, element-wise distance calculation
    l1_dists = cp.sum(cp.abs(x1[:, cp.newaxis, :] - x2[cp.newaxis, :, :]), axis=2)
    gamma = 1.0 / sigma
    kernel_matrix = cp.exp(-gamma * l1_dists)
    
    # Ensures the computation is complete before the matrix is used elsewhere.
    cp.cuda.Stream.null.synchronize()

    return kernel_matrix


def get_hyperparams(X_gpu: cp.ndarray, y_gpu: cp.ndarray) -> tuple[float, float]:
    """
    Finds the optimal hyperparameters (sigma, lambda) using 3-fold cross-validation.

    This function performs a grid search over a predefined range of sigma and
    lambda values to find the combination that yields the lowest Mean
    Absolute Error (MAE) on the validation sets.

    Parameters:
    - X_gpu (cp.ndarray): The feature matrix on the GPU.
    - y_gpu (cp.ndarray): The target vector on the GPU.

    Returns:
    - tuple[float, float]: A tuple containing the best sigma and best lambda.
    """
    np.random.seed(667)
    sigmas = [0.01 * 2**i for i in range(20)]
    lambdas = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]

    best_mae = float('inf')
    best_sigma = None
    best_lambda = None

    kf = KFold(n_splits=3, shuffle=True, random_state=667)

    for sigma in sigmas:
        for lambda_reg in lambdas:
            current_mae = 0.0
            for train_idx, val_idx in kf.split(X_gpu):
                X_train, X_val = X_gpu[train_idx], X_gpu[val_idx]
                y_train, y_val = y_gpu[train_idx], y_gpu[val_idx]

                # --- Training ---
                K_train = laplacian_kernel_gpu(X_train, X_train, sigma)
                # Add regularization directly on the GPU
                C = K_train + cp.eye(len(X_train)) * lambda_reg
                alpha = cp.linalg.solve(C, y_train)

                # --- Prediction ---
                K_val = laplacian_kernel_gpu(X_train, X_val, sigma)
                predictions_gpu = cp.dot(K_val.T, alpha)
                
                # Calculate error (only bring necessary data to CPU)
                error = cp.mean(cp.abs(predictions_gpu - y_val))
                current_mae += error.get()  # .get() moves scalar result to CPU

            avg_mae = current_mae / kf.get_n_splits()

            if avg_mae < best_mae:
                best_mae = avg_mae
                best_sigma = sigma
                best_lambda = lambda_reg

    return best_sigma, best_lambda


def get_predictions(X_train_gpu: cp.ndarray, X_test_gpu: cp.ndarray, y_train_gpu: cp.ndarray,
                      sigma: float, lambda_reg: float) -> np.ndarray:
    """
    Trains a KRR model and makes predictions for the test set.

    Parameters:
    - X_train_gpu (cp.ndarray): The training feature matrix on the GPU.
    - X_test_gpu (cp.ndarray): The test feature matrix on the GPU.
    - y_train_gpu (cp.ndarray): The training target vector on the GPU.
    - sigma (float): The optimised kernel bandwidth.
    - lambda_reg (float): The optimised regularization parameter.

    Returns:
    - np.ndarray: An array of predictions for the test set.
    """
    # --- Training ---
    K_train = laplacian_kernel_gpu(X_train_gpu, X_train_gpu, sigma)
    C = K_train + cp.eye(K_train.shape[0]) * lambda_reg
    alpha = cp.linalg.solve(C, y_train_gpu)

    # --- Prediction ---
    K_test = laplacian_kernel_gpu(X_train_gpu, X_test_gpu, sigma)
    predictions_gpu = cp.dot(K_test.T, alpha)

    # Return predictions as a NumPy array
    return predictions_gpu.get()


def run_loocv_krr(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Performs a full Leave-One-Out Cross-Validation for the KRR model.

    This function iterates through the entire dataset, performing hyperparameter
    optimisation and prediction for each sample.

    Parameters:
    - X (np.ndarray): The full feature matrix.
    - y (np.ndarray): The full target vector.

    Returns:
    - tuple: Contains the final predictions, the actual values, the mean
             MAE, and the mean null model MAE.
    """
    predictions_all = []
    actuals_all = []
    maes = []
    null_maes = []

    # Using tqdm for a progress bar
    for i in tqdm(range(X.shape[0]), desc="Running LOOCV"):
        # Split data into training and test sets for this fold
        X_train, X_test = np.vstack((X[:i], X[i+1:])), X[i:i+1]
        y_train, y_test = np.concatenate((y[:i], y[i+1:])), y[i]

        # --- Move data to GPU ---
        X_train_gpu = cp.asarray(X_train)
        y_train_gpu = cp.asarray(y_train)
        X_test_gpu = cp.asarray(X_test)

        # --- Hyperparameter Optimisation ---
        best_sigma, best_lambda = get_hyperparams(X_train_gpu, y_train_gpu)

        # --- Final Prediction for this Fold ---
        prediction = get_predictions(X_train_gpu, X_test_gpu, y_train_gpu, best_sigma, best_lambda)

        # Clip prediction if it exceeds logical maximum (100% efficiency)
        prediction_clipped = min(prediction[0], 1.0)
        
        # --- Store results ---
        predictions_all.append(prediction_clipped)
        actuals_all.append(y_test)
        maes.append(np.abs(prediction_clipped - y_test))
        null_maes.append(np.abs(np.mean(y_train) - y_test))

    return (np.array(predictions_all), np.array(actuals_all),
            np.mean(maes), np.mean(null_maes))


def create_scatterplot(actual_values: np.ndarray, predicted_values: np.ndarray,
                       metal_labels: np.ndarray, mae: float, null_mae: float, ax: plt.Axes):
    """
    Generates a scatter plot of actual vs. predicted values.

    Parameters:
    - actual_values (np.ndarray): The true target values.
    - predicted_values (np.ndarray): The model's predictions.
    - metal_labels (np.ndarray): The corresponding metal for each data point.
    - mae (float): The Mean Absolute Error of the model.
    - null_mae (float): The Mean Absolute Error of the null model.
    - ax (plt.Axes): The matplotlib axes object on which to draw the plot.
    """
    unique_metals = sorted(list(set(metal_labels)))
    colours = plt.cm.viridis(np.linspace(0, 1, len(unique_metals)))
    colour_map = dict(zip(unique_metals, colours))

    for metal in unique_metals:
        idx = np.array(metal_labels) == metal
        ax.scatter(actual_values[idx] * 100, predicted_values[idx] * 100,
                   color=colour_map[metal], label=metal, s=150, alpha=0.8, edgecolors='k')

    # Plot the y=x line for reference
    min_val = min(actual_values.min(), predicted_values.min()) * 100
    max_val = max(actual_values.max(), predicted_values.max()) * 100
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')

    ax.set_xlabel('Actual Leaching Efficiency [%]')
    ax.set_ylabel('Predicted Leaching Efficiency [%]')

    r2 = r2_score(actual_values, predicted_values)
    title_text = (f'MAE: {mae:.2f}% | Null MAE: {null_mae:.2f}% | '
                  r'R$^2$: ' f'{r2:.2f}')
    ax.set_title(title_text)

    ax.grid(True)
    ax.legend(loc='upper left')
    ax.set_aspect('equal', adjustable='box')


def main():
    """
    Main function to execute the KRR model training and evaluation workflow.
    """
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <path_to_excel_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    # --- 1. Load and Prepare Data ---
    df = get_dataframe(filepath)

    feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]',
                       'time [min/hours]', 'Metal_Co', 'Metal_Li', 'Metal_Mn', 'Metal_Ni']
    target_column = 'Leaching Efficiency'

    X = df[feature_columns].to_numpy()
    y = df[target_column].to_numpy()
    metal_labels = df['Metal'].to_numpy()

    # --- 2. Run Model Training and Validation ---
    predictions, actuals, mae, null_mae = run_loocv_krr(X, y)

    print(f"\nFinal MAE: {mae*100:.2f}%")
    print(f"Null Model MAE: {null_mae*100:.2f}%")

    # --- 3. Visualise Results ---
    fig, ax = plt.subplots(figsize=(10, 10))
    create_scatterplot(actuals, predictions, metal_labels, mae*100, null_mae*100, ax)
    
    plt.tight_layout()
    fig.savefig("leaching_efficiency_prediction.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
