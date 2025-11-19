"""
Utility functions for leaching efficiency prediction models.
"""

import pandas as pd
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error


def get_dataframe(filepath: str) -> pd.DataFrame:
    """
    Loads and preprocesses the experimental data from an Excel file.
    
    Parameters:
    - filepath (str): Path to the Excel file
    
    Returns:
    - pd.DataFrame: Preprocessed dataframe with features and targets
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
    
    return df_final, scaler, encoder


def get_features_and_target(df: pd.DataFrame) -> tuple:
    """
    Extract feature matrix, target vector, and metal labels from dataframe.
    
    Returns:
    - tuple: (X, y, metal_labels, feature_columns)
    """
    base_feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]']
    metal_feature_columns = [col for col in df.columns if col.startswith('Metal_')]
    feature_columns = base_feature_columns + metal_feature_columns
    
    X = df[feature_columns].to_numpy()
    y = df['Leaching Efficiency'].to_numpy()
    metal_labels = df['Metal'].to_numpy()
    
    return X, y, metal_labels, feature_columns


def transform_test_data(filepath: str, scaler, encoder) -> tuple:
    """
    Transform test data using existing scaler and encoder.
    
    Parameters:
    - filepath (str): Path to test data Excel file
    - scaler: Fitted MinMaxScaler from training data
    - encoder: Fitted OneHotEncoder from training data
    
    Returns:
    - tuple: (X_test, y_test, metal_labels_test, df_test)
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
    df[feature_columns] = scaler.transform(df[feature_columns])
    
    id_vars = ['Sample name'] + feature_columns
    value_vars = ['Co', 'Ni', 'Mn', 'Li', 'Al']
    df_melted = df.melt(id_vars=id_vars,
                         value_vars=value_vars,
                         var_name='Metal',
                         value_name='Leaching Efficiency')
    
    metal_encoded = encoder.transform(df_melted[['Metal']])
    metal_encoded_df = pd.DataFrame(metal_encoded, columns=encoder.get_feature_names_out(['Metal']))
    
    df_final = pd.concat([df_melted.reset_index(drop=True), metal_encoded_df], axis=1)
    
    X, y, metal_labels, _ = get_features_and_target(df_final)
    
    return X, y, metal_labels, df_final


def laplacian_kernel_gpu(x1: cp.ndarray, x2: cp.ndarray, sigma: float) -> cp.ndarray:
    """
    Computes the Laplacian kernel matrix on the GPU using CuPy.
    
    Parameters:
    - x1: CuPy array of shape (n_samples1, n_features)
    - x2: CuPy array of shape (n_samples2, n_features)
    - sigma: Kernel bandwidth parameter
    
    Returns:
    - cp.ndarray: Kernel matrix of shape (n_samples1, n_samples2)
    """
    l1_dists = cp.sum(cp.abs(x1[:, cp.newaxis, :] - x2[cp.newaxis, :, :]), axis=2)
    gamma = 1.0 / sigma
    kernel_matrix = cp.exp(-gamma * l1_dists)
    cp.cuda.Stream.null.synchronize()
    return kernel_matrix


def create_subplot_scatterplot(ax, actual_values: np.ndarray, predicted_values: np.ndarray,
                                metal_labels: np.ndarray, mae: float, r2: float, title: str):
    """
    Creates a scatter plot on given axes.
    
    Parameters:
    - ax: Matplotlib axes object
    - actual_values: Actual target values
    - predicted_values: Predicted target values
    - metal_labels: Metal type for each sample
    - mae: Mean Absolute Error
    - r2: R-squared score
    - title: Plot title
    """
    unique_metals = sorted(list(set(metal_labels)))
    colours = plt.cm.viridis(np.linspace(0, 1, len(unique_metals)))
    colour_map = dict(zip(unique_metals, colours))
    
    for metal in unique_metals:
        idx = np.array(metal_labels) == metal
        ax.scatter(actual_values[idx] * 100, predicted_values[idx] * 100,
                   color=colour_map[metal], label=metal, s=100, alpha=0.7, edgecolors='k')
    
    min_val = min(actual_values.min(), predicted_values.min()) * 100
    max_val = max(actual_values.max(), predicted_values.max()) * 100
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')
    
    ax.set_xlabel('Actual [%]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted [%]', fontsize=14, fontweight='bold')
    ax.set_title(f'{title}\nMAE: {mae:.2f}% | R²: {r2:.2f}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_aspect('equal', adjustable='box')


def setup_plot_style():
    """Set global matplotlib parameters."""
    plt.rcParams.update({
        'font.size': 12,
        'font.weight': 'bold',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'grid.linewidth': 0.5,
        'grid.linestyle': '--'
    })
