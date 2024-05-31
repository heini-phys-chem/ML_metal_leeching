import sys

import pandas as pd
import numpy as np

from copy import deepcopy
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import qml
from qml.kernels import gaussian_kernel, laplacian_kernel
from qml.math import svd_solve

import qml2
from qml2.kernels import laplacian_kernel
import cupy as cp

import matplotlib.pyplot as plt
from adjustText import adjust_text

from sklearn.preprocessing import OneHotEncoder
from matplotlib.ticker import FixedLocator, FixedFormatter

import re

SMILES = {
    "acid"    : "[H][C@@]1(OC(=O)C(O)=C1O)[C@@H](O)CO",
    "glycine" : "C(C(=O)O)N",
          }

plt.rcParams.update({
    'font.size': 22,      # Set global font size
    'font.weight': 'bold', # Make all text bold
    'axes.labelsize': 22,  # Set font size for axis labels
    'axes.titlesize': 24,  # Set font size for titles
    'xtick.labelsize': 20, # Set font size for x-tick labels
    'ytick.labelsize': 20, # Set font size for y-tick labels
    'legend.fontsize': 18, # Set font size for legend
    'axes.titleweight': 'bold', # Title weight
    'axes.labelweight': 'bold', # Axis label weight
    'grid.linewidth': 0.5,
    'grid.linestyle': '--'
})

def get_dataframe(f):
    df = pd.read_excel(f, header=1)

    selected_columns = ['Sample name', 'solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]', 'Co', 'Ni', 'Mn', 'Li']

    df = df[selected_columns]
    df = df.dropna(subset=['Sample name'])
    df = df[~df['Sample name'].str.contains('CAM', case=False)]
    df = df[~df['Sample name'].str.contains('LS-CRM-SP', case=False)]
    df = df[~df['Sample name'].str.contains('LS-AS1', case=False)]
    df = df[~df['Sample name'].str.contains('-F1', case=False)]
    df = df[~df['Sample name'].str.contains('-F2', case=False)]
    df = df[~df['Sample name'].str.contains('-F3', case=False)]

    df = df[~df['Sample name'].str.contains('LS-AS00', case=False)]

    df.reset_index(drop=True, inplace=True)

    df['solvent 1'] = df['solvent 1'].astype(str).str.replace('[^\d.]', '', regex=True)
    df['solvent 1'] = pd.to_numeric(df['solvent 1'], errors='coerce')
    df['solvent 2'] = df['solvent 2'].astype(str).str.replace('[^\d.]', '', regex=True)
    df['solvent 2'] = pd.to_numeric(df['solvent 2'], errors='coerce')
    df['time [min/hours]'] = df['time [min/hours]'].astype(str).str.replace('[^\d.]', '', regex=True)
    df['time [min/hours]'] = pd.to_numeric(df['time [min/hours]'], errors='coerce')

    #Scale each feature column separately
    feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]']
    scaler = MinMaxScaler()
    for column in feature_columns:
        df[column] = scaler.fit_transform(df[[column]])
    print(df)
    # Melt the dataframe to have one row per metal type per original sample
    df_melted = df.melt(id_vars=['Sample name', 'solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]'],
                        value_vars=['Co', 'Ni', 'Mn', 'Li'],
                        var_name='Metal', value_name='Leaching Efficiency')

    # One-hot encode the metal column
    encoder = OneHotEncoder(sparse=False)
    metal_encoded = encoder.fit_transform(df_melted[['Metal']])
    metal_encoded_df = pd.DataFrame(metal_encoded, columns=encoder.get_feature_names_out(['Metal']))

    # Combine the one-hot encoded columns back with the original dataframe
    df_final = pd.concat([df_melted, metal_encoded_df], axis=1)

    print(df_final)

    return df_final


def get_hyperparams(X, y):
    np.random.seed(667)

    sigmas  = [0.01 * 2**i for i in range(30)]
    lambdas = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]

    best_MAE = float('inf')
    best_sigma = None
    best_lambda = None

    X = cp.asarray(X)
    y = cp.asarray(y)

    # 3-Fold Cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=667)

    for sigma in sigmas:
        for llambda in lambdas:
            avg_MAE = 0
            for train_index, val_index in kf.split(X):
                # Splitting the data into training and validation sets
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                K_train = laplacian_kernel_gpu(X_train, X_train, sigma)
                K_val = laplacian_kernel_gpu(X_train, X_val, sigma)

                K_train = K_train.get()
                C = K_train + np.eye(len(X_train)) * llambda
                C = cp.asarray(C)
                alpha = cp.linalg.solve(C, y_train)

                predictions = cp.dot(K_val.T, alpha)
                predictions = predictions.get()
                y_val = y_val.get()
                avg_MAE += np.mean(np.abs(predictions - y_val)) / 3  # Dividing by 3 for average

            if avg_MAE < best_MAE:
                best_MAE = avg_MAE
                best_sigma = sigma
                best_lambda = llambda

    return best_sigma, best_lambda


def get_predictions(X_train, X_test, y_train, sigma, llambda):

    X_train = cp.asarray(X_train)
    X_test  = cp.asarray(X_test)
    y_train = cp.asarray(y_train)

    K_train_gpu = laplacian_kernel_gpu(X_train, X_train, sigma)
    K_test_gpu  = laplacian_kernel_gpu(X_train, X_test, sigma)

    K_train = K_train_gpu.get()
    K_test  = K_test_gpu.get()

    C = deepcopy(K_train)
    C += np.eye(C.shape[0]) * llambda

    C = cp.asarray(C)

    alpha = cp.linalg.solve(C, y_train)

    predictions = cp.dot(K_test_gpu.T, alpha)
    predictions = predictions.get()

    return predictions

def get_KRR(X_train, X_test, y_train, y_test):
    np.random.seed(667)

    best_sigma, best_lambda = get_hyperparams(X_train, y_train)

#    best_sigma = 5.12
#    best_lambda = 1e-4

    # Make predictions
    predictions = get_predictions(X_train, X_test, y_train, best_sigma, best_lambda)

    return predictions

def laplacian_kernel_gpu(X_train, X_test, sigma):
    """
    Compute the Laplacian kernel between two sets of vectors.

    Parameters:
    - X_train: CuPy ndarray of shape (n_samples_X_train, n_features), training set vectors.
    - X_test: CuPy ndarray of shape (n_samples_X_test, n_features), test set vectors.
    - sigma: float, the bandwidth of the Laplacian kernel.

    Returns:
    - K: CuPy ndarray of shape (n_samples_X_train, n_samples_X_test), the Laplacian kernel matrix.
    """
    gamma = 1.0 / sigma

    # Compute the L1 norms (Manhattan distance)
    l1_dists = cp.sum(cp.abs(X_train[:, cp.newaxis, :] - X_test[cp.newaxis, :, :]), axis=2)

    # Compute the Laplacian kernel matrix
    K = cp.exp(-gamma * l1_dists)
    cp.cuda.Stream.null.synchronize()

    return K


def get_learning_curve_data(X, Y, test_indices, n_values):
    results = []
    for n in tqdm(n_values, desc="Run over N", leave=False):
        maes = []
        for i in range(5):  # Run 5 times for each n to get a good average
            X_train, X_test = np.delete(X, test_indices, axis=0), X[test_indices]
            y_train, y_test = np.delete(Y, test_indices, axis=0), Y[test_indices]
            indices = np.random.choice(len(X_train), n, replace=False)
            X_train_subset = X_train[indices]
            y_train_subset = y_train[indices]
            predictions = get_KRR(X_train_subset, X_test, y_train_subset, y_test)
            maes.append(np.mean(np.abs(predictions - y_test))*100.)
        results.append((n, np.mean(maes)))
    return results

def plot_learning_curve(results):
    ns, errors = zip(*results)
    plt.figure(figsize=(6, 8))
    plt.loglog(ns, errors, marker='o')

    # Custom x-axis labels
    #plt.xticks(ns, labels=[str(n) for n in ns])
    x_ticks = [30, 60, 160]
#    plt.xticks(x_ticks, labels=[str(x) for x in x_ticks])
    # Custom x-axis labels
    plt.gca().xaxis.set_major_locator(FixedLocator(x_ticks))
    plt.gca().xaxis.set_major_formatter(FixedFormatter([str(n) for n in x_ticks]))

    # Custom y-axis labels
    y_ticks = [3, 4, 5, 6, 7, 8, 9]
    plt.yticks(y_ticks, labels=[str(y) for y in y_ticks])


    # Set custom ticks only
    #plt.gca().set_xticks(ns)
    #plt.gca().set_yticks(y_ticks)

    plt.xlabel(r'$N$')
    plt.ylabel('MAE [%]')
    plt.grid(True, which="both", ls="--")
    plt.savefig("learning_curve_loglog.png")
    plt.tight_layout()
    plt.show()

def main():
    filename = sys.argv[1]
    df = get_dataframe(filename)
    feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]', 'Metal_Li', 'Metal_Co', 'Metal_Mn', 'Metal_Ni']
    target_column = ['Leaching Efficiency']
    X = df[feature_columns].to_numpy()
    Y = df[target_column].to_numpy().flatten()

    n_values = [30, 60, 120, 140, 150, 160]
    all_results = {n: [] for n in n_values}

    for _ in tqdm(range(5), desc="Test runs"):  # Repeat 10 times
        test_indices = []
        for metal in ['Metal_Li', 'Metal_Co', 'Metal_Mn', 'Metal_Ni']:
            metal_indices = df[df[metal] == 1].index
            test_indices.extend(np.random.choice(metal_indices, 5, replace=False))
        test_indices = np.array(test_indices)

        results = get_learning_curve_data(X, Y, test_indices, n_values)

        for n, mae in results:
            all_results[n].append(mae)

    avg_results = [(n, np.mean(all_results[n])) for n in n_values]

    plot_learning_curve(avg_results)


if __name__ == '__main__':
    main()

