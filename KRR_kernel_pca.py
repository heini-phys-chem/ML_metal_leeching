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

import re
from adjustText import adjust_text

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

SMILES = {
    "acid"    : "[H][C@@]1(OC(=O)C(O)=C1O)[C@@H](O)CO",
    "glycine" : "C(C(=O)O)N",
          }

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

    #df = df[(df['Temperature [degC]'].between(70, 80))]
    #df = df[(df['ratio'].between(18, 23))]
    #df = df[(df['solvent 1'].between(0.9, 2.1))]
    #df = df[(df['solvent 2'].between(0.03, 0.09))]
    #df = df[(df['time [min/hours]'].between(0.1, 250))]
    #df.reset_index(drop=True, inplace=True)

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

    sigmas  = [0.01 * 2**i for i in range(20)]
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

                # Apply your kernel function and train your model here
                # Example with Laplacian kernel
                #K_train = laplacian_kernel(X_train, X_train, sigma)
                #K_val = laplacian_kernel(X_train, X_val, sigma)
                K_train = laplacian_kernel_gpu(X_train, X_train, sigma)
                K_val = laplacian_kernel_gpu(X_train, X_val, sigma)

                K_train = K_train.get()
                C = K_train + np.eye(len(X_train)) * llambda
                C = cp.asarray(C)
                #alpha = svd_solve(C, y_train)
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

def center_kernel_matrix(K):
    """
    Center the kernel matrix K.

    Parameters:
    - K: CuPy ndarray of shape (n_samples, n_samples), the kernel matrix.

    Returns:
    - K_centered: CuPy ndarray of shape (n_samples, n_samples), the centered kernel matrix.
    """
    n_samples = K.shape[0]
    one_n = cp.ones((n_samples, n_samples)) / n_samples
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return K_centered

def get_kernel_pca(X_train, y_train, sigma, llambda, n_components):

    X_train = cp.asarray(X_train)

    K_train_gpu = laplacian_kernel_gpu(X_train, X_train, sigma)

    #K_train = K_train_gpu.get()

    # Step 2: Center the kernel matrix
    K_centered = center_kernel_matrix(K_train_gpu)

    # Step 3: Perform eigen-decomposition
    eigvals, eigvecs = cp.linalg.eigh(K_centered)
    print(np.sort(eigvals.get())[-14:])

    # Step 4: Collect the top n_components eigenvectors (principal components)
    X_pc = cp.dot(K_centered, eigvecs[:, -n_components:])

    return X_pc

def get_KRR(X_train, y_train):
    np.random.seed(667)

    best_sigma, best_lambda = get_hyperparams(X_train, y_train)
    print(f"hyperparams: {best_sigma}, {best_lambda}")
#    best_sigma, best_lambda = 400.0, 1e-1

    # Make predictions
    n_components = X_train.shape[0]
    X_pc = get_kernel_pca(X_train, y_train, best_sigma, best_lambda, n_components)

    return X_pc


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


def main():
    filename = sys.argv[1]

    metals = ['Li', 'Co', 'Mn', 'Ni']
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16,10))

    df = get_dataframe(filename)

    print(df)
    feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]', 'Metal_Li', 'Metal_Co', 'Metal_Mn', 'Metal_Ni']
    #feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]', 'Metal_Li']#, 'Metal_Co', 'Metal_Mn', 'Metal_Ni']
    #feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]', 'Metal_Co']#, 'Metal_Co', 'Metal_Mn', 'Metal_Ni']
    #feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]', 'Metal_Mn']#, 'Metal_Co', 'Metal_Mn', 'Metal_Ni']
    #feature_columns = ['solvent 1', 'solvent 2', 'ratio', 'Temperature [degC]', 'time [min/hours]', 'Metal_Ni']#, 'Metal_Co', 'Metal_Mn', 'Metal_Ni']
    target_column = ['Leaching Efficiency']

    X = df[feature_columns].to_numpy()
    Y = df[target_column].to_numpy()

    metal_labels = df['Metal'].to_numpy()

    # Standard scaling the features
    #scaler = StandardScaler()
    #print(X)
    X_scaled = X#scaler.fit_transform(X)
    Y = Y.flatten()
    labels = Y

    X_pc = get_KRR(X, Y)
    n_features = X.shape[1]
    print(n_features)
    #exit()

    # Convert the result to a NumPy array for plotting
    X_pc = cp.asnumpy(X_pc)

    # Plotting the results
    sc = ax.scatter(X_pc[:, 0], X_pc[:, 1], c=labels, cmap='viridis', s=200)
    plt.colorbar(sc, label='Leeching Efficiency [%]')

    # Annotate each point with the corresponding metal label
    #for i, label in enumerate(metal_labels):
#    for i, label in enumerate(metal_labels):
#        plt.annotate(label, (X_pc[i, 0], X_pc[i, 1]), textcoords="offset points", xytext=(5,5), ha='center')

    # Annotate each point with the corresponding metal label
    #texts = []
    #for i, label in enumerate(metal_labels):
    #    texts.append(plt.text(X_pc[i, 0], X_pc[i, 1], label))

    ## Adjust the text annotations to avoid overlap
    #adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'), expand=(2, 2))



    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    #plt.legend()
    plt.show()
    fig.savefig("kernel_pca.png")


if __name__ == '__main__':
    main()

