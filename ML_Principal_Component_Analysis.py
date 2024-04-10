"""
This script demonstrates various operations related to Principal Component Analysis (PCA) and data visualization using matplotlib. 

Functions:
- random_data_helper(): Generates random data with a specific covariance structure.
- arrow(v1, v2): Computes PCA on random data and visualizes the principal axes with arrows.
- pca_flatten(result_Z, result_X): Performs PCA on given data and returns the flattened components.
- diabetes_helper_function(): Performs PCA on the diabetes dataset and visualizes it.
- feature_extraction(): Performs PCA on the diabetes dataset and plots the cumulative explained variance.

The main function orchestrates the execution of these functions and displays the results using subplots.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

def random_data_helper():
    rng = np.random.RandomState(0)
    X = rng.randn(2, 400)
    scale = np.array([[1, 0], [0, 0.4]])  # Standard deviations are 1 and 0.4
    rotate = np.array([[1, -1], [1, 1]]) / math.sqrt(2)
    transform = np.dot(rotate, scale)
    X = np.dot(transform, X)
    X = X.T
    return X

def arrow(v1, v2):
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0)
    ax = plt.gca()
    ax.annotate("", v2, v1, arrowprops=arrowprops)
    pca = PCA(2)
    X = random_data_helper()
    pca.fit(X)
    print("Principal axes:", pca.components_)
    print("Explained variance:", pca.explained_variance_)
    print("Mean:", pca.mean_)
    Z = pca.transform(X)
    plt.close()
    return Z, pca

def pca_flatten(result_Z, result_X):
    pca=PCA(n_components=1)
    pca.fit(result_X)
    result_Z=pca.transform(result_X)
    return result_Z, pca.components_

def diabetes_helper_function():
    X, y = load_diabetes(return_X_y=True)
    pca=PCA(2)
    pca.fit(X)
    print(pca.explained_variance_)
    Z=pca.transform(X)
    plt.axis('equal')
    plt.scatter(Z[:,0], Z[:,1]);

    return plt.show()

def feature_extraction():
    X, y = load_diabetes(return_X_y=True)
    pca=PCA(2)
    pca.fit(X)
    print("Diabetes data explained variance: ", pca.explained_variance_)
    Z=pca.transform(X)
    
    rng=np.random.RandomState(0)
    X=rng.randn(3,400)
    p=rng.rand(10,3)  # Random projection into 10d
    X=np.dot(p, X)
    pca=PCA()
    pca.fit(X)
    v=pca.explained_variance_
    plt.plot(np.arange(1,11), np.cumsum(v))
    return X, v

def main():
    # Generate data
    result_X = random_data_helper()

    # Define v1, v2, and ax or remove this call if unnecessary
    v1 = [0, 0]
    v2 = [1, 1]
    
    # Compute PCA and transform data
    result_Z, pca = arrow(v1, v2)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].axis('equal')
    axes[0].scatter(result_X[:, 0], result_X[:, 1])
    axes[1].axis('equal')
    axes[1].set_xlim(-3, 3)
    axes[1].scatter(result_Z[:, 0], result_Z[:, 1])  # Plot transformed data
    for l, v in zip(pca.explained_variance_, pca.components_):
        axes[0].annotate("", v * l * 3, [0, 0], arrowprops=dict(arrowstyle='->', linewidth=2))
    for l, v in zip([1.0, 0.16], [np.array([1.0, 0.0]), np.array([0.0, 1.0])]):
        axes[1].annotate("", v * l * 3, [0, 0], arrowprops=dict(arrowstyle='->', linewidth=2))
    axes[0].set_title("Original")
    axes[1].set_title("Transformed")
    
    # Display 
    plt.show()

    result_flatten, _ = pca_flatten(result_Z, result_X)
    print("Flattened PCA components are: ", pca.components_)
    plt.axis('equal')
    plt.scatter(result_Z[:,0],np.zeros(400), marker="d", alpha=0.1);
    plt.show()

    # Feature extraction

    diabetes_helper_function()

    data_feature_extraction, result_feature_extraction = feature_extraction()

    plt.plot(np.arange(1,11), np.cumsum(result_feature_extraction));
    plt.show()

if __name__ == "__main__":
    main()
