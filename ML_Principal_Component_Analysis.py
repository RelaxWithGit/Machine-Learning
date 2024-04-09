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

def arrow(v1, v2, ax):
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0)
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
    pca = PCA(n_components=1)
    pca.fit(result_X)
    result_Z = pca.transform(result_X)
    return result_Z, pca.components_

def feature_extraction():
    X, y = load_diabetes(return_X_y=True)
    pca = PCA(2)
    pca.fit(X)
    print("Diabetes data explained variance: ", pca.explained_variance_)
    Z = pca.transform(X)
    v = pca.explained_variance_
    # Reshape v to match the length of np.arange(1,11)
    v_resized = np.concatenate((v, np.zeros(10 - len(v))))
    return Z, v_resized

def main():
    # Generate data
    result_X = random_data_helper()

    # Define v1, v2, and ax or remove this call if unnecessary
    v1 = [0, 0]
    v2 = [1, 1]
    ax = plt.gca()
    
    # Compute PCA and transform data
    result_Z, pca = arrow(v1, v2, ax)

    # Create a single figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    # Plot original data in the first subplot
    axes[0].axis('equal')
    axes[0].scatter(result_X[:, 0], result_X[:, 1])
    axes[0].set_title("Original")

    # Plot transformed data in the second subplot
    axes[1].axis('equal')
    axes[1].set_xlim(-3, 3)
    axes[1].scatter(result_Z[:, 0], result_Z[:, 1])  # Plot transformed data
    for l, v in zip(pca.explained_variance_, pca.components_):
        axes[1].annotate("", v * l * 3, [0, 0], arrowprops=dict(arrowstyle='->', linewidth=2))
    axes[1].set_title("Transformed")

    # Plot flattened PCA components in the third subplot
    result_flatten, pca_components = pca_flatten(result_Z, result_X)
    axes[2].axis('equal')
    axes[2].scatter(result_Z[:, 0], np.zeros(400), marker="d", alpha=0.1)
    axes[2].set_title("Flattened PCA Components")

    # Display the final random data figure
    plt.show()

    # Plot the figures from feature_extraction() in a new figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Call feature_extraction and plot the results
    Z, v_resized = feature_extraction()
    axes[0].axis('equal')
    axes[0].scatter(Z[:, 0], Z[:, 1])
    axes[0].set_title("Feature Extraction Scatter Plot")

    axes[1].plot(np.arange(1, 11), np.cumsum(v_resized))
    axes[1].set_title("Cumulative Explained Variance")

    # Display the final diabetesdata figure
    plt.show()


if __name__ == "__main__":
    main()
