import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
import scipy

def blobs():
    X, y = make_blobs(centers=4, n_samples=200, random_state=0, cluster_std=0.7)
    print(X[:10], y[:10])

    model = KMeans(4)
    model.fit(X)
    print(model.cluster_centers_)

    def find_permutation(n_clusters, real_labels, labels):
        permutation = []
        for i in range(n_clusters):
            idx = labels == i
            mode = scipy.stats.mode(real_labels[idx])[0]
            mode = np.atleast_1d(mode)  # Ensure mode is treated as an array
            new_label = mode[0]  # Choose the most common label among data points in the cluster
            permutation.append(new_label)
        return permutation

    permutation = find_permutation(4, y, model.labels_)
    print(permutation)

    return permutation, model, X, y

def moons():
    X, y = make_moons(200, noise=0.05, random_state=0)
    print(X[:10], y[:10])

    model = DBSCAN(eps=0.3)
    model.fit(X)

    return model, X, y

def digits():
    digits = load_digits()
    print(digits.data.shape)

    model=KMeans(n_clusters = 10, random_state=0)
    model.fit(digits.data)
    model.cluster_centers_.shape

    def find_permutation(n_clusters, real_labels, labels):
        permutation = []
        for i in range(n_clusters):
            idx = labels == i
            mode = scipy.stats.mode(real_labels[idx])[0]
            mode = np.atleast_1d(mode)  # Ensure mode is treated as an array
            new_label = mode[0]  # Choose the most common label among data points in the cluster
            permutation.append(new_label)
        return permutation

    permutation = find_permutation(10, digits.target, model.labels_)
    print(permutation)
    acc = accuracy_score(digits.target, [ permutation[label] for label in model.labels_])
    print("Digit accuracy score is", acc)

    return digits, model

def main():
    # Generate blob data
    permutation_blob, model_blob, X_blob, y_blob = blobs()

    # Generate moon data
    model_moon, X_moon, y_moon = moons()

    # Generate the digit data
    digits_data, digits_model = digits()

    # Create a figure with two sub-plots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot blob data on the first sub-plot
    axs[0].scatter(X_blob[:, 0], X_blob[:, 1], c=model_blob.labels_)
    axs[0].scatter(model_blob.cluster_centers_[:, 0], model_blob.cluster_centers_[:, 1], s=100, color="red")
    axs[0].set_title('Blob Data')

    # Plot moon data on the second sub-plot
    axs[1].scatter(X_moon[:, 0], X_moon[:, 1], c=model_moon.labels_)
    axs[1].set_title('Moon Data')

    # Create digits graphics
    fig, axes = plt.subplots(2,5, subplot_kw=dict(xticks=[], yticks=[]))
    for ax, digit in zip(axes.flat, digits_model.cluster_centers_):
        ax.imshow(digit.reshape(8,8), cmap="gray")

    # Show the plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
