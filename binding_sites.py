import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from scipy.stats import mode
import scipy.cluster.hierarchy as hc

def toint(nucleotide):
    """Converts a nucleotide to an integer."""
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return nucleotide_map[nucleotide]

def get_features_and_labels(filename):
    """Load the contents of the file into a DataFrame and convert the 'X' column into a feature matrix."""
    df = pd.read_csv(filename, sep='\t')
    features = df['X'].apply(lambda x: [toint(c) for c in x])
    features_array = np.array(features.tolist())  # Convert to NumPy array
    labels = df['y'].tolist()
    return features_array, np.array(labels)

def find_permutation(real_labels, predicted_labels):
    """Find the permutation to match predicted labels with real labels."""
    permutation = []
    for i in np.unique(predicted_labels):
        idx = predicted_labels == i
        real_labels_subset = np.array(real_labels)[idx]  # Convert to numpy array
        mode_result = mode(real_labels_subset.astype(int))[0]  # Convert to integers before computing mode
        mode_result = np.atleast_1d(mode_result)
        new_label = mode_result[0]
        permutation.append(new_label)
    return permutation

def cluster_euclidean(filename):
    """Perform hierarchical clustering using euclidean affinity and return the accuracy score."""
    features, labels = get_features_and_labels(filename)
    model = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='euclidean')
    predicted_labels = model.fit_predict(features)
    permutation = find_permutation(labels, predicted_labels)
    new_labels = [permutation[label] for label in predicted_labels]
    return accuracy_score(labels, new_labels)

def cluster_hamming(filename):
    """Perform hierarchical clustering using hamming affinity and return the accuracy score."""
    features, labels = get_features_and_labels(filename)
    distance_matrix = pairwise_distances(features, metric='hamming')
    model = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='precomputed')
    predicted_labels = model.fit_predict(distance_matrix)
    permutation = find_permutation(labels, predicted_labels)
    new_labels = [permutation[label] for label in predicted_labels]
    return accuracy_score(labels, new_labels)

def plot_dendrogram(ax, features, labels, title):
    """Plot dendrogram on given axes."""
    dendrogram = hc.dendrogram(hc.linkage(features, method='average'), ax=ax, labels=labels)
    ax.set_title(title)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Distance')

def main():

    #test "G-A-C-T"
    nucleotide = 'G'
    print("G test returns:",toint(nucleotide))
    nucleotide = 'A'
    print("A test returns:",toint(nucleotide))
    nucleotide = 'C'
    print("C test returns:",toint(nucleotide))
    nucleotide = 'T'
    print("T test returns:",toint(nucleotide))

    print("Accuracy score with Euclidean affinity is", cluster_euclidean("src/data.seq"))
    print("Accuracy score with Hamming affinity is", cluster_hamming("src/data.seq"))

    filename = "src/data.seq"
    
    features, labels = get_features_and_labels(filename)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with 1x2 subplots

    # Plot dendrogram for Euclidean clustering
    plot_dendrogram(axs[0], pairwise_distances(features, metric='euclidean'), labels, 'Cluster Euclidean')

    # Plot dendrogram for Hamming clustering
    plot_dendrogram(axs[1], pairwise_distances(features, metric='hamming'), labels, 'Cluster Hamming')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()