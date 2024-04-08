#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
import scipy.stats
import matplotlib.pyplot as plt

def find_permutation(real_labels, labels):
    permutation = []
    cluster_labels = np.unique(labels)
    num_clusters = len(cluster_labels[cluster_labels != -1])  # Exclude outliers label (-1)
    num_labels = len(np.unique(real_labels))
    
    if num_clusters != num_labels:
        return np.nan, num_clusters, np.sum(labels == -1)
    
    for i in range(num_clusters):
        idx = labels == i
        mode = scipy.stats.mode(real_labels[idx])[0]
        mode = np.atleast_1d(mode)  # Ensure mode is treated as an array
        new_label = mode[0]  # Choose the most common label among data points in the cluster
        permutation.append(new_label)
    
    return accuracy_score(real_labels[labels != -1], [permutation[label] for label in labels[labels != -1]]), num_clusters, np.sum(labels == -1)

def nonconvex_clusters():
    results = []
    eps_values = np.arange(0.05, 0.2, 0.05)
    df = pd.read_csv("src/data.tsv", sep='\t')

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Create a figure with 2x2 subplots

    for i, eps in enumerate(eps_values):
        dbscan = DBSCAN(eps=eps)
        dbscan.fit(df[['X1', 'X2']])
        acc_score, num_clusters, num_outliers = find_permutation(df['y'], dbscan.labels_)
        results.append({'eps': float(eps), 'Score': float(acc_score), 'Clusters': float(num_clusters), 'Outliers': float(num_outliers)})  # Convert values to float
        # Plot the dataset with clusters colored in the corresponding subplot
        ax = axs[i // 2, i % 2]  # Select the current subplot
        sc = ax.scatter(df['X1'], df['X2'], c=dbscan.labels_)
        ax.set_title(f'DBSCAN Clustering with eps={eps}')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        plt.colorbar(sc, ax=ax, label='Cluster')

    plt.tight_layout()  # Adjust layout to prevent overlap
    return pd.DataFrame(results)

def main():

    results_df = nonconvex_clusters()
    print(results_df)
    plt.show()  # Display the figure with subplots

if __name__ == "__main__":
    main()
