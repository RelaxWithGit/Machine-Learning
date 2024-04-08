#!/usr/bin/env python3

import scipy.stats
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

def find_permutation(n_clusters, real_labels, labels):
    permutation=[]
    for i in range(n_clusters):
        idx = labels == i
        # Choose the most common label among data points in the cluster
        mode = scipy.stats.mode(real_labels[idx])[0]
        mode = np.atleast_1d(mode)  # Ensure mode is treated as an array
        new_label = mode[0]  # Choose the most common label among data points in the cluster
        permutation.append(new_label)
    return permutation

def plant_clustering():
    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    model = KMeans(n_clusters=3, random_state=0)
    model.fit(X)
    
    permutation = find_permutation(3, y, model.labels_)

    acc = accuracy_score(y, [permutation[label] for label in model.labels_])
    print("Plant clustering accuracy score is", acc)
    return acc

def main():
    print(plant_clustering())

if __name__ == "__main__":
    main()
