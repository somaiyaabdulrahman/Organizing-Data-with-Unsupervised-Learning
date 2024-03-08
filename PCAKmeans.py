import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

wine_data_path = 'WINE.txt'
wine_df = pd.read_csv(wine_data_path, delim_whitespace=True, header=None)

# Part 1: Principal Component Analysis (PCA)
X = wine_df.iloc[:, 1:]  # Selecting only the feature columns
X_standardized = StandardScaler().fit_transform(X)

# Performing PCA
pca = PCA()
X_pca = pca.fit_transform(X_standardized)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=wine_df[0], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Visualization of WINE Data using Top 2 Principal Components')
plt.colorbar(label='Class')
plt.show()

# Part 2:  K-means Clustering
def kmeans(data, k, n_components, max_iters=100):
    # Selecting the top N principal components for clustering
    data = data[:, :n_components]

    # Randomly initialize centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for i in range(max_iters):
        # Assign points to the nearest centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        closest_centroid = np.argmin(distances, axis=0)

        # Calculate new centroids from the mean of points
        new_centroids = np.array([data[closest_centroid == j].mean(axis=0) for j in range(k)])

        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    # Count number of points in each cluster
    cluster_sizes = np.array([np.sum(closest_centroid == j) for j in range(k)])

    return centroids, cluster_sizes, closest_centroid

# Performing K-means clustering using top 2, 5, and 8 principal components
centroids_2, cluster_sizes_2, labels_2 = kmeans(X_pca, 3, 2)
centroids_5, cluster_sizes_5, labels_5 = kmeans(X_pca, 3, 5)
centroids_8, cluster_sizes_8, labels_8 = kmeans(X_pca, 3, 8)

line = "-" * 65
print("\nCentroids and Cluster Sizes using Top 2 Principal Components:")
print("Centroids:\n", centroids_2)
print("\nCluster Sizes:\n", cluster_sizes_2)

print("\n" + line, "\nCentroids and Cluster Sizes using Top 5 Principal Components:")
print("Centroids:\n", centroids_5)
print("\nCluster Sizes:\n", cluster_sizes_5)

print("\n" + line, "\nCentroids and Cluster Sizes using Top 8 Principal Components:")
print("Centroids:\n", centroids_8)
print("\nCluster Sizes:\n", cluster_sizes_8)

# ska man ha med detta??
# Visualizing the clustering results using top 2 principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_2, cmap='viridis')
plt.scatter(centroids_2[:, 0], centroids_2[:, 1], c='red', marker='X', s=200)  # Marking centroids
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering on WINE Data using Top 2 Principal Components')
plt.show()