# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:56:06 2024

@author: Usuario
"""

import numpy as np

class KMedoids:
    def __init__(self, n_clusters, max_iters=1000, tol=1e-5,random_seed=155):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.medoid_indices = None
        self.labels = None
        self.random_seed = random_seed
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize medoid indices randomly
        self.medoid_indices = np.random.seed(self.random_seed)
        
        for i in range(self.max_iters):
            # Assign each data point to the nearest medoid
            distances = self._calc_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update medoids
            new_medoid_indices = np.zeros(self.n_clusters, dtype=int)
            for j in range(self.n_clusters):
                cluster_points = X[self.labels == j]
                
                if len(cluster_points) > 0:
                    distances_to_points = np.sum(np.abs(cluster_points - cluster_points[:, np.newaxis]), axis=2)
                    total_distances = np.sum(distances_to_points, axis=1)
                    new_medoid_indices[j] = np.argmin(total_distances)
                else:
                    # Handle the case when the cluster is empty
                    print(f"Cluster {j} is empty. Reinitializing medoid.")
                    new_medoid_indices[j] = np.random.choice(n_samples)
                        
            # Check for convergence
            if np.array_equal(new_medoid_indices, self.medoid_indices):
                break
                
            self.medoid_indices = new_medoid_indices
            
    def predict(self, X):
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)
        
    def _calc_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, medoid_index in enumerate(self.medoid_indices):
            distances[:, i] = np.sum(np.abs(X - X[medoid_index]), axis=1)
        return distances
