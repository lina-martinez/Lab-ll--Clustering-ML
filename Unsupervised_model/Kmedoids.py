import utils
import numpy as np
import random

class KMedoids:
    
    def __init__(self, n_clusters, max_iters=1000, tol=1e-5):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.cluster_centers_ = None
        self.tol = tol
        self.medoids = None
        self.labels = None
        np.random.seed(123)

    def fit(self, X):
        n_samples = X.shape[0]
        
        assert self.n_clusters <= n_samples

        # Initialize medoids randomly
        self.medoids = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        for i in range(self.max_iters):
            # Assign each data point to the nearest medoid
            distances = utils.pairwise_distances(X, X[self.medoids])
            self.labels = np.argmin(distances, axis=1)
            
            # Update medoids
            new_medoids = np.zeros(self.n_clusters, dtype=np.int64)
            for j in range(self.n_clusters):
                mask = (self.labels == j)
                if np.sum(mask) > 0:
                    cluster_distances = utils.pairwise_distances(X[mask], X[mask])
                    total_distance = np.sum(cluster_distances, axis=1)
                    new_medoids[j] = np.argmin(total_distance)
                else:
                    new_medoids[j] = self.medoids[j]

            # Check for convergence
            if np.all(new_medoids == self.medoids):
                self.cluster_centers_ = self.medoids
                break
                
            self.medoids = new_medoids
            
    def predict(self, X):
        distances = utils.pairwise_distances(X, X[self.medoids])
        return np.argmin(distances, axis=1)
