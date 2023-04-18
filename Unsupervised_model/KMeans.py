import numpy as np
import utils
import random

random.seed(1111)

class KMeans:

    def __init__(self, n_clusters, max_iters=1000, tol=1e-5):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self,X):
        n_samples  = X.shape[0]
        n_features = X.shape[1]

        # Initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for i in range(self.max_iters):
            # Assign each data point to the nearest centroid
            distances = utils.distances(self,X)
            self.labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for j in range(self.n_clusters):
                new_centroids[j] = np.mean(X[self.labels == j], axis=0)
                
            # Check for convergence
            if np.sum(np.abs(new_centroids - self.centroids)) < self.tol:
                break
                
            self.centroids = new_centroids

    def predict(self, X):
        distances = utils.distances(self,X)
        return np.argmin(distances, axis=1)




        
