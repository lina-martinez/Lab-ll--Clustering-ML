import numpy as np

def pairwise_distances(X, Y=None):
    """
    Computes the pairwise distances between each pair of data points in the X matrix
    """
    if Y is None:
        Y = X
        
    # Compute squared distances
    distance = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)

    # Replace negative values with 0 (due to floating point errors)
    distance[distance < 0] = 0

    # Take the square root to obtain Euclidean distances
    distance = np.sqrt(distance)
    return distance

def distances(self, X):
    '''
    Computes the distance between each point in X and each centroid in self.centroids
    '''
    distances = np.zeros((X.shape[0], self.n_clusters))
    for i, centroid in enumerate(self.centroids):
        distances[:, i] = np.linalg.norm(X - centroid, axis=1)
    return distances

def normalize(self, X):
    """
    Normalizes the rows of X
    """
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X 

def probability(self, distance, perplexity):
    """
    Computes the P-values for the t-SNE algorithm.
    """
    n_samples = distance.shape[0]
    probability = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        beta = 1.0
        done = False
        while not done:
            p = np.exp(-distance[i] * beta)
            p[i] = 0.0
            sum_p = np.sum(p)
            if sum_p == 0.0:
                beta *= 2.0
            else:
                probability[i] = p / sum_p
                done = True
    probability = 0.5 * (probability + probability.T)
    probability = np.maximum(probability, np.finfo(float).eps)
    probability /= np.sum(probability)
    probability = np.maximum(probability, np.finfo(float).eps)
    return probability

def compute_q(self, Y):
    """
    Computes the Q-values for the tSNE algorithm.
    """
    n_samples = Y.shape[0]
    Y_diff = Y[np.newaxis, :] - Y[:, np.newaxis]

    # Get the euclidean distance squared
    distances = np.sum(Y_diff**2, axis=-1)

    # Q-value
    Q = 1.0 / (1.0 + distances)

    # Set the diagonal of the matrix Q to zero
    Q[range(n_samples), range(n_samples)] = 0.0
    Q /= np.sum(Q)
    return Q

def compute_gradient(self, probability , Q , Y, distance):
    """
    Computes the gradients for the t-SNE algorithm.
    """
    # Compute pairwise differences in the embedded space
    Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]

    # Compute the t-SNE gradient
    factor = 4 * (probability - Q)[:,:,np.newaxis] * Y_diff * (1 / (1 + distance**2))[:,:,np.newaxis]
    grad = np.sum(factor, axis=1)
    return grad

