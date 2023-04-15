import numpy as np
import utils
from sklearn.covariance import MinCovDet

class PCA_robust:
    '''
    What strategies do you know (or can think of) in order to make PCA more robust?
    1. Data normalization: Normalizing the data before applying PCA can make it more robust
    2. Incremental PCA: IPCA is a variant of PCA that processes the data in small batches,
       which can make it more computationally efficient and robust
    3. Robust covariance estimation, use another method to calculate the covariance that is not sensitive to outliers,
       such as the Minimum Covariance Determinant (MCD) estimator
    4. Primary Component Pursuit (PCP) approach: Use more robust optimization methods that are less sensitive to outliers

    For this class we apply data normalization and robust covariance estimation:
    Data normalization: Normalizing the data before applying PCA can make it more robust
    Robust covariance estimation, use another method to calculate the covariance such as the Minimum Covariance Determinant (MCD) estimator
    '''

    def __init__(self, n_components):   
        # Initialize the class with the number of main components you want to get
        self.n_components = n_components
        self.mean = None
        self.components = None
    
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        self.transform(X)

    def transform(self, X):
        # Normalize the data
        X = self.utils.normalize(X)

        # Estimate the robust covariance matrix (MCD)
        cov_matrix = MinCovDet().fit(X).covariance_
        
        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Order the eigenvalues and eigenvectors from largest to smallest
        sorted_indexes = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indexes]
        eigenvectors = eigenvectors[:, sorted_indexes]
        
        # Select the first n eigenvector components
        self.components = eigenvectors[:, :self.n_components]
    
    def fit_transform(self, X):
        # Center the data
        if len(X) > 1:
            X = X - self.mean
        else:
            X = (X - np.array(self.mean).reshape(1, -1))

        # Project the data into the principal components
        X_transformed = np.dot(X, self.components)

        return X_transformed
    
    def inverse_transform(self, X): 
        '''
        invert the transform
        '''
        X_reconstructed = (X @ self.components).dot(self.components.T) + np.mean(X, axis=0)      
        return X_reconstructed