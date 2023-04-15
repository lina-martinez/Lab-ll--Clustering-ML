import numpy as np

class PCA:

    def __init__(self, n_components):
        # initialize the class with the number of main components you want to get
        self.n_components = n_components
        self.mean = None
        self.components = None
    
    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        self.transform(X)

    def transform(self, X):
        X = X - self.mean

        # Compute the covariance matrix
        cov_matrix = np.cov(X, rowvar=False)
        
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