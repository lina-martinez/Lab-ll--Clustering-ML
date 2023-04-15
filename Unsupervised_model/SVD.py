import numpy as np

class SVD:

    def __init__(self, n_components):
        # Initialize the class with the number of main components you want to get
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        self.transform(X)
        
    def transform(self, X):
        # Project the data into the new components
        X = X - self.mean

        # Compute the vectors
        self.U, self.S, self.VT = np.linalg.svd(X) 

        # Compute mean and std to standardization   
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)

        if self.n_components is not None:
                self.U = self.U[:, :self.n_components]
                self.S = self.S[:self.n_components]
                self.VT = self.VT[:self.n_components, :]
        self.components = self.VT[:self.n_components].T
    
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
