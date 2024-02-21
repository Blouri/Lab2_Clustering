import numpy as np

class SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Perform Singular Value Decomposition (SVD)
        U, Sigma, VT = np.linalg.svd(X, full_matrices=False)

        # Only keep the first n_components singular values and corresponding vectors
        self.components = VT[:self.n_components, :].T

    def transform(self, X):
        # Center the data
        X = X - self.mean

        # Project the data onto the principal components
        X_transformed = np.dot(X, self.components)

        return X_transformed

    def inverse_transform(self, X_transformed):
        # Reconstruct the data from the transformed space
        X_original = np.dot(X_transformed, self.components.T)

        # Add back the mean to obtain the original data
        X_original = X_original + self.mean

        return X_original

    def fit_transform(self, X):
        # First fit the model to the data
        self.fit(X)

        # Then transform the data
        return self.transform(X)
