from joblib import Parallel, delayed
import numpy as np

class ParallelPCA:
    def __init__(self, n_components, n_jobs=-1):
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.components = None
        self.mean = None

    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Parallelize the computation of the covariance matrix
        cov_matrix_parts = Parallel(n_jobs=self.n_jobs)(
            delayed(np.cov)(X_centered[:, :], rowvar=False)
        )

        # Combine the covariance matrix parts
        cov = np.sum(cov_matrix_parts, axis=0)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort the eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store the first n_components eigenvectors as the principal components
        self.components = eigenvectors[:, : self.n_components]