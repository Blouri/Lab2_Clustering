import numpy as np

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.embedding = None

    def _symmetric_sne(self, P):
        Q = 1 / (1 + np.sum((self.embedding[:, None, :] - self.embedding) ** 2, axis=-1) / self.perplexity)
        Q /= np.sum(Q)
        return Q

    def _kl_divergence(self, P, Q):
        return np.sum(P * np.log(P / Q))

    def _tsne(self, P):
        grad = np.zeros_like(self.embedding)

        for i in range(len(self.embedding)):
            sum_diff = np.sum((self.embedding[i, None, :] - self.embedding) ** 2, axis=-1)
            sum_diff[sum_diff == 0] = np.inf
            Q = 1 / (1 + sum_diff / self.perplexity)
            grad[i, :] = 4 * np.sum((P[i, :] - Q)[:, None] * (self.embedding[i, :] - self.embedding), axis=0)

        grad *= 2
        grad -= 2 * self.learning_rate * np.sum((1 / (1 + np.sum((self.embedding[:, None, :] - self.embedding) ** 2, axis=-1)))[:, :, None] * (self.embedding[:, None, :] - self.embedding), axis=0)
        return grad

    def fit_transform(self, X):
        P = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            distances = np.sum((X[i, :] - X) ** 2, axis=1)
            P[i, :] = self._conditional_probabilities(distances, i)
        P = (P + P.T) / (2 * X.shape[0])

        self.embedding = np.random.randn(X.shape[0], self.n_components)

        for _ in range(self.n_iter):
            Q = self._symmetric_sne(P)
            grad = self._tsne(P)
            self.embedding -= self.learning_rate * grad
            P = self._symmetric_sne(self.embedding)

        return self.embedding

    def _conditional_probabilities(self, distances, i):
        beta_min, beta_max = -np.inf, np.inf
        tol = 1e-5
        target_entropy = np.log(self.perplexity)

        while True:
            beta = (beta_min + beta_max) / 2
            exp_distances = np.exp(-beta * distances)
            sum_exp_distances = np.sum(exp_distances)
            P = exp_distances / sum_exp_distances
            entropy = -np.sum(P * np.log2(P + 1e-12))
            error = entropy - target_entropy

            if np.abs(error) < tol:
                break
            elif error > 0:
                beta_max = beta
            else:
                beta_min = beta

        return P
