import numpy as np
import matplotlib.pyplot as plt
from computations import *


class GMM:
    def __init__(self, probabilities: np.array, mu: list, sigma: list):
        self.p = probabilities
        self.mu = mu
        self.sigma = sigma

        self.dim = self.mu[0].shape[0]  # Dimension of data
        self.K = len(mu)  # Number of clusters

    def sample(self, n):
        samples = np.zeros((n, self.dim))
        labels = np.zeros(n)
        for i in range(n):
            cluster = np.random.choice(np.arange(self.K), p=self.p)
            sample = np.random.multivariate_normal(self.mu[cluster], self.sigma[cluster], size=1)
            samples[i] = sample
            labels[i] = cluster
        return samples, labels

def toy_example():
    ''' Returns similarity matrix for a toy example of euclidean data'''
    # Simulate data from GMM
    p = np.array([0.5, 0.5])
    mu = [np.array([-2, -2]), np.array([2, 2])]
    sigma = [np.eye(2)] * 2

    n_samples = 60
    gmm = GMM(p, mu, sigma)
    samples, labels = gmm.sample(n_samples)
    samples = samples[labels.argsort()]

    D = distance_matrix(samples)
    sigma_sq = D.max() / 16
    S = similarity_matrix(samples, sigma_sq)

    return S


if __name__ == '__main__':
    p = np.array([0.4, 0.3, 0.3])
    mu = [np.array([0, 0]), np.array([15, 2]), np.array([-13, -12])]
    sigma = [np.eye(2)] * 3

    n_samples = 10000
    gmm = GMM(p, mu, sigma)
    samples, labels = gmm.sample(n_samples)
    x = samples[:, 0]
    y = samples[:, 1]

    plt.scatter(x, y)
    plt.show()

    end=True