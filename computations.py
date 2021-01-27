import numpy as np
import generate_data

__all__ = ['distance_matrix', 'similarity_matrix', 'top95_eigenvalues']

def distance_matrix(X):
    ''' Compute the matrix of euclidean distances between n points '''
    n = X.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            distance_matrix[i, j] = np.linalg.norm(X[i] - X[j]) ** 2
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix

def similarity_matrix(X, sigma_sq=1):
    ''' Compute the similiarity matrix between n points'''
    n = X.shape[0]
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            similarity_matrix[i, j] = np.exp(- np.power(np.linalg.norm(X[i] - X[j]), 2) / (2 * sigma_sq))
            similarity_matrix[j, i] = similarity_matrix[i, j]
        similarity_matrix[i, i] = 1.0
    return similarity_matrix

def top95_eigenvalues(S):
    '''Find the number of largest eigenvalues of symmetric matrix S which explains about 95% of the total variance'''
    eigenvalues = np.linalg.eigvals(S)
    eigenvalues = - np.sort(- eigenvalues)  # Sort eigenvalues in reverse order
    threshold = 0.95 * eigenvalues.sum()
    d = 1
    sum = eigenvalues[d - 1]
    while sum < threshold:
        d += 1
        sum += eigenvalues[d - 1]
    return d

def compute_srvf(curve):
    ''' Compute SRVF of curve. 2D curve must be of shape 2xT '''
    T = curve.shape[1]
    gradient = np.gradient(curve, axis=1)
    velocity = np.transpose(gradient)
    velocity_norm = np.linalg.norm(velocity, axis=1)
    velocity_norm = np.transpose(np.vstack([velocity_norm] * 2))

    q = []
    for t in range(T):
        if velocity_norm[t, 0] > 0:
            q_t = velocity[t] / np.sqrt(velocity_norm[t])
        else:
            q_t = 0
        q.append(q_t)

    return np.array(q)


if __name__ == '__main__':
    p = np.array([0.4, 0.3, 0.3])
    mu = [np.array([0, 0]), np.array([15, 2]), np.array([-13, -12])]
    sigma = [np.eye(2)] * 3

    n_samples = 30
    gmm = generate_data.GMM(p, mu, sigma)
    samples, labels = gmm.sample(n_samples)
    samples = samples[labels.argsort()]  # Group samples by cluster

    D = distance_matrix(samples)
    sigma_sq = D.max() / 2
    S = similarity_matrix(samples, sigma_sq)

    end=True