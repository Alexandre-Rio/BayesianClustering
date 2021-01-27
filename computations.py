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


if __name__ == '__main__':

    end=True