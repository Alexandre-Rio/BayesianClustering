import numpy as np
from numpy import linalg as lin


def wishart_pdf(S, Sigma, d):
    n = S.shape[0]
    return np.power(lin.det(S), 0.5 * (d - n - 1)) * np.power(lin.det(lin.inv(Sigma)), 0.5 * d) * \
           np.exp(- 0.5 * d * np.trace(np.dot(lin.inv(Sigma), S)))


def wishart_log_likelihood(S, Sigma, d):
    return - 0.5 * d * np.log(lin.det(Sigma)) - 0.5 * d * np.trace(np.dot(lin.inv(Sigma), S))


def crp(c_new, c_old, ksi):
    '''
    Probability desnity for the Chinese Restaurant Process
    :param c_new: int, new label
    :param c_old: array, old labels
    :param ksi: control parameter for the introduction of new clusters
    :return: probability of new label
    '''
    n = c_old.shape[0] + 1
    denom = n - 1 + ksi
    if c_new in c_old:
        n_j = np.count_nonzero(c_old == c_new)
        proba = n_j / denom
    else:
        proba = ksi / denom
    return proba

if __name__ == '__main__':
    end = True