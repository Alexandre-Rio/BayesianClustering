import numpy as np
import matplotlib.pyplot as plt

from utils import *
from stats import crp
from generate_data import GMM, toy_example
from computations import *

def calculate_B(c, N):
    B = np.zeros(N, N)
    for row in range(N):
        for col in range(N):
            if c[row] == c[col]:
                B[row, col] = 1
    return B


def WishartCluster(d, theta_vect, r0, s0, ksi, M, S, iter=2000):
    '''
    Implementation of the function Wishart Cluster from Chinese Github
    :param d:
    :param theta_vect:
    :param r0:
    :param s0:
    :param ksi:
    :param iter: number of iterations of MCMC
    :param M: number of initial clusters
    :param S: similarity matrix
    :return:
    '''

    dispmcmc = 1

    N = S.shape[0]
    J = len(theta_vect)

    # dd = 1.0

    pp = 1
    fd = 1
    final_B = np.zeros((N, N))
    record = []
    final_rd = []

    c = np.zeros(N)  # Clusters
    cn = np.zeros(M)  # CLuster sizes

    for i in range(N):
        c[i] = (i + 1) % M + 1
    for i in range(M):
        cn[i] = np.sum(c == (i+1))

    while iter > 0:

        # Update parameters
        # d = dd * d
        iter -= 1

        # Update theta
        log_lktheta = np.zeros(J)
        for j in range(J):
            logPhi = 0
            sumSB = 0
            theta = theta_vect[j]

            for i in range(M):
                logPhi += np.log(1 + theta * cn[i])
                Ib = (c == (i + 1))
                sumSB += (theta / (1 + cn[i] * theta)) * np.vdot(Ib, np.dot(S, Ib))

            log_lkhood = (d / 2) * (- logPhi + (N + r0) * (np.log(d / 2) + np.log(np.trace(S) - sumSB + s0)))
            log_lktheta[j] = log_lkhood

        log_lktheta = log_lktheta - np.max(log_lktheta)  # Prior on theta is uniform
        lktheta = np.exp(log_lktheta)
        lktheta = lktheta / np.sum(lktheta)
        theta_idx = np.random.choice(np.arange(J), p=lktheta)
        # Final theta
        theta = theta_vect[theta_idx]

        # Clear parameters
        lkhoodp = 0
        pb = 0

        # Update B
        for j in range(N):

            # On va tester les différentes configurations de clustering pour la donnée j

            cn[int(c[j] - 1)] -= 1 # Update distribution
            # Remove cluster associated with none elements
            if cn[int(c[j] - 1)] == 0:
                oldind = c[j] - 1
                # Move last one to the removed space
                cn[oldind] = cn[M - 1]
                c[c == (M - 1)] = oldind
                M = M - 1
            catp = np.zeros(M + 1)
            log_p = np.zeros(M + 1)

            for k in range(1, M + 1):
                # Temp clustering
                tmpc = c
                tmpc[j] = k
                tmpcn = cn
                tmpcn[k - 1] += 1

                logPhi = 0
                sumSB = 0
                for i in range(M):
                    logPhi += np.log(1 + theta * tmpcn[i])  # cn ou tmpcn ? tmpcn plutôt j'imagine
                    Ib = (tmpc == (i + 1))
                    sumSB += (theta / (1 + tmpcn[i] * theta)) * np.vdot(Ib, np.dot(S, Ib))

                log_lkhood = (d / 2) * (- logPhi + (N + r0) * (np.log(d / 2) + np.log(np.trace(S) - sumSB + s0)))
                log_p[k - 1] = log_lkhood

            # On teste maintenant l'apparition d'un nouveau cluster
            tmpc = c
            tmpc[j] = M + 1
            tmpcn = cn
            tmpcn = np.concatenate([tmpcn, [1]])

            detPhi = 1
            sumSB = 0

            for i in range(M + 1):
                detPhi *= (1 / (1 + theta * tmpcn[i]))
                Ib = (tmpc == (i + 1))
                sumSB += (theta / (1 + tmpcn[i] * theta)) * np.vdot(Ib, np.dot(S, Ib))

            log_detPhi = (d / 2) * np.log(detPhi)
            tr_PhiS = (np.trace(S) - sumSB)
            log_trPhiS = (np.log(tr_PhiS + s0) + np.log(d/2)) * (- (N + r0) * d / 2)
            log_lkhood = log_trPhiS + log_detPhi
            log_p[M] = log_lkhood

            log_p = log_p - np.max(log_p)
            lkhoodp = np.exp(log_p)
            lkhoodp = lkhoodp / np.sum(lkhoodp)

            for k in range(M):
                catp[k] = (cn[k] / (N - 1 + ksi)) * lkhoodp[k]
            catp[M] = (ksi / (N - 1 + ksi)) * lkhoodp[M]

            catp /= np.sum(catp)  # Etape de renormalisation qui n'est pas dans le code initial => A surveiller

            c[j] = np.random.choice(np.arange(len(catp)), p=catp)
            # If cj falls in the new category
            if c[j] > M:
                M = M + 1
                cn[int(M - 1)] = 1
            else:
                cn[int(c[j])] += cn[int(c[j])]

        record.append(M)
        if dispmcmc:
            print(M)

        pp += 1

        if iter < 2000:
            tmp_B = calculate_B(c, N)
            final_B = final_B + tmp_B
            final_rd.append(M)

    return final_B, final_rd

if __name__ == '__main__':
    # Compute metric and estimate hyperpriors2
    S = toy_example()  # Use toy example
    # S = np.load('data/S_tmp_40.npy')  # Use similarity matrix from Git
    d = top95_eigenvalues(S)
    r0 = 2 * 3 / d
    s0 = 2 * 4 / d
    theta_vect = np.array([1000, 2000, 3000, 4000, 5000])
    ksi = 1
    M = 30

    final_B, final_rd = WishartCluster(d, theta_vect, r0, s0, ksi, M, S, iter=5000)

    end=True