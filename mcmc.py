import numpy as np
import matplotlib.pyplot as plt
from numba import jitclass
from numba import int32, float64

from utils import *
from stats import crp
from generate_data import GMM, toy_example
from computations import *
import tqdm


class BayesianClustering:
    def __init__(self, S, **kwargs):
        self.S = S
        self.d = kwargs['d']
        self.r0 = kwargs['r0']
        self.s0 = kwargs['s0']
        self.theta = kwargs['theta']
        self.ksi = kwargs['ksi']
        self.N_theta = self.theta.shape[0]
        self.K = kwargs['K_init']

        self.n = S.shape[0]
        self.c = None  # Cluster indices
        self.cn = None  # Cluster sizes
        self.B = None
        self.B_samples = []  # Membership matrices generated
        self.nb_clusters = []

    def posterior_theta(self, theta, c, cn):
        '''
        Compute the posteriors of the theta's. For computational stability, the log-likelihood is first computed
        c: Cluster indices
        cn: Cluster sizes
        K: Number of clusters
        '''
        log_prod = 0
        sum = 0

        for j in range(len(cn)):
            log_prod += np.log(1 + theta * cn[j])
            Ij = (c == j)
            sum += (theta / (1 + cn[j] * theta)) * np.vdot(Ij, np.dot(self.S, Ij))

        log_lk = - 0.5 * self.d * (log_prod + (self.n + self.r0) *
                                   (np.log(0.5 * self.d) + np.log(np.trace(self.S) - sum + self.s0)))

        return log_lk

    def sample(self, values, p):
        '''
        Sample from values with probability p, while handling NaNs
        :param values: vector of values to sample from
        :param p: probabilities associated to values
        :return: a sample together with its index in the list
        '''
        count_nan = np.count_nonzero(np.isnan(p))  # Account for nan values
        if count_nan == 0:
            sample = np.random.choice(values, p=p)
            index = np.where(values == sample)[0][0]
        elif count_nan < len(values):
            mask = np.isnan(p)
            temp_values = values[~mask]
            p = p[~mask]
            p /= p.sum()
            sample = np.random.choice(temp_values, p=p)
            index = np.where(temp_values == sample)[0][0]
        else:
            sample = np.random.choice(values)
            index = np.where(values == sample)[0][0]

        return sample, index

    def mcmc_sweep(self):
        ''' One sweep of the MCMC algorithm '''
        # Step 1: Sample theta
        log_likelihood_theta = np.zeros(self.N_theta)
        for i in range(self.N_theta):
            theta = self.theta[i]
            assert self.cn.sum() == self.n
            log_lk = self.posterior_theta(theta, self.c, self.cn)
            log_likelihood_theta[i] = log_lk

        log_likelihood = log_likelihood_theta - np.max(log_likelihood_theta)
        likelihood_theta = np.exp(log_likelihood)
        likelihood_theta /= np.sum(likelihood_theta)  # Normalize likelihood

        theta, index_theta = self.sample(self.theta, likelihood_theta)
        print(log_likelihood)

        # Step 2: Update membership vector
        for i in range(self.n):
            c_likelihood = np.zeros(self.K + 1)
            log_likelihood_theta = np.zeros(self.K + 1)

            # Test different clustering configurations for data i
            for k in range(self.K):
                c_temp = self.c.copy()
                c_temp[i] = k
                cn_temp = np.unique(c_temp, return_counts=True)[1]

                assert cn_temp.sum() == self.n
                log_lk = self.posterior_theta(theta, c_temp, cn_temp)
                log_likelihood_theta[k] = log_lk

            # Test apparition of a new cluster
            c_temp = self.c.copy()
            c_temp[i] = self.K
            cn_temp = np.unique(c_temp, return_counts=True)[1]

            assert cn_temp.sum() == self.n
            log_lk = self.posterior_theta(theta, c_temp, cn_temp)
            log_likelihood_theta[self.K] = log_lk

            log_likelihood_theta -= np.max(log_likelihood_theta)
            likelihood_theta = np.exp(log_likelihood_theta)
            likelihood_theta /= np.sum(likelihood_theta)

            for k in range(self.K):
                c_likelihood[k] = likelihood_theta[k] * self.cn[k] / (self.n - 1 - self.ksi)
            c_likelihood[self.K] = likelihood_theta[self.K] * self.ksi / (self.n - 1 - self.ksi)

            c_likelihood /= np.sum(c_likelihood)

            self.c[i] = self.sample(np.arange(self.K + 1), c_likelihood)[0]

        self.B = membership_c2B(self.c)
        self.c = membership_B2c(self.B)
        counter = np.unique(self.c, return_counts=True)
        self.K = len(counter[0])
        self.cn = counter[1]

        return self.B

    def mcmc_sampler(self, iter, burn_in=100):
        ''' MCMC posterior sampling algorithm; Generate a sequence of membership matrices'''
        # Initialization
        self.c = np.random.randint(self.K, size=self.n)
        self.cn = np.zeros(self.K)
        self.nb_clusters.append(self.K)
        for i in range(self.K):
            self.cn[i] = np.sum(self.c == i)

        # Iterations
        for it in range(iter):
            print('{}: {}'.format(it, self.K))
            B = self.mcmc_sweep()
            if it > burn_in:
                self.B_samples.append(B)
                self.nb_clusters.append(self.K)

        return self.B_samples, self.nb_clusters

    def extrinsic_mean(self, B_samples):
        ''' Compute extrinsic mean of the sequence of membership matrices generated by MCMC'''
        M = len(B_samples)

        # Step 1: Find the mode of clusters
        c_samples = [membership_B2c(B) for B in B_samples]
        nb_clusters = [len(np.unique(c_samples[i])) for i in range(len(c_samples))]
        count_nb_clusters = np.unique(nb_clusters, return_counts=True)
        mode = count_nb_clusters[0][count_nb_clusters[1].argmax()]

        # Step 2: Calculate euclidean mean and threshold it on the set of membership matrices
        B_mean = np.array(B_samples).mean(axis=0)
        k = self.n
        iter = M

        B_star_list = []
        while k != mode and iter >= 0:
            print(iter)
            B_mean_copy = B_mean.copy()
            J = np.arange(self.n)
            B_star = np.zeros((self.n, self.n))
            iter -= 1
            t_star = iter / M

            J_copy = J.copy()
            for j in J:
                if j not in J_copy:
                    pass
                v = (B_mean_copy[j, :] > t_star)
                C = np.where(v == 1)[0]
                J_copy = np.setdiff1d(J_copy, C)

                for i in C:
                    B_star[i, :], B_star[:, i] = v.copy(), v.copy()
                    B_mean_copy[i, :], B_mean_copy[:, i] = np.zeros(self.n), np.zeros(self.n)

            c = membership_B2c(B_star.copy())
            k = len(np.unique(c))
            B_star_list.append(B_star.copy())

        return B_star_list[-1]


if __name__ == '__main__':

    # Compute metric and estimate hyperpriors2
    # S = toy_example()  # Use toy example
    # S = np.load('data/S_tmp_40.npy')  # Use similarity matrix from Git
    data = np.load('data/sim_data_3.npy')
    D = distance_matrix(data)
    sigma_sq = D.max() / 12
    S = similarity_matrix(data, sigma_sq)

    d_eb = top95_eigenvalues(S)
    r = 3
    s = 4

    # Set parameters
    params = {'d': d_eb,
              'r0': 2 * r / d_eb,
              's0': 2 * s / d_eb,
              'theta': np.array([1000, 2000, 3000, 4000, 5000]),
              'ksi': 10,
              'K_init': 10
              }

    # Define and run Bayesian Clustering algorithm
    self = BayesianClustering(S, **params)
    B_samples, nb_clusters = self.mcmc_sampler(8000, 1000)
    #np.save('outputs/conv_3_ksi_small_theta_big.npy', np.array(nb_clusters))
    np.save('outputs/B_3_ksi_small_theta_big_bis_5c.npy', np.array(B_samples))
    B_star = self.extrinsic_mean(B_samples)
    np.save('outputs/B_3_star_1_ksi_small_theta_big_bis_5c.npy', B_star)

    #B_samples = np.load('outputs/B_1_ksi_big_theta_small.npy')
    #B_star = np.load('outputs/B_star_1_ksi_big_theta_small.npy')

    c_samples = [membership_B2c(B) for B in B_samples]
    nb_clusters = [len(np.unique(c_samples[i])) for i in range(len(c_samples))]
    count_nb_clusters = np.unique(nb_clusters, return_counts=True)
    mode = count_nb_clusters[0][count_nb_clusters[1].argmax()]

    counts = np.zeros(16)
    for i in range(len(count_nb_clusters[0])):
        counts[count_nb_clusters[0][i]] = count_nb_clusters[1][i]
    counts /= counts.sum()

    # Show the distribution of the number of clusters
    plt.title('Number of clusters - Estimated distribution')
    plt.bar(np.arange(len(counts)), counts)
    plt.xticks(np.arange(len(counts)))
    plt.xlabel('Number of clusters')
    plt.ylabel('Proportion')
    plt.show()

    # Show clustering configuration (for Euclidean data)
    c_star = membership_B2c(B_star)
    plot_data_with_clusters(data, c_star)

    end = True