import numpy as np
import matplotlib.pyplot as plt

from utils import *
from stats import crp
from generate_data import GMM, toy_example
from computations import *

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
        self.B_samples = []  # Membership matrices generated

    def posterior_theta(self, c, cn, K):
        '''
        Compute the posteriors of the theta's. For computational stability, the log-likelihood is first computed
        c: Cluster indices
        cn: Cluster sizes
        K: Number of clusters
        '''

        log_likelihood = np.zeros(self.N_theta)

        for i in range(self.N_theta):
            log_prod = 0
            sum = 0
            theta = self.theta[i]

            for j in range(K):
                log_prod += np.log(1 + theta * cn[j])
                Ij = (c == j)
                sum += (theta / (1 + cn[j] * theta)) * np.vdot(Ij, np.dot(self.S, Ij))

            log_lk = 0.5 * self.d * (- log_prod + (self.n + self.r0) *
                                     (np.log(0.5 * self.d) + np.log(np.trace(self.S) - sum + self.s0)))
            log_likelihood[i] = log_lk

        log_likelihood = log_likelihood - np.max(log_likelihood)
        likelihood = np.exp(log_likelihood)
        likelihood /= np.sum(likelihood)  # Normalize likelihood

        return likelihood

    def B_cond_ksi(self, c):
        ''' Compute the posterior of B conditional on ksi '''
        likelihood = self.ksi / (self.n - 1 + self.ksi)
        for i in range(1, self.n):
            c_curr = c[i]
            c_old = c[0: i]
            likelihood *= crp(c_curr, c_old, self.ksi)
        return likelihood

    def posterior_B(self, c, posterior_theta):
        ''' Compute the posterior of B conditional on S, theta, d and ksi '''
        B_cond_ksi = self.B_cond_ksi(c)
        return posterior_theta * B_cond_ksi / self.N_theta

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
        posteriors_theta = self.posterior_theta(self.c, self.cn, self.K)
        # Handle Nans
        theta, index_theta = self.sample(self.theta, posteriors_theta)

        # Restart here !!!

        if not np.isnan(posteriors_theta[index_theta]):
            posterior_theta = posteriors_theta[index_theta]  # Posterior value used to compute the posterior of B
        else:
            posterior_theta = 1

        # Step 2: Update membership vector
        for i in range(self.n):  # Iterate through observations
            pi = np.zeros(self.K + 1)
            for j in range(1, self.K + 2):  # Iterate though clusters + new cluster
                c_new = self.c.copy()
                c_new[i] = j
                pi[j - 1] = self.posterior_B(c_new, posterior_theta)
            norm_pi = pi / pi.sum()
            self.c[i] = np.random.choice(np.arange(1, self.K + 2), p=norm_pi)  # Update c_i
        self.K = len(np.unique(self.c))  # Update number of clusters

        # Step 3: Obtain new membership matrix
        B = membership_c2B(self.c)

        return B


    def mcmc_sampler(self, iter, burn_in=100):
        ''' MCMC posterior sampling algorithm; Generate a sequence of membership matrices'''
        # Initialization
        self.c = np.random.randint(1, self.K + 1, size=self.n)
        # self.c.sort()
        B = membership_c2B(self.c)
        self.B_samples.append(B)

        # Iterations
        for it in range(1, iter):
            print(it)
            B = self.mcmc_sweep()
            self.B_samples.append(B)

        if len(self.B_samples) > burn_in:
            B_samples = self.B_samples[burn_in:]
        else:
            B_samples = None

        return B_samples

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
            J = np.arange(1, self.n + 1)
            B_star = np.zeros((self.n, self.n))
            iter -= 1
            t_star = iter / M

            for j in J:
                v = (B_mean[j - 1, :] > t_star)
                C = np.where(v == 1)[0]
                J = np.setdiff1d(J, C + 1)

                for i in C:
                    B_star[i, :], B_star[:, i] = v, v
                    B_mean[i, :], B_mean[:, i] = np.zeros(self.n), np.zeros(self.n)

            c = membership_B2c(B_star)
            k = len(np.unique(c))
            B_star_list.append(B_star)

        return B_star_list[-1]


if __name__ == '__main__':

    # Compute metric and estimate hyperpriors2
    S = toy_example() # Use toy example
    # S = np.load('data/S_tmp_40.npy')  # Use similarity matrix from Git
    d_eb = top95_eigenvalues(S)
    r = 3
    s = 4

    # Set parameters
    params = {'d': d_eb,
              'r0': 2 * r / d_eb,
              's0': 2 * s / d_eb,
              'theta': np.array([1000, 2000, 3000, 4000, 5000]),
              'ksi': 1,
              'K_init': 10
              }

    # Define and run Bayesian Clustering algorithm
    self = BayesianClustering(S, **params)
    B_samples = self.mcmc_sampler(4000, 1000)
    np.save('outputs/B_samples3.npy', np.array(B_samples))
    #B_samples = np.load('outputs/B_samples3.npy')
    #B_samples = B_samples[1000:]

    c_samples = [membership_B2c(B) for B in B_samples]
    nb_clusters = [len(np.unique(c_samples[i])) for i in range(len(c_samples))]
    count_nb_clusters = np.unique(nb_clusters, return_counts=True)
    mode = count_nb_clusters[0][count_nb_clusters[1].argmax()]

    # Show the distribution of the number of clusters
    plt.bar(count_nb_clusters[0], count_nb_clusters[1])
    plt.show()

    end = True