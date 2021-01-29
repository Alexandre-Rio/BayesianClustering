import numpy as np
import matplotlib.pyplot as plt
from numba import jit

__all__ = ['membership_B2c', 'membership_c2B', 'plot_save_grid', 'plot_data_with_clusters']

def membership_c2B(c):
    '''
    Convert an array of membership vectors to a membership matrix
    :param c: nx1 array-like
    :return: B, nxn membership matrix
    '''
    n = len(c)
    B = np.eye(n)
    # c_sorted = c.copy()
    # c_sorted.sort()
    for i in range(n):
        for j in range(i+1, n):
            if c[i] == c[j]:
                B[i, j] = B[j, i] = 1
    return B

def membership_B2c_ordered(B):
    '''
    Convert a (ordered) membership matrix to an array of membership vectors
    :param c: nx1 array-like
    :return: B, nxn membership matrix
    '''
    B_temp = B.copy()
    len_clusters = []
    while B_temp.min() < 1:
        i = 0
        while B_temp[i, 0] > 0:
            i += 1
        len_clusters.append(i)
        B_temp = B_temp[i:, i:]
    len_clusters.append(B_temp.shape[0])
    c = []
    cluster = 1
    for k in len_clusters:
        c += [cluster] * k
        cluster += 1
    return c

def membership_B2c(B):
    '''
    Convert a (unordered) membership matrix to an array of membership vectors
    :param c: nx1 array-like
    :return: B, nxn membership matrix
    '''
    n = B.shape[0]
    indices_to_check = np.arange(n)
    clusters = []
    c = np.zeros(n, dtype=int)

    while len(indices_to_check) > 0:
        idx = indices_to_check[0]
        idx_in_cluster = np.where(B[idx, :] == 1)[0]
        clusters.append(idx_in_cluster)
        indices_to_check = np.setdiff1d(indices_to_check, idx_in_cluster)

    for cluster, idx_in_cluster in enumerate(clusters):
        c[idx_in_cluster] = cluster

    return c

def plot_save_grid(dataset, size=4):
    """
    Plot size x size grid to visualize data
    """
    # Choose random shapes
    n = dataset.shape[2]
    shapes_idx = np.random.randint(0, n + 1, size=size ** 2)

    x = dataset[0, :, shapes_idx]
    y = dataset[1, :, shapes_idx]

    fig = plt.figure(figsize=(size, size))
    fig.suptitle("Some shapes from the dataset")
    gridspec = fig.add_gridspec(size, size)
    for idx in range(size ** 2):
        ax = fig.add_subplot(gridspec[idx])
        ax.plot(x[idx], y[idx])
        ax.set_axis_off()
    fig.savefig('images/random_shapes.png')

def plot_data_with_clusters(data, c):
    ''' Plot euclidean data with estimated labels (memberships c) '''
    color_dict = {0: 'b', 1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y', 6: 'k', 7: 'w',
                  8: 'darkcyan', 9: 'darkviolet', 10: 'darkorange', 11: 'dodgerblue'}
    marker_dict = {0: 'o', 1: 'x', 2: 'v', 3: '^', 4: '*', 5: 's', 6: 'h', 7: '+', 8: 'D', 9: '1', 10: 'p', 11: '8',
                   12: 'X'}

    scatter_x = data[:, 0]
    scatter_y = data[:, 1]

    fig, ax = plt.subplots()
    plt.suptitle("Estimated clustering configuration")
    for g in np.unique(c):
        ix = np.where(c == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], c=color_dict[g], marker=marker_dict[g], label=g, s=100)
    ax.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    c = np.array([0, 0, 1, 0, 2, 1, 2, 2])
    c.sort()
    B = membership_c2B(c)

    end=True


