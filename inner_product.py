import numpy as np

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

def compute_curve_from_srvf(q):
    ''' Compute curve from SRVF '''
    pass

if __name__ == '__main__':
    data = np.load('data/shape_data_C.npy')

    curve1 = data[:, :, 0]
    curve2 = data[:, :, 1]
    q1 = compute_srvf(curve1)
    q2 = compute_srvf(curve2)

    end=True