import numpy as np
import scipy.io

#S = scipy.io.loadmat('data/S_tmp_40.mat')['S']
#np.save('data/S_tmp_40.npy', S)

raw_shape_data = scipy.io.loadmat('data/Dataset.mat')
shape_data_C = raw_shape_data['C']
np.save('data/shape_data_C.npy', shape_data_C)
shape_data_Dataset = raw_shape_data['Dataset']
np.save('data/shape_data_Dataset.npy', shape_data_Dataset)