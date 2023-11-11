import numpy as np
a = np.load('./dataset_5/causal_prior.npy')

if a[26][35] == 1:
    print("yes")