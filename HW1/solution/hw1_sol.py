import numpy as np

A = np.loadtxt("A.csv",delimiter=",")
#A = np.load('A.npy')

# TODO: perform SVD on A, zero out top 3 singular values to obtain a new matrix B and compute ||A-B||^2  

U, eigs, V = np.linalg.svd(A,full_matrices=False)

print eigs[0], eigs[1], eigs[2]

num_eigs_keep = 3

eigs[num_eigs_keep:] = 0

B = U.dot(np.diag(eigs).dot(V))

#mse = ((A - B) ** 2).mean(axis=None)

se = ((A - B) ** 2).sum(axis=None)

print se, se*0.99, se*1.01
