from __future__ import division
from scipy.ndimage import imread
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA

# Load the mandrill image as an NxNx3 array. Values range from 0.0 to 255.0.
mandrill = imread('mandrill.png', mode='RGB').astype(float)
N = int(mandrill.shape[0])

M = 2
k = 64

# Store each MxM block of the image as a row vector of X
X = np.zeros((N**2//M**2, 3*M**2))
for i in range(N//M):
    for j in range(N//M):
        X[i*N//M+j,:] = mandrill[i*M:(i+1)*M,j*M:(j+1)*M,:].reshape(3*M**2)

def clusterPoints(X, mu, k, M):
    Z = np.zeros((X.shape[0]), dtype=int)
    for i in range(X.shape[0]):
        A = np.zeros((k))
        for j in range(k):
            A[j] = LA.norm(np.subtract(X[i,:], mu[j,:]))
        Z[i] = np.argmin(A)
    return Z

def updateMu(Z, mu, k, M):
    counter = np.zeros((k), dtype=int)
    sumX = np.zeros((k, 3*M**2))
    for j in range(k):
        for i in range(X.shape[0]):
            if Z[i] == j:
                counter[j] += 1
                sumX[j,:] += X[i,:]
        if counter[j] != 0:
            mu[j,:] = sumX[j,:] / counter[j]
    return mu

oldmu = np.zeros((k, 3*M**2))
mu = np.zeros((k, 3*M**2))
mu = X[np.random.choice(X.shape[0], k, replace=False)]

it = 10
J = np.empty([1])
n = 0
while True:
    if np.allclose(mu, oldmu, rtol=1e-9, atol=1e-9) and n > 5:
        break
    else:
        n += 1
        oldmu = mu
        Z = clusterPoints(X, mu, k, M)
        mu = updateMu(Z, mu, k, M)
        # draw the plot
        temper = 0
        for i in range(0, X.shape[0]):
            for j in range(0, k):
                if Z[i] == j:
                    temper += LA.norm(np.subtract(X[i,:], mu[j,:]))
        J = np.append(J, temper)

J = np.delete(J, 0)
x = range(1, n+1)
plt.plot(x, J, 'b')
plt.ylabel('J')
plt.xlabel('# of iterations')
plt.savefig('foo.png')


# reconstructing the image
temp = np.zeros((X.shape[0], 3*M**2))
for i in range(X.shape[0]):
    temp[i,:] = mu[Z[i],:]

mandrill_cons = np.zeros((N, N, 3))
for i in range(N//M):
    for j in range(N//M):
        mandrill_cons[i*M:(i+1)*M, j*M:(j+1)*M, 0:3] = temp[i*N//M+j,:].reshape(2,2,3)

plt.imshow(mandrill_cons/255)
plt.savefig('foo-2.png')

plt.imshow((mandrill - mandrill_cons + 128)/255)
plt.savefig('foo-3.png')


mae = np.sum(np.absolute(mandrill - mandrill_cons)) / (255 * 3*N**2)
print mae, "mae"

