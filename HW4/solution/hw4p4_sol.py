from __future__ import division
from scipy.ndimage import imread
import numpy as np
from matplotlib import pyplot as plt

mandrill = imread('mandrill.png', mode='RGB').astype(float)
N = int(mandrill.shape[0])

M = 2
k = 64

X = np.zeros((N**2//M**2, 3*M**2))

for i in range(N//M):
    for j in range(N//M):
        X[i*N//M+j,:] = mandrill[i*M:(i+1)*M,j*M:(j+1)*M,:].reshape(3*M**2)

##
centers = X[np.random.choice(range(len(X)), k, replace=False),:]

dists = np.sum(X**2, 1).reshape(len(X),1) - 2*X.dot(centers.T)  + np.sum(centers.T**2, 0)
c = np.argmin(dists, 1)
c0 = c
for i in range(k):
    centers[i,:] = np.mean(X[c == i, :], 0)

obj = []
while True:
    dists = np.sum(X**2, 1).reshape(len(X),1) - 2*X.dot(centers.T)  + np.sum(centers.T**2, 0)
    c = np.argmin(dists, 1)
    obj.append(np.sum(dists[np.arange(len(c)),c]))
    if all(c0 == c):
        break
    c0 = c
    for i in range(k):
        centers[i,:] = np.mean(X[c == i, :], 0)

plt.plot(obj)
plt.show()

Y = np.zeros_like(mandrill)

for i in range(N//M):
    for j in range(N//M):
        Y[i*M:(i+1)*M,j*M:(j+1)*M,:] = centers[c[i*N//M+j],:].reshape((M,M,3))
        
err = np.sum(np.sum(np.sum(np.abs(Y - mandrill))))/255/N**2/3
print(err)

plt.imshow(Y/255)
plt.show()

plt.imshow((Y - mandrill + 128)/255)
plt.show()




