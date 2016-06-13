from __future__ import division
import numpy as np
from numpy import linalg as LA
from math import log, pi
from matplotlib import pyplot as plt

# Generate the data according to the specification in the homework description
N = 10000

# Here's an estimate of gamma for you
G = lambda x: np.log(np.cosh(x))
gamma = np.mean(G(np.random.randn(10**6)))

s1 = np.sin((np.arange(N)+1)/200)
s2 = np.mod((np.arange(N)+1)/200, 2) - 1
S = np.concatenate((s1.reshape((1,N)), s2.reshape((1,N))), 0)

A = np.array([[1,2],[-2,1]])

X = A.dot(S)
# TODO: Implement ICA using a 2x2 rotation matrix on a whitened version of X
lamdaInit, eVectors = LA.eig(np.dot(X, X.transpose()))
lamda = np.zeros((2, 2))
np.fill_diagonal(lamda, lamdaInit)
D = np.sqrt(N) * np.dot(LA.inv(np.sqrt(lamda)), LA.inv(eVectors))
whitenedX = np.dot(D, X)

jays = []
row1 = []
row2 = []
Theta = np.linspace(0.0, pi/2, num=20)
minTheta = 0.0
for theta in Theta:
    wMat = np.zeros((2, 2))
    wMat[0, 0] = wMat[1, 1] = np.cos(theta)
    wMat[0, 1] = -1 * np.sin(theta)
    wMat[1, 0] = np.sin(theta)
    yMat = np.dot(wMat, whitenedX)
    meanY1 = np.mean(G(yMat[0, :]))
    meanY2 = np.mean(G(yMat[1, :]))
    row1.append(meanY1)
    row2.append(meanY2)

    diff1 = (meanY1 - gamma)**2
    diff2 = (meanY2 - gamma)**2
    jay = diff1 + diff2
    jays.append(jay)
    
minIndex = np.argmax(jays)
minTheta = Theta[minIndex]
wMat[0, 0] = wMat[1, 1] = np.cos(minTheta)
wMat[0, 1] = -1 * np.sin(minTheta)
wMat[1, 0] = np.sin(minTheta)
yMat = np.dot(wMat, whitenedX)

plt.plot(Theta, jays, 'r')
plt.ylabel('J(y)')
plt.xlabel('$\\theta$')
plt.savefig('plot-1.png')
plt.clf()

plt.plot(yMat[0, :], 'b')
plt.plot(yMat[1, :]+4, 'g')
plt.savefig('plot-2.png')
