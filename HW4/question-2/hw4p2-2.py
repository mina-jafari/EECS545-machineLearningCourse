from __future__ import division
import numpy as np
from scipy.special import gammaln, polygamma
from math import log
from matplotlib import pyplot as plt
from numpy import linalg as LA
import copy

# Generate the data according to the specification in the homework description
   
N = 1000
m = 5
alpha = np.array([10, 5, 15, 20, 50])
P = np.random.dirichlet(alpha, N)

# TODO: Implement the Newton-Raphson algorithm for estimating the parameters of
# the Dirichlet distribution given observances (rows of P).


def computeF_fPrime(alpha, P, N, m):
    F = np.zeros((N, m))
    Fprime = np.zeros((N, m))
    sumAlpha = np.sum(alpha)
    for i in range(N):
        A = 0.0
        B = 0.0
        C = 0.0
        D = 0.0
        for k in range(m):
#            sumAlpha += alpha[k]
            #B += gammaln(alpha[k])
            B = gammaln(alpha[k])
            for j in range(N):
                C += log(P[j][k])
            #D += (alpha[k]-1) * C
            D = (alpha[k]-1) * C
            A = gammaln(sumAlpha)
        #A = gammaln(sumAlpha)
        #F[i] = N * (A - B + (1/N)*D)
            F[i][k] = N * (A - B + (1/N)*D)
            Fprime[i][k] = N * (polygamma(1, A) - polygamma(1, B) + C)
    return (F, Fprime)

def computeHessian(alpha, P, N, m):
    sumAlpha = np.sum(alpha)
    #Hessians = np.zeros((N, m, m))
    Hessians = np.empty((1, m, m))
    for k in range(N):
        Hess = np.zeros((1, m, m))
        for i in range(m):
            for j in range(m):
                Hess[0][i][j] = N * (polygamma(2, sumAlpha) - (i == j) * polygamma(2, alpha[j]))
        Hessians = np.append(Hessians, Hess, axis=0)
    Hessians = np.delete(Hessians, 0, 0)
#    print Hessians
    return Hessians


alpha_new = np.array([1, 1, 1, 1, 1])
F, Fprime = computeF_fPrime(alpha_new, P, N, m)
#print F
FOld = np.zeros((N, m))
#print FOld
n = 0
J = np.empty([1, 5])

while True:
    if (np.allclose(FOld, F)): # tolerance
        break
    else:
        FOld = copy.deepcopy(F)
        alpha_old = copy.deepcopy(alpha_new)
        F, Fprime = computeF_fPrime(alpha_old, P, N, m)
        print F.shape
        print Fprime.shape
        hess = computeHessian(alpha_old, P, N, m)
        print hess[0].shape
        print LA.pinv(hess[0])
        alpha_new = np.subtract(alpha_old, np.dot(LA.pinv(hess[i]), Fprime[i]))
        n += 1
        print n
        J = np.append(J, F)

J = np.delete(J, 0, 0)
x = range(1, n+1)
plt.plot(x, J, 'r')
plt.ylabel('F')
plt.xlabel('# of iterations')
plt.savefig('plot.png')



