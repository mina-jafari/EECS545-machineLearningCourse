from __future__ import division
import numpy as np
from scipy.special import gammaln, polygamma
from math import log
from matplotlib import pyplot as plt
from numpy import linalg as LA
import copy

def computeF(alpha, P, N, m):
    sumAlpha = np.sum(alpha)
    B = 0.0
    D = 0.0
    for k in range(m):
        C = 0.0
        B += gammaln(alpha[k])
        for j in range(N):
            C += log(P[j][k])
        D += (alpha[k]-1) * (1.0/N)*C
    F = N * (gammaln(sumAlpha) - B + D)
    return F

def computeFPrime(alpha, P, N, m):
    Fprime = np.zeros((m))
    sumAlpha = np.sum(alpha)
    A = polygamma(0, sumAlpha)
    for k in range(m):
        C = 0.0
        for j in range(N):
            C += log(P[j][k])
        Fprime[k] = N * (A - polygamma(0, alpha[k]) + (1.0/N)*C)
    return Fprime

def computeHessian(alpha, P, N, m):
    sumAlpha = np.sum(alpha)
    tempHessians = np.zeros((m))
    for i in range(m):
        tempHessians[i] = -1 * N * (polygamma(1, alpha[i]))
    c = N * polygamma(1, sumAlpha)
    Q = np.diag(tempHessians)
    return (Q, c)


def main():
    N = 1000
    m = 5
    alpha_t = np.array([10, 5, 15, 20, 50])
    P = np.random.dirichlet(alpha_t, N)
    alpha_new = np.array([1, 1, 1, 1, 1])
    F = computeF(alpha_new, P, N, m)
    n = 0
    J = np.empty([1])
    ones = np.ones((m))
    onesT = np.transpose(ones)
    truePar = computeF(alpha_t, P, N, m)
    
    while True:
        FOld = F
        alpha_old = copy.deepcopy(alpha_new)
        Fprime = computeFPrime(alpha_old, P, N, m)
        Q, c = computeHessian(alpha_old, P, N, m)
        Qinv = LA.inv(Q)
        numerator = np.dot( np.dot(Qinv, onesT), np.dot(ones, Qinv)  )
        denom = (1.0/c) + np.dot(ones, np.dot(Qinv, onesT))
        middleTerm = np.subtract(Qinv, np.divide(numerator, denom))
        alpha_new = np.subtract(alpha_old, 0.001*np.dot(middleTerm, Fprime))
        F = computeF(alpha_new, P, N, m)

        if (abs(F - FOld) < 0.0001):
            print "***Alpha: ", alpha_new
            break

        n += 1
        J = np.append(J, F)
    
    J = np.delete(J, 0)
    x = range(1, n+1)
    plt.plot(x, J, 'r')
    plt.ylabel('F')
    plt.xlabel('# of iterations')
    plt.savefig('plot-1.png')
    
    plt.plot((1, n+1), (truePar, truePar), 'k-')
    plt.savefig('plot-2.png')

if __name__ == "__main__":
    main()
