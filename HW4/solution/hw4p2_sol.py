from __future__ import division
import numpy as np
from scipy.special import gammaln, polygamma
from matplotlib import pyplot as plt

def loglike(a, X):
    N = len(X)
    t = np.mean(np.log(X), 0)
    eta = alpha - 1
    A = -N*gammaln(np.sum(a)) + N*np.sum(gammaln(a))
    J = N*eta.dot(t) - A
    dJ = N*(t + polygamma(0, np.sum(a)) - polygamma(0, a))
    q = -1/polygamma(1,a)
    c = polygamma(1, np.sum(a))
    H_inv = (np.diag(q) - np.outer(q,q)*c/(1 + c*np.sum(q)))/N
    return J, dJ, H_inv
    
N = 1000
m = 5
alpha = np.array([10, 5, 15, 20, 50])
X = np.random.dirichlet(alpha, N)

a = np.ones(m)

like = []
while True:
    J, dJ, H_inv = loglike(a, X)
    like.append(J)
    if len(like) > 1 and np.abs(like[-1] - like[-2]) < 1e-4:
        break
    a = a - H_inv.dot(dJ)

print(a)

J, _, _ = loglike(alpha, X)
plt.plot([0, len(like)-1], [J, J])
plt.plot(like)
plt.show()