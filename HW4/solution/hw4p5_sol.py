from __future__ import division
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

def loglike(y, x, pi, w, b, sigma):
    N = len(x)
    K = len(pi)
    gamma = np.zeros((N,K))
    for k in range(K):
         gamma[:,k] = pi[k]*norm.pdf(y,  w[k]*x + b[k], sigma[k])
    ell = np.sum(np.log(np.sum(gamma,1)))
    gamma = gamma / np.sum(gamma,1).reshape((N,1))
    pi = np.mean(gamma,0)
    x_tilde = np.concatenate((x.reshape((N,1)), np.ones((N,1))), 1)
    for k in range(K):
        wb = np.linalg.lstsq(np.sqrt(gamma[:,k]).reshape((N,1))*x_tilde, np.sqrt(gamma[:,k])*y)[0]
        w[k] = wb[0]
        b[k] = wb[1]
        sigma[k] = np.sqrt(np.sum(gamma[:,k]*(y - x_tilde.dot(wb))**2)/np.sum(gamma[:,k]))
    return ell, pi, w, b, sigma

N = 500
x = np.random.rand(N)
pi0 = np.array([0.7, 0.3])
w0 = np.array([-2, 1])
b0 = np.array([0.5, -0.5])
sigma0 = np.array([.4, .3])
y = np.zeros_like(x)

for i in range(N):
    k = 0 if np.random.rand() < pi0[0] else 1
    y[i] = w0[k]*x[i] + b0[k] + np.random.randn()*sigma0[k]

pi = np.array([0.5, 0.5])
w = np.array([1.0, -1.0])
b = np.array([0.0, 0.0])
sigma = np.array([np.std(y), np.std(y)])

ell = -1e8
ells = []
for i in range(1000):
    ell0 = ell
    ell, pi, w, b, sigma = loglike(y, x, pi, w, b, sigma)
    ells.append(ell)
    if np.abs(ell - ell0) < 1e-4:
        break

plt.plot(ells)
plt.show()
    
plt.scatter(x, y, c='r', marker='x')
plt.plot(np.array([0, 1]), np.array([b[0], w[0]+b[0]]))
plt.plot(np.array([0, 1]), np.array([b[1], w[1]+b[1]]))
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([-2,1])
plt.show()