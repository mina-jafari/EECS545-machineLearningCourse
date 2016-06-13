from __future__ import division
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import seaborn

# Generate the data according to the specification in the homework description

N = 10000
G = lambda x: np.log(np.cosh(x))
gamma = np.mean(G(np.random.randn(10**6)))

s1 = np.sin((np.arange(N)+1)/200)
s2 = np.mod((np.arange(N)+1)/200, 2) - 1
S = np.concatenate((s1.reshape((1,N)), s2.reshape((1,N))), 0)
S = S - np.mean(S,1).reshape((2,1))

A = np.array([[1,2],[-2,1]])

X = A.dot(S)

D = sp.linalg.sqrtm(np.linalg.inv(X.dot(X.T)/N))

X_w = D.dot(X)

W = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
J = lambda t: np.sum((np.mean(G(W(t).dot(X_w)),1) - gamma)**2)

ts = np.linspace(0, np.pi/2, 1000)
Js = [J(t) for t in ts]

t_sol = max(zip(Js,ts), key=lambda x: x[0])[1]

Y = W(t_sol).dot(X_w)

plt.plot(ts, Js)
plt.show()

plt.plot(Y[0,:])
plt.plot(Y[1,:]-4)
plt.show()