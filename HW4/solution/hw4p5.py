from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

# Generate the data according to the specification in the homework description

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


# TODO: Implement the EM algorithm for Mixed Linear Regression based on observed
# x and y values.


# Here's the data plotted
plt.scatter(x, y, c='r', marker='x')
plt.show()

    