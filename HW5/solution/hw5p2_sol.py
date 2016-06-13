from __future__ import division
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
# import seaborn # uncomment for nice-looking plots

N = 5000

x1 = [0.0]
x2 = []

for i in xrange(N-1):
    x2.append(np.sqrt(3/4)*np.random.randn() + (x1[-1] + 1)/2 )
    x1.append(np.sqrt(3/4)*np.random.randn() + (x2[-1] + 1)/2 )

x2.append(np.sqrt(3/4)*np.random.randn() + (x1[-1] + 1)/2 )

x = np.linspace(-3,5,100)
plt.hist(x1, 20, normed=True)
plt.plot(x, norm.pdf(x,1,1))
plt.show()

plt.hist(x2, 20, normed=True)
plt.plot(x, norm.pdf(x,1,1))
plt.show()