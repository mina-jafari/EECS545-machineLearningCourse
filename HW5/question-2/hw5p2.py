import numpy as np
import matplotlib.pyplot as plt
from math import pi
import matplotlib.mlab as mlab

def updateX1(x2):
    sigma = np.sqrt(3.0/4.0)
    mu = 0.5*x2+0.5
    x1 = sigma * np.random.randn() + mu
    return x1


def updateX2(x1):
    sigma = np.sqrt(3.0/4.0)
    mu = 0.5*x1+0.5
    x2 = sigma * np.random.randn() + mu
    return x2

def main():
    x1 = 0.0
    N = 5000
    x1Values = np.zeros(N)
    x2Values = np.zeros(N)

    for i in range(N):
        x2 = updateX2(x1)
        x2Values[i] = x2
        x1 = updateX1(x2)
        x1Values[i] = x1

    #plot
    plt.hist(x1Values, bins=20, alpha=0.6, label='calc. marginal', normed=True)
    x = np.linspace(-3,5)
    plt.plot(x,mlab.normpdf(x,1.0,1.0), lw=3)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.savefig('px1')
    plt.clf()

    plt.hist(x2Values, bins=20, alpha=0.6, label='calc. marginal', normed=True)
    x = np.linspace(-3,5)
    plt.plot(x,mlab.normpdf(x,1.0,1.0), lw=3)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.savefig('px2')

if __name__ == "__main__":
    main()
