from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from math import exp, pi, log
from numpy.linalg import inv
from copy import deepcopy



# Generate the data according to the specification in the homework description

N = 500
x = np.random.rand(N)
print x.shape

pi0 = np.array([0.7, 0.3])
w0 = np.array([-2, 1])
b0 = np.array([0.5, -0.5])
sigma0 = np.array([.4, .3])

y1 = np.zeros_like(x)
for i in range(N):
    k = 0 if np.random.rand() < pi0[0] else 1
    y1[i] = w0[k]*x[i] + b0[k] + np.random.randn()*sigma0[k]

y = np.zeros((N, 1))
for j in range(N):
    y[j] = y1[j]
    

# TODO: Implement the EM algorithm for Mixed Linear Regression based on observed
# x and y values.

def computeF(N, K, piVector, sigma, w, x, b, y):
    outerSum = 0.0
    for n in range(N):
        pSum = 0.0
        for k in range(K):
            product = float(piVector[k]) * (1/float(sigma[k])*np.sqrt(2*pi)) * \
                    exp( -1*(float(w[k])*x[n]+b[k]-y[n])**2/(2*sigma[k]**2) )
            #print product
            pSum += product
        #print pSum
        outerSum += log(pSum)
    return outerSum

def computePhi(sigma, w, x, b, y):
    product = (1/sigma*np.sqrt(2*pi)) * exp( -1*(w*x+b-y)**2/(2*sigma**2) )
    return product

def computeR(piVector, sigma, w, x, b, y, N):
    r0 = 0.0
    r1 = 0.0
    for n in range(N):
        r0 += piVector[0] * computePhi(sigma[0], w[0], x[n], b[0], y[0]) / \
                (piVector[0] * computePhi(sigma[0], w[0], x[n], b[0], y[0]) \
                + piVector[1] * computePhi(sigma[1], w[1], x[n], b[1], y[1]))
        r1 += piVector[1] * computePhi(sigma[1], w[1], x[n], b[1], y[1]) / \
                (piVector[0] * computePhi(sigma[0], w[0], x[n], b[0], y[0]) \
                + piVector[1] * computePhi(sigma[1], w[1], x[n], b[1], y[1]))

    return (r0, r1)

def computeWTilde(x, y, r0, r1, N, sigma):
    temp0 = np.zeros((N, N))
    np.fill_diagonal(temp0, -1*r0/(2*sigma[0]**2))
    temp1 = np.zeros((N, N))
    np.fill_diagonal(temp1, -1*r1/(2*sigma[1]**2))

    ones = np.ones((N))
    xTilde = np.vstack((x,ones))
    #print xTilde
    #print 'xtlide', xTilde.shape
    #print 'temp0', temp0.shape
    #print np.dot(temp0, xTilde).shape
    #print np.transpose(xTilde).shape

    first0 = inv(np.dot( xTilde, np.dot(temp0, np.transpose(xTilde))))
    second0 = np.dot( xTilde, np.dot(temp0, y) )
    wb0 = np.zeros((2))
    wb0 = np.dot(first0, second0)

    first1 = inv(np.dot( xTilde, np.dot(temp1, np.transpose(xTilde))))
    second1 = np.dot( xTilde, np.dot(temp1, y) )
    wb1 = np.zeros((2))
    wb1 = np.dot(first1, second1)

    return (wb0, wb1)

    
def computeSigmaSqr(x, N, y, w0, b0, w1, b1, r0, r1):
    ones = np.ones((N))
    xTilde = np.vstack((x, ones))
    #wTilde0 = np.array ([[w0], [b0]])
    #wTilde1 = np.array ([[w1], [b1]])
    wTilde0 = np.array ([w0, b0])
    wTilde1 = np.array ([w1, b1])
    
    R0 = np.zeros((N,N))
    np.fill_diagonal(R0, r0)
    R1 = np.zeros((N,N))
    np.fill_diagonal(R1, r1)

    '''
    print "&&&&&&&&&&&&&&&&&&&&&&&&&"
    print y.shape
    print np.transpose(xTilde).shape
    print wTilde0.shape
    print  (np.dot(np.transpose(xTilde), wTilde0)-y).shape
    print "&&&&&&&&&&&&&&&&&&&&&&&&&"
    '''
    innerVar0  = np.dot(np.transpose(xTilde), wTilde0)-y
    innerVar1  = np.dot(np.transpose(xTilde), wTilde1)-y

    num0 = np.dot(np.transpose(innerVar0), np.dot (R0, innerVar0))
    num1 = np.dot(np.transpose(innerVar1), np.dot (R1, innerVar1))

    sigma0 = num0 / r0
    sigma1 = num1 / r1


    return (float(sigma0), float(sigma1))



piHat = np.array([0.5, 0.5])
wHat = np.array([1, -1])
bHat = np.array([0, 0])
sigmaHat = np.array([np.std(y), np.std(y)])
r0, r1 = computeR(piHat, sigmaHat, wHat, x, bHat, y, N)

F = computeF(N, 2, piHat, sigmaHat, wHat, x, bHat, y)
print "piCurrent: ", piHat
print "sigmaCurrent: ", sigmaHat
print "r0: ", r0
print "r1: ", r1
print "F: ", F
print "======================================"
n = 0
piCurrent = np.zeros(2)

w0Current = np.zeros(2)
w1Current = np.zeros(2)
sigmaOld = deepcopy(sigmaHat)
J = np.empty([1])

while True:
    FOld = F
    r0Old = r0
    r1Old = r1
    piCurrent[0] = r0Old/N
    piCurrent[1] = r1Old/N
    
    wb0Current, wb1Current = computeWTilde(x, y, r0, r1, N, sigmaOld)
    sigmaCurrent = np.sqrt ( computeSigmaSqr(x, N, y, wb0Current[0], wb0Current[1], wb1Current[0], wb1Current[1], r0, r1) )

    sigmaOld = deepcopy(sigmaCurrent)


    F = computeF(N, 2, piCurrent, sigmaCurrent,np.array([wb0Current[0], wb1Current[0]]), x, np.array([wb1Current[0], wb1Current[1]]), y)
    r0, r1 = computeR(piCurrent, sigmaCurrent, np.array([wb0Current[0], wb1Current[0]]), x, np.array([wb1Current[0], wb1Current[1]]), y, N)



    print "piCurrent: ", piCurrent
    print "sigmaCurrent: ", sigmaCurrent
    print "wb0Current: ", wb0Current[0]
    print "wb1Current: ", wb1Current


    print "r0: ", r0
    print "r1: ", r1
    print "F: ", F
    print "FOld: ", FOld
    print "======================================\n"
    J = np.append(J, F)
    n += 1
    if abs(F - FOld) < 0.001:
        print "w: ", w0Current, " ", w1Current, "pi: ", piCurrent, "sigma: ", sigmaCurrent
        break

print J.shape

J = np.delete(J, 0)
kkkkk = range(1, n+1)
plt.plot(kkkkk, J, 'r')
plt.ylabel('F')
plt.xlabel('# of iterations')
plt.savefig('plot-1.png')

# Here's the data plotted
plt.scatter(x, y, c='r', marker='x')
#plt.show()

    
