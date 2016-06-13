from __future__ import division
from itertools import product, islice
import numpy as np
from matplotlib import pyplot as plt

# Generate the data according to the specification in the homework description
# for part (b)

A = np.array([[0.5, 0.2, 0.3], [0.2, 0.4, 0.4], [0.4, 0.1, 0.5]])
phi = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
pi0 = np.array([0.5, 0.3, 0.2])

L = 4
X = []

for _ in xrange(5000):
    z = [np.random.choice([0,1,2], p=pi0)]
    for _ in range(L-1):
        z.append(np.random.choice([0,1,2], p=A[z[-1],:]))
    x = [np.random.choice([0,1], p=phi[zi,:]) for zi in z]
    X.append(x)

# TODO: Implement Baum-Welch for estimating the parameters of the HMM
#%%

def z_prob(z, A, pi0):
    p = pi0[z[0]]
    for i in range(L-1):
        p *= A[z[i], z[i+1]]
    return p

def x_given_z_prob(x, z, phi):
    p = 1
    for i in range(L):
        p *= phi[z[i], x[i]]
    return p

def x_prob(x, A, phi, pi0):
    a = forward(x, A, phi, pi0)
    return np.sum(a[:,-1])

def forward(x, A, phi, pi0):
    a = np.zeros((3,L))
    a[:,0] = pi0*phi[:,x[0]]
    for i in range(1,L):
        a[:,i] = a[:,i-1].dot(A)*phi[:,x[i]]
    return a

def backward(x, A, phi):
    b = np.ones((3,L))
    for i in reversed(range(L-1)):
        b[:,i] = A.dot(b[:,i+1]*phi[:,x[i+1]])
    return b

def bw_vars(x, px, a, b, A, phi, pi0):
    gamma = a*b/px
    xi = np.zeros((3,3,L-1))
    for i in range(L-1):
        xi[:,:,i] = A*np.outer(a[:,i], b[:,i+1]*phi[:,x[i+1]])/px
    return gamma, xi

x_probs = {x: x_prob(x, A, phi, pi0) for x in product(range(2), repeat=L)}


x0 = (0,1,0,1)

prior = {}
likelihood = {}
posterior = {}

for z in product(range(3), repeat=L):
    prior[z] = z_prob(z, A, pi0)
    likelihood[z] = x_given_z_prob(x0, z, phi)
    posterior[z] = prior[z]*likelihood[z]/x_probs[tuple(x0)]

max_posterior = sorted(posterior.items(), key=lambda x: x[1], reverse=True)
for z,v in islice(max_posterior, 3):
    print('%s, %3f, %3f, %4f' % (''.join(str(zi) for zi in z), prior[z], likelihood[z], v))


#%%
    
Ahat0 = np.random.rand(3,3)
Ahat0 /= np.sum(Ahat0, 1).reshape((3,1))
phihat0 = np.random.rand(3,2)
phihat0 /= np.sum(phihat0, 1).reshape((3,1))
pi0hat0 = np.random.rand(3)
pi0hat0 /= np.sum(pi0hat0)

#%%

Ns = [500, 1000, 2000, 5000]
dist_curves = []

for k in range(4):
    N = Ns[k]
    print('Running for N=%d' % N)
    Ahat = Ahat0.copy()
    phihat = phihat0.copy()
    pi0hat = pi0hat0.copy()
    
    x_probshat = {x: x_prob(x, Ahat, phihat, pi0hat) for x in product(range(2), repeat=L)}
    
    dists = []
    like = []
    for j in range(50):                
        gammas = np.zeros((3,L,N))
        gammas_x = np.zeros((3,L,N))
        xis = np.zeros((3,3,L-1,N))
        for n in range(N):
            x = tuple(X[n])
            a = forward(x, Ahat, phihat, pi0hat)
            b = backward(x, Ahat, phihat)
            gamma, xi = bw_vars(x, x_probshat[x], a, b, Ahat, phihat, pi0hat)
            
            gammas[:,:,n] = gamma
    
            for i in range(L):
                gammas_x[:,i,n] = gamma[:,i]*x[i]
            
            xis[:,:,:,n] = xi
        
        gamma_total = np.sum(gammas, 2)
        xi_total = np.sum(xis, 3)
        
        Ahat = np.sum(xi_total, 2)
        Ahat /= np.sum(Ahat, 1).reshape((3,1))
    
        phihat[:,1] = np.sum(np.sum(gammas_x, 2), 1) / np.sum(gamma_total, 1)
        phihat[:,0] = 1 - phihat[:,1]
        
        pi0hat = gamma_total[:,0] / np.sum(gamma_total[:,0])
        
        x_probshat = {x: x_prob(x, Ahat, phihat, pi0hat) for x in product(range(2), repeat=L)}
        
        dists.append(0.5*sum(np.abs(v - x_probs[x]) for x,v in x_probshat.items()))
        like.append(sum(np.log(x_probshat[tuple(x)]) for x in X[:N]))
    
    dist_curves.append(dists)

#%%

for k in range(4):
    plt.plot(dist_curves[k], label='N=%d' % Ns[k])

plt.legend()
plt.axis([0, 49, 0, 1.1*max(max(d) for d in dist_curves)])
plt.show()