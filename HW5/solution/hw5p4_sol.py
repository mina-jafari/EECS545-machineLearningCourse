from __future__ import division
import numpy as np
from sklearn.cluster import MiniBatchKMeans

Xtrain = np.loadtxt('train_noised.csv', delimiter=',', skiprows=1)[:,1:]
Ytrain = np.loadtxt('train_clean.csv', delimiter=',', skiprows=1)[:,1:]
Xtest = np.loadtxt('test_noised.csv', delimiter=',', skiprows=1)[:,1:]
# Ytest = np.loadtxt('test_clean.csv', delimiter=',', skiprows=1)[:,1:]

ntrain = Xtrain.shape[0]
ntest = Xtest.shape[0]

#%%
##
def save_each(fn, X):
    with open(fn, 'w') as file:
        file.write('Id,Val\n')
        for i,x in enumerate(X):
            for j,y in enumerate(x):
                file.write('%d_%d,%d\n' % (i, j, y))
        

#%%
##
def get_patches(X):
    n2 = len(X)
    n = int(np.sqrt(n2))
    X = np.pad(X.reshape((n,n)), ((2, 2), (2, 2)), 'constant')
    patches = np.zeros((n2, 25))
    for i in range(n):
        for j in range(n):
            patches[i*n+j] = X[i:i+5,j:j+5].reshape(25)
    return patches

patches = np.zeros((ntrain*28**2, 25))
outputs = np.zeros(ntrain*28**2)

for k in range(ntrain):
    patches[k*28**2:(k+1)*28**2] = get_patches(Xtrain[k])
    outputs[k*28**2:(k+1)*28**2] = Ytrain[k]

#%%
## Use k-means to quantize the vector input space, then predict the average
#  correct pixel value for the nearest cluster.

K = 500

km = MiniBatchKMeans(K, n_init=30)
labels = km.fit_predict(patches)

#%%

predictions = np.zeros(K)
for k in range(K):
    predictions[k] = np.mean(outputs[labels == k])

#%%

Yout = np.zeros_like(Xtest)
for i in range(ntest):
    patchy = get_patches(Xtest[i])
    Yout[i,:] = np.array([predictions[c] for c in km.predict(patchy)])

# save_each('solution.csv', Yout)