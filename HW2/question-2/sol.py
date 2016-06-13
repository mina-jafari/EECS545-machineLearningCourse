
import numpy as np

# reading in the data file
A = np.loadtxt("steel_composition_train.csv", delimiter=",", skiprows=1)

numOfRows =  A.shape[0]
numOfCols = A.shape[1]
# deleting the first and last column
train = np.delete(A, [0, numOfCols-1], 1)
# saving the last column as target vector
vector_t = A[:,-1]

#from sklearn import datasets, linear_model
#regr = linear_model.LinearRegression()
#regr.fit(train, vector_t)
#vector_w = regr.coef_


B = np.loadtxt("steel_composition_test.csv", delimiter=",", skiprows=1)
test = np.delete(B, 0, 1)

train_trans = train.transpose()

from numpy.linalg import inv

train_T_train = np.dot(train_trans, train)
ident = np.eye(train_T_train.shape[0], train_T_train.shape[1])
lamda = np.exp(-18)
lamda_ident = np.dot(lamda, ident)
train_T_target = np.dot(train_trans, vector_t)
vector_w = np.dot(inv(np.add(lamda_ident, train_T_train)), train_T_target)

pred = np.dot(test, vector_w)
counter = np.arange(1, pred.shape[0]+1)

out_matrix = np.concatenate((counter, pred))
out_matrix = np.reshape(out_matrix, (2, -1))
out_matrix = out_matrix.transpose()

np.savetxt('test.out', out_matrix, fmt='%.6f', delimiter=',', header='id,Strength', comments='')
#train_inv = np.dot((inv(np.exp(-18), np.identity(np.dot(train_trans, train).shape[0]))  + np.dot(train_trans, train)), train_trans)
#iden = np.identity(np.dot(train_trans, train).shape[0])
#print vector_w
#a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

#print values[:, None]
#values = np.ndarray.astype(1,a.shape[0])
#values = values.reshape((10, 1))
#print values
#test = np.append(a, values, axis=1)
#test = np.insert(a, 0, values, axis=0)
#np.resize(test, (10, 2))



