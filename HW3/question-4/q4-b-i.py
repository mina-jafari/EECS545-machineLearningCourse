import numpy as np
from numpy import linalg
from sklearn.metrics import mean_squared_error
from math import exp
from sklearn.preprocessing import normalize

A = np.loadtxt("steel_composition_train.csv", delimiter=",", skiprows=1)
numOfRows =  A.shape[0]
numOfCols = A.shape[1]
trainingData_temp = np.delete(A, [0, numOfCols-1], 1)
targetVector_train = A[:,-1]

# normalization
trainingData = normalize(trainingData_temp, norm='l2', axis=1)

gramMatrix = np.zeros((numOfRows, numOfRows))
for i in range(0, numOfRows):
    for j in range(0, numOfRows):
        temp1 = np.dot(trainingData[i].transpose(), trainingData[j])
        gramMatrix[i][j] = np.power((temp1 + 1), 2)

eye = np.identity(numOfRows)
temp = np.add(gramMatrix, eye)
a = np.dot(linalg.pinv(temp), targetVector_train)
target_pred = np.zeros(numOfRows)
for i in range(0, numOfRows):
    target_pred[i] = np.dot(gramMatrix[:,i], a)
print np.sqrt(mean_squared_error(targetVector_train, target_pred))
