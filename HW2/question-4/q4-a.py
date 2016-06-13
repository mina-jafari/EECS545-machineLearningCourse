import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from numpy.linalg import inv, norm

# reading in the data file
A = np.loadtxt("spambase.train", delimiter=",")

numOfRows =  A.shape[0]
numOfCols = A.shape[1]

trainingData = np.delete(A, [numOfCols-1], 1)
targetVector_train = A[:,-1]

B = np.loadtxt("spambase.test", delimiter=",")
testData = np.delete(B, [B.shape[1]-1], 1)
targetVector_test = B[:,-1]

aggregMatrix = np.vstack((trainingData, testData))
targetVector_aggreg = np.hstack((targetVector_train, targetVector_test))

median = np.empty([aggregMatrix.shape[1]])
for i in range(0, aggregMatrix.shape[1]):
    temp = aggregMatrix[:,i]
    median[i] = np.median(temp)

trainingMapping = np.empty([trainingData.shape[0], trainingData.shape[1]])
for i in range(0, trainingData.shape[0]):
    for j in range(0, trainingData.shape[1]):
        if trainingData[i][j] < median[j]:
            trainingMapping[i][j] = 0
        else:
            trainingMapping[i][j] = 1

testMapping = np.empty([testData.shape[0], testData.shape[1]])
for i in range(0, testData.shape[0]):
    for j in range(0, testData.shape[1]):
        if testData[i][j] < median[j]:
            testMapping[i][j] = 0
        else:
            testMapping[i][j] = 1

sumSpam = 0.0
for elem in targetVector_train:
    if elem == 1:
        sumSpam += 1
sumHam = targetVector_train.shape[0] - sumSpam
probSpam = 0.0
probSpam = sumSpam / targetVector_train.shape[0]
probHam = 1 - probSpam
# finding theta
counterSpam = np.empty([trainingMapping.shape[1]])
counterHam = np.empty([trainingMapping.shape[1]])
for i in range(0, trainingMapping.shape[0]):
    if targetVector_train[i] == 1:
        for j in range(0, trainingMapping.shape[1]):
            if trainingMapping[i][j] == 1:
                counterSpam[j] += 1
    elif targetVector_train[i] == 0:
        for j in range(0, trainingMapping.shape[1]):
            if trainingMapping[i][j] == 1:
                counterHam[j] += 1

thetaSpam = np.empty([trainingData.shape[1]])
thetaHam = np.empty([trainingData.shape[1]])
for j in range(0, counterSpam.shape[0]):
    thetaSpam[j] = counterSpam[j] / sumSpam
    thetaHam[j] = counterHam[j] / sumHam

probabilitySpam = np.empty(testMapping.shape)
probabilityHam = np.empty(testMapping.shape)
for i in range(0, testMapping.shape[0]):
    for j in range(0, testMapping.shape[1]):
        if testMapping[i][j] == 0:
            probabilitySpam[i][j] = 1 - thetaSpam[j]
            probabilityHam[i][j] = 1 - thetaHam[j]
        elif testMapping[i][j] == 1:
            probabilitySpam[i][j] = thetaSpam[j]
            probabilityHam[i][j] = thetaHam[j]

misCalc = 0.0
sanityCalc = 0.0
probGivenSpam = np.ones(testData.shape[0])
probGivenHam = np.ones(testData.shape[0])
probIfSpam = np.ones(testData.shape[0])
probIfHam = np.ones(testData.shape[0])
calcTarget = np.empty([testData.shape[0]])
for i in range(0, probabilitySpam.shape[0]):
    for j in range(0, probabilitySpam.shape[1]):
        probGivenSpam[i] *= probabilitySpam[i][j] 
        probGivenHam[i] *= probabilityHam[i][j]
    probIfSpam[i] = \
    (probGivenSpam[i] * probSpam) / (probGivenSpam[i] * probSpam + probGivenHam[i] * probHam)
    probIfHam[i] = \
    (probGivenHam[i] * probHam) / (probGivenSpam[i] * probSpam + probGivenHam[i] * probHam)
    if probIfSpam[i] > probIfHam[i]:
        calcTarget[i] = 1
    else:
        calcTarget[i] = 0
    if calcTarget[i] != targetVector_test[i]:
        misCalc += 1
    if targetVector_test[i] != 0:
        sanityCalc += 1

error = misCalc / targetVector_test.shape[0]
print error*100
print sanityCalc / targetVector_test.shape[0] * 100
