import numpy as np
from matplotlib import pyplot as plt

trainingData = np.loadtxt("digits_training_data.csv", delimiter=",")
B = np.loadtxt("digits_training_labels.csv", delimiter=",")

targetVector_train = np.zeros(B.shape)
i = 0
for elem in B:
    if elem == 4:
        targetVector_train[i] = -1
    elif elem == 9:
        targetVector_train[i] = 1
    i += 1

numOfRows =  trainingData.shape[0]
numOfCols = trainingData.shape[1]

wStar = np.zeros(numOfCols)
bStar = 0.0
wGrad = np.zeros(numOfCols)
bGrad = 0.0
numOfIterations = 500
pred = np.zeros(numOfIterations)

for j in range(0, numOfIterations):
    sumW = np.zeros(numOfCols)
    sumB = 0.0
    for k in range(0, numOfRows):
        test = np.add( np.dot(targetVector_train[k], np.dot(wStar, trainingData[k])), \
                np.dot(targetVector_train[k], bStar) )
        if test < 1.0:
            tempVec = np.dot(np.dot(targetVector_train[k], trainingData[k]), -3)
            tempB = targetVector_train[k] * -3
        else:
            tempVec = np.zeros(numOfCols)
            tempB = 0.0
        sumW = np.add(sumW, tempVec)
        sumB = sumB + tempB
    wGrad = np.add(wStar, sumW)
    bGrad = sumB

    wStar -= (0.001 / (1 + j*0.001)) * wGrad
    bStar -= (0.001 / (1 + j*0.001)) * bGrad
    correct = 0.0
    target = np.zeros(numOfRows)
    for m in range(0, numOfRows):
        target[m] = np.dot(wStar, trainingData[m]) + bStar
        if target[m] * targetVector_train[m] > 0:
            correct += 1
    perCorr = correct / numOfRows
    pred[j] = perCorr

x = range(1, 501)
plt.plot(x, pred, 'g', label="batch GD")
plt.ylabel('Accuracy')
plt.xlabel('# of iterations')
plt.legend(loc=3)
plt.show()
