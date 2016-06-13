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
pred_batch = np.zeros(numOfIterations)

for j in range(0, numOfIterations):
    sumW = np.zeros(numOfCols)
    sumB = 0.0
    for k in range(0, numOfRows):
        test = np.add( np.dot(targetVector_train[k], np.dot(wStar, trainingData[k])), np.dot(targetVector_train[k], bStar) )
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
    target_batch = np.zeros(numOfRows)
    for m in range(0, numOfRows):
        target_batch[m] = np.dot(wStar, trainingData[m]) + bStar
        if target_batch[m] * targetVector_train[m] > 0:
            correct += 1
    perCorr = correct / numOfRows
    pred_batch[j] = perCorr



wStar_2 = np.zeros(numOfCols)
bStar_2 = 0.0
wGrad_2 = np.zeros(numOfCols)
bGrad_2 = 0.0
pred_sto = np.zeros(numOfIterations)

for j in range(0, numOfIterations):
    array = np.random.permutation(numOfRows)
    for k in array:
        test = np.add( np.dot(targetVector_train[k], np.dot(wStar_2, trainingData[k])), \
               np.dot(targetVector_train[k], bStar_2) )
        if test < 1.0:
            tempVec = np.dot(np.dot(targetVector_train[k], trainingData[k]), -3)
            tempB = targetVector_train[k] * -3
        else:
            tempVec = np.zeros(numOfCols)
            tempB = 0.0
        wGrad_2 = np.add(np.dot(1.0/numOfRows, wStar_2), tempVec)
        bGrad_2 =  tempB

        wStar_2 -= (0.001 / (1 + j*0.001)) * wGrad_2
        bStar_2 -= (0.001 / (1 + j*0.001)) * bGrad_2
    correct = 0.0
    target = np.zeros(numOfRows)
    for m in range(0, numOfRows):
        target[m] = np.dot(wStar_2, trainingData[m]) + bStar_2
        if target[m] * targetVector_train[m] > 0:
            correct += 1
    perCorr = correct / numOfRows
    pred_sto[j] = perCorr

'''


wStar_2 = np.zeros(numOfCols)
bStar_2 = 0.0
wGrad_2 = np.zeros(numOfCols)
bGrad_2 = 0.0
pred_stoch = np.zeros(numOfIterations)

for j in range(0, numOfIterations):
    sumW_2 = np.zeros(numOfCols)
    sumB_2 = 0.0
    array = np.random.permutation(numOfRows)
    for k in array:
        test = np.add( np.dot(targetVector_train[k], np.dot(wStar_2, trainingData[k])), np.dot(targetVector_train[k], bStar_2) )
        if test < 1.0:
            tempVec = np.dot(np.dot(targetVector_train[k], trainingData[k]), -3)
            tempB = targetVector_train[k] * -3
        else:
            tempVec = np.zeros(numOfCols)
            tempB = 0.0
        #sumW_2 = np.add(sumW_2, tempVec)
        #sumB_2 = sumB_2 + tempB
        wStar_2 = np.dot(1.0/numOfRows, wStar_2)
        #wGrad_2 = np.add(wStar_2, sumW_2)
        wGrad_2 = np.add(wStar_2, tempVec)
        #bGrad_2 = sumB_2
        bGrad_2 = tempB

        wStar_2 -= (0.001 / (1 + j*0.001)) * wGrad_2
        bStar_2 -= (0.001 / (1 + j*0.001)) * bGrad_2
    correct = 0.0
    target = np.zeros(numOfRows)
    for m in range(0, numOfRows):
        target[m] = np.dot(wStar_2, trainingData[m]) + bStar_2
        if target[m] * targetVector_train[m] > 0:
            correct += 1
    perCorr = correct / numOfRows
    pred_stoch[j] = perCorr

print pred_stoch[:10]
'''
x = range(1, 501)
plt.plot(x, pred_batch, 'g', label="batch GD")
plt.plot(x, pred_sto, 'b', label="stochastic GD")
plt.ylabel('Accuracy')
plt.xlabel('# of iterations')
plt.legend(loc=3)
plt.show()
