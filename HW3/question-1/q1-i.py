import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt

trainingData = np.loadtxt("digits_training_data.csv", delimiter=",")
A = np.loadtxt("digits_training_labels.csv", delimiter=",")
testData = np.loadtxt("digits_test_data.csv", delimiter=",")
B = np.loadtxt("digits_test_labels.csv", delimiter=",")

targetVector_train = np.zeros(A.shape)
i = 0
for elem in A:
    if elem == 4:
        targetVector_train[i] = -1
    elif elem == 9:
        targetVector_train[i] = 1
    i += 1

targetVector_test = np.zeros(B.shape)
i = 0
for elem in B:
    if elem == 4:
        targetVector_test[i] = -1
    elif elem == 9:
        targetVector_test[i] = 1
    i += 1

clf = SVC(C=1.0, kernel='rbf', gamma=0.0000005, tol=0.001)
clf.fit(trainingData, targetVector_train)
target_pred = clf.predict(testData)
print clf.score(trainingData, targetVector_train)
print clf.score(testData, targetVector_test)
for i in range(0, targetVector_test.shape[0]):
    if targetVector_test[i] != target_pred[i]:
        print i
