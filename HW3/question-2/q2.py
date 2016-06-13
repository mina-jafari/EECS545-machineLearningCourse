import numpy as np
from sklearn.svm import SVC

trainingData = np.loadtxt("trainingData", delimiter=",")
targetVector_train = np.loadtxt("trainingLabels", delimiter=",")
testData = np.loadtxt("testData", delimiter=",")

clf = SVC(C=1.0, kernel='rbf', gamma=0.0000005, tol=0.001)
clf.fit(trainingData, targetVector_train)
target_pred = clf.predict(testData)
f = open('outPut', 'w')
f.write("id,category\n")
for i in range(1, target_pred.shape[0]+1):
        f.write(str(i))
        f.write(",")
        f.write(str(int(target_pred[i-1])))
        f.write("\n")
f.close()
