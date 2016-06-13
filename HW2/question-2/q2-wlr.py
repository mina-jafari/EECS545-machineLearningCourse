import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from numpy.linalg import inv, norm
from sklearn import cross_validation
from sklearn import svm



# reading in the data file
A = np.loadtxt("steel_composition_train_2.csv", delimiter=",", skiprows=1)
numOfRows =  A.shape[0]
numOfCols = A.shape[1]

# deleting the first and last column
trainingData = np.delete(A, [0, numOfCols-1, numOfCols-2], 1)

# saving the last column as target vector
targetVector_train = A[:,-1]

# predict the values from test set
B = np.loadtxt("steel_composition_test.csv", delimiter=",", skiprows=1)
testData = np.delete(B, [0], 1)
#targetVector_test = B[:,-1]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData, targetVector_train, test_size=0.5, random_state=0)

#tau = np.logspace(-2, 1, num=10, base=2)
tau = [10000.]
#rmseTest = np.empty([10])
#print "std" 
#print np.std(trainingData[:,-1])
'''
for elem in tau:
    size = trainingData.shape[0]
    target = np.empty([testData.shape[0]])
    for j in range(0, testData.shape[0]):
        testPoint = testData[j]
        weightVector = np.empty([size])
        k = 0
        for i in range(0, trainingData.shape[0]):
            subt = np.subtract(trainingData[i], testPoint)
            normV = norm(subt)
            weightVector[k] = np.exp(-1 * np.power(normV, 2) / (2 * np.power(elem, 2)) )
            k += 1
        regr = linear_model.LinearRegression()
        vecW = regr.fit(trainingData, targetVector_train, weightVector).coef_
        target[j] = np.dot(vecW, testPoint)
        print target[j]
    #rmseTest[l] = np.sqrt(mean_squared_error(targetVector_test, target))
    #l += 1
plt.plot(tau, rmseTest, 'r')
plt.ylabel('RMSE')
plt.xlabel('$\\tau$')
plt.show()
'''

for elem in tau:
    size = X_train.shape[0]
    target = np.empty([X_test.shape[0]])
    for j in range(0, X_test.shape[0]):
        testPoint = X_test[j]
        weightVector = np.empty([size])
        k = 0
        for i in range(0, X_train.shape[0]):
            subt = np.subtract(X_train[i], testPoint)
            normV = norm(subt)
            weightVector[k] = np.exp(-1 * np.power(normV, 2) / (2 * np.power(elem, 2)) )
            k += 1
        regr = linear_model.LinearRegression()
        vecW = regr.fit(X_train, y_train, weightVector).coef_
        print vecW
        print testPoint
        break
        #target[j] = np.dot(vecW, testPoint)
    target = regr.predict(X_test)
#        print target[j]
print "Error:"
print np.sqrt(mean_squared_error(y_test, target))
#plt.plot(X_train, y_train, 'r')
plt.plot(X_test, y_test, 'g')
#plt.plot(trainingData, targetVector_train, 'r')
plt.plot(X_test, target, 'r')
plt.ylabel('RMSE')
plt.xlabel('$\\tau$')
#plt.show()

f = open('test-wlr.csv', 'w')
f.write("id,Strength\n")
for i in range(1, target.shape[0]+1):
        f.write(str(i))
        f.write(",")
        f.write(str(target[i-1]))
        f.write("\n")
f.close()

