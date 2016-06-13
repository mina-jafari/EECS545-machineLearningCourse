import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# reading in the data file
A = np.loadtxt("train_graphs_f16_autopilot_cruise.csv", delimiter=",", skiprows=1)

numOfRows =  A.shape[0]
numOfCols = A.shape[1]

# deleting the first and last column
trainingData = np.delete(A, [0, numOfCols-1], 1)
# saving the last column as target vector
targetVector_train = A[:,-1]
# predict the values from test set
B = np.loadtxt("test_graphs_f16_autopilot_cruise.csv", delimiter=",", skiprows=1)
testData = np.delete(B, [0, B.shape[1]-1], 1)
targetVector_test = B[:,-1]

def computeFeatures(degree, inMatrix):
    outMatrix = inMatrix
    outMatrix = np.insert(outMatrix, 0, 1, axis=1)
    newColmn = np.empty([inMatrix.shape[0], 1])
    for power in range(2, degree+1):
        for j in range(0, inMatrix.shape[1]):
            for i in range(0, inMatrix.shape[0]):
                x = inMatrix[i][j]
                x_new = np.power(x, power)
                newColmn[i][0] = x_new
            outMatrix = np.insert(outMatrix, outMatrix.shape[1], newColmn.transpose(), axis=1)
    return outMatrix

rmseTraining = np.empty([6, 1])
rmseTest = np.empty([6, 1])
for i in range(1, 7):
    trainFeatures_i = computeFeatures(i, trainingData)
    testFeatures_i = computeFeatures(i, testData)
    # vector w
    coeffVector_i = np.dot(np.linalg.pinv(trainFeatures_i), targetVector_train)
    trainingTarget_i = np.dot(trainFeatures_i, coeffVector_i)
    testTarget_i = np.dot(testFeatures_i, coeffVector_i)
    # training error
    rmseTraining[i-1][0] = np.sqrt(mean_squared_error(targetVector_train, trainingTarget_i))
    # test error
    rmseTest[i-1][0] = np.sqrt(mean_squared_error(targetVector_test, testTarget_i))

x = [1,2,3,4,5,6]
plt.plot(x, rmseTraining, 'r', label="training set")
plt.plot(x, rmseTest, 'g--', label="test set")
plt.ylabel('RMSE')
plt.xlabel('Order of the polynomial features')
plt.legend(loc=2)
plt.show()
