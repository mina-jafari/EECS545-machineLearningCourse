import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from numpy.linalg import inv, pinv

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
            print "computing..."
            outMatrix = np.insert(outMatrix, outMatrix.shape[1], newColmn.transpose(), axis=1)
    return outMatrix

rmseTraining = np.empty([61, 1])
rmseTest = np.empty([61, 1])
for i in range(6, 7):
    trainFeatures_i = computeFeatures(i, trainingData)
    testFeatures_i = computeFeatures(i, testData)

    trainFeatures_i_T = trainFeatures_i.transpose()
    phi_T_phi = np.dot(trainFeatures_i_T, trainFeatures_i)
    ident = np.eye(phi_T_phi.shape[0], phi_T_phi.shape[1])
    j = 0
    for k in range(-40, 21):
        lamda = np.exp(k)
        lamdaEye = np.dot(lamda, ident)
        matrix_1 = np.add(lamdaEye, phi_T_phi)
        matrix_1_inv = inv(matrix_1)
        matrix_2 = np.dot(trainFeatures_i_T, targetVector_train)
        coeffVector_i_k = np.dot(matrix_1_inv, matrix_2)
        trainingTarget_i_k = np.dot(trainFeatures_i, coeffVector_i_k)
        testTarget_i_k = np.dot(testFeatures_i, coeffVector_i_k)
        # training error
        rmseTraining[j][0] = np.sqrt(mean_squared_error(targetVector_train, trainingTarget_i_k))
        # test error
        rmseTest[j][0] = np.sqrt(mean_squared_error(targetVector_test, testTarget_i_k))
        j = j + 1

x = range(-40, 21, 1)
plt.plot(x, rmseTraining, 'r', label="training set")
plt.plot(x, rmseTest, 'g--', label="test set")
plt.ylabel('RMSE')
plt.xlabel('$\ln \lambda$')
plt.legend(loc=2)
plt.show()
