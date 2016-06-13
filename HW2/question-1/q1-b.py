import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from numpy.linalg import inv, norm, pinv

# reading in the data file
A = np.loadtxt("train_graphs_f16_autopilot_cruise.csv", delimiter=",", skiprows=1)

numOfRows =  A.shape[0]
numOfCols = A.shape[1]

# deleting the first and last column
trainingData = np.delete(A, [0, numOfCols-1], 1)
# saving the last column as target vector
targetVector_train = A[:,-1]

# predict the values from test set
B = np.loadtxt("test_locreg_f16_autopilot_cruise.csv", delimiter=",", skiprows=1)
testData = np.delete(B, [0, B.shape[1]-1], 1)
targetVector_test = B[:,-1]

a = np.zeros((3426, 3426), float)
phi_T = np.transpose(trainingData)
tau = np.logspace(-2, 1, num=10, base=2)
l = 0
rmseTest = np.empty([10])
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
        weightVector_2 = np.diag(weightVector)
        inverse = pinv(np.dot(np.dot(phi_T, weightVector_2), trainingData))
        rt = np.dot(weightVector_2, targetVector_train)
        vecW = np.dot(np.dot(inverse, phi_T), rt)
        target[j] = np.dot(vecW, testPoint)
    rmseTest[l] = np.sqrt(mean_squared_error(targetVector_test, target))
    l += 1

plt.plot(tau, rmseTest, 'r')
plt.ylabel('RMSE')
plt.xlabel('$\\tau$')
plt.show()
