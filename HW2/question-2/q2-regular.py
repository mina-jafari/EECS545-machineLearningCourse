import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from numpy.linalg import inv, pinv
from sklearn.preprocessing import PolynomialFeatures
from sklearn import cross_validation
from sklearn import datasets, linear_model

# reading in the data file
A = np.loadtxt("steel_composition_train_2.csv", delimiter=",", skiprows=1)
numOfRows =  A.shape[0]
numOfCols = A.shape[1]

# deleting the first and last column
trainingData = np.delete(A, [0, numOfCols-1], 1)
# saving the last column as target vector
targetVector_train = A[:,-1]

# predict the values from test set
B = np.loadtxt("steel_composition_test.csv", delimiter=",", skiprows=1)
testData = np.delete(B, 0, 1)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData, targetVector_train, test_size=0.4, random_state=0)
poly = PolynomialFeatures(3)
features_train = poly.fit_transform(X_train)
features_test = poly.fit_transform(X_test)

regr = linear_model.LinearRegression()
#trainFeatures_T = features_train.transpose()
trainFeatures_T = trainingData.transpose()
#phi_T_phi = np.dot(trainFeatures_T, features_train)
phi_T_phi = np.dot(trainFeatures_T, trainingData)
ident = np.eye(phi_T_phi.shape[0], phi_T_phi.shape[1])
#for k in range(-40, 41):
for k in range(18, 19):
    lamda = np.exp(k)
    lamdaEye = np.dot(lamda, ident)
    matrix_1 = np.add(lamdaEye, phi_T_phi)
    matrix_1_inv = inv(matrix_1)
#    matrix_2 = np.dot(trainFeatures_T, y_train)
    matrix_2 = np.dot(trainFeatures_T, targetVector_train)
    coeffVector_k = np.dot(matrix_1_inv, matrix_2)
#    testTarget_k = np.dot(features_test, coeffVector_k)
    testTarget_k = np.dot(testData, coeffVector_k)
#    print (k, np.sqrt(mean_squared_error(y_test, testTarget_k)))

f = open('test-reg.csv', 'w')
f.write("id,Strength\n")
for i in range(1, testTarget_k.shape[0]+1):
        f.write(str(i))
        f.write(",")
        f.write(str(testTarget_k[i-1]))
        f.write("\n")
f.close()


'''
x = range(-40, 21, 1)
plt.plot(x, rmseTraining, 'r', label="training set")
plt.ylabel('RMSE')
plt.xlabel('$\ln \lambda$')
plt.legend(loc=2)
plt.show()
'''
