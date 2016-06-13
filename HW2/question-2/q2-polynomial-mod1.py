import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# reading in the data file
A = np.loadtxt("steel_composition_train_2.csv", delimiter=",", skiprows=1)
B = np.loadtxt("steel_composition_test.csv", delimiter=",", skiprows=1)

numOfRows =  A.shape[0]
numOfCols = A.shape[1]

# deleting the first and last column
#trainingData = np.delete(A, [0, 2, 3,5, 6, 8, numOfCols-1], 1)
#trainingData = np.delete(A, [0, 8, numOfCols-1], 1)
trainingData = np.delete(A, [0, 8, numOfCols-1], 1)
testData = np.delete(B, [0, 8,  numOfCols-1], 1)
targetVector_train = A[:,-1]
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainingData, targetVector_train, test_size=0.5, random_state=0)

# saving the last column as target vector

# polynomial regression
poly2 = PolynomialFeatures(2)
#poly2.fit_transform(trainingData, targetVector)
#poly2.fit_transform(X_train, y_train)

model = Pipeline([('poly2', PolynomialFeatures(degree=2)), \
        ('linear', linear_model.LinearRegression(fit_intercept=False))])
model = model.fit(trainingData, targetVector_train)
#model = model.fit(X_train, y_train)
output = model.predict(testData)
#output = model.predict(X_test)

#print "Error:"
#print np.sqrt(mean_squared_error(y_test, output))

f = open('test-degree2-del.csv', 'w')
f.write("id,Strength\n")
for i in range(1, output.shape[0]+1):
	f.write(str(i))
	f.write(",")
	f.write(str(output[i-1]))
	f.write("\n")
f.close()
