
import numpy as np
from sklearn import datasets, linear_model

# reading in the data file
A = np.loadtxt("steel_composition_train.csv", delimiter=",", skiprows=1)

numOfRows =  A.shape[0]
numOfCols = A.shape[1]

# deleting the first and last column
trainingData = np.delete(A, [0, numOfCols-1], 1)

# saving the last column as target vector
targetVector = A[:,-1]

# train the algorithm
regr = linear_model.LinearRegression()
regr.fit(trainingData, targetVector)

# predict the values from test set
B = np.loadtxt("steel_composition_test.csv", delimiter=",", skiprows=1)
testData = np.delete(B, 0, 1)
outMatrix = regr.predict(testData)
#print outMatrix

# print to file
#import os.path
#fname_1 = "test-linear.csv"
#if os.path.isfile(fname_1):
#	os.remove(fname_1)
f = open('test-linear.csv', 'w')
f.write("id,Strength\n")
for i in range(1, outMatrix.shape[0]+1):
	f.write(str(i))
	f.write(",")
	f.write(str(outMatrix[i-1]))
	f.write("\n")
f.close()


# polynomial regression
from sklearn.preprocessing import PolynomialFeatures

poly2 = PolynomialFeatures(2)
poly2.fit_transform(trainingData, targetVector)

from sklearn.pipeline import Pipeline

model = Pipeline([('poly2', PolynomialFeatures(degree=2)), ('linear', linear_model.LinearRegression(fit_intercept=False))])
model = model.fit(trainingData, targetVector)
output = model.predict(testData)

#fname_2 = "test-degree2.csv"
#if os.path.isfile(fname_2):
#	os.remove(fname_2)
f = open('test-degree2.csv', 'w')
f.write("id,Strength\n")
for i in range(1, output.shape[0]+1):
	f.write(str(i))
	f.write(",")
	f.write(str(output[i-1]))
	f.write("\n")
f.close()




# polynomial regression
from sklearn.preprocessing import PolynomialFeatures

poly3 = PolynomialFeatures(3)
poly3.fit_transform(trainingData, targetVector)

from sklearn.pipeline import Pipeline

model_3 = Pipeline([('poly3', PolynomialFeatures(degree=3)), ('linear', linear_model.LinearRegression(fit_intercept=False))])
model_3 = model_3.fit(trainingData, targetVector)
outMatrix_3 = model_3.predict(testData)

f = open('test-degree3.csv', 'w')
f.write("id,Strength\n")
for i in range(1, outMatrix_3.shape[0]+1):
	f.write(str(i))
	f.write(",")
	f.write(str(outMatrix_3[i-1]))
	f.write("\n")
f.close()
