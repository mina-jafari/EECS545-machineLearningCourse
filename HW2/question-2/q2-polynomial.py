import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# reading in the data file
A = np.loadtxt("steel_composition_train_2.csv", delimiter=",", skiprows=1)
B = np.loadtxt("steel_composition_test.csv", delimiter=",", skiprows=1)

numOfRows =  A.shape[0]
numOfCols = A.shape[1]

# deleting the first and last column
trainingData = np.delete(A, [0, numOfCols-1], 1)
testData = np.delete(B, 0, 1)

# saving the last column as target vector
targetVector = A[:,-1]

# polynomial regression
poly2 = PolynomialFeatures(2)
poly2.fit_transform(trainingData, targetVector)

model = Pipeline([('poly2', PolynomialFeatures(degree=2)), \
        ('linear', linear_model.LinearRegression(fit_intercept=False))])
model = model.fit(trainingData, targetVector)
output = model.predict(testData)

f = open('test-degree2.csv', 'w')
f.write("id,Strength\n")
for i in range(1, output.shape[0]+1):
	f.write(str(i))
	f.write(",")
	f.write(str(output[i-1]))
	f.write("\n")
f.close()
