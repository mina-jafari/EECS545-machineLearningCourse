from matplotlib import pyplot as plt
import numpy as np

# reading in the data file
A = np.loadtxt("steel_composition_train_2.csv", delimiter=",", skiprows=1)
numOfRows =  A.shape[0]
numOfCols = A.shape[1]

# deleting the first and last column
trainingData = np.delete(A, [0, numOfCols-1], 1)
# saving the last column as target vector
targetVector_train = A[:,-1]


#plt.plot(trainingData[:,0], targetVector_train, 'ro', label="0") # linear
#plt.plot(trainingData[:,1], targetVector_train, 'go', label="1") #exclude
#plt.plot(trainingData[:,2], targetVector_train, 'bo', label="2") #exclude
#plt.plot(trainingData[:,3], targetVector_train, 'yo', label="3") #sort of exclude
#plt.plot(trainingData[:,4], targetVector_train, 'mo', label="4") # delete below zero
#plt.plot(trainingData[:,5], targetVector_train, 'co', label="5")
#plt.plot(trainingData[:,6], targetVector_train, 'ko', label="6") # eh
plt.plot(trainingData[:,7], targetVector_train, 'ro', label="7") # delete
plt.ylabel('target value')
plt.xlabel('feature')
#plt.legend(loc=2)
plt.show()
