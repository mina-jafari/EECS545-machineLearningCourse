import numpy as np
from matplotlib import pyplot
import matplotlib as mpl
from sklearn.svm import SVR


def show_image(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def get_patches(X):
    m,n = X.shape
    X = np.pad(X, ((2, 2), (2, 2)), 'constant')
    patches = np.zeros((m*n, 25))
    for i in range(m):
        for j in range(n):
            patches[i*n+j] = X[i:i+5,j:j+5].reshape(25)
    return patches

# input
A = np.loadtxt("train_noised.csv", delimiter=",", skiprows=1)
A = A / 255.0
B = np.loadtxt("train_clean.csv", delimiter=",", skiprows=1)
B = B / 255.0
C = np.loadtxt("test_noised.csv", delimiter=",", skiprows=1)
C = C / 255.0
trainData = np.delete(A, 0, 1)
target = np.delete(B, 0, 1)
test = np.delete(C, 0, 1)


def get_patches(X):
    m,n = X.shape
    X = np.pad(X, ((2, 2), (2, 2)), 'constant')
    patches = np.zeros((m*n, 25))
    for i in range(m):
        for j in range(n):
            patches[i*n+j] = X[i:i+5,j:j+5].reshape(25)
    return patches

row, col = trainData.shape
trainNew = np.zeros((row, col, 25))
for i in range(trainData.shape[0]):
    temp = get_patches(trainData[i, :].reshape(28, 28))
    trainNew[i, :, :] = temp[np.newaxis, :, :]

row, col = test.shape
testNew = np.zeros((row, col, 25))
for i in range(test.shape[0]):
    temp = get_patches(test[i, :].reshape(28, 28))
    testNew[i, :, :] = temp[np.newaxis, :, :]

output = np.zeros((row, col))
for i in range(trainData.shape[1]):
    svr_rbf = SVR(kernel='rbf', C=1e4, gamma=0.0005)
    svr_rbf.fit(trainNew[:, i, :], target[:, i])
    pixel1 = svr_rbf.predict(np.matrix(testNew[:, i, :]))
    output[:, i] = pixel1*255.0


f = open('outPut', 'w')
f.write("Id,Val\n")
for i in range(0, test.shape[0]):
    for j in range(784):
        f.write(str(i))
        f.write("_")
        f.write(str(j))
        f.write(",")
        f.write(str(int(output[i, j])))
        f.write("\n")
f.close()
