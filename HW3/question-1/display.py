import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt;
#matplotlib inline

# Read in csv file and conver to a numpy array 
data = np.genfromtxt('./digits_training_data.csv', delimiter=',')

# plot a random training image (row)
#k = int(np.random.random()*data.shape[0])
k = 329
plt.imshow(data[k].reshape((26,26)), interpolation="nearest", cmap=cm.Greys_r)
plt.show()
