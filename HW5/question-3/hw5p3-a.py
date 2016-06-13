import numpy as np
import itertools

A = np.array([[0.5, 0.2, 0.3], [0.2, 0.4, 0.4], [0.4, 0.1, 0.5]])
phi = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
pi0 = np.array([0.5, 0.3, 0.2])

combinations = list(itertools.product([0,1,2], repeat=4))
output = [[0 for x in range(4)] for x in range(81)] 
i = 0
for elem in combinations:
    output[i][0] = elem #combinations
    output[i][1] = (pi0[elem[0]] * A[elem[0]][elem[1]] * A[elem[1]][elem[2]] * A[elem[2]][elem[3]]) #prior
    output[i][2] = (phi[elem[0]][0] * phi[elem[1]][1] * phi[elem[2]][0] * phi[elem[3]][1]) #likelihood
    i += 1

colSum = 0
for i in range(len(output)):
    colSum += output[i][2]

for i in range(len(output)):
    output[i][3] = output[i][1] * output[i][2] / colSum 

output = sorted(output, key=lambda row:row[3], reverse=True)

for col1, col2, col3, col4 in output:
       print (col1, "%.4f" % col2, "%.4f" % col3, "%.4f" % col4)
