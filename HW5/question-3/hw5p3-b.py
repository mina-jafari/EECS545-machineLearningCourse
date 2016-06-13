from __future__ import division
import numpy as np
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import inspect
from copy import deepcopy

# 3 states, 4 observations
# Generate the data according to the specification in the homework description
# for part (b)
A = np.array([[0.5, 0.2, 0.3], [0.2, 0.4, 0.4], [0.4, 0.1, 0.5]])
phi = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
pi = np.array([0.5, 0.3, 0.2])

XData = []

for _ in xrange(5000):
    z = [np.random.choice([0,1,2], p=pi)]
    for _ in range(3):
        z.append(np.random.choice([0,1,2], p=A[z[-1]]))
    x = [np.random.choice([0,1], p=phi[zi]) for zi in z]
    XData.append(x)

# TODO: Implement Baum-Welch for estimating the parameters of the HMM

def fb_algorithm(A_mat, B_mat, pi_mat, observation):
    k = len(observation)
    alpha = np.zeros((3, k))
    beta = np.zeros((3, k))

    # setting initial alpha
    alpha[0, 0] = pi_mat[0] * B_mat[0, observation[0]]
    alpha[1, 0] = pi_mat[1] * B_mat[1, observation[0]]
    alpha[2, 0] = pi_mat[2] * B_mat[2, observation[0]]
    # updating alpha
    for i in range(1, k):
        alpha[0, i] = B_mat[0][observation[i]] * (alpha[0][i-1] * A_mat[0][0] \
                      + alpha[1][i-1] * A_mat[1][0] + alpha[2][i-1] * A_mat[2][0])
        alpha[1, i] = B_mat[1][observation[i]] * (alpha[0][i-1] * A_mat[0][1] \
                      + alpha[1][i-1] * A_mat[1][1] + alpha[2][i-1] * A_mat[2][1])
        alpha[2, i] = B_mat[2][observation[i]] * (alpha[0][i-1] * A_mat[0][2] \
                      + alpha[1][i-1] * A_mat[1][2] + alpha[2][i-1] * A_mat[2][2])

    # setting initial beta
    beta[:, -1] = 1.0
    # updating beta
    for i in range(k-2, -1, -1):
        beta[0, i] = (beta[0][i+1] * A_mat[0][0] * B_mat[0][observation[i+1]]+ \
                      beta[1][i+1] * A_mat[0][1] * B_mat[1][observation[i+1]]+\
                      beta[2][i+1] * A_mat[0][2] * B_mat[2][observation[i+1]])
        beta[1, i] = (beta[0][i+1] * A_mat[1][0] * B_mat[0][observation[i+1]]+\
                      beta[1][i+1] * A_mat[1][1] * B_mat[1][observation[i+1]]+\
                      beta[2][i+1] * A_mat[1][2] * B_mat[2][observation[i+1]])
        beta[2, i] = (beta[0][i+1] * A_mat[2][0] * B_mat[0][observation[i+1]]+\
                      beta[1][i+1] * A_mat[2][1] * B_mat[1][observation[i+1]]+\
                      beta[2][i+1] * A_mat[2][2] * B_mat[2][observation[i+1]])

    gamma = alpha * beta
    p_X = np.sum(gamma[:, -1])
    gamma = gamma / np.sum(gamma, axis=0)

    ksi = np.zeros((3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ksi[i, j, k] = alpha[i][k] * A_mat[i][j] * beta[j][k+1] * B_mat[j][observation[k+1]] / np.sum(alpha[:, -1])

    return gamma, ksi, p_X


def baumWelch(A_mat, B_mat, pi_mat, obs_mat):
    # updating pi
    newPi = np.zeros((3))
    numPi_1 = numPi_2 = numPi_3 = 0.0
    denumPi = 0.0
    numB_00 = numB_10 = numB_20 = numB_01 = numB_11 = numB_21 = 0.0
    denumB_0 = denumB_1 = denumB_2 = 0.0
    newB = np.zeros((3, 2))

    p_X = []
    numOfDataPts = len(obs_mat)
    for i in range(numOfDataPts):
        gamma, ksi, pOfX = fb_algorithm(A_mat, B_mat, pi_mat, obs_mat[i])
        p_X.append(pOfX)
        numPi_1 += np.sum(gamma[0, :])
        numPi_2 += np.sum(gamma[1, :])
        numPi_3 += np.sum(gamma[2, :])
        denumPi += (np.sum(gamma[0, :]) + np.sum(gamma[1, :]) + np.sum(gamma[2, :]))

        for j in range(gamma.shape[1]):
            numB_00 += (obs_mat[i][j] == 0) * gamma[0, j]
            numB_01 += (obs_mat[i][j] == 1) * gamma[0, j]
            numB_10 += (obs_mat[i][j] == 0) * gamma[1, j]
            numB_11 += (obs_mat[i][j] == 1) * gamma[1, j]
            numB_20 += (obs_mat[i][j] == 0) * gamma[2, j]
            numB_21 += (obs_mat[i][j] == 1) * gamma[2, j]
        denumB_0 += np.sum(gamma[0, :])
        denumB_1 += np.sum(gamma[1, :])
        denumB_2 += np.sum(gamma[2, :])
    newPi[0] = numPi_1 / denumPi
    newPi[1] = numPi_2 / denumPi
    newPi[2] = numPi_3 / denumPi
    # updating B
    newB[0, 0] = numB_00 / denumB_0
    newB[0, 1] = numB_01 / denumB_0
    newB[1, 0] = numB_10 / denumB_1
    newB[1, 1] = numB_11 / denumB_1
    newB[2, 0] = numB_20 / denumB_2
    newB[2, 1] = numB_21 / denumB_2

    # updating A
    newA = np.zeros((3, 3))
    for j in range(ksi.shape[0]):
        for k in range(ksi.shape[1]):
            num_jk = denum_jk = 0.0
            for i in range(numOfDataPts):
                gamma, ksi, pOfX = fb_algorithm(A_mat, B_mat, pi_mat, obs_mat[i])
                num_jk += np.sum(ksi[j, k, :])
                denum_jk += np.sum(ksi[j, :, :])
            newA[j, k] = num_jk / denum_jk

    return newPi, newA, newB, p_X

def main():

    N = [500, 1000, 2000, 5000]

    A0 = np.random.uniform(0,1,9).reshape(3, 3)
    A0 = normalize(A0, axis=1, norm='l1')
    B0 = np.random.uniform(0,1,6).reshape(3, 2)
    B0 = normalize(B0, axis=1, norm='l1')
    pi0 = np.random.uniform(0,1,3).reshape(3, 1)
    pi0 = normalize(pi0, axis=1, norm='l1')

    iterNum = 50
    calcDist_500 = []
    calcDist_1000 = []
    calcDist_2000 = []
    calcDist_5000 = []
    for n in N:
        counter = 0
        xDataInput = XData[0:n]
        piTrue, ATrue, BTrue, pOfXTrue = baumWelch(A, phi, pi, XData[0:n])
        piCur, ACur, BCur, pOfX = baumWelch(A0, B0, pi0, XData[0:n])
        while (counter < iterNum):
            piOld = deepcopy(piCur)
            AOld = deepcopy(ACur)
            BOld = deepcopy(BCur)
            piCur, ACur, BCur, pOfX = baumWelch(AOld, BOld, piOld, xDataInput)
            diff = 0.5 * np.sum(np.absolute([a - b for a, b in zip(pOfX, pOfXTrue)]))
            if n == 500:
                calcDist_500.append(diff)
            elif n == 1000:
                calcDist_1000.append(diff)
            elif n == 2000:
                calcDist_2000.append(diff)
            elif n == 5000:
                calcDist_5000.append(diff)
            counter += 1
    
    print ACur
    print piCur
    print BCur

    xAxis = range(iterNum)
    plt.plot(xAxis, calcDist_500, 'r', label='500') 
    plt.plot(xAxis, calcDist_1000, 'g', label='1000')
    plt.plot(xAxis, calcDist_2000, 'b', label='2000')
    plt.plot(xAxis, calcDist_5000, 'y', label='5000')
    plt.ylabel('distribution distance')
    plt.xlabel('iteration number')
    #plt.show()
    plt.legend()
    plt.savefig('plot-b.png')



if __name__ == "__main__":
    main()
