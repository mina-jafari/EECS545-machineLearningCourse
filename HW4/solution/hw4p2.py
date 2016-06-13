from __future__ import division
import numpy as np

# Generate the data according to the specification in the homework description
   
N = 1000
m = 5
alpha = np.array([10, 5, 15, 20, 50])
P = np.random.dirichlet(alpha, N)

# TODO: Implement the Newton-Raphson algorithm for estimating the parameters of
# the Dirichlet distribution given observances (rows of P).
