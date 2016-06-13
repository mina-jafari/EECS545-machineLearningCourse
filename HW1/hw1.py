import numpy as np

# This matrix A contains 183 8x8 images of the handwritten digit 3. 
# Each image has been reshaped into a length 64 row vector.
A = np.loadtxt("A.csv",delimiter=",")

U, s, V = np.linalg.svd(A, full_matrices=True,compute_uv=True)

print s

#print np.dot(U,U.transpose())
#print V
#print s[:3]
s[3:] = 0
s = np.diag(s)
s.resize(U.shape[0], V.shape[1])
B = np.dot(U, np.dot(s, V))
# TODO: perform SVD on A, zero out all but the top 3 singular values to obtain a
# new matrix B and compute ||A-B||^2

R = A - B
print np.power(np.linalg.norm(R), 2)

# B in the following line of code (just a default line of code that will make the plotting at least work) is wrong. 
# Replace it by your code to get correct plots!
#B = np.zeros(A.shape)



# OPTIONAL: You don't need to turn in the output from the code below.
# But if you're curious to see the result of this modification to
# the data using only three singular values, this code snippet will plot
# each handwritten 3 BEFORE (on the top) and AFTER (on the bottom)
# the modification.

# You will need to remove both the lines with ''' to make it run.

# WARNING: You'll need to have matplotlib installed for this to work!

from matplotlib import pyplot as plt

# How many rows and cols in our figure?
NUM_IMGS_TO_SHOW = 5
NUM_PLOT_ROWS = 2
NUM_PLOT_COLS = NUM_IMGS_TO_SHOW


for ind in range(NUM_IMGS_TO_SHOW):

	# The data before and after
	before_vec = A[ind,:]
	after_vec = B[ind,:]

	# We reshape the date into an 8x8 grid
	before_img = np.reshape(before_vec, [8,8])
	after_img = np.reshape(after_vec, [8,8])

	# Now let's plot the before an after into two rows:
	plt.subplot(NUM_PLOT_ROWS,NUM_PLOT_COLS,ind+1)
	plt.imshow(before_img, cmap=plt.cm.gray_r, interpolation='nearest')

	plt.subplot(NUM_PLOT_ROWS,NUM_PLOT_COLS,ind + NUM_IMGS_TO_SHOW + 1)
	plt.imshow(after_img, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()
