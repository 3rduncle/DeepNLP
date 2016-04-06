from __future__ import print_function
import numpy as np

def SelectMaximumProbability(mat):
	row, col = mat.shape
	m = mat.max(axis = 1)
	indices = mat == np.dot(m.reshape((row, 1)), np.ones((1, col)))
	response = np.zeros_like(mat)
	response[indices] = 1.0
	return response

if __name__ == '__main__':
	a = np.array([[1,2,3],[3,1,2]])
	print(SelectMaximumProbability(a))
