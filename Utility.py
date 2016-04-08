from __future__ import print_function
import numpy as np
import random
import os

class StreamDataGenerator(object):
	def __init__(self, path, batch, validation = 0.1, seed = 9527):
		self.fin = open(path)
		self.batch = batch
		self.validation = validation
		self.seed = seed
		self.random = random

	def processor(self, process):
		self.processor = process

	def generate(self):
		while not self.eof():
			train = []
			val = []
			for _ in range(self.batch):
				if self.random.random() > self.validation:
					train.append(self.fin.readline().rstrip('\n'))
				else:
					val.append(self.fin.readline().rstrip('\n'))
			print(len(train), len(val))
			x_train, y_train = self.processor(train)
			x_val, y_val = self.processor(val)
			yield {'train':(x_train, y_train), 'val':(x_val, y_val)}

	def reset(self):
		self.fin.seek(0)
		self.random.seed(self.seed)

	def eof(self):
		return self.fin.tell() == os.fstat(self.fin.fileno()).st_size

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
	sdg = StreamDataGenerator("Utility.py", 5, 0.2)
	sdg.processor(lambda x: (0,0))
	gen = sdg.generate()
	for _ in gen:
		pass
