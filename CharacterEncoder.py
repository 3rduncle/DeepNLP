#coding:utf8
import numpy as np
class CharacterFontTable(object):
	def __init__(self, path):
		self.char_vector = {}
		self.vector_char = {}
		for line in open(path):
			line = line.rstrip('\n')
			pair = line.split('\t')
			if len(pair) != 2 or not pair[1]: continue
			c = pair[0].decode('utf8')
			v = [float(i) for i in pair[1].split(',')]
			i = ''.join([str(term) for term in np.array(v).round().astype(int).tolist()])
			self.char_vector[c] = v
			self.vector_char[i] = c

	def encode(self, uchar):
		return self.char_vector[uchar]

	def decode(self, vector):
		i = ''.join([str(term) for term in np.array(vector).round().astype(int).tolist()])
		return self.vector_char.get(i, u'')

if __name__ == '__main__':
	ct = CharacterFontTable('character.vector')
	v = ct.encode(u'我')
	print(v)
	print(ct.decode(v))
