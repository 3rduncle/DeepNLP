#coding:utf8
from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.layers.core import TimeDistributedDense, Dense, Dropout, RepeatVector, Activation, Masking
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import Sequential, Graph
from CharacterEncoder import CharacterFontTable
from WordSegmentsUtility import GenerateLabel, LabelEmbedding, WordWindow

MAX_FEATURES = 64
HIDDEN_SIZE = 128
BATCH_SIZE = 2000
MAX_LEN = 100
STEP = 20

# B,M,E,S
def PreprocessData(sentences, ct):
	x_data = []
	y_data = []
	for line in sentences:
		sentence, labels = GenerateLabel(line)
		for i in range(0, len(sentence) - MAX_LEN, STEP):
			x_data.append([ct.encode(uchar) for uchar in sentence[i : i + MAX_LEN]])
			y_data.append([LabelEmbedding(char) for char in labels[i : i + MAX_LEN]])
	size = len(x_data)
	x_feature = len(x_data[0][0])
	y_feature = len(y_data[0][0])
	xmax_timestamp = len(max(x_data, key = lambda x:len(x)))
	ymax_timestamp = len(max(y_data, key = lambda x:len(x)))
	print(size, x_feature, y_feature, xmax_timestamp)
	x_np = sequence.pad_sequences(x_data, maxlen = xmax_timestamp)
	y_np = sequence.pad_sequences(y_data, maxlen = ymax_timestamp)
	mask = x_np.max(axis = 2) > 0
	return x_np, y_np, mask

if __name__ == "__main__":
	ct = CharacterFontTable('character.vector')
	path = '../icwb2-data/training/pku_training.utf8'
	sentences = [line.rstrip('\n') for line in open(path)]
	np.random.shuffle(sentences)
	split_at = len(sentences) - len(sentences) / 10
	sen_train = sentences[:split_at]
	sen_val = sentences[split_at:]
	print('Process Data ... ')
	x_train, y_train, m_train = PreprocessData(sen_train, ct)
	x_val, y_val, m_val = PreprocessData(sen_val, ct)
	print(x_train.shape)
	print(y_train.shape)

	print('Build Model ... ')
	_, xmax_timestamp, x_feature = x_train.shape
	_, ymax_timestamp, y_feature = y_train.shape
	model = Graph()
	model.add_input(name='input', input_shape=(xmax_timestamp, x_feature))
	model.add_node(Masking(), name='masking', input='input');
	model.add_node(LSTM(HIDDEN_SIZE, return_sequences=True, activation='relu'), name='forward', input='masking')
	model.add_node(LSTM(HIDDEN_SIZE, go_backwards=True, return_sequences=True, activation='relu'), name='backward', input='masking')
	model.add_node(Dropout(0.2), name='dropout', inputs=['forward', 'backward'])
	model.add_node(TimeDistributedDense(y_feature, activation='sigmoid'), name='sigmoid', input='dropout')
	model.add_node(Activation('softmax'), name='softmax', input='sigmoid')
	model.add_output(name='output', input='softmax')
	model.compile('adam', {'output': 'categorical_crossentropy'})

	from six.moves import range
	for iteration in range(1, 200):
		print()
		print('-' * 50)
		print('Iteration', iteration)
		model.fit({'input': x_train, 'output': y_train}, 
			batch_size=BATCH_SIZE, 
			nb_epoch=1, 
			show_accuracy=True,
			#sample_weight={'output': m_train}
		)
		e = model.evaluate(
			{'input': x_val, 'output': y_val},
			batch_size=len(x_val),
			show_accuracy=True,
			#sample_weight={'output': m_val}
		)
		print('Validation loss: %f -- acc: %f ' % tuple(e))
