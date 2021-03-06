#coding:utf8
from __future__ import print_function
import sys
import numpy as np
#np.random.seed(1337)

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.layers.core import TimeDistributedDense, Dense, Dropout, RepeatVector, Activation
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import Sequential, Graph
from CharacterEncoder import CharacterFontTable
from WordSegmentsUtility import GenerateLabel, LabelEmbedding, WordWindow
from Utility import SelectMaximumProbability, StreamDataGenerator

MAX_FEATURES = 64
HIDDEN_SIZE = 128
BATCH_SIZE = 2000
STREAM_BATCH_SIZE = 1000
MAX_LEN = 5
STEP = 1
XMAX_TIMESTAMP = MAX_LEN
X_FEATURES = 64
Y_FEATURES = 4

# B,M,E,S
def PreprocessData(sentences, ct):
	x_data = []
	y_data = []
	for line in sentences:
		sentence, labels = GenerateLabel(line)
		for window in WordWindow(sentence):
			x_data.append([ct.encode(uchar) for uchar in window])
		for label in labels:
			y_data.append(LabelEmbedding(label))
	size = len(x_data)
	x_feature = len(x_data[0][0])
	y_feature = len(y_data[0])
	xmax_timestamp = len(max(x_data, key = lambda x:len(x)))
	print(size, x_feature, y_feature, xmax_timestamp)
	x_np = np.zeros((size, xmax_timestamp, x_feature))
	y_np = np.zeros((size, y_feature))
	for i, sample in enumerate(x_data):
		for t, timestamp in enumerate(sample):
			x_np[i][t] = np.array(timestamp)
	for i, sample in enumerate(y_data):
		y_np[i] = np.array(sample)
	return x_np, y_np

if __name__ == "__main__":
	print('Process Data ... ')
	ct = CharacterFontTable('character.vector')
	path = './icwb2-data/training/pku_training.utf8'
	#sentences = [line.rstrip('\n') for line in open(path)]
	#np.random.shuffle(sentences)
	#split_at = len(sentences) - len(sentences) / 10
	#sen_train = sentences[:split_at]
	#sen_val = sentences[split_at:]
	#x_train, y_train = PreprocessData(sen_train, ct)
	#x_val, y_val = PreprocessData(sen_val, ct)

	streamData = StreamDataGenerator(path, STREAM_BATCH_SIZE, 0.1)
	streamData.processor(lambda x: PreprocessData(x, ct))

	print('Build Model ... ')
	model = Graph()
	model.add_input(name='input', input_shape=(XMAX_TIMESTAMP, X_FEATURES))
	model.add_node(LSTM(HIDDEN_SIZE), name='forward', input='input')
	model.add_node(LSTM(HIDDEN_SIZE, go_backwards=True), name='backward', input='input')
	model.add_node(Dropout(0.2), name='dropout', inputs=['forward', 'backward'])
	model.add_node(Dense(Y_FEATURES, activation='sigmoid'), name='sigmoid', input='dropout')
	model.add_node(Activation('softmax'), name='softmax', input='sigmoid')
	model.add_output(name='output', input='softmax')
	model.compile('adam', {'output': 'categorical_crossentropy'}, metrics=["accuracy"])

	from six.moves import range
	for iteration in range(1, 200):
		print()
		print('-' * 50)
		print('Iteration', iteration)
		streamData.reset()
		for data in streamData.generate():
			x_train, y_train = data['train']
			x_val, y_val = data['val']
			h = model.fit({'input': x_train, 'output': y_train}, 
				batch_size=BATCH_SIZE, 
				nb_epoch=1, 
				#show_accuracy=True
			)
			train_loss = sum(h.history['loss']) / len(h.history['loss'])
			train_acc = sum(h.history['acc']) / len(h.history['acc'])
			print('Training - loss: %f - acc: %f' % (train_loss, train_acc))
			e = model.evaluate(
					{'input': x_val, 'output': y_val},
					batch_size=len(x_val),
					#show_accuracy=True
				)
			print('Validation - loss: %f - acc: %f' % tuple(e))
