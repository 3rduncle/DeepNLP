#coding:utf8
from __future__ import print_function
import numpy as np
import random
np.random.seed(1337)
random.seed(1337)

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.layers.core import TimeDistributedDense, Dense, Dropout, RepeatVector, Activation, Masking
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.models import Sequential
from CharacterEncoder import CharacterFontTable
from WordSegmentsUtility import GenerateLabel, LabelEmbedding

MAX_FEATURES = 64
HIDDEN_SIZE = 200
BATCH_SIZE = 1000
MAX_LEN = 100
STEP = 20
UNIT = GRU

# B,M,E,S
def PreprocessData(sentences, ct):
	x_data = []
	y_data = []
	for segments in sentences:
		sentence, label = GenerateLabel(segments)
		if len(sentence) < MAX_LEN:
			x_data.append([ct.encode(uchar) for uchar in sentence])
			y_data.append([LabelEmbedding(char) for char in label])
			continue
		for i in range(0, len(sentence) - MAX_LEN, STEP):
			x_data.append([ct.encode(uchar) for uchar in sentence[i: i + MAX_LEN]])
			y_data.append([LabelEmbedding(char) for char in label[i: i + MAX_LEN]])
	size = len(x_data)
	x_feature = len(x_data[0][0])
	y_feature = len(y_data[0][0])
	max_timestamp = len(max(x_data, key = lambda x:len(x)))
	print(size, x_feature, y_feature, max_timestamp)
	x_np = sequence.pad_sequences(x_data, maxlen = max_timestamp)
	y_np = sequence.pad_sequences(y_data, maxlen = max_timestamp)
	mask = np.zeros((size, max_timestamp))
	mask[x_np.max(axis = 2) > 0] = 1
	print(mask.shape)
	return x_np, y_np, mask

if __name__ == "__main__":
	path = '../icwb2-data/training/msr_training.utf8'
	ct = CharacterFontTable('character.vector')

	print('Process Data ... ')
	sentences = [line.rstrip('\n') for line in open(path)]
	#random.shuffle(sentences)
	split_at = len(sentences) - len(sentences) / 10
	sen_train = sentences[:split_at]
	sen_val = sentences[split_at:]
	x_train, y_train, mask_train = PreprocessData(sen_train, ct)
	x_val, y_val, mask_val = PreprocessData(sen_val, ct)
	print('X Train Shape %d %d %d' % x_train.shape)
	print('Y Train Shape %d %d %d' % y_train.shape)

	print('Build Model ... ')
	_, xmax_timestamp, x_feature = x_train.shape
	_, ymax_timestamp, y_feature = y_train.shape
	model = Sequential()
	model.add(Masking(input_shape=(xmax_timestamp, x_feature)))
	model.add(UNIT(HIDDEN_SIZE, activation='relu', return_sequences=True))
	model.add(Dropout(0.2))
	model.add(TimeDistributedDense(y_feature))
	model.add(Activation('softmax'))
	model.compile(
		loss='categorical_crossentropy', 
		optimizer='adam',
		metrics=['accuracy'],
		sample_weight_mode='temporal'
	)

	from six.moves import range
	for iteration in range(1, 200):
		print()
		print('-' * 50)
		print('Iteration', iteration)
		model.fit(
			x_train, 
			y_train, 
			batch_size=BATCH_SIZE, 
			nb_epoch=1,
			#show_accuracy=True,
			sample_weight=mask_train
		)
		e = model.evaluate(
			x_val,
			y_val,
			batch_size=len(x_val),
			#show_accuracy=True,
			sample_weight=mask_val
		)
		print('Validation loss: %f -- acc: %f ' % tuple(e)) 
