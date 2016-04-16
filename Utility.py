#coding:utf8
from __future__ import print_function
from collections import Counter
import itertools
import numpy as np
import random
import os
import re

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()


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

def selectMaximumProbability(mat):
	row, col = mat.shape
	m = mat.max(axis = 1)
	indices = mat == np.dot(m.reshape((row, 1)), np.ones((1, col)))
	response = np.zeros_like(mat)
	response[indices] = 1.0
	return response

def buildVocab(sentences, padding = False):
    '''
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    '''
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    if padding:
        word2index = {x: i + 1 for i, x in enumerate(vocabulary)}
    else:
        word2index = {x: i for i, x in enumerate(vocabulary)}
    return [vocabulary, word2index]

def load_word2vec(coll, vocabulary):
    import pymongo
    '''
    Load word2vec from local mongodb
    '''
    word2vec = {}
    for word in vocabulary:
        response = coll.find_one({'word':word})
        if response:
            word2vec[word] = np.array(response['vector'])
    return word2vec

def embedding_layer_weights(vocabulary, word2vec, dim = 300):
    embedding_weights = np.array([word2vec.get(w, np.random.uniform(-0.25,0.25,dim)) for w in vocabulary])
    return [embedding_weights]

if __name__ == '__main__':
    client = pymongo.MongoClient()
    coll = client['word2vec']['en_google']
    vocab, w2i = buildVocab([['hello', 'world'],['hello', 'python']])
    print(vocab, w2i)
    word2vec = load_word2vec(coll, vocab)
    print(word2vec)
