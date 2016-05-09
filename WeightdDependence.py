#coding:utf8
import numpy as np
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.models import Graph, Model
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Activation, Lambda, Merge
from keras.layers.convolutional import Convolution1D
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras import backend as K
from theano import tensor as T
from qa_utils import extract_sentences, generate_neg
from utility import build_vocab
import theano
import keras
from theano.compile.nanguardmode import NanGuardMode
theano.config.exception_verbosity='high'

class DebugCallback(keras.callbacks.Callback):
	def on_batch_begin(self, batch, logs={}):
		print debug1(xq_np[0:BATCH_SIZE], xa_np[0:BATCH_SIZE]).max()
		print debug2(xq_np[0:BATCH_SIZE], xa_np[0:BATCH_SIZE]).max()
		for param in model.get_weights():
			print np.any(np.isnan(param))	

def compute_similarity(q, a):
    q_sqrt = K.sqrt((q ** 2).sum(axis=1))
    a_sqrt = K.sqrt((a ** 2).sum(axis=1))
    denominator = q_sqrt * a_sqrt
    output = (q * a).sum(axis=1) / (denominator + SAFE_EPSILON)
    return K.expand_dims(output, 1)

class DebugCallback(keras.callbacks.Callback):
    def on_batch_begin(self, batch, logs={}):
        #print debug1(xq_np[0:BATCH_SIZE], xa_np[0:BATCH_SIZE]).max()
        #print debug2(xq_np[0:BATCH_SIZE], xa_np[0:BATCH_SIZE]).max()
        for param in model.get_weights():
            print np.any(np.isnan(param))   

WORD_EMBEDDING_DIM = 300 # 词向量维数
QMAX_TIMESTAMP = 56 # Q的序列长度
AMAX_TIMESTAMP = 1034 # A的序列长度
LOCAL_WINDOW = 3
NB_FILTER = 500 # 卷积核个数
FILTER_LENGTH = 3 # 一维卷积核窗口宽度
BATCH_SIZE = 50 # 批处理数据量
SAFE_EPSILON = 1e-20
EPOCHS = 1000

#pos_q = [line.split() for line in extract_sentences('./data/qg/train.question')]
#pos_a = [line.split() for line in extract_sentences('./data/qg/train.answer')]
pos_q = [line.split() for line in open('/Users/ganlu/Development/PyPath/DeepNLP/data/insurance_qa_python/train.question')]
pos_a = [line.split() for line in open('/Users/ganlu/Development/PyPath/DeepNLP/data/insurance_qa_python/train.answer')]
reversed_vocab, vocab = build_vocab(pos_q + pos_a, start_with=['<PAD>'])
print len(vocab)
pos_q = [map(lambda x: vocab[x], terms) for terms in pos_q]
pos_a = [map(lambda x: vocab[x], terms) for terms in pos_a]
qmax_timestamp = len(max(pos_q, key=lambda x:len(x)))
amax_timestamp = len(max(pos_a, key=lambda x:len(x)))
print qmax_timestamp, amax_timestamp
neg_q, neg_a = generate_neg(pos_q, pos_a)
xq_data = pos_q + neg_q
xa_data = pos_a + neg_a

xq_np = sequence.pad_sequences(xq_data, maxlen = qmax_timestamp)
xa_np = sequence.pad_sequences(xa_data, maxlen = amax_timestamp)
print xq_np.shape, xa_np.shape
y_np = np.array([1] * len(pos_q) + [0] * len(neg_q), dtype='int').reshape((len(pos_q) + len(neg_q),1))

idx = np.arange(xq_np.shape[0])
np.random.shuffle(idx)
xq_np = xq_np[idx]
xa_np = xa_np[idx]
y_np = y_np[idx]
print xq_np.shape, xa_np.shape, y_np.shape
print 'Check Input', np.any(np.isnan(xq_np)), np.any(np.isnan(xa_np)), np.any(np.isnan(y_np))

def build(i):
    #input = Input(shape=(sentence_length,))

    # embedding
    embedding = Embedding(len(vocab), WORD_EMBEDDING_DIM)
    input_embedding = embedding(i)
    conv = Convolution1D(
        nb_filter=NB_FILTER,
        filter_length=FILTER_LENGTH,
        border_mode='same',
        #activation='tanh',
        subsample_length=1,
		W_constraint = maxnorm(3),
		b_constraint = maxnorm(3),
        #input_shape=(AMAX_TIMESTAMP, WORD_EMBEDDING_DIM),
        #name='conv'
    )

    # dropout = Dropout(0.5)
    # dropout
    input_dropout = conv(input_embedding)

    # maxpooling
    maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False),
                     output_shape=lambda x: (x[0], x[2]))
    input_pool = maxpool(input_dropout)

    # activation
    activation = Activation('tanh')
    output = activation(input_pool)
    return output
    #model = Model(input=[i], output=[output])

question, answer = Input(shape=(QMAX_TIMESTAMP,), dtype='int32'), Input(shape=(AMAX_TIMESTAMP,), dtype='int32')

question_output = build(question)
answer_output = build(answer)

similarity = merge(
    [question_output, answer_output], 
    mode=lambda x: compute_similarity(*x), 
    dot_axes=-1,
    output_shape=(BATCH_SIZE,1),
    name='similarity'
)

model = Model([question, answer], [similarity])

#adam = Adam(clipnorm=3)
model.compile(
    optimizer='adam', 
    loss='mse', 
    metrics=['accuracy'], 
    #mode=NanGuardMode(nan_is_error=True)
)
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')
model.summary()
dc = DebugCallback()
model.fit(
    {'input_1':xq_np, 'input_2':xa_np}, 
    {'similarity':y_np},
    batch_size=BATCH_SIZE,
    nb_epoch=EPOCHS,
    #callbacks=[dc]
)
    
