#coding:utf8
import numpy as np
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.models import Graph, Model
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Activation, Lambda, Merge
from keras.layers.convolutional import Convolution1D
from keras.layers.embeddings import Embedding
from keras import backend as K
from theano import tensor as T
from qa_utils import extract_sentences, generate_neg
from utility import build_vocab

WORD_EMBEDDING_DIM = 300 # 词向量维数
QMAX_TIMESTAMP = 33 # Q的序列长度
AMAX_TIMESTAMP = 100 # A的序列长度
LOCAL_WINDOW = 3
NB_FILTER = 500 # 卷积核个数
FILTER_LENGTH = 3 # 一维卷积核窗口宽度
BATCH_SIZE = 50 # 批处理数据量
EPOCHS = 1000

pos_q = [line.split() for line in extract_sentences('./data/qg/train.question')]
pos_a = [line.split() for line in extract_sentences('./data/qg/train.answer')]
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
y_np = np.array([1] * len(pos_q) + [0] * len(pos_q), dtype='int')

idx = np.arange(xq_np.shape[0])
np.random.shuffle(idx)
xq_np = xq_np[idx]
xa_np = xa_np[idx,0:100]
y_np = xq_np[idx]
print xq_np.shape, xa_np.shape

# 注意自定义的merge_mode接收到的tensor带有额外的batch_size，
# 即这一层接收到的tensor的ndim=(batch_size, row, col)
# 至于3d矩阵如何按照样本分别求内积就说来话长了。。看下面的用法
# http://deeplearning.net/software/theano/library/tensor/basic.html
def semantic_matrix(argv):
	assert len(argv) == 2
	q = argv[0]
	a = argv[1]
	inner_product = K.batch_dot(q, K.permute_dimensions(a, [0,2,1]))
	q_l2 = K.l2_normalize(q, axis=2)#.reshape((q.shape[0], q.shape[1], 1))
	a_l2 = K.l2_normalize(a, axis=2)#.reshape((a.shape[0], a.shape[1], 1))
	response = inner_product / K.batch_dot(q_l2, K.permute_dimensions(a_l2, [0,2,1]))
	return response

# 注意idx是二维的矩阵
# 如何执行类似batch index的效果折腾了半天
# 参考https://groups.google.com/forum/#!topic/theano-users/7gUdN6E00Dc
def match_matrix(argv, axis=0, w=3):
	assert len(argv) == 2
	o = argv[0]
	s = argv[1]
	idx = T.argmax(s, axis=1+axis)
	i0 = T.repeat(T.arange(idx.shape[0]), idx.shape[1]).flatten()
	i1 = idx.flatten()
	indexed = o[i0, i1, :]
	return indexed.reshape((idx.shape[0], idx.shape[1], o.shape[2]))

def parallel(source, target):
	einner_product = (source * target).sum(axis=2).reshape((source.shape[0],source.shape[1], 1))
	enorm = (target ** 2).sum(axis=2).reshape((source.shape[0],source.shape[1],1))
	response = target * (einner_product / enorm)
	return response

def compute_similar(source, target):
	s_l2 = K.l2_normalize(source, axis=1).reshape((source.shape[0], 1))
	t_l2 = K.l2_normalize(source, axis=1).reshape((source.shape[0], 1))
	reshaped_source = source.reshape((source.shape[0], source.shape[1], 1))
	reshaped_target = source.reshape((target.shape[0], target.shape[1], 1))
	return K.batch_dot(reshaped_source, K.permute_dimensions(reshaped_target, [0,2,1])) / (s_l2 * t_l2)

#model = Graph()

# Question Network Input
q_input = Input(name='q_input', shape=(QMAX_TIMESTAMP,), dtype='int32')
# Answer Network Input
a_input = Input(name='a_input', shape=(AMAX_TIMESTAMP,), dtype='int32')

embedding = Embedding(
	len(vocab) + 1, 
	WORD_EMBEDDING_DIM, 
	#input_length=QMAX_TIMESTAMP
)

q_embedding = embedding(q_input)
a_embedding = embedding(a_input)
print('Embedding ndim q %d a %d' % (K.ndim(q_embedding), K.ndim(a_embedding)))
print('Embedding shape ', q_embedding._keras_shape, a_embedding._keras_shape)

# compute Semantic Matching
cross = Merge(
#	[q_embedding, a_embedding],
	mode=semantic_matrix,
	output_shape=(QMAX_TIMESTAMP, AMAX_TIMESTAMP),
	name='semantic'
)
semantic = cross([q_embedding, a_embedding])
print('Semantic ndim %d' % K.ndim(semantic))
print('Semantic shape ', semantic._keras_shape)
print('Semantic shape ', cross.get_output_shape_at(0))

# compute cross 
q_match = merge(
	[a_embedding, semantic],
	mode=lambda x: match_matrix(x,axis=1),
	output_shape=(QMAX_TIMESTAMP, WORD_EMBEDDING_DIM),
	name='q_match'
)
print('q_match ', q_match._keras_shape, K.ndim(q_match))

a_match = merge(
	[q_embedding, semantic],
	mode=lambda x: match_matrix(x,axis=0),
	output_shape=(AMAX_TIMESTAMP, WORD_EMBEDDING_DIM),
	name='a_match'
)
print('Match ndim q %d a %d' % (K.ndim(q_match), K.ndim(a_match)))
print('Match shape ', q_match._keras_shape, a_match._keras_shape)

# compute q+, q-, a+, a-
# 注意为什么其他的层不需要加BATCH_SIZE，而这里却突然需要了呢？
# 原因Lambda的坑，Lambda的ouput_shape不需要指定BATCH_SIZE，会
# 自行推导：当Lambda的上层输出中含有BATCH_SIZE时，使用改值作
# 为本层的BATCH_SIZE，如果没有时我就呵呵了，不知道是怎么推的。
# 因此这层Merge给定BATCH_SIZE是填下层Lambda的坑
q_pos = Merge(
	mode=lambda x: parallel(*x),
	output_shape=(BATCH_SIZE, QMAX_TIMESTAMP, WORD_EMBEDDING_DIM),
	name='q+'
)([q_embedding, q_match])

# 注意这里不能直接用1 - q_pos获取，否则会丢掉_keras_shape属性
# 注意这里的output_shape是不需要给batch_size的和Merge不同
q_neg = Lambda(
	lambda x: 1 - x,
	output_shape=(QMAX_TIMESTAMP, WORD_EMBEDDING_DIM),
	name='q-'
)(q_pos)
print('q_pos', q_pos._keras_shape, K.ndim(q_pos))
print('q_neg', q_neg._keras_shape, K.ndim(q_neg))

a_pos = Merge(
	mode=lambda x: parallel(*x),
	output_shape=(BATCH_SIZE, AMAX_TIMESTAMP, WORD_EMBEDDING_DIM),
	name='a+',
)([a_embedding, a_match])
a_neg = Lambda(
	lambda x: 1 - x,
	output_shape=(AMAX_TIMESTAMP, WORD_EMBEDDING_DIM),
	name='a-'
)(a_pos)
print('a_pos', a_pos._keras_shape, K.ndim(a_pos))
print('a_neg', a_neg._keras_shape, K.ndim(a_neg))

# q model
q_conv = Convolution1D(
	nb_filter=NB_FILTER,
	filter_length=FILTER_LENGTH,
	border_mode='valid',
	#activation='relu',
	subsample_length=1,
	input_shape=(QMAX_TIMESTAMP, WORD_EMBEDDING_DIM)
)

q_conv_neg = q_conv(q_neg)
q_conv_pos = q_conv(q_pos)
q_merge = Merge(mode='sum')([q_conv_neg, q_conv_pos])
q_act = Activation('relu')(q_merge)
q_maxpool = Lambda(
	lambda x: K.max(x, axis=1),
	output_shape=(NB_FILTER,),
	name='q_maxpool'
)(q_act)

# a model
a_conv = Convolution1D(
	nb_filter=NB_FILTER,
	filter_length=FILTER_LENGTH,
	border_mode='valid',
	#activation='relu',
	subsample_length=1,
	input_shape=(AMAX_TIMESTAMP, WORD_EMBEDDING_DIM)
)

a_conv_neg = a_conv(a_neg)
a_conv_pos = a_conv(a_pos)
a_merge = Merge(mode='sum')([a_conv_neg, a_conv_pos])
a_act = Activation('relu')(a_merge)
a_maxpool = Lambda(
	lambda x: K.max(x, axis=1),
	output_shape=(NB_FILTER,),
	name='a_maxpool'
)(a_act)

similar = merge(
	[q_maxpool, a_maxpool], 
	mode=lambda x: compute_similar(*x),
	output_shape=(BATCH_SIZE, 1),
	name='similar'
)

model = Model(input=[q_input, a_input], output=[similar])
model.compile(optimizer='adam', loss='mse')
model.fit(
	{'q_input':xq_np, 'a_input':xa_np}, 
	{'similar':y_np},
	batch_size=BATCH_SIZE,
	nb_epoch=EPOCHS
)
