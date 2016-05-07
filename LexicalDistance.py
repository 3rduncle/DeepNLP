#coding:utf8
import numpy as np
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.models import Graph, Model
from keras.constraints import maxnorm
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Activation, Lambda, Merge
from keras.layers.convolutional import Convolution1D
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras import backend as K
from theano import tensor as T
from qa_utils import extract_qapair
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

WORD_EMBEDDING_DIM = 300 # 词向量维数
QMAX_TIMESTAMP = 23 # Q的序列长度
AMAX_TIMESTAMP = 236 # A的序列长度
LOCAL_WINDOW = 3
NB_FILTER = 500 # 卷积核个数
FILTER_LENGTH = 3 # 一维卷积核窗口宽度
BATCH_SIZE = 50 # 批处理数据量
SAFE_EPSILON = 1e-20
EPOCHS = 1000

'''
qg数据集读取
#pos_q = [line.split() for line in extract_sentences('./data/qg/train.question')]
#pos_a = [line.split() for line in extract_sentences('./data/qg/train.answer')]
pos_q = [line.split() for line in open('./data/insurance_qa_python/train.question')]
pos_a = [line.split() for line in open('./data/insurance_qa_python/train.answer')]

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
'''

xq_data, xa_data, labels = extract_qapair('./data/wikiqa/WikiQASent-train.txt')
qmax_timestamp = len(max(xq_data, key=lambda x:len(x)))
amax_timestamp = len(max(xa_data, key=lambda x:len(x)))
reversed_vocab, vocab = build_vocab(xq_data + xa_data, start_with=['<PAD>'])
xq_data = [map(lambda x: vocab[x], terms) for terms in xq_data]
xa_data = [map(lambda x: vocab[x], terms) for terms in xa_data]
xq_np = sequence.pad_sequences(xq_data, maxlen = qmax_timestamp)
xa_np = sequence.pad_sequences(xa_data, maxlen = amax_timestamp)
y_np = np.array(labels)

idx = np.arange(xq_np.shape[0])
np.random.shuffle(idx)
xq_np = xq_np[idx]
xa_np = xa_np[idx]
y_np = y_np[idx]
print 'xq_np', xq_np.shape
print 'xa_np', xa_np.shape, 
print 'y_np', y_np.shape
print 'y+', y_np.sum()

# 注意自定义的merge_mode接收到的tensor带有额外的batch_size，
# 即这一层接收到的tensor的ndim=(batch_size, row, col)
# 至于3d矩阵如何按照样本分别求内积就说来话长了。。看下面的用法
# http://deeplearning.net/software/theano/library/tensor/basic.html
def semantic_matrix(argv):
	assert len(argv) == 2
	q = argv[0]
	a = argv[1]
	q_sqrt = K.sqrt((q ** 2).sum(axis=2, keepdims=True))
	a_sqrt = K.sqrt((a ** 2).sum(axis=2, keepdims=True))
	denominator = K.batch_dot(q_sqrt, K.permute_dimensions(a_sqrt, [0,2,1]))
	return K.batch_dot(q, K.permute_dimensions(a, [0,2,1])) / (denominator + SAFE_EPSILON)

# 注意idx是二维的矩阵
# 如何执行类似batch index的效果折腾了半天
# 参考https://groups.google.com/forum/#!topic/theano-users/7gUdN6E00Dc
# 注意argmax里面是2 - axis
def match_matrix(argv, axis=0, w=3):
	assert len(argv) == 2
	o = argv[0]
	s = argv[1]
	idx = T.argmax(s, axis=2-axis)
	i0 = T.repeat(T.arange(idx.shape[0]), idx.shape[1]).flatten()
	i1 = idx.flatten()
	indexed = o[i0, i1, :]
	return indexed.reshape((idx.shape[0], idx.shape[1], o.shape[2]))

def parallel(source, target):
	einner_product = (source * target).sum(axis=2).reshape((source.shape[0],source.shape[1], 1))
	enorm = (target ** 2).sum(axis=2).reshape((source.shape[0],source.shape[1],1))
	response = target * einner_product / (enorm + SAFE_EPSILON)
	return response

def compute_similar(q, a):
	q_sqrt = K.sqrt((q ** 2).sum(axis=1))
	a_sqrt = K.sqrt((a ** 2).sum(axis=1))
	denominator = q_sqrt * a_sqrt
	output = (q * a).sum(axis=1) / (denominator + SAFE_EPSILON)
	return K.expand_dims(output, 1)

#model = Graph()

# Question Network Input
q_input = Input(name='q_input', shape=(QMAX_TIMESTAMP,), dtype='int32')
# Answer Network Input
a_input = Input(name='a_input', shape=(AMAX_TIMESTAMP,), dtype='int32')

embedding = Embedding(
	len(vocab), 
	WORD_EMBEDDING_DIM, 
	#input_length=QMAX_TIMESTAMP,
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
	mode=lambda x: match_matrix(x,axis=0),
	output_shape=(QMAX_TIMESTAMP, WORD_EMBEDDING_DIM),
	name='q_match'
)
print('q_match ', q_match._keras_shape, K.ndim(q_match))

a_match = merge(
	[q_embedding, semantic],
	mode=lambda x: match_matrix(x,axis=1),
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
q_neg = Merge(
	mode=lambda x: x[0] - x[1],
	output_shape=(BATCH_SIZE, QMAX_TIMESTAMP, WORD_EMBEDDING_DIM),
	name='q-'
)([q_embedding, q_pos])
print('q_pos', q_pos._keras_shape, K.ndim(q_pos))
print('q_neg', q_neg._keras_shape, K.ndim(q_neg))

a_pos = Merge(
	mode=lambda x: parallel(*x),
	output_shape=(BATCH_SIZE, AMAX_TIMESTAMP, WORD_EMBEDDING_DIM),
	name='a+',
)([a_embedding, a_match])
a_neg = Merge(
	mode=lambda x: x[0] - x[1],
	output_shape=(BATCH_SIZE, AMAX_TIMESTAMP, WORD_EMBEDDING_DIM),
	name='a-'
)([a_embedding, a_pos])
print('a_pos', a_pos._keras_shape, K.ndim(a_pos))
print('a_neg', a_neg._keras_shape, K.ndim(a_neg))

# q model
q_conv = Convolution1D(
	nb_filter=NB_FILTER,
	filter_length=FILTER_LENGTH,
	border_mode='valid',
	#activation='relu',
	subsample_length=1,
	W_constraint=maxnorm(3),
	b_constraint=maxnorm(3),
	input_shape=(QMAX_TIMESTAMP, WORD_EMBEDDING_DIM),
	name='q_conv'
)

q_conv_neg = q_conv(q_neg)
q_conv_pos = q_conv(q_pos)
q_merge = Merge(mode='sum')([q_conv_neg, q_conv_pos])
q_act = Activation('tanh')(q_merge)
q_maxpool = Lambda(
	lambda x: K.max(x, axis=1),
	output_shape=(NB_FILTER,),
	name='q_maxpool'
)(q_act)

# a model
a_conv = Convolution1D(
	nb_filter=NB_FILTER,
	filter_length=FILTER_LENGTH,
	border_mode='same',
	#activation='relu',
	subsample_length=1,
	input_shape=(AMAX_TIMESTAMP, WORD_EMBEDDING_DIM),
	name='a_conv'
)

a_conv_neg = a_conv(a_neg)
a_conv_pos = a_conv(a_pos)
a_merge = Merge(mode='sum')([a_conv_neg, a_conv_pos])
a_act = Activation('tanh')(a_merge)
a_maxpool = Lambda(
	lambda x: K.max(x, axis=1),
	output_shape=(NB_FILTER,),
	name='a_maxpool'
)(a_act)

print('q maxpool ', q_maxpool._keras_shape, K.ndim(q_maxpool))
print('a maxpool ', a_maxpool._keras_shape, K.ndim(a_maxpool))

similar = merge(
	[q_maxpool, a_maxpool], 
	mode=lambda x: compute_similar(*x),
	#mode='cos', dot_axes=-1,
	output_shape=(BATCH_SIZE,1),
	name='similar'
)

debug1 = theano.function([q_input,a_input], a_embedding,on_unused_input='ignore')
debug2 = theano.function([q_input,a_input], q_embedding,on_unused_input='ignore')
print(debug1(xq_np[0:BATCH_SIZE], xa_np[0:BATCH_SIZE]).max())
print(debug2(xq_np[0:BATCH_SIZE], xa_np[0:BATCH_SIZE]).max())

model = Model(input=[q_input, a_input], output=[similar])
#adam = Adam(clipnorm=3)
model.compile(
	optimizer='adam', 
	loss='mse', 
	metrics=['accuracy'], 
	#mode=NanGuardMode(nan_is_error=True, inf_is_error=True)
)
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')
model.summary()
dc = DebugCallback()
for _ in xrange(EPOCHS):
	model.fit(
		{'q_input':xq_np, 'a_input':xa_np}, 
		{'similar':y_np},
		batch_size=BATCH_SIZE,
		nb_epoch=1,
		#callbacks=[dc]
	)
	
