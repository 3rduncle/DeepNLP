import pickle
import os
import path

def load(file_name):
	return pickle.load(open(file_name, 'rb'))

train = load('./train')
vocabulary = load('./vocabulary')
answers = load('./answers')
fq = open('train.question', 'w')
fa = open('train.answer', 'w')
for entry in train:
	query = entry['question']
	related_answers = entry['answers']
	for answer in related_answers:
		print >>fq, ' '.join([vocabulary[idx].lower() for idx in query])
		print >>fa, ' '.join([vocabulary[idx].lower() for idx in answers[answer]])
fq.close()
fa.close()
