import sys
import re
import random

def extract_sentences(fname):
	sentences = []
	start_r = re.compile('<\d+>')
	end_r = re.compile('</\d+>')
	for line in open(fname):
		line = line.rstrip('\n')
		if start_r.match(line):
			phrase = []
			hit = True
			continue
		elif end_r.match(line):
			sentences.append(' '.join(phrase))
			phrase = []
			continue
		elif not line:
			hit = True
			continue
		else:
			pass
		if hit:
			hit = False
			phrase.append(line)
	return sentences

def generate_neg(question, answer):
	qsize = len(question)
	asize = len(answer)
	assert qsize == asize
	neg_q = []
	neg_a = []
	for i in xrange(qsize):
		while True:
			qindex = random.randint(0, qsize - 1)
			aindex = random.randint(0, asize - 1)
			if qindex != aindex and question[qindex] != question[aindex]:
				break
		neg_q.append(question[qindex])
		neg_a.append(answer[aindex])
	return neg_q, neg_a

if __name__ == '__main__':
	a = extract_sentence('./data/qg/train.answer')
	b = extract_sentence('./data/qg/train.question')
	c, d = generate_neg(a, b)
	print len(a), len(b), len(c), len(d)
		

