import sys
import csv

def partition_datasets(datasets, splits):
    fsplit = open(splits)
    fsplit.readline()
    id2datasets = {} 
    for line in fsplit:
        sentence, dataset = line.rstrip('\n').split(',')
        id2datasets[sentence] = dataset
    train = {}
    val = {}
    test = {}
    fsent = open(datasets)
    fsent.readline()
    for line in fsent:
        line = line.rstrip('\n')
        id, sentence = line.split('\t')
        dataset = id2datasets[id]
        if dataset == '1':
            train[id] = sentence
        elif dataset == '2':
            test[id] = sentence
        elif dataset == '3':
            val[id] = sentence
        else:
            print >>sys.stderr, line
    return train, val, test

def flatten_phrase(phrase, references):
    labels = {}
    for line in open(references):
        line = line.rstrip('\n')
        id, label = line.split('|')
        labels[id] = label
    response = {}
    for line in open(phrase):
        line = line.rstrip('\n')
        phrase,id = line.split('|')
        label = labels[id]
        response[phrase.decode('utf8')] = label
    return response

def retrive_bracket(sentence):
    return sentence \
        .replace('-LRB-','(').replace('-RRB-',')') \
        .replace('-LSB-','[').replace('-RSB-',']') \
        .replace('-LCB-','{').replace('-RCB-','}')

def flatten_sentences(sentences, references):
    response = {}
    for id, sentence in sentences.items():
        sentence = retrive_bracket(sentence)
        label = references.get(sentence.decode('utf8'), '')
        if not label:
            print >>sys.stderr, '[NOT FOUND] %s' % sentence
            continue
        response[sentence] = label
    return response

def precess_data(root = '../stanfordSentimentTreebank/', INOGRE_NERTUAL=True):
    train, val, test = partition_datasets(root + 'datasetSentences.txt',root + 'datasetSplit.txt')
    phrase2labels = flatten_phrase(root + 'dictionary.txt', root + 'sentiment_labels.txt')
    sentenceNerual = open('sentencesNeural.txt', 'w')
    for term, name in zip((train, val, test), ('Train', 'Val', 'Test')):
        term = flatten_sentences(term, phrase2labels)
        fout = open('sentences%s.txt' % name, 'w')
        for sentence, label in term.items():
            nlabel = get_label(float(label))
            if nlabel == 'IGNORED':
                print >>sentenceNerual, '%s\t%s' % (sentence, label)
                continue
            print >>fout, '%s\t%d' % (sentence, nlabel)
        fout.close()
    fout = open('phraseTrain.txt', 'w')
    for phrase, label in phrase2labels.items():
        if phrase in test or phrase in train or phrase in val:
            continue
        label = get_label(float(label))
        if label == 'IGNORED': continue
        print >>fout, '%s\t%d' % (phrase.encode('utf8'), label)

def get_label(label, INOGRE_NERTUAL=True):
    assert(type(label) == type(0.0))
    if INOGRE_NERTUAL:
        if label > 0.4 and label <= 0.6:
            return 'IGNORED'
        elif label <= 0.4:
            return 0
        else:
            return 1
    else:
        if label <= 0.2:
            return 1
        elif label <= 0.4:
            return 2
        elif label <= 0.6:
            return 3
        elif label <= 0.8:
            return 4
        else:
            return 5

def extract_data_from_cvs(root = '../SST_data_extraction/'):
    def digitalize(fname, mode = 3):
        response = []
        with open(fname, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                assert len(row) == 2
                label = row[-1]
                if label == 'neu': continue
                if label == 'very pos' or label == 'pos':
                    row[-1] = '1'
                else:
                    row[-1] = '0'
                response.append('\t'.join(row))
        return response

    with open('train.txt', 'w') as f:
        f.write('\n'.join(digitalize(root + 'sst5_train_sentences.csv'))) 
        f.write('\n')
    with open('test.txt', 'w') as f:
        f.write('\n'.join(digitalize(root + 'sst5_test.csv')))
        f.write('\n')
    with open('dev.txt', 'w') as f:
        f.write('\n'.join(digitalize(root + 'sst5_dev.csv')))
        f.write('\n')
    with open('phrase.txt', 'w') as f:
        f.write('\n'.join(digitalize(root + 'sst5_train_phrases.csv')))
        f.write('\n')

if __name__ == '__main__':
    extract_data_from_cvs()
