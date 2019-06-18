'''
Created on Sep 14, 2015

@author: bishan
'''

import codecs
import cPickle
from operator import itemgetter, attrgetter
import os
import sys

def save_pkl(data, datafile):
    f = open(datafile, 'wb')
    cPickle.dump(data, f, -1)
    f.close()
    
def load_pkl(datafile):
    f = open(datafile, 'rb')
    data = cPickle.load(f)
    f.close()
    return data

def save_vocabulary(vocab, path):
    with codecs.open(path, 'w', 'utf-8') as f:
        for w in vocab:
            print >>f, w

def load_vocabulary(path):
    with codecs.open(path, 'r', 'utf-8') as f:
        vocab = [line.strip() for line in f if len(line) > 0]
    return dict([(a, i) for i, a in enumerate(vocab)])

def save_word_idx_map(vocab, path):
    with codecs.open(path, 'w', 'utf-8') as f:
        for w, c in vocab.items():
            print >>f, w, c

def load_word_idx_map(path):
    vocab = {}
    with codecs.open(path, 'r', 'utf-8') as f:
        for line in f:
            if len(line) > 0:
                line = line.strip()
                j = line.rfind(' ')
                vocab[line[:j]] = int(line[j+1:])
    return vocab

def sortDict(pdict, keyindex, isReversed=False):
    seq_dict = []
    for k,v in pdict.items():
        seq_dict.append([k,v])
    return sorted(seq_dict, key=itemgetter(keyindex), reverse=isReversed)

def dumpDict(pdict, outfile):
    wf = open(outfile, 'w')
    for k,v in pdict.items():
        wf.write(str(k) + "\t" + str(v) + "\n")
    wf.close()
    