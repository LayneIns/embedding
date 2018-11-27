'''
@Author: Hui Liu
@Date: 2018-11-23 13:33:55
@github: https://github.com/LayneIns
'''

# This code takes an embedding, the corresponding id2word file,
# and evaluates the embedding on the given test file by word similarity measure.


import sys
import os

import numpy as np
import matplotlib.pyplot as plt

# gensim
import gensim

import argparse
from tqdm import tqdm

from scipy import stats


def readEmb(emb_filename):
    f = open(emb_filename, 'r')
    lines = f.readlines()
    N, dim = lines[0].split()
    N = int(N)
    dim = int(dim)
    #print(N)
    #print(dim)
    f.close()

    id2emb = {}
    for i in tqdm(range(len(lines))):
        if i == 0:
            continue
        else:
            line_texts = lines[i].strip().split()
            id = int(line_texts[0])
            embedding = []
            for num in line_texts[1:]:
                if num == '1/2':
                    embedding.append(0.5)
                else:
                    try:
                        embedding.append(float(num))
                    except:
                        continue
            # embedding = [float(num) for num in line_texts[1:]]
            id2emb[id] = np.asarray(embedding)
    
    # ids = np.loadtxt(emb_filename, dtype=int, skiprows=1, usecols=0)
    # #print(ids.shape)
    # embs = np.loadtxt(emb_filename, skiprows=1, usecols=range(1,dim+1))
    # #print(embs.shape)
    
    # for i in range(N):
    #     id2emb[ids[i]] = embs[i]
    return N, dim, id2emb


def readId2Word(id2word_filename):
    id2word = []
    vocab = {}
    with open(id2word_filename, 'r') as f:
        for lines in f:
            l = lines.strip().split()
            id2word.append(l[1])
            vocab[l[1].lower()] = int(l[0])
    #print(id2word)
    #print(vocab)
    return id2word, vocab


def similarity(w1, w2, vocab, id2emb):
    #print("{} {}".format(w1, w2))
    w1_id = vocab[w1]
    w2_id = vocab[w2]
    v1 = id2emb[w1_id]
    v2 = id2emb[w2_id]
    return np.dot(v1, v2)


def main(emb_filename, id2word_filename, test_filename, delimiter='\t'):
    N, dim, id2emb = readEmb(emb_filename)
    id2word, vocab = readId2Word(id2word_filename)

    similarity_gold = []
    similarity_model = []
    oov = 0

    for line_no, line in enumerate(gensim.utils.smart_open(test_filename)):
        line = gensim.utils.to_unicode(line)
        if line.startswith('#'):
            continue
        else:
            a, b, sim = [word.lower() for word in line.split(delimiter)]
            #print("{} {} {}".format(a,b,sim))
            sim = float(sim)
            if a not in vocab or b not in vocab:
                oov += 1
                continue
            similarity_gold.append(sim)
            similarity_model.append(similarity(a, b, vocab, id2emb))
    spearman = stats.spearmanr(similarity_gold, similarity_model)
    pearson = stats.pearsonr(similarity_gold, similarity_model)
    oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100
    
    print("pearson")
    print(pearson)
    print("spearman")
    print(spearman)
    print("oov_ratio")
    print(oov_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, required=True)
    parser.add_argument('--id2word', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    args = parser.parse_args()
    emb_filename = args.embedding
    id2word_filename = args.id2word
    test_filename = args.test_file
    main(emb_filename, id2word_filename, test_filename)
   