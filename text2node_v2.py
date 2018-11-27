'''
@Author: Hui Liu
@Date: 2018-11-19 10:06:13
@github: https://github.com/LayneIns
'''

# This code takes a training text file, and outputs co-occurrence matrix in format (i, j, weight),
# and an id2word file that maps id to the corresponding word.
# The code makes use of multi-thread to speed up output.
# The code also outputs the original co-occurrence matrix and id2word in pickled format.

import sys
import os
import re
from tqdm import tqdm
from multiprocessing import Pool

# import numpy as np

# gensim
import gensim

import argparse

# featurization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


# Preprocess text using given analyzer
def preprocess(text, analyzer):
    preprocessed = analyzer(text)
    return preprocessed


# Preprocess all texts in the file
# tokenization, removal of stopwords
def preprocessAll(filename):
    vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english')
    analyzer = vectorizer.build_analyzer()
    all_text = []
    line_cnt = 0
    with open(filename, 'r') as f:
        for line in f:
            if line_cnt % 10000 == 0:
                sys.stdout.flush()
                sys.stdout.write(" " * 25 + '\r')
                sys.stdout.flush()
                sys.stdout.write(str(line_cnt) + " lines processed.\r")
            line_cnt += 1
            # if line_cnt >= 2000000:
            #     break
            preprocessed = preprocess(line, analyzer)
            all_text.append(preprocessed)

    return all_text


to_remove_list_wiki = [
    "unk", "doc", "id", "url", "en", "wikipedia", "wiki", "org", "curid"
]


def removeWiki(token):
    if token in to_remove_list_wiki:
        return True
    if re.fullmatch('(dg)+', token) is not None:
        return True


# Get all tokens with frequency >= min_c
# text is list of lists of tokens
def getTokenFreq(text, min_c=10):
    token_freq = {}
    for sen in tqdm(text):
        for t in sen:
            if token_freq.get(t) is None:
                token_freq[t] = 0
            token_freq[t] += 1

    # removeWiki is used only for wiki dataset
    to_remove = [
        k for k, v in token_freq.items() if v < min_c or removeWiki(k)
    ]
    for k in to_remove:
        del token_freq[k]

    return token_freq


# Get word2id and id2word correspondence
def getCorrespondence(token_freq):
    word2id = {}
    id2word = []
    idx = 0
    for k, _ in token_freq.items():
        word2id[k] = idx
        idx += 1
        id2word.append(k)
    return word2id, id2word


# Generate co-occurrence matrix
def getCooccur(text, word2id, id2word, win=5):
    print("Start getCooccur---")
    edge_dict = {}
    for sen in tqdm(text):
        sen_len = len(sen)
        for pos1 in range(sen_len):
            id1 = word2id.get(sen[pos1], -1)
            if id1 == -1:
                continue
            for pos2 in range(
                    max(0, pos1 - win), min(sen_len, pos1 + win + 1)):
                id2 = word2id.get(sen[pos2], -1)
                if id2 == -1 or pos1 == pos2:
                    continue
                if id1 > id2:
                    id1, id2 = id2, id1
                key = str(id1) + "_" + str(id2)
                if key not in edge_dict.keys():
                    edge_dict[key] = 0
                edge_dict[key] += 1
    return edge_dict


# Helper function for multi-threaded output
def splitIdx(arr, size):
    arrs = []
    while len(arr) > size:
        a = arr[:size]
        arrs.append(a)
        arr = arr[size:]
    if len(arr) > 0:
        arrs.append(arr)
    return arrs


# Helper function for multi-threaded output
def multiOutput(id, edge_list, min_count):
    M = len(edge_list)
    OUTFILE = open('./output/'+ str(min_count) + '/edgelist_' + str(id), 'w')
    for i in tqdm(range(M)):
        id1, id2 = edge_list[i][0].split("_")
        OUTFILE.write('{} {} {}\n'.format(id1, id2, edge_list[i][1]))
    OUTFILE.close()


def main(train_filename, min_count=10):
    # Preprocess all texts
    all_text = preprocessAll(train_filename)
    print("Finished preprocessAll!!!")

    # Parameters
    print("min count: " + str(min_count))

    token_freq = getTokenFreq(all_text, min_count)
    print("Finished getTokenFreq!!!")

    word2id, id2word = getCorrespondence(token_freq)
    print("Finished getCorrespondence!!!")
    print(len(id2word), "words in the dictionary.")

    edge_dict = getCooccur(all_text, word2id, id2word)
    print("Finished getCooccur!!!")

    print("Start outputting id2word---")
    OUTFILE_id2word = open('output/id2word_'+str(min_count), 'w')
    for idx, word in enumerate(id2word):
        OUTFILE_id2word.write('{} {}\n'.format(idx, word))
    OUTFILE_id2word.close()
    print("Finished outputting id2word!!!")

    edge_list = [(key, value) for key, value in edge_dict.items()]
    # with open("output/id2word.pkl", 'wb') as f:
    #     pickle.dump(id2word, f)
    # with open("output/edgelist.pkl", 'wb') as f:
    #     pickle.dump(edge_list, f)

    print("Start outputting edge list---")
    N = len(edge_list)
    size = 50000000
    n_proc = N // size
    if N % size != 0:
        n_proc += 1

    print("There are", n_proc, "threads in total.")
    with Pool(n_proc) as p:
        p.starmap(multiOutput, [(i, edge_list[i * size: (i + 1) * size], min_count)
                                for i in range(n_proc)])

    print("Finished outputting edge list!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--min', type=int)
    args = parser.parse_args()
    filename = args.infile
    min_count = args.min

    if not os.path.exists(os.path.join("output/", str(min_count))):
        os.makedirs(os.path.join("output/", str(min_count)))

    main(train_filename=filename, min_count=min_count)
