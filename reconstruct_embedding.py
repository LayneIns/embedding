'''
@Author: Hui Liu
@Date: 2018-12-05 14:47:35
@github: https://github.com/LayneIns
'''
import sys
import os
from tqdm import tqdm
import argparse
from gensim.models import KeyedVectors

def readId2word(id2word_filename):
    id2word = {}
    with open(id2word_filename) as fin:
        for line in fin:
            try:
                id, word = line.strip().split(' ')
            except:
                print(line)
                input()
            id = int(id)
            id2word[id] = word
    return id2word


def convert_embedding(id2word, embedding_filename, output_filename):
    line_cnt = 0
    with open(output_filename, "w") as fout:
        with open(embedding_filename) as fin:
            lines = fin.readlines()
            for line in tqdm(lines):
                if line_cnt == 0:
                    line_cnt += 1
                    fout.write(line.strip() + "\n")
                    continue
                line = line.strip()
                line_words = line.split()
                key = int(line_words[0])
                embedding = line_words[1:]
                word = id2word.get(key)
                fout.write("{} {}\n".format(word, " ".join(embedding)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, required=True)
    parser.add_argument('--id2word', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--eval', type=str, required=True)
    args = parser.parse_args()

    id2word_filename = args.id2word
    id2word = readId2word(id2word_filename)
    convert_embedding(id2word, args.embedding, args.output)
    
    evaluate_dir = args.eval
    files = ['SimVerb3500.tsv', 'RW2034.tsv', 'MEN3000.tsv', 'MTurk771.tsv', 'wordsim353.tsv']
    wv = KeyedVectors.load_word2vec_format(args.output, binary=False)
    for file in files:
        print("---------\n", file)
        similarities = wv.evaluate_word_pairs(os.path.join(evaluate_dir, file), dummy4unknown=True)
        print(similarities)



