'''
@Author: Hui Liu
@Date: 2018-12-05 16:09:22
@github: https://github.com/LayneIns
'''
import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id2word', type=str, required=True)
    parser.add_argument('--eval', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    word_list = []
    oov_word = []
    with open(args.id2word) as fin:
        for line in fin:
            key, value = line.strip().split()
            word_list.append(value)
    
    with open(args.eval) as fin:
        line_cnt = 0
        for line in fin:
            if line_cnt == 0:
                line_cnt += 1
                continue
            word1, word2, sim = line.strip().split("\t")
            if word1 not in word_list:
                oov_word.append(word1)
            if word2 not in word_list:
                oov_word.append(word2)
    
    with open(args.output, 'w') as fout:
        fout.write("\n".join(oov_word))
    