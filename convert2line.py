'''
@Author: Hui Liu
@Date: 2018-11-20 15:47:15
@github: https://github.com/LayneIns
'''

# This code takes input of format (i, j, weight) with i < j,
# and outputs both (i, j, weight) and (j, i, weight).
# This is REQUIRED to define an undirected weighted graph to be used in LINE.


import sys
import os
import argparse
from tqdm import tqdm

import numpy as np


def main(infilename, outfilename):
    with open(infilename) as fin:
        with open(outfilename, "w") as fout:
            lines = fin.readlines()
            for line in tqdm(lines):
                line = line.strip()
                id1, id2, weight = line.split()
                fout.write("{} {} {}\n".format(id1, id2, weight)) 
                fout.write("{} {} {}\n".format(id2, id1, weight)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    infilename = args.infile
    outfilename = args.output

    main(infilename, outfilename)
