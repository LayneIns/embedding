# Author: Hui Liu
# Date: Nov. 2018

# This code takes input of format (i, j, weight) with i < j,
# and outputs both (i, j, weight) and (j, i, weight).
# This is REQUIRED to define an undirected weighted graph to be used in LINE.


import sys
import os
import argparse

import numpy as np


def main(infilename, outfilename):
    dt = np.dtype([('id1', np.int32), ('id2', np.int32), ('weight', np.float64)])
    dt1 = np.loadtxt(infilename, dtype=dt)
    print(dt1.shape)
    dt2 = np.zeros(dt1.shape, dtype=dt)
    dt2[:, 0] = dt1[:, 1]
    dt2[:, 1] = dt1[:, 0]
    dt2[:, 2] = dt1[:, 2]
    dt = np.concatenate((dt1, dt2), axis=0)
    np.savetxt(outfilename+".line", dt, fmt='%d %d %.19f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    infilename = args.infile
    outfilename = args.output

    main(infilename, outfilename)
