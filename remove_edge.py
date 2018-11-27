'''
@Author: Hui Liu
@Date: 2018-11-20 13:50:29
@github: https://github.com/LayneIns
'''

# This code takes the separated edge files and combines them, while filtering 
# out the edges with a weight that is less than the threshold.

import sys
import os
from tqdm import tqdm
import argparse


def mergefile(location, num, threshold):
    edge_list = []
    max_weight = 0
    for i in tqdm(range(num+1)):
        filename = os.path.join(location, "edgelist_"+str(i))
        with open(filename) as fin:
            line_cnt = 0
            for line in fin:
                if line_cnt % 100000 == 0:
                    sys.stdout.flush()
                    sys.stdout.write(' '*20 + '\r')
                    sys.stdout.flush()
                    sys.stdout.write(str(line_cnt)+"\r")
                line_cnt += 1

                id1, id2, weight = line.strip().split()
                weight = int(weight)
                if weight >= threshold:
                    if weight > max_weight:
                        max_weight = weight
                    edge_list.append([id1, id2, weight])

    return edge_list, max_weight


def output(output_file, edge_list, max_weight):
    with open(output_file, "w") as fout:
        for edge in tqdm(edge_list):
            fout.write("{} {} {}\n".format(edge[0], edge[1], edge[2]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--num', type=int, required=True)
    parser.add_argument('--t', type=int, required=True)
    parser.add_argument('--output', type=str, default="all_edges")
    args = parser.parse_args()
    file_location = args.location
    file_num = args.num
    threshold = args.t
    output_name = args.output

    edge_list, max_weight = mergefile(file_location, file_num, threshold)

    print("Output merged file...")
    output(os.path.join(file_location, output_name), edge_list, max_weight)

