import nltk
import pickle
import argparse
from collections import Counter
import json
import numpy as np

def message_passing(data):
    
    new_factorgraph = []
    for i in range(len(data)):
        
        graph = data[i]
        graph.reset_factor_graph_prior() # reset the prior, this is necessary
        graph.message_passing()
        new_factorgraph.append(graph)

    return new_factorgraph 

def main(args):

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    new_factorgraph = message_passing(data)

    with open(args.out_path, 'wb+') as f:
        pickle.dump(new_factorgraph, f)
        f.close()

    print(" Message passing done \n")


# build the vocab given the description
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='',
                        help='path for train data file')
    parser.add_argument('--out_path', type=str, default='',
                        help='the result for next path')

    args = parser.parse_args()
    main(args)
