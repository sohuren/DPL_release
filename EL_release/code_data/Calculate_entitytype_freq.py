import nltk
import pickle
import argparse
from collections import Counter
import json
import numpy as np


# test procedure
def GetResult(data, entity_type):
    
    freq = Counter()

    for trial in data.keys():
        
        for i in range(len(data[trial]['inc'])):

            instance = data[trial]['inc'][i]
            text_inc = instance['text']

            # only care the specified entity type
            for item in instance['matched'][entity_type]:        
                entity = " ".join(text_inc[item[0]:item[1]+1])
                freq.update([entity])

        for i in range(len(data[trial]['exc'])):
            
            instance = data[trial]['exc'][i]
            text_exc = instance['text']

            # only care the specified entity type
            for item in instance['matched'][entity_type]:    
                entity = " ".join(text_exc[item[0]:item[1]+1])
                freq.update([entity])

    total = float(sum(freq.values()))
    freq = dict(freq)
    
    for key in freq.keys():
        
        freq[key] /= total
    
    return freq

def main(args):

    fp = open(args.data_path, "r")		
    data = pickle.load(fp)
    fp.close()

    freq = GetResult(data, args.entity_type)
    # save the matching result to the file
    fp = open(args.result_file, "wb+")		
    pickle.dump(freq, fp)
    fp.close()


# build the vocab given the description
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='./data/pubmed_parsed/pubmed_parsed_splits.pkl',
                        help='')
    parser.add_argument('--entity_type', type=str,
                        default='gene',
                        help='entity type')
    parser.add_argument('--result_file', type=str,
                        default='./data/pubmed_parsed/entity_type.pkl',
                        help='save the frequency')
    args = parser.parse_args()
    main(args)
