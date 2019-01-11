from __future__ import generators
from pdb import set_trace as st
import os
import numpy as np
import argparse
import json
import time
from nltk.tokenize import word_tokenize
import thread 
from collections import Counter 
import pickle

def generate_data(clinlical_match, min_confidence):


    keys = clinlical_match.keys()
    freq = Counter()

    for trial in keys:
		
        # currently we ignore the other type such as the disease&drug
        # currently we only care about the disease and gene 
        # for the inclusion

        for instance in clinlical_match[trial]['inc']:
            text = instance['text']
            freq.update(text)

        for instance in clinlical_match[trial]['exc']:
            text = instance['text']
            freq.update(text)

    # normalize the count
    total = float(sum(freq.values()))

    freq = dict(freq)
    for key in freq.keys():
        freq[key] /= total

    sort_value = sorted(freq.values())
    median = sort_value[int(len(sort_value)*min_confidence)-1]
    print len(sort_value)

    filter_freq = {}
    for key in freq.keys():
        if freq[key] <= median:    
            filter_freq[key] = freq[key]
    
    print len(filter_freq)

    return filter_freq

def	main():

	parser = argparse.ArgumentParser("calculate the unigram frequency")
	parser.add_argument('--input_clinlical_matching', help='input clinlical matching result from basic string matching', type=str, default='./data/pubmed_parsed/pubmed_parsed_splits_0.pkl')
	parser.add_argument('--result_file', help='the freq of each token', type=str, default="./data/pubmed_parsed/freq_uni.pkl")
	parser.add_argument('--min_confidence', type=float,  default=1, help='how much entity we want to keep')

	args = parser.parse_args()
	# don't merge drugs now
	# output it as json file		
	fp = open(args.input_clinlical_matching, 'r')
	clinlical_matching = pickle.load(fp)
	fp.close()
	
	freq = generate_data(clinlical_matching, args.min_confidence)
    
    # save the matching result to the file
	fp = open(args.result_file, "wb+")		
	pickle.dump(freq, fp)
	fp.close()
    
if __name__ == '__main__':
    main()
	
