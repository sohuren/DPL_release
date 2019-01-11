from pdb import set_trace as st
import os
import numpy as np
import argparse
import pickle
from collections import Counter 
import pickle
import operator
from copy import deepcopy
import json

def Convert_data(data):

	# all the example
	cnt = 0
	total = len(data.keys())
	new_data = deepcopy(data)

	for trial in data.keys():
    
		for i in range(len(data[trial]['inc'])):	
			instance = data[trial]['inc'][i]
			if instance['tree']:
    				
				result_dict = json.loads(instance['tree'].encode('ascii', 'ignore'))
				if 'sentences' in result_dict.keys():
					for idx in range(len(result_dict['sentences'])):
						if 'text' in result_dict['sentences'][idx].keys(): 
							result_dict['sentences'][idx]['text'] = result_dict['sentences'][idx]['text'].split()
				new_data[trial]['inc'][i]['tree'] = result_dict
			else:
				new_data[trial]['inc'][i]['tree'] = {}

		for i in range(len(data[trial]['exc'])):	
			instance = data[trial]['exc'][i]
			if instance['tree']:
				result_dict = json.loads(instance['tree'].encode('ascii', 'ignore'))
				if 'sentences' in result_dict.keys():
					for idx in range(len(result_dict['sentences'])):
						if 'text' in result_dict['sentences'][idx].keys():
							result_dict['sentences'][idx]['text'] = result_dict['sentences'][idx]['text'].split()
				new_data[trial]['exc'][i]['tree'] = result_dict
			else:
				new_data[trial]['exc'][i]['tree'] = {}
			
		if cnt%10 == 0:
			print ("finished: {}/{}".format(cnt, total))

		cnt += 1

	return new_data

def	main():

	parser = argparse.ArgumentParser("convert the dependency matching result fromn string to dict so that we can easily parse it")
	parser.add_argument('--input_file', help='the input file', type=str, default='./data/pubmed_parsed/pubmed_parsed_splits.pkl')
	parser.add_argument('--output_file', help='output the new files with new format', type=str, default='./data/pubmed_parsed/pubmed_parsed_splits_dict.pkl')
    
	args = parser.parse_args()
	fp = open(args.input_file, 'rb')
	data = pickle.load(fp)

	# break the data into different 
	parsed_data = Convert_data(data)

	# save the data as new format
	#fp = open(args.output_file, 'wb')
	#pickle.dump(parsed_data, fp)
	#fp.close()

    # save the data as new format
	p = pickle.Pickler(open(args.output_file, 'wb'))
	p.fast = True
	p.dump(parsed_data) # we can use this as the

if __name__ == '__main__':
    main()
