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
#import matplotlib.pyplot as plt
#import pandas
import operator
from visualizer2 import make_html_file
from indirect_supervision import indirect_supervision
import sys
import random
import string

def check_ambuigity(query, class_type, entity_linking):
    
    for key in entity_linking.keys():
        if key != class_type:
            for i in range(len(entity_linking[key])):
                a = entity_linking[key][i][0]
                b = entity_linking[key][i][1]
                if query[0] == a and query[1] == b:
                    return True

    return False

def check_length(query, text, min_length):
    
    # only one word
    if query[0] == query[1]:
        w = text[query[0]]
        if len(w)>= min_length or (w.upper()==w and len(w)>=min_length):
            return True
        else:
            return False

    return True

def analyze_data(data):
    
    total_length = 0
    total_entity = 0
    total_overlap = 0
    cnt = 0
    
    token_num_statistic = Counter()
    alphabet_num_statistic = Counter()

    for trial in data.keys():
        
        for instance in data[trial]['inc']:
            total_length += len(instance['text'])
            total_entity += len(instance['pos_neg_example'])
            cnt += 1
        for instance in data[trial]['exc']:    
            total_length += len(instance['text'])
            total_entity += len(instance['pos_neg_example'])
            cnt += 1

        length =[]
        for instance in data[trial]['inc']:    
            length.extend([item[1]-item[0]+1 for item in instance['pos_neg_example']])
        for instance in data[trial]['exc']:    
            length.extend([item[1]-item[0]+1 for item in instance['pos_neg_example']])
            
        token_num_statistic.update(length)

        # the string length
        length =[]
        for instance in data[trial]['inc']:
            length.extend(len(" ".join(instance['text'][item[0]:item[1]+1])) for item in instance['pos_neg_example'])
        for instance in data[trial]['exc']:    
            length.extend(len(" ".join(instance['text'][item[0]:item[1]+1])) for item in instance['pos_neg_example'])

        alphabet_num_statistic.update(length)

    print "average tokens per case: {}".format(float(total_length)/cnt) 
    print "average entities per case: {}".format(float(total_entity)/cnt) 

    # get the length statistics 
    alphabet_num_statistic = dict(alphabet_num_statistic)
    for key in alphabet_num_statistic.keys():
        alphabet_num_statistic[key] = alphabet_num_statistic[key]/float(total_entity)

    # show the length statistics
    # print alphabet_num_statistic

    # get the length statistics 
    token_num_statistic = dict(token_num_statistic)
    for key in token_num_statistic.keys():
        token_num_statistic[key] = token_num_statistic[key]/float(total_entity)

    # show the length statistics
    # print token_num_statistic

def analyze_data_test(data, entity_type):
    
    total_type = 0
    total_over = 0
    cnt = 0

    for trial in data.keys():
        
        for instance in data[trial]['inc']:        
            for key in instance['matched'].keys(): 
                if key == entity_type:
                    # inclusion
                    for item in instance['matched'][key]:
                        total_type += 1
                        if check_ambuigity(item, key, instance['matched']):
                            total_over += 1
            cnt += 1                
        for instance in data[trial]['exc']:        
            for key in instance['matched'].keys(): 
                if key == entity_type:
                    # inclusion
                    for item in instance['matched'][key]:
                        total_type += 1
                        if check_ambuigity(item, key, instance['matched']):
                            total_over += 1
            cnt += 1

    print "average matched entity: {}".format(total_type) 
    print "average overlapped entities: {}".format(total_over)                
    print "average matched entity per case: {}".format(float(total_type)/cnt) 
    print "average overlapped entities per case: {}".format(float(total_over)/cnt) 
    print "average overlapped ratio per case: {}".format(float(total_over)/total_type)


def generate_train_data(clinlical_matching, train_keys, entity_type, min_length_train, entitytype_freq, entity_ctx_freq, hard_em, factor_set):
    
    # now do the paralell message passing updating by writing all the files      
    chunk = {}
    chunk_size = 1000
    if hard_em:
        chunk_size = 5000 # takes around 15 minutes on slurm, totally will result 100 cpu jobs on slurm
        
    idx = 1
    total_chunk = 0
    randomstring = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30))
    random_dir = "./temp/%s"%randomstring
    if os.path.exists(random_dir):
        os.rmdir(random_dir)  
    os.mkdir(random_dir)
    
    for trial in train_keys:
        
        chunk[trial] = clinlical_matching[trial]
        if idx%chunk_size == 0:
            input_file = os.path.join(random_dir, "chunk_"+str(total_chunk) + "_" + entity_type + ".pkl")
            fp = open(input_file, "wb+")
            pickle.dump(chunk, fp)
            fp.close()
            # now call the shell to lanuch the job
            output_file = os.path.join(random_dir, "chunk_train_"+str(total_chunk) + "_" + entity_type + ".pkl")
            command = "bash submit_generatedata.sh %s %s %s %s %s %s %s %d" % (input_file, output_file, entity_type, min_length_train, entitytype_freq, entity_ctx_freq, hard_em, factor_set)
            print (command)
            os.system(command)
            time.sleep(10)
            print ("submitted generation job to slurm \n")
            chunk = {}
            total_chunk += 1

        idx += 1
    
    # the last chunks 
    if len(chunk.keys()) > 0:        
        input_file = os.path.join(random_dir, "chunk_"+str(total_chunk) + "_" + entity_type + ".pkl")
        fp = open(input_file, "wb+")
        pickle.dump(chunk, fp)
        fp.close()
        # now call the shell to lanuch the job
        output_file = os.path.join(random_dir, "chunk_train_"+str(total_chunk) + "_" + entity_type + ".pkl")
        command = "bash submit_generatedata.sh %s %s %s %s %s %s %s %d" % (input_file, output_file, entity_type, min_length_train, entitytype_freq, entity_ctx_freq, hard_em, factor_set)
        print (command)
        os.system(command)
        print ("submitted generation job to slurm \n")
        total_chunk += 1

    print ("total %d jobs submitted \n"%total_chunk)

    # checking all the new files generated
    while True:
        
        total_finished = 0
        for i in range(total_chunk):
            if not os.path.exists(os.path.join(random_dir, "chunk_train_"+str(i) + "_" + entity_type + ".pkl")):
                time.sleep(5)    
            else:
                total_finished += 1
        
        if total_finished == total_chunk:
            cacheds = []
            new_state = []
            for i in range(total_chunk):
                cached = os.stat(os.path.join(random_dir, "chunk_train_"+str(i) + "_" + entity_type + ".pkl")).st_mtime
                cacheds.append(cached)
            time.sleep(60)
            for i in range(total_chunk):
                curr = os.stat(os.path.join(random_dir, "chunk_train_"+str(i) + "_" + entity_type + ".pkl")).st_mtime
                new_state.append(curr)

            if new_state == cacheds:         
                break

    print ("data generation finished \n")
    # checking the result until get all of them, and then return the new marginal probability
    result = {}
    for i in range(total_chunk):
        fp = open(os.path.join(random_dir, "chunk_train_"+str(i) + "_" + entity_type + ".pkl"), "rb")
        chunk = pickle.load(fp)
        result.update(chunk)
        fp.close()

    return result


def generate_valid_test_data(clinlical_match, val_keys, test_keys, entity_type, min_length):

	# analyze overlapping between different types
	# disease and gene
    cnt = 0
    val = {}
    test = {}

    total = len(clinlical_match)

    data = {}

    total_positive = 0
    total_negative = 0
    total_entity = 0

    for trial in val_keys:
		
        # currently we ignore the other type such as the disease & drug
        # currently we only care about the disease and gene 
        # for the inclusion

        data[trial.encode("ascii")] = {}
        data[trial.encode("ascii")]['inc'] = []
        data[trial.encode("ascii")]['exc'] = []

        for instance in clinlical_match[trial]['inc']:
            
            entity_label_inc = []
            text = instance['text']

            # instance['tree'].keys(): sentences
            # instance['tree']['sentences'][0].keys(): parsetree, text, dependencies, words
            if 'sentences' in instance['tree'].keys():

                if len(instance['tree']['sentences']) == 1 and instance['tree']['sentences'][0]['text'] == text:

                    for key in instance['matched'].keys():
                                
                        item = instance['matched'][key]

                        if entity_type == key:

                            for k in item:
                                total_entity += 1
                                # simple hurestic, this might filter lots of single entry data, but also make the training data clean
                                # checking whether this exist in other dataset
                                # at least three 2 
                                if  check_length(k, text, min_length) and not check_ambuigity(k, key, instance['matched']):
                                    new_item = (int(k[0]), int(k[1]), [0, 1])  # probability == 1
                                    if new_item not in entity_label_inc:
                                        entity_label_inc.append(new_item)
                                        total_positive += 1
                        else:
                            # fake data, only get from the matched string, not from the random   
                            for k in item:
                                total_entity += 1
                                # only not in other type
                                if not check_ambuigity(k, key, instance['matched']) and total_negative <= total_positive:
                                    new_item = (int(k[0]), int(k[1]), [1, 0]) # probability == 1
                                    if new_item not in entity_label_inc:
                                        entity_label_inc.append(new_item)
                                        total_negative += 1                

                    # key data strucutre to save the file                 
                    instance['pos_neg_example'] = entity_label_inc
                    if len(entity_label_inc) > 0:
                        del instance['tree']
                        del instance['matched']               
                        data[trial.encode("ascii")]['inc'].append(instance) 

        for instance in clinlical_match[trial]['exc']:
            
            entity_label_exc = []
            text = instance['text']

            if 'sentences' in instance['tree'].keys():

                if len(instance['tree']['sentences']) == 1 and instance['tree']['sentences'][0]['text'] == text:

                    for key in instance['matched'].keys():
            
                        item = instance['matched'][key]

                        if entity_type == key:
                        
                            for k in item:
                                total_entity += 1
                                # fake data, only get from the matched string, not from the random
                                if check_length(k, text, min_length) and not check_ambuigity(k, key, instance['matched']):
                                    new_item = (int(k[0]), int(k[1]), [0, 1]) 
                                    if new_item not in entity_label_exc:
                                        entity_label_exc.append(new_item)
                                        total_positive += 1
                        else:
                            # fake data, only get from the matched string, not from the random    
                            for k in item:
                                total_entity += 1
                                # only not in other type
                                if not check_ambuigity(k, key, instance['matched']) and total_negative <= total_positive:
                                    new_item = (int(k[0]), int(k[1]), [1, 0]) 
                                    if new_item not in entity_label_exc:
                                        entity_label_exc.append(new_item)
                                        total_negative += 1
		
                    # key data strucutre to save the file                 
                    instance['pos_neg_example'] = entity_label_exc
                    if len(entity_label_exc) > 0:
                        del instance['tree']
                        del instance['matched']
                        data[trial.encode("ascii")]['exc'].append(instance)

    #print "total validation and test instance: {}".format(total_entity)    
    print "total valid & test negative instance: {}".format(total_negative)
    print "total valid & test positive instance: {}".format(total_positive)

    val = data        
    # test data for visualization
    for key in test_keys:
        test[key] = clinlical_match[key]
                
    return val, test

def display_sample(data, html_file):
    
    sample = {}
    for key in data.keys():
        sample[key] ={}
        sample[key]['inc'] = []
        sample[key]['exc'] = []

        for instance in data[key]['inc']:
            context_inc = instance['text']
            matches_inc = instance['pos_neg_example']
            sample[key]['inc'].append((context_inc, matches_inc))

        for instance in data[key]['inc']:            
            context_exc = instance['text']
            matches_exc = instance['pos_neg_example']            
            sample[key]['exc'].append((context_exc, matches_exc))

    make_html_file(sample, html_file)


def	main():

	parser = argparse.ArgumentParser("generate the train, test and validation dataset for the entity linking problem")
	parser.add_argument('--input_clinlical_matching', help='input clinlical matching result from basic string matching', type=str, default='./data/pubmed_parsed/pubmed_parsed_splits_dict_0.pkl')
	parser.add_argument('--ouput_train', help='output the training data', type=str, default='')
	parser.add_argument('--ouput_test', help='output the test data', type=str, default='')
	parser.add_argument('--ouput_val', help='output the validation data', type=str, default='')
	parser.add_argument('--train_ratio', help='training data ratio', type=float, default=0.8)
	parser.add_argument('--test_ratio', help='the test data ratio', type=float, default=0.1)
	parser.add_argument('--val_ratio', help='the validation data ratio', type=float, default=0.1)
	parser.add_argument('--entity_type', help='which data type', type=str, default='gene')
	parser.add_argument('--min_length', help='minumum length', type=int, default=5) # gene: 6 disease: 10
	parser.add_argument('--html_file', help='html file for visualization', type=str, default='')
	parser.add_argument('--min_length_train', help='minumum length', type=int, default=3) # gene: 6 disease: 10

	parser.add_argument('--input_entitytype_freq', help='input entity type freq', type=str, default='./data/pubmed_parsed/entity_type.pkl')
	parser.add_argument('--input_entity_ctx_freq', help='input entity ctx freq', type=str, default='./data/pubmed_parsed/freq_uni.pkl')

	parser.add_argument('--hard_em', help='whether use hard em', type=int, default=1)
	parser.add_argument('--cv', help='cross validation, from [0,1,2,3,4]', type=int, default=1) # make sure each fold has the same 
	parser.add_argument('--factor_set', help='factor set to activate', type=int, default=3)

	sys.setrecursionlimit(20000)

	#print ("remove all previous chunk files \n")
	#command = "rm ./result/chunk_*.pkl"
	#os.system(command)


	args = parser.parse_args()
	# don't merge drugs now
	# output it as json file
	print (args.hard_em)
	print (args.input_clinlical_matching)		
	fp = open(args.input_clinlical_matching, 'r')
	clinlical_matching = pickle.load(fp)
	fp.close()
    
	hard_em = False
	if args.hard_em == 1:
		hard_em = True

	# deal with the cross validation, shuffle the keys and get the corresponding keys
	keys = clinlical_matching.keys()
	np.random.seed(args.cv)
	np.random.shuffle(keys)
    
	train_cnt = int(len(keys)*args.train_ratio)
	val_cnt = int(len(keys)*args.val_ratio)
	test_cnt = int(len(keys)*args.test_ratio)
	train_keys = keys[:train_cnt]
	val_keys = keys[train_cnt:train_cnt+val_cnt]
	test_keys = keys[train_cnt+val_cnt:]

	train = generate_train_data(clinlical_matching, train_keys, args.entity_type, args.min_length_train, args.input_entitytype_freq, args.input_entity_ctx_freq, hard_em, args.factor_set)
	val, test = generate_valid_test_data(clinlical_matching, val_keys, test_keys, args.entity_type, args.min_length)
	
    # we don't need to analyze it
    # analyze_data(train)
	
	analyze_data(val)

    # display the valid examples 
	display_sample(val, args.html_file)

    # analyze the test data
	analyze_data_test(test, args.entity_type)
    
	
    # save the matching result to the file
	fp = open(args.ouput_train, "wb+")		
	pickle.dump(train, fp)
	fp.close()

	fp = open(args.ouput_val, "wb+")		
	pickle.dump(val, fp)
	fp.close()        

	fp = open(args.ouput_test, "wb+")		
	pickle.dump(test, fp)
	fp.close()
	

if __name__ == '__main__':
    main()
	
