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
from indirect_supervision import indirect_supervision
import sys

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

def generate_train_data(clinlical_match, entity_type, min_length, entity_type_freq, ctx_freq, hard_em, factor_set):

    # analyze overlapping between different types
    # disease and gene
    cnt = 0

    total = len(clinlical_match)
    data = {}
    total_positive = 0
    total_negative = 0
    total_entity = 0
    keys = clinlical_match.keys()

    opt = {}

    # single factor
    if factor_set >= 1:
        opt['token_length'] = True
        opt['str_length'] = True
        opt['postag'] = True
        opt['case_token'] = True
        opt['case_str'] = True
        opt['mention_ctx_entropy'] = True
        opt['parsetree_label'] = True
        opt['parsetree_label_neg'] = True
        opt['entity_entropy'] = True
        opt['special_char'] = True
        opt['special_token'] = True

    opt['logical_connective'] = False
    opt['acronym'] = False
    opt['same_mention'] = False
    opt['acronym_pairwise'] = False
    opt['cross_sentence'] = False

    if factor_set >= 2:
        # pairwise interaction for training
        # group 1
        opt['logical_connective'] = True
        # group 2
        opt['acronym'] = True
        opt['same_mention'] = True
        # group 3
        opt['acronym_pairwise'] = True
    
    if factor_set >= 3:
        # across sentence factor
        opt['cross_sentence'] = True

    # entity type, need to be careful
    opt['entity_type'] = entity_type

    # need to update this one later if we have more hurestics
    opt['entity_exclude_keys'] = {
        "gene": set(['drug', 'disease', '%']),
        "drug": set(['gene', '', '%'])
    }
    
    cnt = 0
    instance_cnt = 0

    # only cares about the training instances
    for trial in keys:
		
        data[trial.encode("ascii")] = {}
        data[trial.encode("ascii")]['inc'] = []
        data[trial.encode("ascii")]['exc'] = []
        predesessor = []

        for instance in clinlical_match[trial]['inc']:

            entity_label_inc = []
            text = instance['text']
            instance_cnt += 1

            # now it contains tree 
            if 'sentences' in instance['tree'].keys():

                if len(instance['tree']['sentences']) == 1 and instance['tree']['sentences'][0]['text'] == text: 

                    has_pos_mention = False
                    has_neg_mention = False

                    for key in instance['matched'].keys():
                                
                        item = instance['matched'][key]

                        if entity_type == key and has_pos_mention == False:
                            for k in item:
                                total_entity += 1
                                if check_length(k, text, min_length) and not check_ambuigity(k, key, instance['matched']):
                                    if not hard_em: 
                                        new_item = (int(k[0]), int(k[1]), [0.2, 0.8]) # initialized with uniform distribution, p(1) 
                                    else:
                                        new_item = (int(k[0]), int(k[1]), [0, 1]) # initialized with uniform distribution, p(1)
                                    
                                    if new_item not in entity_label_inc:
                                        entity_label_inc.append(new_item)
                                        total_positive += 1
                                        has_pos_mention = True

                        elif has_pos_mention == False:
                            for k in item:
                                total_entity += 1
                                if not check_ambuigity(k, key, instance['matched']) and total_negative <= total_positive*0.7:
                                    if not hard_em:
                                        new_item = (int(k[0]), int(k[1]), [0.8, 0.2]) # initialized with uniform distribution, p(1)
                                    else:
                                        new_item = (int(k[0]), int(k[1]), [1, 0]) # initialized with uniform distribution, p(1)

                                    if new_item not in entity_label_inc:
                                        entity_label_inc.append(new_item)
                                        total_negative += 1                
                                        has_neg_mention = True

                    # key data strucutre to save the file                 
                    instance['pos_neg_example'] = entity_label_inc # all the instances

                    if (has_pos_mention or has_neg_mention) and not hard_em:

                        # construct the factor graph for this sentences, might including more prior
                        graph = indirect_supervision(instance, opt, entity_type_freq, ctx_freq, predesessor)
                        # only call it one time 
                        pred = graph.construct_factor_graph()
                        predesessor.extend(pred) # extend all previous acronmy

                        instance['graph'] = graph
                        graph.message_passing()
                        cnt += 1

                        # do message passing one time
                        new_pos_neg_example = []
                        for idx, item in enumerate(instance['pos_neg_example']):    
                            # update the marginal probability used for future training
                            marginal = instance['graph'].get_marginals("z"+str(idx))
                            idx1 = item[0]
                            idx2 = item[1]
                            new_pos_neg_example.append((idx1, idx2, marginal.squeeze().tolist())) # get the probability 0                               
                        instance['pos_neg_example'] = new_pos_neg_example
                    
                    # keep only the instance that has mentions
                    if (has_pos_mention or has_neg_mention):
                        del instance['tree']
                        del instance['matched']
                        data[trial.encode("ascii")]['inc'].append(instance) 

        for instance in clinlical_match[trial]['exc']:
            
            entity_label_exc = []
            text = instance['text']
            instance_cnt += 1

            if 'sentences' in instance['tree'].keys():
                
                if len(instance['tree']['sentences']) == 1 and instance['tree']['sentences'][0]['text'] == text: 

                    has_pos_mention = False
                    has_neg_mention = False

                    for key in instance['matched'].keys():
            
                        item = instance['matched'][key]

                        if entity_type == key and has_pos_mention == False:
                            
                            for k in item:
                                total_entity += 1
                                if check_length(k, text, min_length) and not check_ambuigity(k, key, instance['matched']):
                                    if not hard_em:        
                                        new_item = (int(k[0]), int(k[1]), [0.2, 0.8]) # initialized with uniform distribution, p(1)
                                    else:
                                        new_item = (int(k[0]), int(k[1]), [0, 1]) # initialized with uniform distribution, p(1)

                                    if new_item not in entity_label_exc:
                                        entity_label_exc.append(new_item)
                                        total_positive += 1
                                        has_pos_mention = True

                        elif has_pos_mention == False:

                            # fake data, only get from the matched string, not from the random    
                            for k in item:
                                total_entity += 1
                                if not check_ambuigity(k, key, instance['matched']) and total_negative <= total_positive*0.7:
                                    if not hard_em:
                                        new_item = (int(k[0]), int(k[1]), [0.8, 0.2]) # initialized with uniform distribution, p(1)
                                    else:
                                        new_item = (int(k[0]), int(k[1]), [1, 0]) # initialized with uniform distribution, p(1)

                                    if new_item not in entity_label_exc:
                                        entity_label_exc.append(new_item)
                                        total_negative += 1
                                        has_neg_mention = True

                    # key data strucutre to save the file                 
                    instance['pos_neg_example'] = entity_label_exc # all the instances

                    if (has_pos_mention or has_neg_mention) and not hard_em:
                        graph = indirect_supervision(instance, opt, entity_type_freq, ctx_freq, predesessor)
                        # only call it one time 
                        pred = graph.construct_factor_graph()
                        predesessor.extend(pred) # extend all previous acronmy
                        instance['graph'] = graph
                        graph.message_passing()
                        cnt += 1

                        # do message passing one time
                        new_pos_neg_example = []
                        for idx, item in enumerate(instance['pos_neg_example']):    
                            # update the marginal probability used for future training
                            marginal = instance['graph'].get_marginals("z"+str(idx))
                            idx1 = item[0]
                            idx2 = item[1]
                            new_pos_neg_example.append((idx1, idx2, marginal.squeeze().tolist())) # get the probability 0                               
                        instance['pos_neg_example'] = new_pos_neg_example
                    
                    # keep only the instance that has mentions 
                    if has_pos_mention or has_neg_mention:
                        del instance['tree']
                        del instance['matched']
                        data[trial.encode("ascii")]['exc'].append(instance)

    print "total negative instance: {}".format(total_negative)
    print "total positive instance: {}".format(total_positive)

    print "total instance: %d \n" % instance_cnt    
    print "total factor graph: %d \n" % cnt         

    return data

def	main():
    
    parser = argparse.ArgumentParser("generate the train dataset for the entity linking problem")
    parser.add_argument('--input_clinlical_matching', help='input clinlical matching result from basic string matching', type=str, default='./data/pubmed_parsed/pubmed_parsed_splits_dict_0.pkl')
    parser.add_argument('--ouput_train', help='output the training data', type=str, default='')
    parser.add_argument('--entity_type', help='which data type', type=str, default='gene')
    parser.add_argument('--min_length_train', help='minumum length', type=int, default=3) # gene: 6 disease: 10
    parser.add_argument('--input_entitytype_freq', help='input entity type freq', type=str, default='./data/pubmed/entity_type.pkl')
    parser.add_argument('--input_entity_ctx_freq', help='input entity ctx freq', type=str, default='./data/pubmed/freq_uni.pkl')
    parser.add_argument('--hard_em', help='whether use hard em', action='store_true', default=False)
    parser.add_argument('--factor_set', help='factor set to activate', type=int, default=3)

    sys.setrecursionlimit(2000)

    args = parser.parse_args()
    # don't merge drugs now
    # output it as json file
    fp = open(args.input_clinlical_matching, 'r')
    clinlical_matching = pickle.load(fp)
    fp.close()

    fp = open(args.input_entitytype_freq, 'r')
    entitytype_freq = pickle.load(fp)
    fp.close()

    fp = open(args.input_entity_ctx_freq, 'r')
    entity_ctx_freq = pickle.load(fp)
    fp.close()

    hard_em = False
    if args.hard_em:
        hard_em = True
        
    data = generate_train_data(clinlical_matching, args.entity_type, args.min_length_train, entitytype_freq, entity_ctx_freq, hard_em, args.factor_set)
    fp = open(args.ouput_train, 'wb+')
    pickle.dump(data, fp)
    fp.close()

if __name__ == '__main__':
    main()
