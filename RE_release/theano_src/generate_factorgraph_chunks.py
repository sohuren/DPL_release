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
import operator
from indirect_supervision import indirect_supervision
import sys


def generate_factorgraph(data, entity_type, entity_type_freq, ctx_freq, arc_type_dict, factor_set, multi_instance):

    cnt = 0
    opt = {}

    # single factor related to the gene, drug, var, entity@2017.10.30
    if factor_set == 1:
        # entity factor
        opt['token_length'] = True
        opt['str_length'] = True
        opt['postag'] = False
        opt['case_token'] = True
        opt['case_str'] = True
        opt['mention_fig'] = True
        opt['mention_ctx_entropy'] = False
        opt['parsetree_label'] = False
        opt['parsetree_label_neg'] = False
        opt['entity_entropy'] = False
        opt['special_char'] = True
        opt['entity_gene_score'] = True
        opt['drug_sentivity'] = True
        opt['gene'] = True

        # relation factor
        opt['cross_sentence'] = False
        opt['acronym'] = False
        opt['table'] = False
        opt['ref'] = False
        opt['intro'] = False
        opt['fig_table'] = False
        opt['unnormal_seq'] = False
        opt['triple_pair'] = False # disable when do the first message passing @ 2018.1.26
        opt['gene_mutation'] = False # disable when do the first message passing @ 2018.1.26
        opt["z_re"] = False

    if factor_set == 2:

        # entity factor
        opt['token_length'] = False
        opt['str_length'] = False
        opt['postag'] = False
        opt['case_token'] = False
        opt['case_str'] = False
        opt['mention_fig'] = False
        opt['mention_ctx_entropy'] = False
        opt['parsetree_label'] = False
        opt['parsetree_label_neg'] = False
        opt['entity_entropy'] = False
        opt['special_char'] = False
        opt['entity_gene_score'] = False
        opt['drug_sentivity'] = False
        opt['gene'] = False

        # relation factor
        opt['cross_sentence'] = True
        opt['acronym'] = True
        opt['table'] = True
        opt['ref'] = True
        opt['intro'] = True
        opt['fig_table'] = True
        opt['unnormal_seq'] = True
        opt['triple_pair'] = True # disable when do the first message passing, able it in active learning setting @ 2018.1.26
        opt['gene_mutation'] = True # disable when do the first message passing, able it in active learning setting @ 2018.1.26
        opt["z_re"] = False

    if factor_set >= 3:
        
        # entity factor
        opt['token_length'] = True
        opt['str_length'] = True
        opt['postag'] = False
        opt['case_token'] = True
        opt['case_str'] = True
        opt['mention_fig'] = True
        opt['mention_ctx_entropy'] = False
        opt['parsetree_label'] = False
        opt['parsetree_label_neg'] = False
        opt['entity_entropy'] = False
        opt['special_char'] = True
        opt['entity_gene_score'] = True
        opt['drug_sentivity'] = True
        opt['gene'] = True

        # relation factor
        opt['cross_sentence'] = True
        opt['acronym'] = True
        opt['table'] = True
        opt['ref'] = True
        opt['intro'] = True
        opt['fig_table'] = True
        opt['unnormal_seq'] = True
        opt['triple_pair'] = True # disable when do the first message passing, able it in active learning setting @ 2018.1.26 
        opt['gene_mutation'] = True # disable when do the first message passing, able it in active learning setting @ 2018.1.26
        opt["z_re"] = False

    # entity type, need to be careful
    opt['entity_type'] = entity_type

    # need to update this one later if we have more hurestics
    opt['entity_exclude_keys'] = {
        "gene": set(['drug', 'disease', '%'])
    }
    
    predesessor = []
    factor_graph = []
    
    # check about the same sentences to implement at at least one method

    # only cares about the training instances
    for i in range(len(data)):
		
        instance = data[i]
        # construct the factor graph for this sentences, might including more prior
        graph = indirect_supervision(instance, opt, entity_type_freq, ctx_freq, arc_type_dict, predesessor)
        # only call it one time 
        pred = graph.construct_factor_graph()
        graph.message_passing()
        predesessor.extend(pred) # extend all previous acronmy
        factor_graph.append(graph)
        cnt += 1
    
    if multi_instance:
        
        # culster the data now due to MIL
        cluster = []
        for i in range(len(data)):
            x, y, mentions, dependencies = data[i]
            cluster.append(" ".join(x)) 

        # modify the method due to MIL
        dis_cluster = list(set(cluster))
        text2cluster = {}
        for i in range(len(dis_cluster)):
            text = dis_cluster[i]
            # find all the same text
            index = [idx for (idx, t) in enumerate(cluster) if t==text ]
            text2cluster[text] = index

        text2prob = {}
        for key in text2cluster.keys():
            index = text2cluster[key]
            prob = 1
            for i in index:
                graph = factor_graph[i]
                mariginal = graph.get_marginals('z0')
                prob_0 = mariginal.tolist()[0][0] # p(0)
                prob *= prob_0
            factor = 1.0/(1.01 - prob) # approximated solution
            text2prob[key] = [1-0.5*factor, 0.5*factor]    

        # modify the method due to MIL
        for i in range(len(factor_graph)):             
            graph = factor_graph[i]
            x, y, mentions, dependencies = data[i]
            key = " ".join(x)
            # this assumption only applied to the positive instance
            if y != "None" or y != "1 0":
                graph.update_factor_graph_pairwise('MIL0', 'z0', np.asarray([text2prob[key]]))
                # do the new message passing again
                graph.message_passing()
            factor_graph[i] = graph

        print "total factor graph: %d \n" % cnt         

    return factor_graph

def	main():
    
    parser = argparse.ArgumentParser("do the message passing for the RE problem")
    
    parser.add_argument('--input_data', help='input clinlical matching result from basic string matching', type=str, default='')
    parser.add_argument('--ouput_data', help='output the factor graph', type=str, default='')
    parser.add_argument('--input_entitytype_freq', help='input entity type freq', type=str, default='../data/entity_type.pkl')
    parser.add_argument('--input_entity_ctx_freq', help='input entity ctx freq', type=str, default='../data/freq_uni.pkl')
    parser.add_argument('--factor_set', help='factor set to activate', type=int, default=3)
    parser.add_argument('--multi_instance', help='MIL', type=bool, default=False) # change this if want MIL

    sys.setrecursionlimit(2000)

    args = parser.parse_args()
    # don't merge drugs now
    # output it as json file
    fp = open(args.input_data, 'r')
    clinlical_matching = pickle.load(fp)
    fp.close()
    data = clinlical_matching['data']
    arc_type_dict = clinlical_matching['arc_type']

    fp = open(args.input_entitytype_freq, 'r')
    entitytype_freq = pickle.load(fp)
    fp.close()

    fp = open(args.input_entity_ctx_freq, 'r')
    entity_ctx_freq = pickle.load(fp)
    fp.close()

    #entity_type= ['gene', 'drug']    
    entity_type= 'gene'
    factor_graph = generate_factorgraph(data, entity_type, entitytype_freq, entity_ctx_freq, arc_type_dict, args.factor_set, args.multi_instance)
    
    fp = open(args.ouput_data, 'wb+')
    pickle.dump(factor_graph, fp)
    fp.close()

if __name__ == '__main__':
    main()
