import nltk
import pickle
import argparse
from collections import Counter
import json
import numpy as np
import codecs as cs
import os
import random
import string
import time

def read_raw_file(filename, num_entities=2):
    corpus_x = []
    corpus_y = []
    corpus_idx = []
    with cs.open(filename, 'r', encoding='utf-8') as inf:
        line_count = 0
        for line in inf:
            line_count += 1
            line = line.strip()
            if len(line) == 0:
                continue
            #sentence, entity_ids_1, entity_ids_2, label = line.split('\t') 
            elems = line.split('\t') 
            entity_id_arry = []
            for ett in elems[1:1+num_entities]:
                entity_id = map(int, ett.split(' '))
                entity_id_arry.append(entity_id)
            assert len(entity_id_arry) == num_entities
            assert len(elems) == num_entities + 2
            x = elems[0].lower().split(' ')
            label = elems[-1]
            try:
                for i in range(num_entities):
                    assert entity_id_arry[i][-1] < len(x)
            except:
                sys.stderr.write('abnormal entity ids:'+str(entity_id_arry)+', sentence length:'+str(len(x))+'\n')
                continue
            
            if len(x) < 1:
                print x 
                continue
             
            y = label.strip()
            corpus_x.append(x)
            corpus_y.append(y)
            corpus_idx.append(entity_id_arry)
    print 'read file', filename, len(corpus_x), len(corpus_y), len(corpus_idx)
    return corpus_x, corpus_y, corpus_idx 

def read_graph_dependencies(graph_file, arc_type_dict, fine_grain=True):
    dep_graphs = []
    if 'adjtok' not in arc_type_dict:
        arc_type_dict['adjtok'] = 0
    with open(graph_file, 'r') as parents_f:
        while True:
            cur_parents = parents_f.readline()
            if not cur_parents :
                break
            cur_deps = [[elem.split('::') for elem in p.split(',,,')] for p in cur_parents.strip().split(' ')]
            for p in cur_parents.strip().split(' '):
                for elem in p.split(',,,'):
                    temp = elem.split('::')
                    try:
                        assert len(temp) == 2
                    except:
                        print elem, p
            dep_graphs.append(construct_graph_deps(cur_deps, arc_type_dict, fine_grain))
    return dep_graphs, arc_type_dict

# get a dict of the arc_types and dependencies with types
def construct_graph_deps(dep_array, arc_type_dict, fine_grain=True):
    dep_graph = []
    ignore_types = ['prevsent', 'coref', 'discSenseInv', 'adjsent', 'discSense', 'nextsent', 'depsent', 'discExplicitInv', 'discExplicit', 'depinv']
    focus_types = ['deparc'] #, 'depinv']
    for i, elem in enumerate(dep_array):
        local_dep = []
        for pair in elem:
            arc_type = pair[0].split(':')[0]
            if fine_grain and arc_type in focus_types:
                arc_type = pair[0][:11]
            dep_node = int(pair[1])
            if dep_node < 0 or arc_type in ignore_types: # or arc_type == 'deparc' or arc_type == 'depinv':   
                continue
            # I modified this to support bi-direction
            if dep_node != i:
                if arc_type not in arc_type_dict:
                    arc_type_dict[arc_type] = len(arc_type_dict) 
                local_dep.append((dep_node, arc_type_dict[arc_type]))
        try:
            assert (len(local_dep) > 0 or i == 0)
        except:
            #print i, elem
            pass
        dep_graph.append(local_dep)
    return dep_graph


def main(args):

    arc_type_dict=dict()

    # read the data
    corpus_x, corpus_y, corpus_idx = read_raw_file(args.raw_file, args.num_entities)
    dependencies, arc_type_dict = read_graph_dependencies(args.graph_file, arc_type_dict)
    
    fp = open(args.input_gene_prob_file, "r")
    gene_prob = fp.readlines()
    fp.close()
    for idx in range(len(gene_prob)):
        gene_prob[idx] = float(gene_prob[idx].strip())

    # construct the data
    merge_data = []
    assert (len(corpus_x) == len(dependencies))
    chunk = []
    idx = 0
    chunk_idx = 0
    
    randomstring = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30))
    random_dir = "../data/temp/%s"%randomstring
    if os.path.exists(random_dir):
        os.rmdir(random_dir)  
    os.mkdir(random_dir)

    for i in range(len(corpus_x)):            
        instance = (corpus_x[i], corpus_y[i], corpus_idx[i], dependencies[i], gene_prob[i])
        chunk.append(instance)
        idx += 1
        if idx% args.chunk_size == 0:
            file_name = os.path.join(random_dir, str(chunk_idx))
            fp = open(file_name, 'wb+')
            chunk_data = {'data': chunk, 'arc_type': arc_type_dict}
            pickle.dump(chunk_data, fp)
            fp.close()
            
            # now call the shell to lanuch the job
            output_file = os.path.join(random_dir, "_temp_"+str(chunk_idx))
            command = "bash submit_generate_factorgraph.sh %s %s %s %s %d" % (file_name, output_file, args.input_entitytype_freq, args.input_entity_ctx_freq, args.factor_set)
            os.system(command)
            time.sleep(10)

            chunk = []
            chunk_idx += 1

    # the last dimension of the data
    file_name = os.path.join(random_dir, str(chunk_idx))
    fp = open(file_name, 'wb+')
    chunk_data = {'data': chunk, 'arc_type': arc_type_dict}
    pickle.dump(chunk_data, fp)
    fp.close()

    # now call the shell to lanuch the job
    output_file = os.path.join(random_dir, "_temp_"+str(chunk_idx))
    command = "bash submit_generate_factorgraph.sh %s %s %s %s %d" % (file_name, output_file, args.input_entitytype_freq, args.input_entity_ctx_freq, args.factor_set)
    print (command)
    os.system(command)
    time.sleep(10)
    chunk_idx += 1
    

    # call the shell for launch the script to build the factorgraph
    # checking all the new files generated
    total_chunk = chunk_idx
    print ("total %d jobs submitted \n"% total_chunk)
    
    while True:
        
        total_finished = 0
        for i in range(total_chunk):
            if not os.path.exists(os.path.join(random_dir, "_temp_"+str(i))):
                time.sleep(5)    
            else:
                total_finished += 1
        
        if total_finished == total_chunk:
            cacheds = []
            new_state = []
            for i in range(total_chunk):
                cached = os.stat(os.path.join(random_dir, "_temp_"+str(i))).st_mtime
                cacheds.append(cached)
            time.sleep(60)
            for i in range(total_chunk):
                curr = os.stat(os.path.join(random_dir, "_temp_"+str(i))).st_mtime
                new_state.append(curr)

            if new_state == cacheds:         
                break

    print ("data generation finished \n")
    # checking the result until get all of them, and then return the new marginal probability
    result = []
    for i in range(total_chunk):
        fp = open(os.path.join(random_dir, "_temp_"+str(i)), "rb")
        chunk = pickle.load(fp)
        result.extend(chunk)
        fp.close()

    # save the file 
    fp = open(args.result_file, "wb+")
    pickle.dump(result, fp)
    fp.close()
    
    # output the probability to text file
    fp = open(args.input_probability_file, "r")
    lines = fp.readlines()
    fp.close()

    assert (len(lines) == len(result))
    fp = open(args.output_probability_file, 'wt+')

    for i in range(len(result)):
        factor_graph = result[i]
        marginal = factor_graph.get_marginals('z'+str(0)).squeeze().tolist()   # assume we only have one factor
        text = "\t".join(lines[i].strip().split("\t")[:-1])
        fp.write( text + "\t" + " ".join([str(x) for x in marginal]) + "\n")
    fp.close()

    # generate the fake probability to test the pipeline
    '''
    prediction_f = args.input_probability_file
    prediction_f = prediction_f.replace("sentences_2nd", "prediction")
    print (prediction_f)
    fp = open(prediction_f, "wt+")
    for i in range(len(result)):
        factor_graph = result[i]
        marginal = factor_graph.get_marginals('z'+str(0)).squeeze().tolist()
        fp.write(" ".join([str(x) for x in marginal]) + "\n")
    fp.close()
    '''

# build the vocab given the description
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_file', type=str,
                        default='../data/drug_var/0/sentences_raw',
                        help='path for raw sentence file')
    
    # concatenate this input_probability_file with the updated message passing file
    parser.add_argument('--input_probability_file', type=str, default='../data/drug_var/0/sentences_2nd',
                        help='the result for next path')
    parser.add_argument('--input_gene_prob_file', type=str, default='../data/drug_var/0/sentences_2nd',
                        help='the result for next path')

    # this is the output_probability_file, used for training in next iteration
    parser.add_argument('--output_probability_file', type=str, default='../data/drug_var/0/sentences_3nd',
                        help='the result for next path')

    parser.add_argument('--graph_file', type=str, default='../data/drug_var/0/graph_arcs',
                        help='the result for next path')

    parser.add_argument('--result_file', type=str, default='../data/drug_var/0/factorgraph.pkl',
                        help='the result for next path')

    parser.add_argument('--chunk_size', type=int, default=500,
                        help='chunk size for each data type')
    parser.add_argument('--input_entitytype_freq', 
                        help='input entity type freq', type=str, default='../data/entity_type.pkl')
    parser.add_argument('--input_entity_ctx_freq', 
                        help='input entity ctx freq', type=str, default='../data/freq_uni.pkl')
    parser.add_argument('--factor_set', type=int, default=3,
                        help='factor set we used')
    parser.add_argument('--num_entities', type=int, default=2,
                        help='num_entities')

    args = parser.parse_args()
    main(args)