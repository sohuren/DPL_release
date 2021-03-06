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
    
    # re-read the predicted probability
    fp = open(args.input_probability_file, "r")
    corpus_y = [] 
    lines = fp.readlines()
    pos = 0
    index = []
    idx = 0
    for line in lines:
      if float(line.split("\t")[1].strip()) >= 0.5:
        corpus_y.append("0 1") # predicted as truth
        pos += 1
        index.append(idx)
      else:
        corpus_y.append("1 0") # predicted as false
      idx += 1

    fp.close()
    assert (len(corpus_x) == len(corpus_y))
    print("total pos: %d \n" % pos)
    assert len(index) == pos

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
    random_dir = "./data/temp/%s"%randomstring
    if os.path.exists(random_dir):
        os.rmdir(random_dir)  
    os.mkdir(random_dir)

    # only use half of the all data for active learning
    for i in range(len(corpus_x)/2):
        if i in index:
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
                command = "bash ./theano_src/submit_generate_factorgraph2.sh %s %s %s %s %d" % (file_name, output_file, args.input_entitytype_freq, args.input_entity_ctx_freq, args.factor_set)
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
    command = "bash ./theano_src/submit_generate_factorgraph2.sh %s %s %s %s %d" % (file_name, output_file, args.input_entitytype_freq, args.input_entity_ctx_freq, args.factor_set)
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
            time.sleep(30)
            for i in range(total_chunk):
                curr = os.stat(os.path.join(random_dir, "_temp_"+str(i))).st_mtime
                new_state.append(curr)

            if new_state == cacheds:         
                break

    print ("data generation finished \n")
    time.sleep(180)

    # checking the result until get all of them, and then return the new marginal probability
    result = []
    for i in range(total_chunk):
        fp = open(os.path.join(random_dir, "_temp_"+str(i)), "rb")
        chunk = pickle.load(fp)
        result.extend(chunk)
        fp.close()
    
    #assert (len(index) == len(result))

    # open the annonymized file
    fp = open(args.anno_file, 'r')
    anno = fp.readlines()
    for idx, instance in enumerate(anno):
        anno[idx] = "\t".join(instance.strip().split("\t")[:-1])
    fp.close()

    # open the graph file
    fp = open(args.graph_file, 'r')
    graph = fp.readlines()
    for idx, instance in enumerate(graph):
        graph[idx] = instance.strip()
    fp.close()

    # choose the one that its label is different from the prediction file, i.e., from 1 to 0 under factors
    active_samples = []
    graph_samples = []
    #fp = open("factor_true", "wt+")
    for i in range(len(result)):
        factor_graph = result[i]
        marginal = factor_graph.get_marginals('z'+str(0)).squeeze().tolist()   # assume we only have one factor
        # only cares about the false positive
        if corpus_y[index[i]] == "0 1" and marginal[0] >= marginal[1]:
          active_samples.append(anno[index[i]] + "\t" + " ".join([str(x) for x in marginal]))
          graph_samples.append(graph[index[i]])

        #elif corpus_y[i] == "0 1":
          #fp.write(anno[i] + "\t" + " ".join([str(x) for x in marginal]) + "\n")
          #print ("find some\n")
    #fp.close()

    # merge those two files together
    previous_sample = []
    if os.path.exists(args.input_previous_sample_file):
        fp = open(args.input_previous_sample_file, 'r')
        previous_sample = fp.readlines()
        for idx, instance in enumerate(previous_sample):
            previous_sample[idx] = instance.strip()
        fp.close()

    previous_sample_graph = []
    if os.path.exists(args.input_previous_graph_file):
        fp = open(args.input_previous_graph_file, 'r')
        previous_sample_graph = fp.readlines()
        for idx, instance in enumerate(previous_sample_graph):
            previous_sample_graph[idx] = instance.strip()
        fp.close()

    # extend the previous example
    assert len(active_samples) == len(graph_samples)
    
    for i in range(len(active_samples)):
        if active_samples[i] not in previous_sample:
            previous_sample.append(active_samples[i])
            previous_sample_graph.append(graph_samples[i])
    
    assert len(previous_sample) == len(previous_sample_graph)

    # write all the examples to file
    fp = open(args.output_samples_file, 'wt+')
    for idx, instance in enumerate(previous_sample):
        fp.write(instance + "\n")
    fp.close()

    fp = open(args.output_graph_file, 'wt+')
    for idx, instance in enumerate(previous_sample_graph):
        fp.write(instance + "\n")
    fp.close()


# build the vocab given the description
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input 
    parser.add_argument('--raw_file', type=str,
                        default='../data/drug_var/0/sentences_raw',
                        help='path for raw sentence file')
    parser.add_argument('--anno_file', type=str, default='../data/drug_var/0/graph_arcs',
                        help='the result for next path')
    parser.add_argument('--input_probability_file', type=str, default='../data/drug_var/0/sentences_2nd',
                        help='the result for next path')

    parser.add_argument('--input_previous_sample_file', type=str, default='../data/drug_var/0/sentences_2nd',
                        help='the result for next path')
    parser.add_argument('--input_previous_graph_file', type=str, default='../data/drug_var/0/sentences_2nd',
                        help='the result for next path')

    # input for messaging passing
    parser.add_argument('--input_gene_prob_file', type=str, default='../data/drug_var/0/sentences_2nd',
                        help='the result for next path')
    parser.add_argument('--graph_file', type=str, default='../data/drug_var/0/graph_arcs',
                        help='the result for next path')

    # config for the message passing
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='chunk size for each data type')
    parser.add_argument('--input_entitytype_freq', 
                        help='input entity type freq', type=str, default='../data/entity_type.pkl')
    parser.add_argument('--input_entity_ctx_freq', 
                        help='input entity ctx freq', type=str, default='../data/freq_uni.pkl')
    parser.add_argument('--factor_set', type=int, default=3,
                        help='factor set we used')
    parser.add_argument('--num_entities', type=int, default=3,
                        help='num_entities')

    # selected candidates, used for training in next iteration
    parser.add_argument('--output_samples_file', type=str, default='../data/drug_var/0/sentences_3nd',
                        help='the result for next path')
    parser.add_argument('--output_graph_file', type=str, default='../data/drug_var/0/sentences_3nd',
                        help='the result for next path')

    args = parser.parse_args()
    main(args)
