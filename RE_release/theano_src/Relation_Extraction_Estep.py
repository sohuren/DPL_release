import nltk
import pickle
import argparse
from collections import Counter
import json
import numpy as np
import codecs as cs
from utils import update_graph_parameters
from utils import accumulate_gradient
import random
import string
import time
import os
from copy import deepcopy

def distribute_message_passing(factor_graph, marginals):
    
    chunk = []
    idx = 0
    chunk_idx = 0

    randomstring = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30))
    random_dir = "./data/temp/%s"%randomstring
    if os.path.exists(random_dir):
        os.rmdir(random_dir)  
    os.mkdir(random_dir)

    for i in range(len(factor_graph)):
                    
        instance = factor_graph[i]
        marginal = marginals[i]

        # reset the DL factor, do the message passing under the new graph parameters
        instance.update_factor_graph_unary('DL0', np.asarray(marginal, dtype=np.float32))
        chunk.append(instance)
        idx += 1

        if idx% args.chunk_size == 0:
            input_file = os.path.join(random_dir, str(chunk_idx) + ".pkl") # for the temp files
            fp = open(input_file, 'wb+')
            pickle.dump(chunk, fp)
            fp.close()
            
            # now call the shell to lanuch the job
            output_file = os.path.join(random_dir, "mp_"+str(chunk_idx)+".pkl")
            command = "bash ./theano_src/submit_message_passing.sh %s %s" % (input_file, output_file)
            print (command)
            os.system(command)
            time.sleep(10)
            chunk = []
            chunk_idx += 1

    # the last dimension of the data
    input_file = os.path.join(random_dir, str(chunk_idx) + ".pkl")
    fp = open(input_file, 'wb+')
    pickle.dump(chunk, fp)
    fp.close()

    # now call the shell to lanuch the job
    output_file = os.path.join(random_dir, "mp_"+str(chunk_idx)+".pkl")
    command = "bash ./theano_src/submit_message_passing.sh %s %s" % (input_file, output_file)
    print (command)
    os.system(command)
    time.sleep(10)
    chunk_idx += 1

    total_chunk = chunk_idx
    # call the shell for launch the script to build the factorgraph
    # checking all the new files generated
    while True:
        total_finished = 0
        for i in range(total_chunk):
            if not os.path.exists(os.path.join(random_dir, "mp_"+str(i)+".pkl")):
                time.sleep(5)    
            else:
                total_finished += 1
        
        if total_finished == total_chunk:
            cacheds = []
            new_state = []
            for i in range(total_chunk):
                cached = os.stat(os.path.join(random_dir, "mp_"+str(i)+".pkl")).st_mtime
                cacheds.append(cached)
            time.sleep(60)
            for i in range(total_chunk):
                curr = os.stat(os.path.join(random_dir, "mp_"+str(i)+".pkl")).st_mtime
                new_state.append(curr)

            if new_state == cacheds:         
                break

    print ("message passing finished \n")

    # checking the result until get all of them, and then return the new marginal probability
    result = []
    for i in range(total_chunk):
        fp = open(os.path.join(random_dir, "mp_"+str(i)+".pkl"), "rb")
        chunk = pickle.load(fp)
        result.extend(chunk)
        fp.close()

    # get the new marginal 
    new_marginal = []
    for i in range(len(result)):
        instance = result[i]
        marginal = instance.get_marginals("z0") # get the mariginal
        new_marginal.append(marginal.squeeze().tolist())

    return result, new_marginal   

def optimize_factorgraph(data, marginals):
    
    # calculate the gradient over all the samples
    current_parameters = {} # only need to get it one time
    grads = []

    for i in range(len(data)):
        graph = data[i]
        label = np.argmax(marginals[i])
        graph.set_label([label]) # set the new label
        current_parameters.update(graph.get_current_parameters()) # reset the prior, this is necessary
        grads.append(graph.compute_gradient()) # get the gradient from this sample
                        
    # update the graph parameters given all the parameter key 
    sample_gradient = accumulate_gradient(grads, current_parameters.keys())
    print ("old parameters ")
    print (current_parameters)

    print ("gradient ")
    print (sample_gradient)
    new_parameters = update_graph_parameters(current_parameters, sample_gradient, args.learn_rate_graph, args.graph_regularizer)

    print ("new parameters ")
    print (new_parameters)

    new_graph = []
    for i in range(len(data)):
        graph = data[i]
        graph.update_all_factor_graph_pairwise(new_parameters) # reset the prior, this is necessary
        new_graph.append(graph)

    return new_graph

def main(args):

    # add one more splits here on the new data set
    total_split = [0, 1, 2, 3, 4, 5]
    cv = args.cv
    lens_split = []

    # read all 4 files
    factorgraph = []
    marginals = []

    for split in total_split:
        if split not in cv:
            file_path = os.path.join(args.data_dir, str(split)+"/factorgraph_%d.pkl"%args.factor_set)
            fp = open(file_path, 'rb')
            chunk = pickle.load(fp)
            factorgraph.extend(chunk)

            # read the data from dl as new ground truth
            file_path = os.path.join(args.data_dir, str(split)+"/prediction")
            fp = open(file_path, 'r')
            lines = fp.readlines()
            lens_split.append(len(lines))

            for line in lines:
                prob_1 = float(line.split()[1]) # the file record is the probability of 1
                marginal = [1-prob_1, prob_1]
                marginals.append(marginal)

    # optimize the graph given the new marginal
    assert(len(factorgraph) == len(marginals))

    factorgraph = optimize_factorgraph(factorgraph, marginals)

    # update the marginal with the new deep learning factor, also, the new graph
    new_factorgraph, new_marginal = distribute_message_passing(factorgraph, marginals)

    # write the marginal to file
    start_idx = 0
    idx = 0
    for split in total_split:
        if split not in cv:
            chunk = new_marginal[start_idx:start_idx+lens_split[idx]] 
            dl_marginals = marginals[start_idx:start_idx+lens_split[idx]]
            file_path = os.path.join(args.data_dir, str(split)+"/sentences_2nd")
            fp = open(file_path, 'r')
            lines = fp.readlines()
            fp.close()
            assert (len(chunk) == len(lines))
            output_file = os.path.join(args.data_dir, str(split)+"/sentences_3nd_"+str(args.epoch))
            fp = open(output_file, 'wt+')

            # generate the new probability files
            for i in range(len(chunk)):
                
                marginal = [chunk[i][j]*dl_marginals[i][j] for j in range(len(chunk[i])]
                # multiply with previous marginal probability and normalize it
                marginal /= sum(marginal) 
                text = "\t".join(lines[i].strip().split("\t")[:-1])
                fp.write(text + "\t" + " ".join([str(x) for x in marginal]) + "\n")
            fp.close()
            start_idx += lens_split[idx]
            idx += 1

    # save the factor graph for next iteration
    start_idx = 0
    idx = 0
    for split in total_split:
        if split not in cv:
            chunk = new_factorgraph[start_idx:start_idx+lens_split[idx]] 
            file_path = os.path.join(args.data_dir, str(split)+"/factorgraph_%d.pkl"%args.factor_set)
            fp = open(file_path, 'wb+')
            pickle.dump(chunk, fp)
            fp.close()
            start_idx += lens_split[idx]
            idx += 1

# build the vocab given the description
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        default='../data/drug_gene_var',
                        help='path for raw sentence file')

    parser.add_argument('--cv', type=int, metavar='N', nargs='+',
                        help='cross validation split, we can speed up the procedure here')

    parser.add_argument('--epoch', type=int, default=0,
                        help='the result for next M step')

    parser.add_argument('--factor_set', type=int, default=3,
                        help='factor set we want to do message passing [1, 2, 3]')

    parser.add_argument('--learn_rate_graph', help='learning rate of graph parameters', type=float, default=0.01)
    parser.add_argument('--graph_regularizer', help='regularizer of graph parameters', type=float, default=1e-4)
    parser.add_argument('--chunk_size', type=int, default=500,
                        help='chunk size for each data type')
                        
    args = parser.parse_args()
    main(args)