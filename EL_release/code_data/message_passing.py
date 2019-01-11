import nltk
import pickle
import argparse
from collections import Counter
import json
import numpy as np

def message_passing(result):

    for trial in result.keys():
        
        for i in range(len(result[trial]['inc'])):
            instance = result[trial]['inc'][i]
            text_inc = instance['text']

            # only care the specified entity type
            if 'sentences' in instance['tree'].keys():

                if len(instance['tree']['sentences']) == 1 and (instance['tree']['sentences'][0]['text'] == text_inc):
    
                    if len(instance['pos_neg_example']) >= 1:
                        instance['graph'].reset_factor_graph_prior() # reset the prior, this is necessary
                        instance['graph'].message_passing()
                    new_pos_neg_example = []
                    #print (instance['text'])
                    for idx, item in enumerate(instance['pos_neg_example']):    
                        # update the marginal probability used for future training
                        marginal = instance['graph'].get_marginals("z"+str(idx))
                        idx1 = instance['pos_neg_example'][idx][0]
                        idx2 = instance['pos_neg_example'][idx][1]
                        #print (instance['text'][idx1:idx2+1])
                        #print (marginal.squeeze().tolist())
                        new_pos_neg_example.append((idx1, idx2, marginal.squeeze().tolist())) # get the probability 0
                    instance['pos_neg_example'] = new_pos_neg_example
            
            result[trial]['inc'][i] = instance

        for i in range(len(result[trial]['exc'])):
            
            instance = result[trial]['exc'][i]
            text_exc = instance['text']

            # only care the specified entity type

            if 'sentences' in instance['tree'].keys():

                if len(instance['tree']['sentences']) == 1 and (instance['tree']['sentences'][0]['text'] == text_exc):

                    if len(instance['pos_neg_example']) >= 1:
                        instance['graph'].reset_factor_graph_prior()  # reset the prior, this is necessary  
                        instance['graph'].message_passing()
                    new_pos_neg_example = []
                    #print (instance['text'])
                    for idx, item in enumerate(instance['pos_neg_example']):
                        # update the marginal probability used for future training
                        marginal = instance['graph'].get_marginals("z"+str(idx))
                        idx1 = instance['pos_neg_example'][idx][0]
                        idx2 = instance['pos_neg_example'][idx][1]
                        #print (instance['text'][idx1:idx2+1])
                        #print (marginal.squeeze().tolist())
                        new_pos_neg_example.append((idx1, idx2, marginal.squeeze().tolist())) # get the probability 0
                    instance['pos_neg_example'] = new_pos_neg_example

            result[trial]['exc'][i] = instance
        
    return result

def main(args):

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    result = message_passing(data)
    with open(args.out_path, 'wb+') as f:
        pickle.dump(result, f)
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
