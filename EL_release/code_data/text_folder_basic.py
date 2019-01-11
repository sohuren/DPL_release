###############################################################################
# Base Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code
###############################################################################

import torch.utils.data as data
import os
import os.path
import pickle
import torch


def make_dataset(file_name):
    
    assert os.path.isfile(file_name), '%s is not a valid file' % file_name
    fp = open(file_name, "rb")
    dataset = pickle.load(fp)
    fp.close()
    return dataset

class TxtFolder_MLP(data.Dataset):

    def __init__(self, file_name, vocab, window_size, valid_token):
        
        dataset = make_dataset(file_name)

        if len(dataset) == 0:
            raise(RuntimeError("Found 0 clinlical trial in: " + file_name + "\n"))

        self.file_name = file_name
        self.vocab = vocab

        # Load the valid token
        with open(valid_token, 'rb') as f:
            valid_tokens = pickle.load(f)
            valid_tokens = valid_tokens.keys()
            f.close()
            print ("the valid tokens are loaded \n")

        data = []
        maxi_num = 0
        # generate all the data when initialize the dataset instance
        for key in dataset.keys():
            if maxi_num >= 1000000:
                break

            for instance in dataset[key]['inc']:
                 
                text_inc = instance['text']
                # assume that the label here is the "+1" or "-1" and the triple 
                label_inc = instance['pos_neg_example']

                for item in label_inc:
                    # inclusion, get the window size
                    tokens = text_inc[item[0]-4*window_size:item[0]]
                    tokens.extend(text_inc[item[1]+1:item[1]+1+4*window_size]) # need to check this one
                    tokens_inc = []
                    tokens_inc.extend([vocab(token) for token in tokens if token in valid_tokens and vocab(token)!=vocab('<unk>') ])
                    # only care the one that has full history 
                    if len(tokens_inc) >= window_size:
                        mild = len(tokens_inc)/2
                        # get the most close context that has valid tokens
                        target_inc = torch.LongTensor(tokens_inc[mild-window_size/2:mild+window_size/2])
                        data.append((target_inc, torch.LongTensor([item[2]])))
                        maxi_num += 1
            
            for instance in dataset[key]['exc']:
                
                text_exc = instance['text']
                # assume that the label here is the "+1" or "-1" and the triple 
                label_exc = instance['pos_neg_example']
                
                for item in label_exc:   
                    # exclusion, get the window size
                    tokens = text_exc[item[0]-4*window_size:item[0]] 
                    tokens.extend(text_exc[item[1]+1:item[1]+1+4*window_size]) # need to check this one
                    tokens_exc = []
                    tokens_exc.extend([vocab(token) for token in tokens if token in valid_tokens and vocab(token)!=vocab('<unk>') ])
                    if len(tokens_exc) >= window_size:
                        mild = len(tokens_exc)/2
                        # get the most close context that has valid tokens
                        target_exc = torch.LongTensor(tokens_exc[mild-window_size/2:mild+window_size/2])                        
                        data.append((target_exc, torch.LongTensor([item[2]])))
                        maxi_num += 1

        self.data = data

    def __getitem__(self, index):
        
        data = self.data[index] # near-by window, and the target label
        return data[0], data[1]  

    def __len__(self):
        return len(self.data)
