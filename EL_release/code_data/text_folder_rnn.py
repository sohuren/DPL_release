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

# need to re-write this code
def make_dataset(file_name):    

    assert os.path.isfile(file_name), '%s is not a valid file' % file_name
    fp = open(file_name, "rb")
    dataset = pickle.load(fp)
    fp.close()
    
    '''
    for key in dataset.keys():
        for instance in dataset[key]['inc']:
            if 'graph' in instance.keys():
                instance['graph'] = []
        for instance in dataset[key]['exc']:
            if 'graph' in instance.keys():
                instance['graph'] = []
    '''
    
    return dataset

def pro_dataset(dataset):    

    '''
    for key in dataset.keys():
        for instance in dataset[key]['inc']:
            if 'graph' in instance.keys():
                instance['graph'] = []
        for instance in dataset[key]['exc']:
            if 'graph' in instance.keys():
                instance['graph'] = []
    '''    
    return dataset

class TxtFolder_RNN(data.Dataset):

    def __init__(self, file_name, vocab):

        if isinstance(file_name, dict):
            dataset = pro_dataset(file_name) # if the file name is already the variable, no need to load from disk to save time
        else:    
            dataset = make_dataset(file_name) # if the file name is string, load from disk

        if len(dataset) == 0:
            raise(RuntimeError("Found 0 clinlical trial in: " + file_name + "\n"))

        self.data = dataset
        self.instance = []
        maxi_num = 0
        self.text = []
        self.lens = []

        for key in self.data.keys():

            if maxi_num >= 10000000:
               break

            for instance in self.data[key]['inc']:
                text = instance['text']
                len_text = len(text)
                tokens = []
                tokens.extend([vocab(token) for token in text])
                tokens = torch.LongTensor(tokens)
                self.text.append(tokens)
                for example in instance['pos_neg_example']:
                    mask = [example[0], example[1]]
                    self.instance.append((len(self.text)-1, mask, torch.FloatTensor(example[2]))) # marginal probability for p(1) 
                    self.lens.append(len_text)
                    maxi_num += 1

            for instance in self.data[key]['exc']:
                text = instance['text']
                len_text = len(text)
                tokens = []
                tokens.extend([vocab(token) for token in text])
                tokens = torch.LongTensor(tokens)
                self.text.append(tokens)
                for example in instance['pos_neg_example']:
                    mask = [example[0], example[1] ]                                 
                    self.instance.append((len(self.text)-1, mask, torch.FloatTensor(example[2]))) # marginal probability for p(1)
                    self.lens.append(len_text)
                    maxi_num += 1

        self.vocab = vocab
        
        # sort the data according to the data len to reduce the memory
        indices = sorted(range(len(self.lens)), key=lambda k: self.lens[k])
        self.sort_instance = self.instance
        for i in range(len(indices)):
            self.sort_instance[i] = self.instance[indices[i]]
        
        self.instance = self.sort_instance
        # del this varaible to save memory
        del self.data
        del self.sort_instance

    # we need to modify this rnn version    
    def __getitem__(self, index):
                
        # text label is different
        data = self.instance[index]
        # the text is re-used here to reduce the memory  
        return self.text[data[0]], data[1], data[2]

    def __len__(self):
        return len(self.instance)
