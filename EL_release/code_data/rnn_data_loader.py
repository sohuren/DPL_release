import random
import torch.utils.data
import torchvision.transforms as transforms
from base_data_loader import BaseDataLoader
from text_folder_rnn import TxtFolder_RNN
from pdb import set_trace as st
from builtins import object
import math
import pickle
from operator import itemgetter

def collate_fn(data):

    """Creates mini-batch tensors from the list of tuples (image, caption).
    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: (text, caption).
            - text: torch tensor of shape (?); variable length.
    Returns:
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """

    # Sort a data list by caption length (descending order)

    data.sort(key=lambda x: len(x[0]), reverse=True)

    text, mask, label = zip(*data)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in text]
    
    # merge captions 
    targets = torch.zeros(len(text), max(lengths)).long()
    for i, cap in enumerate(text):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    # masks: mask for the candidates, 1 is the candidate position    
    masks = torch.zeros(len(text), max(lengths)).long()
    for i, ma in enumerate(mask):
        masks[i, ma[0]:ma[1]+1] = 1  # start index and end index

    # batch_masks: mask for the length, used when calculate the attention, 1 is the paded element
    batch_masks = torch.ones(len(text), max(lengths)).long()
    
    for i, ma in enumerate(text):
        
        # choice 1: whole context
        end = lengths[i]
        # batch_masks[i, :end] = 1  # 0 is the original element
        # choice 2 :just cares about the short context (-10:10)
        ma = mask[i]
        start_pos = max([ma[0]-10, 0])
        end_pos = min([ma[1]+10, end])
        batch_masks[i, start_pos:end_pos] = 0
    

    labels = torch.zeros(len(mask), 2).float()
    for i, l in enumerate(label):
        labels[i, :] = l

    return targets, batch_masks, masks, lengths, labels 

class RnnDataLoader(BaseDataLoader):
    
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        
        dataset = TxtFolder_RNN(file_name = opt.file_path, vocab = opt.vocab)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=int(self.opt.nThreads),
            drop_last=True,
            collate_fn = collate_fn
        )
        self.dataset = dataset
        self.rnn_data = data_loader

    def name(self):
        return 'RNNDataLoader'

    def load_data(self):
        return self.rnn_data

    def __len__(self):
        return len(self.dataset)