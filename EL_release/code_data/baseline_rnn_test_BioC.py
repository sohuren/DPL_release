from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from network import EncoderRNN
from data_loader import CreateDataLoader
import os
import pickle
from build_vocab import Vocabulary
from visualizer import make_html_file
from visualizer import make_html_file_confidence
from copy import deepcopy
from utils import make_csv_file
import numpy as np
import sys
import time

# Test code with the pytorch
parser = argparse.ArgumentParser(description='Test code: entities disambuity with PyTorch')

parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--word_embedding', type=str, default='./data/pubmed_parsed/embedding_vec_gene.pkl',
                    help='initial word embedding file')
parser.add_argument('--dataroot', type=str, default='./data/bioc',
                    help='the data root')
parser.add_argument('--test_data', type=str, default='bio2genenormplus.txt',
                    help='test data')
parser.add_argument('--vocab_path', type = str, default="./data/pubmed_parsed/vocab_gene.pkl",
                    help='the vocab path')
parser.add_argument('--embed_size', type=int,  default=200,
                    help='the initial word embedding size')
parser.add_argument('--entity_type', type=str,  default='gene',
                    help='the current entity type we are trained on')
parser.add_argument('--initial_model', type=str,  default='./model/model.pkl10',
                    help='the current entity type we are trained on')
parser.add_argument('--out_file', type=str,  default='./data/bio2genenormplus_genescore.txt',
                    help='the html that can write')
parser.add_argument('--class_label', type=int,  default=2,
                    help='output label number')
parser.add_argument('--hidden_size', type=int,  default=128,
                    help='hidden_size of rnn')
parser.add_argument('--num_layer', type=int,  default=1,
                    help='output label number')
parser.add_argument('--cell', type=str,  default="lstm",
                    help='lstm or rnn')

# adjust the threshold for evaluation                     
parser.add_argument('--threshold', type=float,  default=0.5,
                    help='adjust the threshold for evaluation')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

sys.setrecursionlimit(20000)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# we can specify the model here later
wordvec = None
if args.word_embedding:
    fp = open(args.word_embedding, "rb")
    wordvec = pickle.load(fp)
    fp.close()

# Load vocabulary wrapper
with open(args.vocab_path, 'rb') as f:
     vocab = pickle.load(f)
     f.close()

vocab_size = len(vocab)
print ("vocab size:{}".format(vocab_size))
args.vocab = vocab


# model
model = EncoderRNN(args.embed_size, args.hidden_size , vocab_size, args.num_layer, args.cell, wordvec, args.class_label)

if args.cuda:
    model.cuda()

# load the pre-trained model
if args.initial_model:
    print ("load pre-trained model")
    model.load_state_dict(torch.load(args.initial_model))
    if args.cuda:
        model = model.cuda()

# test procedure
def GetResult(data, entity_type, vocab):

    model.eval()
    total = 0
    labels = []
    for line in data:

        line = line.strip().split('\t') 
        text_inc = line[0].split()
        len_text = len(text_inc)
        
        mentions = line[1].split() # mention index for drug
        mentions = [int(x) for x in mentions]
        relation = line[-1]

        tokens_inc = []
        tokens_inc.extend([vocab(token) for token in text_inc])
        tokens_inc = torch.LongTensor(tokens_inc)

        if args.cuda:
            tokens_inc = tokens_inc.cuda()
        input_data = Variable(tokens_inc, volatile=True)    
            
        # vectorize the mask
        mask = [0 for k in range(len_text)]
        batch_mask = [0 for k in range(len_text)]

        for k in range(len(mask)):
            if (k in mentions):
                mask[k] = 1
                
        mask = torch.LongTensor(mask)
        batch_mask = torch.LongTensor(batch_mask)

        if args.cuda:
            mask = mask.cuda()
            batch_mask = batch_mask.byte().cuda()

        mask = Variable(mask, volatile=True)

        output = model.forward(input_data[None], batch_mask[None], mask[None])
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        prob = np.exp(output.data.max(1)[0].cpu().numpy()[0])
        label = pred.cpu().numpy()[0]
        
        if label == 0 and 1 - prob >= args.threshold:
            label = 1
                
        if label == 1 and prob < args.threshold:
            label = 0

        total += 1.0        
        
        labels.append(label)

    return labels

def make_text_file(lines, out_file):
    
    fp = open(out_file, "wt+")
    for idx in range(len(lines)):
        fp.write(str(lines[idx])+"\n")
    
    fp.close()

fp = open(os.path.join(args.dataroot, args.test_data), 'r')
data = fp.readlines()
fp.close()

# ignore this first
labels = GetResult(data, args.entity_type, vocab)

# visualization the result using the visualizer
print ("writing the result to text \n")
make_text_file(labels, args.out_file)
