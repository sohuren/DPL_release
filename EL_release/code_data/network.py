import torch
import torch.nn as nn
from torch.autograd import Variable
from pdb import set_trace as st
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as models
import torch.nn.functional as F
import math
_INF = float('inf')

# intialize the weight
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or  classname.find('InstanceNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.01, 0.02)
        m.bias.data.fill_(0) 


class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, inputs, context):
        """
        inputs: batch x dim
        context: batch x sourceL x dim
        """

        targetT = self.linear_in(inputs).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -_INF) # 1: -inf, 0: keep the same 
        attn = self.sm(attn)

        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, inputs), 1) # concatenate the inputs and the weighted context
        contextOutput = self.tanh(self.linear_out(contextCombined)) # linear outputs of the model
        return contextOutput, attn # attn: softmaxed attention weight


# Defines the LSTM Decoder
class EncoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, cell, wordvec, class_label):
        """Set the hyper-parameters and build the layers."""
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
    
        if cell == 'lstm':
           self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        else:
           self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # since this bi-directional network
        self.wordvec = wordvec
        self.init_weights()
        #self.att = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.linear = nn.Linear(self.hidden_size*2, class_label)
        self.attention = GlobalAttention(self.hidden_size*2)

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        # initialize with glove word embedding
        self.embed.weight.data.copy_(torch.from_numpy(self.wordvec))

    # the forward method, which compute the hidden state vector
    def forward(self, text, batch_mask, mask):
        """run the lstm to decode the text"""
        embeddings = self.embed(text)
        hiddens, _ = self.rnn(embeddings) # b*l*f_size
        batch = embeddings.size()[0]
        # only get the last hidden states
        attvec = hiddens 
        
        # the h_{t} at the mentions       
        mask = torch.unsqueeze(mask, 2)
        mask = mask.expand(attvec.size())
        mask = mask.float()
        ht = torch.mean(mask*attvec, 1) # get the mean vector of the mentions
        
        # calculate the attention
        self.attention.applyMask(batch_mask)
        output, att = self.attention(ht.view(batch, -1), hiddens) # batch * dim
        
        response = self.linear(output.view(batch, -1)) 
        
        return F.log_softmax(response)


