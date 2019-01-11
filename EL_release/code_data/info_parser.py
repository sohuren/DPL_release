import random
import torch.utils.data
import torchvision.transforms as transforms
from base_data_loader import BaseDataLoader
from pdb import set_trace as st
from builtins import object
import math
import pickle


'''
given parsed result from stanford corenlp as dict
this class will return the corresponding value  
'''
class Info_Parser():

    # calculate the feature for each token
    # feature depends on the logical operator's position
    def __init__(self, tree):

        self.tree = tree 
        self.lookuptable = {}

        self.lookuptable['pos'] = {}
        self.lookuptable['dep'] = {}
        self.lookuptable['ner'] = {}
        
        # define the lookup table
        self.lookuptable['pos'] = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP',
        'PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','.',',',':']
        
        self.lookuptable['dep'] = ['acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'aux', 'auxpass', 'cc', 'ccomp', 'conj', 'cop', 'csubj',
        'csubjpass', 'dep', 'det', 'discourse', 'dobj', 'expl', 'goeswith', 'iobj', 'mark', 'mwe', 'neg', 'nn', 'npadvmod', 'nsubj', 'nsubjpass',
        'num', 'number', 'parataxis', 'pcomp', 'pobj', 'poss', 'possessive', 'preconj', 'predet', 'prep', 'prepc', 'prt', 'punct', 'quantmod',
         'rcmod', 'ref', 'root','tmod','vmod','xcomp','xsubj','case','', 'nmod', 'acl']
        
        self.lookuptable['ner'] = ['LOCATION', 'ORGANIZATION', 'MISC', 'DATE', 'MONEY', 'PERSON', 'PERCENT', 'TIME','O', 'NUMBER','SET','DURATION','ORDINAL','SET']

    # calculate the feature for each token
    # feature depends on the logical operator's position
    def get_pos(self, index):

        # assume that the extraction is based on one sentence     
        tree = self.tree
        pos = ""

        if index == 0:
            pos = tree['words'][0][1]['PartOfSpeech']
        else:
            data = tree['parsetree'].encode('ascii', 'ignore').replace('[','')
            data = data.replace(']','').split()
            pos = []
            for x in data:
                if 'PartOfSpeech' in x:
                    pos.append(x)
            if index-1 < len(pos):        
                info = pos[index-1]
                pos = info.split("=")[1]
            else:
                pos = '.'

        if pos in self.lookuptable['pos']:
            return self.lookuptable['pos'].index(pos)
        else:
            return len(self.lookuptable['pos']) + 1


    # calculate the feature for each token
    # feature depends on the logical operator's position
    def check_NP(self, index):

        # assume that the extraction is based on one sentence     
        idx = self.get_pos(index)
        if idx < len(self.lookuptable['pos']):
           if self.lookuptable['pos'][idx] in ['NN','NNS','NNP','NNPS']:
               return True
        
        return False       

    # calculate the tree strucuture for each token
    def get_tree(self):

        # assume that the extraction is based on one sentence     
        deptree = json.loads(self.tree['dependencies'].encode('ascii', 'ignore'))
        text = self.tree['text']

        # saved for the result
        parents = []
        rels = []

        for i in range(len(deptree)):
            
            label = deptree[i][0]
            parent = deptree[i][1]
            child = deptree[i][2]
            
            rels.append(label)
            index = [k for k, m in enumerate(text) if m == parent]
            # choose the one that cloest to the current token
            index_gap = 10000
            for item in index:
                if abs(item-i) < index_gap:
                    index_gap = abs(item-i)
                    true_index = item

            # save the true parent          
            parents.append(true_index)
        
        return parents, rels

    # calculate the feature for each token
    # feature depends on the logical operator's position
    def get_ner(self, index):

        # assume that the extraction is based on one sentence     
        tree = self.tree
        ner = ""
        if index == 0:
            ner = tree['words'][0][1]['NamedEntityTag']
        else:
            data = tree['parsetree'].encode('ascii', 'ignore').replace('[','')
            data = data.replace(']','').split()
            ner = []
            for x in data:
                if 'NamedEntityTag' in x and 'NormalizedNamedEntityTag' not in x:
                    ner.append(x)
            if index - 1 < len(ner):        
                info = ner[index-1]
                ner = info.split("=")[1]
            else:
                ner = 'O'
                                
        if ner in self.lookuptable['ner']:
            return self.lookuptable['ner'].index(ner)
        else:
            return len(self.lookuptable['ner']) + 1

    # calculate the feature for each token
    # feature depends on the logical operator's position
    def get_dependency(self, w1, w2):

        # assume that the extraction is based on one sentence     
        tree = self.tree
        depen = ""
        
        for item in tree['dependencies']:
            if item[1] == w1 and item[2] == w2:
                depen = item[0]
            if item[2] == w2 and item[1] == w2:
                depen = item[0]
        
        if depen.split(":")[0] in self.lookuptable['dep']:
            return self.lookuptable['dep'].index(depen.split(":")[0])
        else:
            return len(self.lookuptable['dep']) + 1   


    # calculate the feature for each token
    # feature depends on the logical operator's position
    def get_dependencies(self, w1, text):
        # assume that the extraction is based on one sentence     
        labels = []
        for w2 in text:
            idx = self.get_dependency(w1, w2)
            if idx < len(self.lookuptable['dep']):
                label = self.lookuptable['dep'][idx]
                if label in ['advcl', 'advmod', 'amod', 'dobj', 'iobj', 'nsubj', 'nsubjpass', 'num', 'number', 'quantmod', 'rcmod','tmod','vmod','xcomp','xsubj', 'nmod', 'acl']:
                    labels.append(1)
                else:
                    labels.append(0)        

        return labels
