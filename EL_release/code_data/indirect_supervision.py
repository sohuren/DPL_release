import factorgraph as fg
import numpy as np
import utils
import info_parser
import itertools
from string import digits

# each sentence has a indirect_supervisoin instance
# here, we assume we fixed the pairwise potentials since there is no better way to learn it through data
# one way is use grid search to tune the parameters based on accuracy on validation accuracy

class indirect_supervision():

    def __init__(self, instance, opt, entity_type_freq, ctx_freq, acronym_list):
        
        self.instance = instance
        self.opt = opt

        self.g = fg.Graph()

        self.entity_type_freq = entity_type_freq
        self.ctx_freq = ctx_freq
        self.acronym_list = acronym_list
    
        self.initial_potential = {}
        self.base_assignment= {}
        self.current_DL_potential = {}
        self.current_pairwisepotential = {}
        self.N = len(self.instance['pos_neg_example'])
        
    # called one time, might time consuming
    def construct_factor_graph(self):

        # assume we only have one sentence    
        parser = info_parser.Info_Parser(self.instance['tree']['sentences'][0])
        text = self.instance['text']
        mentions = []
        mentions_ctx = []
        
        for idx, item in enumerate(self.instance['pos_neg_example']):
            mentions.append(text[int(item[0]):int(item[1])+1])
            ctx = text[int(item[0])-5:int(item[0])] 
            ctx.extend(text[int(item[1]):int(item[1])+5])
            temp = [self.ctx_freq[x] for x in ctx if x in self.ctx_freq.keys()]
            mentions_ctx.append(sum(temp))

        # the hidden variable    
        for idx, item in enumerate(self.instance['pos_neg_example']):
            self.g.rv('z'+str(idx), 2)
            self.g.factor(['z'+str(idx)], potential=np.array(item[2])) # probability 0, 1, initialized from initial probability


        # the DL factor    
        for idx, item in enumerate(self.instance['pos_neg_example']):
            self.g.rv('DL'+str(idx), 2)
            self.g.factor(['DL'+str(idx)], potential=np.array(item[2])) # probability 0, 1, initialized from initial probability
            self.current_DL_potential['DL'+str(idx)] = np.array(item[2])
            #self.base_assignment['DL'+str(idx)] = 1

        # single factor         
        # the mention's token number
        if self.opt['token_length']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.rv('token_length'+str(idx), 3)
                if len(mentions[idx]) >= 3:
                    self.g.factor(['token_length'+str(idx)], potential=np.array([1, 0, 0], dtype=np.float64)) # encoding the 0, 1
                    self.initial_potential['token_length'+str(idx)] = np.array([1, 0, 0], dtype=np.float64)
                    self.base_assignment['token_length'+str(idx)] = 0
                elif len(mentions[idx]) == 2:
                    self.g.factor(['token_length'+str(idx)], potential=np.array([0, 1, 0], dtype=np.float64)) # encoding the 0, 1
                    self.initial_potential['token_length'+str(idx)] = np.array([0, 1, 0], dtype=np.float64)
                    self.base_assignment['token_length'+str(idx)] = 1
                elif len(mentions[idx]) == 1: # one token
                    self.g.factor(['token_length'+str(idx)], potential=np.array([0, 0, 1], dtype=np.float64))
                    self.initial_potential['token_length'+str(idx)] = np.array([0, 0, 1], dtype=np.float64)
                    self.base_assignment['token_length'+str(idx)] = 2

        # the mentions's string length   
        if self.opt['str_length']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.rv('str_length'+str(idx), 3)
                mention_str = " ".join(mentions[idx])
                if len(mention_str) >= 10:
                    self.g.factor(['str_length'+str(idx)], potential=np.array([1, 0, 0], dtype=np.float64)) # probability 0, 1
                    self.initial_potential['str_length'+str(idx)] = np.array([1, 0, 0], dtype=np.float64)
                    self.base_assignment['str_length'+str(idx)] = 0
                elif len(mention_str) <10 and len(mention_str) >= 5:
                    self.g.factor(['str_length'+str(idx)], potential=np.array([0, 1, 0], dtype=np.float64)) # probability 0, 1
                    self.initial_potential['str_length'+str(idx)] = np.array([0, 1, 0], dtype=np.float64)
                    self.base_assignment['str_length'+str(idx)] = 1
                else:
                    self.g.factor(['str_length'+str(idx)], potential=np.array([0, 0, 1], dtype=np.float64))
                    self.initial_potential['str_length'+str(idx)] = np.array([0, 0, 1], dtype=np.float64)
                    self.base_assignment['str_length'+str(idx)] = 2    

        # the mentions's postag   
        if self.opt['postag']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.rv('postag'+str(idx), 2)                
                labels = 0
                for i in range(int(item[0]), int(item[1])+1):
                    if parser.check_NP(i):
                        labels += 1
                
                ratio = labels/float(item[1]+1-item[0])
                if ratio >= 0.5:
                    self.g.factor(['postag'+str(idx)], potential=np.array([1, 0], dtype=np.float64)) # probability 0, 1
                    self.initial_potential['postag'+str(idx)] = np.array([1, 0], dtype=np.float64)
                    self.base_assignment['postag'+str(idx)] = 0
                else:
                    self.g.factor(['postag'+str(idx)], potential=np.array([0, 1], dtype=np.float64)) # probability 0, 1
                    self.initial_potential['postag'+str(idx)] = np.array([0, 1], dtype=np.float64)
                    self.base_assignment['postag'+str(idx)] = 1


        # the mentions contains special token   
        if self.opt['special_token']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                mention_str = " ".join(mentions[idx]).lower()
                self.g.rv('special_token'+str(idx), 2)
                if mention_str in ['spatial', 'insulin', ';', 'mother', 'brother', 'ghrelin', 'anova', 'lobe', 'star', 'fish', 'leptin', 'hole', 'statin', 'pompe disease']:
                    self.g.factor(['special_token'+str(idx)], potential=np.array([1, 0], dtype=np.float64)) # probability 0, 1
                    self.initial_potential['special_token'+str(idx)] = np.array([1, 0], dtype=np.float64)
                    self.base_assignment['special_token'+str(idx)] = 0
                else:
                    self.g.factor(['special_token'+str(idx)], potential=np.array([0, 1], dtype=np.float64)) # probability 0, 1
                    self.initial_potential['special_token'+str(idx)] = np.array([0, 1], dtype=np.float64)
                    self.base_assignment['special_token'+str(idx)] = 1

        # the mentions's token case   
        if self.opt['case_token']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                mention_str = " ".join(mentions[idx])
                ratio_token = utils.check_uppercase_cnt(mention_str)
                self.g.rv('case_token'+str(idx), 2)
                if ratio_token >= 0.4:
                    self.g.factor(['case_token'+str(idx)], potential=np.array([1, 0], dtype=np.float64)) # probability 0, 1
                    self.initial_potential['case_token'+str(idx)] = np.array([1, 0], dtype=np.float64)
                    self.base_assignment['case_token'+str(idx)] = 0
                else:
                    self.g.factor(['case_token'+str(idx)], potential=np.array([0, 1], dtype=np.float64)) # probability 0, 1
                    self.initial_potential['case_token'+str(idx)] = np.array([0, 1], dtype=np.float64)
                    self.base_assignment['case_token'+str(idx)] = 1

        # the mentions's string case
        if self.opt['case_str']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):                
                mention_str = " ".join(mentions[idx])
                ratio = utils.check_uppercase_str_cnt(mention_str)
                self.g.rv('case_str'+str(idx), 2)
                if ratio >= 0.3:
                    self.g.factor(['case_str'+str(idx)], potential=np.array([1, 0], dtype=np.float64)) # probability 0, 1
                    self.initial_potential['case_str'+str(idx)] = np.array([1, 0], dtype=np.float64)
                    self.base_assignment['case_str'+str(idx)] = 0
                else:
                    self.g.factor(['case_str'+str(idx)], potential=np.array([0, 1], dtype=np.float64)) # probability 0, 1
                    self.initial_potential['case_str'+str(idx)] = np.array([0, 1], dtype=np.float64)
                    self.base_assignment['case_str'+str(idx)] = 1

        # the mention context importance reflected by the entropy
        # get all the context surronding this mentions and calculate their entropy
        # large entropy means less information
           
        if self.opt['mention_ctx_entropy']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.rv('mention_ctx_entropy'+str(idx), 3)

                ratio = mentions_ctx[idx]
                if ratio >= 0.05: # less important since too much entropy
                    self.g.factor(['mention_ctx_entropy'+str(idx)], potential=np.array([0, 0, 1], dtype=np.float64))
                    self.initial_potential['mention_ctx_entropy'+str(idx)] = np.array([0, 0, 1], dtype=np.float64)
                    self.base_assignment['mention_ctx_entropy'+str(idx)] = 2
                elif ratio<0.05 and ratio>=1e-5:
                    self.g.factor(['mention_ctx_entropy'+str(idx)], potential=np.array([0, 1, 0], dtype=np.float64))
                    self.initial_potential['mention_ctx_entropy'+str(idx)] = np.array([0, 1, 0], dtype=np.float64)
                    self.base_assignment['mention_ctx_entropy'+str(idx)] = 1 
                else:
                    self.g.factor(['mention_ctx_entropy'+str(idx)], potential=np.array([1, 0, 0], dtype=np.float64)) # most important
                    self.initial_potential['mention_ctx_entropy'+str(idx)] = np.array([1, 0, 0], dtype=np.float64)
                    self.base_assignment['mention_ctx_entropy'+str(idx)] = 0

        # parser tree information   
        if self.opt['parsetree_label']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.rv('parsetree_label'+str(idx), 2)
                dep_labels = []
                for i in range(item[0], item[1]+1):
                    dep_label = parser.get_dependencies(text[i], text)
                    dep_labels.extend(dep_label) # 1, 0
                if sum(dep_labels)/float(len(dep_labels)) >= 0.4: 
                    self.g.factor(['parsetree_label'+str(idx)], potential=np.array([1, 0], dtype=np.float64))
                    self.initial_potential['parsetree_label'+str(idx)] = np.array([1, 0], dtype=np.float64)
                    self.base_assignment['parsetree_label'+str(idx)] = 0 
                else:
                    self.g.factor(['parsetree_label'+str(idx)], potential=np.array([0, 1], dtype=np.float64)) 
                    self.initial_potential['parsetree_label'+str(idx)] = np.array([0, 1], dtype=np.float64)
                    self.base_assignment['parsetree_label'+str(idx)] = 1

        # parser tree information --    
        if self.opt['parsetree_label_neg']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.rv('parsetree_label_neg'+str(idx), 2)
                dep_labels = []
                for i in range(item[0], item[1]+1):
                    for j in range(len(text)):
                        dep_label = parser.get_dependency(text[i], text[j])
                        dep_labels.append(dep_label)

                # filter the one, if it's root, then it cannot be mentions
                dep_label = []
                for x in dep_labels:
                    if x<len(parser.lookuptable['dep']): 
                        dep_label.append(parser.lookuptable['dep'][x])

                if 'root' not in dep_label: 
                    self.g.factor(['parsetree_label_neg'+str(idx)], potential=np.array([1, 0], dtype=np.float64))
                    self.initial_potential['parsetree_label_neg'+str(idx)] = np.array([1, 0], dtype=np.float64)
                    self.base_assignment['parsetree_label_neg'+str(idx)] = 0 
                else:
                    self.g.factor(['parsetree_label_neg'+str(idx)], potential=np.array([0, 1], dtype=np.float64)) 
                    self.initial_potential['parsetree_label_neg'+str(idx)] = np.array([0, 1], dtype=np.float64)
                    self.base_assignment['parsetree_label_neg'+str(idx)] = 1

        # the mention entropy, basically it's a function of the entity's probability     
        if self.opt['entity_entropy']:
            # for each entity type
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.rv('entity_entropy'+str(idx), 3)
                mention = " ".join(mentions[idx])
                
                if mention in self.entity_type_freq.keys():
                    ratio = self.entity_type_freq[mention]
                else:
                    ratio = 0.05 # assume high entropy since too small probability

                if ratio >= 0.02: # less important since too much entropy
                    self.g.factor(['entity_entropy'+str(idx)], potential=np.array([0, 0, 1], dtype=np.float64))
                    self.initial_potential['entity_entropy'+str(idx)] = np.array([0, 0, 1], dtype=np.float64)
                    self.base_assignment['entity_entropy'+str(idx)] = 2
                elif ratio <  0.02 and ratio >= 1e-6:
                    self.g.factor(['entity_entropy'+str(idx)], potential=np.array([0, 1, 0], dtype=np.float64))
                    self.initial_potential['entity_entropy'+str(idx)] = np.array([0, 1, 0], dtype=np.float64)
                    self.base_assignment['entity_entropy'+str(idx)] = 1 
                else:
                    self.g.factor(['entity_entropy'+str(idx)], potential=np.array([1, 0, 0], dtype=np.float64)) # most important
                    self.initial_potential['entity_entropy'+str(idx)] = np.array([1, 0, 0], dtype=np.float64)
                    self.base_assignment['entity_entropy'+str(idx)] = 0
        
        if self.opt['special_char']:
            
            # for each entity type
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.rv('special_char'+str(idx), 2)
                mention = " ".join(mentions[idx])
                if '-' in mention or utils.hasNumbers(mention): # important since contains special char
                    self.g.factor(['special_char'+str(idx)], potential=np.array([1, 0], dtype=np.float64))
                    self.initial_potential['special_char'+str(idx)] = np.array([1, 0], dtype=np.float64)
                    self.base_assignment['special_char'+str(idx)] = 0 
                else:
                    self.g.factor(['special_char'+str(idx)], potential=np.array([0, 1], dtype=np.float64)) # most important
                    self.initial_potential['special_char'+str(idx)] = np.array([0, 1], dtype=np.float64)
                    self.base_assignment['special_char'+str(idx)] = 1

        # the acronym on the left and right
        temp_acronym_list = [] 
        acronym_list = []
        if self.opt['acronym']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                left = text[item[0]-2:item[0]]
                len_text = item[1]-item[0]+1
                left_ctx = set(text[item[0]-len_text-1:item[0]])
                right = text[item[1]+1:item[1]+2]
                right_ctx = text[item[1]+1:item[1]+len_text+2]
                
                # () XXX
                if '(' in left and ')' in right:
                    self.g.rv('acronym'+str(idx), 2)
                    potential_type = True
                    if len(left_ctx.intersection(self.opt['entity_exclude_keys'][self.opt['entity_type']])) == 0: 
                        self.g.factor(['acronym'+str(idx)], potential=np.array([1, 0], dtype=np.float64)) # it's the type we want
                        self.initial_potential['acronym'+str(idx)] = np.array([1, 0], dtype=np.float64)
                        self.base_assignment['acronym'+str(idx)] = 0 
                    else:
                        self.g.factor(['acronym'+str(idx)], potential=np.array([0, 1], dtype=np.float64))
                        self.initial_potential['acronym'+str(idx)] = np.array([0, 1], dtype=np.float64)
                        self.base_assignment['acronym'+str(idx)] = 1

                        # potentially a positive example, just in initial case change it
                        if item[2][0] < item[2][1]:
                            new_prob = [1 - x for x in item[2]] # inverse the probability
                            self.g.factor(['z'+str(idx)], potential=np.array(new_prob, dtype=np.float64))
                            self.g.factor(['DL'+str(idx)], potential=np.array(new_prob, dtype=np.float64))
                            potential_type = False
                    self.g.factor(['acronym'+str(idx), 'z'+str(idx)], potential=np.array([
                    [0.1, 0.9],
                    [0.9, 0.1],
                    ]))

                    # save current potential to calculate the gradient
                    self.current_pairwisepotential['acronym'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
                    [0.1, 0.9],
                    [0.9, 0.1],
                    ])

                    acronym_list.append(idx)
                    temp_acronym_list.append([mentions[idx], potential_type])

                ## XXX ()
                elif '(' in right and ')' in right_ctx:
                    len_acronym = right_ctx.index(')') - right_ctx.index('(')
                    if abs(len_acronym-len_text) <= 1 and len_acronym > 0:
                        self.g.rv('acronym'+str(idx), 2)
                        potential_type = True
                        if len(set(right_ctx).intersection(self.opt['entity_exclude_keys'][self.opt['entity_type']])) == 0: 
                            self.g.factor(['acronym'+str(idx)], potential=np.array([1, 0], dtype=np.float64)) # it's the type we want
                            self.initial_potential['acronym'+str(idx)] = np.array([1, 0], dtype=np.float64)
                            self.base_assignment['acronym'+str(idx)] = 0
                        else:
                            self.g.factor(['acronym'+str(idx)], potential=np.array([0, 1], dtype=np.float64))
                            self.initial_potential['acronym'+str(idx)] = np.array([0, 1], dtype=np.float64)
                            self.base_assignment['acronym'+str(idx)] = 1

                            # potentially a positive example
                            if item[2][0] < item[2][1]:
                                new_prob = [1 - x for x in item[2]] # inverse the probability
                                self.g.factor(['z'+str(idx)], potential=np.array(new_prob, dtype=np.float64))
                                self.g.factor(['DL'+str(idx)], potential=np.array(new_prob, dtype=np.float64))
                                potential_type = False

                        self.g.factor(['acronym'+str(idx), 'z'+str(idx)], potential=np.array([
                        [0.1, 0.9],
                        [0.9, 0.1],
                        ]))                        

                        # save current potential to calculate the gradient
                        self.current_pairwisepotential['acronym'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
                        [0.1, 0.9],
                        [0.9, 0.1],
                        ])

                        acronym_list.append(idx)
                        temp_acronym_list.append([mentions[idx], potential_type])

        # the acronym   
        if self.opt['cross_sentence']:
            # for each predessor
            for idx, item in enumerate(self.acronym_list):
                self.g.rv('cross_sentence'+str(idx), 2)
                if item[1]:
                    self.g.factor(['cross_sentence'+str(idx)], potential=np.array([1, 0], dtype=np.float64)) # probability 0, 1
                    self.initial_potential['cross_sentence'+str(idx)] = np.array([1, 0], dtype=np.float64)
                    self.base_assignment['cross_sentence'+str(idx)] = 0
                else:
                    self.g.factor(['cross_sentence'+str(idx)], potential=np.array([0, 1], dtype=np.float64))    
                    self.initial_potential['cross_sentence'+str(idx)] = np.array([0, 1], dtype=np.float64)
                    self.base_assignment['cross_sentence'+str(idx)] = 1

        # single factor         
        # the mention's token number
        # the dl factor should be consitent with z
        for idx, item in enumerate(self.instance['pos_neg_example']):
            self.g.factor(['DL'+str(idx), 'z'+str(idx)], potential=np.array([
            [0.7, 0.3],
            [0.3, 0.7],
            ]))

            # save current potential to calculate the gradient
            self.current_pairwisepotential['DL'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
            [0.7, 0.3],
            [0.3, 0.7],
            ])

        if self.opt['token_length']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.factor(['token_length'+str(idx), 'z'+str(idx)], potential=np.array([
                [0.3, 0.7],
                [0.5, 0.5],
                [0.7, 0.3],
                ]))
                
                # save current potential to calculate the gradient
                self.current_pairwisepotential['token_length'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
                [0.3, 0.7],
                [0.5, 0.5],
                [0.7, 0.3],
                ])
    
        if self.opt['str_length']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.factor(['str_length'+str(idx), 'z'+str(idx)], potential=np.array([
                [0.3, 0.7],
                [0.5, 0.5],
                [0.7, 0.3],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['str_length'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
                [0.3, 0.7],
                [0.5, 0.5],
                [0.7, 0.3],
                ])

        if self.opt['postag']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.factor(['postag'+str(idx), 'z'+str(idx)], potential=np.array([
                [0.3, 0.7],
                [0.6, 0.4],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['postag'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
                [0.3, 0.7],
                [0.6, 0.4],
                ])

        if self.opt['case_token']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.factor(['case_token'+str(idx), 'z'+str(idx)], potential=np.array([
                [0.3, 0.7],
                [0.7, 0.3],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['case_token'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
                [0.3, 0.7],
                [0.7, 0.3],
                ])

        if self.opt['special_token']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.factor(['special_token'+str(idx), 'z'+str(idx)], potential=np.array([
                [0.8, 0.2],
                [0.5, 0.5],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['special_token'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
                [0.8, 0.2],
                [0.5, 0.5],
                ])

        if self.opt['case_str']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.factor(['case_str'+str(idx), 'z'+str(idx)], potential=np.array([
                [0.3, 0.7],
                [0.7, 0.3],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['case_str'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
                [0.3, 0.7],
                [0.7, 0.3],
                ])

        if self.opt['mention_ctx_entropy']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.factor(['mention_ctx_entropy'+str(idx), 'z'+str(idx)], potential=np.array([
                [0.2, 0.8],
                [0.4, 0.6],
                [0.6, 0.4],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['mention_ctx_entropy'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
                [0.2, 0.8],
                [0.4, 0.6],
                [0.6, 0.4],
                ])

        if self.opt['parsetree_label']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.factor(['parsetree_label'+str(idx), 'z'+str(idx)], potential=np.array([
                [0.3, 0.7],
                [0.7, 0.3],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['parsetree_label'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
                [0.3, 0.7],
                [0.7, 0.3],
                ])

        if self.opt['parsetree_label_neg']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.factor(['parsetree_label_neg'+str(idx), 'z'+str(idx)], potential=np.array([
                [0.3, 0.7],
                [0.7, 0.3],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['parsetree_label_neg'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
                [0.3, 0.7],
                [0.7, 0.3],
                ])

        if self.opt['entity_entropy']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.factor(['entity_entropy'+str(idx), 'z'+str(idx)], potential=np.array([
                [0.2, 0.8],
                [0.4, 0.6],
                [0.6, 0.4],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['entity_entropy'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
                [0.2, 0.8],
                [0.4, 0.6],
                [0.6, 0.4],
                ])

        if self.opt['special_char']:
            # for each mentions
            for idx, item in enumerate(self.instance['pos_neg_example']):
                self.g.factor(['special_char'+str(idx), 'z'+str(idx)], potential=np.array([
                [0.3, 0.7],
                [0.7, 0.3],
                ]))
                
                # save current potential to calculate the gradient
                self.current_pairwisepotential['special_char'+str(idx) + '&&' + 'z'+str(idx)] = np.array([
                [0.3, 0.7],
                [0.7, 0.3],
                ])

        # now specify the pairwise potential function -- might add more later, need to be careful
        if self.opt['same_mention']:
            for idx1 in acronym_list:
                for idx2 in range(len(self.instance['pos_neg_example'])):
                    if mentions[idx1] == mentions[idx2] and idx1 != idx2:
                        item = self.instance['pos_neg_example'][idx2][2]
                        if item[0] < item[1]:
                            new_prob = [1 - x for x in item] # inverse the probability
                            self.g.factor(['z'+str(idx2)], potential=np.array(new_prob, dtype=np.float64))
                            self.g.factor(['DL'+str(idx)], potential=np.array(new_prob, dtype=np.float64))

                        self.g.factor(['z'+str(idx1), 'z'+str(idx2)], potential=np.array([
                        [0.9, 0.1],
                        [0.1, 0.9],
                        ]))

                        # save current potential to calculate the gradient
                        self.current_pairwisepotential['z'+str(idx1) + '&&' + 'z'+str(idx2)] = np.array([
                        [0.9, 0.1],
                        [0.1, 0.9],
                        ])

        if self.opt['logical_connective']:
            for idx1, item1 in enumerate(self.instance['pos_neg_example']):
                for idx2 in range(idx1+1, len(self.instance['pos_neg_example'])):
                    item2 = self.instance['pos_neg_example'][idx2]    
                    span = text[item1[1]+1:item2[0]]
                    if len(span) <= 10 and ('and' in span or 'or' in span) and '.' not in span:
                        self.g.factor(['z'+str(idx1), 'z'+str(idx2)], potential=np.array([
                        [0.6, 0.4],
                        [0.4, 0.6],
                        ]))

                        # save current potential to calculate the gradient
                        self.current_pairwisepotential['z'+str(idx1) + '&&' + 'z'+str(idx2)] = np.array([
                        [0.6, 0.4],
                        [0.4, 0.6],
                        ])
                
        if self.opt['acronym_pairwise']:
            
            # for each mentions
            for idx1, item1 in enumerate(self.instance['pos_neg_example']):
                acronym = ''
                # find it's acronym
                pos = len(self.instance['pos_neg_example'])
                for idx2 in range(idx1+1, len(self.instance['pos_neg_example'])):
                    mention_2 = mentions[idx2]
                    item2 = self.instance['pos_neg_example'][idx2]    
                    span = text[item1[1]+1:item2[0]]
                    right_ctx = text[item2[1]+1:item2[1]+2] # max noise is 1
                    if '(' in span and ')' in right_ctx and len(span) <= 2:
                        self.g.factor(['z'+str(idx1), 'z'+str(idx2)], potential=np.array([
                        [0.9, 0.1],
                        [0.1, 0.9],
                        ]))
                        
                        # save current potential to calculate the gradient
                        self.current_pairwisepotential['z'+str(idx1) + '&&' + 'z'+str(idx2)] = np.array([
                        [0.9, 0.1],
                        [0.1, 0.9],
                        ])

                        acronym = mention_2
                        pos = idx2
                        temp_acronym_list.append([mentions[pos], True]) # potential it's true
                        break
                
                # check the following acronym, might not need it     
                for idx3 in range(pos+1, len(self.instance['pos_neg_example'])):
                    mention_3 = mentions[idx3]
                    if mention_3 == acronym and acronym:
                        self.g.factor(['z'+str(idx1), 'z'+str(idx3)], potential=np.array([
                        [0.9, 0.1],
                        [0.1, 0.9],
                        ]))        

                        # save current potential to calculate the gradient
                        self.current_pairwisepotential['z'+str(idx1) + '&&' + 'z'+str(idx3)] = np.array([
                        [0.9, 0.1],
                        [0.1, 0.9],
                        ])

        # modify it's initial value since the signal it's too strong
        if self.opt['cross_sentence']:
            for item in self.acronym_list:
                for idx2 in range(len(self.instance['pos_neg_example'])):
                    if item[0] == mentions[idx2]:
                        if item[1] == False:
                            item2 = self.instance['pos_neg_example'][idx2][2]
                            if item2[0] < item2[1]:
                                new_prob = [1 - x for x in item2] # inverse the probability
                                self.g.factor(['z'+str(idx2)], potential=np.array(new_prob, dtype=np.float64))
                                self.g.factor(['DL'+str(idx)], potential=np.array(new_prob, dtype=np.float64))

        # the acronym factor 
        if self.opt['cross_sentence']:
            # for each predessor
            for idx1, item in enumerate(self.acronym_list):
                for idx2 in range(len(self.instance['pos_neg_example'])):
                    if item[0] == mentions[idx2]:
                        
                        # might need to modify it - reverse
                        self.g.factor(['cross_sentence'+str(idx1), 'z'+str(idx2)], potential=np.array([
                        [0.9, 0.1],
                        [0.1, 0.9],
                        ]))

                        # save current potential to calculate the gradient
                        # might need to modify it
                        self.current_pairwisepotential['cross_sentence'+str(idx1) + '&&' + 'z'+str(idx2)] = np.array([
                        [0.9, 0.1],
                        [0.1, 0.9],
                        ])        


        # delete all useless instance to save the memory
        del self.instance
        del self.opt
        del self.entity_type_freq
        del self.ctx_freq
        del self.acronym_list
    
        return temp_acronym_list


    def update_factor_graph_unary(self, key, potentials):

        self.g.factor([key], potential=np.array(potentials)) # uniformly probability
        # keep the current potentials
        if 'DL' in key:
            self.current_DL_potential[key] = np.array(potentials)

    def get_factor_graph_unary(self, key):
    
        # keep the current potentials
        assert ("DL" in key)
        return self.current_DL_potential[key]
                
    # reset all the prior to the initial state, needed in each iteration no matter learn the weight or not
    def reset_factor_graph_prior(self):
        
        for key in self.initial_potential.keys():
            self.g.factor([key], potential = self.initial_potential[key]) # initial probability

    # get the factor graph pairwise w.r.t specific key1 and key2
    def get_factor_graph_pairwise(self, key1, key2):
        
        return self.current_pairwisepotential[key1 + '&&' + key2]

    def update_factor_graph_pairwise(self, key1, key2, potentials):

        # now specify the pairwise function
        self.g.factor([key1, key2], potential=np.array(potentials))
        # also update in our dict so we can get it later
        self.current_pairwisepotential[key1 + '&&' + key2] = np.array(potentials)

    # input is the parameters w.r.t to current group.    
    def update_all_factor_graph_pairwise(self, parameters):
        # now specify the pairwise function
        for pairwise_keys in self.current_pairwisepotential.keys():
            key1 = pairwise_keys.split('&&')[0]
            key2 = pairwise_keys.split('&&')[1]
            para_key = key1.translate(None, digits) + "&&" + key2.translate(None, digits)    
            self.g.factor([key1, key2], potential=np.array(parameters[para_key]))
            # also update in our dict so we can get it later
            self.current_pairwisepotential[pairwise_keys] = np.array(parameters[para_key])

    # evaluate all the possible assignment, p(y|x)
    def set_label(self, x):
        
        # get the dl assignment and extend it to DL factor
        for idx in range(len(x)):
            self.base_assignment['DL'+str(idx)] = int(np.argmax(self.current_DL_potential['DL'+str(idx)]))
    
        self.label = self.base_assignment
        for idx in range(len(x)):
            self.label['z'+str(idx)] = int(x[idx])
      
    # evaluate all the possible assignment, p(y|x)
    def compute_expectation(self):
        
        all_assignments = list(itertools.product([0, 1], repeat = self.N)) # all the possible solution
    
        # get the full assignment for the current factor
        assignments = []
        for x in all_assignments:
            x = list(x)
            assign = self.base_assignment
            for idx in range(len(x)):
                assign['z'+str(idx)] = x[idx]
            assignments.append(assign)

        Z = []            
        # construct the possible assignment
        for x in assignments:
            poten = np.exp(self.g.joint(x))
            Z.append(poten)
        
        sum_Z = sum(Z)
        prob = [x/sum_Z for x in Z]

        # return the best assignment for z....z, all assignment and its corrspoding probability
        
        return assignments, prob    

    # get all current parameters
    def get_current_parameters(self):
        
        current_parameters = {}
        for pairwise_keys in self.current_pairwisepotential.keys():    
            key1 = pairwise_keys.split('&&')[0]
            key2 = pairwise_keys.split('&&')[1]
            para_key = key1.translate(None, digits) + "&&" + key2.translate(None, digits)    
            # also update in our dict so we can get it later
            current_parameters[para_key] = self.current_pairwisepotential[pairwise_keys]
    
        return current_parameters

    # count all current parameters, ignore the idx, done    
    def count(self, assignment, prob):

        cnt_pairwise = {}
        for pairwise_keys in self.current_pairwisepotential.keys():
            # remove the idx w.r.t to the mention id sicne they shared
            key1 = pairwise_keys.split('&&')[0]
            key2 = pairwise_keys.split('&&')[1]
            para_key = key1.translate(None, digits) + "&&" + key2.translate(None, digits)
            cnt_pairwise[para_key] = np.zeros_like(self.current_pairwisepotential[pairwise_keys])

        for pairwise_key in self.current_pairwisepotential.keys():
            pairwise_key = pairwise_key.split("&&")
            key1 = pairwise_key[0]
            key2 = pairwise_key[1]
            v1 = assignment[key1]
            key1 = key1.translate(None, digits)  
            v2 = assignment[key2]
            key2 = key2.translate(None, digits)
            para_key = key1 + "&&" + key2 
            cnt_pairwise[para_key][v1][v2] += prob 
    
        return cnt_pairwise

    # compute the gradient w.r.t to single instance, done
    def compute_gradient(self):
        
        all_assignments, prob = self.compute_expectation()

        # ground truth label    
        total_grad = self.count(self.label , 1) 
        # all others with label
        grads = []
        for idx, assign in enumerate(all_assignments):
            grad = self.count(assign, prob[idx])
            grads.append(grad)     # grad w.r.t current asignment 

        for pairwise_key in total_grad.keys():
            expect = np.zeros_like(total_grad[pairwise_key])
            for idx in range(len(grads)):
                expect += grads[idx][pairwise_key]
            total_grad[pairwise_key] -= expect

        return total_grad

    def message_passing(self):
        
        # Run (loopy) belief propagation (LBP)
        iters, converged = self.g.lbp(normalize=True)
        print ('LBP ran for %d iterations. Converged = %r' % (iters, converged))

    def get_marginals(self, key):

        tuples = self.g.get_rv_marginals()    
        for rv, marg in tuples:
            if str(rv) == key:
                vals = range(rv.n_opts)
                if len(rv.labels) > 0:
                    vals = rv.labels
                marginal = np.zeros((1, len(vals)), dtype = np.float32)
                for i in range(len(vals)):
                    marginal[0][vals[i]] = marg[i]
        
        return marginal
