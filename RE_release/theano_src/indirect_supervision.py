import factorgraph as fg
import numpy as np
import utils
import info_parser
import itertools
from string import digits
import re
import KB
# each sentence has a indirect_supervision instance
# here, we assume we fixed the pairwise potentials since there is no better way to learn it through data
# one way is use grid search to tune the parameters based on accuracy on validation accuracy

class indirect_supervision():

    def __init__(self, instance, opt, entity_type_freq, ctx_freq, arc_type_dict, acronym_list):
        
        self.instance = instance
        self.opt = opt

        self.g = fg.Graph()

        self.entity_type_freq = entity_type_freq
        self.ctx_freq = ctx_freq
        self.acronym_list = acronym_list
        self.arc_type_dict = arc_type_dict

        self.initial_potential = {}
        self.base_assignment= {}
        self.current_pairwisepotential = {}
        self.current_DL_potential = {}

    # called one time, might time consuming
    def construct_factor_graph(self):

        # core data to contrsuct the data, add new gene probability score
        x, y, mentions, dependencies, gene_prob = self.instance

        span_min = 1000
        span_max = 0
        for i in range(len(mentions)):
            for j in mentions[i]:
                if span_min >= int(j):    
                    span_min = int(j)
                if span_max <= int(j):
                    span_max = int(j)

        # no matter whether what's the relation, then mentions are always ok        
        # the hidden variable

        if y == 'None' or y == "1 0":
            initial_potential = [0.8, 0.2] # modify here make small difference in neg examples, [1,0]
        else:
            initial_potential = [0.2, 0.8] # [0.2, 0.8]

        # we only have one target factor each sentence
        idx = 0
        self.g.rv('z'+str(idx), 2)
        self.g.factor(['z'+str(idx)], potential=np.asarray(initial_potential, dtype=np.float32)) # probability 0, 1, initialized from initial probability
        self.g.rv('DL'+str(idx), 2)
        self.g.factor(['DL'+str(idx)], potential=np.asarray(initial_potential, dtype=np.float32))
        self.current_DL_potential['DL'+str(idx)] = np.asarray(initial_potential, dtype=np.float32)

        temp_acronym_list = []
        # the minimal factor we need to debug, this is also the hard em or soft em begins
        self.g.factor(['z'+str(idx), 'DL'+str(idx)], potential=np.array([
                        [0.8, 0.2],
                        [0.2, 0.8],
                        ]))
        self.current_pairwisepotential['z'+str(idx) + "&&" + 'DL'+str(idx)] = np.array([
                        [0.8, 0.2],
                        [0.2, 0.8],
                        ])


        # add MIL factor
        self.g.rv('MIL'+str(idx), 1)
        self.g.factor(['MIL'+str(idx)], potential=np.asarray([1], dtype=np.float32))
        self.initial_potential['MIL'+str(idx)] = np.asarray([1], dtype=np.float32)
        self.base_assignment['MIL'+str(idx)] = 0
        self.g.factor(['MIL'+str(idx), 'z'+str(idx)], potential=np.array([
                        [0.5, 0.5],
                        ]))
        self.current_pairwisepotential['MIL'+str(idx) + "&&" + 'z'+str(idx) ] = np.array([
                        [0.5, 0.5],
                        ])

        # new factor
        self.g.rv('z_overlap', 2)
        if  len(set(mentions[0]).intersection(set(mentions[1]))) >=1 or len(set(mentions[0]).intersection(set(mentions[2]))) >=1 or len(set(mentions[1]).intersection(set(mentions[2]))) >= 1:
            self.g.factor(['z_overlap'], potential=np.asarray([1, 0], dtype=np.float32)) # probability 0, 1, initialized from initial probability          
            self.initial_potential['z_overlap'] = np.array([1, 0], dtype=np.float64)
            self.base_assignment['z_overlap'] = 0
#        elif abs(mentions[0][-1]-mentions[1][0]) <=1 or abs(mentions[0][-1] - mentions[2][0]) <=1:
#            self.g.factor(['z_overlap'], potential=np.asarray([1, 0], dtype=np.float32)) # probability 0, 1, initialized from initial probability          
#            self.initial_potential['z_overlap'] = np.array([1, 0], dtype=np.float64)
#            self.base_assignment['z_overlap'] = 0
        else:
            self.g.factor(['z_overlap'], potential=np.asarray([0, 1], dtype=np.float32)) # probability 0, 1, initialized from initial probability          
            self.initial_potential['z_overlap'] = np.array([0, 1], dtype=np.float64)
            self.base_assignment['z_overlap'] = 1
            
        self.g.factor(['z_overlap', 'z0'], potential=np.array([
                        [1, 0],
                        [0.5, 0.5],
                        ]))
        self.current_pairwisepotential['z_overlap' + "&&" + 'z0'] = np.array([
                        [1, 0],
                        [0.5, 0.5],
                        ])

        # check drug's mention
        if self.opt['drug_sentivity']:
            # for each mentions
            self.g.rv('drug_sentivity', 2)
            mention_str = " ".join([x[i] for i in mentions[0]]).lower() 
            if ('cofactor' in mention_str or 'histidine' in mention_str or ('-sensitive' in mention_str and len(mentions[0])==1)
                or ('-resistant' in mention_str and len(mentions[0])==1) or ('-insensitive' in mention_str and len(mentions[0])==1)
                or ('-resistance' in mention_str and len(mentions[0])==1) or 'alcohol' in mention_str or 'g/ml' in mention_str or 'ethanol' in mention_str):
                self.g.factor(['drug_sentivity'], potential=np.array([0, 1], dtype=np.float64))
                self.initial_potential['drug_sentivity'] = np.array([0, 1], dtype=np.float64)
                self.base_assignment['drug_sentivity'] = 1    
            else:
                self.g.factor(['drug_sentivity'], potential=np.array([1, 0], dtype=np.float64)) # probability 0, 1
                self.initial_potential['drug_sentivity'] = np.array([1, 0], dtype=np.float64)
                self.base_assignment['drug_sentivity'] = 0

        # check drug's mention
        if self.opt['triple_pair']:
            # for each mentions
            self.g.rv('triple_pair', 2)
            mention_drug = " ".join([x[i] for i in mentions[0]]).lower()
            mention_gene = " ".join([x[i] for i in mentions[1]]).lower()
            mention_var = " ".join([x[i] for i in mentions[2]]).lower()
            # each variant only apprears in one gene
            extact_match = False
            for kb in KB.KB:
                if mention_drug == kb[0] and mention_gene == kb[1] and mention_var == kb[2]:
                    extact_match = True
                    break

            #print pair_gene, mention_gene    
            if not extact_match:
                self.g.factor(['triple_pair'], potential=np.array([0, 1], dtype=np.float64))
                self.initial_potential['triple_pair'] = np.array([0, 1], dtype=np.float64)
                self.base_assignment['triple_pair'] = 1    
            else:
                self.g.factor(['triple_pair'], potential=np.array([1, 0], dtype=np.float64)) # probability 0, 1
                self.initial_potential['triple_pair'] = np.array([1, 0], dtype=np.float64)
                self.base_assignment['triple_pair'] = 0

        # check drug's mention
        if self.opt['gene']:
            # for each mentions
            self.g.rv('gene', 2)
            mention_str = " ".join([x[i] for i in mentions[1]]).lower() 
            if ('insulin' == mention_str or '>a' in mention_str or 'a>' in mention_str or "not" == mention_str or "was" == mention_str or "san" == mention_str or "for" == mention_str
                or "all" == mention_str or "fig" in mention_str or "has" == mention_str or "end-point" == mention_str or "red" == mention_str or "had" == mention_str
                or "can" == mention_str or "far" == mention_str or "best" == mention_str or "fact" == mention_str or "proteins" == mention_str or "cell" == mention_str
                or "mice" == mention_str or "bar" == mention_str or "wild-type" == mention_str or "April" == mention_str or "six" == mention_str or "is a" == mention_str
                or "men" == mention_str or "them" == mention_str or "as a" == mention_str or "she" == mention_str or "in a" == mention_str or "arms" == mention_str
                or "med" == mention_str or "large" == mention_str or "clinicaltrials.gov" == mention_str or "age" == mention_str or "pan-cancer" == mention_str or "mark" == mention_str):
                self.g.factor(['gene'], potential=np.array([0, 1], dtype=np.float64))
                self.initial_potential['gene'] = np.array([0, 1], dtype=np.float64)
                self.base_assignment['gene'] = 1    
            else:
                self.g.factor(['gene'], potential=np.array([1, 0], dtype=np.float64)) # probability 0, 1
                self.initial_potential['gene'] = np.array([1, 0], dtype=np.float64)
                self.base_assignment['gene'] = 0

        if self.opt['mention_fig']:
            self.g.rv('mention_fig', 2)
            mention_wrong = False             
            for i in range(len(x)):  
              if x[i] in ['Fig', 'Figure', 'Figures', 'Figs', 'Fig.', 'Figure.']:
                if i+1 in mentions[0] or i+1 in mentions[1] or i+1 in mentions[2] or i in mentions[0] or i in mentions[1] or i in mentions[2]:
                  mention_wrong = True
                  break
            if mention_wrong:
                self.g.factor(['mention_fig'], potential=np.array([1, 0], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['mention_fig'] = np.array([1, 0], dtype=np.float64)
                self.base_assignment['mention_fig'] = 0
            else:
                self.g.factor(['mention_fig'], potential=np.array([0, 1], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['mention_fig'] = np.array([0, 1], dtype=np.float64)
                self.base_assignment['mention_fig'] = 1


        if self.opt['gene_mutation']:

            self.g.rv('gene_mutation', 2)
            gene_mention_str = [x[i] for i in mentions[1]]
            var_mention_str = [x[i] for i in mentions[2]]
            gene_index = []
            gene_index.extend(mentions[1])
            var_index = []
            var_index.extend(mentions[2])

            for i in range(len(x)):
              if x[i:i+len(gene_mention_str)] == gene_mention_str:
                gene_index.extend(range(i,i+len(gene_mention_str)))

              if x[i:i+len(var_mention_str)] == var_mention_str:
                var_index.extend(range(i,i+len(var_mention_str)))

            min_dis = 1000000
            min_i = 0
            min_j = 0
            for i in gene_index:
                for j in var_index:
                    if abs(i-j) < min_dis:
                        min_dis = abs(i-j)
                        min_i = i
                        min_j = j

            interval = ""
            if min_i <= min_j:
                interval = x[min_i+1:min_j]
            if min_j < min_i:
                interval = x[min_j+1:min_i]

            other_gene = False
            #print interval
            gene_keys = []
            for key in self.entity_type_freq.keys():
                if len(key.split()) > 1:
                    gene_keys.append(key.lower())
                elif len(key) >=4:
                    gene_keys.append(key.lower())
            #print gene_keys
            for item in interval:
                if item in gene_keys:
                    other_gene = True

            if other_gene:
                print "find other genes" 
                self.g.factor(['gene_mutation'], potential=np.array([1, 0], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['gene_mutation'] = np.array([1, 0], dtype=np.float64)
                self.base_assignment['gene_mutation'] = 0
            else:
                self.g.factor(['gene_mutation'], potential=np.array([0, 1], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['gene_mutation'] = np.array([0, 1], dtype=np.float64)
                self.base_assignment['gene_mutation'] = 1


        # the mentions's string length   
        # change the mentions now to less change the code
        new_mentions = [mentions[1], mentions[2]]
        mentions = new_mentions

        #print (x) # list
        #print (y) # string
        #print (idx) # list [[], []]
        #print (dependencies) # 

        # no matter whether what's the relation, then mentions are always ok
        initial_mention_potential = [1, 0]
        for idx, item in enumerate(mentions):
          self.g.rv('z_mention'+str(idx), 2)
          self.g.factor(['z_mention'+str(idx)], potential=np.asarray(initial_mention_potential, dtype=np.float32)) # probability 0, 1, initialized from initial probability          
          self.initial_potential['z_mention'+str(idx)] = np.array(initial_mention_potential, dtype=np.float64)
          self.base_assignment['z_mention'+str(idx)] = 0

          self.g.factor(['z_mention'+str(idx), 'z0'], potential=np.array([
                        [0.5, 0.5],
                        [0.7, 0.3],
                        ]))
          self.current_pairwisepotential['z_mention'+str(idx) + "&&" + 'z0'] = np.array([
                        [0.5, 0.5],
                        [0.7, 0.3],
                        ])

        # consider the context between the mentions to decide whether they have the relation
        if self.opt["z_re"]:
            special_token_exist = ['leading', 'significant', 'lead', 'result', 'resulting', 'results', 'because', 'influence', 'resistance', 'confer', 
            'demonstrate', 'inhibiting', 'sensitizing', 'inhibit', 'receptor', 'regulate', 'regulation', 'negative', 'sensitive', 'indicate','sensitivity','regulatory', 
            'differ', 'identified', 'identify', 'activate', 'pathways', 'demonstrated', 'response', 'sensitivity', 'predictive',]
            special_token_no_exist = ['no', 'No', 'not', 'none', 'None' 'figure', 'appendix', 'does not', 'low', 'lower']
            cnt_exist = 0
            cnt_no_exist = 0
            for idx, word in enumerate(x):
                if word.lower() in special_token_exist:
                    cnt_exist += 1
                if word in special_token_no_exist:
                    if span_min <= idx and span_max >= idx: 
                        cnt_no_exist += 1    

            idx = 0
            self.g.rv('z_re'+str(idx), 3)
            if cnt_exist >= 2 and cnt_no_exist <= 1:
                self.g.factor(['z_re'+str(idx)], potential=np.asarray([1, 0, 0], dtype=np.float32)) # probability 0, 1, initialized from initial probability          
                self.initial_potential['z_re'+str(idx)] = np.array([1, 0, 0], dtype=np.float64)
                self.base_assignment['z_re'+str(idx)] = 0
            elif cnt_no_exist >= 2:
                self.g.factor(['z_re'+str(idx)], potential=np.asarray([0, 0, 1], dtype=np.float32)) # probability 0, 1, initialized from initial probability          
                self.initial_potential['z_re'+str(idx)] = np.array([0, 0, 1], dtype=np.float64)
                self.base_assignment['z_re'+str(idx)] = 2
            else:        
                self.g.factor(['z_re'+str(idx)], potential=np.asarray([0, 1, 0], dtype=np.float32)) # probability 0, 1, initialized from initial probability          
                self.initial_potential['z_re'+str(idx)] = np.array([0, 1, 0], dtype=np.float64)
                self.base_assignment['z_re'+str(idx)] = 1

            self.g.factor(['z_re'+str(idx), 'z0'], potential=np.array([
                        [0.2, 0.8],
                        [0.5, 0.5],
                        [0.8, 0.2],
                        ]))
            self.current_pairwisepotential['z_re'+str(idx) + "&&" + 'z0'] = np.array([
                        [0.2, 0.8],
                        [0.5, 0.5],
                        [0.8, 0.2],
                        ])


        if self.opt['table']:
            self.g.rv('table', 2)
            ratio = 0
            long_seq = 0
            for item in x:
                if item.replace('.','').isdigit():
                    ratio += 1
                    long_seq += 1
                else:
                    long_seq = 0

            ratio /= float(len(x))
            print ratio
            print long_seq             
            if ratio >= 0.25 and long_seq >= 2:
                self.g.factor(['table'], potential=np.array([1, 0], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['table'] = np.array([1, 0], dtype=np.float64)
                self.base_assignment['table'] = 0
            else:
                self.g.factor(['table'], potential=np.array([0, 1], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['table'] = np.array([0, 1], dtype=np.float64)
                self.base_assignment['table'] = 1


        if self.opt['entity_gene_score']:
            self.g.rv('entity_gene_score', 2)
            # threshold == 0.65
            if gene_prob <= 0.55:
                self.g.factor(['entity_gene_score'], potential=np.array([1, 0], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['entity_gene_score'] = np.array([1, 0], dtype=np.float64)
                self.base_assignment['entity_gene_score'] = 0
            else:
                self.g.factor(['entity_gene_score'], potential=np.array([0, 1], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['entity_gene_score'] = np.array([0, 1], dtype=np.float64)
                self.base_assignment['entity_gene_score'] = 1


        # added by haiwang@2018.1.1
        if self.opt['ref']:  
            self.g.rv('ref', 2)
            ratio = 0
            # check the pattern "XXX et al, 2015"
            RE_D = re.compile('\d')
            for i in range(len(x)-4):
                item = x[i]
                item_next = x[i+1]
                item_pre = x[i-3:i]
                # check people's name
                name = False
                for cha in item_pre:
                  if cha[0].isupper() and cha[1:].islower():
                      name = True
                item_after = " ".join(x[i+1:i+4])
                if (item == 'et' and (item_next == 'al' or item_next == 'al.') ) and name and RE_D.search(item_after) != None:
                    ratio += 1
                # [8] [9]    
                if item == '[' and ']' in item_after or item == '(' and ')' in item_after:
                    ratio += 1

            if ratio >= 3:
                self.g.factor(['ref'], potential=np.array([1, 0], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['ref'] = np.array([1, 0], dtype=np.float64)
                self.base_assignment['ref'] = 0
            else:
                self.g.factor(['ref'], potential=np.array([0, 1], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['ref'] = np.array([0, 1], dtype=np.float64)
                self.base_assignment['ref'] = 1

        if self.opt['intro']:  
            self.g.rv('intro', 2)
            ratio = 0
            # check the pattern "10.1200/JCO.2012.42.2592"
            for i in range(len(x)):
                item = x[i]
                if '/' in item:
                  data = [1 if num.isdigit() else 0 for num in item.split(".")]
                else:
                  data = []  
                if (sum(data) >= 3 or (item == 'doi' and x[i+1] == ":") or item == 'journal' or (item == "Cancer" and x[i+1] == "Res") or item == "Carcinogenesis"
                    or (item == "Proc" and x[i+1] == "Natl" and x[i+2] == "Acad") or (item == "Clin" and x[i+1]=="Invest") or (item == "Engl" and "Med" in x[i+1:x+5])
                    or (item == "J" and x[i+1] == "Clin" and x[i+2] == "Oncol") or (item == "Oncogene") or (item == "Nat" and x[i+1] == "Protoc") 
                    or (item == "Surg" and x[i+1] == "Oncol") or (item == "J" and x[i+1] == "Oncol") or (item == "Mol" and x[i+1] == "Sci") or (item == "Lung" and x[i+1] == "Cancer")
                    or (item == "Mol" and x[i+1] == "Cancer") or (item == "Lancet" and x[i+1] == "Oncol") or item == "PMID" or (item == "Cancer" and x[i+1] == "Med")
                    or (item == "Transl" and x[i+1] == "Med") or (item == "PLoS" and x[i+1] == "Med")):
                    ratio += 1
            if ratio >= 3:
                self.g.factor(['intro'], potential=np.array([1, 0], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['intro'] = np.array([1, 0], dtype=np.float64)
                self.base_assignment['intro'] = 0
            else:
                self.g.factor(['intro'], potential=np.array([0, 1], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['intro'] = np.array([0, 1], dtype=np.float64)
                self.base_assignment['intro'] = 1

        if self.opt['unnormal_seq']:
            self.g.rv('unnormal_seq', 2)
            ratio = 0
            long_seq = 0
            for item in x:
                if item !=',' and item !='.' and item !=';':
                  # some special char such as XXX/-+XXX    
                  if (item[0].isupper() or item[-1].isupper() or item.isdigit() or '/' in item or '>' in item or '+' in item or '-' in item):
                      ratio += 1
                      long_seq += 1
                  else:
                      long_seq = 0

            ratio /= float(len(x))         
            if ratio >= 0.4 and long_seq >= 5:
                self.g.factor(['unnormal_seq'], potential=np.array([1, 0], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['unnormal_seq'] = np.array([1, 0], dtype=np.float64)
                self.base_assignment['unnormal_seq'] = 0
            else:
                self.g.factor(['unnormal_seq'], potential=np.array([0, 1], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['unnormal_seq'] = np.array([0, 1], dtype=np.float64)
                self.base_assignment['unnormal_seq'] = 1

        # caption for the figure or table    
        if self.opt['fig_table']:
            self.g.rv('fig_table', 2)
            ratio = 0
            long_seq = 0
            for item in x:
                if item.lower() == 'table' or item.lower() in ['figure', 'fig', 'figures', 'figs']:
                    # some explnationation text to resolve    
                    ratio += 1
            if ratio >= 3:
                self.g.factor(['fig_table'], potential=np.array([1, 0], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['fig_table'] = np.array([1, 0], dtype=np.float64)
                self.base_assignment['fig_table'] = 0
            else:
                self.g.factor(['fig_table'], potential=np.array([0, 1], dtype=np.float64)) # encoding the 0, 1
                self.initial_potential['fig_table'] = np.array([0, 1], dtype=np.float64)
                self.base_assignment['fig_table'] = 1

        # single factor for the mentions         
        # the mention's token number
        if self.opt['token_length']:
            # for each mentions
            for idx, item in enumerate(mentions):
                self.g.rv('token_length'+str(idx), 2)
                if len(mentions[idx]) >= 2:
                    self.g.factor(['token_length'+str(idx)], potential=np.array([1, 0], dtype=np.float64)) # encoding the 0, 1
                    self.initial_potential['token_length'+str(idx)] = np.array([1, 0], dtype=np.float64)
                    self.base_assignment['token_length'+str(idx)] = 0
                elif len(mentions[idx]) == 1:
                    self.g.factor(['token_length'+str(idx)], potential=np.array([0, 1], dtype=np.float64)) # encoding the 0, 1
                    self.initial_potential['token_length'+str(idx)] = np.array([0, 1], dtype=np.float64)
                    self.base_assignment['token_length'+str(idx)] = 1

        # the mentions's string length   
        if self.opt['str_length']:
            # for each mentions
            for idx, item in enumerate(mentions):
                self.g.rv('str_length'+str(idx), 3)
                mention_str = " ".join([x[i] for i in mentions[idx]] )
                if len(mention_str) >= 6:
                    self.g.factor(['str_length'+str(idx)], potential=np.array([1, 0, 0], dtype=np.float64)) # probability 0, 1
                    self.initial_potential['str_length'+str(idx)] = np.array([1, 0, 0], dtype=np.float64)
                    self.base_assignment['str_length'+str(idx)] = 0
                elif len(mention_str) < 6 and len(mention_str) >= 3:
                    self.g.factor(['str_length'+str(idx)], potential=np.array([0, 1, 0], dtype=np.float64)) # probability 0, 1
                    self.initial_potential['str_length'+str(idx)] = np.array([0, 1, 0], dtype=np.float64)
                    self.base_assignment['str_length'+str(idx)] = 1
                else:
                    self.g.factor(['str_length'+str(idx)], potential=np.array([0, 0, 1], dtype=np.float64))
                    self.initial_potential['str_length'+str(idx)] = np.array([0, 0, 1], dtype=np.float64)
                    self.base_assignment['str_length'+str(idx)] = 2    


        # the mentions's token case   
        if self.opt['case_token']:
            # for each mentions
            for idx, item in enumerate(mentions):
                mention_str = " ".join([x[i] for i in mentions[idx]] )
                ratio_token = utils.check_uppercase_cnt(mention_str)
                self.g.rv('case_token'+str(idx), 2)
                if ratio_token >= 0.3:
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
            for idx, item in enumerate(mentions):                
                mention_str = " ".join([x[i] for i in mentions[idx]] )
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
            for idx, item in enumerate(mentions):
                self.g.rv('mention_ctx_entropy'+str(idx), 3)
                left_ctx = x[mentions[idx][0]-10:mentions[idx][0]] 
                right_ctx = x[mentions[idx][-1]:mentions[idx][-1]+10]
                ratio = 0; ctx = []; ctx.extend(left_ctx); ctx.extend(right_ctx)
                for word in ctx: 
                    if word in self.ctx_freq.keys():
                        ratio += self.ctx_freq[word]

                if ratio >= 0.05: # less important since too much entropy
                    self.g.factor(['mention_ctx_entropy'+str(idx)], potential=np.array([0, 0, 1], dtype=np.float64))
                    self.initial_potential['mention_ctx_entropy'+str(idx)] = np.array([0, 0, 1], dtype=np.float64)
                    self.base_assignment['mention_ctx_entropy'+str(idx)] = 2
                elif ratio<0.05 and ratio>=1e-6:
                    self.g.factor(['mention_ctx_entropy'+str(idx)], potential=np.array([0, 1, 0], dtype=np.float64))
                    self.initial_potential['mention_ctx_entropy'+str(idx)] = np.array([0, 1, 0], dtype=np.float64)
                    self.base_assignment['mention_ctx_entropy'+str(idx)] = 1 
                else:
                    self.g.factor(['mention_ctx_entropy'+str(idx)], potential=np.array([1, 0, 0], dtype=np.float64)) # most important
                    self.initial_potential['mention_ctx_entropy'+str(idx)] = np.array([1, 0, 0], dtype=np.float64)
                    self.base_assignment['mention_ctx_entropy'+str(idx)] = 0

        # the mention entropy, basically it's a function of the entity's probability     
        if self.opt['entity_entropy']:
            # for each entity type
            for idx, item in enumerate(mentions):
                self.g.rv('entity_entropy'+str(idx), 3)
                mention = " ".join([x[i] for i in mentions[idx]] )
                # check the drug, sure, it's not in gene
                if idx == 0:  
                  if mention in self.entity_type_freq.keys():
                      ratio = self.entity_type_freq[mention]
                  else:
                      ratio = 1e-7 # assume important since too small probability
                
                # check gene, ignore the variant
                elif idx == 1:
                  if mention in self.entity_type_freq.keys():
                      ratio = self.entity_type_freq[mention]
                  else:
                      ratio = 1e-7 # assume important since too small probability                  

                if ratio >= 0.02: # less important since too much entropy
                    self.g.factor(['entity_entropy'+str(idx)], potential=np.array([0, 0, 1], dtype=np.float64))
                    self.initial_potential['entity_entropy'+str(idx)] = np.array([0, 0, 1], dtype=np.float64)
                    self.base_assignment['entity_entropy'+str(idx)] = 2
                elif ratio < 0.02 and ratio >= 1e-6:
                    self.g.factor(['entity_entropy'+str(idx)], potential=np.array([0, 1, 0], dtype=np.float64))
                    self.initial_potential['entity_entropy'+str(idx)] = np.array([0, 1, 0], dtype=np.float64)
                    self.base_assignment['entity_entropy'+str(idx)] = 1 
                else:
                    self.g.factor(['entity_entropy'+str(idx)], potential=np.array([1, 0, 0], dtype=np.float64)) # most important
                    self.initial_potential['entity_entropy'+str(idx)] = np.array([1, 0, 0], dtype=np.float64)
                    self.base_assignment['entity_entropy'+str(idx)] = 0
        
        if self.opt['special_char']:
            
            # for each entity type
            for idx, item in enumerate(mentions):
                self.g.rv('special_char'+str(idx), 2)
                mention = " ".join([x[i] for i in mentions[idx]] )
                if '-' in mention or utils.hasNumbers(mention): # important since contains special char
                    self.g.factor(['special_char'+str(idx)], potential=np.array([1, 0], dtype=np.float64))
                    self.initial_potential['special_char'+str(idx)] = np.array([1, 0], dtype=np.float64)
                    self.base_assignment['special_char'+str(idx)] = 0 
                else:
                    self.g.factor(['special_char'+str(idx)], potential=np.array([0, 1], dtype=np.float64)) # most important
                    self.initial_potential['special_char'+str(idx)] = np.array([0, 1], dtype=np.float64)
                    self.base_assignment['special_char'+str(idx)] = 1

        # for this sentences
        if self.opt['table']:
            self.g.factor(['table', 'z0'], potential=np.array([
            [1, 0],
            [0.5, 0.5],
            ]))
                
            # save current potential to calculate the gradient
            self.current_pairwisepotential['table' + '&&' + 'z0'] = np.array([
            [1, 0],
            [0.5, 0.5],
            ])

        if self.opt['ref']:
            self.g.factor(['ref', 'z0'], potential=np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            ]))
                
            # save current potential to calculate the gradient
            self.current_pairwisepotential['ref' + '&&' + 'z0'] = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            ])

        # adding the gene score
        if self.opt['entity_gene_score']:
            self.g.factor(['entity_gene_score', 'z0'], potential=np.array([
            [0.7, 0.3],
            [0.4, 0.6],
            ]))
                
            # save current potential to calculate the gradient
            self.current_pairwisepotential['entity_gene_score' + '&&' + 'z0'] = np.array([
            [0.7, 0.3],
            [0.4, 0.6],
            ])

        if self.opt['intro']:
            self.g.factor(['intro', 'z0'], potential=np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            ]))
                
            # save current potential to calculate the gradient
            self.current_pairwisepotential['intro' + '&&' + 'z0'] = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            ])


        if self.opt['fig_table']:
            self.g.factor(['fig_table', 'z0'], potential=np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            ]))
                
            # save current potential to calculate the gradient
            self.current_pairwisepotential['fig_table' + '&&' + 'z0'] = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            ])

        if self.opt['mention_fig']:
            self.g.factor(['mention_fig', 'z0'], potential=np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            ]))
                
            # save current potential to calculate the gradient
            self.current_pairwisepotential['mention_fig' + '&&' + 'z0'] = np.array([
            [0.9, 0.1],
            [0.5, 0.5],
            ])


        if self.opt['gene_mutation']:
            self.g.factor(['gene_mutation', 'z0'], potential=np.array([
            [0.7, 0.3],
            [0.5, 0.5],
            ]))
                
            # save current potential to calculate the gradient
            self.current_pairwisepotential['gene_mutation' + '&&' + 'z0'] = np.array([
            [0.7, 0.3],
            [0.5, 0.5],
            ])


        if self.opt['unnormal_seq']:
            self.g.factor(['unnormal_seq', 'z0'], potential=np.array([
            [1, 0],
            [0.5, 0.5],
            ]))
                
            # save current potential to calculate the gradient
            self.current_pairwisepotential['unnormal_seq' + '&&' + 'z0'] = np.array([
            [1, 0],
            [0.5, 0.5],
            ])


        # the acronym on the left and right
        temp_acronym_list = [] 
        acronym_list = []
        if self.opt['acronym']:
            # for each mentions
            for idx, item in enumerate(mentions):
                
                left = x[item[0]-2:item[0]]
                len_text = item[-1]-item[0]+1
                left_ctx = set(x[item[0]-len_text-1:item[0]])
                right = x[item[-1]+1:item[-1]+2]
                right_ctx = x[item[-1]+1:item[-1]+len_text+2]
                
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

                        # exclude this postive mentions
                        new_prob = [1 - k for k in initial_mention_potential] # inverse the probability
                        self.g.factor(['z_mention'+str(idx)], potential=np.array(new_prob, dtype=np.float64))
                        self.initial_potential['z_mention'+str(idx)] = np.array(new_prob, dtype=np.float64)
                        potential_type = False
                            
                    self.g.factor(['acronym'+str(idx), 'z0'], potential=np.array([
                    [0.4, 0.6],
                    [0.6, 0.4],
                    ]))

                    # save current potential to calculate the gradient
                    self.current_pairwisepotential['acronym'+str(idx) + '&&' + 'z0'] = np.array([
                    [0.4, 0.6],
                    [0.6, 0.4],
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
                            new_prob = [1 - k for k in initial_mention_potential] # inverse the probability
                            self.g.factor(['z_mention'+str(idx)], potential=np.array(new_prob, dtype=np.float64))
                            self.initial_potential['z_mention'+str(idx)] = np.array(new_prob, dtype=np.float64)
                            potential_type = False

                        self.g.factor(['acronym'+str(idx), 'z0'], potential=np.array([
                        [0.4, 0.6],
                        [0.6, 0.4],
                        ]))                        

                        # save current potential to calculate the gradient
                        self.current_pairwisepotential['acronym'+str(idx) + '&&' + 'z0'] = np.array([
                        [0.4, 0.6],
                        [0.6, 0.4],
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
        if self.opt['token_length']:
            # for each mentions
            for idx, item in enumerate(mentions):
                self.g.factor(['token_length'+str(idx), 'z0'], potential=np.array([
                [0.4, 0.6],
                [0.6, 0.4],
                ]))
                
                # save current potential to calculate the gradient
                self.current_pairwisepotential['token_length'+str(idx) + '&&' + 'z0'] = np.array([
                [0.4, 0.6],
                [0.6, 0.4],
                ])
    
        if self.opt['str_length']:
            # for each mentions
            for idx, item in enumerate(mentions):
                self.g.factor(['str_length'+str(idx), 'z0'], potential=np.array([
                [0.4, 0.6],
                [0.5, 0.5],
                [0.6, 0.4],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['str_length'+str(idx) + '&&' + 'z0'] = np.array([
                [0.4, 0.6],
                [0.5, 0.5],
                [0.6, 0.4],
                ])

        if self.opt['drug_sentivity']:
            # for each mentions
            self.g.factor(['drug_sentivity', 'z0'], potential=np.array([
            [0.4, 0.6],
            [1.0, 0],
            ]))

            # save current potential to calculate the gradient
            self.current_pairwisepotential['drug_sentivity' + '&&' + 'z0'] = np.array([
            [0.4, 0.6],
            [1.0, 0],
            ])


        if self.opt['gene']:
            # for each mentions
            self.g.factor(['gene', 'z0'], potential=np.array([
            [0.4, 0.6],
            [1.0, 0],
            ]))

            # save current potential to calculate the gradient
            self.current_pairwisepotential['gene' + '&&' + 'z0'] = np.array([
            [0.4, 0.6],
            [1.0, 0],
            ])

        if self.opt['triple_pair']:
            # for each mentions
            self.g.factor(['triple_pair', 'z0'], potential=np.array([
            [0.1, 0.9],
            [0.5, 0.5],
            ]))

            # save current potential to calculate the gradient
            self.current_pairwisepotential['triple_pair' + '&&' + 'z0'] = np.array([
            [0.1, 0.9],
            [0.5, 0.5],
            ])

        if self.opt['case_token']:
            # for each mentions
            for idx, item in enumerate(mentions):
                self.g.factor(['case_token'+str(idx), 'z0'], potential=np.array([
                [0.4, 0.6],
                [0.6, 0.4],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['case_token'+str(idx) + '&&' + 'z0'] = np.array([
                [0.4, 0.6],
                [0.6, 0.4],
                ])

        if self.opt['case_str']:
            # for each mentions
            for idx, item in enumerate(mentions):
                self.g.factor(['case_str'+str(idx), 'z0'], potential=np.array([
                [0.4, 0.6],
                [0.6, 0.4],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['case_str'+str(idx) + '&&' + 'z0'] = np.array([
                [0.4, 0.6],
                [0.6, 0.4],
                ])

        if self.opt['mention_ctx_entropy']:
            # for each mentions
            for idx, item in enumerate(mentions):
                self.g.factor(['mention_ctx_entropy'+str(idx), 'z0'], potential=np.array([
                [0.4, 0.6],
                [0.5, 0.5],
                [0.6, 0.4],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['mention_ctx_entropy'+str(idx) + '&&' + 'z0'] = np.array([
                [0.4, 0.6],
                [0.5, 0.5],
                [0.6, 0.4],
                ])

        if self.opt['entity_entropy']:
            # for each mentions
            for idx, item in enumerate(mentions):
                self.g.factor(['entity_entropy'+str(idx), 'z0'], potential=np.array([
                [0.4, 0.6],
                [0.5, 0.5],
                [0.6, 0.4],
                ]))

                # save current potential to calculate the gradient
                self.current_pairwisepotential['entity_entropy'+str(idx) + '&&' + 'z0'] = np.array([
                [0.4, 0.6],
                [0.5, 0.5],
                [0.6, 0.4],
                ])

        if self.opt['special_char']:
            # for each mentions
            for idx, item in enumerate(mentions):
                self.g.factor(['special_char'+str(idx), 'z0'], potential=np.array([
                [0.4, 0.6],
                [0.6, 0.4],
                ]))
                
                # save current potential to calculate the gradient
                self.current_pairwisepotential['special_char'+str(idx) + '&&' + 'z0'] = np.array([
                [0.4, 0.6],
                [0.6, 0.4],
                ])

                
        # modify it's initial value since the signal it's too strong
        if self.opt['cross_sentence']:
            for item in self.acronym_list:
                for idx2 in range(len(mentions)):
                    if item[0] == " ".join([x[i] for i in mentions[idx2]] ):
                        if item[1] == False:
                            new_prob = [1 - k for k in initial_mention_potential] # inverse the probability
                            self.g.factor(['z_mention'+str(idx2)], potential=np.array(new_prob, dtype=np.float64))
                            self.initial_potential['z_mention'+str(idx)] = np.array(new_prob, dtype=np.float64)

        # the acronym factor 
        if self.opt['cross_sentence']:
            # for each predessor
            for idx1, item in enumerate(self.acronym_list):
                for idx2 in range(len(mentions)):
                    if item[0] == " ".join([x[i] for i in mentions[idx2]] ):
                        self.g.factor(['cross_sentence'+str(idx1), 'z0'], potential=np.array([
                        [0.4, 0.6],
                        [0.6, 0.4],
                        ]))

                        # save current potential to calculate the gradient
                        self.current_pairwisepotential['cross_sentence'+str(idx1) + '&&' + 'z0'] = np.array([
                        [0.4, 0.6],
                        [0.6, 0.4],
                        ]) 
        

        # delete all useless instance to save the memory
        del self.instance
        del self.opt
        del self.entity_type_freq
        del self.ctx_freq
        del self.acronym_list
        del self.arc_type_dict
            
        return temp_acronym_list
        

    def update_factor_graph_unary(self, key, potentials):

        self.g.factor([key], potential=np.array(potentials)) # uniformly probability

        # keep the current potentials
        if 'DL' in key:
            self.current_DL_potential[key] = np.array(potentials)

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
        
        self.N = 1
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
