from copy import deepcopy
import numpy as np

def make_csv_file(vcs_file, confidence):
    
    fp = open(vcs_file, 'wt+')

    for instance in confidence:
            
        text_inc = " ".join(instance['inc_text']).replace(","," ")
        text_exc = " ".join(instance['exc_text']).replace(","," ")

        for item in instance['inc']:
            entity = " ".join(item[0]).replace(",","")
            fp.write(text_inc + "," + entity + "," + str(item[1]) + "," + str(item[2][0][0]) + "," + str(item[2][0][1]) + "\n")
        for item in instance['exc']:
            entity = " ".join(item[0]).replace(",","")
            fp.write(text_exc + "," + entity + "," + str(item[1]) + "," + str(item[2][0][0]) + "," + str(item[2][0][1]) + "\n")
    
    fp.close()
        

def check_uppercase_cnt(s):

    s = s.split()
    total = len(s)
    cnt = 0
    for token in s:
        if token.upper() == token:
            cnt += 1

    return float(cnt)/total        


def check_uppercase_str_cnt(s):

    total = len(s)
    cnt = 0
    for token in s:
        if token.upper() == token and token != ' ':
            cnt += 1

    return float(cnt)/total

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

# updates the graph parameters, delta is the single sample weight
def update_graph_parameters(current, gradient, eta, delta):

    new_parameters = deepcopy(current)    
    for pairwise_key in current.keys():
        gradient[pairwise_key] = delta*current[pairwise_key] - gradient[pairwise_key]
    
    new_parameters[pairwise_key] = current[pairwise_key] - gradient[pairwise_key]*eta

    return new_parameters

# accumulate_gradient over different graph instances
# need to check the keys since different graph might has different edges

def accumulate_gradient(gradients, all_keys):

    accu_gradient = {}
    sample_size = {} # independent sample size for keys
    for key in all_keys:
        sample_size[key] = 0

    for pairwise_key in all_keys:
        for grad in gradients:
            if pairwise_key in grad.keys():
                sample_size[pairwise_key] += 1
                if pairwise_key not in accu_gradient.keys(): 
                    accu_gradient[pairwise_key] = grad[pairwise_key]
                else:
                    accu_gradient[pairwise_key] += grad[pairwise_key]

    # normalize the gradient over probability
    for key in accu_gradient.keys():
        accu_gradient[key] /= sample_size[key]

    return accu_gradient
