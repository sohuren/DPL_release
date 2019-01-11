from copy import deepcopy

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

def longest_common_substring(s1, s2):
   m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
   longest, x_longest = 0, 0
   for x in xrange(1, 1 + len(s1)):
       for y in xrange(1, 1 + len(s2)):
           if s1[x - 1] == s2[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
           else:
               m[x][y] = 0
   return s1[x_longest - longest: x_longest]

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
