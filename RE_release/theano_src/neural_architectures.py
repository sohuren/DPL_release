import theano.tensor as T
from neural_lib import *

# Note: need to refactor many places to return regularizable params  lists for the optimization.

def calculate_params_needed(chips):
    l = []
    for c, _ in chips:
        l += c.needed_key()
    return l

''' Automatically create(initialize) layers and hook them together'''
def stackLayers(chips, current_chip, params, feature_size=0, entity_size=2):
    instantiated_chips = []
    print 'stack layers!!!'
    for e in chips:
        previous_chip = current_chip
        if e[1].endswith('feature_emission_trans'):
            current_chip = e[0](e[1], params).prepend(previous_chip, feature_size)
        elif e[1].endswith('target_columns'):
            current_chip = e[0](e[1], params).prepend(previous_chip, entity_size)
        else:
            current_chip = e[0](e[1], params).prepend(previous_chip)
        instantiated_chips.append((current_chip, e[1]))
        print 'current chip:', e[1], "In_dim:", current_chip.in_dim, "Out_dim:", current_chip.out_dim
        print 'needed keys:'
        for e in current_chip.needed_key():
            print (e, params[e])
    return instantiated_chips 

''' Compute the initialized layers by feed in the inputs. '''
def computeLayers(instantiated_chips, current_chip, params, feature_input=None, entities_input=None, mask=None, sample_weights=None):
    print 'compute layers!!!'
    regularizable_params = []
    for e in instantiated_chips:
        previous_chip = current_chip
        current_chip = e[0]
        print 'current chip:', e[1], "In_dim:", current_chip.in_dim, "Out_dim:", current_chip.out_dim
        if e[1].endswith('feature_emission_trans'):
            internal_params = current_chip.parameters
            current_chip.compute(previous_chip.output_tv, feature_input)
        elif e[1].endswith('target_columns') or e[1].endswith('Entity_Att'):
            internal_params = current_chip.parameters
            current_chip.compute(previous_chip.output_tv, entities_input)
        elif e[1].endswith('lstm'):
            internal_params = current_chip.parameters
            current_chip.compute(previous_chip.output_tv, mask)
        elif e[1].endswith('logistic_regression'):
            internal_params = current_chip.parameters
            print ("compute layers")
            print sample_weights
            current_chip.compute(previous_chip.output_tv, sample_weights)
        else:
            internal_params = current_chip.parameters
            current_chip.compute(previous_chip.output_tv)
            
        assert current_chip.output_tv is not None
        for k in internal_params:
            print 'internal_params:', k.name
            assert k.is_regularizable
            params[k.name] = k
            regularizable_params.append(k)
    return regularizable_params

''' Compile the architectures for single-task learning: stack the layers, compute the forward pass.'''
def RelationStackMaker(chips, params, graph=False, weighted=False, batched=False):
    if batched:
        emb_input = T.itensor3('emb_input')
        entities_tv = [T.fmatrix('enidx_'+str(i)).astype(theano.config.floatX) for i in range(params['num_entity'])]
        sample_weights = T.fvector('sample_weight')
        if graph:
            if weighted:
                masks = T.ftensor4('child_mask')
            else:
                masks = T.ftensor3('child_mask')
        else:
            masks = T.fmatrix('batch_mask')
        
    else:
        emb_input = T.imatrix('emb_input')
        entities_tv = [T.fvector('enidx_'+str(i)).astype(theano.config.floatX) for i in range(params['num_entity'])]
        sample_weights = T.fvector('sample_weight')
        if graph:
            if weighted:
                masks = T.ftensor3('child_mask')
            else:
                masks = T.fmatrix('child_mask')
        else:
            masks = None
    #print masks, type(masks), masks.ndim
    current_chip = Start(params['voc_size'], emb_input)  
    print '\n', 'Building Stack now', '\n', 'Start: ', params['voc_size'], 'out_tv dim:', current_chip.output_tv.ndim
    instantiated_chips = stackLayers(chips, current_chip, params, entity_size=params['num_entity'])
    regularizable_params = computeLayers(instantiated_chips, current_chip, params, entities_input=entities_tv, mask=masks, sample_weights=sample_weights )
    ### Debug use: Get the attention co-efficiency and visualize. ###
    for c in instantiated_chips:
        if c[1].endswith('Entity_Att'):
            assert hasattr(c[0], 'att_wt_arry')
            assert hasattr(c[0], 'entity_tvs')
            attention_weights = c[0].att_wt_arry
            entity_tvs = c[0].entity_tvs
    
    current_chip = instantiated_chips[-1][0]
    if current_chip.output_tv.ndim == 2:
        pred_y = current_chip.output_tv #T.argmax(current_chip.output_tv, axis=1)
    else:
        pred_y = current_chip.output_tv #T.argmax(current_chip.output_tv) #, axis=1)
    gold_y = (current_chip.gold_y
            if hasattr(current_chip, 'gold_y')
            else None)
    # Show all parameters that would be needed in this system
    params_needed = calculate_params_needed(instantiated_chips)
    print "Parameters Needed", params_needed
    for k in params_needed:
        assert k in params, k
        print k, params[k]
    assert hasattr(current_chip, 'score')
    cost = current_chip.score #/ params['nsentences'] 
    cost_arr = [cost]
    for layer in instantiated_chips[:-1]:
        if hasattr(layer[0], 'score'):
            print layer[1]
            cost += params['cost_coef'] * layer[0].score
            cost_arr.append(params['cost_coef'] * layer[0].score)

    grads = T.grad(cost,
            wrt=regularizable_params)
            #[params[k] for k in params if (hasattr(params[k], 'is_regularizable') and params[k].is_regularizable)])
    print 'Regularizable parameters:'
    for k, v in params.items():
        if hasattr(v, 'is_regularizable'):
            print k, v, v.is_regularizable
    if graph or batched:
        #return (emb_input, masks, entities_tv, attention_weights, entity_tvs, gold_y, pred_y, cost, grads, regularizable_params) 
        return (emb_input, masks, entities_tv, sample_weights, gold_y, pred_y, cost, grads, regularizable_params) 
    else: 
        return (emb_input, entities_tv, sample_weights, gold_y, pred_y, cost, grads, regularizable_params, sample_weights) 


''' Single task architectures, suitable for CNN, BiLSTM (with/without input attention). 
    The major difference between this and the graphLSTM single task architecture is the final line: the paremeters given to RelationStackMaker function'''
def LSTMRelation(params):
    chips = [
            (Embedding          ,'wemb1'),
            (BiLSTM               ,'lstm'),
            (TargetHidden       ,'get_target_columns'),
            #(Entity_attention   ,'hidden_Entity_Att'),
            #(BiasedLinear       ,'MLP_hidden'),
            #(Activation         ,'MLP_activation'),
            (LogitRegression        ,'logistic_regression'),
            (L2Reg,             'L2Reg'),
            ]
    return RelationStackMaker(chips, params, batched=(params['batch_size']>1))

def CNNRelation(params):
    chips = [
            (Embedding          ,'wemb1'),
            (Entity_attention   ,'input_Entity_Att'),
            (Convolutional_NN   ,'CNN'),
            (LogitRegression        ,'logistic_regression'),
            (L2Reg,             'L2Reg'),
            ]
    return RelationStackMaker(chips, params, batched=(params['batch_size']>1))



def GraphLSTMRelation(params):
    chips = [
            (Embedding          ,'wemb1'),
            (BiGraphLSTM  ,'lstm'),
            (TargetHidden       ,'get_target_columns'),
            (LogitRegression        ,'logistic_regression')
            ]
    return RelationStackMaker(chips, params, graph=True, batched=(params['batch_size']>1))


def WeightedGraphLSTMRelation(params):
    chips = [
            (Embedding          ,'wemb1'),
            (BiGraphLSTM_Wtd  ,'lstm'),
            (TargetHidden       ,'get_target_columns'),
            (LogitRegression        ,'logistic_regression'),
            ]
    return RelationStackMaker(chips, params, graph=True, weighted=True, batched=(params['batch_size']>1))


def WeightedAddGraphLSTMRelation(params):
    chips = [
            (Embedding          ,'wemb1'),
            (BiGraphLSTM_WtdEmbMult  ,'lstm'),
            (TargetHidden       ,'get_target_columns'),
            (LogitRegression        ,'logistic_regression'),
            ]
    return RelationStackMaker(chips, params, graph=True, weighted=True, batched=(params['batch_size']>1))
