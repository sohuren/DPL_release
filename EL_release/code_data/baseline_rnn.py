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
from utils import accumulate_gradient
from utils import update_graph_parameters

# Training settings with the pytorch
parser = argparse.ArgumentParser(description='entities disambuity with PyTorch (rnn settings)')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=6, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--word_embedding', type=str, default='./data/pubmed_parsed/embedding_vec_gene.pkl',
                    help='initial word embedding file')
parser.add_argument('--classifier_type', type = str, default="rnn",
                    help='the classifier type')
parser.add_argument('--windowSize', type=int, default=5,
                    help='the window size')
parser.add_argument('--dataroot', type=str, default='./data/pubmed_parsed',
                    help='the data root')
parser.add_argument('--train_data', type=str, default='chunk_train_41_gene.pkl',
                    help='train data')
parser.add_argument('--val_data', type=str, default='validation_gene_1_soft.pkl',
                    help='val data')
parser.add_argument('--test_data', type=str, default='test_gene_1_soft.pkl',
                    help='test data')
parser.add_argument('--vocab_path', type = str, default="./data/pubmed_parsed/vocab_gene.pkl",
                    help='the vocab path')
parser.add_argument('--embed_size', type=int,  default=200,
                    help='the initial word embedding size')
parser.add_argument('--fix_embed', type=bool,  default=False,
                    help='whether fix the embedding or not')
parser.add_argument('--nThreads', type=int,  default=1,
                    help='number of thread for the data reading')
parser.add_argument('--entity_type', type=str,  default='gene',
                    help='the current entity type we are trained on')
parser.add_argument('--initial_model', type=str,  default='',
                    help='the current entity type we are trained on')
parser.add_argument('--save_path', type=str,  default='./model/model.pkl',
                    help='the current entity type we are trained on')
parser.add_argument('--visulization_html', type=str,  default='./result/mlp_vis.html',
                    help='the html that can write')
parser.add_argument('--combine', type=str,  default='concatenate',
                    help='how to combine the wordvector: mean, max, product, concatenate, maxpool')
parser.add_argument('--confidence_html', type=str,  default='./result/confidence.html',
                    help='display the confidence for each prediction')
parser.add_argument('--gene_key', type=str,  default='./data/gene_key.pkl',
                    help='display the confidence for each prediction')
parser.add_argument('--window_size', type=int,  default=5,
                    help='lstm or rnn')
parser.add_argument('--class_label', type=int,  default=2,
                    help='output label number')
parser.add_argument('--hidden_size', type=int,  default=128,
                    help='hidden_size of rnn')
parser.add_argument('--num_layer', type=int,  default=1,
                    help='output label number')
parser.add_argument('--cell', type=str,  default="lstm",
                    help='lstm or rnn')
parser.add_argument('--max_confidence_instance', type=int,  default=200,
                    help='maximum positive or negative example')

# adjust the threshold for evaluation                     
parser.add_argument('--threshold', type=float,  default=0.5,
                    help='adjust the threshold for evaluation')

# different optimization method
parser.add_argument('--hard_em', help='whether use hard em', type=bool, default=False)
parser.add_argument('--stochastic', help='whether use the incremental EM or not', type=bool, default=False)
parser.add_argument('--multiple_M', help='whether train multiple M step', type=int, default=2)
parser.add_argument('--learn_graph', help='whether learn the graph parameters or not', action='store_true', default=False)
parser.add_argument('--learn_rate_graph', help='learning rate of graph parameters', type=float, default=0.01)
parser.add_argument('--graph_regularizer', help='regularizer of graph parameters', type=float, default=1e-4)

# E step or M step, split the task into different machine
parser.add_argument('--stage', help='E step or M step', type=str, default="E")
parser.add_argument('--prediction_file', help='prediction file', type=str, default="")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

sys.setrecursionlimit(20000)

torch.manual_seed(args.seed)
print (" use cuda: %d \n" % args.cuda)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

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

# dataset
args.file_path = os.path.join(args.dataroot, args.train_data)
data_loader = CreateDataLoader(args)
train_loader = data_loader.load_data()
args.file_path = os.path.join(args.dataroot, args.val_data)
data_loader = CreateDataLoader(args)
val_loader = data_loader.load_data()

# re-evaluate the probability under current rnn model
fp = open(os.path.join(args.dataroot, args.train_data), "rb")
train_data = pickle.load(fp)
fp.close()

fp = open(os.path.join(args.dataroot, args.val_data), "rb")
valid_data = pickle.load(fp)
fp.close()

# for visualization
args.file_path = os.path.join(args.dataroot, args.test_data)
fp = open(args.file_path)
test_data = pickle.load(fp)
fp.close()

# model
model = EncoderRNN(args.embed_size, args.hidden_size , vocab_size, args.num_layer, args.cell, wordvec, args.class_label)

if args.cuda:
    model.cuda()

# load the pre-trained model
if args.initial_model and os.path.exists(args.initial_model):
    print ("load pre-trained model")
    model.load_state_dict(torch.load(args.initial_model))
    if args.cuda:
        model = model.cuda()

if not args.fix_embed:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

#print ("remove all previous message passing \n")
#command = "rm ./result/mp_*.pkl"
#os.system(command)

# train procedure, we can use more complicated optimization method
def train_Mstep_RNN(epoch):

    model.train()
    all_instance = []

    for batch_idx, (data, batch_mask, mask, length, target) in enumerate(train_loader):

        if args.cuda:
            data, batch_mask, mask, target = data.cuda(), batch_mask.cuda().byte(), mask.cuda(), target.cuda()

        # make the data balance
        num_pos = float(sum(target[:,0] <= target[:, 1]))
        num_neg = float(sum(target[:,0] > target[:, 1]))
        weight = torch.ones(target.size(0), 1)
        mask_pos = (target[:,0] <= target[:, 1]).cpu().float()   
        mask_neg = (target[:,0] > target[:, 1]).cpu().float()
        weight = mask_pos*(num_pos + num_neg)/num_pos
        weight += mask_neg*(num_pos + num_neg)/num_neg
        weight = Variable(weight).cuda()
        data, mask, target = Variable(data), Variable(mask), Variable(target)
        optimizer.zero_grad()
        output = model.forward(data, batch_mask, mask)
        loss = F.kl_div(output, target, reduce=False)
        loss = loss.sum(dim=1)*weight
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0])) 


# train procedure, we can use more complicated optimization method
def train_Mstep_Potential(epoch, result):

    # calculate the gradient over all the samples
    current_parameters = {} # only need to get it one time
    grads = [] 

    for trial in result.keys():
        
        for i in range(len(result[trial]['inc'])):
            instance = result[trial]['inc'][i]
            text_inc = instance['text']        
            if len(instance['pos_neg_example']) >= 1:
                current_parameters.update(instance['graph'].get_current_parameters()) # reset the prior, this is necessary
                grads.append(instance['graph'].compute_gradient()) # get the gradient from this sample
                        
        for i in range(len(result[trial]['exc'])):
            
            instance = result[trial]['exc'][i]
            text_exc = instance['text']

            # only care the specified entity type
            if len(instance['pos_neg_example']) >= 1:
                current_parameters.update(instance['graph'].get_current_parameters())  # reset the prior, this is necessary  
                grads.append(instance['graph'].compute_gradient()) # get the gradient from this sample

    # update the graph parameters given all the parameter key 
    sample_gradient = accumulate_gradient(grads, current_parameters.keys())
    print ("old parameters ")
    print (current_parameters)

    print ("gradient ")
    print (sample_gradient)
    new_parameters = update_graph_parameters(current_parameters, sample_gradient, args.learn_rate_graph, args.graph_regularizer)

    print ("new parameters ")
    print (new_parameters)

    # reset all the parameter for the factor graph
    for trial in result.keys():
        
        for i in range(len(result[trial]['inc'])):
            instance = result[trial]['inc'][i]
            text_inc = instance['text']

            if len(instance['pos_neg_example']) >= 1:
                instance['graph'].update_all_factor_graph_pairwise(new_parameters) # reset the prior, this is necessary
                        
            result[trial]['inc'][i] = instance

        for i in range(len(result[trial]['exc'])):
            
            instance = result[trial]['exc'][i]
            text_exc = instance['text']

            # only care the specified entity type
            if len(instance['pos_neg_example']) >= 1:
                instance['graph'].update_all_factor_graph_pairwise(new_parameters)  # reset the prior, this is necessary  
                        
            result[trial]['exc'][i] = instance
    
    return result

# train procedure, we can use more complicated optimization method

def train_Estep(epoch, data, entity_type, vocab):

    model.eval()
    
    result = deepcopy(data)
    examples = {}
    factor_cnt = 0

    for trial in data.keys():
        
        for i in range(len(data[trial]['inc'])):
            instance = data[trial]['inc'][i]
            text_inc = instance['text']
            len_text = len(text_inc)

            # will resave the result
            new_instance = deepcopy(instance)
            
            tokens_inc = []
            tokens_inc.extend([vocab(token) for token in text_inc])
            tokens_inc = torch.LongTensor(tokens_inc)

            if args.cuda:
                tokens_inc = tokens_inc.cuda()
            input_data = Variable(tokens_inc, volatile=True)    
            
            # only care the specified entity type

            label = []
            potential = []    
            
            for idx, item in enumerate(instance['pos_neg_example']):

                # vectorize the mask
                mask = [0 for k in range(len_text)]
                for k in range(len(mask)):
                    if (k>=item[0] and k<=item[1]):
                        mask[k] = 1

                mask = torch.LongTensor(mask)

                if args.cuda:
                    mask = mask.cuda()

                mask = Variable(mask, volatile=True)
                output = model.forward(input_data[None], None, mask[None])
                marginal = output.data.squeeze().cpu().numpy() # get the index of the max log-probability
                # update each marginal probability for each mentions
                label.append(np.argmax(marginal))
                potential.append(marginal)

            if len(label) > 0: 
                new_instance['graph'].set_label(label)   
            for idx, item in enumerate(instance['pos_neg_example']):
                new_instance['graph'].update_factor_graph_unary("DL"+str(idx), potential[idx])     

            result[trial]['inc'][i] = new_instance

        for i in range(len(data[trial]['exc'])):
            
            instance = data[trial]['exc'][i]
            text_exc = instance['text']
            len_text = len(text_exc)

            # will resave the result
            new_instance = deepcopy(instance)

            tokens_exc = []
            tokens_exc.extend([vocab(token) for token in text_exc])
            tokens_exc = torch.LongTensor(tokens_exc)

            if args.cuda:
                tokens_exc = tokens_exc.cuda()
            input_data = Variable(tokens_exc, volatile=True)    
            
            # only care the specified entity type

            label = []
            potential = []
            for idx, item in enumerate(instance['pos_neg_example']):
                
                # vectorize the mask
                mask = [0 for k in range(len_text)]
                for k in range(len(mask)):
                    if (k>item[0]) and (k<=item[1]):
                        mask[k] = 1

                mask = torch.LongTensor(mask)

                if args.cuda:
                    mask = mask.cuda()

                mask = Variable(mask, volatile=True)
                output = model.forward(input_data[None], None, mask[None])
                marginal = output.data.squeeze().cpu().numpy() # get the index of the max log-probability
                label.append(np.argmax(marginal))                        
                potential.append(marginal)

            if len(label) > 0:
                new_instance['graph'].set_label(label)
            for idx, item in enumerate(instance['pos_neg_example']):
                new_instance['graph'].update_factor_graph_unary("DL"+str(idx), potential[idx])                         

            result[trial]['exc'][i] = new_instance

    return result

def distribute_message_passing(result):

    # now do the paralell message passing updating by writing all the files      
    chunk = {}
    chunk_size = 200 # takes around 15 minutes on slurm, totally will result 100 cpu jobs on slurm
    idx = 1
    total_chunk = 0
    randomstring = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30))
    random_dir = "./temp/%s"%randomstring
    if os.path.exists(random_dir):
        os.rmdir(random_dir)        
    os.mkdir(random_dir)

    for trial in result.keys():
        chunk[trial] = result[trial]
        if idx%chunk_size == 0:
            input_file = os.path.join(random_dir, "mp_"+str(total_chunk)+ "_" + str(epoch)+ ".pkl")
            fp = open(input_file, "wb+")
            pickle.dump(chunk, fp)
            fp.close()
            # now call the shell to lanuch the job
            output_file = os.path.join(random_dir, "mp_"+str(total_chunk)+ "_" + str(epoch+1)+ ".pkl")
            command = "bash submit_mp.sh %s %s "  % (input_file, output_file)
            os.system(command)
            time.sleep(10)
            print ("submitted job to slurm \n")
            chunk = {}
            total_chunk += 1

        idx += 1
    
    # the last chunks 
    if len(chunk.keys()) > 0:        
        input_file = os.path.join(random_dir, "mp_"+str(total_chunk)+ "_" + str(epoch)+ ".pkl")
        fp = open(input_file, "wb+")
        pickle.dump(chunk, fp)
        fp.close()
        # now call the shell to lanuch the job
        output_file = os.path.join(random_dir, "mp_"+str(total_chunk)+ "_" + str(epoch+1)+ ".pkl")
        command = "bash submit_mp.sh %s %s "  % (input_file, output_file)
        os.system(command)
        print ("submitted job to slurm \n")
        total_chunk += 1

    print ("total %d jobs submitted \n"%total_chunk)

    # checking all the new files generated
    while True:
        
        total_finished = 0
        for i in range(total_chunk):
            if not os.path.exists(os.path.join(random_dir, "mp_"+str(i)+ "_" + str(epoch+1)+ ".pkl")):
                time.sleep(5)    
            else:
                total_finished += 1
        
        if total_finished == total_chunk:
            cacheds = []
            new_state = []
            for i in range(total_chunk):
                cached = os.stat(os.path.join(random_dir, "mp_"+str(i)+ "_" + str(epoch+1)+ ".pkl")).st_mtime
                cacheds.append(cached)
            time.sleep(100)
            for i in range(total_chunk):
                curr = os.stat(os.path.join(random_dir, "mp_"+str(i)+ "_" + str(epoch+1)+ ".pkl")).st_mtime
                new_state.append(curr)

            if new_state == cacheds:         
                break

    print ("all message passing finished \n")
    # checking the result until get all of them, and then return the new marginal probability
    result = {}
    for i in range(total_chunk):
        fp = open(os.path.join(random_dir, "mp_"+str(i)+ "_" + str(epoch+1)+ ".pkl"), "rb")
        chunk = pickle.load(fp)
        result.update(chunk)
        fp.close()

    return result


# test procedure
def test(epoch):

    model.eval()
    val_loss = 0
    correct = 0
    prob_pos = []
    prob_neg = []

    for data, batch_mask, mask, length, target in val_loader:
        
        if args.cuda:
            data, batch_mask, mask, target = data.cuda(), batch_mask.byte().cuda(), mask.cuda(), target.cuda()

        data, mask, target = Variable(data, volatile=True), Variable(mask, volatile=True), Variable(target.select(1, 1).contiguous().view(-1).long())
        output = model.forward(data, batch_mask, mask)

        val_loss += F.nll_loss(output, target).data[0] # need to check here 
        pred = output.data.max(1)[1] # get the index of the max log-probability
        
        pos_prob = [np.exp(output.data.cpu().numpy()[idx][1]) for idx, l in enumerate(target.data.cpu().numpy()) if l == 1]
        neg_prob = [np.exp(output.data.cpu().numpy()[idx][1]) for idx, l in enumerate(target.data.cpu().numpy()) if l == 0]
         
        prob_pos.extend(pos_prob)
        prob_neg.extend(neg_prob)

        correct += pred.eq(target.data).cpu().sum()

    # adjust the threhold according to the sampled label
    all_prob = deepcopy(prob_pos)
    all_prob.extend(prob_neg)
    all_prob = np.asarray(all_prob)
    prob_pos = np.asarray(prob_pos)
    prob_neg = np.asarray(prob_neg)

    # 0.5 is the standard threshold 
    tp = np.sum(prob_pos >= 0.5)
    fp = np.sum(prob_pos < 0.5)
    tn = np.sum(prob_neg < 0.5)
    fn = np.sum(prob_neg >= 0.5)
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    f1 = 2.0*precision*recall/(precision+recall)
    
    val_loss = val_loss
    val_loss /= len(val_loader) # loss function already averages over batch size
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), precision: ({:.4f}), recall: ({:.4f}), f1: ({:.4f}) \n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset), precision, recall, f1))

    max_acc = float(correct)/len(val_loader.dataset)    
    
    for threshold in all_prob:
        
      tp = np.sum(prob_pos >= threshold)
      fp = np.sum(prob_pos < threshold)
      tn = np.sum(prob_neg < threshold)
      fn = np.sum(prob_neg >= threshold)
      acc = (tp + tn)/float(len(all_prob))

      if acc > max_acc:
        max_acc = acc
        args.threshold = threshold
        precision = tp/float(tp+fp)
        recall = tp/float(tp+fn)
        f1 = 2.0*precision*recall/(precision+recall)
        
    #print('\nVal set: Average loss: {:.4f}, Adjusted Accuracy: {}/{} ({:.4f}%), precision: ({:.4f}), recall: ({:.4f}), f1: ({:.4f})\n'.format(
    #    val_loss, len(val_loader.dataset)*max_acc, len(val_loader.dataset),
    #    100. * max_acc, precision, recall, f1))

# test procedure
def GetResult_valid(data, vocab, file_name):

    model.eval()
    fp = open(file_name, "wt+")
    fp.write("threshold: %f \n" % args.threshold)

    for trial in data.keys():
        for i in range(len(data[trial]['inc'])):
            instance = data[trial]['inc'][i]
            text_inc = instance['text']
            len_text = len(text_inc)
            tokens_inc = []
            tokens_inc.extend([vocab(token) for token in text_inc])
            tokens_inc = torch.LongTensor(tokens_inc)

            if args.cuda:
                tokens_inc = tokens_inc.cuda()
            input_data = Variable(tokens_inc, volatile=True)    
            
            for item in instance['pos_neg_example']:

                # vectorize the mask, for the entity
                mask = [0 for k in range(len_text)]

                # need to change the mask settings, for entity's attention: -10:10 window size
                batch_mask = [1 for k in range(len_text)] # ignore all first
                start_pos = max([item[0]-10, 0])
                end_pos = min([item[1]+10, len_text])
                batch_mask[start_pos:end_pos] = [0]*(end_pos-start_pos)

                for k in range(len(mask)):
                    if (k>=item[0] and k<item[1]+1):
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
                
                # write the prediction to file
                fp.write(trial + "\t")
                fp.write(" ".join(text_inc) + "\t")
                fp.write("true label:" + str(item[2]) + "\t")
                fp.write("prediction:" + str(label) + "\t")
                fp.write("p(x=1):" + str(np.exp(output.data.cpu().numpy()[0][1])) + "\n")

    fp.close()



# test procedure
def GetResult(data, entity_type, vocab):

    result = deepcopy(data)
    model.eval()
    examples = {}
    
    confidence = {}
    confidence['positive'] = []
    confidence['negative'] = []

    for trial in data.keys():
        
        for i in range(len(data[trial]['inc'])):
            instance = data[trial]['inc'][i]
            text_inc = instance['text']
            len_text = len(text_inc)
            # will resave the result
            new_instance = deepcopy(instance)
            new_instance['matched'][entity_type] = []

            tokens_inc = []
            tokens_inc.extend([vocab(token) for token in text_inc])
            tokens_inc = torch.LongTensor(tokens_inc)

            if args.cuda:
                tokens_inc = tokens_inc.cuda()
            input_data = Variable(tokens_inc, volatile=True)    
            
            # only care the specified entity type
            checked_item = []
            for item in instance['matched'][entity_type]:

                if item in checked_item:
                    continue
                
                # add some filters here
                if len(" ".join(text_inc[item[0]:item[1]+1])) <= 3:
                    continue

                checked_item.append(item)    
                # vectorize the mask
                mask = [0 for k in range(len_text)]

                # need to change the mask settings, for entity's attention: -10:10 window size
                batch_mask = [1 for k in range(len_text)] # ignore all first
                start_pos = max([item[0]-10, 0])
                end_pos = min([item[1]+10, len_text])
                batch_mask[start_pos:end_pos] = [0]*(end_pos-start_pos)

                for k in range(len(mask)):
                    if (k>=item[0] and k<item[1]+1):
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
                
                # save the result to the data type
                new_instance['matched'][entity_type].append((item[0], item[1], label, np.exp(output.data.cpu().numpy())))
                
            result[trial]['inc'][i] = new_instance

        for i in range(len(data[trial]['exc'])):
            instance = data[trial]['exc'][i]
            text_exc = instance['text']
            len_text = len(text_exc)

            # will resave the result
            new_instance = deepcopy(instance)
            new_instance['matched'][entity_type] = []

            tokens_exc = []
            tokens_exc.extend([vocab(token) for token in text_exc])
            tokens_exc = torch.LongTensor(tokens_exc)

            if args.cuda:
                tokens_exc = tokens_exc.cuda()
            input_data = Variable(tokens_exc, volatile=True)    
            
            # only care the specified entity type
            checked_item = []
            for item in instance['matched'][entity_type]:

                if item in checked_item:
                    continue

                # add some filters here
                if len(" ".join(text_exc[item[0]:item[1]+1])) <= 3:
                    continue

                checked_item.append(item)    
                
                # vectorize the mask
                mask = [0 for k in range(len_text)]

                # need to change the mask settings, for entity's attention: -10:10 window size
                batch_mask = [1 for k in range(len_text)] # ignore all first
                start_pos = max([item[0]-10, 0])
                end_pos = min([item[1]+10, len_text])
                batch_mask[start_pos:end_pos] = [0]*(end_pos-start_pos)

                for k in range(len(mask)):
                    if (k>=item[0]) and (k<item[1]+1):
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

                # save the result to the data type
                label = pred.cpu().numpy()[0]

                if label == 0 and 1 - prob >= args.threshold:
                    label = 1
                
                if label == 1 and prob < args.threshold:
                    label = 0

                new_instance['matched'][entity_type].append((item[0], item[1], label, np.exp(output.data.cpu().numpy())))
                
            result[trial]['exc'][i] = new_instance

    # prepration writing prediction to html file            
    for trial in result.keys():
        
        examples[trial] = {}
        examples[trial]['inc'] = []
        examples[trial]['exc'] = []
        
        for instance in result[trial]['inc']:
            context_inc = " ".join(instance['text']).encode("ascii", "ignore").split()
            matches_inc = {}
            # matched for inc
            for key in instance['matched'].keys():
                matches_inc[key] = instance['matched'][key]

            examples[trial]['inc'].append((context_inc, matches_inc))

        for instance in result[trial]['exc']:
            context_exc = " ".join(instance['text']).encode("ascii", "ignore").split()
            matches_exc = {}
            # matched for inc
            for key in instance['matched'].keys():
                matches_exc[key] = instance['matched'][key]

            examples[trial]['exc'].append((context_exc, matches_exc))


    # sample the confidence example for evaluation
    positive = 0
    negative = 0

    for trial in result.keys():

        random_seed = True
        if np.random.randint(10) >= 5:
            random_seed = False
            
        # now checking each instance
        for instance in result[trial]['inc']: 
            context_inc = " ".join(instance['text']).encode("ascii", "ignore").split()
            matches_inc = {}

            # matched for inc
            for key in instance['matched'].keys():
                matches_inc[key] = instance['matched'][key]

            if  random_seed == True:       
                # sub-sample some negative from inclusion examples        
                for key in matches_inc.keys():
                    if key == entity_type:
                        for item in matches_inc[key]:
                            if item[2] == 0 and negative <= args.max_confidence_instance:        
                                confidence['negative'].append((trial, context_inc, item)) # remove all the other detection
                                negative += 1    
                # sub-sample some examples from exclusion examples       
                for key in matches_inc.keys():
                    if key == entity_type:
                        for item in matches_inc[key]:
                            if item[2] == 1 and positive <= args.max_confidence_instance:        
                                confidence['positive'].append((trial, context_inc, item)) # remove all the other detection
                                positive += 1
        
        for instance in result[trial]['exc']: 
            context_exc = " ".join(instance['text']).encode("ascii", "ignore").split()
            matches_exc = {}

            # matched for inc
            for key in instance['matched'].keys():
                matches_exc[key] = instance['matched'][key]
        
            # now choose the positive and negative example
            if random_seed == False:
                # sub-sample some negative from inclusion examples        
                for key in matches_exc.keys():
                    if key == entity_type:
                        for item in matches_exc[key]:
                            if item[2] == 0 and negative <= args.max_confidence_instance:        
                                confidence['negative'].append((trial, context_exc, item)) # remove all the other detection
                                negative += 1

                # sub-sample some examples from exclusion examples       
                for key in matches_exc.keys():
                    if key == entity_type:
                        for item in matches_exc[key]:
                            if item[2] == 1 and positive <= args.max_confidence_instance:        
                                confidence['positive'].append((trial, context_exc, item)) # remove all the other detection
                                positive += 1
    
    return examples, confidence

for epoch in range(1, args.epochs + 1):

    # train the nn model
    if not args.hard_em:
        
        if not args.stochastic:

            # initial update the nn with all the examples
            if args.stage == "M":
                    
                for k in range(args.multiple_M):    
                    train_Mstep_RNN(epoch)
                    print (" threshold: %f \n"%args.threshold)
                    # test after each epoch
                    test(epoch)
                    GetResult_valid(valid_data, vocab, args.prediction_file)

                # evaluate on the batch we sampled
                test(epoch)
                print (" threshold: %f \n"%args.threshold)
                GetResult_valid(valid_data, vocab, args.prediction_file)
                
                # save the model at each epoch, always use the newest one	
                torch.save(model.state_dict(), args.save_path)

            else: # only do the E step
                
                # train the pairwise potential model
                if args.learn_graph:
                
                    # get the label for each instance, train_data is fresh
                    if not result:
                        result = train_data
                  
                    result = train_Estep(epoch, result, args.entity_type, vocab)     
                    # train the graphical model with those labels
                    result = train_Mstep_Potential(epoch, result)
                    # reset the prior, message-passing again 
                    result = distribute_message_passing(result)

                    # save the data to disk so next time when readed, the marginal probability is changed
                    # slightly change the path to avoid re-write
                    # don't use it to save time for large dataset
            
                    file_path = os.path.join(args.dataroot, "new_"+args.train_data)
                    fp = open(file_path, "wb+")		
                    pickle.dump(result, fp) # save the new result to the disk  
                    fp.close()
                    
                    # directly read from the data
                    #args.file_path = result
                    #data_loader = CreateDataLoader(args)
                    #train_loader = data_loader.load_data()

    else:
      
        train_Mstep_RNN(epoch)
        print (" threshold: %f \n"%args.threshold)
        test(epoch) # initial test
        GetResult_valid(valid_data, vocab, args.prediction_file)  
        # save the model at each epoch, always use the newest one	
        torch.save(model.state_dict(), args.save_path)

        
# ignore this first
test_result, confidence = GetResult(test_data, args.entity_type, vocab)

# visualization the result using the visualizer
print ("writing the result to html \n")
make_html_file(test_result, args.visulization_html, args.entity_type)

# Load gene key data
with open(args.gene_key, 'rb') as f:
     gene_key = pickle.load(f)
     f.close()

print ("writing the confidence to html \n")
make_html_file_confidence(confidence, args.confidence_html, gene_key)

# write the the confidence to file for calculating the precision and recall
# make_csv_file(args.csv_file, confidence)

# save the final model
torch.save(model.state_dict(), args.save_path)
