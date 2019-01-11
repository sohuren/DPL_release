import nltk
import pickle
import argparse
from collections import Counter
import json
import numpy as np

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def lower_case_word(w):
    if w.upper() == w:
       return w
    else:
       return w.lower()

def build_vocab(file_name, threshold, wordvec_file):
    """Build a simple vocabulary wrapper."""

    counter = Counter()
    with open(file_name, 'r') as data_file:
        data = pickle.load(data_file)

    for i, key in enumerate(data.keys()):
        for instance in data[key]['inc']:
            tokens = instance['text']
            counter.update(tokens)
        for instance in data[key]['exc']:    
            tokens = instance['text']
            counter.update(tokens)
        
        if i % 1000 == 0:
            print("[%d/%d] Tokenized the description." %(i, len(data)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    
    count = len(vocab)
    print ("vocab size: %d"%count)

    # load the pre-trained wordvec file from bin
    embedding_vec = None
    # load data from the pre-trained word embedding
    fp = open(wordvec_file, "rb")
    wordvec = pickle.load(fp)
    fp.close() # wordvec is a dict
    print ("total candidate word vector:%s"%len(wordvec))
    word_keys = wordvec.keys()
    hit = 0
    cnt = 0 

    for word in vocab.word2idx.keys():
        # they just every case here, slightly different from what we have done
        word = lower_case_word(word) # convert the word to lower case
        if word in word_keys:
            vec = wordvec[word]
            vec_dim = vec.shape[0]
            if embedding_vec is None:
                embedding_vec = np.zeros((len(vocab), vec_dim), dtype=np.float32) # initialize with the zero vector
            else:    
                embedding_vec[vocab.word2idx[word], :] = vec
                hit += 1
                
        cnt += 1

        if cnt%10000 == 0:
            print (hit/float(count))

    return vocab, embedding_vec

def main(args):

    vocab, embedding_vec = build_vocab(file_name=args.data_path,threshold=args.threshold, wordvec_file=args.wordvec)                        
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

    embedding_path = args.embedding_path
    with open(embedding_path, 'wb') as f:
        pickle.dump(embedding_vec, f, pickle.HIGHEST_PROTOCOL)

    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


# build the vocab given the description
# feel free to change other data 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='./sample_data/train_multi.pkl',
                        help='path for train data file')
    parser.add_argument('--vocab_path', type=str, default='./sample_data/vocab_multi.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=3,
                        help='minimum word count threshold')
    parser.add_argument('--wordvec', type=str, default='./resources/wordvec.pkl',
                        help='the pre-trained wordvec from other group')
    parser.add_argument('--embedding_path', type=str, default='./sample_data/embedding_vec_multi.pkl',
                        help='save the pre-fetched word vector')
                        
                        
    args = parser.parse_args()
    main(args)  
