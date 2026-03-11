import os

import torch

from .lstm import CustomRNNModel


def get_lstm_model(mode_name, pretrained=True, dropout=None):
    corpus = Corpus('/data/leizhao_data/ptb')
    vocab_size = len(corpus.dictionary)
    if mode_name == 'lstm':
        if dropout is not None:
            model = CustomRNNModel(vocab_size=vocab_size, ninp=650, nhid=650, nlayers=2, dropout=dropout)
        else:
            model = CustomRNNModel(vocab_size=vocab_size, ninp=650, nhid=650, nlayers=2)
    else:
        print("unrecognized network")
        exit(0)

    return model


class Dictionary(object):
    '''
        Vocabulary of the whole dataset, including train, valid and test
    '''
    def __init__(self):
        self.word2idx = {} # word: index
        self.idx2word = [] # position(index): word

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    '''
        train: tokenized training data, each word is represented by its ID in vocabulary
        valid: tokenized validation data, each word is represented by its ID in vocabulary
        test : tokenized testing data, each word is represented by its ID in vocabulary
    '''
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                # line to list of token + eos
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def batchify(data, bsz):
    nbatch = data.size(0) // bsz                 # Work out how cleanly we can divide the dataset into bsz parts.
    data = data.narrow(0, 0, nbatch * bsz)       # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.view(bsz, -1).t().contiguous()   # Evenly divide the data across the bsz batches.
    return data


def prepare_ptb_data(batch_size, workers):
    corpus = Corpus('/data/leizhao_data/ptb')
    train_data = batchify(corpus.train, batch_size)
    val_data = batchify(corpus.valid, batch_size)
    test_data = batchify(corpus.test, batch_size)

    return train_data, test_data