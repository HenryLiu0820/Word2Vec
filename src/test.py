import pickle
import numpy as np
import torch
import os
from utils import *



if __name__ == '__main__':
    datadir = '/scratch/zhliu/repos/Word2Vec/data/preprocessed'
    ckptdir = '/scratch/zhliu/checkpoints/sgns/epoch_10/batch_size_1024/lr_0.001/weight_decay_1e-4'
    ############################## 1. get word embeddings ################################
    print('loading the word embeddings...')
    word2idx = pickle.load(open(os.path.join(datadir, 'word2idx.dat'), 'rb'))
    vocabulary = pickle.load(open(os.path.join(datadir, 'vocabulary.dat'), 'rb'))
    text_idx = pickle.load(open(os.path.join(datadir, 'text_idx.dat'), 'rb'))
    words_freq = pickle.load(open(os.path.join(datadir, 'words_freq.dat'), 'rb'))
    vocab_size = len(vocabulary)
    embeddings = torch.load(os.path.join(ckptdir, 'sgns.pt'))['in_embed.weight'].cpu().numpy()
    print('embeddings shape: {}'.format(embeddings.shape))

    ############################## 2. calculate top-5 closest words of the given word ################################
    word = 'smart'
    word_idx = word2idx[word]
    dist, max_idx = distance_matrix(word_idx, embeddings)

    print('the top-5 closest words to {} are:'.format(word))
    most_similar(word, embeddings, vocabulary, word2idx)

