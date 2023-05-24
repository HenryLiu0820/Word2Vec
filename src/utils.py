import json
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F


def save_args_to_file(args, output_file_path):
    with open(output_file_path, "w") as output_file:
        json.dump(vars(args), output_file, indent=4)

def one_hot(mat, vocab_size):
    '''
    convert the last dimension with index to one-hot vector
    '''

    return F.one_hot(mat, num_classes=vocab_size)


def cosine_distance(v1, v2):
    '''
    calculate the cosine distance of two word vectors
    '''
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def distance_matrix(w_idx, mat):
    '''
    calculate the distance of a given word to all the words in the word embedding matrix
    input:
        w_idx: an index of a word -- scalar
        mat: a word embedding matrix -- numpy array(word_num, embed_dim)

    output:
        mat: a distance matrix -- (word_num, 1)
        top5idx: the indices 5 of the word with the maximum distance to the given word -- numpy array(5)
    '''
    w_vec = mat[w_idx]
    mat = np.dot(mat, w_vec) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(w_vec))
    max_idx = np.argsort(mat)[-5:]

    return mat, max_idx

def most_similar(word, word_vector, words, num, topn=5):
    '''
    Get the most similar words for the given word, using trained embedding
    Used for testing (during training)
    '''
    word = word.lower()
    if word not in words:
        print('word %s not in vocabulary' % word)
        return
    word_vector = word_vector / np.linalg.norm(word_vector, axis=1, keepdims=True)
    word_id = num[word]
    word_vec = word_vector[word_id]
    word_vec = word_vec / np.linalg.norm(word_vec)
    sim = np.dot(word_vector, word_vec)
    sim_idx = np.argsort(-sim)
    sim_idx = sim_idx[1:topn + 1]
    for idx in sim_idx:
        print('word: %s, similarity %f' % (words[idx], sim[idx]))
