import json
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
import os
import math


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

def calc_err(embedding, num, args):
    data = []
    X = []
    lines = []
    test_dir = os.path.join(args.datadir, 'wordsim353_agreed.txt')
    with open(test_dir, 'r') as f:
        for line in f.readlines():
            lines.append(line.strip())
            data.append(line.split())
            X.append((num.get(data[-1][1], -1), num.get(data[-1][2], -1)))
    X = np.array(X)
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)  # 归一化
    pos = np.any(X == -1, axis=1)  # 带-1的pair
    embedding_x = embedding[X[:, 0]]
    embedding_y = embedding[X[:, 1]]

    score = np.sum(embedding_x * embedding_y, axis=1)
    score[pos] = 0
    avg_err = 0
    score = [0 if math.isnan(x) else x for x in score]
    for i, sim in enumerate(score):
        avg_err += abs(float(data[i][-1]) - (sim + 1) * 5)
    avg_err /= len(score)
    print("avg test err: %f" % (avg_err))
    return avg_err