import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
import os
from sklearn.metrics.pairwise import cosine_similarity

# load training data
with open("/scratch/zhliu/repos/Word2Vec/data/text8.txt", "r") as f:
    corpus = f.read().split()

# construct the dictionary
vocab = sorted(set(corpus))
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

# define the window size
K = 5

# sparce co-occurrence matrix
row = []
col = []
data = []
for i in range(len(corpus) - K):
    center = word2idx[corpus[i]] # center word index
    context = [word2idx[corpus[j]] for j in range(i + 1, i + K + 1)] # context word index
    for c in context:
        row.append(center)
        col.append(c)
        data.append(1)

# create a sparse matrix objext
co_matrix = coo_matrix((data, (row, col)), shape=(len(vocab), len(vocab)))
co_matrix = co_matrix.astype(np.float32)

# perform SVD
U, S, Vt = svds(co_matrix, k=100)
vec_svd = U

# eval dataset
df = pd.read_csv("/scratch/zhliu/repos/Word2Vec/data/wordsim353_agreed.txt", sep="\t", header=None)
word1 = df[1].values 
word2 = df[2].values

# calculate the cosine similarity
sim_svd = []
for w1, w2 in zip(word1, word2):
    if w1 in word2idx and w2 in word2idx:
        v1 = vec_svd[word2idx[w1]]
        v2 = vec_svd[word2idx[w2]]
        sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
        sim_svd.append(sim)
    else:   # if the word is not in the vocabulary, set the similarity to 0
        sim_svd.append(0)

# save the result
df["sim_svd"] = sim_svd
df.to_csv(os.path.join('/scratch/zhliu/repos/Word2Vec/data', '2020212296.txt'), sep="\t", header=None, index=False)