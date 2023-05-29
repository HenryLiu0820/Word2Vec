'''
Data preprocessing, extract and build corpus from the given training data file
'''

import os
import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import collections
import numpy as np
from tqdm import tqdm
import time
import pandas

class preprocess:

    def __init__(self, args):
        self.args = args

    def build(self):
        print('Building vocabulary...')
        self.word_count = {self.args.unk: 1}
        with open(os.path.join(self.args.datadir, self.args.filename), 'r') as file:
            corpus = file.read().strip("\n")
        file.close()
        corpus = corpus.strip().lower()
        corpus = corpus.split(" ")
        print('Finished extracting words. Total words: {}'.format(len(corpus)))

        # build the corpus
        counter = collections.Counter(corpus)
        # get only the most common max_vocab words (rest are treated as unknown), sorted by frequency
        most_freq = counter.most_common(self.args.max_vocab)
        vocabulary = [word for word, _ in most_freq]
        words_freq = [freq for word, freq in most_freq]
        
        vocabulary = vocabulary[: len(vocabulary) - 1]
        vocabulary.append(self.args.unk)
        words_freq = words_freq[: len(vocabulary) - 1]
        words_freq.append(len(corpus) - sum(words_freq))

        word2idx = {word: idx for idx, word in enumerate(vocabulary)}
        idx2word = {idx: word for idx, word in enumerate(vocabulary)}

        # convert all the words in the text to their corresponding index, 
        # if not in word2idx, then use the index of unknown token
        text_idx = [word2idx.get(word, 0) for word in corpus]

        # downsample the frequent words
        words_freq = np.array(words_freq) / len(corpus)
        sub_sampled_data = []
        sub_freq = (np.sqrt(words_freq / 0.001) + 1) * 0.001 / words_freq
        sub_sampled_data = [word for word in tqdm(text_idx) if np.random.rand() < sub_freq[word]]
        text_idx = np.array(sub_sampled_data)

        words_freq = words_freq ** (3. / 4)
        words_freq = words_freq / np.sum(words_freq)

        # create new directory in datadir to save the corpus
        save_path = os.path.join(self.args.datadir, 'preprocessed')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pickle.dump(word2idx, open(os.path.join(save_path, 'word2idx.dat'), 'wb'))          # find index for the given word
        pickle.dump(vocabulary, open(os.path.join(save_path, 'vocabulary.dat'), 'wb'))      # find word for given index
        pickle.dump(text_idx, open(os.path.join(save_path, 'text_idx.dat'), 'wb'))          # find indices for the given corpus
        pickle.dump(words_freq, open(os.path.join(save_path, 'words_freq.dat'), 'wb'))      # find frequency for the given word

        print('Finished building corpus. Total words in the corpus: {}'.format(len(vocabulary)))

        return word2idx, vocabulary, text_idx, words_freq

        
        
class SGNSDataset(Dataset):
    def __init__(self, word2idx, text_idx, words_freq, args):
        super().__init__()
        self.text_idx = torch.tensor(text_idx, dtype=torch.int32)
        self.word2idx = word2idx
        self.length = len(text_idx)
        self.words_freq = torch.tensor(words_freq, dtype=torch.float32)
        self.window_size = args.window_size
        self.n_negs = args.n_negs
        self.args = args

    def __len__(self):
        return len(self.text_idx)

    def __getitem__(self, idx):
        """
        get the center word, context words and negative samples of a given index

        Get the skipgram pairs from the given sentence
        from the paper by Mikolov et.al: https://arxiv.org/pdf/1301.3781.pdf
        """
        iword = self.text_idx[idx]  # index of the center word

        left = list(range(idx - self.window_size, idx))
        right = list(range(idx + 1, idx + self.window_size + 1))
        owords_idx = [i % self.length for i in left + right]
        # pad the left and right with unknown tokens
        owords = self.text_idx[owords_idx]
        
        # negative sampling
        nwords = torch.multinomial(self.words_freq, self.n_negs, True)
        return iword, owords, nwords

class IterDataset(Dataset):
    def __init__(self, words_freq, co_matrix_dir='./co_matrix.txt', window_size=2, ratio=2, batch_size=256):
        super().__init__()
        self.words_freq = words_freq
        print('loading co_matrix...')
        ti = time.time()
        self.co_matrix = pandas.read_table(co_matrix_dir, header=None, sep=' ', dtype=np.int32).values
        # 将稀疏矩阵按中心词排序即[:,0]
        self.co_matrix = self.co_matrix[np.argsort(self.co_matrix[:, 0])]
        self.freq = self.co_matrix[:, 2].astype(float)  # 单独存储词频需要float32
        # 计算每个中心词最后一个位置的下标, 以便后续在负采样时快速定位, 例如[0, 0, 0, 1, 1, 1, 2, 2, 2] -> [3, 6, 9]
        self.co_matrix_idx = np.cumsum(np.unique(self.co_matrix[:, 0], return_counts=True)[1])
        # 计算每个中心词所有上下文的总词频
        cum_sum = np.cumsum(self.freq)
        self.co_matrix_freq_sum = cum_sum[self.co_matrix_idx - 1]
        self.co_matrix_freq_sum[1:] = self.co_matrix_freq_sum[1:] - self.co_matrix_freq_sum[:-1]
        # 当索引为-1时, 会取到最后一个位置=0
        self.co_matrix_idx = np.insert(self.co_matrix_idx, len(self.co_matrix_idx), 0)
        print('co_matrix loaded, time cost: ', time.time() - ti)
        self.window_size = window_size
        self.ratio = ratio
        self.batch_size = batch_size

    def __iter__(self):
        """
        根据稀疏共现矩阵进行正采样，基于词频进行负采样
        每次返回一批数据，包含中心词，上下文词，负采样词

        :return:
        """
        for _ in range(len(self.words_freq)):
            w = np.random.choice(len(self.words_freq), self.batch_size, p=self.words_freq)
            positive_c = []
            # self.co_matrix_idx中存储的是稀疏矩阵中心词最后一个位置的下标,
            for i in w:
                l, r = self.co_matrix_idx[i - 1], self.co_matrix_idx[i]
                if self.freq[l] >= 1:
                    self.freq[l:r] = self.freq[l:r] / self.co_matrix_freq_sum[i]
                    self.freq[l:r] = np.power(self.freq[l:r], 0.75)
                    self.freq[l:r] = self.freq[l:r] / self.freq[l:r].sum()
                positive_c.append(np.random.choice(self.co_matrix[l:r, 1], size=self.window_size * 2, p=self.freq[l:r]))
            positive_c = np.array(positive_c)
            negative_c = np.random.choice(len(self.words_freq),
                                          size=(self.batch_size, int(self.window_size * 2 * self.ratio)),
                                          p=self.words_freq)
            yield torch.cuda.LongTensor(w), torch.cuda.LongTensor(positive_c), torch.cuda.LongTensor(negative_c)
