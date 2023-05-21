'''
Data preprocessing, extract and build corpus from the given training data file
'''

import os
import re
import json
import pickle
import argparse
import codecs


class preprocess:

    def __init__(self, args):
        self.args = args

    def skipgram(self, sentence, i):
        '''
        Get the skipgram pairs from the given sentence
        :param: sentence: list of words, i: index of the cecnter word
        :return: skipgram list around the center word
        from the paper by Mikolov et.al: https://arxiv.org/pdf/1301.3781.pdf
        '''
        iword = sentence[i]
        left = sentence[max(i - self.args.window_size, 0): i]
        right = sentence[i + 1: i + self.args.window_size + 1]
        return iword, [self.args.unk for _ in range(self.args.window_size - len(left))] + left + right + [self.args.unk for _ in range(self.args.window_size - len(right))]
    
    def build(self):
        print('Building corpus...')
        step = 0
        self.word_count = {self.args.unk: 1}
        with codecs.open(os.path.join(self.args.datadir, self.args.filename), 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                sent = line.split()
                for word in sent:
                    self.word_count[word] = self.word_count.get(word, 0) + 1
        print("")
        self.idx2word = sorted(self.word_count, key=self.word_count.get, reverse=True)[: self.args.max_vocab - 1]
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        self.corpus = set([word for word in self.word2idx])

        # create new directory in datadir to save the corpus
        save_path = os.path.join(self.args.datadir, 'preprocessed')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pickle.dump(self.word_count, open(os.path.join(save_path, 'word_count.dat'), 'wb'))
        pickle.dump(self.corpus, open(os.path.join(save_path, 'corpus.dat'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(save_path, 'idx2word.dat'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(save_path, 'word2idx.dat'), 'wb'))
        print("build done")

    def convert(self):
        '''
        Convert the corpus to the trainable skipgram pairs
        '''
        print('Converting corpus to trainable data...')
        step = 1
        data = []
        with codecs.open(os.path.join(self.args.datadir, self.args.filename), 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                sent = []
                for word in line.split():
                    if word not in self.corpus:
                        word = self.args.unk
                    else: sent.append(word)
                for i in range(len(sent)):
                    iword, owords = self.skipgram(sent, i)
                    data.append((self.word2idx[iword], [self.word2idx[oword] for oword in owords]))

        print("")
        pickle.dump(data, open(os.path.join(self.args.datadir, 'train.dat'), 'wb'))
        print("convert done")
        
        
