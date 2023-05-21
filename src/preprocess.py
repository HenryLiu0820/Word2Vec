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
        self.word_count = {self.args.unk: 1}
        with open(os.path.join(self.args.datadir, self.args.filename), 'r') as file:
            text = file.read().strip("\n")
        file.close()
        text = text.strip().lower()
        self.text = text.split(" ")
        print('Finished extracting words. Total words: {}'.format(len(text)))
        # word count
        for word in self.text:
            self.word_count[word] = self.word_count.get(word, 0) + 1
        
        print("")
        self.freq_dict = sorted(list(self.word_count.items()), key=lambda x: x[1], reverse=True)
        self.idx2word = [self.args.unk] + [word for word, _ in self.freq_dict]
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
        data = []
        for i in range(len(self.text)):
            iword, owords = self.skipgram(self.text, i)
            data.append((self.word2idx[iword], [self.word2idx[oword] for oword in owords]))

        print("")
        pickle.dump(data, open(os.path.join(self.args.datadir, 'train.dat'), 'wb'))
        print("convert done")
        
        
