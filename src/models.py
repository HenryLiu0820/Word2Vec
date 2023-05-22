# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from torch import LongTensor as LT
from torch import FloatTensor as FT

class SGNS(nn.Module):
    def __init__(self, args, vocab_size):
        super(SGNS, self).__init__()
        self.args = args
        self.in_embed = nn.Embedding(vocab_size, args.e_dim)
        self.out_embed = nn.Embedding(vocab_size, args.e_dim)
        self.n_negs = args.n_negs
        self.vocab_size = vocab_size

    def forward(self, iword, owords, nwords):
        '''
        shapes:
            iword: Tensor(batch_size)
            owords: Tensor(batch_size, context_size)
            nwords: Tensor(batch_size, context_size * n_negs)
        '''

        # get center word embedding, its shape should be Tensor(batch_size, e_dim, 1)
        embed_i = self.in_embed(iword).unsqueeze(2) # added a third dimension

        # get context words embedding, its shape should be Tensor(batch_size, context_size, e_dim)
        embed_o = self.out_embed(owords)

        # negative sampling, its shape should be Tensor(batch_size, context_size * n_negs, e_dim)
        embed_n = self.out_embed(nwords).neg()

        # calculate the loss
        score_o = torch.bmm(embed_o, embed_i).squeeze().sigmoid().log().mean(1)
        score_n = torch.bmm(embed_n, embed_i).squeeze().sigmoid().log().view(-1, owords.size()[1], self.n_negs).sum(2).mean(1)

        loss = -(score_o + score_n).mean()
        # print('loss: ', loss, loss.requires_grad)
        return loss, score_o, score_n

    def get_embeddings(self, embedding_type='in'):
        '''
        return the embedding matrix
        '''
        if embedding_type == 'in':
            return self.in_embed.weight.data.cpu().numpy()
        elif embedding_type == 'out':
            return self.out_embed.weight.data.cpu().numpy()

