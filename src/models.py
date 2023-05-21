# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn

from torch import LongTensor as LT
from torch import FloatTensor as FT

class SGNS(nn.Module):
    def __init__(self, args):
        super(SGNS, self).__init__()
        self.args = args
        self.in_embed = nn.Embedding(args.vocab_size, args.e_dim)
        self.out_embed = nn.Embedding(args.vocab_size, args.e_dim)
        self.n_negs = args.n_negs

    def forward(self, iword, owords):
        # convert data to long tensor and cuda settings
        iword = LT(iword)
        owords = LT(owords)
        nwords = FT(self.args.batch_size, owords.size()[1] * self.n_negs).uniform_(0, self.args.vocab_size - 1).long()
        if self.args.cuda == 'True':
            iword = iword.cuda()
            owords = owords.cuda()
            nwords = nwords.cuda()

        # get center word embedding
        embed_i = self.in_embed(iword).unsqueeze(2)
        # get context words embedding
        embed_o = self.out_embed(owords)
        # negative sampling
        embed_n = self.out_embed(nwords).neg()

        # calculate the loss
        score_o = t.bmm(embed_o, embed_i).squeeze().sigmoid().log().mean(1)
        score_n = t.bmm(embed_n, embed_i).squeeze().sigmoid().log().view(-1, owords.size()[1], self.n_negs).sum(2).mean(1)
        return -(score_o + score_n).mean()

    def get_embeddings(self, embedding_type='in'):
        '''
        return the embedding matrix
        '''
        if embedding_type == 'in':
            return self.in_embed.weight.data.cpu().numpy()
        elif embedding_type == 'out':
            return self.out_embed.weight.data.cpu().numpy()

