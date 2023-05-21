from preprocess import *
from utils import *
from parse_args import *
from models import *
import argparse
import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
import torch
from torch.optim import Adam


if __name__ == '__main__':

    ############################## 1. Parse arguments ################################
    print('parsing arguments...')
    args = parse_args()


    if args.print_tofile == 'True':
        # Open files for stdout and stderr redirection
        stdout_file = open(os.path.join(args.ckpt_path, 'stdout.log'), 'w')
        stderr_file = open(os.path.join(args.ckpt_path, 'stderr.log'), 'w')
        # Redirect stdout and stderr to the files
        sys.stdout = stdout_file
        sys.stderr = stderr_file

    save_args_to_file(args, os.path.join(args.ckpt_path, 'args.json'))

    # print args
    print(args)

    ############################## 2. preprocess and loading the training data ################################
    if args.load == 'False':
        print('preprocessing the data...')
        pre = preprocess(args)
        pre.build()
        pre.convert()
        print('preprocessing finished')

    # load the training data
    print('loading the data...')
    idx2word = pickle.load(open(os.path.join(args.datadir, 'preprocessed', 'idx2word.dat'), 'rb'))
    word_count = pickle.load(open(os.path.join(args.datadir, 'preprocessed', 'word_count.dat'), 'rb'))


    ############################## 3. build the model ################################
    wf = np.array([word_count[word] for word in idx2word])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(args.ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2word)
    weights = wf

    model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim)
    modelpath = os.path.join(args.ckpt_path, '{}.pt'.format(args.name))
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=args.n_negs, weights=weights)
    sgns = torch.nn.DataParallel(sgns, device_ids=range(torch.cuda.device_count()))
    optim = Adam(sgns.parameters())


    ############################## 4. train the model ################################
    for epoch in range(1, args.epoch + 1):
        dataset = PermutedSubsampledCorpus(os.path.join(args.data_dir, 'train.dat'))
        dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True)
        total_batches = int(np.ceil(len(dataset) / args.mb))
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        for iword, owords in pbar:
            loss = sgns(iword, owords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())

    idx2vec = model.ivectors.weight.data.cpu().numpy()
    pickle.dump(idx2vec, open(os.path.join(args.datadir, 'idx2vec.dat'), 'wb'))
    torch.save(sgns.state_dict(), os.path.join(args.ckpt_path, '{}.pt'.format(args.name)))
    torch.save(optim.state_dict(), os.path.join(args.ckpt_path, '{}.optim.pt'.format(args.name)))

    
    
