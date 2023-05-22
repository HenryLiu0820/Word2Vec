from preprocess import *
from utils import *
from parse_args import *
from models import *
import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch import LongTensor as LT
from torch import FloatTensor as FT

if __name__ == '__main__':

    ############################## 1. Parse arguments ################################
    print('parsing arguments...')
    args = parse_args()

    # create the checkpoint directory if it does not exist
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)


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

    modelpath = os.path.join(args.ckpt_path, '{}.pt'.format(args.name))
    sgns = SGNS(args, vocab_size=vocab_size)
    # sgns = torch.nn.DataParallel(sgns, device_ids=range(torch.cuda.device_count()))
    if args.cuda == 'True':
        sgns.cuda()
    optim = Adam(sgns.parameters(), lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)

    training_stats = {
        'epoch': [],
        'train_loss': [],
        'loss_o': [],
        'loss_n': []
    }

    # flush the output
    sys.stdout.flush()

    ############################## 4. train the model ################################
    for epoch in range(1, args.epoch + 1):
        dataset = PermutedSubsampledCorpus(os.path.join(args.datadir, 'train.dat'))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        total_batches = int(np.ceil(len(dataset) / args.batch_size))
        train_loss = 0
        loss_o = 0
        loss_n = 0
        step = 0
        print('Starting epoch: {}'.format(epoch))
        for _, (iword, owords) in enumerate(dataloader):
            step += 1
            nwords = FT(owords.size()[0], args.n_negs).uniform_(0, vocab_size - 1).long()
            iword = LT(iword)
            owords = LT(owords)
            if epoch == 1 and step <= 1:
                print('iword: {}, shape: {}'.format(iword, iword.shape))
                print('owords: {}, shape: {}'.format(owords, owords.shape))
                print('nwords: {}, shape: {}'.format(nwords, nwords.shape))
            sys.stdout.flush()
            if args.cuda == 'True':
                iword = iword.cuda()
                owords = owords.cuda()
                nwords = nwords.cuda()
            loss, score_o, score_n = sgns(iword, owords, nwords)
            train_loss += loss.item()
            loss_o += score_o.mean().item()
            loss_n += score_n.mean().item()
            optim.zero_grad()
            loss.backward()
            optim.step()

        train_loss /= step
        loss_o /= step
        loss_n /= step
        print('Finished Epoch: {}, train_loss: {}, loss_o: {}, loss_n: {}'.format(epoch, train_loss, loss_o, loss_n))

        # update training stats
        training_stats['epoch'].append(epoch)
        training_stats['train_loss'].append(train_loss)
        training_stats['loss_o'].append(loss_o)
        training_stats['loss_n'].append(loss_n)

        # flush the output
        sys.stdout.flush()


    np.save(os.path.join(args.ckpt_path, "training_stats.npy"), training_stats)

    idx2vec = sgns.get_embeddings('in')
    pickle.dump(idx2vec, open(os.path.join(args.datadir, 'idx2vec.dat'), 'wb'))
    torch.save(sgns.state_dict(), os.path.join(args.ckpt_path, '{}.pt'.format(args.name)))
    torch.save(optim.state_dict(), os.path.join(args.ckpt_path, '{}.optim.pt'.format(args.name)))
    
    if args.print_tofile == 'True':
        # Close the files to flush the output
        stdout_file.close()
        stderr_file.close()