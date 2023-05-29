from preprocess import *
from utils import *
from parse_args import *
from models import *
import os
import sys
import numpy as np
import pickle
import random
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch import LongTensor as LT
from torch import FloatTensor as FT
from torch.utils.data import DataLoader

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
        print('preprocessing finished')

    # load the training data
    print('loading the data...')
    word2idx = pickle.load(open(os.path.join(args.datadir, 'preprocessed', 'word2idx.dat'), 'rb'))
    vocabulary = pickle.load(open(os.path.join(args.datadir, 'preprocessed', 'vocabulary.dat'), 'rb'))
    text_idx = pickle.load(open(os.path.join(args.datadir, 'preprocessed', 'text_idx.dat'), 'rb'))
    words_freq = pickle.load(open(os.path.join(args.datadir, 'preprocessed', 'words_freq.dat'), 'rb'))
    vocab_size = len(vocabulary)
    print('loading finished, vocabulary size: {}'.format(vocab_size))


    ############################## 3. build the model ################################
    

    modelpath = os.path.join(args.ckpt_path, '{}.pt'.format(args.name))
    sgns = SGNS(args, vocab_size=vocab_size)
    sgns.train()
    # sgns = torch.nn.DataParallel(sgns, device_ids=range(torch.cuda.device_count()))
    if args.cuda == 'True':
        sgns.cuda()
    optim = Adam(sgns.parameters(), lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)

    training_stats = {
        'epoch': [],
        'train_loss': [],
        'loss_o': [],
        'loss_n': [],
        'avg_err': []
    }

    # flush the output
    sys.stdout.flush()

    ############################## 4. train the model ################################
    for epoch in range(1, args.epoch + 1):
        dataset = SGNSDataset(word2idx, text_idx, words_freq, args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print('dataset size: {}'.format(len(dataset)))
        print('batch num: {}'.format(len(dataloader)))
        train_loss = 0
        loss_o = 0
        loss_n = 0
        step = 0
        avg_err = 0
        print('Starting epoch: {}'.format(epoch))
        for _, (iword, owords, nwords) in enumerate(dataloader):
            step += 1
            if epoch == 1 and step <= 1:
                print('iword: {}, shape: {}, max: {}'.format(iword, iword.shape, iword.max()))
                print('owords: {}, shape: {}, max: {}'.format(owords, owords.shape, owords.max()))
                print('nwords: {}, shape: {}, max: {}'.format(nwords, nwords.shape, nwords.max()))
            sys.stdout.flush()
            optim.zero_grad()
            if args.cuda == 'True':
                iword, owords, nwords = iword.cuda(), owords.cuda(), nwords.cuda()
            score_o, score_n = sgns(iword, owords, nwords)
            # loss = torch.mean(score_o + score_n)
            loss = score_o + score_n
            train_loss += loss.item()
            loss_o += score_o.item()
            loss_n += score_n.item()
            loss.backward()
            optim.step()

            # print the training stats
            if step % 1000 == 0:
                in_embed = sgns.get_embeddings('in')
                avg_err = calc_err(in_embed, word2idx, args)
                print('Epoch: {}, Step: {}, train_loss: {}, score_o: {}, score_n: {}, avg_error: {}'.format(epoch, step, loss.item(), score_o.item(), score_n.item(), avg_err))
                sys.stdout.flush()

                # test the embedding
                test_list = ['rain', 'utah', 'computer', 'brother', 'house']
                for test_w in test_list:
                    print('Words closest to chosen word: {}'.format(test_w))
                    most_similar(test_w, in_embed, vocabulary, word2idx)

        train_loss /= step
        loss_o /= step
        loss_n /= step
        print('Finished Epoch: {}, train_loss: {}, loss_o: {}, loss_n: {}, avg error: {}'.format(epoch, train_loss, loss_o, loss_n, avg_err))

        # update training stats
        training_stats['epoch'].append(epoch)
        training_stats['train_loss'].append(train_loss)
        training_stats['loss_o'].append(loss_o)
        training_stats['loss_n'].append(loss_n)
        training_stats['avg_err'] = avg_err

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