from preprocess import *
from utils import *
from parse_args import *
import argparse
import os
import sys
import numpy as np


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
    if args.load == 'True':
        print('preprocessing the data...')
        pre = preprocess(args)
        pre.build()
        pre.convert()
        print('preprocessing finished')

    # load the training data
    print('loading the data...')
    idx2word = pickle.load(open(os.path.join(args.datadir, 'preprocessed', 'idx2word.dat'), 'rb'))
    word_count = pickle.load(open(os.path.join(args.datadir, 'preprocessed', 'word_count.dat'), 'rb'))
    wf = np.array([word_count[word] for word in idx2word])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(args.ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2word)

    

    
    
