
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import h5py
import os
import numpy as np
import random

import torch
import torch.utils.data as data

import multiprocessing

def get_npy_data(ix, fc_file, att_file, use_att):
    if use_att == True:
        return (np.load(fc_file), np.load(att_file)['feat'], ix)
    else:
        return (np.load(fc_file), np.zeros((1,1,1)), ix)





class DataLoader(data.Dataset):
    def reset_iterator(self,split):
        pass

    def get_vocab_size(self):
        pass

    def get_vocab(self):
        pass

    def get_seq_length(self):
        pass

    def __init__(self, opt):
        self.opt=opt
        self.batch_size=opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.use_att = getattr(opt, 'use_att', True)

        # load the json file which contains additional information about the dataset


        print('DataLoader loading json file: ', opt.input_json)

        self.info = json.load(open(self.opt.input_json))

        self.ix_to_word = self.info['ix_to_word']

