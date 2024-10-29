#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Rhizome
# Version beta 0.0, August 2023
# Property of IBM Research, Accelerated Discovery
#

"""
PLEASE NOTE THIS IMPLEMENTATION INCLUDES THE ORIGINAL SOURCE CODE (AND SOME ADAPTATIONS)
OF THE MHG IMPLEMENTATION OF HIROSHI KAJINO AT IBM TRL ALREADY PUBLICLY AVAILABLE. 
THIS MIGHT INFLUENCE THE DECISION OF THE FINAL LICENSE SO CAREFUL CHECK NEEDS BE DONE. 
"""

""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2018"
__version__ = "0.1"
__date__ = "Apr 18 2018"

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


def left_padding(sentence_list, max_len, pad_idx=-1, inverse=False):
    ''' pad left

    Parameters
    ----------
    sentence_list : list of sequences of integers
    max_len : int
        maximum length of sentences.
        if a sentence is shorter than `max_len`, its left part is padded.
    pad_idx : int
        integer for padding
    inverse : bool
        if True, the sequence is inversed.

    Returns
    -------
    List of torch.LongTensor
        each sentence is left-padded.
    '''
    max_in_list = max([len(each_sen) for each_sen in sentence_list])
        
    if max_in_list > max_len:
        raise ValueError('`max_len` should be larger than the maximum length of input sequences, {}.'.format(max_in_list))

    if inverse:
        return [torch.LongTensor([pad_idx] * (max_len - len(each_sen)) + each_sen[::-1]) for each_sen in sentence_list]
    else:
        return [torch.LongTensor([pad_idx] * (max_len - len(each_sen)) + each_sen) for each_sen in sentence_list]


def right_padding(sentence_list, max_len, pad_idx=-1):
    ''' pad right

    Parameters
    ----------
    sentence_list : list of sequences of integers
    max_len : int
        maximum length of sentences.
        if a sentence is shorter than `max_len`, its right part is padded.
    pad_idx : int
        integer for padding

    Returns
    -------
    List of torch.LongTensor
        each sentence is right-padded.
    '''
    max_in_list = max([len(each_sen) for each_sen in sentence_list])
    if max_in_list > max_len:
        raise ValueError('`max_len` should be larger than the maximum length of input sequences, {}.'.format(max_in_list))

    return [torch.LongTensor(each_sen + [pad_idx] * (max_len - len(each_sen))) for each_sen in sentence_list]


class HRGDataset(Dataset):

    '''
    A class of HRG data
    '''

    def __init__(self, hrg, prod_rule_seq_list, max_len, target_val_list=None, inversed_input=False):
        self.hrg = hrg
        self.left_prod_rule_seq_list = left_padding(prod_rule_seq_list,
                                                    max_len,
                                                    inverse=inversed_input)
            
        self.right_prod_rule_seq_list = right_padding(prod_rule_seq_list, max_len)
        self.inserved_input = inversed_input
        self.target_val_list = target_val_list
        if target_val_list is not None:
            if len(prod_rule_seq_list) != len(target_val_list):
                raise ValueError(f'prod_rule_seq_list and target_val_list have inconsistent lengths: {len(prod_rule_seq_list)}, {len(target_val_list)}')

    def __len__(self):
        return len(self.left_prod_rule_seq_list)

    def __getitem__(self, idx):
        if self.target_val_list is not None:
            return self.left_prod_rule_seq_list[idx], self.right_prod_rule_seq_list[idx], np.float32(self.target_val_list[idx])
        else:
            return self.left_prod_rule_seq_list[idx], self.right_prod_rule_seq_list[idx]

    @property
    def vocab_size(self):
        return self.hrg.num_prod_rule

def batch_padding(each_batch, batch_size, padding_idx):
    num_pad = batch_size - len(each_batch[0])
    if num_pad:
        each_batch[0] = torch.cat([each_batch[0],
                                   padding_idx * torch.ones((batch_size - len(each_batch[0]),
                                                             len(each_batch[0][0])), dtype=torch.int64)], dim=0)
        each_batch[1] = torch.cat([each_batch[1],
                                   padding_idx * torch.ones((batch_size - len(each_batch[1]),
                                                             len(each_batch[1][0])), dtype=torch.int64)], dim=0)
    return each_batch, num_pad
