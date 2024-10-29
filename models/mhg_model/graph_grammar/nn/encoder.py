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
__date__ = "Aug 9 2018"


import abc
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import List


class EncoderBase(nn.Module):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, in_seq):
        ''' forward model

        Parameters
        ----------
        in_seq_emb : Variable, shape (batch_size, max_len, input_dim)

        Returns
        -------
        hidden_seq_emb : Tensor, shape (batch_size, max_len, 1 + bidirectional, hidden_dim)
        '''
        pass

    @abc.abstractmethod
    def init_hidden(self):
        ''' initialize the hidden states
        '''
        pass


class GRUEncoder(EncoderBase):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 bidirectional: bool, dropout: float, batch_size: int, use_gpu: bool,
                 no_dropout=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.model = nn.GRU(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional,
                            dropout=self.dropout if not no_dropout else 0)
        if self.use_gpu:
            self.model.cuda()
        self.init_hidden()


    def init_hidden(self):
        self.h0 = torch.zeros(((self.bidirectional + 1) * self.num_layers,
                               self.batch_size,
                               self.hidden_dim),
                              requires_grad=False)
        if self.use_gpu:        
            self.h0 = self.h0.cuda()

    def forward(self, in_seq_emb):
        ''' forward model

        Parameters
        ----------
        in_seq_emb : Tensor, shape (batch_size, max_len, input_dim)

        Returns
        -------
        hidden_seq_emb : Tensor, shape (batch_size, max_len, 1 + bidirectional, hidden_dim)
        '''
        max_len = in_seq_emb.size(1)
        hidden_seq_emb, self.h0 = self.model(
            in_seq_emb, self.h0)
        hidden_seq_emb = hidden_seq_emb.view(self.batch_size,
                                             max_len,
                                             1 + self.bidirectional,
                                             self.hidden_dim)
        return hidden_seq_emb


class LSTMEncoder(EncoderBase):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 bidirectional: bool, dropout: float, batch_size: int, use_gpu: bool,
                 no_dropout=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.model = nn.LSTM(input_size=self.input_dim,
                             hidden_size=self.hidden_dim,
                             num_layers=self.num_layers,
                             batch_first=True,
                             bidirectional=self.bidirectional,
                             dropout=self.dropout if not no_dropout else 0)
        if self.use_gpu:
            self.model.cuda()
        self.init_hidden()

    def init_hidden(self):
        self.h0 = torch.zeros(((self.bidirectional + 1) * self.num_layers,
                               self.batch_size,
                               self.hidden_dim),
                              requires_grad=False)
        self.c0 = torch.zeros(((self.bidirectional + 1) * self.num_layers,
                               self.batch_size,
                               self.hidden_dim),
                              requires_grad=False)
        if self.use_gpu:        
            self.h0 = self.h0.cuda()
            self.c0 = self.c0.cuda()

    def forward(self, in_seq_emb):
        ''' forward model

        Parameters
        ----------
        in_seq_emb : Tensor, shape (batch_size, max_len, input_dim)

        Returns
        -------
        hidden_seq_emb : Tensor, shape (batch_size, max_len, 1 + bidirectional, hidden_dim)
        '''
        max_len = in_seq_emb.size(1)
        hidden_seq_emb, (self.h0, self.c0) = self.model(
            in_seq_emb, (self.h0, self.c0))
        hidden_seq_emb = hidden_seq_emb.view(self.batch_size,
                                             max_len,
                                             1 + self.bidirectional,
                                             self.hidden_dim)
        return hidden_seq_emb


class FullConnectedEncoder(EncoderBase):

    def __init__(self, input_dim: int, hidden_dim: int, max_len: int, hidden_dim_list: List[int],
                 batch_size: int, use_gpu: bool):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.hidden_dim_list = hidden_dim_list
        self.use_gpu = use_gpu
        in_out_dim_list = [input_dim * max_len] + list(hidden_dim_list) + [hidden_dim]
        self.linear_list = nn.ModuleList(
            [nn.Linear(in_out_dim_list[each_idx], in_out_dim_list[each_idx + 1])\
             for each_idx in range(len(in_out_dim_list) - 1)])

    def forward(self, in_seq_emb):
        ''' forward model

        Parameters
        ----------
        in_seq_emb : Tensor, shape (batch_size, max_len, input_dim)

        Returns
        -------
        hidden_seq_emb : Tensor, shape (batch_size, max_len, 1 + bidirectional, hidden_dim)
        '''
        batch_size = in_seq_emb.size(0)
        x = in_seq_emb.view(batch_size, -1)
        for each_linear in self.linear_list:
            x = F.relu(each_linear(x))
        return x.view(batch_size, 1, -1)

    def init_hidden(self):
        pass
