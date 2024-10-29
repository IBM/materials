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
from torch import nn


class DecoderBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_dict = {}

    @abc.abstractmethod
    def forward_one_step(self, tgt_emb_in):
        ''' one-step forward model

        Parameters
        ----------
        tgt_emb_in : Tensor, shape (batch_size, input_dim)

        Returns
        -------
        Tensor, shape (batch_size, hidden_dim)
        '''
        tgt_emb_out = None
        return tgt_emb_out

    @abc.abstractmethod
    def init_hidden(self):
        ''' initialize the hidden states
        '''
        pass

    @abc.abstractmethod
    def feed_hidden(self, hidden_dict_0):
        for each_hidden in self.hidden_dict.keys():
            self.hidden_dict[each_hidden][0] = hidden_dict_0[each_hidden]


class GRUDecoder(DecoderBase):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, batch_size: int, use_gpu: bool,
                 no_dropout=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.model = nn.GRU(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=False,
                            dropout=self.dropout if not no_dropout else 0
        )
        if self.use_gpu:
            self.model.cuda()
        self.init_hidden()

    def init_hidden(self):
        self.hidden_dict['h'] = torch.zeros((self.num_layers,
                                             self.batch_size,
                                             self.hidden_dim),
                                            requires_grad=False)
        if self.use_gpu:
            self.hidden_dict['h'] = self.hidden_dict['h'].cuda()

    def forward_one_step(self, tgt_emb_in):
        ''' one-step forward model

        Parameters
        ----------
        tgt_emb_in : Tensor, shape (batch_size, input_dim)

        Returns
        -------
        Tensor, shape (batch_size, hidden_dim)
        '''
        tgt_emb_out, self.hidden_dict['h'] \
            = self.model(tgt_emb_in.view(self.batch_size, 1, -1),
                         self.hidden_dict['h'])
        return tgt_emb_out


class LSTMDecoder(DecoderBase):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, batch_size: int, use_gpu: bool,
                 no_dropout=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.model = nn.LSTM(input_size=self.input_dim,
                             hidden_size=self.hidden_dim,
                             num_layers=self.num_layers,
                             batch_first=True,
                             bidirectional=False,
                             dropout=self.dropout if not no_dropout else 0)
        if self.use_gpu:
            self.model.cuda()
        self.init_hidden()

    def init_hidden(self):
        self.hidden_dict['h'] = torch.zeros((self.num_layers,
                                             self.batch_size,
                                             self.hidden_dim),
                                            requires_grad=False)
        self.hidden_dict['c'] = torch.zeros((self.num_layers,
                                             self.batch_size,
                                             self.hidden_dim),
                                            requires_grad=False)
        if self.use_gpu:
            for each_hidden in self.hidden_dict.keys():
                self.hidden_dict[each_hidden] = self.hidden_dict[each_hidden].cuda()

    def forward_one_step(self, tgt_emb_in):
        ''' one-step forward model

        Parameters
        ----------
        tgt_emb_in : Tensor, shape (batch_size, input_dim)

        Returns
        -------
        Tensor, shape (batch_size, hidden_dim)
        '''
        tgt_hidden_out, self.hidden_dict['h'], self.hidden_dict['c'] \
            = self.model(tgt_emb_in.view(self.batch_size, 1, -1),
                         self.hidden_dict['h'], self.hidden_dict['c'])
        return tgt_hidden_out
