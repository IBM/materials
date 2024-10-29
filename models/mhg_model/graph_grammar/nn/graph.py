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
__date__ = "Jan 1 2018"

import numpy as np
import torch
import torch.nn.functional as F
from graph_grammar.graph_grammar.hrg import ProductionRuleCorpus
from torch import nn
from torch.autograd import Variable

class MolecularProdRuleEmbedding(nn.Module):
    
    ''' molecular fingerprint layer
    '''

    def __init__(self, prod_rule_corpus, layer2layer_activation, layer2out_activation,
                 out_dim=32, element_embed_dim=32,
                 num_layers=3, padding_idx=None, use_gpu=False):
        super().__init__()
        if padding_idx is not None:
            assert padding_idx == -1, 'padding_idx must be -1.'
        self.prod_rule_corpus = prod_rule_corpus
        self.layer2layer_activation = layer2layer_activation
        self.layer2out_activation = layer2out_activation
        self.out_dim = out_dim
        self.element_embed_dim = element_embed_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.use_gpu = use_gpu

        self.layer2layer_list = []
        self.layer2out_list = []

        if self.use_gpu:
            self.atom_embed = torch.randn(self.prod_rule_corpus.num_edge_symbol,
                                          self.element_embed_dim, requires_grad=True).cuda()
            self.bond_embed = torch.randn(self.prod_rule_corpus.num_node_symbol,
                                          self.element_embed_dim, requires_grad=True).cuda()
            self.ext_id_embed = torch.randn(self.prod_rule_corpus.num_ext_id,
                                            self.element_embed_dim, requires_grad=True).cuda()
            for _ in range(num_layers):
                self.layer2layer_list.append(nn.Linear(self.element_embed_dim, self.element_embed_dim).cuda())
                self.layer2out_list.append(nn.Linear(self.element_embed_dim, self.out_dim).cuda())
        else:
            self.atom_embed = torch.randn(self.prod_rule_corpus.num_edge_symbol,
                                          self.element_embed_dim, requires_grad=True)
            self.bond_embed = torch.randn(self.prod_rule_corpus.num_node_symbol,
                                          self.element_embed_dim, requires_grad=True)
            self.ext_id_embed = torch.randn(self.prod_rule_corpus.num_ext_id,
                                            self.element_embed_dim, requires_grad=True)
            for _ in range(num_layers):
                self.layer2layer_list.append(nn.Linear(self.element_embed_dim, self.element_embed_dim))
                self.layer2out_list.append(nn.Linear(self.element_embed_dim, self.out_dim))


    def forward(self, prod_rule_idx_seq):
        ''' forward model for mini-batch

        Parameters
        ----------
        prod_rule_idx_seq : (batch_size, length)

        Returns
        -------
        Variable, shape (batch_size, length, out_dim)
        '''
        batch_size, length = prod_rule_idx_seq.shape
        if self.use_gpu:
            out = Variable(torch.zeros((batch_size, length, self.out_dim))).cuda()
        else:
            out = Variable(torch.zeros((batch_size, length, self.out_dim)))
        for each_batch_idx in range(batch_size):
            for each_idx in range(length):
                if int(prod_rule_idx_seq[each_batch_idx, each_idx]) == len(self.prod_rule_corpus.prod_rule_list):
                    continue
                else:
                    each_prod_rule = self.prod_rule_corpus.prod_rule_list[int(prod_rule_idx_seq[each_batch_idx, each_idx])]
                    layer_wise_embed_dict = {each_edge: self.atom_embed[
                        each_prod_rule.rhs.edge_attr(each_edge)['symbol_idx']]
                                             for each_edge in each_prod_rule.rhs.edges}
                    layer_wise_embed_dict.update({each_node: self.bond_embed[
                        each_prod_rule.rhs.node_attr(each_node)['symbol_idx']]
                                                  for each_node in each_prod_rule.rhs.nodes})
                    for each_node in each_prod_rule.rhs.nodes:
                        if 'ext_id' in each_prod_rule.rhs.node_attr(each_node):
                            layer_wise_embed_dict[each_node] \
                                = layer_wise_embed_dict[each_node] \
                                + self.ext_id_embed[each_prod_rule.rhs.node_attr(each_node)['ext_id']]

                    for each_layer in range(self.num_layers):
                        next_layer_embed_dict = {}
                        for each_edge in each_prod_rule.rhs.edges:
                            v = layer_wise_embed_dict[each_edge]
                            for each_node in each_prod_rule.rhs.nodes_in_edge(each_edge):
                                v = v + layer_wise_embed_dict[each_node]
                            next_layer_embed_dict[each_edge] = self.layer2layer_activation(self.layer2layer_list[each_layer](v))
                            out[each_batch_idx, each_idx, :] \
                                = out[each_batch_idx, each_idx, :] + self.layer2out_activation(self.layer2out_list[each_layer](v))
                        for each_node in each_prod_rule.rhs.nodes:
                            v = layer_wise_embed_dict[each_node]
                            for each_edge in each_prod_rule.rhs.adj_edges(each_node):
                                v = v + layer_wise_embed_dict[each_edge]
                            next_layer_embed_dict[each_node] = self.layer2layer_activation(self.layer2layer_list[each_layer](v))
                            out[each_batch_idx, each_idx, :]\
                                = out[each_batch_idx, each_idx, :] + self.layer2out_activation(self.layer2out_list[each_layer](v))
                        layer_wise_embed_dict = next_layer_embed_dict
                        
        return out


class MolecularProdRuleEmbeddingLastLayer(nn.Module):
    
    ''' molecular fingerprint layer
    '''

    def __init__(self, prod_rule_corpus, layer2layer_activation, layer2out_activation,
                 out_dim=32, element_embed_dim=32,
                 num_layers=3, padding_idx=None, use_gpu=False):
        super().__init__()
        if padding_idx is not None:
            assert padding_idx == -1, 'padding_idx must be -1.'
        self.prod_rule_corpus = prod_rule_corpus
        self.layer2layer_activation = layer2layer_activation
        self.layer2out_activation = layer2out_activation
        self.out_dim = out_dim
        self.element_embed_dim = element_embed_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.use_gpu = use_gpu

        self.layer2layer_list = []
        self.layer2out_list = []

        if self.use_gpu:
            self.atom_embed = nn.Embedding(self.prod_rule_corpus.num_edge_symbol, self.element_embed_dim).cuda()
            self.bond_embed = nn.Embedding(self.prod_rule_corpus.num_node_symbol, self.element_embed_dim).cuda()
            for _ in range(num_layers+1):
                self.layer2layer_list.append(nn.Linear(self.element_embed_dim, self.element_embed_dim).cuda())
                self.layer2out_list.append(nn.Linear(self.element_embed_dim, self.out_dim).cuda())
        else:
            self.atom_embed = nn.Embedding(self.prod_rule_corpus.num_edge_symbol, self.element_embed_dim)
            self.bond_embed = nn.Embedding(self.prod_rule_corpus.num_node_symbol, self.element_embed_dim)
            for _ in range(num_layers+1):
                self.layer2layer_list.append(nn.Linear(self.element_embed_dim, self.element_embed_dim))
                self.layer2out_list.append(nn.Linear(self.element_embed_dim, self.out_dim))


    def forward(self, prod_rule_idx_seq):
        ''' forward model for mini-batch

        Parameters
        ----------
        prod_rule_idx_seq : (batch_size, length)

        Returns
        -------
        Variable, shape (batch_size, length, out_dim)
        '''
        batch_size, length = prod_rule_idx_seq.shape
        if self.use_gpu:
            out = Variable(torch.zeros((batch_size, length, self.out_dim))).cuda()
        else:
            out = Variable(torch.zeros((batch_size, length, self.out_dim)))
        for each_batch_idx in range(batch_size):
            for each_idx in range(length):
                if int(prod_rule_idx_seq[each_batch_idx, each_idx]) == len(self.prod_rule_corpus.prod_rule_list):
                    continue
                else:
                    each_prod_rule = self.prod_rule_corpus.prod_rule_list[int(prod_rule_idx_seq[each_batch_idx, each_idx])]

                    if self.use_gpu:
                        layer_wise_embed_dict = {each_edge: self.atom_embed(
                            Variable(torch.LongTensor(
                                [each_prod_rule.rhs.edge_attr(each_edge)['symbol_idx']]
                            ), requires_grad=False).cuda())
                                                 for each_edge in each_prod_rule.rhs.edges}
                        layer_wise_embed_dict.update({each_node: self.bond_embed(
                                                         Variable(
                                                             torch.LongTensor([
                                                                     each_prod_rule.rhs.node_attr(each_node)['symbol_idx']]),
                                                             requires_grad=False).cuda()
                                                     ) for each_node in each_prod_rule.rhs.nodes})
                    else:
                        layer_wise_embed_dict = {each_edge: self.atom_embed(
                            Variable(torch.LongTensor(
                                [each_prod_rule.rhs.edge_attr(each_edge)['symbol_idx']]
                            ), requires_grad=False))
                                                 for each_edge in each_prod_rule.rhs.edges}
                        layer_wise_embed_dict.update({each_node: self.bond_embed(
                                                         Variable(
                                                             torch.LongTensor([
                                                                     each_prod_rule.rhs.node_attr(each_node)['symbol_idx']]), 
                                                             requires_grad=False)
                                                     ) for each_node in each_prod_rule.rhs.nodes})

                    for each_layer in range(self.num_layers):
                        next_layer_embed_dict = {}
                        for each_edge in each_prod_rule.rhs.edges:
                            v = layer_wise_embed_dict[each_edge]
                            for each_node in each_prod_rule.rhs.nodes_in_edge(each_edge):
                                v += layer_wise_embed_dict[each_node]
                            next_layer_embed_dict[each_edge] = self.layer2layer_activation(self.layer2layer_list[each_layer](v))
                        for each_node in each_prod_rule.rhs.nodes:
                            v = layer_wise_embed_dict[each_node]
                            for each_edge in each_prod_rule.rhs.adj_edges(each_node):
                                v += layer_wise_embed_dict[each_edge]
                            next_layer_embed_dict[each_node] = self.layer2layer_activation(self.layer2layer_list[each_layer](v))
                        layer_wise_embed_dict = next_layer_embed_dict
                    for each_edge in each_prod_rule.rhs.edges:
                        out[each_batch_idx, each_idx, :] = self.layer2out_activation(self.layer2out_list[self.num_layers](v))
                    for each_edge in each_prod_rule.rhs.edges:
                        out[each_batch_idx, each_idx, :] = self.layer2out_activation(self.layer2out_list[self.num_layers](v))
                        
        return out


class MolecularProdRuleEmbeddingUsingFeatures(nn.Module):
    
    ''' molecular fingerprint layer
    '''

    def __init__(self, prod_rule_corpus, layer2layer_activation, layer2out_activation,
                 out_dim=32, num_layers=3, padding_idx=None, use_gpu=False):
        super().__init__()
        if padding_idx is not None:
            assert padding_idx == -1, 'padding_idx must be -1.'
        self.feature_dict, self.feature_dim = prod_rule_corpus.construct_feature_vectors()
        self.prod_rule_corpus = prod_rule_corpus
        self.layer2layer_activation = layer2layer_activation
        self.layer2out_activation = layer2out_activation
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.use_gpu = use_gpu

        self.layer2layer_list = []
        self.layer2out_list = []

        if self.use_gpu:
            for each_key in self.feature_dict:
                self.feature_dict[each_key] = self.feature_dict[each_key].to_dense().cuda()
            for _ in range(num_layers):
                self.layer2layer_list.append(nn.Linear(self.feature_dim, self.feature_dim).cuda())
                self.layer2out_list.append(nn.Linear(self.feature_dim, self.out_dim).cuda())
        else:
            for _ in range(num_layers):
                self.layer2layer_list.append(nn.Linear(self.feature_dim, self.feature_dim))
                self.layer2out_list.append(nn.Linear(self.feature_dim, self.out_dim))


    def forward(self, prod_rule_idx_seq):
        ''' forward model for mini-batch

        Parameters
        ----------
        prod_rule_idx_seq : (batch_size, length)

        Returns
        -------
        Variable, shape (batch_size, length, out_dim)
        '''
        batch_size, length = prod_rule_idx_seq.shape
        if self.use_gpu:
            out = Variable(torch.zeros((batch_size, length, self.out_dim))).cuda()
        else:
            out = Variable(torch.zeros((batch_size, length, self.out_dim)))
        for each_batch_idx in range(batch_size):
            for each_idx in range(length):
                if int(prod_rule_idx_seq[each_batch_idx, each_idx]) == len(self.prod_rule_corpus.prod_rule_list):
                    continue
                else:
                    each_prod_rule = self.prod_rule_corpus.prod_rule_list[int(prod_rule_idx_seq[each_batch_idx, each_idx])]
                    edge_list = sorted(list(each_prod_rule.rhs.edges))
                    node_list = sorted(list(each_prod_rule.rhs.nodes))
                    adj_mat = torch.FloatTensor(each_prod_rule.rhs_adj_mat(edge_list + node_list).todense() + np.identity(len(edge_list)+len(node_list)))
                    if self.use_gpu:
                        adj_mat = adj_mat.cuda()
                    layer_wise_embed = [
                        self.feature_dict[each_prod_rule.rhs.edge_attr(each_edge)['symbol']]
                        for each_edge in edge_list]\
                            + [self.feature_dict[each_prod_rule.rhs.node_attr(each_node)['symbol']]
                               for each_node in node_list]
                    for each_node in each_prod_rule.ext_node.values():
                        layer_wise_embed[each_prod_rule.rhs.num_edges + node_list.index(each_node)] \
                                = layer_wise_embed[each_prod_rule.rhs.num_edges + node_list.index(each_node)] \
                                + self.feature_dict[('ext_id', each_prod_rule.rhs.node_attr(each_node)['ext_id'])]
                    layer_wise_embed = torch.stack(layer_wise_embed)

                    for each_layer in range(self.num_layers):
                        message = adj_mat @ layer_wise_embed
                        next_layer_embed = self.layer2layer_activation(self.layer2layer_list[each_layer](message))
                        out[each_batch_idx, each_idx, :] \
                                = out[each_batch_idx, each_idx, :] \
                                + self.layer2out_activation(self.layer2out_list[each_layer](message)).sum(dim=0)
                        layer_wise_embed = next_layer_embed
        return out
