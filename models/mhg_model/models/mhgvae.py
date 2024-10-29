# -*- coding:utf-8 -*-
# Rhizome
# Version beta 0.0, August 2023
# Property of IBM Research, Accelerated Discovery
#

"""
PLEASE NOTE THIS IMPLEMENTATION INCLUDES ADAPTED SOURCE CODE
OF THE MHG IMPLEMENTATION OF HIROSHI KAJINO AT IBM TRL ALREADY PUBLICLY AVAILABLE, 
E.G., GRUEncoder/GRUDecoder, GrammarSeq2SeqVAE AND EVEN SOME METHODS OF GrammarGINVAE.
THIS MIGHT INFLUENCE THE DECISION OF THE FINAL LICENSE SO CAREFUL CHECK NEEDS BE DONE. 
"""

import numpy as np
import logging

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool


from ..graph_grammar.graph_grammar.symbols import NTSymbol
from ..graph_grammar.nn.encoder import EncoderBase
from ..graph_grammar.nn.decoder import DecoderBase

def get_atom_edge_feature_dims():
    from torch_geometric.utils.smiles import x_map, e_map
    func = lambda x: len(x[1])
    return list(map(func, x_map.items())), list(map(func, e_map.items()))


class FeatureEmbedding(nn.Module):
    def __init__(self, input_dims, embedded_dim):
        super().__init__()
        self.embedding_list = nn.ModuleList()
        for dim in input_dims:
            embedding = nn.Embedding(dim, embedded_dim)
            self.embedding_list.append(embedding)
    
    def forward(self, x):
        output = 0
        for i in range(x.shape[1]):
            input = x[:, i].to(torch.int)
            device = next(self.parameters()).device
            if device != input.device:
                input = input.to(device)
            emb = self.embedding_list[i](input)
            output += emb
        return output
    

class GRUEncoder(EncoderBase):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 bidirectional: bool, dropout: float, batch_size: int, rank: int=-1,
                 no_dropout: bool=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_size = batch_size
        self.rank = rank
        self.model = nn.GRU(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional,
                            dropout=self.dropout if not no_dropout else 0)
        if self.rank >= 0:
            if torch.cuda.is_available():
                self.model = self.model.to(rank)
            else:
                # support mac mps
                self.model = self.model.to(torch.device("mps", rank))
        self.init_hidden(self.batch_size)
        
    def init_hidden(self, bsize):
        self.h0 = torch.zeros(((self.bidirectional + 1) * self.num_layers,
                               min(self.batch_size, bsize),
                               self.hidden_dim),
                              requires_grad=False)
        if self.rank >= 0:        
            if torch.cuda.is_available():
                self.h0 = self.h0.to(self.rank)
            else:
                # support mac mps
                self.h0 = self.h0.to(torch.device("mps", self.rank))

    def to(self, device):
        newself = super().to(device)
        newself.model = newself.model.to(device)
        newself.h0 = newself.h0.to(device)
        newself.rank = next(newself.parameters()).get_device()
        return newself

    def forward(self, in_seq_emb):
        ''' forward model

        Parameters
        ----------
        in_seq_emb : Tensor, shape (batch_size, max_len, input_dim)

        Returns
        -------
        hidden_seq_emb : Tensor, shape (batch_size, max_len, 1 + bidirectional, hidden_dim)
        '''
        # Kishi: I think original MHG had this init_hidden()
        self.init_hidden(in_seq_emb.size(0))
        max_len = in_seq_emb.size(1)
        hidden_seq_emb, self.h0 = self.model(
            in_seq_emb, self.h0)
        # As shown as returns, convert hidden_seq_emb: (batch_size, seq_len, (1 or 2) * hidden_size) -->
        # (batch_size, seq_len, 1 or 2, hidden_size)
        # In the original input the original GRU/LSTM with bidirectional encoding 
        # has contactinated tensors 
        # (first half for forward RNN, latter half for backward RNN)
        # so convert them in a more friendly format packed for each RNN
        hidden_seq_emb = hidden_seq_emb.view(-1,
                                             max_len,
                                             1 + self.bidirectional,
                                             self.hidden_dim)
        return hidden_seq_emb


class GRUDecoder(DecoderBase):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, batch_size: int, rank: int=-1,
                 no_dropout: bool=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.rank = rank
        self.model = nn.GRU(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=False,
                            dropout=self.dropout if not no_dropout else 0
        )
        if self.rank >= 0:
            if torch.cuda.is_available():
                self.model = self.model.to(self.rank)
            else:
                # support mac mps
                self.model = self.model.to(torch.device("mps", self.rank))
        self.init_hidden(self.batch_size)

    def init_hidden(self, bsize):
        self.hidden_dict['h'] = torch.zeros((self.num_layers,
                                             min(self.batch_size, bsize),
                                             self.hidden_dim),
                                            requires_grad=False)
        if self.rank >= 0:
            if torch.cuda.is_available():
                self.hidden_dict['h'] = self.hidden_dict['h'].to(self.rank)
            else:
                self.hidden_dict['h'] = self.hidden_dict['h'].to(torch.device("mps", self.rank))

    def to(self, device):
        newself = super().to(device)
        newself.model = newself.model.to(device)
        for k in self.hidden_dict.keys():
            newself.hidden_dict[k] = newself.hidden_dict[k].to(device)
        newself.rank = next(newself.parameters()).get_device()
        return newself

    def forward_one_step(self, tgt_emb_in):
        ''' one-step forward model

        Parameters
        ----------
        tgt_emb_in : Tensor, shape (batch_size, input_dim)

        Returns
        -------
        Tensor, shape (batch_size, hidden_dim)
        '''
        bsize = tgt_emb_in.size(0)
        tgt_emb_out, self.hidden_dict['h'] \
            = self.model(tgt_emb_in.view(bsize, 1, -1),
                         self.hidden_dict['h'])
        return tgt_emb_out


class NodeMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.nbat = nn.BatchNorm1d(hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.nbat(x)
        x = x.relu()
        x = self.lin2(x)
        return x
    
    
class GINLayer(MessagePassing):
    def __init__(self, node_input_size, node_output_size, node_hidden_size, edge_input_size):
        super().__init__()
        self.node_mlp = NodeMLP(node_input_size, node_output_size, node_hidden_size)
        self.edge_mlp = FeatureEmbedding(edge_input_size, node_output_size)
        self.eps = nn.Parameter(torch.tensor([0.0]))
        
    def forward(self, x, edge_index, edge_attr):
        msg = self.propagate(edge_index, x=x ,edge_attr=edge_attr)
        x = (1.0 + self.eps) * x + msg
        x = x.relu()
        x = self.node_mlp(x)
        return x
    
    def message(self, x_j, edge_attr):
        edge_attr = self.edge_mlp(edge_attr)
        x_j = x_j + edge_attr
        x_j = x_j.relu()
        return x_j
    
    def update(self, aggr_out):
        return aggr_out

#TODO implement the case where features of atoms and edges are considered
# Check GraphMVP and ogb (open graph benchmark) to realize this
class GIN(torch.nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, hidden_channels=64, 
                 proximity_size=3, dropout=0.1):
        super().__init__()
        #print("(num node features, num edge features)=", (node_feature_size, edge_feature_size))
        hsize = hidden_channels * 2
        atom_dim, edge_dim = get_atom_edge_feature_dims()
        self.trans = FeatureEmbedding(atom_dim, hidden_channels)
        ml = []
        for _ in range(proximity_size):
            ml.append(GINLayer(hidden_channels, hidden_channels, hsize, edge_dim))
        self.mlist = nn.ModuleList(ml)
        #It is possible to calculate relu with x.relu() where x is an output
        #self.activations = nn.ModuleList(actl)
        self.dropout = dropout
        self.proximity_size = proximity_size
        
    def forward(self, x, edge_index, edge_attr, batch_size):
        x = x.to(torch.float)
        #print("before: edge_weight.shape=", edge_attr.shape)
        edge_attr = edge_attr.to(torch.float)
        #print("after: edge_weight.shape=", edge_attr.shape)
        x = self.trans(x)
        # TODO Check if this x is consistent with global_add_pool
        hlist = [global_add_pool(x, batch_size)]
        for id, m in enumerate(self.mlist):
            x = m(x, edge_index=edge_index, edge_attr=edge_attr)
            #print("Done with one layer") 
            ###if id != self.proximity_size - 1:
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            #h = global_mean_pool(x, batch_size)
            h = global_add_pool(x, batch_size)
            hlist.append(h)
            #print("Done with one relu call: x.shape=", x.shape)
        #print("calling golbal mean pool")
        #print("calling dropout x.shape=", x.shape)
        #print("x=", x)
        #print("hlist[0].shape=", hlist[0].shape)
        x = torch.cat(hlist, dim=1)
        #print("x.shape=", x.shape)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x


# TODO copied from MHG implementation and adapted here. 
class GrammarSeq2SeqVAE(nn.Module):

    '''
    Variational seq2seq with grammar.
    TODO: rewrite this class using mixin
    '''

    def __init__(self, hrg, rank=-1, latent_dim=64, max_len=80,
                 batch_size=64, padding_idx=-1, 
                 encoder_params={'hidden_dim': 384, 'num_layers': 3, 'bidirectional': True,
                                 'dropout': 0.1},
                 decoder_params={'hidden_dim': 384, #'num_layers': 2,
                                 'num_layers': 3,
                                 'dropout': 0.1},
                 prod_rule_embed_params={'out_dim': 128},
                 no_dropout=False):

        super().__init__()
        # TODO USE GRU FOR ENCODING AND DECODING
        self.hrg = hrg
        self.rank = rank
        self.prod_rule_corpus = hrg.prod_rule_corpus
        self.prod_rule_embed_params = prod_rule_embed_params

        self.vocab_size = hrg.num_prod_rule + 1
        self.batch_size = batch_size
        self.padding_idx = np.mod(padding_idx, self.vocab_size)
        self.no_dropout = no_dropout

        self.latent_dim = latent_dim
        self.max_len = max_len
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        
        # TODO Simple embedding is used. Check if a domain-dependent embedding works or not. 
        embed_out_dim = self.prod_rule_embed_params['out_dim']        
        #use MolecularProdRuleEmbedding later on
        self.src_embedding = nn.Embedding(self.vocab_size, embed_out_dim,
                                          padding_idx=self.padding_idx)
        self.tgt_embedding = nn.Embedding(self.vocab_size, embed_out_dim,
                                          padding_idx=self.padding_idx)

        # USE a GRU-based encoder in MHG
        self.encoder = GRUEncoder(input_dim=embed_out_dim, batch_size=self.batch_size,
                                      rank=self.rank, no_dropout=self.no_dropout,
                                  **self.encoder_params)

        lin_dim = (self.encoder_params.get('bidirectional', False) + 1) * self.encoder_params['hidden_dim']
        lin_out_dim = self.latent_dim
        self.hidden2mean = nn.Linear(lin_dim, lin_out_dim, bias=False)
        self.hidden2logvar = nn.Linear(lin_dim, lin_out_dim)
   
        # USE a GRU-based decoder in MHG
        self.decoder = GRUDecoder(input_dim=embed_out_dim, batch_size=self.batch_size,
                                  rank=self.rank, no_dropout=self.no_dropout, **self.decoder_params)
        self.latent2tgt_emb = nn.Linear(self.latent_dim, embed_out_dim)
        self.latent2hidden_dict = nn.ModuleDict()
        dec_lin_out_dim = self.decoder_params['hidden_dim']
        for each_hidden in self.decoder.hidden_dict.keys():
            self.latent2hidden_dict[each_hidden] = nn.Linear(self.latent_dim, dec_lin_out_dim)
            if self.rank >= 0:
                if torch.cuda.is_available():
                    self.latent2hidden_dict[each_hidden] = self.latent2hidden_dict[each_hidden].to(self.rank)
                else:
                    # support mac mps
                    self.latent2hidden_dict[each_hidden] = self.latent2hidden_dict[each_hidden].to(torch.device("mps", self.rank))

        self.dec2vocab = nn.Linear(dec_lin_out_dim, self.vocab_size)
        self.encoder.init_hidden(self.batch_size)
        self.decoder.init_hidden(self.batch_size)

        # TODO Do we need this?
        if hasattr(self.src_embedding, 'weight'):
            self.src_embedding.weight.data.uniform_(-0.1, 0.1)
        if hasattr(self.tgt_embedding, 'weight'):
            self.tgt_embedding.weight.data.uniform_(-0.1, 0.1)
            
        self.encoder.init_hidden(self.batch_size)
        self.decoder.init_hidden(self.batch_size)
    
    def to(self, device):
        newself = super().to(device)
        newself.src_embedding = newself.src_embedding.to(device)
        newself.tgt_embedding = newself.tgt_embedding.to(device)
        newself.encoder = newself.encoder.to(device)
        newself.decoder = newself.decoder.to(device)
        newself.dec2vocab = newself.dec2vocab.to(device)
        newself.hidden2mean = newself.hidden2mean.to(device)
        newself.hidden2logvar = newself.hidden2logvar.to(device)
        newself.latent2tgt_emb = newself.latent2tgt_emb.to(device)
        newself.latent2hidden_dict = newself.latent2hidden_dict.to(device)
        return newself
    
    def forward(self, in_seq, out_seq):
        ''' forward model

        Parameters
        ----------
        in_seq : Variable, shape (batch_size, length)
            each element corresponds to word index.
            where the index should be less than `vocab_size`

        Returns
        -------
        Variable, shape (batch_size, length, vocab_size)
            logit of each word (applying softmax yields the probability)
        '''
        mu, logvar = self.encode(in_seq)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, out_seq), mu, logvar

    def encode(self, in_seq):
        src_emb = self.src_embedding(in_seq)
        src_h = self.encoder.forward(src_emb)
        if self.encoder_params.get('bidirectional', False):
            concat_src_h = torch.cat((src_h[:, -1, 0, :], src_h[:, 0, 1, :]), dim=1)
            return self.hidden2mean(concat_src_h), self.hidden2logvar(concat_src_h)
        else:
            return self.hidden2mean(src_h[:, -1, :]), self.hidden2logvar(src_h[:, -1, :])

    def reparameterize(self, mu, logvar, training=True):
        if training:
            std = logvar.mul(0.5).exp_()
            device = next(self.parameters()).device
            eps = Variable(std.data.new(std.size()).normal_())
            if device != eps.get_device():
                eps.to(device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    #TODO Not tested. Need to implement this in case of molecular structure generation
    def sample(self, sample_size=-1, deterministic=True, return_z=False):
        self.eval()
        self.init_hidden()
        if sample_size == -1:
            sample_size = self.batch_size

        num_iter = int(np.ceil(sample_size / self.batch_size))
        hg_list = []
        z_list = []
        for _ in range(num_iter):
            z = Variable(torch.normal(
                torch.zeros(self.batch_size, self.latent_dim),
                torch.ones(self.batch_size * self.latent_dim))).cuda()
            _, each_hg_list = self.decode(z, deterministic=deterministic)
            z_list.append(z)
            hg_list += each_hg_list
        z = torch.cat(z_list)[:sample_size]
        hg_list = hg_list[:sample_size]
        if return_z:
            return hg_list, z.cpu().detach().numpy()
        else:
            return hg_list

    def decode(self, z=None, out_seq=None, deterministic=True):
        if z is None:
            z = Variable(torch.normal(
                torch.zeros(self.batch_size, self.latent_dim),
                torch.ones(self.batch_size * self.latent_dim)))
        if self.rank >= 0:
            z = z.to(next(self.parameters()).device)

        hidden_dict_0 = {}
        for each_hidden in self.latent2hidden_dict.keys():
            hidden_dict_0[each_hidden] = self.latent2hidden_dict[each_hidden](z)
        bsize = z.size(0)
        self.decoder.init_hidden(bsize) 
        self.decoder.feed_hidden(hidden_dict_0)

        if out_seq is not None:
            tgt_emb0 = self.latent2tgt_emb(z)
            tgt_emb0 = tgt_emb0.view(tgt_emb0.shape[0], 1, tgt_emb0.shape[1])
            out_seq_emb = self.tgt_embedding(out_seq)
            tgt_emb = torch.cat((tgt_emb0, out_seq_emb), dim=1)[:, :-1, :]
            tgt_emb_pred_list = []
            for each_idx in range(self.max_len):
                tgt_emb_pred = self.decoder.forward_one_step(tgt_emb[:, each_idx, :].view(bsize, 1, -1))
                tgt_emb_pred_list.append(tgt_emb_pred)
            vocab_logit = self.dec2vocab(torch.cat(tgt_emb_pred_list, dim=1))
            return vocab_logit
        else:
            with torch.no_grad():
                tgt_emb = self.latent2tgt_emb(z)
                tgt_emb = tgt_emb.view(tgt_emb.shape[0], 1, tgt_emb.shape[1])
                tgt_emb_pred_list = []
                stack_list = []
                hg_list = []
                nt_symbol_list = []
                nt_edge_list = []
                gen_finish_list = []
                for _ in range(bsize):
                    stack_list.append([])
                    hg_list.append(None)
                    nt_symbol_list.append(NTSymbol(degree=0,
                                                   is_aromatic=False,
                                                   bond_symbol_list=[]))
                    nt_edge_list.append(None)
                    gen_finish_list.append(False)

                for idx in range(self.max_len):
                    tgt_emb_pred = self.decoder.forward_one_step(tgt_emb)
                    tgt_emb_pred_list.append(tgt_emb_pred)
                    vocab_logit = self.dec2vocab(tgt_emb_pred)
                    for each_batch_idx in range(bsize):
                        if not gen_finish_list[each_batch_idx]: # if generation has not finished
                            # get production rule greedily
                            prod_rule = self.hrg.prod_rule_corpus.sample(vocab_logit[each_batch_idx, :, :-1].squeeze().cpu().numpy(),
                                                                         nt_symbol_list[each_batch_idx],
                                                                         deterministic=deterministic)
                            # convert production rule into an index
                            tgt_id = self.hrg.prod_rule_list.index(prod_rule)
                            # apply the production rule
                            hg_list[each_batch_idx], nt_edges = prod_rule.applied_to(hg_list[each_batch_idx], nt_edge_list[each_batch_idx])
                            # add non-terminals to the stack
                            stack_list[each_batch_idx].extend(nt_edges[::-1])
                            # if the stack size is 0, generation has finished!
                            if len(stack_list[each_batch_idx]) == 0:
                                gen_finish_list[each_batch_idx] = True
                            else:
                                nt_edge_list[each_batch_idx] = stack_list[each_batch_idx].pop()
                                nt_symbol_list[each_batch_idx] = hg_list[each_batch_idx].edge_attr(nt_edge_list[each_batch_idx])['symbol']
                        else:
                            tgt_id = np.mod(self.padding_idx, self.vocab_size)
                        indice_tensor = torch.LongTensor([tgt_id])
                        device = next(self.parameters()).device
                        if indice_tensor.device != device: 
                            indice_tensor = indice_tensor.to(device)
                        tgt_emb[each_batch_idx, :] = self.tgt_embedding(indice_tensor)
                vocab_logit = self.dec2vocab(torch.cat(tgt_emb_pred_list, dim=1))
                #for id, v in enumerate(gen_finish_list):
                #if not v:
                #    print("bacth id={} not finished generating a sequence: ".format(id))
                return gen_finish_list, vocab_logit, hg_list


# TODO A lot of duplicates with GrammarVAE. Clean up it if necessary
class GrammarGINVAE(nn.Module):

    '''
    Variational autoencoder based on GIN and grammar
    '''

    def __init__(self, hrg, rank=-1, max_len=80,
                 batch_size=64, padding_idx=-1, 
                 encoder_params={'node_feature_size': 4, 'edge_feature_size': 3, 
                                 'hidden_channels': 64, 'proximity_size': 3,
                                 'dropout': 0.1},
                 decoder_params={'hidden_dim': 384, 'num_layers': 3,
                                 'dropout': 0.1},
                 prod_rule_embed_params={'out_dim': 128},
                 no_dropout=False):

        super().__init__()
        # TODO USE GRU FOR ENCODING AND DECODING
        self.hrg = hrg
        self.rank = rank
        self.prod_rule_corpus = hrg.prod_rule_corpus
        self.prod_rule_embed_params = prod_rule_embed_params

        self.vocab_size = hrg.num_prod_rule + 1
        self.batch_size = batch_size
        self.padding_idx = np.mod(padding_idx, self.vocab_size)
        self.no_dropout = no_dropout
        self.max_len = max_len
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        
        # TODO Simple embedding is used. Check if a domain-dependent embedding works or not. 
        embed_out_dim = self.prod_rule_embed_params['out_dim']        
        #use MolecularProdRuleEmbedding later on
        self.tgt_embedding = nn.Embedding(self.vocab_size, embed_out_dim,
                                          padding_idx=self.padding_idx)

        self.encoder = GIN(**self.encoder_params)
        self.latent_dim = self.encoder_params['hidden_channels']
        self.proximity_size = self.encoder_params['proximity_size']
        hidden_dim = self.decoder_params['hidden_dim']
        self.hidden2mean = nn.Linear(self.latent_dim * (1 + self.proximity_size), self.latent_dim, bias=False)
        self.hidden2logvar = nn.Linear(self.latent_dim * (1 + self.proximity_size), self.latent_dim)
   
        self.decoder = GRUDecoder(input_dim=embed_out_dim, batch_size=self.batch_size,
                                  rank=self.rank, no_dropout=self.no_dropout, **self.decoder_params)
        self.latent2tgt_emb = nn.Linear(self.latent_dim, embed_out_dim)
        self.latent2hidden_dict = nn.ModuleDict()
        for each_hidden in self.decoder.hidden_dict.keys():
            self.latent2hidden_dict[each_hidden] = nn.Linear(self.latent_dim, hidden_dim)
            if self.rank >= 0:
                if torch.cuda.is_available():
                    self.latent2hidden_dict[each_hidden] = self.latent2hidden_dict[each_hidden].to(self.rank)
                else:
                    # support mac mps
                    self.latent2hidden_dict[each_hidden] = self.latent2hidden_dict[each_hidden].to(torch.device("mps", self.rank))

        self.dec2vocab = nn.Linear(hidden_dim, self.vocab_size)
        self.decoder.init_hidden(self.batch_size)

        # TODO Do we need this?
        if hasattr(self.tgt_embedding, 'weight'):
            self.tgt_embedding.weight.data.uniform_(-0.1, 0.1)
        self.decoder.init_hidden(self.batch_size)
    
    def to(self, device):
        newself = super().to(device)
        newself.encoder = newself.encoder.to(device)
        newself.decoder = newself.decoder.to(device)
        newself.rank = next(newself.encoder.parameters()).get_device()
        return newself
    
    def forward(self, x, edge_index, edge_attr, batch_size, out_seq=None, sched_prob = None):
        mu, logvar = self.encode(x, edge_index, edge_attr, batch_size)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, out_seq, sched_prob=sched_prob), mu, logvar

    #TODO Not tested. Need to implement this in case of molecular structure generation
    def sample(self, sample_size=-1, deterministic=True, return_z=False):
        self.eval()
        self.init_hidden()
        if sample_size == -1:
            sample_size = self.batch_size

        num_iter = int(np.ceil(sample_size / self.batch_size))
        hg_list = []
        z_list = []
        for _ in range(num_iter):
            z = Variable(torch.normal(
                torch.zeros(self.batch_size, self.latent_dim),
                torch.ones(self.batch_size * self.latent_dim))).cuda()
            _, each_hg_list = self.decode(z, deterministic=deterministic)
            z_list.append(z)
            hg_list += each_hg_list
        z = torch.cat(z_list)[:sample_size]
        hg_list = hg_list[:sample_size]
        if return_z:
            return hg_list, z.cpu().detach().numpy()
        else:
            return hg_list

    def decode(self, z=None, out_seq=None, deterministic=True, sched_prob=None):
        if z is None:
            z = Variable(torch.normal(
                torch.zeros(self.batch_size, self.latent_dim),
                torch.ones(self.batch_size * self.latent_dim)))
        if self.rank >= 0:
            z = z.to(next(self.parameters()).device)

        hidden_dict_0 = {}
        for each_hidden in self.latent2hidden_dict.keys():
            hidden_dict_0[each_hidden] = self.latent2hidden_dict[each_hidden](z)
        bsize = z.size(0)
        self.decoder.init_hidden(bsize) 
        self.decoder.feed_hidden(hidden_dict_0)

        if out_seq is not None:
            tgt_emb0 = self.latent2tgt_emb(z)
            tgt_emb0 = tgt_emb0.view(tgt_emb0.shape[0], 1, tgt_emb0.shape[1])
            out_seq_emb = self.tgt_embedding(out_seq)
            tgt_emb = torch.cat((tgt_emb0, out_seq_emb), dim=1)[:, :-1, :]
            tgt_emb_pred_list = []
            tgt_emb_pred = None
            for each_idx in range(self.max_len):
                if tgt_emb_pred is None or sched_prob is None or torch.rand(1)[0] <= sched_prob:
                    inp = tgt_emb[:, each_idx, :].view(bsize, 1, -1)
                else:
                    cur_logit = self.dec2vocab(tgt_emb_pred)
                    yi = torch.argmax(cur_logit, dim=2)
                    inp = self.tgt_embedding(yi)
                tgt_emb_pred = self.decoder.forward_one_step(inp)
                tgt_emb_pred_list.append(tgt_emb_pred)
            vocab_logit = self.dec2vocab(torch.cat(tgt_emb_pred_list, dim=1))
            return vocab_logit
        else:
            with torch.no_grad():
                tgt_emb = self.latent2tgt_emb(z)
                tgt_emb = tgt_emb.view(tgt_emb.shape[0], 1, tgt_emb.shape[1])
                tgt_emb_pred_list = []
                stack_list = []
                hg_list = []
                nt_symbol_list = []
                nt_edge_list = []
                gen_finish_list = []
                for _ in range(bsize):
                    stack_list.append([])
                    hg_list.append(None)
                    nt_symbol_list.append(NTSymbol(degree=0,
                                                   is_aromatic=False,
                                                   bond_symbol_list=[]))
                    nt_edge_list.append(None)
                    gen_finish_list.append(False)

                for _ in range(self.max_len):
                    tgt_emb_pred = self.decoder.forward_one_step(tgt_emb)
                    tgt_emb_pred_list.append(tgt_emb_pred)
                    vocab_logit = self.dec2vocab(tgt_emb_pred)
                    for each_batch_idx in range(bsize):
                        if not gen_finish_list[each_batch_idx]: # if generation has not finished
                            # get production rule greedily
                            prod_rule = self.hrg.prod_rule_corpus.sample(vocab_logit[each_batch_idx, :, :-1].squeeze().cpu().numpy(),
                                                                         nt_symbol_list[each_batch_idx],
                                                                         deterministic=deterministic)
                            # convert production rule into an index
                            tgt_id = self.hrg.prod_rule_list.index(prod_rule)
                            # apply the production rule
                            hg_list[each_batch_idx], nt_edges = prod_rule.applied_to(hg_list[each_batch_idx], nt_edge_list[each_batch_idx])
                            # add non-terminals to the stack
                            stack_list[each_batch_idx].extend(nt_edges[::-1])
                            # if the stack size is 0, generation has finished!
                            if len(stack_list[each_batch_idx]) == 0:
                                gen_finish_list[each_batch_idx] = True
                            else:
                                nt_edge_list[each_batch_idx] = stack_list[each_batch_idx].pop()
                                nt_symbol_list[each_batch_idx] = hg_list[each_batch_idx].edge_attr(nt_edge_list[each_batch_idx])['symbol']
                        else:
                            tgt_id = np.mod(self.padding_idx, self.vocab_size)
                        indice_tensor = torch.LongTensor([tgt_id])
                        if self.rank >= 0:
                            indice_tensor = indice_tensor.to(next(self.parameters()).device)
                        tgt_emb[each_batch_idx, :] = self.tgt_embedding(indice_tensor)
                vocab_logit = self.dec2vocab(torch.cat(tgt_emb_pred_list, dim=1))
                return gen_finish_list, vocab_logit, hg_list

    #TODO Not tested. Need to implement this in case of molecular structure generation
    def conditional_distribution(self, z, tgt_id_list):
        self.eval()
        self.init_hidden()
        z = z.cuda()

        hidden_dict_0 = {}
        for each_hidden in self.latent2hidden_dict.keys():
            hidden_dict_0[each_hidden] = self.latent2hidden_dict[each_hidden](z)
        self.decoder.feed_hidden(hidden_dict_0)

        with torch.no_grad():
            tgt_emb = self.latent2tgt_emb(z)
            tgt_emb = tgt_emb.view(tgt_emb.shape[0], 1, tgt_emb.shape[1])
            nt_symbol_list = []
            stack_list = []
            hg_list = []
            nt_edge_list = []
            gen_finish_list = []
            for _ in range(self.batch_size):
                nt_symbol_list.append(NTSymbol(degree=0,
                                               is_aromatic=False,
                                               bond_symbol_list=[]))
                stack_list.append([])
                hg_list.append(None)
                nt_edge_list.append(None)
                gen_finish_list.append(False)

            for each_position in range(len(tgt_id_list[0])):
                tgt_emb_pred = self.decoder.forward_one_step(tgt_emb)
                for each_batch_idx in range(self.batch_size):
                    if not gen_finish_list[each_batch_idx]: # if generation has not finished
                        # use the prespecified target ids
                        tgt_id = tgt_id_list[each_batch_idx][each_position]
                        prod_rule = self.hrg.prod_rule_list[tgt_id]
                        # apply the production rule
                        hg_list[each_batch_idx], nt_edges = prod_rule.applied_to(hg_list[each_batch_idx], nt_edge_list[each_batch_idx])
                        # add non-terminals to the stack
                        stack_list[each_batch_idx].extend(nt_edges[::-1])
                        # if the stack size is 0, generation has finished!
                        if len(stack_list[each_batch_idx]) == 0:
                            gen_finish_list[each_batch_idx] = True
                        else:
                            nt_edge_list[each_batch_idx] = stack_list[each_batch_idx].pop()
                            nt_symbol_list[each_batch_idx] = hg_list[each_batch_idx].edge_attr(nt_edge_list[each_batch_idx])['symbol']
                    else:
                        tgt_id = np.mod(self.padding_idx, self.vocab_size)
                    indice_tensor = torch.LongTensor([tgt_id])
                    indice_tensor = indice_tensor.cuda()
                    tgt_emb[each_batch_idx, :] = self.tgt_embedding(indice_tensor)

            # last one step
            conditional_logprob_list = []
            tgt_emb_pred = self.decoder.forward_one_step(tgt_emb)
            vocab_logit = self.dec2vocab(tgt_emb_pred)
            for each_batch_idx in range(self.batch_size):
                if not gen_finish_list[each_batch_idx]: # if generation has not finished
                    # get production rule greedily
                    masked_logprob = self.hrg.prod_rule_corpus.masked_logprob(
                        vocab_logit[each_batch_idx, :, :-1].squeeze().cpu().numpy(),
                        nt_symbol_list[each_batch_idx])
                    conditional_logprob_list.append(masked_logprob)
                else:
                    conditional_logprob_list.append(None)
        return conditional_logprob_list

    #TODO Not tested. Need to implement this in case of molecular structure generation
    def decode_with_beam_search(self, z, beam_width=1):
        ''' Decode a latent vector using beam search.

        Parameters
        ----------
        z
            latent vector
        beam_width : int
            parameter for beam search

        Returns
        -------
        List of Hypergraphs
        '''
        if self.batch_size != 1:
            raise ValueError('this method works only under batch_size=1')
        if self.padding_idx != -1:
            raise ValueError('this method works only under padding_idx=-1')
        top_k_tgt_id_list = [[]] * beam_width
        logprob_list = [0.] * beam_width

        for each_len in range(self.max_len):
            expanded_logprob_list = np.repeat(logprob_list, self.vocab_size) # including padding_idx
            expanded_length_list = np.array([0] * (beam_width * self.vocab_size))
            for each_beam_idx, each_candidate in enumerate(top_k_tgt_id_list):
                conditional_logprob = self.conditional_distribution(z, [each_candidate])[0]
                if conditional_logprob is None:
                    expanded_logprob_list[(each_beam_idx + 1) * self.vocab_size - 1]\
                        = logprob_list[each_beam_idx]
                    expanded_logprob_list[each_beam_idx * self.vocab_size : (each_beam_idx + 1) * self.vocab_size - 1]\
                        = -np.inf
                    expanded_length_list[each_beam_idx * self.vocab_size : (each_beam_idx + 1) * self.vocab_size]\
                        = len(each_candidate)
                else:
                    expanded_logprob_list[each_beam_idx * self.vocab_size : (each_beam_idx + 1) * self.vocab_size - 1]\
                        = logprob_list[each_beam_idx] + conditional_logprob
                    expanded_logprob_list[(each_beam_idx + 1) * self.vocab_size - 1]\
                        = -np.inf
                    expanded_length_list[each_beam_idx * self.vocab_size : (each_beam_idx + 1) * self.vocab_size]\
                        = len(each_candidate) + 1
            score_list = np.array(expanded_logprob_list) / np.array(expanded_length_list)
            if each_len == 0:
                top_k_list = np.argsort(score_list[:self.vocab_size])[::-1][:beam_width]
            else:
                top_k_list = np.argsort(score_list)[::-1][:beam_width]
            next_top_k_tgt_id_list = []
            next_logprob_list = []
            for each_top_k in top_k_list:
                beam_idx = each_top_k // self.vocab_size
                vocab_idx = each_top_k % self.vocab_size
                if vocab_idx == self.vocab_size - 1:
                    next_top_k_tgt_id_list.append(top_k_tgt_id_list[beam_idx])
                    next_logprob_list.append(expanded_logprob_list[each_top_k])
                else:
                    next_top_k_tgt_id_list.append(top_k_tgt_id_list[beam_idx] + [vocab_idx])
                    next_logprob_list.append(expanded_logprob_list[each_top_k])
            top_k_tgt_id_list = next_top_k_tgt_id_list
            logprob_list = next_logprob_list

        # construct hypergraphs
        hg_list = []
        for each_tgt_id_list in top_k_tgt_id_list:
            hg = None
            stack = []
            nt_edge = None
            for each_idx, each_prod_rule_id in enumerate(each_tgt_id_list):
                prod_rule = self.hrg.prod_rule_list[each_prod_rule_id]
                hg, nt_edges = prod_rule.applied_to(hg, nt_edge)
                stack.extend(nt_edges[::-1])
                try:
                    nt_edge = stack.pop()
                except IndexError:
                    if each_idx == len(each_tgt_id_list) - 1:
                        break
                    else:
                        raise ValueError('some bugs')
            hg_list.append(hg)
        return hg_list

    def graph_embed(self, x, edge_index, edge_attr, batch_size):
        src_h = self.encoder.forward(x, edge_index, edge_attr, batch_size)
        return src_h        

    def encode(self, x, edge_index, edge_attr, batch_size):
        #print("device for src_emb=", src_emb.get_device())
        #print("device for self.encoder=", next(self.encoder.parameters()).get_device())
        src_h = self.graph_embed(x, edge_index, edge_attr, batch_size)
        mu, lv = self.get_mean_var(src_h)
        return mu, lv
    
    def get_mean_var(self, src_h):
        #src_h = torch.tanh(src_h)
        mu = self.hidden2mean(src_h)
        lv = self.hidden2logvar(src_h)
        mu = torch.tanh(mu)
        lv = torch.tanh(lv)
        return mu, lv

    def reparameterize(self, mu, logvar, training=True):
        if training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            if self.rank >= 0:
                eps = eps.to(next(self.parameters()).device)
            return eps.mul(std).add_(mu)
        else:
            return mu

# Copied from the MHG implementation and adapted
class GrammarVAELoss(_Loss):

    '''
    a loss function for Grammar VAE

    Attributes
    ----------
    hrg : HyperedgeReplacementGrammar
    beta : float
        coefficient of KL divergence
    '''

    def __init__(self, rank, hrg, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.hrg = hrg
        self.beta = beta
        self.rank = rank

    def forward(self, mu, logvar, in_seq_pred, in_seq):
        ''' compute VAE loss

        Parameters
        ----------
        in_seq_pred : torch.Tensor, shape (batch_size, max_len, vocab_size)
            logit
        in_seq : torch.Tensor, shape (batch_size, max_len)
            each element corresponds to a word id in vocabulary.
        mu : torch.Tensor, shape (batch_size, hidden_dim)
        logvar : torch.Tensor, shape (batch_size, hidden_dim)
            mean and log variance of the normal distribution
        '''
        batch_size = in_seq_pred.shape[0]
        max_len = in_seq_pred.shape[1]
        vocab_size = in_seq_pred.shape[2]
        mask = torch.zeros(in_seq_pred.shape)
        
        for each_batch in range(batch_size):
            flag = True
            for each_idx in range(max_len):
                prod_rule_idx = in_seq[each_batch, each_idx]
                if prod_rule_idx == vocab_size - 1:
                    #### DETERMINE WHETHER THIS SHOULD BE SKIPPED OR NOT
                    mask[each_batch, each_idx, prod_rule_idx] = 1
                    #break
                    continue
                lhs = self.hrg.prod_rule_corpus.prod_rule_list[prod_rule_idx].lhs_nt_symbol
                lhs_idx = self.hrg.prod_rule_corpus.nt_symbol_list.index(lhs)
                mask[each_batch, each_idx, :-1] = torch.FloatTensor(self.hrg.prod_rule_corpus.lhs_in_prod_rule[lhs_idx])
            if self.rank >= 0:
                mask = mask.to(next(self.parameters()).device)
        in_seq_pred = mask * in_seq_pred

        cross_entropy = F.cross_entropy(
            in_seq_pred.view(-1, vocab_size),
            in_seq.view(-1),
            reduction='sum',
            #ignore_index=self.ignore_index if self.ignore_index is not None else -100
            )
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return cross_entropy + self.beta * kl_div


class VAELoss(_Loss):
    def __init__(self, beta=0.01):
        super().__init__()
        self.beta = beta

    def forward(self, mean, log_var, dec_outputs, targets):

        device = mean.get_device()
        if device >= 0:
            targets = targets.to(mean.get_device())
        reconstruction = F.cross_entropy(dec_outputs.view(-1, dec_outputs.size(2)), targets.view(-1), reduction='sum')
        
        KL = 0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var))
        loss = - self.beta * KL + reconstruction
        return loss
