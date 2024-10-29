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
__date__ = "Jun 4 2018"

from ..hypergraph import Hypergraph
from copy import deepcopy
from typing import List
import numpy as np


def common_node_list(hg1: Hypergraph, hg2: Hypergraph) -> List[str]:
    """ return a list of common nodes

    Parameters
    ----------
    hg1, hg2 : Hypergraph

    Returns
    -------
    list of str
        list of common nodes
    """
    if hg1 is None or hg2 is None:
        return [], False
    else:
        node_set = hg1.nodes.intersection(hg2.nodes)
        node_dict = {}
        if 'order4hrg' in hg1.node_attr(list(hg1.nodes)[0]):
            for each_node in node_set:
                node_dict[each_node] = hg1.node_attr(each_node)['order4hrg']
        else:
            for each_node in node_set:
                node_dict[each_node] = hg1.node_attr(each_node)['symbol'].__hash__()
        node_list = []
        for each_key, _ in sorted(node_dict.items(), key=lambda x:x[1]):
            node_list.append(each_key)
        edge_name = hg1.has_edge(node_list, ignore_order=True)
        if edge_name:
            if not hg1.edge_attr(edge_name).get('terminal', True):
                node_list = hg1.nodes_in_edge(edge_name)
            return node_list, True
        else:
            return node_list, False


def _node_match(node1, node2):
    # if the nodes are hyperedges, `atom_attr` determines the match
    if node1['bipartite'] == 'edge' and node2['bipartite'] == 'edge':
        return node1["attr_dict"]['symbol'] == node2["attr_dict"]['symbol']
    elif node1['bipartite'] == 'node' and node2['bipartite'] == 'node':
        # bond_symbol
        return node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']
    else:
        return False

def _easy_node_match(node1, node2):
    # if the nodes are hyperedges, `atom_attr` determines the match
    if node1['bipartite'] == 'edge' and node2['bipartite'] == 'edge':
        return node1["attr_dict"].get('symbol', None) == node2["attr_dict"].get('symbol', None)
    elif node1['bipartite'] == 'node' and node2['bipartite'] == 'node':
        # bond_symbol
        return node1['attr_dict'].get('ext_id', -1) == node2['attr_dict'].get('ext_id', -1)\
            and node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']
    else:
        return False


def _node_match_prod_rule(node1, node2, ignore_order=False):
    # if the nodes are hyperedges, `atom_attr` determines the match
    if node1['bipartite'] == 'edge' and node2['bipartite'] == 'edge':
        return node1["attr_dict"]['symbol'] == node2["attr_dict"]['symbol']
    elif node1['bipartite'] == 'node' and node2['bipartite'] == 'node':
        # ext_id, order4hrg, bond_symbol
        if ignore_order:
            return node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']
        else:
            return node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']\
                and node1['attr_dict'].get('ext_id', -1) == node2['attr_dict'].get('ext_id', -1)
    else:
        return False


def _edge_match(edge1, edge2, ignore_order=False):
    #return True
    if ignore_order:
        return True
    else:
        return edge1["order"] == edge2["order"]

def masked_softmax(logit, mask):
    ''' compute a probability distribution from logit

    Parameters
    ----------
    logit : array-like, length D
        each element indicates how each dimension is likely to be chosen
        (the larger, the more likely)
    mask : array-like, length D
        each element is either 0 or 1.
        if 0, the dimension is ignored
        when computing the probability distribution.

    Returns
    -------
    prob_dist : array, length D
        probability distribution computed from logit.
        if `mask[d] = 0`, `prob_dist[d] = 0`.
    '''
    if logit.shape != mask.shape:
        raise ValueError('logit and mask must have the same shape')
    c = np.max(logit)
    exp_logit = np.exp(logit - c) * mask
    sum_exp_logit = exp_logit @ mask
    return exp_logit / sum_exp_logit
