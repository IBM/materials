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

from collections import Counter
from functools import partial
from .utils import _easy_node_match, _edge_match, _node_match, common_node_list, _node_match_prod_rule
from networkx.algorithms.isomorphism import GraphMatcher
import os


class CliqueTreeCorpus(object):

    ''' clique tree corpus

    Attributes
    ----------
    clique_tree_list : list of CliqueTree
    subhg_list : list of Hypergraph
    '''

    def __init__(self):
        self.clique_tree_list = []
        self.subhg_list = []

    @property
    def size(self):
        return len(self.subhg_list)

    def add_clique_tree(self, clique_tree):
        for each_node in clique_tree.nodes:
            subhg = clique_tree.nodes[each_node]['subhg']
            subhg_idx = self.add_subhg(subhg)
            clique_tree.nodes[each_node]['subhg_idx'] = subhg_idx
        self.clique_tree_list.append(clique_tree)

    def add_to_subhg_list(self, clique_tree, root_node):
        parent_node_dict = {}
        current_node = None
        parent_node_dict[root_node] = None
        stack = [root_node]
        while stack:
            current_node = stack.pop()
            current_subhg = clique_tree.nodes[current_node]['subhg']
            for each_child in clique_tree.adj[current_node]:
                if each_child != parent_node_dict[current_node]:
                    stack.append(each_child)
                    parent_node_dict[each_child] = current_node
            if parent_node_dict[current_node] is not None:
                parent_subhg = clique_tree.nodes[parent_node_dict[current_node]]['subhg']
                common, _ = common_node_list(parent_subhg, current_subhg)
                parent_subhg.add_edge(set(common), attr_dict={'tmp': True})

        parent_node_dict = {}
        current_node = None
        parent_node_dict[root_node] = None
        stack = [root_node]
        while stack:
            current_node = stack.pop()
            current_subhg = clique_tree.nodes[current_node]['subhg']
            for each_child in clique_tree.adj[current_node]:
                if each_child != parent_node_dict[current_node]:
                    stack.append(each_child)
                    parent_node_dict[each_child] = current_node
            if parent_node_dict[current_node] is not None:
                parent_subhg = clique_tree.nodes[parent_node_dict[current_node]]['subhg']
                common, _ = common_node_list(parent_subhg, current_subhg)
                for each_idx, each_node in enumerate(common):
                    current_subhg.set_node_attr(each_node, {'ext_id': each_idx})

            subhg_idx, is_new = self.add_subhg(current_subhg)
            clique_tree.nodes[current_node]['subhg_idx'] = subhg_idx
        return clique_tree

    def add_subhg(self, subhg):
        if len(self.subhg_list) == 0:
            node_dict = {}
            for each_node in subhg.nodes:
                node_dict[each_node] = subhg.node_attr(each_node)['symbol'].__hash__()
            node_list = []
            for each_key, _ in sorted(node_dict.items(), key=lambda x:x[1]):
                node_list.append(each_key)
            for each_idx, each_node in enumerate(node_list):
                subhg.node_attr(each_node)['order4hrg'] = each_idx
            self.subhg_list.append(subhg)
            return 0, True
        else:
            match = False
            subhg_bond_symbol_counter \
                = Counter([subhg.node_attr(each_node)['symbol'] \
                           for each_node in subhg.nodes])
            subhg_atom_symbol_counter \
                = Counter([subhg.edge_attr(each_edge).get('symbol', None) \
                           for each_edge in subhg.edges])
            for each_idx, each_subhg in enumerate(self.subhg_list):
                each_bond_symbol_counter \
                    = Counter([each_subhg.node_attr(each_node)['symbol'] \
                               for each_node in each_subhg.nodes])
                each_atom_symbol_counter \
                    = Counter([each_subhg.edge_attr(each_edge).get('symbol', None) \
                               for each_edge in each_subhg.edges])
                if not match \
                   and (subhg.num_nodes == each_subhg.num_nodes
                        and subhg.num_edges == each_subhg.num_edges
                        and subhg_bond_symbol_counter == each_bond_symbol_counter
                        and subhg_atom_symbol_counter == each_atom_symbol_counter):
                    gm = GraphMatcher(each_subhg.hg,
                                      subhg.hg,
                                      node_match=_easy_node_match,
                                      edge_match=_edge_match)
                    try:
                        isomap = next(gm.isomorphisms_iter())
                        match = True
                        for each_node in each_subhg.nodes:
                            subhg.node_attr(isomap[each_node])['order4hrg'] \
                                = each_subhg.node_attr(each_node)['order4hrg']
                            if 'ext_id' in each_subhg.node_attr(each_node):
                                subhg.node_attr(isomap[each_node])['ext_id'] \
                                    = each_subhg.node_attr(each_node)['ext_id']
                        return each_idx, False
                    except StopIteration:
                        match = False    
            if not match:
                node_dict = {}
                for each_node in subhg.nodes:
                    node_dict[each_node] = subhg.node_attr(each_node)['symbol'].__hash__()
                node_list = []
                for each_key, _ in sorted(node_dict.items(), key=lambda x:x[1]):
                    node_list.append(each_key)
                for each_idx, each_node in enumerate(node_list):
                    subhg.node_attr(each_node)['order4hrg'] = each_idx

                #for each_idx, each_node in enumerate(subhg.nodes):
                #    subhg.node_attr(each_node)['order4hrg'] = each_idx
                self.subhg_list.append(subhg)
                return len(self.subhg_list) - 1, True
