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
__date__ = "Jan 31 2018"

from copy import deepcopy
from typing import List, Dict, Tuple
import networkx as nx
import numpy as np
import os


class Hypergraph(object):
    '''
    A class of a hypergraph.
    Each hyperedge can be ordered. For the ordered case,
    edges adjacent to the hyperedge node are labeled by their orders.

    Attributes
    ----------
    hg : nx.Graph
        a bipartite graph representation of a hypergraph
    edge_idx : int
        total number of hyperedges that exist so far
    '''
    def __init__(self):
        self.hg = nx.Graph()
        self.edge_idx = 0
        self.nodes = set([])
        self.num_nodes = 0
        self.edges = set([])
        self.num_edges = 0
        self.nodes_in_edge_dict = {}

    def add_node(self, node: str, attr_dict=None):
        ''' add a node to hypergraph

        Parameters
        ----------
        node : str
            node name
        attr_dict : dict
            dictionary of node attributes
        '''
        self.hg.add_node(node, bipartite='node', attr_dict=attr_dict)
        if node not in self.nodes:
            self.num_nodes += 1
        self.nodes.add(node)

    def add_edge(self, node_list: List[str], attr_dict=None, edge_name=None):
        ''' add an edge consisting of nodes `node_list`

        Parameters
        ----------
        node_list : list 
            ordered list of nodes that consist the edge
        attr_dict : dict
            dictionary of edge attributes
        '''
        if edge_name is None:
            edge = 'e{}'.format(self.edge_idx)
        else:
            assert edge_name not in self.edges
            edge = edge_name
        self.hg.add_node(edge, bipartite='edge', attr_dict=attr_dict)
        if edge not in self.edges:
            self.num_edges += 1
        self.edges.add(edge)
        self.nodes_in_edge_dict[edge] = node_list
        if type(node_list) == list:
            for node_idx, each_node in enumerate(node_list):
                self.hg.add_edge(edge, each_node, order=node_idx)
                if each_node not in self.nodes:
                    self.num_nodes += 1
                self.nodes.add(each_node)

        elif type(node_list) == set:
            for each_node in node_list:
                self.hg.add_edge(edge, each_node, order=-1)
                if each_node not in self.nodes:
                    self.num_nodes += 1
                self.nodes.add(each_node)
        else:
            raise ValueError
        self.edge_idx += 1
        return edge

    def remove_node(self, node: str, remove_connected_edges=True):
        ''' remove a node

        Parameters
        ----------
        node : str
            node name
        remove_connected_edges : bool
            if True, remove edges that are adjacent to the node
        '''
        if remove_connected_edges:
            connected_edges = deepcopy(self.adj_edges(node))
            for each_edge in connected_edges:
                self.remove_edge(each_edge)
        self.hg.remove_node(node)
        self.num_nodes -= 1
        self.nodes.remove(node)

    def remove_nodes(self, node_iter, remove_connected_edges=True):
        ''' remove a set of nodes

        Parameters
        ----------
        node_iter : iterator of strings
            nodes to be removed
        remove_connected_edges : bool
            if True, remove edges that are adjacent to the node        
        '''
        for each_node in node_iter:
            self.remove_node(each_node, remove_connected_edges)

    def remove_edge(self, edge: str):
        ''' remove an edge

        Parameters
        ----------
        edge : str
            edge to be removed
        '''
        self.hg.remove_node(edge)
        self.edges.remove(edge)
        self.num_edges -= 1
        self.nodes_in_edge_dict.pop(edge)

    def remove_edges(self, edge_iter):
        ''' remove a set of edges

        Parameters
        ----------
        edge_iter : iterator of strings
            edges to be removed
        '''
        for each_edge in edge_iter:
            self.remove_edge(each_edge)

    def remove_edges_with_attr(self, edge_attr_dict):
        remove_edge_list = []
        for each_edge in self.edges:
            satisfy = True
            for each_key, each_val in edge_attr_dict.items():
                if not satisfy:
                    break
                try:
                    if self.edge_attr(each_edge)[each_key] != each_val:
                        satisfy = False
                except KeyError:
                    satisfy = False
            if satisfy:
                remove_edge_list.append(each_edge)
        self.remove_edges(remove_edge_list)

    def remove_subhg(self, subhg):
        ''' remove subhypergraph.
        all of the hyperedges are removed.
        each node of subhg is removed if its degree becomes 0 after removing hyperedges.

        Parameters
        ----------
        subhg : Hypergraph
        '''
        for each_edge in subhg.edges:
            self.remove_edge(each_edge)
        for each_node in subhg.nodes:
            if self.degree(each_node) == 0:
                self.remove_node(each_node)

    def nodes_in_edge(self, edge):
        ''' return an ordered list of nodes in a given edge.

        Parameters
        ----------
        edge : str
            edge whose nodes are returned

        Returns
        -------
        list or set
            ordered list or set of nodes that belong to the edge
        '''
        if edge.startswith('e'):
            return self.nodes_in_edge_dict[edge]
        else:
            adj_node_list = self.hg.adj[edge]
            adj_node_order_list = []
            adj_node_name_list = []
            for each_node in adj_node_list:
                adj_node_order_list.append(adj_node_list[each_node]['order'])
                adj_node_name_list.append(each_node)
            if adj_node_order_list == [-1] * len(adj_node_order_list):
                return set(adj_node_name_list)
            else:
                return [adj_node_name_list[each_idx] for each_idx
                        in np.argsort(adj_node_order_list)]

    def adj_edges(self, node):
        ''' return a dict of adjacent hyperedges

        Parameters
        ----------
        node : str

        Returns
        -------
        set
            set of edges that are adjacent to `node`
        '''
        return self.hg.adj[node]

    def adj_nodes(self, node):
        ''' return a set of adjacent nodes

        Parameters
        ----------
        node : str

        Returns
        -------
        set
            set of nodes that are adjacent to `node`
        '''
        node_set = set([])
        for each_adj_edge in self.adj_edges(node):
            node_set.update(set(self.nodes_in_edge(each_adj_edge)))
        node_set.discard(node)
        return node_set

    def has_edge(self, node_list, ignore_order=False):
        for each_edge in self.edges:
            if ignore_order:
                if set(self.nodes_in_edge(each_edge)) == set(node_list):
                    return each_edge
            else:
                if self.nodes_in_edge(each_edge) == node_list:
                    return each_edge
        return False

    def degree(self, node):
        return len(self.hg.adj[node])

    def degrees(self):
        return {each_node: self.degree(each_node) for each_node in self.nodes}

    def edge_degree(self, edge):
        return len(self.nodes_in_edge(edge))

    def edge_degrees(self):
        return {each_edge: self.edge_degree(each_edge) for each_edge in self.edges}

    def is_adj(self, node1, node2):
        return node1 in self.adj_nodes(node2)

    def adj_subhg(self, node, ident_node_dict=None):
        """ return a subhypergraph consisting of a set of nodes and hyperedges adjacent to `node`.
        if an adjacent node has a self-loop hyperedge, it will be also added to the subhypergraph.

        Parameters
        ----------
        node : str
        ident_node_dict : dict
            dict containing identical nodes. see `get_identical_node_dict` for more details

        Returns
        -------
        subhg : Hypergraph
        """
        if ident_node_dict is None:
            ident_node_dict = self.get_identical_node_dict()
        adj_node_set = set(ident_node_dict[node])
        adj_edge_set = set([])
        for each_node in ident_node_dict[node]:
            adj_edge_set.update(set(self.adj_edges(each_node)))
        fixed_adj_edge_set = deepcopy(adj_edge_set)
        for each_edge in fixed_adj_edge_set:
            other_nodes = self.nodes_in_edge(each_edge)
            adj_node_set.update(other_nodes)

            # if the adjacent node has self-loop edge, it will be appended to adj_edge_list.
            for each_node in other_nodes:
                for other_edge in set(self.adj_edges(each_node)) - set([each_edge]):
                    if len(set(self.nodes_in_edge(other_edge)) \
                           - set(self.nodes_in_edge(each_edge))) == 0:
                        adj_edge_set.update(set([other_edge]))
        subhg = Hypergraph()
        for each_node in adj_node_set:
            subhg.add_node(each_node, attr_dict=self.node_attr(each_node))
        for each_edge in adj_edge_set:
            subhg.add_edge(self.nodes_in_edge(each_edge),
                           attr_dict=self.edge_attr(each_edge),
                           edge_name=each_edge)
        subhg.edge_idx = self.edge_idx
        return subhg

    def get_subhg(self, node_list, edge_list, ident_node_dict=None):
        """ return a subhypergraph consisting of a set of nodes and hyperedges adjacent to `node`.
        if an adjacent node has a self-loop hyperedge, it will be also added to the subhypergraph.

        Parameters
        ----------
        node : str
        ident_node_dict : dict
            dict containing identical nodes. see `get_identical_node_dict` for more details

        Returns
        -------
        subhg : Hypergraph
        """
        if ident_node_dict is None:
            ident_node_dict = self.get_identical_node_dict()
        adj_node_set = set([])
        for each_node in node_list:
            adj_node_set.update(set(ident_node_dict[each_node]))
        adj_edge_set = set(edge_list)

        subhg = Hypergraph()
        for each_node in adj_node_set:
            subhg.add_node(each_node,
                           attr_dict=deepcopy(self.node_attr(each_node)))
        for each_edge in adj_edge_set:
            subhg.add_edge(self.nodes_in_edge(each_edge),
                           attr_dict=deepcopy(self.edge_attr(each_edge)),
                           edge_name=each_edge)
        subhg.edge_idx = self.edge_idx
        return subhg

    def copy(self):
        ''' return a copy of the object
        
        Returns
        -------
        Hypergraph
        '''
        return deepcopy(self)

    def node_attr(self, node):
        return self.hg.nodes[node]['attr_dict']

    def edge_attr(self, edge):
        return self.hg.nodes[edge]['attr_dict']

    def set_node_attr(self, node, attr_dict):
        for each_key, each_val in attr_dict.items():
            self.hg.nodes[node]['attr_dict'][each_key] = each_val

    def set_edge_attr(self, edge, attr_dict):
        for each_key, each_val in attr_dict.items():
            self.hg.nodes[edge]['attr_dict'][each_key] = each_val

    def get_identical_node_dict(self):
        ''' get identical nodes
        nodes are identical if they share the same set of adjacent edges.
        
        Returns
        -------
        ident_node_dict : dict
            ident_node_dict[node] returns a list of nodes that are identical to `node`.
        '''
        ident_node_dict = {}
        for each_node in self.nodes:
            ident_node_list = []
            for each_other_node in self.nodes:
                if each_other_node == each_node:
                    ident_node_list.append(each_other_node)
                elif self.adj_edges(each_node) == self.adj_edges(each_other_node) \
                   and len(self.adj_edges(each_node)) != 0:
                    ident_node_list.append(each_other_node)
            ident_node_dict[each_node] = ident_node_list
        return ident_node_dict
    '''
        ident_node_dict = {}
        for each_node in self.nodes:
            ident_node_dict[each_node] = [each_node]
        return ident_node_dict
    '''

    def get_leaf_edge(self):
        ''' get an edge that is incident only to one edge

        Returns
        -------
        if exists, return a leaf edge. otherwise, return None.
        '''
        for each_edge in self.edges:
            if len(self.adj_nodes(each_edge)) == 1:
                if 'tmp' not in self.edge_attr(each_edge):
                    return each_edge
        return None

    def get_nontmp_edge(self):
        for each_edge in self.edges:
            if 'tmp' not in self.edge_attr(each_edge):
                return each_edge
        return None

    def is_subhg(self, hg):
        ''' return whether this hypergraph is a subhypergraph of `hg`

        Returns
        -------
        True if self \in hg,
        False otherwise.
        '''
        for each_node in self.nodes:
            if each_node not in hg.nodes:
                return False
        for each_edge in self.edges:
            if each_edge not in hg.edges:
                return False
        return True

    def in_cycle(self, node, visited=None, parent='', root_node='') -> bool:
        ''' if `node` is in a cycle, then return True. otherwise, False.

        Parameters
        ----------
        node : str
            node in a hypergraph
        visited : list
            list of visited nodes, used for recursion
        parent : str
            parent node, used to eliminate a cycle consisting of two nodes and one edge.

        Returns
        -------
        bool
        '''
        if visited is None:
            visited = []
        if parent == '':
            visited = []
        if root_node == '':
            root_node = node
        visited.append(node)
        for each_adj_node in self.adj_nodes(node):
            if each_adj_node not in visited:
                if self.in_cycle(each_adj_node, visited, node, root_node):
                    return True
            elif each_adj_node != parent and each_adj_node == root_node:
                return True
        return False

    
    def draw(self, file_path=None, with_node=False, with_edge_name=False):
        ''' draw hypergraph
        '''
        import graphviz
        G = graphviz.Graph(format='png')
        for each_node in self.nodes:
            if 'ext_id' in self.node_attr(each_node):
                G.node(each_node, label='',
                       shape='circle', width='0.1', height='0.1', style='filled',
                       fillcolor='black')
            else:
                if with_node:
                    G.node(each_node, label='',
                           shape='circle', width='0.1', height='0.1', style='filled',
                           fillcolor='gray')
        edge_list = []
        for each_edge in self.edges:
            if self.edge_attr(each_edge).get('terminal', False):
                G.node(each_edge,
                       label=self.edge_attr(each_edge)['symbol'].symbol if not with_edge_name \
                       else self.edge_attr(each_edge)['symbol'].symbol + ', ' + each_edge,
                       fontcolor='black', shape='square')
            elif self.edge_attr(each_edge).get('tmp', False):
                G.node(each_edge, label='tmp' if not with_edge_name else 'tmp, ' + each_edge,
                       fontcolor='black', shape='square')
            else:
                G.node(each_edge,
                       label=self.edge_attr(each_edge)['symbol'].symbol if not with_edge_name \
                       else self.edge_attr(each_edge)['symbol'].symbol + ', ' + each_edge,
                       fontcolor='black', shape='square', style='filled')
            if with_node:
                for each_node in self.nodes_in_edge(each_edge):
                    G.edge(each_edge, each_node)
            else:
                for each_node in self.nodes_in_edge(each_edge):
                    if 'ext_id' in self.node_attr(each_node)\
                       and set([each_node, each_edge]) not in edge_list:
                        G.edge(each_edge, each_node)
                        edge_list.append(set([each_node, each_edge]))
                for each_other_edge in self.adj_nodes(each_edge):
                    if set([each_edge, each_other_edge]) not in edge_list:
                        num_bond = 0
                        common_node_set = set(self.nodes_in_edge(each_edge))\
                                          .intersection(set(self.nodes_in_edge(each_other_edge)))
                        for each_node in common_node_set:
                            if self.node_attr(each_node)['symbol'].bond_type in [1, 2, 3]:
                                num_bond += self.node_attr(each_node)['symbol'].bond_type
                            elif self.node_attr(each_node)['symbol'].bond_type in [12]:
                                num_bond += 1
                            else:
                                raise NotImplementedError('unsupported bond type')
                        for _ in range(num_bond):
                            G.edge(each_edge, each_other_edge)
                        edge_list.append(set([each_edge, each_other_edge]))
        if file_path is not None:
            G.render(file_path, cleanup=True)
            #os.remove(file_path)
        return G

    def is_dividable(self, node):
        _hg = deepcopy(self.hg)
        _hg.remove_node(node)
        return (not nx.is_connected(_hg))

    def divide(self, node):
        subhg_list = []

        hg_wo_node = deepcopy(self)
        hg_wo_node.remove_node(node, remove_connected_edges=False)
        connected_components = nx.connected_components(hg_wo_node.hg)
        for each_component in connected_components:
            node_list = [node]
            edge_list = []
            node_list.extend([each_node for each_node in each_component
                              if each_node.startswith('bond_')])
            edge_list.extend([each_edge for each_edge in each_component
                              if each_edge.startswith('e')])
            subhg_list.append(self.get_subhg(node_list, edge_list))
            #subhg_list[-1].set_node_attr(node, {'divided': True})
        return subhg_list

