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
__copyright__ = "(c) Copyright IBM Corp. 2017"
__version__ = "0.1"
__date__ = "Dec 11 2017"

from copy import deepcopy
from itertools import combinations
from ..hypergraph import Hypergraph
import networkx as nx
import numpy as np


class CliqueTree(nx.Graph):
    ''' clique tree object

    Attributes
    ----------
    hg : Hypergraph
        This hypergraph will be decomposed.
    root_hg : Hypergraph
        Hypergraph on the root node.
    ident_node_dict : dict
        ident_node_dict[key_node] gives a list of nodes that are identical (i.e., the adjacent hyperedges are common)
    '''
    def __init__(self, hg=None, **kwargs):
        self.hg = deepcopy(hg)
        if self.hg is not None:
            self.ident_node_dict = self.hg.get_identical_node_dict()
        else:
            self.ident_node_dict = {}
        super().__init__(**kwargs)

    @property
    def root_hg(self):
        ''' return the hypergraph on the root node
        '''
        return self.nodes[0]['subhg']

    @root_hg.setter
    def root_hg(self, hypergraph):
        ''' set the hypergraph on the root node
        '''
        self.nodes[0]['subhg'] = hypergraph

    def insert_subhg(self, subhypergraph: Hypergraph) -> None:
        ''' insert a subhypergraph, which is extracted from a root hypergraph, into the tree.

        Parameters
        ----------
        subhg : Hypergraph
        '''
        num_nodes = self.number_of_nodes()
        self.add_node(num_nodes, subhg=subhypergraph)
        self.add_edge(num_nodes, 0)
        adj_nodes = deepcopy(list(self.adj[0].keys()))
        for each_node in adj_nodes:
            if len(self.nodes[each_node]["subhg"].nodes.intersection(
                    self.nodes[num_nodes]["subhg"].nodes)\
                   - self.root_hg.nodes) != 0 and each_node != num_nodes:
                self.remove_edge(0, each_node)
                self.add_edge(each_node, num_nodes)

    def to_irredundant(self) -> None:
        ''' convert the clique tree to be irredundant
        '''
        for each_node in self.hg.nodes:
            subtree = self.subgraph([
                each_tree_node for each_tree_node in self.nodes()\
                if each_node in self.nodes[each_tree_node]["subhg"].nodes]).copy()
            leaf_node_list = [x for x in subtree.nodes() if subtree.degree(x)==1]
            redundant_leaf_node_list = []
            for each_leaf_node in leaf_node_list:
                if len(self.nodes[each_leaf_node]["subhg"].adj_edges(each_node)) == 0:
                    redundant_leaf_node_list.append(each_leaf_node)
            for each_red_leaf_node in redundant_leaf_node_list:
                current_node = each_red_leaf_node
                while subtree.degree(current_node) == 1 \
                      and len(subtree.nodes[current_node]["subhg"].adj_edges(each_node)) == 0:
                    self.nodes[current_node]["subhg"].remove_node(each_node)
                    remove_node = current_node
                    current_node = list(dict(subtree[remove_node]).keys())[0]
                    subtree.remove_node(remove_node)

        fixed_node_set = deepcopy(self.nodes)
        for each_node in fixed_node_set:
            if self.nodes[each_node]["subhg"].num_edges == 0:
                if len(self[each_node]) == 1:
                    self.remove_node(each_node)
                elif len(self[each_node]) == 2:
                    self.add_edge(*self[each_node])
                    self.remove_node(each_node)
                else:
                    pass
            else:
                pass

        redundant = True
        while redundant:
            redundant = False
            fixed_edge_set = deepcopy(self.edges)
            remove_node_set = set()
            for node_1, node_2 in fixed_edge_set:
                if node_1 in remove_node_set or node_2 in remove_node_set:
                    pass
                else:
                    if self.nodes[node_1]['subhg'].is_subhg(self.nodes[node_2]['subhg']):
                        redundant = True
                        adj_node_list = set(self.adj[node_1]) - {node_2}
                        self.remove_node(node_1)
                        remove_node_set.add(node_1)
                        for each_node in adj_node_list:
                            self.add_edge(node_2, each_node)

                    elif self.nodes[node_2]['subhg'].is_subhg(self.nodes[node_1]['subhg']):
                        redundant = True
                        adj_node_list = set(self.adj[node_2]) - {node_1}
                        self.remove_node(node_2)
                        remove_node_set.add(node_2)
                        for each_node in adj_node_list:
                            self.add_edge(node_1, each_node)

    def node_update(self, key_node: str, subhg) -> None:
        """ given a pair of a hypergraph, H, and its subhypergraph, sH, return a hypergraph H\sH.

        Parameters
        ----------
        key_node : str
            key node that must be removed.
        subhg : Hypegraph
        """
        for each_edge in subhg.edges:
            self.root_hg.remove_edge(each_edge)
        self.root_hg.remove_nodes(self.ident_node_dict[key_node])

        adj_node_list = list(subhg.nodes)
        for each_node in subhg.nodes:
            if each_node not in self.ident_node_dict[key_node]:
                if set(self.root_hg.adj_edges(each_node)).issubset(subhg.edges):
                    self.root_hg.remove_node(each_node)
                    adj_node_list.remove(each_node)
            else:
                adj_node_list.remove(each_node)

        for each_node_1, each_node_2 in combinations(adj_node_list, 2):
            if not self.root_hg.is_adj(each_node_1, each_node_2):
                self.root_hg.add_edge(set([each_node_1, each_node_2]), attr_dict=dict(tmp=True))
        
        subhg.remove_edges_with_attr({'tmp' : True})
        self.insert_subhg(subhg)

    def update(self, subhg, remove_nodes=False):
        """ given a pair of a hypergraph, H, and its subhypergraph, sH, return a hypergraph H\sH.

        Parameters
        ----------
        subhg : Hypegraph
        """
        for each_edge in subhg.edges:
            self.root_hg.remove_edge(each_edge)
        if remove_nodes:
            remove_edge_list = []
            for each_edge in self.root_hg.edges:
                if set(self.root_hg.nodes_in_edge(each_edge)).issubset(subhg.nodes)\
                   and self.root_hg.edge_attr(each_edge).get('tmp', False):
                    remove_edge_list.append(each_edge)
            self.root_hg.remove_edges(remove_edge_list)

        adj_node_list = list(subhg.nodes)
        for each_node in subhg.nodes:
            if self.root_hg.degree(each_node) == 0:
                self.root_hg.remove_node(each_node)
                adj_node_list.remove(each_node)

        if len(adj_node_list) != 1 and not remove_nodes:
            self.root_hg.add_edge(set(adj_node_list), attr_dict=dict(tmp=True))
        '''
        else:
            for each_node_1, each_node_2 in combinations(adj_node_list, 2):
                if not self.root_hg.is_adj(each_node_1, each_node_2):
                    self.root_hg.add_edge(
                        [each_node_1, each_node_2], attr_dict=dict(tmp=True))
        '''
        subhg.remove_edges_with_attr({'tmp':True})
        self.insert_subhg(subhg)


def _get_min_deg_node(hg, ident_node_dict: dict, mode='mol'):
    if mode == 'standard':
        degree_dict = hg.degrees()
        min_deg_node = min(degree_dict, key=degree_dict.get)
        min_deg_subhg = hg.adj_subhg(min_deg_node, ident_node_dict)
        return min_deg_node, min_deg_subhg
    elif mode == 'mol':
        degree_dict = hg.degrees()
        min_deg = min(degree_dict.values())
        min_deg_node_list = [each_node for each_node in hg.nodes if degree_dict[each_node]==min_deg]
        min_deg_subhg_list = [hg.adj_subhg(each_min_deg_node, ident_node_dict)
                              for each_min_deg_node in min_deg_node_list]
        best_score = np.inf
        best_idx = -1
        for each_idx in range(len(min_deg_subhg_list)):
            if min_deg_subhg_list[each_idx].num_nodes < best_score:
                best_idx = each_idx
        return min_deg_node_list[each_idx], min_deg_subhg_list[each_idx]
    else:
        raise ValueError


def tree_decomposition(hg, irredundant=True):
    """ compute a tree decomposition of the input hypergraph

    Parameters
    ----------
    hg : Hypergraph
        hypergraph to be decomposed
    irredundant : bool
        if True, irredundant tree decomposition will be computed.

    Returns
    -------
    clique_tree : nx.Graph
        each node contains a subhypergraph of `hg`
    """
    org_hg = hg.copy()
    ident_node_dict = hg.get_identical_node_dict()
    clique_tree = CliqueTree(org_hg)
    clique_tree.add_node(0, subhg=org_hg)
    while True:
        degree_dict = org_hg.degrees()
        min_deg_node = min(degree_dict, key=degree_dict.get)
        min_deg_subhg = org_hg.adj_subhg(min_deg_node, ident_node_dict)
        if org_hg.nodes == min_deg_subhg.nodes:
            break

        # org_hg and min_deg_subhg are divided
        clique_tree.node_update(min_deg_node, min_deg_subhg)

    clique_tree.root_hg.remove_edges_with_attr({'tmp' : True})

    if irredundant:
        clique_tree.to_irredundant()
    return clique_tree


def tree_decomposition_with_hrg(hg, hrg, irredundant=True, return_root=False):
    ''' compute a tree decomposition given a hyperedge replacement grammar.
    the resultant clique tree should induce a less compact HRG.
    
    Parameters
    ----------
    hg : Hypergraph
        hypergraph to be decomposed
    hrg : HyperedgeReplacementGrammar
        current HRG
    irredundant : bool
        if True, irredundant tree decomposition will be computed.

    Returns
    -------
    clique_tree : nx.Graph
        each node contains a subhypergraph of `hg`
    '''
    org_hg = hg.copy()
    ident_node_dict = hg.get_identical_node_dict()
    clique_tree = CliqueTree(org_hg)
    clique_tree.add_node(0, subhg=org_hg)
    root_node = 0
    
    # construct a clique tree using HRG
    success_any = True
    while success_any:
        success_any = False
        for each_prod_rule in hrg.prod_rule_list:
            org_hg, success, subhg = each_prod_rule.revert(org_hg, True)
            if success:
                if each_prod_rule.is_start_rule: root_node = clique_tree.number_of_nodes()
                success_any = True
                subhg.remove_edges_with_attr({'terminal' : False})
                clique_tree.root_hg = org_hg
                clique_tree.insert_subhg(subhg)
    
    clique_tree.root_hg = org_hg
    
    for each_edge in deepcopy(org_hg.edges):
        if not org_hg.edge_attr(each_edge)['terminal']:
            node_list = org_hg.nodes_in_edge(each_edge)
            org_hg.remove_edge(each_edge)
            
            for each_node_1, each_node_2 in combinations(node_list, 2):
                if not org_hg.is_adj(each_node_1, each_node_2):
                    org_hg.add_edge([each_node_1, each_node_2], attr_dict=dict(tmp=True))

    # construct a clique tree using the existing algorithm
    degree_dict = org_hg.degrees()
    if degree_dict:
        while True:
            min_deg_node, min_deg_subhg = _get_min_deg_node(org_hg, ident_node_dict)
            if org_hg.nodes == min_deg_subhg.nodes: break

            # org_hg and min_deg_subhg are divided
            clique_tree.node_update(min_deg_node, min_deg_subhg)

    clique_tree.root_hg.remove_edges_with_attr({'tmp' : True})
    if irredundant:
        clique_tree.to_irredundant()

    if return_root:
        if root_node == 0 and 0 not in clique_tree.nodes:
            root_node = clique_tree.number_of_nodes()
            while root_node not in clique_tree.nodes:
                root_node -= 1
        elif root_node not in clique_tree.nodes:
            while root_node not in clique_tree.nodes:
                root_node -= 1
        else:
            pass
        return clique_tree, root_node
    else:
        return clique_tree


def tree_decomposition_from_leaf(hg, irredundant=True):
    """ compute a tree decomposition of the input hypergraph

    Parameters
    ----------
    hg : Hypergraph
        hypergraph to be decomposed
    irredundant : bool
        if True, irredundant tree decomposition will be computed.

    Returns
    -------
    clique_tree : nx.Graph
        each node contains a subhypergraph of `hg`
    """
    def apply_normal_decomposition(clique_tree):
        degree_dict = clique_tree.root_hg.degrees()
        min_deg_node = min(degree_dict, key=degree_dict.get)
        min_deg_subhg = clique_tree.root_hg.adj_subhg(min_deg_node, clique_tree.ident_node_dict)
        if clique_tree.root_hg.nodes == min_deg_subhg.nodes:
            return clique_tree, False
        clique_tree.node_update(min_deg_node, min_deg_subhg)
        return clique_tree, True

    def apply_min_edge_deg_decomposition(clique_tree):
        edge_degree_dict = clique_tree.root_hg.edge_degrees()
        non_tmp_edge_list = [each_edge for each_edge in clique_tree.root_hg.edges \
                             if not clique_tree.root_hg.edge_attr(each_edge).get('tmp')]
        if not non_tmp_edge_list:
            return clique_tree, False
        min_deg_edge = None
        min_deg = np.inf
        for each_edge in non_tmp_edge_list:
            if min_deg > edge_degree_dict[each_edge]:
                min_deg_edge = each_edge
                min_deg = edge_degree_dict[each_edge]
        node_list = clique_tree.root_hg.nodes_in_edge(min_deg_edge)
        min_deg_subhg = clique_tree.root_hg.get_subhg(
            node_list, [min_deg_edge], clique_tree.ident_node_dict)
        if clique_tree.root_hg.nodes == min_deg_subhg.nodes:
            return clique_tree, False
        clique_tree.update(min_deg_subhg)
        return clique_tree, True

    org_hg = hg.copy()
    clique_tree = CliqueTree(org_hg)
    clique_tree.add_node(0, subhg=org_hg)

    success = True
    while success:
        clique_tree, success = apply_min_edge_deg_decomposition(clique_tree)
        if not success:
            clique_tree, success = apply_normal_decomposition(clique_tree)

    clique_tree.root_hg.remove_edges_with_attr({'tmp' : True})
    if irredundant:
        clique_tree.to_irredundant()
    return clique_tree

def topological_tree_decomposition(
        hg, irredundant=True, rip_labels=True, shrink_cycle=False, contract_cycles=False):
    ''' compute a tree decomposition of the input hypergraph

    Parameters
    ----------
    hg : Hypergraph
        hypergraph to be decomposed
    irredundant : bool
        if True, irredundant tree decomposition will be computed.

    Returns
    -------
    clique_tree : CliqueTree
        each node contains a subhypergraph of `hg`
    '''
    def _contract_tree(clique_tree):
        ''' contract a single leaf

        Parameters
        ----------
        clique_tree : CliqueTree

        Returns
        -------
        CliqueTree, bool
            bool represents whether this operation succeeds or not.
        '''
        edge_degree_dict = clique_tree.root_hg.edge_degrees()
        leaf_edge_list = [each_edge for each_edge in clique_tree.root_hg.edges \
                          if (not clique_tree.root_hg.edge_attr(each_edge).get('tmp'))\
                          and edge_degree_dict[each_edge] == 1]
        if not leaf_edge_list:
            return clique_tree, False
        min_deg_edge = leaf_edge_list[0]
        node_list = clique_tree.root_hg.nodes_in_edge(min_deg_edge)
        min_deg_subhg = clique_tree.root_hg.get_subhg(
            node_list, [min_deg_edge], clique_tree.ident_node_dict)
        if clique_tree.root_hg.nodes == min_deg_subhg.nodes:
            return clique_tree, False
        clique_tree.update(min_deg_subhg)
        return clique_tree, True

    def _rip_labels_from_cycles(clique_tree, org_hg):
        ''' rip hyperedge-labels off

        Parameters
        ----------
        clique_tree : CliqueTree
        org_hg : Hypergraph

        Returns
        -------
        CliqueTree, bool
            bool represents whether this operation succeeds or not.
        '''
        ident_node_dict = clique_tree.ident_node_dict #hg.get_identical_node_dict()
        for each_edge in clique_tree.root_hg.edges:
            if each_edge in org_hg.edges:
                if org_hg.in_cycle(each_edge):
                    node_list = clique_tree.root_hg.nodes_in_edge(each_edge)
                    subhg = clique_tree.root_hg.get_subhg(
                        node_list, [each_edge], ident_node_dict)
                    if clique_tree.root_hg.nodes == subhg.nodes:
                        return clique_tree, False
                    clique_tree.update(subhg)
                    '''
                    in_cycle_dict = {each_node: org_hg.node_attr(each_node)['is_in_ring'] for each_node in node_list}
                    if not all(in_cycle_dict.values()):
                        node_not_in_cycle = [each_node for each_node in in_cycle_dict.keys() if not in_cycle_dict[each_node]][0]
                        node_list = [node_not_in_cycle]
                        node_list.extend(clique_tree.root_hg.adj_nodes(node_not_in_cycle))
                        edge_list = clique_tree.root_hg.adj_edges(node_not_in_cycle)
                        import pdb; pdb.set_trace()
                        subhg = clique_tree.root_hg.get_subhg(
                            node_list, edge_list, ident_node_dict)
                        
                        clique_tree.update(subhg)
                    '''
                    return clique_tree, True
        return clique_tree, False

    def _shrink_cycle(clique_tree):
        ''' shrink a cycle

        Parameters
        ----------
        clique_tree : CliqueTree

        Returns
        -------
        CliqueTree, bool
            bool represents whether this operation succeeds or not.
        '''
        def filter_subhg(subhg, hg, key_node):
            num_nodes_cycle = 0
            nodes_in_cycle_list = []
            for each_node in subhg.nodes:
                if hg.in_cycle(each_node):
                    num_nodes_cycle += 1
                    if each_node != key_node:
                        nodes_in_cycle_list.append(each_node)
                if num_nodes_cycle > 3:
                    break
            if num_nodes_cycle != 3:
                return False
            else:
                for each_edge in hg.edges:
                    if set(nodes_in_cycle_list).issubset(hg.nodes_in_edge(each_edge)):
                        return False
                return True

        #ident_node_dict = hg.get_identical_node_dict()
        ident_node_dict = clique_tree.ident_node_dict
        for each_node in clique_tree.root_hg.nodes:
            if clique_tree.root_hg.in_cycle(each_node)\
               and filter_subhg(clique_tree.root_hg.adj_subhg(each_node, ident_node_dict),
                                clique_tree.root_hg,
                                each_node):
                target_node = each_node
                target_subhg = clique_tree.root_hg.adj_subhg(target_node, ident_node_dict)
                if clique_tree.root_hg.nodes == target_subhg.nodes:
                    return clique_tree, False
                clique_tree.update(target_subhg)
                return clique_tree, True
        return clique_tree, False

    def _contract_cycles(clique_tree):
        '''
        remove a subhypergraph that looks like a cycle on a leaf.

        Parameters
        ----------
        clique_tree : CliqueTree

        Returns
        -------
        CliqueTree, bool
            bool represents whether this operation succeeds or not.
        '''
        def _divide_hg(hg):
            ''' divide a hypergraph into subhypergraphs such that
            each subhypergraph is connected to each other in a tree-like way.

            Parameters
            ----------
            hg : Hypergraph

            Returns
            -------
            list of Hypergraphs
                each element corresponds to a subhypergraph of `hg`
            '''
            for each_node in hg.nodes:
                if hg.is_dividable(each_node):
                    adj_edges_dict = {each_edge: hg.in_cycle(each_edge) for each_edge in hg.adj_edges(each_node)}
                    '''
                    if any(adj_edges_dict.values()):
                        import pdb; pdb.set_trace()
                        edge_in_cycle = [each_key for each_key, each_val in adj_edges_dict.items() if each_val][0]
                        subhg1, subhg2, subhg3 = hg.divide(each_node, edge_in_cycle)
                        return _divide_hg(subhg1) + _divide_hg(subhg2) + _divide_hg(subhg3)
                    else:
                    '''
                    subhg1, subhg2 = hg.divide(each_node)
                    return _divide_hg(subhg1) + _divide_hg(subhg2)
            return [hg]

        def _is_leaf(hg, divided_subhg) -> bool:
            ''' judge whether subhg is a leaf-like in the original hypergraph

            Parameters
            ----------
            hg : Hypergraph
            divided_subhg : Hypergraph
                `divided_subhg` is a subhypergraph of `hg`

            Returns
            -------
            bool
            '''
            '''
            adj_edges_set = set([])
            for each_node in divided_subhg.nodes:
                adj_edges_set.update(set(hg.adj_edges(each_node)))


            _hg = deepcopy(hg)
            _hg.remove_subhg(divided_subhg)
            if nx.is_connected(_hg.hg) != (len(adj_edges_set - divided_subhg.edges) == 1):
                import pdb; pdb.set_trace()
            return len(adj_edges_set - divided_subhg.edges) == 1
            '''
            _hg = deepcopy(hg)
            _hg.remove_subhg(divided_subhg)
            return nx.is_connected(_hg.hg)
        
        subhg_list = _divide_hg(clique_tree.root_hg)
        if len(subhg_list) == 1:
            return clique_tree, False
        else:
            while len(subhg_list) > 1:
                max_leaf_subhg = None
                for each_subhg in subhg_list:
                    if _is_leaf(clique_tree.root_hg, each_subhg):
                        if max_leaf_subhg is None:
                            max_leaf_subhg = each_subhg
                        elif max_leaf_subhg.num_nodes < each_subhg.num_nodes:
                            max_leaf_subhg = each_subhg
                clique_tree.update(max_leaf_subhg)
                subhg_list.remove(max_leaf_subhg)
            return clique_tree, True

    org_hg = hg.copy()
    clique_tree = CliqueTree(org_hg)
    clique_tree.add_node(0, subhg=org_hg)

    success = True
    while success:
        '''
        clique_tree, success = _rip_labels_from_cycles(clique_tree, hg)
        if not success:
            clique_tree, success = _contract_cycles(clique_tree)
        '''
        clique_tree, success = _contract_tree(clique_tree)
        if not success:
            if rip_labels:
                clique_tree, success = _rip_labels_from_cycles(clique_tree, hg)
            if not success:
                if shrink_cycle:
                    clique_tree, success = _shrink_cycle(clique_tree)
                if not success:
                    if contract_cycles:
                        clique_tree, success = _contract_cycles(clique_tree)
    clique_tree.root_hg.remove_edges_with_attr({'tmp' : True})
    if irredundant:
        clique_tree.to_irredundant()
    return clique_tree

def molecular_tree_decomposition(hg, irredundant=True):
    """ compute a tree decomposition of the input molecular hypergraph

    Parameters
    ----------
    hg : Hypergraph
        molecular hypergraph to be decomposed
    irredundant : bool
        if True, irredundant tree decomposition will be computed.

    Returns
    -------
    clique_tree : CliqueTree
        each node contains a subhypergraph of `hg`
    """
    def _divide_hg(hg):
        ''' divide a hypergraph into subhypergraphs such that
        each subhypergraph is connected to each other in a tree-like way.

        Parameters
        ----------
        hg : Hypergraph

        Returns
        -------
        list of Hypergraphs
            each element corresponds to a subhypergraph of `hg`
        '''
        is_ring = False
        for each_node in hg.nodes:
            if hg.node_attr(each_node)['is_in_ring']:
                is_ring = True
            if not hg.node_attr(each_node)['is_in_ring'] \
               and hg.degree(each_node) == 2:
                subhg1, subhg2 = hg.divide(each_node)
                return _divide_hg(subhg1) + _divide_hg(subhg2)

        if is_ring:
            subhg_list = []
            remove_edge_list = []
            remove_node_list = []
            for each_edge in hg.edges:
                node_list = hg.nodes_in_edge(each_edge)
                subhg = hg.get_subhg(node_list, [each_edge], hg.get_identical_node_dict())
                subhg_list.append(subhg)
                remove_edge_list.append(each_edge)
                for each_node in node_list:
                    if not hg.node_attr(each_node)['is_in_ring']:
                        remove_node_list.append(each_node)
            hg.remove_edges(remove_edge_list)
            hg.remove_nodes(remove_node_list, False)
            return subhg_list + [hg]
        else:
            return [hg]

    org_hg = hg.copy()
    clique_tree = CliqueTree(org_hg)
    clique_tree.add_node(0, subhg=org_hg)

    subhg_list = _divide_hg(deepcopy(clique_tree.root_hg))
    #_subhg_list = deepcopy(subhg_list)
    if len(subhg_list) == 1:
        pass
    else:
        while len(subhg_list) > 1:
            max_leaf_subhg = None
            for each_subhg in subhg_list:
                if _is_leaf(clique_tree.root_hg, each_subhg) and not _is_ring(each_subhg):
                    if max_leaf_subhg is None:
                        max_leaf_subhg = each_subhg
                    elif max_leaf_subhg.num_nodes < each_subhg.num_nodes:
                        max_leaf_subhg = each_subhg

            if max_leaf_subhg is None:
                for each_subhg in subhg_list:
                    if _is_ring_label(clique_tree.root_hg, each_subhg):
                        if max_leaf_subhg is None:
                            max_leaf_subhg = each_subhg
                        elif max_leaf_subhg.num_nodes < each_subhg.num_nodes:
                            max_leaf_subhg = each_subhg
            if max_leaf_subhg is not None:
                clique_tree.update(max_leaf_subhg)
                subhg_list.remove(max_leaf_subhg)
            else:
                for each_subhg in subhg_list:
                    if _is_leaf(clique_tree.root_hg, each_subhg):
                        if max_leaf_subhg is None:
                            max_leaf_subhg = each_subhg
                        elif max_leaf_subhg.num_nodes < each_subhg.num_nodes:
                            max_leaf_subhg = each_subhg
                if max_leaf_subhg is not None:
                    clique_tree.update(max_leaf_subhg, True)
                    subhg_list.remove(max_leaf_subhg)
                else:
                    break
    if len(subhg_list) > 1:
        '''
        for each_idx, each_subhg in enumerate(subhg_list):
            each_subhg.draw(f'{each_idx}', True)
        clique_tree.root_hg.draw('root', True)
        import pickle
        with open('buggy_hg.pkl', 'wb') as f:
            pickle.dump(hg, f)
        return clique_tree, subhg_list, _subhg_list
        '''
        raise RuntimeError('bug in tree decomposition algorithm')
    clique_tree.root_hg.remove_edges_with_attr({'tmp' : True})

    '''
    for each_tree_node in clique_tree.adj[0]:
        subhg = clique_tree.nodes[each_tree_node]['subhg']
        for each_edge in subhg.edges:
            if set(subhg.nodes_in_edge(each_edge)).issubset(clique_tree.root_hg.nodes):
                clique_tree.root_hg.add_edge(set(subhg.nodes_in_edge(each_edge)), attr_dict=dict(tmp=True))
    '''
    if irredundant:
        clique_tree.to_irredundant()
    return clique_tree #, _subhg_list

def _is_leaf(hg, subhg) -> bool:
    ''' judge whether subhg is a leaf-like in the original hypergraph

    Parameters
    ----------
    hg : Hypergraph
    subhg : Hypergraph
        `subhg` is a subhypergraph of `hg`

    Returns
    -------
    bool
    '''
    if len(subhg.edges) == 0:
        adj_edge_set = set([])
        subhg_edge_set = set([])
        for each_edge in hg.edges:
            if set(hg.nodes_in_edge(each_edge)).issubset(subhg.nodes) and hg.edge_attr(each_edge).get('tmp', False):
                subhg_edge_set.add(each_edge)
        for each_node in subhg.nodes:
            adj_edge_set.update(set(hg.adj_edges(each_node)))
        if subhg_edge_set.issubset(adj_edge_set) and len(adj_edge_set.difference(subhg_edge_set)) == 1:
            return True
        else:
            return False
    elif len(subhg.edges) == 1:
        adj_edge_set = set([])
        subhg_edge_set = subhg.edges
        for each_node in subhg.nodes:
            for each_adj_edge in hg.adj_edges(each_node):
                adj_edge_set.add(each_adj_edge)
        if subhg_edge_set.issubset(adj_edge_set) and len(adj_edge_set.difference(subhg_edge_set)) == 1:
            return True
        else:
            return False
    else:
        raise ValueError('subhg should be nodes only or one-edge hypergraph.')

def _is_ring_label(hg, subhg):
    if len(subhg.edges) != 1:
        return False
    edge_name = list(subhg.edges)[0]
    #assert edge_name in hg.edges, f'{edge_name}'
    is_in_ring = False
    for each_node in subhg.nodes:
        if subhg.node_attr(each_node)['is_in_ring']:
            is_in_ring = True
        else:
            adj_edge_list = list(hg.adj_edges(each_node))
            adj_edge_list.remove(edge_name)
            if len(adj_edge_list) == 1:
                if not hg.edge_attr(adj_edge_list[0]).get('tmp', False):
                    return False
            elif len(adj_edge_list) == 0:
                pass
            else:
                raise ValueError
    if is_in_ring:
        return True
    else:
        return False

def _is_ring(hg):
    for each_node in hg.nodes:
        if not hg.node_attr(each_node)['is_in_ring']:
            return False
    return True        

