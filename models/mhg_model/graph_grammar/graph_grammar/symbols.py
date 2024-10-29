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

from typing import List

class TSymbol(object):

    ''' terminal symbol

    Attributes
    ----------
    degree : int
        the number of nodes in a hyperedge
    is_aromatic : bool
        whether or not the hyperedge is in an aromatic ring
    symbol : str
        atomic symbol
    num_explicit_Hs : int
        the number of hydrogens associated to this hyperedge
    formal_charge : int
        charge
    chirality : int
        chirality
    '''

    def __init__(self, degree, is_aromatic,
                 symbol, num_explicit_Hs, formal_charge, chirality):
        self.degree = degree
        self.is_aromatic = is_aromatic
        self.symbol = symbol
        self.num_explicit_Hs = num_explicit_Hs
        self.formal_charge = formal_charge
        self.chirality = chirality

    @property
    def terminal(self):
        return True

    def __eq__(self, other):
        if not isinstance(other, TSymbol):
            return False
        if self.degree != other.degree:
            return False
        if self.is_aromatic != other.is_aromatic:
            return False
        if self.symbol != other.symbol:
            return False
        if self.num_explicit_Hs != other.num_explicit_Hs:
            return False
        if self.formal_charge != other.formal_charge:
            return False
        if self.chirality != other.chirality:
            return False
        return True

    def __hash__(self):
        return self.__str__().__hash__()

    def __str__(self):
        return f'degree={self.degree}, is_aromatic={self.is_aromatic}, '\
            f'symbol={self.symbol}, '\
            f'num_explicit_Hs={self.num_explicit_Hs}, '\
            f'formal_charge={self.formal_charge}, chirality={self.chirality}'


class NTSymbol(object):

    ''' non-terminal symbol

    Attributes
    ----------
    degree : int
        degree of the hyperedge
    is_aromatic : bool
        if True, at least one of the associated bonds must be aromatic.
    node_aromatic_list : list of bool
        indicate whether each of the nodes is aromatic or not.
    bond_type_list : list of int
        bond type of each node"
    '''

    def __init__(self, degree: int, is_aromatic: bool,
                 bond_symbol_list: list,
                 for_ring=False):
        self.degree = degree
        self.is_aromatic = is_aromatic
        self.for_ring = for_ring
        self.bond_symbol_list = bond_symbol_list

    @property
    def terminal(self) -> bool:
        return False

    @property
    def symbol(self):
        return f'NT{self.degree}'

    def __eq__(self, other) -> bool:
        if not isinstance(other, NTSymbol):
            return False

        if self.degree != other.degree:
            return False
        if self.is_aromatic != other.is_aromatic:
            return False
        if self.for_ring != other.for_ring:
            return False
        if len(self.bond_symbol_list) != len(other.bond_symbol_list):
            return False
        for each_idx in range(len(self.bond_symbol_list)):
            if self.bond_symbol_list[each_idx] != other.bond_symbol_list[each_idx]:
                return False
        return True

    def __hash__(self):
        return self.__str__().__hash__()

    def __str__(self) -> str:
        return f'degree={self.degree}, is_aromatic={self.is_aromatic}, '\
            f'bond_symbol_list={[str(each_symbol) for each_symbol in self.bond_symbol_list]}'\
            f'for_ring={self.for_ring}'


class BondSymbol(object):
    

    ''' Bond symbol

    Attributes
    ----------
    is_aromatic : bool
        if True, at least one of the associated bonds must be aromatic.
    bond_type : int
        bond type of each node"
    '''

    def __init__(self, is_aromatic: bool,
                 bond_type: int,
                 stereo: int):
        self.is_aromatic = is_aromatic
        self.bond_type = bond_type
        self.stereo = stereo

    def __eq__(self, other) -> bool:
        if not isinstance(other, BondSymbol):
            return False

        if self.is_aromatic != other.is_aromatic:
            return False
        if self.bond_type != other.bond_type:
            return False
        if self.stereo != other.stereo:
            return False
        return True

    def __hash__(self):
        return self.__str__().__hash__()

    def __str__(self) -> str:
        return f'is_aromatic={self.is_aromatic}, '\
            f'bond_type={self.bond_type}, '\
            f'stereo={self.stereo}, '
