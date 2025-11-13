# -*- coding:utf-8 -*-
"""
ChemGraph.py

Package for IBM Molecule Generation Experience

MIT License

Copyright (c) 2022 International Business Machines Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
PLEASE NOTE THIS IMPLEMENTATION INCLUDES ADAPTED SOURCE CODE 
OF THE MOLGX ALREADY PULICLY AVAILABLE
"""


from rdkit import Chem
from rdkit.Chem.rdchem import BondType, BondDir, BondStereo, ChiralType
from rdkit import RDLogger

from operator import attrgetter
from collections import Counter, defaultdict
import copy
import heapq
import re
import itertools

import logging
logger = logging.getLogger(__name__)
rdlogger = RDLogger.logger()


class ChemVertex(object):
    """Base class for a vertex in a chemical graph.

    Attributes:
        index (int): unique index of a vertex in a graph starting from 0
        edges (list): a list of edges starting from a vertex
        hyper_edges (list): a list of hyper_edges starting from a vertex
        label (int): index of a vertex labelled by canonical labeling
        visit (int): pre-order index during depth first search of the graph
        exact_match_fragment (set): a set of matched fragments including a vertex
    """

    wild_card_atom = '*'
    """str: dummy atom symbol of rdkit"""

    ATOM_DICT = {}
    """dict: mapping from atom name to Chem.Atom object"""

    @staticmethod
    def atom_to_symbol(atom, charge):
        """Get a symbol of an atom with charge

        Args:
            atom (str): atom name
            charge (int): charge

        Returns:
            str: symbol of an atom
        """
        if charge == 0:
            symbol = atom
        elif charge == 1:
            symbol = atom + '+'
        elif charge == -1:
            symbol = atom + '-'
        elif charge > 0:
            symbol = '{0}{1}{2}'.format(atom, '+', charge)
        else: # charge > 0
            symbol = '{0}{1}{2}'.format(atom, '-', -charge)
        return symbol

    @staticmethod
    def symbol_to_atom(symbol):
        """Get an atom name and charge from atom symbol

        Args:
            symbol (str): symbol of atom including charge

        Returns:
            tuple: a tuple of atom name and charge
        """
        match = re.match(r"([\w*]+)([+-]*)(\d*)", symbol)
        if match:
            atom = match.group(1)
            sign = match.group(2)
            value = match.group(3)
            if sign == '+':
                if value == '':
                    charge = 1
                else:
                    charge = 1 * int(value)
            elif sign == '-':
                if value == '':
                    charge = -1
                else:
                    charge = -1 * int(value)
            else:
                charge = 0
            return atom, charge
        else:
            return '', 0

    @classmethod
    def register_vertex(cls, atom):
        """Generate atom info from atom name and register it with a key of atom name.

        Args:
            atom (str): atom symbol including charge

        Returns:
            registered Chem.Atom object for an atom name
        """
        if atom not in cls.ATOM_DICT:
            try:
                (base_atom, charge) = cls.symbol_to_atom(atom)
                mw = Chem.RWMol()
                rd_atom = Chem.Atom(base_atom)
                rd_atom.SetFormalCharge(charge)
                mw.AddAtom(rd_atom)
                mol = mw.GetMol()
                mol.UpdatePropertyCache()
                rd_atom = mol.GetAtomWithIdx(0)
                cls.ATOM_DICT[atom] = (base_atom, charge, rd_atom.GetTotalValence())
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('register atom:%s success valence=%d',
                                 rd_atom.GetSymbol(), rd_atom.GetTotalValence())
            except AttributeError:
                logger.error('register atom:unknown atom %s', atom)
                cls.ATOM_DICT[atom] = None
            except:
                logger.error('register atom:unknown atom %s', atom)
                cls.ATOM_DICT[atom] = None
        return cls.ATOM_DICT[atom]

    def __init__(self, index):
        """Constructor of ChemVertex with a specified index.

        Args:
            index (int): index of a vertex
        """
        self.index = index
        self.edges = []
        self.hyper_edges = []
        self.label = 0
        self.visit = 0
        self.exact_match_fragment = set()

    def clear(self):
        """Clear all the attributes except index.
        """
        self.index = 0
        while len(self.edges) > 0:
            self.pop_edge()
        while len(self.hyper_edges) > 0:
            self.pop_hyper_edge()
        self.label = 0
        self.visit = 0
        self.exact_match_fragment.clear()

    def add_edge(self, e):
        """Add an edge starting from a vertex.

        Args:
            e (ChemEdge): an edge starting from a vertex
        """
        self.edges.append(e)

    def add_normal_edge(self, e):
        """Add an edge starting from a vertex.

        Args:
            e (ChemEdge): an edge starting from a vertex
        """
        self.edges.append(e)

    def add_hyper_edge(self, e):
        """Add a hyper_edge starting from a vertex.

        Args:
            e (ChemEdge): an edge starting from a vertex
        """
        self.hyper_edges.append(e)

    def pop_edge(self, edge=None):
        """Pop an edge added last if edge is not specified. Otherwise remove a specified edge.

        Args:
            edge (ChemEdge, optional): a edge starting from a vertex. Defaults to None.
        """
        if edge is None:
            return self.edges.pop()
        else:
            self.edges.remove(edge)
            return edge

    def pop_hyper_edge(self, edge=None):
        """Pop a hyper_edge added last if edge is not specified. Otherwise remove a specified edge.

        Args:
            edge (ChemEdge, optional): a edge starting from a vertex. Defaults to None.
        """
        if edge is None:
            return self.hyper_edges.pop()
        else:
            self.hyper_edges.remove(edge)
            return edge

    def num_edge(self):
        """Get the number of edges.

        Returns:
            int: the number of edge
        """
        return len(self.edges)

    def num_bond(self):
        """Get the number of bonds.

        Returns:
            int: the number of bonds
        """
        return sum(e.get_bond_order() for e in self.edges)

    def num_bond2(self):
        """Get the squared number of bonds for canonical labeling.

        Returns:
            int: square of the number of bonds
        """
        return sum(e.get_bond_num()*e.get_bond_num() for e in self.edges)

    def num_hyper_edge(self):
        """Get the number of hyper_edge.

        Returns:
            int: the number of hyper_edges
        """
        return len(self.hyper_edges)

    def get_exact_match_fragment(self):
        """Get a set of exactly matched fragments.

        Returns:
             set: exactly matched fragments
        """
        return self.exact_match_fragment

    def bond_available(self, bond_type):
        raise NotImplementedError('ChemVertex:bond_available()')

    def num_atom(self):
        raise NotImplementedError('ChemVertex:num_atom()')

    def num_ring_atom(self):
        raise NotImplementedError('ChemVertex:num_ring_atom()')

    def num_aromatic_atom(self):
        raise NotImplementedError('ChemVertex:num_aromatic_atom()')

    def num_valence(self):
        raise NotImplementedError('ChemVertex:num_valance()')

    def color(self):
        raise NotImplementedError('ChemVertex:color()')


class ChemEdge(object):
    """Base class for an edge in a chemical graph.

    Attributes:
        start (ChemVertex): start vertex of an edge
        end (ChemVertex): end vertex of an edge
        bond_type (BondType): bond type of an edge
        direction (int): direction of an edge (0: forward, 1:backward in case of dative bond)
        alternative_bond (set): a set of alternative bond type for matching
    """

    def __init__(self, start, end, bond_type=BondType.SINGLE, direction=0):
        """Constructor of ChemEdge.

        Args:
            start (ChemVertex): staring vertex
            end (ChemVertex): ending vertex
            bond_type (BondType, optional): bond type. Defaults to BondType.SINGLE.
            direction (int, optional): direction of an edge (0:forward, 1:backward). Defaults to 0.
        """
        self.start = start
        self.end = end
        self.bond_type = bond_type
        self.direction = direction
        self.alternative_bond = None

    def get_reverse_edge(self):
        """Get an edge of reverse direction

        Returns:
            ChemEdge: a reverse edge
        """
        for e in self.end.edges:
            if e.end == self.start:
                return e
        logger.error('no reverse edge')
        return None

    def get_bond_num(self):
        """Get number representation of bond for canonical labeling

         Returns:
             float: number representation of bond
         """
        if self.alternative_bond is None:
            return self.get_bond_order()
        else:
            val = self.get_bond_order()
            for b in self.alternative_bond:
                bo = ChemGraph.get_bond_order(b)
                val += (bo/10.0)*(bo/10.0)
            return val

    def get_bond_order(self):
        """Get an order of the bond type (1 for SINGLE, 1.5 for AROMATIC)

        Returns:
            float: bond order
        """
        return ChemGraph.get_bond_order(self.bond_type)

    def set_alternative_bond(self, alt):
        """Set alternative edge types for matching sub-graph

        Args:
            alt (list): a set of alternative bond types
        """
        self.alternative_bond = \
            set([BondType.UNSPECIFIED if x == '*' else ChemGraph.BOND_NAME[x] for x in alt])

    def match_bond(self, e):
        """Check the matching to given edge

        Args:
            e (edge): an edge
        """
        if self.bond_type == e.bond_type and self.direction == e.direction:
            return True
        elif self.alternative_bond is not None:
            if e.bond_type in self.alternative_bond or \
                    BondType.UNSPECIFIED in self.alternative_bond:
                return True
            else:
                return False
        else:
            return False


class ChemGraph(object):
    """Base calss for a graph of a molecule.

    Attributes:
        vertices (list): a list of vertices
        ring_vertices (list): a list of vertices representing a ring
        graph_vertices (list): a list of vertices representing a sub-structure
    """

    BOND_ORDER = {
        BondType.UNSPECIFIED: 0,
        BondType.IONIC: 0,
        BondType.ZERO: 0,
        BondType.SINGLE: 1,
        BondType.DOUBLE: 2,
        BondType.TRIPLE: 3,
        BondType.QUADRUPLE: 4,
        BondType.QUINTUPLE: 5,
        BondType.HEXTUPLE: 6,
        BondType.ONEANDAHALF: 1.5,
        BondType.TWOANDAHALF: 2.5,
        BondType.THREEANDAHALF: 3.5,
        BondType.FOURANDAHALF: 4.5,
        BondType.FIVEANDAHALF: 5.5,
        BondType.AROMATIC: 1.5,
        BondType.DATIVEONE: 1.0,
        BondType.DATIVE: 1.0,
    }
    """dict: rdkit mapping from bond_type to bond order"""

    BOND_NAME = {
        BondType.UNSPECIFIED.name: BondType.UNSPECIFIED,
        BondType.IONIC.name: BondType.IONIC,
        BondType.ZERO.name: BondType.ZERO,
        BondType.SINGLE.name: BondType.SINGLE,
        BondType.DOUBLE.name: BondType.DOUBLE,
        BondType.TRIPLE.name: BondType.TRIPLE,
        BondType.QUADRUPLE.name: BondType.QUADRUPLE,
        BondType.QUINTUPLE.name: BondType.QUINTUPLE,
        BondType.HEXTUPLE.name: BondType.HEXTUPLE,
        BondType.ONEANDAHALF.name: BondType.ONEANDAHALF,
        BondType.TWOANDAHALF.name: BondType.TWOANDAHALF,
        BondType.THREEANDAHALF.name: BondType.THREEANDAHALF,
        BondType.FOURANDAHALF.name: BondType.FOURANDAHALF,
        BondType.FIVEANDAHALF.name: BondType.FIVEANDAHALF,
        BondType.AROMATIC.name: BondType.AROMATIC,
        BondType.DATIVEONE.name: BondType.DATIVEONE,
        BondType.DATIVE.name: BondType.DATIVE,
    }
    """dict: rdkit mapping from bond name to bond_type"""

    BOND_CHAR = {
        BondType.SINGLE: '',
        BondType.DOUBLE: '=',
        BondType.TRIPLE: '#',
        BondType.AROMATIC: ':',
        BondType.DATIVE: '>',
    }
    """dict: mapping from bond to display character"""

    CHAR_BOND = {
        '': BondType.SINGLE,
        '=': BondType.DOUBLE,
        '#': BondType.TRIPLE,
        ':': BondType.AROMATIC,
        '>': BondType.DATIVE,
    }
    """dict: mapping from display charcter to bond"""

    @classmethod
    def canonical_smiles(cls, smiles):
        """Translating given SMILES to canonical SMILES using rdkit.

        Args:
            smiles (str): SMILES

        Returns:
            str: canonical SMILES
        """
        try:
            return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))
        except Exception as err:
            logger.error('canonical_smiles: wrong smiles=%s', smiles)
            logger.error('canonical_smiles: %s', err)
        return None

    @classmethod
    def canonical_smarts(cls, smiles):
        """Translating given SMILES to canonical SMILES using rdkit.

        Args:
            smiles (str): SMILES

        Returns:
            str: canonical SMARTS
        """
        try:
            return Chem.MolToSmarts(Chem.MolFromSmiles(smiles, sanitize=False))
        except Exception as err:
            logger.error('canonical_smarts: wrong smiles=%s', smiles)
            logger.error('canonical_smarts: %s', err)
        return None

    @classmethod
    def get_bond_order(cls, bond_type):
        """Get bond order from bond type (1 for SINGLE, 1.5 for AROMATIC)

        Args:
            bond_type (BondType): bond type

        Returns:
            float : bond order
        """
        return cls.BOND_ORDER[bond_type]

    @classmethod
    def get_bond_char(cls, bond_type):
        """Get SMILES bond character from bond type

        Args:
            bond_type (BondType): bond type

        Returns:
            str : bond character
        """
        return cls.BOND_CHAR[bond_type]

    @classmethod
    def get_bond_type(cls, bond_char):
        """Get bond type from SMILES bond character

        Args:
            bond_char (str): bond charactter

        Returns:
            BondType: bond type
        """
        return cls.CHAR_BOND[bond_char]

    def __init__(self):
        """Constructor of an empty graph.
        """
        self.vertices = []
        self.ring_vertices = []
        self.graph_vertices = []

    def num_vertices(self):
        """Get the number of vertices.

        Returns:
            int: the number of vertices
        """
        return len(self.vertices)

    def num_ring_graph_vertices(self):
        """Get the number of vertices representing a ring.

        Returns:
            int: the number of vertices representing a ring
        """
        return len(self.ring_vertices)

    def num_graph_graph_vertices(self):
        """Get the number of vertices representing a sub-structure.

        Returns:
            int: the number of vertices representing a sub-structure
        """
        return len(self.graph_vertices)

    def num_all_free_hand(self):
        """Get the total number of free hands to connect a new atom.

        Returns:
            int: total number of free hands
        """
        return sum(v.num_free_hand() for v in self.vertices)

    def num_atom(self):
        """Get the total number of atoms.

        Returns:
            int: total number of atoms
        """
        natom = sum(v.num_atom() for v in self.vertices)
        natom += sum(v.num_atom() for v in self.ring_vertices if not v.expand)
        natom += sum(v.num_atom() for v in self.graph_vertices if not v.expand)
        return natom

    def num_ring_atom(self):
        """Get the total number of atoms composing a ring.

        Returns:
            int: total nubmer of atoms
        """
        natom = sum(v.num_ring_atom() for v in self.vertices)
        natom += sum(v.num_ring_atom() for v in self.ring_vertices if not v.expand)
        natom += sum(v.num_ring_atom() for v in self.graph_vertices if not v.expand)
        return natom

    def num_aromatic_atom(self):
        """Get the total number of atoms composing an aromatic ring.

        Returns:
            int: total number of atoms
        """
        natom = sum(v.num_aromatic_atom() for v in self.vertices)
        natom += sum(v.num_aromatic_atom() for v in self.ring_vertices if not v.expand)
        natom += sum(v.num_aromatic_atom() for v in self.graph_vertices if not v.expand)
        return natom

    def apply_label(self):
        """Replace an index of a vertex with a label of a vertex
        """
        for v in self.vertices:
            v.index = v.label

    def reorder_canonical(self):
        """Reorder the vertices according to the canonical smiles

        Returns:
            list: a list of canonical atom index order
        """
        rdmol = self.to_mol()
        Chem.MolToSmiles(rdmol)
        index_list = [int(idx) for idx in rdmol.GetProp('_smilesAtomOutputOrder').strip('[,]').split(',')]
        self.reorder_vertices([self.vertices[idx] for idx in index_list])
        return index_list

    def reorder_vertices(self, vlist):
        """Reorder the vertices accoring to given list of vertices.

        Args:
            vlist (list): list of ChemVertex
        """
        index = 0
        for v in vlist:
            v.index = index
            index += 1
        self.vertices = sorted(self.vertices, key=lambda vv: vv.index)
        for v in self.vertices:
            v.edges = sorted(v.edges, key=lambda ee: ee.end.index)

    def to_readable_string(self):
        """Get a string representation of a graph, which can be used as an input for a constructor.

        Returns:
            str: string representation of a graph
        """
        str = '{'
        for v in self.vertices:
            str += v.to_readable_string()
            if v != self.vertices[-1]:
                str += ','
        str += '}'
        return str

    def to_string(self):
        """Get a string representation of a graph.

        Returns:
            str: string representation of a graph
        """
        str = 'Graph:{'
        for v in self.vertices:
            str += v.to_string()
            if v != self.vertices[-1]:
                str += ','
        str += '}'
        return str

    def to_string_label(self):
        """Get a string representation of a graph based on labeling resutls.

        Returns:
            str: string representation of a graph
        """
        vertices = sorted(self.vertices,key=attrgetter('label'))
        str = 'Graph(cl):{'
        for v in vertices:
            str += v.to_string_label()
            if v != vertices[-1]:
                str += ','
        str += '}'
        return str

    def to_mol(self):
        """Get a rdkit Chem.Mol object.

        Returns:
            Chem.Mol: mol object
        """
        vertex_map = dict()
        mw = Chem.RWMol()
        for vertex in self.vertices:
            new_atom = Chem.Atom(vertex.base_atom)
            new_atom.SetFormalCharge(vertex.charge)
            if vertex.charge == 0 and vertex.num_free_hand() >= 0:
                new_atom.SetNumExplicitHs(vertex.explicit_h)
            new_atom.SetAtomMapNum(vertex.atom_map_num)
            new_atom_index = mw.AddAtom(new_atom)
            vertex_map[vertex] = new_atom_index
        for vertex in self.vertices:
            vertex.visit = 1
            for e in vertex.edges:
                if e.end.visit == 0 and e.end in vertex_map:
                    mw.AddBond(vertex_map[vertex], vertex_map[e.end], e.bond_type)
        for vertex in self.vertices:
            vertex.visit = 0
        mol = mw.GetMol()
        for vertex in self.vertices:
            if vertex.root > 0:
                mol.GetAtomWithIdx(vertex_map[vertex]).SetUnsignedProp('root', vertex.root - 1)
            if vertex.repeat != 0:
                mol.GetAtomWithIdx(vertex_map[vertex]).SetIntProp('repeat', vertex.repeat)
        mol.UpdatePropertyCache()
        return mol

    def to_smiles(self):
        """Get a SMILES notation of a graph.

        Returns:
            string: SMILES notation of a graph
        """
        mol = self.to_mol()
        return Chem.MolToSmiles(mol)

    def to_smarts(self):
        """Get a SMARTS notation of a graph.

        Returns:
            string: SMARTS notation of a graph
        """
        mol = self.to_mol()
        return Chem.MolToSmarts(mol)

    def debug_check_consistency(self):
        """Check the consistency of vertex and edge relations in a graph for debugging purpose.

        Returns:
            bool: True if consistent, False otherwise
        """
        if logger.isEnabledFor(logging.INFO):
            logger.info('check:%s',self.to_string())
        consistent = True
        vertex_index = set()
        for v in self.vertices:
            vertex_index.add(v.index)
        for v in self.vertices:
            for e in v.edges:
                if e.start != v:
                    logger.error('inconsistent vertex edge')
                    consistent = False
                if e.end.index not in vertex_index:
                    logger.error('unknown end vertex in vertex edge')
                    consistent = False
        for v in self.ring_vertices:
            if v.expand:
                for v0 in v.vertices:
                    if v0.index not in vertex_index:
                        logger.error('unknown ring vertex')
                        consistent = False
                    for e in v0.edges:
                        if e.start != v0:
                            logger.error('inconsistent vertex edge')
                            consistent = False
                        if e.end.index not in vertex_index:
                            logger.error('unknown end vertex in vertex edge')
                            consistent = False
            else:
                if v.connection_vertex.index not in vertex_index:
                    logger.error('unknown connection vertex')
                    consistent = False
                for e in v.connection_vertex.edges:
                    if e.start != v.connection_vertex:
                        logger.error('inconsistent vertex edge')
                        consistent = False
                    if e.end.index not in vertex_index:
                        logger.error('unknown end vertex in vertex edge')
                        consistent = False
        if not consistent:
            logger.error('inconsistent vertex and edge')


class AtomVertex(ChemVertex):
    """Vertex of AtomGraph, which has a valence of an atom as an upper bound of acceptable edges.

    Attributes:
        base_atom: (str): atom name
        atom (str): atom name with charge
        charge (int): charge of an atom
        valence (int): valence of an atom represented by a vertex
        explicit_h (int): number of explicitly specified H connection
        atom_map_num (int): number of atom map of Rdkit Mol
        root (int): depth from a root atom (0 means not root atom)
        repeat (int): identifier of repeated structure
        aromatic (bool): flag of aromatic atom
        in_ring (bool): falg of atom in ring
        atom_color (str): color of atom for labeling
        alternative_atom (set): a set of alternative atom symbols for matching
    """

    def __init__(self, index, atom):
        """Constructor of AtomVertex.

        Args:
            index (int): index of a vertex
            atom (str): atom name of a vertex
        """
        super().__init__(index)
        (base_atom, charge, valence) = self.register_vertex(atom)
        self.base_atom = base_atom
        self.atom = atom
        self.charge = charge
        self.valence = valence
        self.explicit_h = 0
        self.atom_map_num = 0
        self.root = 0
        self.repeat = 0
        self.aromatic = False
        self.in_ring = False
        self.atom_color = atom
        self.alternative_atom = None

    def set_explicit_h(self, explicit_h):
        """Set explicit 'H'

        Args:
            explicit_h (int): the number of explicit 'H'
        """
        self.explicit_h = explicit_h

    def set_color(self, color):
        """Set color of atom

        Args:
            color (str): color of atom
        """
        self.atom_color = color

    def set_root_index(self, index):
        """Set index of root vertex

        Args:
            index (int): root index
        """
        self.root = index

    def set_terminal_index(self, index):
        """Set index of terminal vertex

        Args:
            index (int): terminal index
        """
        self.repeat = index
        self.atom_color = '{0}:{1}'.format(self.atom, index)

    def set_alternative_atom(self, alt):
        """Set alternative atom symbols for matching

        Args:
            alt (list): a list of alternative atom symbols
        """
        self.alternative_atom = set(alt)

    def match_atom(self, v):
        """Matching atom symbol to a give vertex

        Args:
            v (AtomVertex): a vertex
        """
        if self.atom == v.atom:
            return True
        elif self.alternative_atom is not None:
            if v.atom in self.alternative_atom or \
                    ChemVertex.wild_card_atom in self.alternative_atom:
                return True
            else:
                return False
        else:
            return False

    def num_atom(self):
        """Get the number of atom.

        Returns:
            int: 1 constant
        """
        return 1

    def ring_atom(self):
        """Get if it is an atom in ring

        Returns:
            bool: true if atom in ring
        """
        return self.in_ring

    def num_ring_atom(self):
        """Get the number of atom composing a ring.

        Returns:
            int: 1 if atom in ring, 0 otherwise
        """
        return 1 if self.in_ring else 0

    def aromatic_atom(self):
        """Get if it is an aromatic atom

        Returns:
            bool: true if aromatic atom
        """
        return self.aromatic

    def num_aromatic_atom(self):
        """Get the number of aromatic atom.

        Returns:
            int: 1 if aromatic atom, 0 othewise
        """
        return 1 if self.aromatic else 0

    def num_valence(self):
        """Get the number of valence of an atom.

        Returns:
            int: the number of valence
        """
        return self.valence

    def color(self):
        """Get the color of a vertex used in labeling.

        Returns:
            str: color of a vertex
        """
        return self.atom_color

    def num_free_hand(self):
        """Get the number of free hands as valence minus total bonds.

        Returns:
            int: the number of free hands
        """
        return self.valence - int(sum(e.get_bond_order() for e in self.edges))

    def num_all_free_hand(self):
        """Get the number of total free hands (same as num_free_hand()).

        Returns:
            int: the total number of free hands
        """
        return self.num_free_hand()

    def bond_degree(self):
        """Get the total number of bonds

        Returns:
             int: total number of bonds
        """
        return min(int(sum([e.get_bond_order() for e in self.edges])), self.num_valence())

    def get_connect_vertex(self):
        """Get real vertex of connecting to a graph

        Returns:
            AtomVertex: vertex
        """
        return self

    def get_good_bad_atoms(self, bond_type):
        """Get good and bad atoms for bond type

        Args:
            bond_type (BondType): a bond type

        Returns:
            set, set: a set of good atoms, a set of bad atoms
        """
        return None

    def bond_available(self, bond_type):
        """Get availability of adding a new bond of give type.

        Args:
            bond_type (BondType): bond type

        Returns:
            bool: True if bond type is acceptable, False otherwise
        """
        return self.num_free_hand() >= ChemGraph.get_bond_order(bond_type)

    def replace_symbol(self, new_atom):
        """Replace the atom name by specified name with valence checked.

        Args:
            new_atom (str): new atom name for replacement

        Returns:
            AtomVertex: self if new valence is acceptable, None otherwise
        """
        (base_atom, charge, valence) = self.register_vertex(new_atom)
        if valence < int(self.num_bond()):
            # not enough valence for replace
            return None
        else:
            self.atom = new_atom
            self.base_atom = base_atom
            self.charge = charge
            self.valence = valence
            self.atom_color = new_atom
            return self

    def to_readable_string(self):
        """Get string representation of an atom, which can be read by constructor.

        Returns:
            str: string representation of an atom
        """
        str = '['
        for x in self.edges:
            str += "'%d%s'"%(x.end.index, ChemGraph.BOND_CHAR[x.bond_type])
            if x != self.edges[-1]:
                str += ','
        str += ']'
        return "%d:('%s',%s)"%(self.index, self.atom, str)

    def to_string(self):
        """Get string representation of an atom.

        Returns:
            str: string representation of an atom
        """
        str = '['
        for x in self.edges:
            str += '{0}{1}'.format(x.end.index, ChemGraph.BOND_CHAR[x.bond_type])
            if x != self.edges[-1]:
                str += ','
        str += ']'
        if self.repeat > 0:
            return '{0}:({1}*{2},{3})'.format(self.index, self.atom, self.repeat, str)
        else:
            return '{0}:({1},{2})'.format(self.index, self.atom, str)

    def to_string_label(self):
        """Get string representation of an atom with label as index.

        Returns:
            str: string representation of an atom
        """
        vertices = sorted(self.edges, key=lambda x: x.end.label)
        str = '['
        for x in vertices:
            str += '{0}{1}'.format(x.end.label, ChemGraph.BOND_CHAR[x.bond_type])
            if x != vertices[-1]:
                str += ','
        str += ']'
        if self.repeat > 0:
            return '{0}:({1}*{2},{3})'.format(self.label, self.atom, self.repeat, str)
        else:
            return '{0}:({1},{2})'.format(self.label, self.atom, str)


class GraphVertex(AtomVertex):
    """Vertex of AtomGraph, which is composing a sub-structure in molecular generation.

    Attributes:
        graph_index (int): index of a vertex in a sub-structure graph
        graph_edge (int): edges of a vertex in a sub-structure graph
        connect_bond (float): bond order used in a sub-structure graph
        available_bond (Counter): counter of available bonds for connecting a new vertex
        free_count (int): the number of can-be-free connections
        good_atoms (list): a list of sets of good atoms
        bad_atoms (list): a list of sets of bad atoms
    """

    def __init__(self, index, graph_index, atom):
        """Constructor of a graph vertex.

        Args:
            index (int): index of a vertex
            graph_index (int): index of a vertex in a sub-structure graph
            atom (str): atom name of a vertex
        """
        super().__init__(index, atom)
        self.graph_index = graph_index
        self.graph_edges = []
        self.connect_bond = 0
        self.available_bond = Counter()
        self.free_count = 0
        self.good_atoms = defaultdict(list)
        self.bad_atoms = defaultdict(list)

    def add_edge(self, e):
        """Add an edge starting from a vertex. Available bond is also updated.

        Args:
            e (ChemEdge): an edge starting from a vertex
        """
        super().add_edge(e)
        self.available_bond[e.bond_type] -= 1

    def pop_edge(self, e=None):
        """Pop an edge added last if not specified. Remove an edge otherwise. Available bond is also updated.

        Args:
            e (ChemEdge, optional): an edge staring from a vertex. Defaults to None.
        """
        edge = super().pop_edge(e)
        self.available_bond[edge.bond_type] += 1
        return edge

    def add_graph_edge(self, e):
        """Add an edge composing a sub-structure.

        Args:
            e (ChemEdge): an edge composing a sub-structure
        """
        self.graph_edges.append(e)

    def num_free_hand(self):
        """Get the number of free hands. Bonds of connect_bond is already used composing a sub-structure.

        Returns:
            int: the number of free hands
        """
        return self.valence - int(self.connect_bond + sum(e.get_bond_order() for e in self.edges))

    def can_free_hand(self):
        """Get the number of hands that can be free in the completion

        Returns:
            int: the number of can-be-free hands
        """
        return self.free_count

    def get_good_bad_atoms(self, bond_type):
        """Get good and bad atoms for bond type

        Args:
            bond_type (BondType): a bond type

        Returns:
            set, set: a set of good atoms, a set of bad atoms
        """
        index = self.available_bond[bond_type] - 1
        return self.good_atoms[bond_type][index], self.bad_atoms[bond_type][index]

    def bond_available(self, bond_type):
        """Get the availability of bond.

        Args:
            bond_type (BondType): a bond type

        Returns:
            bool: True if bond type is available, False otherwise
        """
        if bond_type in self.available_bond and self.available_bond[bond_type] >= 0:
            return True
        else:
            return False


class RingVertex(AtomVertex):
    """Vertex of AtomGraph, which is composing a ring in molecular generation.

    Attributes:
        ring_index (int): index of a vertex in a ring graph
        ring_edge (int): edges of a vertex in a ring graph
        aromatic (bool): flag of aromatic atom
        connect_bond (float): bond used in a ring graph
    """

    def __init__(self, index, ring_index, atom):
        """Constructor of aring vertex.

        Args:
            index (int): index of a vertex
            ring_index (int): index of a vertex in a ring graph
            atom (str): atom name
        """
        super().__init__(index, atom)
        self.ring_index = ring_index
        self.ring_edges = []
        self.connect_bond = 0

    def add_ring_edge(self, e):
        """Add an edge composing a ring.

        Args:
            e (ChemEdge): an edge composing a ring graph
        """
        self.ring_edges.append(e)

    def num_free_hand(self):
        """Get the number of free hands. Bonds of connect_bond is already used for composing a ring.

        Returns:
            int: the number of free hands
        """
        return self.valence - int(self.connect_bond + sum(e.get_bond_order() for e in self.edges))


class ConnectionVertex(AtomVertex):
    """Vertex of AtomGraph, representing a connection point of a sub-structure.
    If a sub-structure is not expanded in a AtomGraph, this vertex represents a sub-structure.

    Attributes:
        connect (AtomVertex): real connecting vertex in a sub-structure
        graph_valence (int): total valence of a sub-structure
    """

    def __init__(self, index, connect, graph_valence):
        """Constructor for graph connection vertex.

        Args:
            index (int): index of a vertex
            connect (AtomVertex): real connecting vertex
            graph_valence (int): total valence of a sub-structure
        """
        super().__init__(index, '*')
        self.connect = connect
        self.graph_valence = graph_valence

    def bond_available(self, bond_type):
        """Get the availability of the bond of real connecting vertex.

        Returns:
            bool: True if available, False othewise
        """
        return self.connect.bond_available(bond_type)

    def num_free_hand(self):
        """Get the number of free hands of real connecting vertex.

        Returns:
            int: the number of free hands
        """
        return self.connect.num_free_hand()

    def num_all_free_hand(self):
        """Get the total number of free hands of real connecting vertex.

        Returns:
            int: the total number of free hands
        """
        return self.graph_valence - sum(e.get_bond_order() for e in self.edges)

    def num_atom(self):
        """Get the number of atoms.

        Returns:
            int: 1 constant
        """
        return 1

    def get_exact_match_fragment(self):
        """Get a set of exactly matched fragments

        Returns:
            set: a set of exactly matched fragments
        """
        return self.connect.exact_match_fragment

    def get_connect_vertex(self):
        """Get real vertex of connecting to a graph

        Returns:
            AtomVertex: vertex
        """
        return self.connect


class GraphConnectionVertex(ConnectionVertex):
    """Vertex of AtomGraph, representing a connection point of a sub-structure.
    If a sub-structure is not expanded in a AtomGraph, this vertex represents a sub-structure.

    Attributes:
        symbol (str): SMILES of sub-structure as a key to AtomGraph
        connect (AtomVertex): real connecting vertex in a sub-structure
        graph_valence (int): total valence of a sub-structure
    """

    def __init__(self, index, symbol, connect, graph_valence):
        """Constructor for graph connection vertex.

        Args:
            index (int): index of a vertex
            graph (str): SMILES of sub-structure as a key to AtomGraph
            connect (AtomVertex): real connecting vertex
            graph_valence (int): total valence of a sub-structure
        """
        super().__init__(index, connect, graph_valence)
        self.atom = '{0}:{1}'.format(symbol, connect.graph_index)
        if connect.repeat > 0:
            # terminal atom has special name
            # head of normal terminal atom: '  '
            # head of iteration terminal atom: '||'
            self.atom_color = symbol[0:2] + self.atom
            self.repeat = connect.repeat
        else:
            self.atom_color = '||' + self.atom

    def add_edge(self, e):
        """Add an edge starting form a vertex.
        available_bond and connect_bond of real connecting vertex is also update.

        Args:
            e (ChemEdge): an edge stating from a vertex
        """
        super().add_edge(e)
        self.connect.available_bond[e.bond_type] -= 1
        self.connect.connect_bond += e.get_bond_order()

    def pop_edge(self, e=None):
        """Pop an edge if not specified. Remove an edge otherwise.

        Args:
            e (ChemEdge, optional): an edge starting from a vertex. Defaults to None.
        """
        edge = super().pop_edge(e)
        self.connect.available_bond[edge.bond_type] += 1
        self.connect.connect_bond -= edge.get_bond_order()
        return edge


class RingConnectionVertex(ConnectionVertex):
    """Vertex of AtomGraph, representing a connection point of a ring.
    If a ring is not expanded in a AtomGraph, this vertex represents a ring

    Attributes:
        symbol (str): SMILES of ring as a key to AtomGraph
        connect (AtomVertex): real connecting vertex in a sub-structure
        graph_valence (int): total valence of a sub-structure
    """

    def __init__(self, index, symbol, connect, graph_valence):
        """Constructor for ring connection vertex.

        Args:
            index (int): index of a vertex
            ring (str): SMILES of ring as a key to AtomGraph
            connect (AtomVertex): real connecting vertex
            graph_valence (int): total valence of a ring
        """
        super().__init__(index, connect, graph_valence)
        self.atom = '{0}:{1}'.format(symbol, connect.ring_index)
        self.atom_color = '|'+self.atom

    def add_edge(self, e):
        """Add an edge starting form a vertex.
        available_bond and connect_bond of real connecting vertex is also update.

        Args:
            e (ChemEdge): an edge stating from a vertex
        """
        super().add_edge(e)
        self.connect.connect_bond += e.get_bond_order()

    def pop_edge(self, e=None):
        """Pop an edge if not specified. Remove an edge otherwise.

        Args:
            e (ChemEdge, optional): an edge starting from a vertex. Defaults to None.
        """
        edge = super().pop_edge(e)
        self.connect.connect_bond -= edge.get_bond_order()
        return edge


class AtomGraph(ChemGraph):
    """Chemical structure represented as a graph of AtomVertex.
    A single vertex sometimes represents a sub-graph such as a ring and a sub-structure in AtomGraph,
    and such vertex is expanded into real a graph of atom vertices if necessary

    Attributes:
        expanded_ring (list): a list of RingAsVertex expanded to a sub-graph of AtomVertex
        expanded_graph (list): a list of GraphAsVertex expanded to a sub-graph of AtomVertex
        expanded_vertex (list): a list of AtomVertex expanded to a AtomVertex
    """

    def __init__(self, graph=None, smiles=None, smarts=None, mol=None):
        """Constructor of AtomGraph from graph notation, SMILES/SMARTS notation, or RDKIT mol object
        An instance can be initialized in three ways.

        Args:
            graph (str): dict-type string representation of graph
            smiles (str): SMILES representation of a molecule
            smarts (str): SMARTS representation of a molecule
            mol (Chem.Mol): mol object of rdkit

        Example:
            graph notation::
            AtomGraph(graph='{0:('C',[1]), 1:('C',[0])}')
            smiles notation::
            AtomGraph(smiles='CC')
            mol notation::
            AtomGraph(mol=Chem.MolFromSMILES('CC'))
        """
        super().__init__()
        self.expanded_ring = []
        self.expanded_graph = []
        self.expanded_vertex = []
        if graph is not None:
            vertices = {}
            for id, atom in graph.items():
                vertices[id] = self.add_vertex(atom[0])
            for id, atom in graph.items():
                for v in atom[1]:
                    index = v
                    if isinstance(v, str):
                        if v[-1].isdigit():
                            index = int(v)
                            self.add_directed_edge(vertices[id], vertices[index])
                        else:
                            index = int(v[0:-1])
                            bond = v[-1]
                            self.add_directed_edge(vertices[id], vertices[index], ChemGraph.CHAR_BOND[bond])
                    else:
                        self.add_directed_edge(vertices[id], vertices[index])
            # check ring vertex
            self.update_atom_in_ring()
        if smiles is not None:
            try:
                rdlogger.setLevel(RDLogger.CRITICAL)
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                Chem.SanitizeMol(mol)
            except ValueError as verr:
                logger.error('failed to read smiles: %s %s', smiles, verr)
            rdlogger.setLevel(RDLogger.WARNING)
        if smarts is not None:
            try:
                rdlogger.setLevel(RDLogger.CRITICAL)
                mol = Chem.MolFromSmarts(smarts)
            except ValueError as verr:
                logger.error('failed to read smiles: %s %s', smiles, verr)
            rdlogger.setLevel(RDLogger.WARNING)
        if mol is not None:
            vertex_map = dict()
            for atom in mol.GetAtoms():
                atom_symbol = ChemVertex.atom_to_symbol(atom.GetSymbol(), atom.GetFormalCharge())
                new_v = self.add_vertex(atom_symbol)
                new_v.explicit_h = atom.GetNumExplicitHs()
                new_v.atom_map_num = atom.GetAtomMapNum()
                new_v.in_ring = atom.IsInRing()
                vertex_map[atom.GetIdx()] = new_v
                if atom.HasProp('root'):
                    new_v.root = atom.GetUnsignedProp('root')+1
                if atom.HasProp('repeat'):
                    new_v.repeat = atom.GetIntProp('repeat')
            for bond in mol.GetBonds():
                self.add_edge(vertex_map[bond.GetBeginAtomIdx()],
                              vertex_map[bond.GetEndAtomIdx()],
                              bond.GetBondType())
        # sort vertices and edges
        self.vertices = sorted(self.vertices, key=lambda vv: vv.index)
        for v in self.vertices:
            v.edges = sorted(v.edges, key=lambda ee: ee.end.index)

    def get_atom_count(self):
        """Get counter of each atom.

        Returns:
            Counter: a counter of each atom
        """
        atom_count = Counter()
        for v in self.vertices:
            atom_count[v.atom] += 1
        return atom_count

    def get_ring_atom_count(self):
        """Get counter of each ring atom.

        Returns:
            Counter: a counter of each ring atom
        """
        atom_count = Counter()
        for v in self.vertices:
                atom_count[v.atom] += v.num_ring_atom()
        return atom_count

    def get_aromatic_atom_count(self):
        """Get counter of each aromatic atom.

        Returns:
            Counter: a counter of each aromatic atom
        """
        atom_count = Counter()
        for v in self.vertices:
            atom_count[v.atom] += v.num_aromatic_atom()
        return atom_count

    def update_atom_in_ring(self):
        """Update in_ring flag of vertices
        """
        for v in self.vertices:
            v.in_ring = False
        for v in self.get_ring_vertices():
            v.in_ring = True

    def get_ring_vertices(self):
        """Get vertices composing a ring in case graph is not constructed from mol object.

        Returns:
            list: a list of ring vertices
        """
        ring_vertices = set()
        used_vertices = set()
        for v in sorted(self.vertices, key=lambda x:len(x.edges), reverse=True):
            if len(v.edges) <= 1:
                # ring vertex has more than 2 edges
                used_vertices.add(v)
                continue
            elif v in used_vertices:
                # vertex is already scanned
                continue
            v.visit = 1
            dfs_nodes = [[e for e in v.edges]]
            # depth first search
            while len(dfs_nodes) > 0:
                node = dfs_nodes[-1]
                if len(node) == 0:
                    # all children are scanned
                    dfs_nodes.pop()
                    if len(dfs_nodes) > 0:
                        node = dfs_nodes[-1]
                        e = node.pop()
                        e.end.visit = 0
                        used_vertices.add(e.end)
                    continue
                # set last children as current node
                current = node[-1]
                current.end.visit = 1
                children = []
                for e in current.end.edges:
                    if e.end == current.start:
                        # ignore reverse direction
                        continue
                    elif e.end.visit > 0:
                        # search tree closed
                        # collect ring vertices of closed path
                        path_edge = e
                        index = -1
                        while path_edge.start != e.end:
                            ring_vertices.add(path_edge.end)
                            path_edge = dfs_nodes[index][-1]
                            index -= 1
                    else:
                        children.append(e)
                # add new node
                dfs_nodes.append(children)
            v.visit = 0
        return ring_vertices

    def get_connected_ring_vertices(self):
        """Get a lit of connected ring vertices.

        Returns:
            list: a list of connected ring vertices
        """
        ring_vertices_set = []
        used_vertices = set()
        for v in sorted(self.vertices, key=lambda x:len(x.edges), reverse=True):
            if len(v.edges) <= 1:
                # ring vertex has more than 2 edges
                used_vertices.add(v)
                continue
            elif v in used_vertices:
                # vertex is already scanned
                continue
            ring_vertices = set()
            v.visit = 1
            dfs_nodes = [[e for e in v.edges]]
            # depth first search
            while len(dfs_nodes) > 0:
                node = dfs_nodes[-1]
                if len(node) == 0:
                    # all children are scanned
                    dfs_nodes.pop()
                    if len(dfs_nodes) > 0:
                        node = dfs_nodes[-1]
                        e = node.pop()
                        e.end.visit = 0
                        used_vertices.add(e.end)
                    continue
                # set last children as current node
                current = node[-1]
                current.end.visit = 1
                children = []
                for e in current.end.edges:
                    if e.end == current.start:
                        # ignore reverse direction
                        continue
                    elif e.end.visit > 0:
                        # search tree closed
                        # collect ring vertices of closed path
                        path_edge = e
                        index = -1
                        while path_edge.start != e.end:
                            ring_vertices.add(path_edge.end)
                            path_edge = dfs_nodes[index][-1]
                            index -= 1
                        ring_vertices.add(path_edge.end)
                        ring_vertices_set.append(ring_vertices)
                        ring_vertices = set()
                    else:
                        children.append(e)
                # add new node
                dfs_nodes.append(children)
            v.visit = 0
        # make unique sets of ring vertices
        unique_ring_sets = []
        for ring in ring_vertices_set:
            if all([ring != r for r in unique_ring_sets]):
                unique_ring_sets.append(ring)
        unique_ring_sets.sort(key=lambda x: len(x), reverse=True)
        # make disjoint sets of ring vertices
        disjoint = False
        while not disjoint:
            disjoint = True
            new_ring_sets = []
            for i in range(len(unique_ring_sets)):
                for j in range(0, i):
                    if unique_ring_sets[i].isdisjoint(unique_ring_sets[j]):
                        new_ring_sets.append(unique_ring_sets[j])
                    else:
                        new_ring_sets.append(unique_ring_sets[i] | unique_ring_sets[j])
                        disjoint = False
                for j in range(i + 1, len(unique_ring_sets)):
                    if unique_ring_sets[i].isdisjoint(unique_ring_sets[j]):
                        new_ring_sets.append(unique_ring_sets[j])
                    else:
                        new_ring_sets.append(unique_ring_sets[i] | unique_ring_sets[j])
                        disjoint = False
                if not disjoint:
                    unique_ring_sets = new_ring_sets
                    break
        unique_ring_lists = []
        for ring in sorted(unique_ring_sets, key=lambda x:len(x)):
            unique_ring_lists.append(sorted(list(ring), key=lambda x:x.index))
        return unique_ring_lists

    def get_connected_ring_graphs(self):
        """Get a list of connected ring subgraphs

        Returns:
            list: a list of connect ring subgraphs
        """
        ring_lists = self.get_connected_ring_vertices()
        # add exceptional atom attached to aromatic ring
        for ring in ring_lists:
            additional_atoms = []
            for vertex in ring:
                if vertex.aromatic_atom() and vertex.num_free_hand() < 0:
                    for e in vertex.edges:
                        if e.end not in ring:
                            additional_atoms.append(e.end)
            ring.extend(additional_atoms)
        # make graphs of disjoint rings
        ring_graphs = []
        for ring in ring_lists:
            (ring_graph, ring_vertex_map) = self.make_subgraph(ring)
            ring_graphs.append(ring_graph)
        return ring_graphs

    def count_rings_sssr(self):
        """Count the number of ring by SSSR.
        if rings share more than one bonds, count all the rings not SSSR.

        Returns:
            dict, dict: a map of ring size and its count, a map of aromatic ring and its count
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('sssr graph:{0}'.format(self.to_string()))
        num_ring = sum(len(v.edges) for v in self.vertices)/2 - len(self.vertices) + 1
        used_vertex = Counter()
        used_edge = Counter()
        rings = Counter()
        aromatic = Counter()
        found_path_vertices = defaultdict(list)

        # find smallest rings phase 1
        smallest_ring_paths = []
        for v in sorted(self.vertices, key=lambda x:len(x.edges), reverse=True):
            if len(v.edges) <= 1:
                # ring vertex has more than 2 edges
                continue
            elif v in used_vertex:
                # vertex is already scanned
                continue
            # start depth first search
            v.visit = 1
            dfs_nodes = [[e for e in v.edges]]
            smallest_ring_path = self.search_smallest_ring(dfs_nodes, used_vertex, used_edge)
            v.visit = 0
            if len(smallest_ring_path) > 0:
                # check duplication
                path_vertices = sorted([e0.end.index for e0 in smallest_ring_path])
                duplicate = False
                for old_path_vertices in found_path_vertices[len(path_vertices)]:
                    if path_vertices == old_path_vertices:
                        duplicate = True
                        break
                if not duplicate:
                    # smallest ring is found
                    found_path_vertices[len(path_vertices)].append(path_vertices)
                    self.update_used_counter(smallest_ring_path, used_vertex, used_edge)
                    smallest_ring_paths.append(smallest_ring_path)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('ph1 start={0} ring={1}'.
                                     format(v.index, [e0.end.index for e0 in smallest_ring_path]))
                    rings[len(smallest_ring_path)] += 1
                    if all(e.start.aromatic for e in smallest_ring_path):
                        aromatic[len(smallest_ring_path)] += 1

        # find smallest rings phase2
        # follows from not used edges
        used_vertex = Counter()
        not_used_edges = []
        for v in self.vertices:
            for e in v.edges:
                if e.start.index < e.end.index and e in used_edge and used_edge[e] == 0:
                    not_used_edges.append(e)
        for e in not_used_edges:
            e.start.visit = 1
            dfs_nodes = [[e]]
            smallest_ring_path = self.search_smallest_ring(dfs_nodes, used_vertex, used_edge)
            e.start.visit = 0
            if len(smallest_ring_path) > 0:
                # check duplication
                path_vertices = sorted([e0.end.index for e0 in smallest_ring_path])
                duplicate = False
                for old_path_vertices in found_path_vertices[len(path_vertices)]:
                    if path_vertices == old_path_vertices:
                        duplicate = True
                        break
                if not duplicate:
                    # smallest ring is found
                    found_path_vertices[len(path_vertices)].append(path_vertices)
                    self.update_used_counter(smallest_ring_path, used_vertex, used_edge)
                    smallest_ring_paths.append(smallest_ring_path)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('ph2 start=({0}-{1}) ring={2}'.format(e.start.index, e.end.index, [e0.end.index for e0 in smallest_ring_path]))
                    rings[len(smallest_ring_path)] += 1
                    if all(e.start.aromatic for e in smallest_ring_path):
                        aromatic[len(smallest_ring_path)] += 1

        # check alternative ring path
        for smallest_ring_path in smallest_ring_paths:
            # check if rings shares more than one edges
            another_ring_paths = self.search_another_ring(smallest_ring_path, used_edge)
            for another_ring_path in another_ring_paths:
                # check duplication
                path_vertices = sorted([e0.end.index for e0 in another_ring_path])
                duplicate = False
                for old_path_vertices in found_path_vertices[len(path_vertices)]:
                    if path_vertices == old_path_vertices:
                        duplicate = True
                        break
                if not duplicate:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('orig={0} another ring={1}'.
                                     format([e.end.index for e in smallest_ring_path],
                                            [e.end.index for e in another_ring_path]))
                    found_path_vertices[len(path_vertices)].append(path_vertices)
                    rings[len(another_ring_path)] += 1
                    if all(e.start.aromatic for e in another_ring_path):
                        aromatic[len(another_ring_path)] += 1
                    num_ring += 1

        if sum(rings.values()) >= num_ring:
            return rings, aromatic

        # find smallest ring phase3
        once_used_edges = []
        for v in self.vertices:
            for e in v.edges:
                if e.start.index < e.end.index and used_edge[e] == 1:
                    once_used_edges.append(e)
        found_path = []
        for e in once_used_edges:
            # set used edge count 2
            used_edge[e] = 2
            used_edge[e.get_reverse_edge()] = 2
            e.start.visit = 1
            dfs_nodes = [[e]]
            smallest_ring_path = self.search_smallest_ring(dfs_nodes, used_vertex, used_edge, phase3=True)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('ph3 start={0} ring={1}'.format(v.index, [e.end.index for e in smallest_ring_path]))
            e.start.visit = 0
            if len(smallest_ring_path) > 0:
                found_path.append(smallest_ring_path)
                self.update_used_counter(smallest_ring_path, used_vertex, used_edge)
        for path in sorted(found_path, key=lambda x: len(x)):
            rings[len(path)] += 1
            if all(e.start.aromatic for e in path):
                aromatic[len(path)] += 1
            if sum(rings.values()) >= num_ring:
                break
        return rings, aromatic

    def update_used_counter(self, path, used_vertex, used_edge):
        """Update used vertex/edge counter in the path.

        Args:
            path (list): a list of edges
            used_vertex (Counter): counter of used vertex
            used_edge (Counter): counter of used edge
        """
        for e in path:
            used_vertex[e.start] += 1
            used_edge[e] += 1
            used_edge[e.get_reverse_edge()] += 1

    def search_smallest_ring(self, dfs_nodes, used_vertex, used_edge, phase3=False):
        """Find the smallest ring starting from given edges.

        Args:
            def_nodes (list): node of depth first search (a list of a list of edges)
            used_vertex (Counter): counter of used vertex
            used_edge (Counter): counter of used edge
            phase3 (bool, optional): extention for phase3 SSSR algorith. Defaults to False.
        Returns:
             (list): a list of edges as a ring path
        """
        smallest_ring_path = []
        # depth first search
        while len(dfs_nodes) > 0:
            node = dfs_nodes[-1]
            if len(node) == 0:
                # all children are scanned
                dfs_nodes.pop()
                if len(dfs_nodes) > 0:
                    node = dfs_nodes[-1]
                    e = node.pop()
                    e.end.visit = 0
                continue
            # set last children as current node
            current = node[-1]
            current.end.visit = 1
            children = []
            for e in current.end.edges:
                if e.end == current.start:
                    # ignore reverse direction
                    continue
                elif phase3 and \
                        ( len(e.start.edges) < 3 or
                          len(e.end.edges) < 3 ):
                    # in phase3 ignore edge with small degrees
                    continue
                elif e.end.visit > 0:
                    # search tree closed
                    # check if returned to the start vertex
                    if e.end != dfs_nodes[0][-1].start:
                        continue
                    # make a path of a ring
                    path = []
                    path_edge = e
                    index = -1
                    while path_edge.start != e.end:
                        path.append(path_edge)
                        path_edge = dfs_nodes[index][-1]
                        index -= 1
                    path.append(path_edge)
                    if phase3:
                        # check edge usage count
                        if len([e for e in path if used_edge[e] >= 2]) > 1:
                            continue
                    # mark edge in a ring
                    for e in path:
                        if used_edge[e] == 0:
                            used_edge[e] = 0
                            used_edge[e.get_reverse_edge()] = 0
                    # check is path is smallest ring
                    if len(smallest_ring_path) == 0:
                        smallest_ring_path = path
                    elif len(smallest_ring_path) > len(path):
                        smallest_ring_path = path
                    elif len(smallest_ring_path) == len(path):
                        # compare the used vertex counter
                        smallest_vertex_used = sum(used_vertex[e.start] for e in smallest_ring_path)
                        vertex_used = sum(used_vertex[e.start] for e in path)
                        if smallest_vertex_used > vertex_used:
                            smallest_ring_path = path
                        elif smallest_vertex_used == vertex_used:
                            # compare the used edge counter
                            smallest_edge_used = sum(used_edge[e] for e in smallest_ring_path)
                            edge_used = sum(used_edge[e] for e in path)
                            if smallest_edge_used < edge_used:
                                smallest_ring_path = path
                            elif smallest_edge_used == edge_used:
                                # compare the degree
                                smallest_connectivity = sum(len(e.start.edges) for e in smallest_ring_path)
                                connectivity = sum(len(e.start.edges) for e in path)
                                if smallest_connectivity < connectivity:
                                    smallest_ring_path = path
                else:
                    children.append(e)
            # add new node
            dfs_nodes.append(children)
        return smallest_ring_path

    def search_another_ring(self, ring_path, used_edge):
        """Find another ring starting from given edges.

        Args:
            ring_path (list): a list of edges in a ring path
            used_edge (Counter): counter of used edge

        Returns:
             (list): a list of edges as a ring path
        """
        connected_edges = []
        for index, e in enumerate(ring_path):
            if used_edge[e] > 1:
                # shared edge
                if len(connected_edges) == 0:
                    connected_edges.append([index])
                elif ring_path[connected_edges[-1][-1]].start == e.end:
                    connected_edges[-1].append(index)
                else:
                    connected_edges.append([index])

        # find other paths connecting start/end of connected_edges
        alternative_paths = []
        for connected_edge in [edges for edges in connected_edges if len(edges) > 1]:
            if len(connected_edge) == len(ring_path):
                # if all the edge is connected, remove last one
                connected_edge.pop()
            s_edge = ring_path[(connected_edge[-1] + 1) % len(ring_path)]
            e_edge = ring_path[connected_edge[0] - 1]
            connected_path = [ring_path[ind] for ind in connected_edge]
            residue_path = ring_path[connected_edge[-1] + 1:]
            residue_path.extend(ring_path[:connected_edge[0]])
            residue_path = [e.get_reverse_edge() for e in reversed(residue_path)]
            s_edge.end.visit = 1
            dfs_nodes = [[e for e in s_edge.end.edges]]
            # depth first search
            while len(dfs_nodes) > 0:
                node = dfs_nodes[-1]
                if len(node) == 0:
                    # all children are scanned
                    dfs_nodes.pop()
                    if len(dfs_nodes) > 0:
                        node = dfs_nodes[-1]
                        e = node.pop()
                        e.end.visit = 0
                    continue
                # set last children as current node
                current = node[-1]
                current.end.visit = 1
                children = []
                for e in current.end.edges:
                    if e.end == current.start:
                        # ignore reverse direction
                        continue
                    if e.end.visit > 0:
                        # skip visited vertex
                        continue
                    if len(dfs_nodes) + 1 == len(connected_edge):
                        if e.end == e_edge.start:
                            # reach the end vertex
                            path = [node[-1] for node in dfs_nodes]
                            path.append(e)
                            path.reverse()
                            if any([e0 != e1 for (e0, e1) in zip(path, connected_path)]) and \
                                    (len(path) != len(residue_path) or
                                     any([e0 != e1 for (e0, e1) in zip(path, residue_path)])):
                                alternative_paths.append((connected_edge, path))
                    else:
                        children.append(e)
                # add new node
                dfs_nodes.append(children)
            s_edge.end.visit = 0

        # make another ring path replacing alternative_paths
        other_ring_paths = []
        for num in range(len(alternative_paths)):
            for alt_paths in itertools.combinations(alternative_paths, num+1):
                other_ring_path = [e for e in ring_path]
                for (edges, path) in alt_paths:
                    other_ring_path[edges[0]:edges[-1]+1] = path
                other_ring_paths.append(other_ring_path)

        return other_ring_paths

    def add_vertex(self, atom):
        """Add an AtomVertex with atom name.

        Args:
            atom (str): atom name

        Returns:
            AtomVertex: a vertex object
        """
        # simple atom vertex
        index = len(self.vertices)
        v = AtomVertex(index, atom)
        self.vertices.append(v)
        return v

    def add_graph_graph_vertex(self, graph, pos):
        """Add a sub-structure (GraphGraphVertex) as if it is a single vertex.

        Args:
            graph (str): SMILES of a sub-structure
            pos (int): vertex index in a sub-structure to connect to an existing graph

        Returns:
             GraphAsVertex: a new vertex
        """
        # graph vertex
        index = len(self.vertices)
        v = GraphAsVertex(index, graph, pos)
        self.graph_vertices.append(v)
        self.vertices.append(v.connection_vertex)
        return v

    def add_ring_graph_vertex(self, ring, pos):
        """Add a ring (RingGraphVertex) as if it is a single vertex.

        Args:
            ring (str): SMILES of a ring
            pos (int): vretex index in a ring to connect to an existing graph

        Returns:
            RingAsVertex: a new vertex
        """
        # ring vertex
        index = len(self.vertices)
        v = RingAsVertex(index, ring, pos)
        self.ring_vertices.append(v)
        self.vertices.append(v.connection_vertex)
        return v

    def add_graph_graph_vertex_by_graph(self, graph_graph, pos):
        """Add a sub-structure by AtomGraph object.

        Args:
            graph_graph (GraphAsVertex): sub-structure as vertex object
            pos (int): vertex index in a sub-structure to connect to an existing graph

        Returns:
            GraphAsVertex: a new vertex
        """
        index = len(self.vertices)
        v = graph_graph
        v.index = index
        v.connection_vertex.index = index
        connect = v.vertices[pos]
        connect.connect_bond = 0
        v.connection_vertex.connect = connect
        v.connection_vertex.atom = '%s:%d'%(v.symbol, connect.graph_index)
        self.graph_vertices.append(v)
        self.vertices.append(v.connection_vertex)
        return v

    def add_ring_graph_vertex_by_graph(self, ring_graph, pos):
        """Add a ring by AtomGraph object.

        Args:
            ring_graph (RingAsVertex): ring as vertex object
            pos (int): vretex index in a ring to connect to an existing graph

        Returns:
            RingAsVertex: a new vertex
        """
        index = len(self.vertices)
        v = ring_graph
        v.index = index
        v.connection_vertex.index = index
        connect = v.vertices[pos]
        connect.connect_bond = 0
        v.connection_vertex.connect = connect
        v.connection_vertex.atom = '%s:%d'%(v.symbol, connect.ring_index)
        self.ring_vertices.append(v)
        self.vertices.append(v.connection_vertex)
        return v

    def get_graph_graph_vertex(self, connection_vertex):
        """Get graph graph vretex object corresponding to graph connection vertex.

        Args:
            connection_vertex (GraphConnectionVertex): a connection point of a sub-structure

        Returns:
            GraphAsVertex: a sub-structure vertex
        """
        for v in self.graph_vertices:
            if v.connection_vertex == connection_vertex:
                return v
        return None

    def get_ring_graph_vertex(self, connection_vertex):
        """Get ring graph vretex object corresponding to ring connection vertex.

        Args:
            connection_vertex (RingConnectionVertex): a connection point of a ring

        Returns:
            RingAsVertex: a ring vertex
        """
        for v in self.ring_vertices:
            if v.connection_vertex == connection_vertex:
                return v
        return None

    def expand_graph(self):
        """Expand all the ring/graph graph vertex and remember the expanded vertices.
        """
        self.expanded_ring = self.expand_ring_graph()
        self.expanded_graph = self.expand_graph_graph()

    def shrink_graph(self):
        """Shrink all the expanded ring/graph graph vertex and remember the expanded vertices.
        """
        self.shrink_ring_graph(self.expanded_ring)
        self.expanded_ring = []
        self.shrink_graph_graph(self.expanded_graph)
        self.expanded_graph = []

    def expand_graph_vertex(self, vertex):
        """Expand a specified ring/graph vertex to real atom sub-graphs.
        """
        if isinstance(vertex, RingAsVertex):
            self.expand_ring_graph_vertex(vertex)
            self.expanded_vertex.append(vertex)
        elif isinstance(vertex, GraphAsVertex):
            self.expand_graph_graph_vertex(vertex)
            self.expanded_vertex.append(vertex)
        else:
            self.expanded_vertex.append(vertex)

    def shrink_graph_vertex(self):
        """Shrink a real atom sub-graph expanded last to a single ring/graph connection vertex.
        """
        vertex = self.expanded_vertex.pop()
        if isinstance(vertex, RingAsVertex):
            self.shrink_ring_graph_vertex(vertex)
        elif isinstance(vertex, GraphAsVertex):
            self.shrink_graph_graph_vertex(vertex)
        else:
            pass

    def expand_ring_graph(self):
        """Expand all the shrunk ring vertices to real atom sub-graphs,
        and returns expanded vertices.

        Returns:
            list: a list of shrunk ring vertices
        """
        shrunk_vertices = []
        for v in self.ring_vertices:
            if not v.expand:
                shrunk_vertices.append(v)
                self.expand_ring_graph_vertex(v)
        return shrunk_vertices

    def shrink_ring_graph(self, vertices):
        """Shrink specified ring vertices to a single ring connection vertex.

        Args:
            vertices (list): a list of expanded ring vertices
        """
        for v in reversed(vertices):
            self.shrink_ring_graph_vertex(v)

    def expand_ring_graph_vertex(self, vertex):
        """Expand a specified ring graph vertex to real atom sub-graph.

        Args:
            vertex (RingAsVertex): a ring graph vertex

        Returns:
            RingAsVertex: a ring graph vertex
        """
        # replace connection vertex with ring vertices
        single_vertex = (len(self.vertices) == 1)
        index = vertex.connection_vertex.index
        size = 1
        self.vertices[index:index+size] = vertex.vertices
        for v in self.vertices[index:]:
            v.index = index
            index += 1
        # replace edge of connection vertex
        if not single_vertex:
            replace_vertex = vertex.vertices[vertex.connection_vertex.connect.ring_index]
            replace_edge = vertex.connection_vertex.pop_edge()
            replace_edge.start = replace_vertex
            replace_vertex.add_edge(replace_edge)
            for e in replace_edge.end.edges:
                if e.end == vertex.connection_vertex:
                    e.end = replace_vertex
                    break
        vertex.expand = True
        return vertex

    def shrink_ring_graph_vertex(self, vertex):
        """Shrink a specified ring graph to a single ring connection vertex.

        Args:
            vertex (RingAsVertex): a ring graph vertex

        Returns:
            RingAsVertex: a ring graph vertex
        """
        # replace ring vertices with connection vertex
        single_vertex = (len(self.vertices) == len(vertex.vertices))
        index = vertex.vertices[0].index
        size = len(vertex.vertices)
        self.vertices[index:index+size] = [vertex.connection_vertex]
        for v in self.vertices[index:]:
            v.index = index
            index += 1
        # replace edge of connection vertex
        if not single_vertex:
            replace_vertex = vertex.vertices[vertex.connection_vertex.connect.ring_index]
            replace_edge = replace_vertex.pop_edge()
            replace_edge.start = vertex.connection_vertex
            vertex.connection_vertex.add_edge(replace_edge)
            for e in replace_edge.end.edges:
                if e.end == replace_vertex:
                    e.end = vertex.connection_vertex
        vertex.expand = False
        return vertex

    def expand_graph_graph(self):
        """Expand all the shrunk graph vertices to real atom sub-graphs,
        and returns expanded vertices.

        Returns:
            list: a list of shurunk graph vertices
        """
        shrunk_vertices = []
        for v in self.graph_vertices:
            if not v.expand:
                shrunk_vertices.append(v)
                self.expand_graph_graph_vertex(v)
        return shrunk_vertices

    def shrink_graph_graph(self, vertices):
        """Shrink specified graph vertices to a single graph connection vertex.

        Args:
            vertices (list): a list of expanded graph vertices
        """
        for v in reversed(vertices):
            self.shrink_graph_graph_vertex(v)

    def expand_graph_graph_vertex(self, vertex):
        """Expand a specified graph graph vretex to real atom sub-graph.

        Args:
            vretex (GraphAsVertex): a graph graph vertex

        Returns:
            GraphAsVertex: a graph graph vretex
        """
        # replace connection vertex with graph vertices
        single_vertex = (len(self.vertices) == 1)
        index = vertex.connection_vertex.index
        size = 1
        self.vertices[index:index+size] = vertex.vertices
        for v in self.vertices[index:]:
            v.index = index
            index += 1
        # replace edge of connection vertex
        if not single_vertex:
            replace_vertex = vertex.vertices[vertex.connection_vertex.connect.graph_index]
            replace_edge = vertex.connection_vertex.pop_edge()
            replace_edge.start = replace_vertex
            replace_vertex.add_edge(replace_edge)
            for e in replace_edge.end.edges:
                if e.end == vertex.connection_vertex:
                    e.end = replace_vertex
                    break
        vertex.expand = True
        return vertex

    def shrink_graph_graph_vertex(self, vertex):
        """Shrink a specified graph graph to a single graph connection vertex.

        Args:
            vertex (GraphAsVertex): a graph graph vertex

        Returns:
            GraphAsVertex: a graph graph vertex
        """
        # replace graph vertices with connection vertex
        single_vertex = (len(self.vertices) == len(vertex.vertices))
        index = vertex.vertices[0].index
        size = len(vertex.vertices)
        self.vertices[index:index+size] = [vertex.connection_vertex]
        for v in self.vertices[index:]:
            v.index = index
            index += 1
        # replace edge of connection vertex
        if not single_vertex:
            replace_vertex = vertex.vertices[vertex.connection_vertex.connect.graph_index]
            replace_edge = replace_vertex.pop_edge()
            replace_edge.start = vertex.connection_vertex
            vertex.connection_vertex.add_edge(replace_edge)
            for e in replace_edge.end.edges:
                if e.end == replace_vertex:
                    e.end = vertex.connection_vertex
        vertex.expand = False
        return vertex

    def get_expand_automorphism(self, automorphism):
        """Get an automorphism of expanded graph from automorphism of shrinked graph

        Args:
            automorhims (AutoMorphism): automorphism of shrinked graph

        Returns:
            Automorphis: autoMorphism of expanded graph
        """
        from .ChemGraphLabeling import AutoMorphism, Permutation

        if len(self.ring_vertices) + len(self.graph_vertices) == 0:
            # no expandable vertex
            return automorphism
        perm_list = automorphism.get_generator_by_index_list()
        vertex_perm_list = dict()
        # make a mapping from shrinked vertex index to expanded vertex index
        expand_index = list(range(len(self.vertices)))
        expand_vertex_index = dict()
        expand_vertices = self.ring_vertices + self.graph_vertices
        expand_vertices = sorted(expand_vertices, key=lambda x:x.index)
        additional_vertex_size = 0
        for v in expand_vertices:
            if not v.expand:
                vertex_index = v.connection_vertex.index
                new_index = expand_index[vertex_index]
                size = len(v.vertices)
                additional_vertex_size += size - 1
                expand_vertex_index[vertex_index] = list(map(lambda x: x + new_index, list(range(size))))
                expand_index[vertex_index+1:] = list(map(lambda x: x + size - 1, expand_index[vertex_index+1:]))
                vertex_perm_list[vertex_index] = v.automorphism.get_generator_by_index_list()
        if len(expand_vertex_index) == 0:
            # no expandable vertex
            return automorphism
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('expand_index:{0}'.format(expand_index))
            logger.debug('expand_vertex_index:{0}'.format(expand_vertex_index))
            logger.debug('vertex_perm_list:{0}'.format(vertex_perm_list))
        # make permutations of expanded graph from automorphism of whole graph and graph vertex
        id_permutation = list(range(len(self.vertices) + additional_vertex_size))
        expand_perm_list = []
        if len(perm_list) > 0:
            for perm in perm_list:
                new_perm = copy.copy(id_permutation)
                # apply perm
                for index, perm_index in enumerate(perm):
                    new_perm[expand_index[index]] = expand_index[perm[index]]
                    # handle permutation between graph vertex
                    if index in vertex_perm_list and index != perm_index:
                        size = len(expand_vertex_index[index])
                        vertex_index = expand_index[index]
                        new_perm[vertex_index:vertex_index+size] = expand_vertex_index[perm_index]
                expand_perm_list.append(new_perm)
        # handle permutation within graph vertex
        for vertex_index, vertex_perms in vertex_perm_list.items():
            if len(vertex_perms) > 0:
                for vertex_perm in vertex_perms:
                    new_perm = copy.copy(id_permutation)
                    size = len(expand_vertex_index[vertex_index])
                    new_vertex_index = expand_index[vertex_index]
                    new_perm[new_vertex_index:new_vertex_index+size] = list(map(lambda x: x + new_vertex_index, vertex_perm))
                    expand_perm_list.append(new_perm)
        # create a new automorphism
        new_automorphism = AutoMorphism()
        for perm in expand_perm_list:
            permutaion = Permutation.from_index_list(perm)
            new_automorphism.add_generator(permutaion)

        return new_automorphism

    def add_edge(self, start, end, bond_type=BondType.SINGLE, direction=0):
        """Add both direction edges from a start vertex to an end vertex with specified bonds.
        If start and end vertices belong to different category (simple atom vertex, or graph vertex
        representing a sub-graph), hyper_edge is created.

        Args:
            start (ChemVertex): start vertex of an edge
            end (ChemVertex): end vertex of an edge
            bond_type (BondType, optional): bond type of an edge. Defaults to BondType.SINGLE
            direction (int, optional): direction of a dative bond. Defaults to 0.

        Returns:
            ChemEdge, ChemEdge: a pair of forward and backward edges
        """
        e1 = ChemEdge(start, end, bond_type, direction)
        e2 = ChemEdge(end, start, bond_type, direction)
        if bond_type == BondType.AROMATIC:
            start.aromatic = True
            end.aromatic = True
        # an edge between same level vertices is added to normal edges
        # otherwise, added to hyper edge
        if isinstance(start, SubStructureAsVertex) == isinstance(end, SubStructureAsVertex):
            start.add_edge(e1)
            end.add_edge(e2)
        else:
            start.add_hyper_edge(e1)
            end.add_hyper_edge(e2)
        return e1, e2

    def add_directed_edge(self, start, end, bond_type=BondType.SINGLE, direction=0):
        """Add a directed edge from a start vertex to an end vretex with specified bonds
        If start and end vertices belong to different category (simple atom vertex, or graph vertex
        representing a sub-graph), hyper_edge is created.

        Args:
            start (ChemVertex): start vertex of an edge
            end (ChemVertex): end vertex of an edge
            bond_type (BondType, optional): bond type of an edge. Defaults to BondType.SINGLE.
            direction (int, optional): direction of a dative bond. Defaults to 0.

        Returns:
            ChemEdge: a edge from start to end
        """
        e = ChemEdge(start, end, bond_type, direction)
        if bond_type == BondType.AROMATIC:
            start.aromatic = True
            end.aromatic = True
        # an edge between same level vertices is added to normal edges
        # otherwise, added to hyper edge
        if isinstance(start, SubStructureAsVertex) == isinstance(end, SubStructureAsVertex):
            start.add_edge(e)
        else:
            start.add_hyper_edge(e)
        return e

    def pop_vertex(self, vertex=None):
        """Pop a atom vertex added last if a vertex is not specified. Remove it otherwise.
        At the same time, edges connecting the vertex are also removed.
        Args:
            vertex (AtomVertex, optional): a atom vertex
        Returns:
            AtomVertex: a removed vertex
        """
        if vertex is None:
            v = self.vertices.pop()
            for e in v.edges:
                e.end.pop_edge()
            for e in v.hyper_edges:
                e.end.pop_hyper_edge()
            v.clear()
            return v
        else:
            v = vertex
            self.vertices.remove(v)
            for e in v.edges:
                for x in e.end.edges:
                    if x.end == v:
                        e.end.pop_edge(x)
                for x in e.end.hyper_edges:
                    if x.end == v:
                        e.end.pop_hyper_edge(x)
            v.clear()
            return v

    def pop_graph_graph_vertex(self, vertex=None):
        """Pop a graph graph vertex added last if a vertex is not specified. Remove it otherwise.
        At the same time, edges connecting the vertex are also removed.

        Args:
            vertex (GraphGraphVertex, optional): a graph graph vertex. Defaults to None.

        Returns:
            GraphAsVertex: a removed vertex
        """
        if vertex is None:
            gv = self.graph_vertices[-1]
            if gv.expand:
                self.shrink_graph_graph_vertex(gv)
            self.graph_vertices.pop()
            self.pop_vertex()
            for e in gv.edges:
                e.end.pop_edge()
            for e in gv.hyper_edges:
                e.end.pop_hyper_edge()
            gv.clear()
            return gv
        else:
            gv = vertex
            if gv.expand:
                self.shrink_graph_graph_vertex(gv)
            self.graph_vertices.remove(gv)
            self.pop_vertex(gv.connection_vertex)
            for e in gv.edges:
                for x in e.end.edges:
                    if x.end == gv:
                        e.end.pop_edge(x)
                for x in e.end.hyper_edges:
                    if x.end == gv:
                        e.end.pop_hyper_edge(x)
            gv.clear()
            return gv

    def pop_ring_graph_vertex(self, vertex=None):
        """Pop a ring graph vertex added last if a vertex is not specified. Remove it otherwise.
        At the same time, edges connecting the vertex are also removed.

        Args:
            vertex (RingGraphVertex, optional): a ring graph vertex. Defaults to None.

        Returns:
            RingAsVertex: a removed vertex
        """
        if vertex is None:
            rv = self.ring_vertices[-1]
            if rv.expand:
                self.shrink_ring_graph_vertex(rv)
            self.ring_vertices.pop()
            self.pop_vertex()
            for e in rv.edges:
                e.end.pop_edge()
            for e in rv.hyper_edges:
                e.end.pop_hyper_edge()
            rv.clear()
            return rv
        else:
            rv = vertex
            if rv.expand:
                self.shrink_ring_graph_vertex(rv)
            self.ring_vertices.remove(rv)
            self.pop_vertex(rv.connection_vertex)
            for e in rv.edges:
                for x in e.end.edges:
                    if x.end == rv:
                        e.end.pop_edge(x)
                for x in e.end.hyper_edges:
                    if x.end == rv:
                        e.end.pop_hyper_edge(x)
            rv.clear()
            return rv

    def translate_to_mol(self, vertices=None, sanitize=True):
        """Make rdkit Mol object from a graph structure. If a subset of vertices are given,
        only induced sub-graph is translated to rdkit Mol.

        Args:
            vertices (list): a list of vertices for the translation. Default to None.

        Returns:
            Mol: rdkit Mol object
        """
        if vertices is None:
            vertices = self.vertices
        vertex_map = dict()
        mw = Chem.RWMol()
        for vertex in vertices:
            new_atom = Chem.Atom(vertex.base_atom)
            new_atom.SetFormalCharge(vertex.charge)
            if vertex.charge == 0 and vertex.num_free_hand() >= 0:
                new_atom.SetNumExplicitHs(vertex.explicit_h)
            new_atom.SetAtomMapNum(vertex.atom_map_num)
            new_atom_index = mw.AddAtom(new_atom)
            vertex_map[vertex] = new_atom_index
        for vertex in vertices:
            vertex.visit = 1
            for e in vertex.edges:
                if e.end.visit == 0 and e.end in vertex_map:
                    mw.AddBond(vertex_map[vertex], vertex_map[e.end], e.bond_type)
        for vertex in vertices:
            vertex.visit = 0
        mol = mw.GetMol()
        for vertex in vertices:
            if vertex.root > 0:
                mol.GetAtomWithIdx(vertex_map[vertex]).SetUnsignedProp('root', vertex.root - 1)
            if vertex.repeat != 0:
                mol.GetAtomWithIdx(vertex_map[vertex]).SetIntProp('repeat', vertex.repeat)
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except ValueError as err:
                logger.warning('sanitization error for mol from graph: %s', Chem.MolToSmiles(mol))

        return mol

    def make_subgraph(self, vertices=None):
        """ Make a new sub-graph from given vertices of a graph.

        Args:
              vertices (list): a list of vertices for a sub-graph. Default to None (all vertices)

        Returns:
            AtomGraph, dict: a new sub-graph, a map of vertices from old graph to new graph
        """
        if vertices is None:
            vertices = self.vertices
        vertex_map = dict()
        new_graph = AtomGraph()
        for index, v in enumerate(vertices):
            new_v = copy.copy(v)
            new_v.index = index
            new_v.edges = []
            new_graph.vertices.append(new_v)
            vertex_map[v] = new_v
        for v in vertices:
            for e in v.edges:
                if e.end in vertex_map:
                    new_e = new_graph.add_directed_edge(vertex_map[e.start], vertex_map[e.end],
                                                        e.bond_type)
                    new_e.alternative_bond = e.alternative_bond
        # sort vertices and edges
        new_graph.vertices = sorted(new_graph.vertices, key=lambda vv: vv.index)
        for v in self.vertices:
            v.edges = sorted(v.edges, key=lambda ee: ee.end.index)
        return new_graph, vertex_map

    def get_terminal_index(self):
        """Get a sum of terminal indices of a graph

        Returns:
            int: sum of termina indices
        """
        return sum([v.repeat for v in self.vertices])

    @staticmethod
    def find_shortest_path(start, end):
        """Find a shortest path from star to end

        Args:
            start (list): a list of starting vertex
            end (list): a list of ending vertex

        Returns:
            list: a list of vertex on the shortest path
        """
        goal = set(end)
        heap = []
        distance = dict()
        for v in start:
            heapq.heappush(heap, (0, v.index, v))
            distance[v] = (0, None)
        while len(heap) > 0:
            (dist, index, vertex) = heapq.heappop(heap)
            if vertex in goal:
                break
            else:
                dist += 1
                for e in vertex.edges:
                    if e.end in distance:
                        if dist < distance[e.end][0]:
                            distance[e.end] = (dist, vertex)
                            heapq.heappush(heap, (dist, e.end.index, e.end))
                    else:
                        distance[e.end] = (dist, vertex)
                        heapq.heappush(heap, (dist, e.end.index, e.end))
        shortest_path = [vertex]
        while vertex not in start:
            vertex = distance[vertex][1]
            shortest_path.append(vertex)
        shortest_path.reverse()
        return shortest_path

    @staticmethod
    def get_connected_component(edge):
        """Get a connected component starting from an edge

        Args:
            edge (ChemEdge): a starting edge

        Returns:
            list: a list of reached vertices
        """
        edge.start.visit = 1
        vertices = [edge.end]
        visit_vertices = []
        while len(vertices) > 0:
            vertex = vertices.pop()
            if vertex.visit > 0:
                continue
            vertex.visit = 1
            visit_vertices.append(vertex)
            for e in vertex.edges:
                if e.end.visit == 0:
                    vertices.append(e.end)
        for v in visit_vertices:
            v.visit = 0
        edge.start.visit = 0
        return visit_vertices


class SubStructureAsVertex(ChemVertex, AtomGraph):
    """Common base class of a vertex representing a sub-structure.

    Attributes:
        expand (bool): flag if a sub-structure is expanded
        automorphism (Automorphism): automorphism of graph labeling
        atom_count (Counter): counter of atoms
        ring_count (Counter): counter of rings
        aromatic_count (Counter): counter of aromatic rings
        atom_degree_count (dict): counter of atom degrees
        total_ring_count (int): counter of total rings
        total_aromatic_count (int): counter of total aromatic rings
        connection_vertex (ConnectionVertex): vertex for connecting to a molecular graph
    """

    def __init__(self, index):
        """Constructor with an index of a vertex.

        Args:
            index (int): an index of a vertex
        """
        super(SubStructureAsVertex, self).__init__(index)
        super(ChemVertex, self).__init__()
        self.expand = False
        self.automorphism = None
        self.atom_count = Counter()
        self.ring_count = Counter()
        self.aromatic_count = Counter()
        self.atom_degree_count = dict()
        self.total_ring_count = 0
        self.total_aromatic_count = 0
        self.connection_vertex = None

    def set_automorphism(self, automorphism):
        """Set an automorphism of graph labeling

        Args:
              automorphism (AutoMorhism): graph automorphism
        """
        self.automorphism = automorphism


class GraphAsVertex(SubStructureAsVertex):
    """A vertex representing a sub-structure of molecular graph

    Attributes:
        smiles (str): SMILES of a sub-structure
        valance_vertices (list): a list of vertices with free hands
    """

    def __init__(self, index, symbol, graph, pos, fill_graph=False, connection_condition=None):
        """Constructor of GraphAsVertex.

        connection_condition is a dictionary of tuple with vertex index as a key.
        (fill, good_atoms, bad_atoms)
        fill: true is the dummy connection should be filled by an atom
        good_atoms: a list of acceptable atom symbols
        bad_atoms: a list of prohibited atom symbols

        Args:
            index (int): an index of a vertex
            smiles (str): SMILES of a sub-sructure
            graph (AtomGraph): graph  of a sub-structure
            pos (int): an index of a vertex in a sub-structure for connecting to a molecular graph
            fill_graph (bool): flag of filling graph hands by default
            connection_condition (dict): conditions for connecting at dummy atoms
        """
        super().__init__(index)
        self.symbol = symbol
        self.valence_vertices = []
        # make internal graph
        vertex_map = {}
        for v in graph.vertices:
            if v.atom != ChemVertex.wild_card_atom:
                new_v = self.add_graph_vertex(v.atom)
                new_v.valence = v.valence
                new_v.explicit_h = v.explicit_h
                new_v.in_ring = v.in_ring
                new_v.repeat = v.repeat
                vertex_map[v] = new_v
        for v in graph.vertices:
            if v.atom != ChemVertex.wild_card_atom:
                vv = vertex_map[v]
                # update the valance to the current hands
                vv.valence = int(sum(e.get_bond_order() for e in v.edges))
                for e in v.edges:
                    if e.end.atom != ChemVertex.wild_card_atom:
                        self.add_graph_edge(vv, vertex_map[e.end], e.bond_type)
            else:
                # set connect conditions and available bond types to corresponding vertex
                connect_e = v.edges[0]
                connect_v = vertex_map[connect_e.end]
                if len(connect_v.available_bond) == 0:
                    self.valence_vertices.append(connect_v)
                connect_v.available_bond[connect_e.bond_type] += 1
                # get connect condition for the dummy vertex
                if connection_condition is not None and v.index in connection_condition:
                    (fill, good_atoms, bad_atoms) = connection_condition[v.index]
                    connect_v.free_count += 0 if fill else connect_e.get_bond_order()
                    connect_v.good_atoms[connect_e.bond_type].append(set(good_atoms))
                    connect_v.bad_atoms[connect_e.bond_type].append(set(bad_atoms))
                else:
                    connect_v.free_count += 0 if fill_graph else connect_e.get_bond_order()
                    connect_v.good_atoms[connect_e.bond_type].append(set([]))
                    connect_v.bad_atoms[connect_e.bond_type].append(set([]))

        # update vertex color
        for v in self.vertices:
            if v.num_edge() == 1:
                v.set_color('||' + v.color())

        # update counter
        for v in graph.vertices:
            if v.atom !=  ChemVertex.wild_card_atom:
                self.atom_count[v.atom] += 1
                if v.atom not in self.atom_degree_count:
                    self.atom_degree_count[v.atom] = [0] * (v.num_valence() + 1)
                self.atom_degree_count[v.atom][v.bond_degree()] += 1
        self.ring_count, self.aromatic_count = graph.count_rings_sssr()
        self.total_ring_count = sum(self.ring_count.values())
        self.total_aromatic_count = sum(self.aromatic_count.values())

        # set connection vertex
        graph_valence = 0
        for v in self.valence_vertices:
            graph_valence += max(0, v.num_free_hand())
        if graph_valence > 0:
            connect = self.valence_vertices[pos]
        else:
            # use the first vertex as a dummy connection
            connect = self.vertices[0]
        self.connection_vertex = GraphConnectionVertex(index, symbol, connect, graph_valence)

    def add_graph_vertex(self, atom):
        """Add a graph vertex to a graph representing a sub-structure.

        Args:
            atom (str): atom name

        Returns:
            GraphVertex: a vertex
        """
        index = len(self.vertices)
        v = GraphVertex(index, index, atom)
        self.vertices.append(v)
        return v

    def add_graph_edge(self, start, end, bond_type=BondType.SINGLE, direction=0):
        """Add an edge of graph vertices to a graph representing a sub-structure.

        Args:
            start (GraphVertex): a start vertex
            end (GraphVertex): a end vertex
            bond_type (BondType, optional): bond type of an edge. Defaults to BondType.SINGLE.
            direction (int, optional): direction of a dative bond. Defaults to 0.
        Returns:
            ChemEdge: an edge
        """
        e = ChemEdge(start, end, bond_type, direction)
        if bond_type == BondType.AROMATIC:
            start.aromatic = True
            end.aromatic = True
        start.add_graph_edge(e)
        start.add_normal_edge(e)
        return e

    def num_atom(self):
        """Get the number of atoms in a sub-structure.

        Returns:
            int: the number of atoms
        """
        return len(self.vertices)

    def num_ring_atom(self):
        """Get the number of ring atoms in a sub-structure.

        Returns:
            int: the number of ring atoms
        """
        return len(self.vertices)

    def num_aromatic_atom(self):
        """Get the number of aromatic ring atoms in a sub-structure.

        Returns:
            int: the number of aromatic ring atoms
        """
        return sum(v.num_aromatic_atom() for v in self.vertices)

    def num_valence(self):
        """Get the number of valence of a sub-structure.

        Returns:
            int: valence
        """
        return sum(v.num_valence() for v in self.vertices)

    def color(self):
        """Get the name of the vertex as a color.

        Returns:
            str: color of a vertex
        """
        return '||'+self.symbol


class RingAsVertex(SubStructureAsVertex):
    """A vertex representing a ring of molecular graph.

    Attributes:
        ring (str): SMILES of a ring
    """

    def __init__(self, index, symbol, graph, pos):
        """Constructor.

        Args:
            index (int): an index of a vertex
            ring (str): SMILES of a ring
            pos (int): an index of a vertex in a ring for connecting to a molecular graph
        """
        super().__init__(index)
        self.symbol = symbol
        # make internal ring graph
        vertex_map = {}
        for v in graph.vertices:
            new_v = self.add_ring_vertex(v.atom)
            new_v.valence = v.valence
            new_v.explicit_h = v.explicit_h
            new_v.in_ring = v.in_ring
            vertex_map[v] = new_v
        for v in graph.vertices:
            for e in v.edges:
                self.add_ring_edge(vertex_map[v], vertex_map[e.end], e.bond_type)

        # count atoms
        for v in graph.vertices:
            self.atom_count[v.atom] += 1
            if v.atom not in self.atom_degree_count:
                self.atom_degree_count[v.atom] = [0] * (v.num_valence() + 1)
            self.atom_degree_count[v.atom][v.bond_degree()] += 1
        self.ring_count, self.aromatic_count = graph.count_rings_sssr()
        self.total_ring_count = sum(self.ring_count.values())
        self.total_aromatic_count = sum(self.aromatic_count.values())

        # set connection vertex
        graph_valence = 0
        for v in self.vertices:
            if v.num_edge() == 1:
                v.set_color('|' + v.color())
            graph_valence += max(0, v.num_free_hand())
        connect = vertex_map[graph.vertices[pos]]
        self.connection_vertex = RingConnectionVertex(index, symbol, connect, graph_valence)

    def add_ring_vertex(self, atom):
        """Add a graph vertex to a graph representing a ring.

        Args:
            atom (str): atom name

        Returns:
            RingVertex: a vertex
        """
        index = len(self.vertices)
        v = RingVertex(index, index, atom)
        self.vertices.append(v)
        return v

    def add_ring_edge(self, start, end, bond_type=BondType.SINGLE, direction=0):
        """Add an edge of ring vertices to a graph representing a ring.

        Args:
            start (RingVertex): a start vertex
            end (RingVertex): a end vertex
            bond_type (BondType, optional): bond type of an edge. Defaults to BondType.SINGLE.
            direction (int, optional): direction of a dative bond. Defaults to 0.

        Returns:
            ChemEdge: an edge
        """
        e = ChemEdge(start, end, bond_type, direction)
        if bond_type == BondType.AROMATIC:
            start.aromatic = True
            end.aromatic = True
        start.add_ring_edge(e)
        start.add_normal_edge(e)
        return e

    def num_atom(self):
        """Get the number of atoms in a ring.

        Returns:
            int: the number of atoms
        """
        return len(self.vertices)

    def num_ring_atom(self):
        """Get the number of ring atoms in a ring.

        Returns:
            int: the number of atoms
        """
        return len(self.vertices)

    def num_aromatic_atom(self):
        """Get the number of aromatic ring atoms in a ring.

        Returns:
            int: the number of atoms
        """
        return sum(v.num_aromatic_atom() for v in self.vertices)

    def num_valence(self):
        """Get the number of valence of a ring.

        Returns:
            int: valence
        """
        return sum(v.num_valence() for v in self.vertices)

    def color(self):
        """Get the name of the vertex as a color.

        Returns:
            str: color of a vertex
        """
        return '|'+self.symbol

