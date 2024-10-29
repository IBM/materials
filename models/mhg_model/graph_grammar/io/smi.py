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
__date__ = "Jan 12 2018"

from copy import deepcopy
from rdkit import Chem
from rdkit import RDLogger
import networkx as nx
import numpy as np
from ..hypergraph import Hypergraph
from ..graph_grammar.symbols import TSymbol, BondSymbol

# supress warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class HGGen(object):
    """
    load .smi file and yield a hypergraph.

    Attributes
    ----------
    path_to_file : str
        path to .smi file
    kekulize : bool
        kekulize or not
    add_Hs : bool
        add implicit hydrogens to the molecule or not.
    all_single : bool
        if True, all multiple bonds are summarized into a single bond with some attributes

    Yields
    ------
    Hypergraph
    """
    def __init__(self, path_to_file, kekulize=True, add_Hs=False, all_single=True):
        self.num_line = 1
        self.mol_gen = Chem.SmilesMolSupplier(path_to_file, titleLine=False)
        self.kekulize = kekulize
        self.add_Hs = add_Hs
        self.all_single = all_single

    def __iter__(self):
        return self

    def __next__(self):
        '''
        each_mol = None
        while each_mol is None:
            each_mol = next(self.mol_gen)
        '''
        # not ignoring parse errors
        each_mol = next(self.mol_gen)
        if each_mol is None:
            raise ValueError(f'incorrect smiles in line {self.num_line}')
        else:
            self.num_line += 1
        return mol_to_hg(each_mol, self.kekulize, self.add_Hs)


def mol_to_bipartite(mol, kekulize):
    """
    get a bipartite representation of a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        molecule object

    Returns
    -------
    nx.Graph
        a bipartite graph representing which bond is connected to which atoms.
    """
    try:
        mol = standardize_stereo(mol)
    except KeyError:
        print(Chem.MolToSmiles(mol))
        raise KeyError
        
    if kekulize:
        Chem.Kekulize(mol)

    bipartite_g = nx.Graph()
    for each_atom in mol.GetAtoms():
        bipartite_g.add_node(f"atom_{each_atom.GetIdx()}",
                             atom_attr=atom_attr(each_atom, kekulize))

    for each_bond in mol.GetBonds():
        bond_idx = each_bond.GetIdx()
        bipartite_g.add_node(
            f"bond_{bond_idx}",
            bond_attr=bond_attr(each_bond, kekulize))
        bipartite_g.add_edge(
            f"atom_{each_bond.GetBeginAtomIdx()}",
            f"bond_{bond_idx}")
        bipartite_g.add_edge(
            f"atom_{each_bond.GetEndAtomIdx()}",
            f"bond_{bond_idx}")
    return bipartite_g


def mol_to_hg(mol, kekulize, add_Hs):
    """
    get a bipartite representation of a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        molecule object
    kekulize : bool
        kekulize or not
    add_Hs : bool
        add implicit hydrogens to the molecule or not.

    Returns
    -------
    Hypergraph
    """
    if add_Hs:
        mol = Chem.AddHs(mol)

    if kekulize:
        Chem.Kekulize(mol)

    bipartite_g = mol_to_bipartite(mol, kekulize)
    hg = Hypergraph()
    for each_atom in [each_node for each_node in bipartite_g.nodes()
                      if each_node.startswith('atom_')]:
        node_set = set([])
        for each_bond in bipartite_g.adj[each_atom]:
            hg.add_node(each_bond,
                        attr_dict=bipartite_g.nodes[each_bond]['bond_attr'])
            node_set.add(each_bond)
        hg.add_edge(node_set,
                    attr_dict=bipartite_g.nodes[each_atom]['atom_attr'])
    return hg


def hg_to_mol(hg, verbose=False):
    """ convert a hypergraph into Mol object

    Parameters
    ----------
    hg : Hypergraph

    Returns
    -------
    mol : Chem.RWMol
    """
    mol = Chem.RWMol()
    atom_dict = {}
    bond_set = set([])
    for each_edge in hg.edges:
        atom = Chem.Atom(hg.edge_attr(each_edge)['symbol'].symbol)
        atom.SetNumExplicitHs(hg.edge_attr(each_edge)['symbol'].num_explicit_Hs)
        atom.SetFormalCharge(hg.edge_attr(each_edge)['symbol'].formal_charge)
        atom.SetChiralTag(
            Chem.rdchem.ChiralType.values[
                hg.edge_attr(each_edge)['symbol'].chirality])
        atom_idx = mol.AddAtom(atom)
        atom_dict[each_edge] = atom_idx

    for each_node in hg.nodes:
        edge_1, edge_2 = hg.adj_edges(each_node)
        if edge_1+edge_2 not in bond_set:
            if hg.node_attr(each_node)['symbol'].bond_type <= 3:
                num_bond = hg.node_attr(each_node)['symbol'].bond_type
            elif hg.node_attr(each_node)['symbol'].bond_type == 12:
                num_bond = 1
            else:
                raise ValueError(f'too many bonds; {hg.node_attr(each_node)["bond_symbol"].bond_type}')
            _ = mol.AddBond(atom_dict[edge_1],
                            atom_dict[edge_2],
                            order=Chem.rdchem.BondType.values[num_bond])
            bond_idx = mol.GetBondBetweenAtoms(atom_dict[edge_1], atom_dict[edge_2]).GetIdx()

            # stereo
            mol.GetBondWithIdx(bond_idx).SetStereo(
                Chem.rdchem.BondStereo.values[hg.node_attr(each_node)['symbol'].stereo])
            bond_set.update([edge_1+edge_2])
            bond_set.update([edge_2+edge_1])
    mol.UpdatePropertyCache()
    mol = mol.GetMol()
    not_stereo_mol = deepcopy(mol)
    if Chem.MolFromSmiles(Chem.MolToSmiles(not_stereo_mol)) is None:
        raise RuntimeError('no valid molecule was obtained.')
    try:
        mol = set_stereo(mol)
        is_stereo = True
    except:
        import traceback
        traceback.print_exc()
        is_stereo = False
    mol_tmp = deepcopy(mol)
    Chem.SetAromaticity(mol_tmp)
    if Chem.MolFromSmiles(Chem.MolToSmiles(mol_tmp)) is not None:
        mol = mol_tmp
    else:
        if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is None:
            mol = not_stereo_mol
    mol.UpdatePropertyCache()
    Chem.GetSymmSSSR(mol)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if verbose:
        return mol, is_stereo
    else:
        return mol

def hgs_to_mols(hg_list, ignore_error=False):
    if ignore_error:
        mol_list = []
        for each_hg in hg_list:
            try:
                mol = hg_to_mol(each_hg)
            except:
                mol = None
            mol_list.append(mol)
    else:
        mol_list = [hg_to_mol(each_hg) for each_hg in hg_list]
    return mol_list

def hgs_to_smiles(hg_list, ignore_error=False):
    mol_list = hgs_to_mols(hg_list, ignore_error)
    smiles_list = []
    for each_mol in mol_list:
        try:
            smiles_list.append(
                Chem.MolToSmiles(
                    Chem.MolFromSmiles(
                        Chem.MolToSmiles(
                            each_mol))))
        except:
            smiles_list.append(None)
    return smiles_list

def atom_attr(atom, kekulize):
    """
    get atom's attributes

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
    kekulize : bool
        kekulize or not

    Returns
    -------
    atom_attr : dict
        "is_aromatic" : bool
            the atom is aromatic or not.
        "smarts" : str
            SMARTS representation of the atom.
    """
    if kekulize:
        return {'terminal': True,
                'is_in_ring': atom.IsInRing(),
                'symbol': TSymbol(degree=0,
                                  #degree=atom.GetTotalDegree(),
                                  is_aromatic=False,
                                  symbol=atom.GetSymbol(),
                                  num_explicit_Hs=atom.GetNumExplicitHs(),
                                  formal_charge=atom.GetFormalCharge(),
                                  chirality=atom.GetChiralTag().real
                )}
    else:
        return {'terminal': True,
                'is_in_ring': atom.IsInRing(),
                'symbol': TSymbol(degree=0,
                                  #degree=atom.GetTotalDegree(),
                                  is_aromatic=atom.GetIsAromatic(),
                                  symbol=atom.GetSymbol(),
                                  num_explicit_Hs=atom.GetNumExplicitHs(),
                                  formal_charge=atom.GetFormalCharge(),
                                  chirality=atom.GetChiralTag().real
                )}

def bond_attr(bond, kekulize):
    """
    get atom's attributes

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
    kekulize : bool
        kekulize or not

    Returns
    -------
    bond_attr : dict
        "bond_type" : int
        {0: rdkit.Chem.rdchem.BondType.UNSPECIFIED,
         1: rdkit.Chem.rdchem.BondType.SINGLE,
         2: rdkit.Chem.rdchem.BondType.DOUBLE,
         3: rdkit.Chem.rdchem.BondType.TRIPLE,
         4: rdkit.Chem.rdchem.BondType.QUADRUPLE,
         5: rdkit.Chem.rdchem.BondType.QUINTUPLE,
         6: rdkit.Chem.rdchem.BondType.HEXTUPLE,
         7: rdkit.Chem.rdchem.BondType.ONEANDAHALF,
         8: rdkit.Chem.rdchem.BondType.TWOANDAHALF,
         9: rdkit.Chem.rdchem.BondType.THREEANDAHALF,
         10: rdkit.Chem.rdchem.BondType.FOURANDAHALF,
         11: rdkit.Chem.rdchem.BondType.FIVEANDAHALF,
         12: rdkit.Chem.rdchem.BondType.AROMATIC,
         13: rdkit.Chem.rdchem.BondType.IONIC,
         14: rdkit.Chem.rdchem.BondType.HYDROGEN,
         15: rdkit.Chem.rdchem.BondType.THREECENTER,
         16: rdkit.Chem.rdchem.BondType.DATIVEONE,
         17: rdkit.Chem.rdchem.BondType.DATIVE,
         18: rdkit.Chem.rdchem.BondType.DATIVEL,
         19: rdkit.Chem.rdchem.BondType.DATIVER,
         20: rdkit.Chem.rdchem.BondType.OTHER,
         21: rdkit.Chem.rdchem.BondType.ZERO}
    """
    if kekulize:
        is_aromatic = False
        if bond.GetBondType().real == 12:
            bond_type = 1
        else:
            bond_type = bond.GetBondType().real
    else:
        is_aromatic = bond.GetIsAromatic()
        bond_type = bond.GetBondType().real
    return {'symbol': BondSymbol(is_aromatic=is_aromatic,
                                 bond_type=bond_type,
                                 stereo=int(bond.GetStereo())),
            'is_in_ring': bond.IsInRing()}


def standardize_stereo(mol):
    '''
 0: rdkit.Chem.rdchem.BondDir.NONE,
 1: rdkit.Chem.rdchem.BondDir.BEGINWEDGE,
 2: rdkit.Chem.rdchem.BondDir.BEGINDASH,
 3: rdkit.Chem.rdchem.BondDir.ENDDOWNRIGHT,
 4: rdkit.Chem.rdchem.BondDir.ENDUPRIGHT,

    '''
    # mol = Chem.AddHs(mol) # this removes CIPRank !!!
    for each_bond in mol.GetBonds():
        if int(each_bond.GetStereo()) in [2, 3]: #2=Z (same side), 3=E
            begin_stereo_atom_idx = each_bond.GetBeginAtomIdx()
            end_stereo_atom_idx = each_bond.GetEndAtomIdx()
            atom_idx_1 = each_bond.GetStereoAtoms()[0]
            atom_idx_2 = each_bond.GetStereoAtoms()[1]
            if mol.GetBondBetweenAtoms(atom_idx_1, begin_stereo_atom_idx):
                begin_atom_idx = atom_idx_1
                end_atom_idx = atom_idx_2
            else:
                begin_atom_idx = atom_idx_2
                end_atom_idx = atom_idx_1

            begin_another_atom_idx = None
            assert len(mol.GetAtomWithIdx(begin_stereo_atom_idx).GetNeighbors()) <= 3
            for each_neighbor in mol.GetAtomWithIdx(begin_stereo_atom_idx).GetNeighbors():
                each_neighbor_idx = each_neighbor.GetIdx()
                if each_neighbor_idx not in [end_stereo_atom_idx, begin_atom_idx]:
                    begin_another_atom_idx = each_neighbor_idx

            end_another_atom_idx = None
            assert len(mol.GetAtomWithIdx(end_stereo_atom_idx).GetNeighbors()) <= 3
            for each_neighbor in mol.GetAtomWithIdx(end_stereo_atom_idx).GetNeighbors():
                each_neighbor_idx = each_neighbor.GetIdx()
                if each_neighbor_idx not in [begin_stereo_atom_idx, end_atom_idx]:
                    end_another_atom_idx = each_neighbor_idx

            ''' 
            relationship between begin_atom_idx and end_atom_idx is encoded in GetStereo
            '''
            begin_atom_rank = int(mol.GetAtomWithIdx(begin_atom_idx).GetProp('_CIPRank'))
            end_atom_rank = int(mol.GetAtomWithIdx(end_atom_idx).GetProp('_CIPRank'))
            try:
                begin_another_atom_rank = int(mol.GetAtomWithIdx(begin_another_atom_idx).GetProp('_CIPRank'))
            except:
                begin_another_atom_rank = np.inf
            try:
                end_another_atom_rank = int(mol.GetAtomWithIdx(end_another_atom_idx).GetProp('_CIPRank'))
            except:
                end_another_atom_rank = np.inf
            if begin_atom_rank < begin_another_atom_rank\
               and end_atom_rank < end_another_atom_rank:
                pass
            elif begin_atom_rank < begin_another_atom_rank\
                 and end_atom_rank > end_another_atom_rank:
                # (begin_atom_idx +) end_another_atom_idx should be in StereoAtoms
                if each_bond.GetStereo() == 2:
                    # set stereo
                    each_bond.SetStereo(Chem.rdchem.BondStereo.values[3])
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 3)
                elif each_bond.GetStereo() == 3:
                    # set stereo
                    each_bond.SetStereo(Chem.rdchem.BondStereo.values[2])
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 4)
                else:
                    raise ValueError
                each_bond.SetStereoAtoms(begin_atom_idx, end_another_atom_idx)
            elif begin_atom_rank > begin_another_atom_rank\
                 and end_atom_rank < end_another_atom_rank:
                # (end_atom_idx +) begin_another_atom_idx should be in StereoAtoms
                if each_bond.GetStereo() == 2:
                    # set stereo
                    each_bond.SetStereo(Chem.rdchem.BondStereo.values[3])
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 0)
                elif each_bond.GetStereo() == 3:
                    # set stereo
                    each_bond.SetStereo(Chem.rdchem.BondStereo.values[2])
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 3)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 0)
                else:
                    raise ValueError
                each_bond.SetStereoAtoms(begin_another_atom_idx, end_atom_idx)
            elif begin_atom_rank > begin_another_atom_rank\
                 and end_atom_rank > end_another_atom_rank:
                # begin_another_atom_idx + end_another_atom_idx should be in StereoAtoms
                if each_bond.GetStereo() == 2:
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 3)
                elif each_bond.GetStereo() == 3:
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 4)
                else:
                    raise ValueError
                each_bond.SetStereoAtoms(begin_another_atom_idx, end_another_atom_idx)
            else:
                raise RuntimeError
    return mol


def set_stereo(mol):
    '''
 0: rdkit.Chem.rdchem.BondDir.NONE,
 1: rdkit.Chem.rdchem.BondDir.BEGINWEDGE,
 2: rdkit.Chem.rdchem.BondDir.BEGINDASH,
 3: rdkit.Chem.rdchem.BondDir.ENDDOWNRIGHT,
 4: rdkit.Chem.rdchem.BondDir.ENDUPRIGHT,
    '''
    _mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    Chem.Kekulize(_mol, True)
    substruct_match = mol.GetSubstructMatch(_mol)
    if not substruct_match:
        ''' mol and _mol are kekulized.
        sometimes, the order of '=' and '-' changes, which causes mol and _mol not matched.
        '''
        Chem.SetAromaticity(mol)
        Chem.SetAromaticity(_mol)
        substruct_match = mol.GetSubstructMatch(_mol)
    try:
        atom_match = {substruct_match[_mol_atom_idx]: _mol_atom_idx for _mol_atom_idx in range(_mol.GetNumAtoms())} # mol to _mol
    except:
        raise ValueError('two molecules obtained from the same data do not match.')
        
    for each_bond in mol.GetBonds():
        begin_atom_idx = each_bond.GetBeginAtomIdx()
        end_atom_idx = each_bond.GetEndAtomIdx()
        _bond = _mol.GetBondBetweenAtoms(atom_match[begin_atom_idx], atom_match[end_atom_idx])
        _bond.SetStereo(each_bond.GetStereo())

    mol = _mol
    for each_bond in mol.GetBonds():
        if int(each_bond.GetStereo()) in [2, 3]: #2=Z (same side), 3=E
            begin_stereo_atom_idx = each_bond.GetBeginAtomIdx()
            end_stereo_atom_idx = each_bond.GetEndAtomIdx()
            begin_atom_idx_set = set([each_neighbor.GetIdx()
                                      for each_neighbor
                                      in mol.GetAtomWithIdx(begin_stereo_atom_idx).GetNeighbors()
                                      if each_neighbor.GetIdx() != end_stereo_atom_idx])
            end_atom_idx_set = set([each_neighbor.GetIdx()
                                    for each_neighbor
                                    in mol.GetAtomWithIdx(end_stereo_atom_idx).GetNeighbors()
                                    if each_neighbor.GetIdx() != begin_stereo_atom_idx])
            if not begin_atom_idx_set:
                each_bond.SetStereo(Chem.rdchem.BondStereo(0))
                continue
            if not end_atom_idx_set:
                each_bond.SetStereo(Chem.rdchem.BondStereo(0))
                continue
            if len(begin_atom_idx_set) == 1:
                begin_atom_idx = begin_atom_idx_set.pop()
                begin_another_atom_idx = None
            if len(end_atom_idx_set) == 1:
                end_atom_idx = end_atom_idx_set.pop()
                end_another_atom_idx = None
            if len(begin_atom_idx_set) == 2:
                atom_idx_1 = begin_atom_idx_set.pop()
                atom_idx_2 = begin_atom_idx_set.pop()
                if int(mol.GetAtomWithIdx(atom_idx_1).GetProp('_CIPRank')) < int(mol.GetAtomWithIdx(atom_idx_2).GetProp('_CIPRank')):
                    begin_atom_idx = atom_idx_1
                    begin_another_atom_idx = atom_idx_2
                else:
                    begin_atom_idx = atom_idx_2
                    begin_another_atom_idx = atom_idx_1
            if len(end_atom_idx_set) == 2:
                atom_idx_1 = end_atom_idx_set.pop()
                atom_idx_2 = end_atom_idx_set.pop()
                if int(mol.GetAtomWithIdx(atom_idx_1).GetProp('_CIPRank')) < int(mol.GetAtomWithIdx(atom_idx_2).GetProp('_CIPRank')):
                    end_atom_idx = atom_idx_1
                    end_another_atom_idx = atom_idx_2
                else:
                    end_atom_idx = atom_idx_2
                    end_another_atom_idx = atom_idx_1

            if each_bond.GetStereo() == 2: # same side
                mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 4)
                each_bond.SetStereoAtoms(begin_atom_idx, end_atom_idx)
            elif each_bond.GetStereo() == 3: # opposite side
                mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 3)
                each_bond.SetStereoAtoms(begin_atom_idx, end_atom_idx)
            else:
                raise ValueError
    return mol


def safe_set_bond_dir(mol, atom_idx_1, atom_idx_2, bond_dir_val):
    if atom_idx_1 is None or atom_idx_2 is None:
        return mol
    else:
        mol.GetBondBetweenAtoms(atom_idx_1, atom_idx_2).SetBondDir(Chem.rdchem.BondDir.values[bond_dir_val])
        return mol
        
