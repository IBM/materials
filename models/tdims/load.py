from rdkit import Chem
from rdkit.Chem import AllChem, BRICS

from tdims.ChemGenerator.ChemGraph import AtomGraph

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from collections import Counter
import itertools

import pandas as pd
from collections import defaultdict
import types

import warnings
warnings.simplefilter('ignore')

import logging
logger = logging.getLogger(__name__)

import numpy as np


class TDiMS():
    def __init__(self, sm_list, radius=1, func_dis=-2, func_merge=sum, fragment_set=True, atom_set=True, fingerprint_set=True, nBit=2048):
        """Constructor of TDiMS class.

        Args:
            sm_list: smiles list of dataset
            radius (int, optional): radius of finger print. Default to 1.
            func_dis (int, float, function, optional) : Calculation method for computing the feature value from bonds distance. Default to -2 (inverse square)
            func_merge (function, optional) :  Calculation method for merging feature values of distance in the same set of substructures. Default to sum.
            fragment_set (bool, optional) : True if you want to include this substructure type to extract the distance. Default to True.
            atom_set (bool, optional) : Types True if you want to include this substructure type to extract the distance. Default to True
            fingerprint_set (bool, optional) : True if you want to include this substructure type to extract the distance. Default to True
        """
        
        self.sm_list = sm_list
        self.radius = radius
        self.func_dis = func_dis
        self.func_merge = func_merge
        self.fragment_set = fragment_set
        self.atom_set = atom_set
        self.fingerprint_set = fingerprint_set
        self.nBit = nBit
        
    def mfp_subset(self, mol):
        
        bitinfo = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.nBit, bitInfo=bitinfo) 

        mfp=[]
        for k,v in bitinfo.items():
            r = v[0][1]
            atom_index = v[0][0]

            if r != 0:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_index)
                submol=Chem.PathToSubmol(mol,env)
                sub_smile = Chem.MolToSmiles(submol)

            else:
                sub_smile=mol.GetAtomWithIdx(atom_index).GetSymbol()

            mfp.append(sub_smile)

        mfp_set = sorted(set(mfp), key=len, reverse=True)
        
        return mfp_set
        
    def extract_mol_features(self, sm):
        
        topological_distance = dict()
        
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            print(f"RDKit failed to read SMILES: {sm}")
            return topological_distance
        
        all_dic = dict()
        mol_distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)
        eps = 1.0e-10
        
         # atom index of Fragment
        if self.fragment_set:
            sm_list = ['C1C=CC=C1',
                        'S1N=C2C=CN=CC2=N1',
                        'S1N=C2C=CC=CC2=N1',
                        'O1C=CC2=CSC=C12',
                        'C1=CC=NC=C1',
                        'C1=NC=NC=N1',
                        'O1C=CC=C1',
                        '[SiH2]1C=CC=C1',
                        '[SiH2]1C=C2C=CC=CC2=C1',
                        'N1C=CC2=CSC=C12',
                        'C1=CC2=CC=CC=C2C=C1',
                        'S1C=CC=C1',
                        'S1C=C2SC=CC2=C1',
                        'O1C=C2C=CC=CC2=C1',
                        '[Se]1C=CC=C1',
                        'N1C=C2C=CC=CC2=C1',
                        'N1C=CC=C1',
                        'S1C=C2C(=C1)C1=CC=CC=C1C1=CC=CC=C21',
                        'C1=CC=CC=C1',
                        'S1C=C2N=CC=NC2=C1',
                        'S1C=CN=C1',
                        '[SiH2]1C=CC2=CSC=C12',
                        'S1C=C2C=CC=CC2=C1',
                        'S1C=C2[Se]C=CC2=C1',
                        'C1C=C2C=CC=CC2=C1',
                        'C1C=CC2=CSC=C12']
            
            fragment_atomidx_dic = defaultdict(list)
            for smiles in sm_list:
                atm_set=mol.GetSubstructMatches(Chem.MolFromSmiles(smiles))
                if atm_set != ():
                    fragment_atomidx_dic[f'{smiles}_CEPfrag']=[list(atm_idx) for atm_idx in atm_set]
            all_dic.update(fragment_atomidx_dic)

        # atom index of HeavyAtom
        if self.atom_set:
            atom_set=set()
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                atom_set.add(symbol)
                
            heteroAtm_smiles=[atom for atom in atom_set if atom != 'C']
            hetero_atomidx_dic = defaultdict(list)
            for hatm_sm in heteroAtm_smiles:
                try:
                    hatm_idx_defalts = mol.GetSubstructMatches(Chem.MolFromSmiles(hatm_sm))
                    hatm_idx = []
                    for hatm_idx_ in hatm_idx_defalts:
                        hatm_idx = [hatm_idx_[0]]
                        hetero_atomidx_dic[hatm_sm].append(hatm_idx)
                except:
                    pass
            all_dic.update(hetero_atomidx_dic)
            
        if self.fingerprint_set:
            sub_atomidx_dic = defaultdict(list)
            mol_graph = AtomGraph(mol=mol)
            
            for mol_distance in mol_distance_matrix:
                sub_atmidx = [idx for idx in range(len(mol_distance)) if mol_distance[idx] <= self.radius]
                sub_vertices = [mol_graph.vertices[idx] for idx in sub_atmidx]
                sub_mol = mol_graph.translate_to_mol(vertices=sub_vertices, sanitize=False)
                sub_atomidx_dic[Chem.MolToSmiles(sub_mol)].append(sub_atmidx)

            sub_atomidx_dic_sort = sorted(sub_atomidx_dic, key=len, reverse=True)
            all_atmidx_setlist_defalt = []
            sub_atomidx_dic_slc = defaultdict(list)
            for i in all_dic.values():
                for j in i:
                    all_atmidx_setlist_defalt.append(j)

            for sub_sm in sub_atomidx_dic_sort:
                for x in sub_atomidx_dic[sub_sm]:
                    flag = True

                    for y in all_atmidx_setlist_defalt:
                        if len(set(x) & set(y)) == len(x):
                            flag = False

                    if flag == True:
                        sub_atomidx_dic_slc[sub_sm].append(list(x))
                        all_atmidx_setlist_defalt.append(x)

            all_dic.update(sub_atomidx_dic_slc)

                        
        dis_dic = defaultdict(list)

        for (sm1, sm2) in itertools.combinations_with_replacement(sorted(all_dic.keys(), key=len, reverse=True), 2):
            sub_pair = f'{sm1} & {sm2}'
            
            # collect each pair of substracture distance
            target1 = all_dic[sm1]
            target2 = all_dic[sm2]
            if sm1 == sm2:
                for i in range(len(target1)):
                    for j in range(i+1, len(target2)):
                        distance_tmp = []
                        for x in target1[i]:
                            for y in target2[j]:
                                distance_tmp.append(mol_distance_matrix[x][y])
                        av_dis = sum(distance_tmp)/len(distance_tmp)

                        if isinstance(self.func_dis, types.FunctionType):
                            calc_dis = self.func_dis(av_dis)

                        else:
                            calc_dis = av_dis**self.func_dis

                        dis_dic[sub_pair].append(calc_dis)
                        
            else:
                for i in range(len(target1)):
                    for j in range(len(target2)):
                        distance_tmp = []
                        for x in target1[i]:
                            for y in target2[j]:
                                distance_tmp.append(mol_distance_matrix[x][y])
                        av_dis = sum(distance_tmp) / len(distance_tmp)

                        if isinstance(self.func_dis,types.FunctionType):
                            calc_dis = self.func_dis(av_dis)

                        else:
                            calc_dis = av_dis ** self.func_dis

                        dis_dic[sub_pair].append(calc_dis)
        

            if dis_dic[sub_pair] == []:
                distance_final = 0
            else:
                distance_final = self.func_merge(dis_dic[sub_pair])


            if distance_final-eps > 0:
                topological_distance[sub_pair] = distance_final

            elif distance_final < 0:
                logger.error(f'Minus feature for mol:{Chem.MolToSmiles(mol)} feature:{sub_pair}')

        return topological_distance        
        
    def feature_extraction(self):
        
        key_all={}
        dic_all=[]
        for sm in self.sm_list:
            tdims_dic = self.extract_mol_features(sm)
            dic_all.append(tdims_dic)
            key_all = {**key_all, **tdims_dic}
        
        X=[]
        for dic in dic_all:
            x = [dic.get(key, 0) for key in key_all]
            X.append(x)
        X = np.array(X)
        return X, key_all
    
def encode(sm_list, radius=1, func_dis=-2, func_merge=sum, fragment_set=True, atom_set=True, fingerprint_set=True):
    tdims = TDiMS(sm_list, radius=radius, func_dis=func_dis, func_merge=func_merge, fragment_set=fragment_set, atom_set=atom_set, fingerprint_set=fingerprint_set)
    X, key_all = tdims.feature_extraction()
    
    return X, key_all