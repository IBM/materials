# Standard library
from heapq import nsmallest

# Numerical computing & plotting
import numpy as np
import matplotlib.pyplot as plt

# PySCF core (quantum chemistry)
from pyscf import lib
from pyscf import gto, scf, dft
from pyscf.dft import numint, gen_grid
from pyscf.tools import cubegen
from pyscf.geomopt.berny_solver import optimize
from pyscf.semiempirical import mindo3

# PyTorch
import torch.nn.functional as F

# RDKit (molecule parsing & conformer generation)
from rdkit import Chem
from rdkit.Chem import rdDistGeom, AllChem


def change_grid_size(tensor, size):
    new_grid = F.interpolate(
        tensor,
        size=size,
        mode="trilinear",
        align_corners=False
    )
    return new_grid


def get_grid_from_smiles(data_smi_l):
    density_grids = []  
    for smi_it in data_smi_l:
        fin_tmp_l = []

        mol_it = Chem.MolFromSmiles(smi_it, sanitize=True)

        # normalize to canonical SMILES for bookkeeping
        if mol_it != None:
            try:
                can_smi_it = Chem.MolToSmiles(mol_it, kekuleSmiles=True)
            except:
                can_smi_it = Chem.MolToSmiles(mol_it, kekuleSmiles=False)

        print('\n molecule ', smi_it)
        
        # Embedding with Molecular Force Field
        #     embed 50 conformations, optimize with rdkit: N_MMFF = 50
        #     select 1 most stable MMFF conformations, optimize with pyscf N_PYSCF = 1

        N_MMFF = 50
        N_PYSCF = 1
        
        confmol = Chem.AddHs(Chem.Mol(mol_it))
        param = rdDistGeom.ETKDGv2()
        param.pruneRmsThresh = 0.1
        cids = rdDistGeom.EmbedMultipleConfs(confmol, N_MMFF, param)

        if len(cids) == 0:
            continue

        try:
            res = AllChem.MMFFOptimizeMoleculeConfs(confmol)
            energies = {c: res[c][1] for c in range(len(res))}
            opt_mols = {}
            top_energies = {}
            top_cids = nsmallest(N_PYSCF, energies, key=energies.get)
        except Exception as error:
            print('Something went wrong, MMFFOptimize')

        # PySCF optimization and cube generation
        for cid in top_cids:
            print('\n ----> Conformer ', str(cid), '\n')
            molstr = Chem.MolToXYZBlock(confmol, confId=cid)
            mol = gto.M(atom='; '.join(molstr.split('\n')[2:]))
            mf = scf.RHF(mol)

            mol_eq = optimize(mf, maxsteps=200)
            opt_mols[cid] = mol_eq
            mol_eq_f = scf.RHF(mol_eq).run()
            top_energies[cid] = mol_eq_f.e_tot

            box0 = max(mol_eq_f.mol.atom_coords()[:, 0]) - min(mol_eq_f.mol.atom_coords()[:, 0])
            box1 = max(mol_eq_f.mol.atom_coords()[:, 1]) - min(mol_eq_f.mol.atom_coords()[:, 1])
            box2 = max(mol_eq_f.mol.atom_coords()[:, 2]) - min(mol_eq_f.mol.atom_coords()[:, 2])
            n0 = 6 * (int(box0) + 2)
            n1 = 6 * (int(box1) + 2)
            n2 = 6 * (int(box2) + 2)
            el_cube = cubegen.density(
                mol_eq_f.mol,
                f"SMILES_{data_smi_l.index(smi_it)}_{cid}.cube",
                mol_eq_f.make_rdm1(),
                nx=n0,
                ny=n1,
                nz=n2,
            )
            rho = el_cube                
            density_grids.append(
                {
                    "smiles": smi_it,
                    "name": f"SMILES_{data_smi_l.index(smi_it)}_{cid}",
                    "rho":  rho
                }
            )

    return density_grids 


def plot_voxel_grid(tensor, thresholds=[0.5, 0.25, 0.125, 0.0125], title='Voxel Grid Plot'):
    """
    Plots a 3D voxel grid from a tensor and shows it inline.
    
    Args:
        tensor (torch.Tensor): input shape [1,1,D,H,W]
        thresholds (list of float): visibility cutoffs
        title (str): plot title
    """
    # Convert to NumPy and squeeze out batch/channel dims
    data_np = tensor.detach().squeeze().cpu().numpy()

    # Build normalized grid coordinates
    x, y, z = np.indices(np.array(data_np.shape) + 1) / (max(data_np.shape) + 1)

    # Predefine colors & alpha
    alpha = 0.3
    colors = [
        [1.00, 0.00, 0.00, alpha],
        [0.75, 0.00, 0.25, alpha],
        [0.50, 0.00, 0.50, alpha],
        [0.25, 0.00, 0.75, alpha],
        [0.00, 0.00, 1.00, alpha],
    ]

    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.tick_params(left=False, right=False, labelleft=False,
                   labelbottom=False, bottom=False)
    ax.grid(False)
    if title:
        ax.set_title(title)
    # Plot one layer per threshold
    for i, thr in enumerate(thresholds):
        mask = np.clip(data_np - thr, 0, 1)
        ax.voxels(x, y, z, mask, facecolors=colors[i % len(colors)], linewidth=0.5, alpha=alpha)

    plt.tight_layout()
    plt.show()