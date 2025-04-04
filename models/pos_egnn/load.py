import torch
from .posegnn.calculator import PosEGNNCalculator
import ase
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

def smiles_to_atoms(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol)
    ase_atoms = ase.Atoms(
        numbers=[
            atom.GetAtomicNum() for atom in mol.GetAtoms()
        ],
        positions=mol.GetConformer().GetPositions()
    )
    return ase_atoms

class POSEGNN():
    def __init__(self, use_gpu=True):
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.device = device
        self.calculator = None

    def load(self, checkpoint=None):
        repo_id = "ibm-research/materials.pos-egnn"
        filename = "pytorch_model.bin"
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.calculator = PosEGNNCalculator(model_path, device=self.device, compute_stress=False)

    def encode(self, smiles_list, return_tensor=False, batch_size=32):
        results = []

        # make batch-wise processing with progress bar
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Batch Encoding"):
            batch = smiles_list[i:i+batch_size]
            atoms_batch = []

            for smiles in batch:
                try:
                    atoms = smiles_to_atoms(smiles)
                    atoms.calc = self.calculator
                    atoms_batch.append(atoms)
                except Exception as e:
                    print(f"Skipping {smiles}: {e}")

            if atoms_batch:
                embeddings = [a.get_invariant_embeddings().mean(dim=0).cpu() for a in atoms_batch]
                batch_tensor = torch.stack(embeddings)
                results.append(batch_tensor)

        if not results:
            raise RuntimeError("No valid SMILES could be processed.")

        all_embeddings = torch.cat(results, dim=0)
        return all_embeddings if return_tensor else pd.DataFrame(all_embeddings.numpy())

