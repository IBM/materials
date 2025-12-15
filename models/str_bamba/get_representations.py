import pandas as pd
import pubchempy
import selfies as sf
import requests
import argparse

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
from tqdm import tqdm
tqdm.pandas()
RDLogger.DisableLog('rdApp.*')
CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"


def smiles_to_iupac_from_cactus(smiles):
    rep = "iupac_name"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url)
    response.raise_for_status()
    return response.text
    

def normalize_smiles(smi, canonical=True, isomeric=False):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), 
            canonical=canonical, 
            isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized


def calculate_representations(row):
    # convert to rdkit object
    try:
        smiles = row['canon_smiles']
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(f'Failed to read SMILES {smiles}:', e)
        return '<formula>', '<iupac>', '<inchi>', '<selfies>'

    # molecular formula
    try:
        formula = '<formula>'+rdMolDescriptors.CalcMolFormula(mol)
    except Exception as e:
        print('Formula failed:', e)
        formula = '<formula>'

    # IUPAC name
    try:
        # try first with pubchempy
        iupac = '<iupac>'+pubchempy.get_compounds(smiles, namespace='smiles')[0].iupac_name
    except Exception as e:
        print('PubchemPy failed:', e)
        try:
            # otherwise, try with CACTUS
            iupac = '<iupac>'+smiles_to_iupac_from_cactus(smiles)
        except Exception as e:
            print('CACTUS failed:', e)
            iupac = '<iupac>'

    # InChI
    try:
        inchi = '<inchi>'+Chem.inchi.MolToInchi(mol)
    except Exception as e:
        print('InChI failed:', e)
        inchi = '<inchi>'

    # SELFIES
    try:
        selfies = '<selfies>'+sf.encoder(smiles)
    except Exception as e:
        print('SELFIES failed:', e)
        selfies = '<selfies>'
        
    return formula, iupac, inchi, selfies


def main(args):
    # load dataset
    df = pd.read_csv(args.dataset)

    # canonicalize SMILES
    df['CANONICAL_SMILES'] = df['smiles'].progress_apply(normalize_smiles)

    # get representations
    print('Getting representations...')
    df[['MOLECULAR_FORMULA', 'IUPAC_NAME', 'INCHI', 'SELFIES']] = df.progress_apply(calculate_representations, axis=1, result_type="expand")

    # add special token to SMILES
    df['CANONICAL_SMILES'] = '<smiles>'+df['CANONICAL_SMILES']

    # save dataset
    df.to_csv(args.dataset, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Dataset path in csv")
    args = parser.parse_args()
    print('Starting script...')
    main(args)