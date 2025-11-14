# materials.smi-TED - INFERENCE (Regression)
# Fixed model weights


dataset_name = 'esol' # 'lipophilicity' | 'freesolv' | 'qm8' | 'qm9' | 'esol'
ycolumn_name = {
    'lipophilicity': 'y',
    'freesolv': 'expt',
    'qm8': 'E1-CAM',
    'qm9': 'gap',
    'esol': 'measured log solubility in mols per litre'
    }  # target column name in the dataset


import os
import sys

# Data
import torch
import pandas as pd
import numpy as np

# Chemistry
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors

# ML
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# materials.smi-ted (smi-ted)

inference_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'smi_ted', 'inference', 'smi_ted_light'))
finetune_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'smi_ted', 'finetune'))

sys.path.append(inference_dir)

from load import load_smi_ted

# function to canonicalize SMILES
def normalize_smiles(smi, canonical=True, isomeric=False):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized

# ### Import smi-ted

def main():

    model_smi_ted = load_smi_ted(
        folder=inference_dir,
        ckpt_filename='smi-ted-Light_40.pt'
    )

    # Experiments - Data Load

    df_train = pd.read_csv(f"{finetune_dir}/moleculenet/{dataset_name}/train.csv")
    df_test = pd.read_csv(f"{finetune_dir}/moleculenet/{dataset_name}/test.csv")

    # SMILES canonization

    df_train['norm_smiles'] = df_train['smiles'].apply(normalize_smiles)
    df_train_normalized = df_train.dropna()
    print(f'df_train_normalized.shape: {df_train_normalized.shape}')

    df_test['norm_smiles'] = df_test['smiles'].apply(normalize_smiles)
    df_test_normalized = df_test.dropna()
    print(f'df_test_normalized.shape: {df_test_normalized.shape}')

    # smi-ted embeddings (encoding into latent space)
    # This converts smiles into contextualized embeddings by passing them through the smi-ted encoder
    # Then aggregating the token embeddings into a single fixed-size vector with an MLP pooling layer

    with torch.no_grad():
        df_embeddings_train = model_smi_ted.encode(df_train_normalized['norm_smiles'])

    with torch.no_grad():
        df_embeddings_test = model_smi_ted.encode(df_test_normalized['norm_smiles'])

    # XGBoost prediction using the whole Latent Space

    xgb_predict = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=4)
    xgb_predict.fit(df_embeddings_train, df_train_normalized[ycolumn_name[dataset_name]])

    # get XGBoost predictions
    y_pred = xgb_predict.predict(df_embeddings_test)

    rmse = np.sqrt(mean_squared_error(df_test_normalized[ycolumn_name[dataset_name]], y_pred))
    print(f"RMSE Score: {rmse:.4f}")

if __name__ == "__main__":
    
    main()