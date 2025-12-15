import pandas as pd
from str_tokenizer import load_tokenizer
from tqdm import tqdm
from collections import namedtuple


df_pubchem = pd.read_csv('./normprops_100k_sample.csv')
df_polymer = pd.read_csv('./polymer_100k_sample.csv')
df_formulation = pd.read_csv('./formulation_data.csv')

data = (
    df_pubchem['MOLECULAR_FORMULA'].to_list()
    + df_pubchem['CANONICAL_SMILES'].to_list()
    + df_pubchem['IUPAC_NAME'].to_list()
    + df_pubchem['INCHI'].to_list()
    + df_pubchem['SELFIES'].to_list()
    + df_polymer['POLYMER SMILES'].to_list()
    + df_formulation['FORMULATION'].to_list()
)
print(len(data))

tokenizer = load_tokenizer("str_bamba_tokenizer.json")

AboveStatistics = namedtuple('AboveStatistics', ['formula', 'smiles', 'iupac', 'inchi', 'selfies', 'polymer', 'formulation'])
above_202 = AboveStatistics(0, 0, 0, 0, 0, 0, 0)
above_2048 = AboveStatistics(0, 0, 0, 0, 0, 0, 0)
len_tokens = []
for d in tqdm(data):
    num_tokens = len(tokenizer(d)['input_ids'])

    # more than 202 tokens
    if num_tokens > 202 and d.startswith('<formula>'):
        above_202 = AboveStatistics(
            above_202.formula + 1,
            above_202.smiles,
            above_202.iupac,
            above_202.inchi,
            above_202.selfies,
            above_202.polymer,
            above_202.formulation
        )
    elif num_tokens > 202 and d.startswith('<smiles>'):
        above_202 = AboveStatistics(
            above_202.formula,
            above_202.smiles + 1,
            above_202.iupac,
            above_202.inchi,
            above_202.selfies,
            above_202.polymer,
            above_202.formulation
        )
    elif num_tokens > 202 and d.startswith('<iupac>'):
        above_202 = AboveStatistics(
            above_202.formula,
            above_202.smiles,
            above_202.iupac + 1,
            above_202.inchi,
            above_202.selfies,
            above_202.polymer,
            above_202.formulation
        )
    elif num_tokens > 202 and d.startswith('<inchi>'):
        above_202 = AboveStatistics(
            above_202.formula,
            above_202.smiles,
            above_202.iupac,
            above_202.inchi + 1,
            above_202.selfies,
            above_202.polymer,
            above_202.formulation
        )
    elif num_tokens > 202 and d.startswith('<selfies>'):
        above_202 = AboveStatistics(
            above_202.formula,
            above_202.smiles,
            above_202.iupac,
            above_202.inchi,
            above_202.selfies + 1,
            above_202.polymer,
            above_202.formulation
        )
    elif num_tokens > 202 and d.startswith('<polymer_spg>'):
        above_202 = AboveStatistics(
            above_202.formula,
            above_202.smiles,
            above_202.iupac,
            above_202.inchi,
            above_202.selfies,
            above_202.polymer + 1,
            above_202.formulation
        )
    elif num_tokens > 202 and d.startswith('<formulation_start>'):
        above_202 = AboveStatistics(
            above_202.formula,
            above_202.smiles,
            above_202.iupac,
            above_202.inchi,
            above_202.selfies,
            above_202.polymer,
            above_202.formulation + 1
        )
    
    # more than 2048 tokens
    if num_tokens > 2048 and d.startswith('<formula>'):
        above_2048 = AboveStatistics(
            above_2048.formula + 1,
            above_2048.smiles,
            above_2048.iupac,
            above_2048.inchi,
            above_2048.selfies,
            above_2048.polymer,
            above_2048.formulation
        )
    elif num_tokens > 2048 and d.startswith('<smiles>'):
        above_2048 = AboveStatistics(
            above_2048.formula,
            above_2048.smiles + 1,
            above_2048.iupac,
            above_2048.inchi,
            above_2048.selfies,
            above_2048.polymer,
            above_2048.formulation
        )
    elif num_tokens > 2048 and d.startswith('<iupac>'):
        above_2048 = AboveStatistics(
            above_2048.formula,
            above_2048.smiles,
            above_2048.iupac + 1,
            above_2048.inchi,
            above_2048.selfies,
            above_2048.polymer,
            above_2048.formulation
        )
    elif num_tokens > 2048 and d.startswith('<inchi>'):
        above_2048 = AboveStatistics(
            above_2048.formula,
            above_2048.smiles,
            above_2048.iupac,
            above_2048.inchi + 1,
            above_2048.selfies,
            above_2048.polymer,
            above_2048.formulation
        )
    elif num_tokens > 2048 and d.startswith('<selfies>'):
        above_2048 = AboveStatistics(
            above_2048.formula,
            above_2048.smiles,
            above_2048.iupac,
            above_2048.inchi,
            above_2048.selfies + 1,
            above_2048.polymer,
            above_2048.formulation
        )
    elif num_tokens > 2048 and d.startswith('<polymer_spg>'):
        above_2048 = AboveStatistics(
            above_2048.formula,
            above_2048.smiles,
            above_2048.iupac,
            above_2048.inchi,
            above_2048.selfies,
            above_2048.polymer + 1,
            above_2048.formulation
        )
    elif num_tokens > 2048 and d.startswith('<formulation_start>'):
        above_2048 = AboveStatistics(
            above_2048.formula,
            above_2048.smiles,
            above_2048.iupac,
            above_2048.inchi,
            above_2048.selfies,
            above_2048.polymer,
            above_2048.formulation + 1
        )

    len_tokens.append(num_tokens)

print('Molecules with more than 202 tokens:', above_202)
print('Molecules with more than 2048 tokens:', above_2048)
print(pd.Series(len_tokens).describe())
pd.Series(len_tokens).to_csv('count_tokens_sample.csv', index=False)
