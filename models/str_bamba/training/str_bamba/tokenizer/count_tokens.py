from str_tokenizer import load_tokenizer
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import itertools


def main():
    tokenizer = load_tokenizer("str_bamba_tokenizer.json")

    pubchem_files = [f'normprops_{i}.csv' for i in range(67)]
    ds_pubchem = load_dataset('./str_pubchem/', data_files={'train': pubchem_files}, split="train", trust_remote_code=True)
    ds_polymer = load_dataset('./str_polymer/', data_files={'train': ['polymer_pretrain_v1.csv']}, split="train", trust_remote_code=True)
    ds_formulation = load_dataset('./str_formulation/', data_files={'train': ['formulation_data.csv']}, split="train", trust_remote_code=True)
    ds = concatenate_datasets([ds_pubchem, ds_polymer, ds_formulation])
    print(ds)

    total_tokens = 0
    for batch in (pbar := tqdm(ds.iter(batch_size=1), total=len(ds))):
        representations = [
            batch['MOLECULAR_FORMULA'],
            batch['CANONICAL_SMILES'],
            batch['IUPAC_NAME'],
            batch['INCHI'],
            batch['SELFIES'],
            batch['POLYMER SMILES'],
            batch['FORMULATION'],
        ]
        molecules = list(filter(None, itertools.chain.from_iterable(representations)))
        tokens = tokenizer(molecules)['input_ids']
        total_tokens += sum([len(x) for x in tokens])

        pbar.set_postfix(total_tokens=total_tokens)
        pbar.update()

    with open('str-bamba_total_tokens.txt', 'w') as f:
        f.write(str(total_tokens))


if __name__ == '__main__':
    main()