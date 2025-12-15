"""
Code adapted from:
    - https://github.com/huggingface/transformers/issues/17862
    - https://github.com/huggingface/tokenizers/blob/main/bindings/python/examples/train_with_datasets.py
    - https://github.com/huggingface/tokenizers/blob/main/bindings/python/examples/custom_components.py
"""


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split, Sequence, ByteLevel, Whitespace, PreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset, concatenate_datasets
from str_tokenizer import MoleculePreTokenizer, load_tokenizer
from special_tokens import STR_SPECIAL_TOKENS

import pandas as pd
import itertools
import random
import re
from tqdm import tqdm


def batch_iterator(dataset):
    batch_size = 1
    for batch in tqdm(dataset.iter(batch_size=batch_size), total=len(dataset)):
        representations = [
            batch['MOLECULAR_FORMULA'],
            batch['CANONICAL_SMILES'],
            batch['IUPAC_NAME'],
            batch['INCHI'],
            batch['SELFIES'],
            batch['POLYMER SMILES'],
            batch['FORMULATION'],
        ]
        batch = list(filter(None, itertools.chain.from_iterable(representations)))
        yield batch


def main():
    ### dataset ###
    rand_files = random.sample(range(67), 3)  # randomly choose 3 .csv files
    pubchem_files = [f'normprops_{i}.csv' for i in rand_files]
    ds_pubchem = load_dataset('./str_pubchem/', data_files={'train': pubchem_files}, split="train", trust_remote_code=True)
    ds_polymer = load_dataset('./str_polymer/', data_files={'train': ['polymer_pretrain_v1.csv']}, split="train", trust_remote_code=True)
    ds_formulation = load_dataset('./str_formulation/', data_files={'train': ['formulation_data.csv']}, split="train", trust_remote_code=True)
    ds = concatenate_datasets([ds_pubchem, ds_polymer, ds_formulation])
    print(ds)
    print('Starting training')

    ### trainer ###
    trainer = BpeTrainer(
        vocab_size=5000,
        show_progress=False,
        special_tokens=[
            # basic
            STR_SPECIAL_TOKENS['BOS_TOKEN'],
            STR_SPECIAL_TOKENS['EOS_TOKEN'],
            STR_SPECIAL_TOKENS['PAD_TOKEN'],
            STR_SPECIAL_TOKENS['MASK_TOKEN'],
            STR_SPECIAL_TOKENS['UNK_TOKEN'],
            # representations
            STR_SPECIAL_TOKENS['MOLECULAR_FORMULA_TOKEN'],
            STR_SPECIAL_TOKENS['SMILES_TOKEN'],
            STR_SPECIAL_TOKENS['IUPAC_TOKEN'],
            STR_SPECIAL_TOKENS['INCHI_TOKEN'],
            STR_SPECIAL_TOKENS['SELFIES_TOKEN'],
            STR_SPECIAL_TOKENS['POLYMER_SPG_TOKEN'],
            STR_SPECIAL_TOKENS['FORMULATION_START_TOKEN'],
            STR_SPECIAL_TOKENS['FORMULATION_END_TOKEN'],
            # force tokens
            STR_SPECIAL_TOKENS['INCHI_INITIAL_TOKEN'],
            STR_SPECIAL_TOKENS['INCHI_COMMA_TOKEN'],
            STR_SPECIAL_TOKENS['INCHI_DASH_TOKEN'],
            STR_SPECIAL_TOKENS['INCHI_FORWARDSLASH_TOKEN'],
            STR_SPECIAL_TOKENS['INCHI_QUESTIONMARK_TOKEN'],
            STR_SPECIAL_TOKENS['INCHI_PARENTHESIS_OPEN_TOKEN'],
            STR_SPECIAL_TOKENS['INCHI_PARENTHESIS_CLOSE_TOKEN'],
            STR_SPECIAL_TOKENS['POLYMER_ARROW_TOKEN'],
        ]
    )

    ### tokenizer ###
    tokenizer = Tokenizer(BPE(unk_token=STR_SPECIAL_TOKENS['UNK_TOKEN']))

    # pre-tokenizer
    tokenizer.pre_tokenizer = PreTokenizer.custom(MoleculePreTokenizer())

    # training
    tokenizer.train_from_iterator(batch_iterator(ds), trainer=trainer)

    # post-processor
    tokenizer.post_processor = TemplateProcessing(
        single=STR_SPECIAL_TOKENS['BOS_TOKEN'] + " $A " + STR_SPECIAL_TOKENS['EOS_TOKEN'],
        pair=f"{STR_SPECIAL_TOKENS['BOS_TOKEN']} $A {STR_SPECIAL_TOKENS['EOS_TOKEN']} $B:1 {STR_SPECIAL_TOKENS['EOS_TOKEN']}:1",
        special_tokens=[
            (STR_SPECIAL_TOKENS['BOS_TOKEN'], tokenizer.token_to_id(STR_SPECIAL_TOKENS['BOS_TOKEN'])),
            (STR_SPECIAL_TOKENS['EOS_TOKEN'], tokenizer.token_to_id(STR_SPECIAL_TOKENS['EOS_TOKEN'])),
        ],
    )

    print('Vocabulary size:', tokenizer.get_vocab_size())

    ### save tokenizer ###
    tokenizer.pre_tokenizer = Whitespace()  # resolve issue with tokenizer package to serialize pre_tokenizer
    tokenizer.save('str_bamba_tokenizer.json')
    
    ### testing ###
    tok = load_tokenizer("./str_bamba_tokenizer.json")
    print(len(tok))

    input_ids = tok('<polymer>[*:1]CC(CO)(CCSC)CO[*:2];2->1')['input_ids']
    print('')
    print(' '.join([tok.convert_ids_to_tokens(idx) for idx in input_ids]))

    input_ids = tok('<polymer>[*:1]NC(Cc1cccnc1N1C(=O)c2cc(Oc3ccccc3C)c(-c3ccc4c(c3)C(=O)N([*:2])C4=O)cc2C1=O)c1ccncc1C;2->1')['input_ids']
    print('')
    print(' '.join([tok.convert_ids_to_tokens(idx) for idx in input_ids]))

    input_ids = tok('<selfies>[C][=C][C][C][C][C][C][C][C][C][C][Branch1][S][O][C][C][C][Branch1][C][C][C][C][C][=C][Branch1][C][C][C][O][C][C][C][Branch1][C][C][C][C][C][=C][Branch1][C][C][C]')['input_ids']
    print('')
    print(' '.join([tok.convert_ids_to_tokens(idx) for idx in input_ids]))

    input_ids = tok('<smiles>Cc1cc(C)c(COc2ccc(Br)cc2)cc1C=C(C#N)C(N)=S')['input_ids']
    print('')
    print(' '.join([tok.convert_ids_to_tokens(idx) for idx in input_ids]))

    input_ids = tok('<iupac>N,N-diethyl-3,3-diphenylprop-2-en-1-amine')['input_ids']
    print('')
    print(' '.join([tok.convert_ids_to_tokens(idx) for idx in input_ids]))

    input_ids = tok('<formula>C22H20BrFN4O2')['input_ids']
    print('')
    print(' '.join([tok.convert_ids_to_tokens(idx) for idx in input_ids]))

    input_ids = tok('<inchi>InChI=1S/C22H16BrN3O/c1-14-5-4-8-21(24-14)26-22(27)18-13-20(15-9-11-16(23)12-10-15)25-19-7-3-2-6-17(18)19/h2-13H,1H3,(H,24,26,27)')['input_ids']
    print('')
    print(' '.join([tok.convert_ids_to_tokens(idx) for idx in input_ids]))

    input_ids = tok('<formulation_start>COCCC#N<sep>56.0<sep>COCCC#N<sep>42.0<sep>O=C1N(C)CCCN1C<sep>0.0<sep>[Li+].[N+](=O)([O-])[O-]<sep>2.0<formulation_end>')['input_ids']
    print('')
    print(' '.join([tok.convert_ids_to_tokens(idx) for idx in input_ids]))


if __name__ == '__main__':
    main()