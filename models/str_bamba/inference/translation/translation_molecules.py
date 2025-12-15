from str_bamba.load import load_bamba
from utils import (
    decode,
    generate_multiples,
    tanimoto_similarity,
    valid_smiles_selfies,
    exact_match
)

import torch
import pandas as pd
import numpy as np
import argparse
import random
import re
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # load model
    model = load_bamba(
        base_folder='./str_bamba',
        config_filename='config_encoder-decoder_436M.json',
        ckpt_filename='STR-Bamba_decoder_126000.pt', 
        tokenizer_filename='tokenizer_v3.json',
        device='cuda',
        dtype=torch.bfloat16
    )
    model.eval()

    # load data
    df = pd.read_csv('../data/datasets/pubchem_translation_sample_3K.csv')
    print(df.shape)

    # define input and output
    input_representation = df[args.input_representation]
    target_representation = df[args.output_representation]

    ts_score = 0
    valid_score = 0
    match_score = 0
    for idx, (inp, tgt) in enumerate(pbar := tqdm(list(zip(input_representation, target_representation)))):
        encoder_input = inp
        decoder_input = re.findall(r'<.*?>', tgt)[0]  # define special token for decoder input, i.e., <smiles>
        decoder_target = tgt  # target for decoder
        target = decoder_target.replace(decoder_input, '')  # target for metrics
        
        if args.input_representation == 'MOLECULAR_FORMULA' and (args.output_representation == 'CANONICAL_SMILES' or args.output_representation == 'SELFIES'):
            # print('formula')
            deterministic_gen = decode(model, encoder_input, decoder_input, decoder_target, verbose=False, device=device)[0]
            generateds = generate_multiples(model, encoder_input, decoder_input, decoder_target, n=4, verbose=False, device=device)
            generateds.append(deterministic_gen)
            # generateds = [deterministic_gen]

            ts = []
            valid = []
            match = []
            for gen in generateds:
                # print(gen)
                ts_idx = tanimoto_similarity(target, gen, selfies=True if args.output_representation == 'SELFIES' else False)
                valid_idx = valid_smiles_selfies(gen, selfies=True if args.output_representation == 'SELFIES' else False)
                match_idx = exact_match(target, gen)

                ts.append(ts_idx)
                valid.append(int(valid_idx))
                match.append(int(match_idx))

            ts_score += max(ts)
            valid_score += max(valid)
            match_score += max(match)
        else:
            generated = decode(model, encoder_input, decoder_input, decoder_target, verbose=False, device=device)[0]
            # print(generated)

            ts = tanimoto_similarity(target, generated, selfies=True if args.output_representation == 'SELFIES' else False)
            valid = valid_smiles_selfies(generated, selfies=True if args.output_representation == 'SELFIES' else False)
            match = exact_match(target, generated)

            ts_score += ts
            valid_score += int(valid)
            match_score += int(match)

        pbar.set_postfix(
            ts_score=ts_score/(idx+1), 
            valid_score=valid_score/(idx+1), 
            match_score=match_score/(idx+1)
        )
        pbar.update()

    df_results = pd.DataFrame({
        'Translation': [f'{args.input_representation} -> {args.output_representation}'],
        'Tanimoto Similarity': [ts_score/len(input_representation)],
        'Valid SMILES/SELFIES': [valid_score/len(input_representation)],
        'Exact Match String': [match_score/len(input_representation)],
    })
    df_results.to_csv(f'results_{args.input_representation}_to_{args.output_representation}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_path", type=str, help="The file to process.", required=True)
    parser.add_argument("--input_representation", type=str, help="The file to process.", required=True)
    parser.add_argument("--output_representation", type=str, help="The file to process.", required=True)
    args = parser.parse_args()
    main(args)