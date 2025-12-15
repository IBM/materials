from str_bamba.load import load_bamba
from utils import (
    decode,
    get_bleu1,
    get_bleu2,
    jaccard_similarity,
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
        ckpt_filename='STR-Bamba_decoder_36000.pt', 
        tokenizer_filename='tokenizer_v3.json',
        device='cuda',
        dtype=torch.float16
    )
    model.eval()

    # load data
    df = pd.read_csv('../data/datasets/pubchem_translation_sample_3K.csv')
    print(df.shape)

    # define input and output
    input_representation = df[args.input_representation]
    target_representation = df[args.output_representation]

    match_score = 0
    bleu1_score = 0
    bleu2_score = 0
    jaccard_score = 0
    for idx, (inp, tgt) in enumerate(pbar := tqdm(list(zip(input_representation, target_representation)))):
        encoder_input = inp
        decoder_input = re.findall(r'<.*?>', tgt)[0]
        decoder_target = tgt
        target = decoder_target.replace(decoder_input, '')
        
        generated = decode(model, encoder_input, decoder_input, decoder_target, verbose=False)[0]
        # print(generated)

        match = exact_match(target, generated)
        bleu1 = get_bleu1(model.tokenizer, target, generated)
        bleu2 = get_bleu2(model.tokenizer, target, generated)
        jaccard = jaccard_similarity(set(target), set(generated))

        bleu1_score += bleu1
        bleu2_score += bleu2
        match_score += int(match)
        jaccard_score += jaccard

        pbar.set_postfix(
            match_score=match_score/(idx+1), 
            bleu1_score=bleu1_score/(idx+1), 
            bleu2_score=bleu2_score/(idx+1), 
            jaccard_score=jaccard_score/(idx+1)
        )
        pbar.update()

    df_results = pd.DataFrame({
        'Translation': [f'{args.input_representation} -> {args.output_representation}'],
        'BLEU-1': [bleu1_score/len(input_representation)],
        'BLEU-2': [bleu2_score/len(input_representation)],
        'Jaccard Similarity': [jaccard_score/len(input_representation)],
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