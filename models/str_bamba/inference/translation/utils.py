from nltk.translate.bleu_score import sentence_bleu
import selfies as sf

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def tanimoto_similarity(str1, str2, selfies=False):

    if selfies:
        try:
            smiles1 = sf.decoder(str1)
            smiles2 = sf.decoder(str2)
        except:
            return 0
    else:
        smiles1 = str1
        smiles2 = str2

    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
    
        # Morgan fingerprint
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0

    return similarity


def valid_smiles_selfies(str1, selfies=False):
    
    if selfies:
        try:
            smiles = sf.decoder(str1)
        except:
            return False
    else:
        smiles = str1

    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return False

    return mol is not None


def exact_match(str1, str2):
    return str1 == str2


def encode_str(tokenizer, string):
    ids = tokenizer(string, padding=True, truncation=True)['input_ids']
    tokens = tokenizer.batch_decode(ids)
    return tokens


def get_bleu1(tokenizer, target, generated):
    candidate = encode_str(tokenizer, generated)
    references = encode_str(tokenizer, target)
    
    return sentence_bleu([references], candidate, weights=(1,))


def get_bleu2(tokenizer, target, generated):
    candidate = encode_str(tokenizer, generated)
    references = encode_str(tokenizer, target)
    
    return sentence_bleu([references], candidate, weights=(0.5, 0.5))


def jaccard_similarity(set1, set2):
    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))
    
    return intersection / union


def decode(model, encoder_input, decoder_input, decoder_target, verbose=True, device='cuda', **kwargs):
    # tokenization
    encoder_input_ids = model.tokenizer(encoder_input, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)
    decoder_input_ids = model.tokenizer(decoder_input, padding=True, truncation=True, return_tensors='pt')['input_ids'][:, :-1].to(device)
    decoder_target_ids = model.tokenizer(decoder_target, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)

    if verbose:
        print('Encoder input:', model.tokenizer.batch_decode(encoder_input_ids))
        print('Decoder input:', model.tokenizer.batch_decode(decoder_input_ids))
        print('Decoder target:', model.tokenizer.batch_decode(decoder_target_ids))
        print('Target:', decoder_target_ids)
    
    # encoder forward
    encoder_hidden_states = model.encoder(encoder_input_ids).hidden_states
    if verbose:
        print('Encoder hidden states:', encoder_hidden_states.shape)

    # decoder generation
    output = model.decoder.generate(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden_states,
        max_length=decoder_target_ids.shape[1],
        cg=False,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=1.0,
        top_k=1,
        top_p=.0,
        min_p=0.,
        repetition_penalty=1,
        # eos_token_id=1,
    )
    decoded = model.tokenizer.batch_decode(output.sequences, clean_up_tokenization_spaces=True, skip_special_tokens=False)

    outs = []
    for out in decoded:
        generated = ''.join(''.join(out).split(' ')).replace('<bos>', '').replace('<sep>', '').replace(decoder_input, '')
        outs.append(generated)
    
    return outs


def generate_multiples(model, encoder_input, decoder_input, decoder_target, n=3, verbose=True, device='cuda'):
    # tokenization
    encoder_input_ids = model.tokenizer(encoder_input, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)
    decoder_input_ids = model.tokenizer(decoder_input, padding=True, truncation=True, return_tensors='pt')['input_ids'][:, :-1].to(device)
    decoder_target_ids = model.tokenizer(decoder_target, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)

    if verbose:
        print('Encoder input:', model.tokenizer.batch_decode(encoder_input_ids))
        print('Decoder input:', model.tokenizer.batch_decode(decoder_input_ids))
        print('Decoder target:', model.tokenizer.batch_decode(decoder_target_ids))
        print('Target:', decoder_target_ids)
    
    # encoder forward
    encoder_hidden_states = model.encoder(encoder_input_ids).hidden_states
    if verbose:
        print('Encoder hidden states:', encoder_hidden_states.shape)

    # decoder generation
    list_generated = []
    while len(list_generated) < n:
        output = model.decoder.generate(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            max_length=decoder_target_ids.shape[1],
            # max_length=202,
            cg=False,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=1.5,
            top_k=5,
            top_p=0.,
            min_p=0.,
            repetition_penalty=1,
            # eos_token_id=1,
        )
        decoded = model.tokenizer.batch_decode(output.sequences, clean_up_tokenization_spaces=True, skip_special_tokens=False)
    
        outs = []
        for out in decoded:
            generated = ''.join(''.join(out).split(' ')).replace('<bos>', '').replace('<sep>', '').replace(decoder_input, '')
            outs.append(generated)

        if outs[0] not in list_generated:
            list_generated.extend(outs)
            # print(outs)
    
    return list_generated