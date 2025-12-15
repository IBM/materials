import regex as re
import torch
import numpy as np
import random
import gc
import collections
from transformers import PreTrainedTokenizerFast
from str_bamba.tokenizer.str_tokenizer import load_tokenizer
from str_bamba.tokenizer.special_tokens import STR_SPECIAL_TOKENS

class TextEncoder():

    def __init__(self, tokenizer_path, is_encoder, max_length=2048, decoder_n_alternatives=2, bag=None):

        self.max_length = max_length
        self.min_length = 1
        self.mod_length = 42
        self.avg_length = 66
        self.tail = 122
        self.mlm_probability = .15
        self.nsp_probability = .5
        self.b0_cache=collections.deque()
        self.b1_cache=collections.deque()
        self.b2_cache=collections.deque()
        self.b3_cache=collections.deque()
        self.bucket0=collections.deque()
        self.bucket1=collections.deque()
        self.bucket2=collections.deque()
        self.bucket3=collections.deque()

        self.b0_max=1100
        self.b1_max=700
        self.b2_max=150
        self.b3_max=20

        self.decoder_n_alternatives = decoder_n_alternatives

        # tokenizer
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.char2id = self.tokenizer.vocab
        self.vocab_size = len(self.tokenizer)

        # basic tokens
        self.bos  = self.char2id[STR_SPECIAL_TOKENS['BOS_TOKEN']]
        self.eos  = self.char2id[STR_SPECIAL_TOKENS['EOS_TOKEN']]
        self.pad  = self.char2id[STR_SPECIAL_TOKENS['PAD_TOKEN']]
        self.mask = self.char2id[STR_SPECIAL_TOKENS['MASK_TOKEN']]
        self.unk = self.char2id[STR_SPECIAL_TOKENS['UNK_TOKEN']]

        # special tokens to not mask
        self.special_tokens = [
            self.bos, 
            self.eos, 
            self.pad, 
            self.char2id[STR_SPECIAL_TOKENS['MOLECULAR_FORMULA_TOKEN']],
            self.char2id[STR_SPECIAL_TOKENS['SMILES_TOKEN']],
            self.char2id[STR_SPECIAL_TOKENS['IUPAC_TOKEN']],
            self.char2id[STR_SPECIAL_TOKENS['INCHI_TOKEN']],
            self.char2id[STR_SPECIAL_TOKENS['SELFIES_TOKEN']],
            self.char2id[STR_SPECIAL_TOKENS['POLYMER_SPG_TOKEN']],
            self.char2id[STR_SPECIAL_TOKENS['FORMULATION_START_TOKEN']],
            self.char2id[STR_SPECIAL_TOKENS['FORMULATION_END_TOKEN']],
        ]

        self.allowed_str = [
            'MOLECULAR_FORMULA',
            'CANONICAL_SMILES',
            'IUPAC_NAME',
            'INCHI',
            'SELFIES',
            'POLYMER SMILES',
            'FORMULATION'
        ]

        self.bag = bag
        self.is_encoder = is_encoder

    def encoder(self, sentence_a, sentence_b):
        raise NotImplementedError

    def process_text(self, text):
        #random length sequences seems to help training
        mod_length = self.mod_length
        avg_length = self.avg_length
        for mol in text:
            # fill up buckets and caches
            sentence_a = mol['sequence_a']

            # 50% NSP
            if self.is_encoder:
                sentence_b = None
                next_sentence_label = None

                if random.random() > self.nsp_probability:
                    sentence_b = random.choice(mol['str_alternatives'])
                    next_sentence_label = 0  # same molecule
                else:
                    while sentence_b is None:
                        rand_idx = random.randint(0, len(self.bag)-1)
                        bag_values = self.bag[rand_idx].values()
                        if sentence_a not in bag_values:
                            options = [self.bag[rand_idx][key] for key in self.bag[rand_idx].keys() if key in self.allowed_str]
                            sentence_b = random.choice(options)
                            next_sentence_label = 1  # different molecule
                        del bag_values
                        gc.collect()

                # Sentence tokenization
                encodings = self.encoder(sentence_a, sentence_b)
                encodings['nsp_label'] = next_sentence_label
                length = encodings['input_ids'].shape[1]
            else:
                encodings = self.encoder(sentence_a)
                if self.decoder_n_alternatives > len(mol['str_alternatives']):
                    alternatives_ls = mol['str_alternatives']
                else:
                    alternatives_ls = random.sample(mol['str_alternatives'], self.decoder_n_alternatives)
                encodings['alternatives'] = [self.encoder(s)['input_ids'].squeeze() for s in alternatives_ls]
                length = max([t.size(0) for t in [encodings['input_ids'].squeeze()] + encodings['alternatives']])

            # length = encodings['input_ids'].shape[1]
            if length > self.min_length and length < mod_length:
                if len(self.bucket0) < self.b0_max:
                    self.bucket0.append(encodings)
                else:
                    self.b0_cache.append(encodings)
            elif length >= mod_length and length < avg_length:
                if len(self.bucket1) < self.b1_max:
                    self.bucket1.append(encodings)
                else:
                    self.b1_cache.append(encodings)
            elif length >= avg_length and length < self.tail:
                if len(self.bucket2) < self.b2_max:
                   self.bucket2.append(encodings)
                else:
                   self.b2_cache.append(encodings)
            elif length >= self.tail and length < self.max_length:
                if len(self.bucket3) < self.b3_max:
                   self.bucket3.append(encodings)
                else:
                   self.b3_cache.append(encodings)

        #print('before Cache size  {} {} {} {}'.format(len(self.b0_cache), len(self.b1_cache), len(self.b2_cache), len(self.b3_cache)))
        #pour cache elements into any open bucket
        if len(self.bucket0) < self.b0_max and len(self.b0_cache) > 0:
            cache_size = len(self.b0_cache)
            max_margin = self.b0_max-len(self.bucket0)
            range0 = min(cache_size, max_margin)
            outbucket0 = [self.bucket0.pop() for item in range(len(self.bucket0))] + [self.b0_cache.pop() for i in range(range0)]
            #self.b0_cache =  collections.deque(self.b0_cache[:self.b0_max-len(bucket0)])
            #print('0 type {}'.format(type(self.b0_cache)))
        else:
            outbucket0 = [self.bucket0.pop() for item in range(len(self.bucket0))]

        if len(self.bucket1) < self.b1_max and len(self.b1_cache) > 0:
            cache_size = len(self.b1_cache)
            max_margin = self.b1_max-len(self.bucket1)
            range1 = min(cache_size, max_margin)
            outbucket1 = [self.bucket1.pop() for item in range(len(self.bucket1))] + [self.b1_cache.pop() for i in range(range1)]
        else:
            outbucket1 = [self.bucket1.pop() for item in range(len(self.bucket1))]

        if len(self.bucket2) < self.b2_max and len(self.b2_cache) > 0:
            cache_size = len(self.b2_cache)
            max_margin = self.b2_max-len(self.bucket2)
            range2 = min(cache_size, max_margin)
            outbucket2 = [self.bucket2.pop() for item in range(len(self.bucket2))] + [self.b2_cache.pop() for i in range(range2)]
        else:
            outbucket2 = [self.bucket2.pop() for item in range(len(self.bucket2))]

        if len(self.bucket3) < self.b3_max and len(self.b3_cache) > 0:
            cache_size = len(self.b3_cache)
            max_margin = self.b3_max-len(self.bucket3)
            range3 = min(cache_size, max_margin)
            outbucket3 = [self.bucket3.pop() for item in range(len(self.bucket3))] + [self.b3_cache.pop() for i in range(range3)]
        else:
            outbucket3 = [self.bucket3.pop() for item in range(len(self.bucket3))]

        return outbucket0, outbucket1, outbucket2, outbucket3

    def mask_tokens( self, inputs, special_tokens_mask= None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.size(), self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            #special_tokens_mask = special_tokens_mask.bool()

        #print(special_tokens_mask.size())
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.size(), 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.size(), 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.char2id.keys()), labels.size(), dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def pack_tensors(self, tokens, sequences, bag):
        raise NotImplementedError
    
    def process(self, text):
        raise NotImplementedError


class TextEncoder4ModelEncoder(TextEncoder):

    def __init__(self, tokenizer_path, max_length=2048, decoder_n_alternatives=2, bag=None):
        super().__init__(tokenizer_path, True, max_length, decoder_n_alternatives, bag)

    def encoder(self, sentence_a, sentence_b):
        return self.tokenizer(sentence_a, sentence_b, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
    
    def pack_tensors(self, encodings):
        array_ids = [item['input_ids'].squeeze() for item in encodings]
        attention_masks = [item['attention_mask'].squeeze() for item in encodings]
        tokens_type_ids = [item['token_type_ids'].squeeze() for item in encodings]
        next_sentence_label = [item['nsp_label'] for item in encodings]

        array =  torch.nn.utils.rnn.pad_sequence(array_ids, batch_first=True, padding_value=self.pad)
        attention_masks =  torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        tokens_type_ids =  torch.nn.utils.rnn.pad_sequence(tokens_type_ids, batch_first=True, padding_value=0)
        lengths = (array!=self.pad).sum(dim=-1)

        # Bert tokenization
        masked_array = []
        masked_labels = []
        # MLM for each sequence instead of full string
        for idx in range(array.shape[0]):
            cut_idx = torch.where(array[idx] == self.eos)[0][0]

            sequence_a = array[idx][:cut_idx]
            sequence_b = array[idx][cut_idx:]

            special_token_mask_a = [list(map(lambda x: 1 if x in self.special_tokens else 0, stuff)) for stuff in [sequence_a]]
            special_token_mask_b = [list(map(lambda x: 1 if x in self.special_tokens else 0, stuff)) for stuff in [sequence_b]]
            
            masked_array_a, masked_labels_a = self.mask_tokens(sequence_a.unsqueeze(0), special_token_mask_a)
            masked_array_b, masked_labels_b = self.mask_tokens(sequence_b.unsqueeze(0), special_token_mask_b)

            masked_array_idx = torch.cat([masked_array_a, masked_array_b], dim=1)
            masked_labels_idx = torch.cat([masked_labels_a, masked_labels_b], dim=1)

            assert array[idx].shape[0] == masked_array_idx.shape[1], 'Error data encoder'
            assert array[idx].shape[0] == masked_labels_idx.shape[1], 'Error data encoder'

            masked_array.append(masked_array_idx)
            masked_labels.append(masked_labels_idx)
            
        masked_array = torch.cat(masked_array, dim=0)
        masked_labels = torch.cat(masked_labels, dim=0)

        return masked_array, masked_labels, array, attention_masks, tokens_type_ids, torch.LongTensor(next_sentence_label), lengths

    def process(self, text):
        masked_arrays = []
        masked_labels = []
        arrays_ids = []
        attn_masks = []
        tokens_type_ids = []
        nsp_labels = []
        lengths = []
        for tokens in self.process_text(text):
            if len(tokens) > 0:
                mask_arr, mask_label, arr_ids, attn_mask, token_type, nsp, lgt = self.pack_tensors(tokens)
                masked_arrays.append(mask_arr)
                masked_labels.append(mask_label)
                arrays_ids.append(arr_ids)
                attn_masks.append(attn_mask)
                tokens_type_ids.append(token_type)
                nsp_labels.append(nsp)
                lengths.append(lgt)
        return masked_arrays, masked_labels, arrays_ids, attn_masks, tokens_type_ids, nsp_labels, lengths


class TextEncoder4ModelDecoder(TextEncoder):

    def __init__(self, tokenizer_path, max_length=2048, decoder_n_alternatives=2, bag=None):
        super().__init__(tokenizer_path, False, max_length, decoder_n_alternatives, bag)

    def encoder(self, sentence_a):
        return self.tokenizer(sentence_a, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
    
    def pack_tensors(self, encodings):
        # Sentence tokenization
        array_ids = [item['input_ids'].squeeze() for item in encodings]
        attention_masks = [item['attention_mask'].squeeze() for item in encodings]
        alternatives = [item['alternatives'] for item in encodings]

        array =  torch.nn.utils.rnn.pad_sequence(array_ids, batch_first=True, padding_value=self.pad)
        attention_masks =  torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        lengths = (array!=self.pad).sum(dim=-1)

        # Bert tokenization
        special_token_mask = [list(map(lambda x: 1 if x in self.special_tokens else 0, stuff)) for stuff in array.tolist()]
        masked_array, masked_labels = self.mask_tokens(array, special_token_mask)

        return masked_array, masked_labels, array_ids, attention_masks, alternatives, lengths
    
    def process(self, text):
        masked_arrays = []
        masked_labels = []
        arrays_ids = []
        attn_masks = []
        alternatives = []
        lengths = []
        for tokens in self.process_text(text):
            if len(tokens) > 0:
                mask_arr, mask_label, arr_ids, attn_mask, alt, lgt = self.pack_tensors(tokens)
                masked_arrays.append(mask_arr)
                masked_labels.append(mask_label)
                arrays_ids.append(arr_ids)
                attn_masks.append(attn_mask)
                alternatives.append(alt)
                lengths.append(lgt)
        return masked_arrays, masked_labels, arrays_ids, attn_masks, alternatives, lengths
    


def main():
    import pandas as pd
    df = pd.read_csv('../tokenization/polymer_pretrain_v1.csv')
    encoder = TextEncoder4ModelDecoder('str_bamba_tokenizer.json', 4096, bag=df['POLYMER SMILES'].to_list())

    batch = encoder.process([
        {'sentence_a': '<smiles>CCCCccccccccO', 'other_representations': ['<iupac>N,N-diethyl-3,3-diphenylprop-2-en-1-amine']},
        {'sentence_a': 'InChI=1S/C19H23N/c1-3-20(4-2)16-15-19(17-11-7-5-8', 'other_representations': ['<smiles>CCCCccccccccO']},
        {'sentence_a': '<iupac>N,N', 'other_representations': ['<smiles>CCCC']},
        {'sentence_a': '<iupac>amine', 'other_representations': ['<smiles>OcccN']},
        {'sentence_a': '<selfies>[C][C][N][Branch1][Ring1][C][C][C][C][=C][Branch1][=Branch2][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=C][C][=C][C][=C][Ring1][=Branch1]', 'other_representations': ['<smiles>CCCCccccccccO']},
    ])
    print(batch)

    # print(encoder.process_text([{'text': 'CCCCccccccccO'}, {'text': 'CCCCccccccccNNN'}, {'text': 'CCCCccccccccOCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC'}]))


if __name__ == '__main__':
    main()
