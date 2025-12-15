from typing import List

from tokenizers import NormalizedString, PreTokenizedString
from tokenizers.pre_tokenizers import PreTokenizer
from transformers import PreTrainedTokenizerFast

import re


ATOM_REGEX_PATTERN = r"""(<(.*?)>|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
FORMULATION_REGEX_PATTERN = r"""(<(.*?)>|[-+]?\d*\.\d+|[-+]?\d+\.?\d*[eE][-+]?\d+|[-+]?\d+|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
NUMBER_REGEX_PATTERN = r"""(\d{2}|\d[a-zA-Z]\d|\d[a-zA-Z]|[a-zA-Z]\d+|\(|\))"""
# NUMBER_REGEX_PATTERN = r"""((?<!\d)\d{2}(?!\d)|\d[a-zA-Z]\d|\d[a-zA-Z]|[a-zA-Z]\d)"""
# NUMBER_REGEX_PATTERN = r"""(\d[a-zA-Z]\d|\d[a-zA-Z]|[a-zA-Z]\d|\b\d{2}\b)"""
SPECIAL_REGEX_PATTERN = r"""<(.*?)>"""


class MoleculePreTokenizer:

    def molecule_based_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        splits = []
        if str(normalized_string).startswith(('<smiles>', '<selfies>', '<polymer_spg>')):
            for m in re.finditer(ATOM_REGEX_PATTERN, str(normalized_string)):
                start = m.start(0)
                stop = m.end(0)
                if start == 0:  # remove special tokens
                    continue
                splits.append(normalized_string[start:stop])
        elif str(normalized_string).startswith('<formulation_start>'):
            for m in re.finditer(FORMULATION_REGEX_PATTERN, str(normalized_string)):
                start = m.start(0)
                stop = m.end(0)
                if start == 0 or stop == len(str(normalized_string)):  # remove special tokens
                    continue
                splits.append(normalized_string[start:stop])
        elif str(normalized_string).startswith(('<formula>', '<inchi>')):
            for m in re.finditer(NUMBER_REGEX_PATTERN, str(normalized_string)):
                start = m.start(0)
                stop = m.end(0)
                if start == 0:  # remove special tokens
                    continue
                splits.append(normalized_string[start:stop])
        else:
            last = 0
            for m in re.finditer(SPECIAL_REGEX_PATTERN, str(normalized_string)):  # remove special tokens
                start = m.start(0)
                stop = m.end(0)
                # splits.append(normalized_string[start:stop])
                last = stop
            splits.append(normalized_string[last:])

        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.molecule_based_split)


class MultiMolTranBertTokenizer(PreTrainedTokenizerFast):
        def __init__(self, vocab_file: str = '',
                    do_lower_case=False,
                    cls_token='<bos>',
                    eos_token='<sep>',
                    pad_token='<pad>',
                    unk_token='<unk>',
                    mask_token='<mask>',
                    **kwargs):

            super().__init__(
                tokenizer_file=vocab_file,
                bos_token=cls_token,
                eos_token=eos_token,
                pad_token=pad_token,
                unk_token=unk_token,
                mask_token=mask_token
            )

        def get_padding_idx(self):
            return 2

        def convert_idx_to_tokens(self, idx_tensor):
            tokens = [self.convert_ids_to_tokens(idx) for idx in idx_tensor.tolist()]
            return tokens

        def convert_tokens_to_string(self, tokens):
            stopwords = ['<bos>', '<eos>']
            clean_tokens = [word for word in tokens if word not in stopwords]
            out_string = ''.join(clean_tokens)
            return out_string

        def idx_to_smiles(self, torch_model, idx):
            '''Convert tokens idx back to SMILES text'''
            rev_tokens = torch_model.tokenizer.convert_idx_to_tokens(idx)
            flat_list_tokens = [item for sublist in rev_tokens for item in sublist]
            decoded_smiles = torch_model.tokenizer.convert_tokens_to_string(flat_list_tokens)
            return decoded_smiles


def load_tokenizer(vocab_file, **kwargs):
    tokenizer = MultiMolTranBertTokenizer(vocab_file, **kwargs)
    tokenizer.backend_tokenizer.pre_tokenizer = PreTokenizer.custom(MoleculePreTokenizer())
    return tokenizer