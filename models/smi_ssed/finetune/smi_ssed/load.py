PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# Tokenizer
from transformers import BertTokenizer

# Mamba
from mamba_ssm.models.mixer_seq_simple import MixerModel

# Data
import pandas as pd
import numpy as np

# Standard library
import regex as re
import random
import os
import gc
from tqdm import tqdm
tqdm.pandas()


class MolTranBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file: str = '',
                 do_lower_case=False,
                 unk_token='<pad>',
                 sep_token='<eos>',
                 pad_token='<pad>',
                 cls_token='<bos>',
                 mask_token='<mask>',
                 **kwargs):
        super().__init__(vocab_file,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         pad_token=pad_token,
                         cls_token=cls_token,
                         mask_token=mask_token,
                         **kwargs)

        self.regex_tokenizer = re.compile(PATTERN)
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None
        with open(vocab_file) as f:
            self.padding_idx = f.readlines().index(pad_token+'\n')

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens
    
    def convert_idx_to_tokens(self, idx_tensor):
        tokens = [self.convert_ids_to_tokens(idx) for idx in idx_tensor.tolist()]
        return tokens

    def convert_tokens_to_string(self, tokens):
        stopwords = ['<bos>', '<eos>']
        clean_tokens = [word for word in tokens if word not in stopwords]
        out_string = ''.join(clean_tokens)
        return out_string
    
    def get_padding_idx(self):
        return self.padding_idx
    
    def idx_to_smiles(self, torch_model, idx):
        '''Convert tokens idx back to SMILES text'''
        rev_tokens = torch_model.tokenizer.convert_idx_to_tokens(idx)
        flat_list_tokens = [item for sublist in rev_tokens for item in sublist]
        decoded_smiles = torch_model.tokenizer.convert_tokens_to_string(flat_list_tokens)
        return decoded_smiles


class AutoEncoderLayer(nn.Module):

    def __init__(self, feature_size, latent_size):
        super().__init__()
        self.encoder = self.Encoder(feature_size, latent_size)
        self.decoder = self.Decoder(feature_size, latent_size)
    
    class Encoder(nn.Module):

        def __init__(self, feature_size, latent_size):
            super().__init__()
            self.is_cuda_available = torch.cuda.is_available()
            self.fc1 = nn.Linear(feature_size, latent_size)
            self.ln_f = nn.LayerNorm(latent_size)
            self.lat = nn.Linear(latent_size, latent_size, bias=False)
        
        def forward(self, x):
            if self.is_cuda_available:
                self.fc1.cuda()
                self.ln_f.cuda()
                self.lat.cuda()
                x = x.cuda()
            x = F.gelu(self.fc1(x))
            x = self.ln_f(x)
            x = self.lat(x)
            return x # -> (N, D)
    
    class Decoder(nn.Module):

        def __init__(self, feature_size, latent_size):
            super().__init__()
            self.is_cuda_available = torch.cuda.is_available()
            self.fc1 = nn.Linear(latent_size, latent_size)
            self.ln_f = nn.LayerNorm(latent_size)
            self.rec = nn.Linear(latent_size, feature_size, bias=False)
        
        def forward(self, x):
            if self.is_cuda_available:
                self.fc1.cuda()
                self.ln_f.cuda()
                self.rec.cuda()
                x = x.cuda()
            x = F.gelu(self.fc1(x))
            x = self.ln_f(x)
            x = self.rec(x)
            return x # -> (N, L*D)


class LangLayer(nn.Module):
    def __init__(self, n_embd, n_vocab):
        super().__init__()
        self.is_cuda_available = torch.cuda.is_available()
        self.embed = nn.Linear(n_embd, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_vocab, bias=False)
    def forward(self, tensor):
        if self.is_cuda_available:
            self.embed.cuda()
            self.ln_f.cuda()
            self.head.cuda()
            tensor = tensor.cuda()
        tensor = self.embed(tensor)
        tensor = F.gelu(tensor)
        tensor = self.ln_f(tensor)
        tensor = self.head(tensor)
        return tensor
    

class Net(nn.Module):
    
    def __init__(self, smiles_embed_dim, n_output=1, dropout=0.2):
        super().__init__()
        self.desc_skip_connection = True
        self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.GELU()
        self.final = nn.Linear(smiles_embed_dim, n_output)

    def forward(self, smiles_emb, multitask=False):
        x_out = self.fc1(smiles_emb)
        x_out = self.dropout1(x_out)
        x_out = self.relu1(x_out)

        if self.desc_skip_connection is True:
            x_out = x_out + smiles_emb

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.relu2(z)
        if self.desc_skip_connection is True:
            z = self.final(z + x_out)
        else:
            z = self.final(z)

        if multitask:
            return F.sigmoid(z)
        return z


class MolEncoder(nn.Module):

    def __init__(self, config, n_vocab):
        super().__init__()
        
        self.config = config
        self.mamba = MixerModel(
            d_model=config['n_embd'], 
            n_layer=config['n_layer'],
            ssm_cfg=dict(
                d_state=config['d_state'],
                d_conv=config['d_conv'],
                expand=config['expand_factor'],
                dt_rank=config['dt_rank'],
                dt_min=config['dt_min'],
                dt_max=config['dt_max'],
                dt_init=config['dt_init'],
                dt_scale=config['dt_scale'],
                dt_init_floor=config['dt_init_floor'],
                conv_bias=bool(config['conv_bias']),
                bias=False,
            ),
            vocab_size=n_vocab,
            rms_norm=False,
            fused_add_norm=False,
        )

        # classification
        self.lang_model = LangLayer(config['n_embd'], n_vocab)
        

class MoLDecoder(nn.Module):

    def __init__(self, n_vocab, max_len, n_embd, n_gpu=None):
        super(MoLDecoder, self).__init__()

        self.max_len = max_len
        self.n_embd = n_embd
        self.n_gpu = n_gpu
        self.autoencoder = AutoEncoderLayer(n_embd*max_len, n_embd)
        self.lm_head = LangLayer(n_embd, n_vocab)


class Smi_ssed(nn.Module):
    """granite.materials.smi-ssed 336M Parameters"""

    def __init__(self, tokenizer, config=None):
        super(Smi_ssed, self).__init__()
        
        # configuration
        self.config = config
        self.tokenizer = tokenizer
        self.padding_idx = tokenizer.get_padding_idx()
        self.n_vocab = len(self.tokenizer.vocab)
        self.is_cuda_available = torch.cuda.is_available()

        # instantiate modules
        if self.config:
            self.encoder = MolEncoder(self.config, self.n_vocab)
            self.decoder = MoLDecoder(self.n_vocab, self.config['max_len'], self.config['n_embd'])
            self.net = Net(self.config['n_embd'], n_output=self.config['n_output'], dropout=self.config['d_dropout'])

    def load_checkpoint(self, ckpt_path, n_output):
        # load checkpoint file
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

        # load hyparameters
        self.config = checkpoint['hparams']
        self.max_len = self.config['max_len']
        self.n_embd = self.config['n_embd']
        self._set_seed(self.config['seed'])

        # instantiate modules
        self.encoder = MolEncoder(self.config, self.n_vocab)
        self.decoder = MoLDecoder(self.n_vocab, self.max_len, self.n_embd)
        self.net = Net(self.n_embd, n_output=self.config['n_output'] if 'n_output' in self.config else n_output, dropout=self.config['dropout'])

        # load weights
        self.load_state_dict(checkpoint['MODEL_STATE'], strict=False)

        # load RNG states each time the model and states are loaded from checkpoint
        if 'rng' in self.config:
            rng = self.config['rng']
            for key, value in rng.items():
                if key =='torch_state':
                    torch.set_rng_state(value.cpu())
                elif key =='cuda_state':
                    torch.cuda.set_rng_state(value.cpu())
                elif key =='numpy_state':
                    np.random.set_state(value)
                elif key =='python_state':
                    random.setstate(value)
                else:
                    print('unrecognized state')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_seed(self, value):
        print('Random Seed:', value)
        random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed(value)
        torch.cuda.manual_seed_all(value)
        np.random.seed(value)
        cudnn.deterministic = True
        cudnn.benchmark = False
            
    def tokenize(self, smiles):
        """Tokenize a string into tokens."""
        if isinstance(smiles, str):
            batch = [smiles]
        else:
            batch = smiles
        
        tokens = self.tokenizer(
            batch, 
            padding=True,
            truncation=True, 
            add_special_tokens=True, 
            return_tensors="pt",
            max_length=self.max_len,
        )
        
        idx = tokens['input_ids'].clone().detach()
        mask = tokens['attention_mask'].clone().detach()

        if self.is_cuda_available:
            return idx.cuda(), mask.cuda()
        
        return idx, mask

    def extract_embeddings(self, smiles):
        """Extract token and SMILES embeddings."""
        # evaluation mode
        if self.is_cuda_available:
            self.encoder.cuda()
            self.decoder.cuda()
        
        # tokenizer
        idx, mask = self.tokenize(smiles)
        
        # encoder forward
        x = self.encoder.mamba(idx)

        # add padding
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        mask_embeddings = (token_embeddings * input_mask_expanded)
        token_embeddings = F.pad(mask_embeddings, pad=(0, 0, 0, self.config['max_len'] - mask_embeddings.shape[1]), value=0)

        # aggregate token embeddings (similar to mean pooling)
        # CAUTION: use the embeddings from the autoencoder.
        smiles_embeddings = self.decoder.autoencoder.encoder(token_embeddings.view(-1, self.max_len*self.n_embd))

        return smiles_embeddings

    def __str__(self):
        return 'smi-ssed'


def load_smi_ssed(folder="./smi_ssed", 
              ckpt_filename="smi-ssed_130.pt",
              vocab_filename="bert_vocab_curated.txt",
              n_output=1
              ):
    tokenizer = MolTranBertTokenizer(os.path.join(folder, vocab_filename))
    model = Smi_ssed(tokenizer)
    model.load_checkpoint(os.path.join(folder, ckpt_filename), n_output)
    print('Vocab size:', len(tokenizer.vocab))
    print(f'[FINETUNE MODE - {str(model)}]')
    return model