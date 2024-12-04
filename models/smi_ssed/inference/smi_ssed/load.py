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

# Chemistry
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
PandasTools.RenderImagesInAllDataFrames(True)

# Standard library
import regex as re
import random
import os
import gc
from tqdm import tqdm
tqdm.pandas()


# function to canonicalize SMILES
def normalize_smiles(smi, canonical=True, isomeric=False):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized


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
                bias=bool(config['bias']),
            ),
            vocab_size=n_vocab,
            rms_norm=False,
            fused_add_norm=False,
        )

        # classification
        self.lang_model = LangLayer(config['n_embd'], n_vocab)

    def forward(self, idx, mask):
        x = self.mamba(idx)

        # add padding
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        mask_embeddings = (token_embeddings * input_mask_expanded)
        token_embeddings = F.pad(mask_embeddings, pad=(0, 0, 0, self.config['max_len'] - mask_embeddings.shape[1]), value=0)

        return token_embeddings
        

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

    def load_checkpoint(self, ckpt_path):
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
        self.net = Net(self.n_embd, n_output=self.config['n_output'] if 'n_output' in self.config else 1, dropout=self.config['d_dropout'])

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

    def forward(self, smiles, batch_size=100):
        return self.decode(self.encode(smiles, batch_size=batch_size, return_torch=True))
            
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
    
    def extract_all(self, smiles):
        """Extract all elements from each part of smi-ssed. Be careful."""
        # evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        if self.is_cuda_available:
            self.encoder.cuda()
            self.decoder.cuda()

        # handle single str or a list of str
        smiles = pd.Series(smiles) if isinstance(smiles, str) else pd.Series(list(smiles))

        # SMILES normalization
        smiles = smiles.apply(normalize_smiles)
        null_idx = smiles[smiles.isnull()].index.to_list()  # keep track of SMILES that cannot normalize
        smiles = smiles.dropna()
        
        # tokenizer
        idx, mask = self.tokenize(smiles)
        
        ###########
        # Encoder #
        ###########
        # encoder forward
        x = self.encoder.mamba(idx)

        # mean pooling
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        true_set = sum_embeddings / sum_mask # DO NOT USE THIS FOR DOWNSTREAM TASKS, USE `pred_set` INSTEAD

        # add padding
        mask_embeddings = (token_embeddings * input_mask_expanded)
        token_embeddings = F.pad(mask_embeddings, pad=(0, 0, 0, self.max_len - mask_embeddings.shape[1]), value=0)
        idx = F.pad(idx, pad=(0, self.max_len - idx.shape[1], 0, 0), value=2)

        true_ids = idx
        true_cte = token_embeddings
        true_cte = true_cte.view(-1, self.max_len*self.n_embd)

        ###########
        # Decoder #
        ###########
        # CTE autoencoder
        pred_set = self.decoder.autoencoder.encoder(true_cte)
        pred_cte = self.decoder.autoencoder.decoder(pred_set)

        # reconstruct tokens
        pred_ids = self.decoder.lm_head(pred_cte.view(-1, self.max_len, self.n_embd))
        pred_ids = torch.argmax(pred_ids, axis=-1)

        # replacing null SMILES with NaN values
        for idx in null_idx:
            true_ids = true_ids.tolist()
            pred_ids = pred_ids.tolist()
            true_cte = true_cte.tolist()
            pred_cte = pred_cte.tolist()
            true_set = true_set.tolist()
            pred_set = pred_set.tolist()

            true_ids.insert(idx, np.array([np.nan]*self.config['max_len']))
            pred_ids.insert(idx, np.array([np.nan]*self.config['max_len']))
            true_cte.insert(idx, np.array([np.nan] * (self.config['max_len']*self.config['n_embd'])))
            pred_cte.insert(idx, np.array([np.nan] * (self.config['max_len']*self.config['n_embd'])))
            true_set.insert(idx, np.array([np.nan]*self.config['n_embd']))
            pred_set.insert(idx, np.array([np.nan]*self.config['n_embd']))
        
        if len(null_idx) > 0:
            true_ids = torch.tensor(true_ids)
            pred_ids = torch.tensor(pred_ids)
            true_cte = torch.tensor(true_cte)
            pred_cte = torch.tensor(pred_cte)
            true_set = torch.tensor(true_set)
            pred_set = torch.tensor(pred_set)
        
        return ((true_ids, pred_ids), # tokens
               (true_cte, pred_cte),  # token embeddings
               (true_set, pred_set))  # smiles embeddings

    def extract_embeddings(self, smiles):
        """Extract token and SMILES embeddings."""
        # evaluation mode
        self.encoder.eval()
        if self.is_cuda_available:
            self.encoder.cuda()
        
        # tokenizer
        idx, mask = self.tokenize(smiles)
        
        # encoder forward
        token_embeddings = self.encoder(idx, mask)

        # aggregate token embeddings (similar to mean pooling)
        # CAUTION: use the embeddings from the autoencoder.
        smiles_embeddings = self.decoder.autoencoder.encoder(token_embeddings.view(-1, self.max_len*self.n_embd))

        # add padding
        idx = F.pad(idx, pad=(0, self.max_len - idx.shape[1], 0, 0), value=self.padding_idx)

        return idx, token_embeddings, smiles_embeddings
    
    def encode(self, smiles, useCuda=False, batch_size=100, return_torch=False):
        """Extract efficiently SMILES embeddings per batches."""
        # TODO: remove useCuda argument

        # handle single str or a list of str
        smiles = pd.Series(smiles) if isinstance(smiles, str) else pd.Series(list(smiles))

        # SMILES normalization
        smiles = smiles.apply(normalize_smiles)
        null_idx = smiles[smiles.isnull()].index.to_list()  # keep track of SMILES that cannot normalize
        smiles = smiles.dropna()

        # process in batches
        n_split = smiles.shape[0] // batch_size if smiles.shape[0] >= batch_size else smiles.shape[0]
        embeddings = [
            self.extract_embeddings(list(batch))[2].cpu().detach().numpy() 
                for batch in tqdm(np.array_split(smiles, n_split))
        ]
        flat_list = [item for sublist in embeddings for item in sublist]

        # clear GPU memory
        if self.is_cuda_available:
            torch.cuda.empty_cache()
            gc.collect()

        # replacing null SMILES with NaN values
        for idx in null_idx:
            flat_list.insert(idx, np.array([np.nan]*self.config['n_embd']))
        flat_list = np.asarray(flat_list)

        if return_torch:
            return torch.tensor(flat_list)
        return pd.DataFrame(flat_list)

    def embd_to_smiles(self, embds):
        # evaluation mode
        self.decoder.eval()
        if self.is_cuda_available:
            self.decoder.cuda()

        # reconstruct token embeddings
        pred_token_embds = self.decoder.autoencoder.decoder(embds)

        # reconstruct tokens
        pred_idx = self.decoder.lm_head(pred_token_embds.view(-1, self.max_len, self.n_embd))
        pred_idx = torch.argmax(pred_idx, axis=-1).cpu().detach().numpy()

        # convert idx to tokens
        pred_smiles = []
        for i in range(pred_idx.shape[0]):
            idx = pred_idx[i]
            smiles = self.tokenizer.idx_to_smiles(self, idx)
            smiles = smiles.replace('<bos>', '') # begin token
            smiles = smiles.replace('<eos>', '') # end token
            smiles = smiles.replace('<pad>', '') # pad token
            pred_smiles.append(smiles)
        return pred_smiles
    
    def decode(self, smiles_embeddings, batch_size=100):
        """Decode SMILES embeddings back to SMILES."""
        # process in batches
        n_split = smiles_embeddings.shape[0] // batch_size if smiles_embeddings.shape[0] >= batch_size else smiles_embeddings.shape[0]
        embeddings = [
            self.embd_to_smiles(batch) for batch in tqdm(np.array_split(smiles_embeddings, n_split))
        ]
        pred_smiles = [item for sublist in embeddings for item in sublist]

        # clear GPU memory
        if self.is_cuda_available:
            torch.cuda.empty_cache()
            gc.collect()

        return pred_smiles

    def __str__(self):
        return 'smi-ssed'


def load_smi_ssed(folder="./smi_ssed", 
              ckpt_filename="smi-ssed_130.pt",
              vocab_filename="bert_vocab_curated.txt"
              ):
    tokenizer = MolTranBertTokenizer(os.path.join(folder, vocab_filename))
    model = Smi_ssed(tokenizer)
    model.load_checkpoint(os.path.join(folder, ckpt_filename))
    model.eval()
    print('Vocab size:', len(tokenizer.vocab))
    print(f'[INFERENCE MODE - {str(model)}]')
    return model