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
import numpy as np

# Standard library
import regex as re
import random
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


class MolEncoder(nn.Module):

    def __init__(self, config, n_vocab):
        super().__init__()
        
        self.mamba = MixerModel(
            d_model=config.n_embd, 
            n_layer=config.n_layer,
            ssm_cfg=dict(
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand_factor,
                dt_rank=config.dt_rank,
                dt_min=config.dt_min,
                dt_max=config.dt_max,
                dt_init=config.dt_init,
                dt_scale=config.dt_scale,
                dt_init_floor=config.dt_init_floor,
                conv_bias=bool(config.conv_bias),
                bias=bool(config.bias),
            ),
            vocab_size=n_vocab,
            rms_norm=False,
            fused_add_norm=False,
        )

        # classification
        self.lang_model = LangLayer(config.n_embd, n_vocab)

    def forward(self, idx, mask=None, inference=False):
        if not inference:
            x = self.mamba(idx)
            logits = self.lang_model(x)
            return logits
        else:
            x = self.mamba(idx)
        
            # mean pooling
            token_embeddings = x
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            true_set = sum_embeddings / sum_mask

            return true_set, token_embeddings
        

class MoLDecoder(nn.Module):

    def __init__(self, n_vocab, max_len, n_embd, n_gpu=None):
        super(MoLDecoder, self).__init__()

        self.max_len = max_len
        self.n_embd = n_embd
        self.n_gpu = n_gpu
        self.autoencoder = AutoEncoderLayer(n_embd*max_len, n_embd)
        self.lm_head = LangLayer(n_embd, n_vocab)

    def forward(self, token_embeddings):
        pred_set = self.autoencoder.encoder(token_embeddings) # (N, D)
        pred_cte = self.autoencoder.decoder(pred_set) # (N, L*D)
        pred_ids = self.lm_head(pred_cte.view(-1, self.max_len, self.n_embd))
        return pred_set, pred_ids


class Smi_ssed(nn.Module):
    """granite.materials.smi-ssed 336M Parameters"""

    def __init__(self, config, vocab):
        super(Smi_ssed, self).__init__()
        
        self.config = config

        self.padding_idx = 2
        self.is_cuda_available = torch.cuda.is_available()
        n_vocab = len(vocab.keys())
        print(n_vocab, config.n_embd)

        self.encoder = MolEncoder(config, n_vocab)
        self.decoder = MoLDecoder(n_vocab, config.max_len, config.n_embd)

        self._set_seed(config.seed)
        print('Vocab size:', n_vocab)
        print(f'[PRE-TRAINING MODE - {str(self)}]')

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

    def __str__(self):
        return 'smi-ssed'

