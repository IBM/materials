PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# Transformers
from fast_transformers.attention import AttentionLayer
from fast_transformers.events import EventDispatcher, QKVEvent
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer
from fast_transformers.builders.base import BaseBuilder
from fast_transformers.builders.transformer_builders import BaseTransformerEncoderBuilder
from fast_transformers.builders.attention_builders import AttentionBuilder
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from fast_transformers.masking import LengthMask 

from transformers import BertTokenizer

# Data
import numpy as np

# Standard library
from functools import partial
import regex as re
import random


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

## Transformer layers

class RotaryEmbedding(torch.nn.Module):
    
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = 0 
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            
            self.cos_cached = emb.cos()[None,:, None, :]
            self.sin_cached = emb.sin()[None,:, None, :]
                
        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1) # dim=-1 triggers a bug in earlier torch versions

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotateAttentionLayer(AttentionLayer):
    """Rotate attention layer inherits from fast_transformer attention layer. 
        The only thing added is an Embedding encoding, for more information
        on the attention layer see the fast_transformers code
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, event_dispatcher=""):
        super(RotateAttentionLayer, self).__init__(attention,d_model, n_heads, d_keys=d_keys,
                 d_values=d_values, event_dispatcher=event_dispatcher)

        self.rotaryemb = RotaryEmbedding(d_keys)
        print('Using Rotation Embedding')

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """
        Using the same frame work as the fast_Transformers attention layer
        but injecting rotary information to the queries and the keys
        after the keys and queries are projected. 
        In the argument description we make use of the following sizes
            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'
        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        cos, sin = self.rotaryemb(queries)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        values = self.value_projection(values).view(N, S, H, -1)
        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))


        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)

class RotateEncoderBuilder(BaseTransformerEncoderBuilder):
    """Build a batch transformer encoder with Relative Rotary embeddings
    for training or processing of sequences all elements at a time.
    Example usage:
        builder = RotateEncoderBuilder()
        builder.n_layers = 12
        builder.n_heads = 8
        builder.feed_forward_dimensions = 1024
        builder.query_dimensions = 64
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.attention_dropout = 0.1
        builder.attention_type = "linear"
        transformer = builder.get()
    """
    def _get_attention_builder(self):
        """Return an instance of the appropriate attention builder."""
        return AttentionBuilder()

    def _get_attention_layer_class(self):
        """Return the class for the layer that projects queries keys and
        values."""
        return RotateAttentionLayer

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        return TransformerEncoder

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        return TransformerEncoderLayer


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


class MoLEncoder(nn.Module):

    def __init__(self, config, n_vocab):
        super(MoLEncoder, self).__init__()

        # embeddings
        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
        self.drop = nn.Dropout(config.d_dropout)

        # transformer
        builder = RotateEncoderBuilder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd//config.n_head,
            value_dimensions=config.n_embd//config.n_head,
            feed_forward_dimensions=None,
            attention_type='linear',
            # unless we do deterministic_eval here, we will have random outputs
            feature_map=partial(GeneralizedRandomFeatures, 
                                n_dims=config.num_feats, 
                                deterministic_eval=False),
            activation='gelu'
        )
        self.blocks = builder.get()

        # classification
        self.lang_model = LangLayer(config.n_embd, n_vocab)

    def forward(self, idx, mask=None, inference=False):
        if not inference:
            x = self.tok_emb(idx) # each index maps to a (learnable) vector
            x = self.drop(x)

            #masking of the length of the inputs its handled in the Masked language part of the code
            #do not attempt to handle it in the forward of the transformer
            x = self.blocks(x)
            logits = self.lang_model(x)

            return logits
        else:
            x = self.tok_emb(idx) # each index maps to a (learnable) vector
            x = self.drop(x)

            #masking of the length of the inputs its handled in the Masked language part of the code
            #do not attempt to handle it in the forward of the transformer
            x = self.blocks(x, length_mask=LengthMask(mask.sum(-1), max_len=idx.shape[1]))
            
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
        self.lang_model = LangLayer(n_embd, n_vocab)

    def forward(self, token_embeddings):
        pred_set = self.autoencoder.encoder(token_embeddings) # (N, D)
        pred_cte = self.autoencoder.decoder(pred_set) # (N, L*D)
        pred_ids = self.lang_model(pred_cte.view(-1, self.max_len, self.n_embd))
        return pred_set, pred_ids


class Smi_ted(nn.Module):
    """materials.smi-ted-Large 738M Parameters"""

    def __init__(self, config, vocab):
        super(Smi_ted, self).__init__()
        
        self.config = config
        self.padding_idx = 2
        self.is_cuda_available = torch.cuda.is_available()
        n_vocab = len(vocab.keys())
        print(n_vocab, config.n_embd)

        self.encoder = MoLEncoder(config, n_vocab)
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
        return 'smi-ted-Large'