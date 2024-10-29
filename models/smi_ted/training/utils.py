# Deep learning
import torch

# Data
from pubchem_encoder import Encoder
from datasets import load_dataset

# Standard library
import os
import getpass
import glob


class MoleculeModule:
    def __init__(self,  max_len, dataset, data_path):
        super().__init__()
        self.dataset = dataset
        self.data_path = data_path
        self.text_encoder = Encoder(max_len)

    def prepare_data(self):
        pass

    def get_vocab(self):
        #using home made tokenizer, should look into existing tokenizer
        return self.text_encoder.char2id

    def get_cache(self):
        return self.cache_files
    
    def setup(self, stage=None):
        #using huggingface dataloader
        # create cache in tmp directory of locale mabchine under the current users name to prevent locking issues
        pubchem_path = {'train': self.data_path}
        if 'canonical' in pubchem_path['train'].lower():
            pubchem_script = './pubchem_canon_script.py'
        else:
            pubchem_script = './pubchem_script.py'
        zinc_path = './data/ZINC'
        global dataset_dict
        if 'ZINC' in self.dataset or 'zinc' in self.dataset:
            zinc_files = [f for f in glob.glob(os.path.join(zinc_path,'*.smi'))]
            for zfile in zinc_files:
                print(zfile)
            self.dataset = {'train': zinc_files}
            dataset_dict = load_dataset('./zinc_script.py', data_files=self.dataset, cache_dir=os.path.join('/tmp',getpass.getuser(), 'zinc'),split='train', trust_remote_code=True)

        elif 'pubchem' in self.dataset:
            dataset_dict =  load_dataset(pubchem_script, data_files=pubchem_path, cache_dir=os.path.join('/tmp',getpass.getuser(), 'pubchem'), split='train')
        elif 'both' in self.dataset or 'Both' in self.dataset or 'BOTH' in self.dataset:
            dataset_dict_pubchem =  load_dataset(pubchem_script, data_files=pubchem_path, cache_dir=os.path.join('/tmp',getpass.getuser(), 'pubchem'),split='train', trust_remote_code=True)
            zinc_files = [f for f in glob.glob(os.path.join(zinc_path,'*.smi'))]
            for zfile in zinc_files:
                print(zfile)
            self.dataset = {'train': zinc_files}
            dataset_dict_zinc =  load_dataset('./zinc_script.py', data_files=self.dataset, cache_dir=os.path.join('/tmp',getpass.getuser(), 'zinc'),split='train', trust_remote_code=True)
            dataset_dict = concatenate_datasets([dataset_dict_zinc, dataset_dict_pubchem])
        self.pubchem= dataset_dict
        print(dataset_dict.cache_files)
        self.cache_files = []

        for cache in dataset_dict.cache_files:
            tmp = '/'.join(cache['filename'].split('/')[:4])
            self.cache_files.append(tmp)


def get_optim_groups(module):
    # setup optimizer
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in module.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in module.named_parameters()}
    
    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    return optim_groups