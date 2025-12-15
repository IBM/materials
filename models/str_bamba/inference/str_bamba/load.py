from .bamba_config import BambaEncoderDecoderConfig
from .bamba import BambaConfig, BambaEncoderDecoder
from .tokenizer.str_tokenizer import load_tokenizer
import torch
import numpy as np
import random
import json
import os


def load_strbamba(ckpt_filename,
			   base_folder='./str_bamba', 
			   config_filename='config_encoder-decoder_436M.json',  
			   tokenizer_filename='str_bamba_tokenizer.json', 
			   eval_model=True, 
			   device='cuda:0', 
			   dtype=torch.float32
			   ):
	# load config
	with open(os.path.join(base_folder, f'config/{config_filename}')) as json_data:
		config_json = json.load(json_data)
	bamba_config = BambaEncoderDecoderConfig(
        encoder_config=BambaConfig(**config_json['encoder_config']),
        decoder_config=BambaConfig(**config_json['decoder_config']),
        tie_word_embeddings=config_json['tie_word_embeddings'],
        seed=config_json['seed']
    )

	# load tokenizer
	tokenizer = load_tokenizer(os.path.join(base_folder, f'tokenizer/{tokenizer_filename}'))

	# load model
	model = BambaEncoderDecoder(bamba_config, tokenizer, device=device, dtype=dtype)

	# load weights
	ckpt_dict = torch.load(
		os.path.join(base_folder, f'checkpoints/{ckpt_filename}'), 
		map_location=device, 
		weights_only=False
	)
	model.load_state_dict(ckpt_dict['module'])

	# load RNG states each time the model and states are loaded from checkpoint
	if 'rng' in ckpt_dict:
		rng = ckpt_dict['rng']
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

	if eval_model:
		return model.eval()
	return model