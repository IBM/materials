from str_bamba.load import load_strbamba
import torch

torch.random.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
	# load model
	model = load_strbamba(
		ckpt_filename='STR-Bamba_Decoder_smiles_0.pt', 
		device='cuda',
		dtype=torch.float16
	)

	# set encoder decoder inputs
	encoder_input = '<smiles>*CC(*)c1ccccc1C(=O)OCCCCCC'
	decoder_input = '<smiles>'
	decoder_target = '<smiles>*CC(*)c1ccccc1C(=O)OCCCCCC'

	# tokenization
	encoder_input_ids = model.tokenizer(encoder_input, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)
	decoder_input_ids = model.tokenizer(decoder_input, padding=True, truncation=True, return_tensors='pt')['input_ids'][:, :-1].to(device)
	decoder_target_ids = model.tokenizer(decoder_target, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)
	print('Encoder input:', model.tokenizer.batch_decode(encoder_input_ids))
	print('Decoder input:', model.tokenizer.batch_decode(decoder_input_ids))
	print('Decoder target:', model.tokenizer.batch_decode(decoder_target_ids))
	print('Target:', decoder_target_ids)

	# encoder forward
	encoder_hidden_states = model.encoder(encoder_input_ids).hidden_states
	print('Encoder hidden states:', encoder_hidden_states.shape)

	# decoder generation
	output = model.decoder.generate(
		input_ids=decoder_input_ids,
		encoder_hidden_states=encoder_hidden_states,
		max_length=decoder_target_ids.shape[1],
		cg=True,
		return_dict_in_generate=True,
		output_scores=True,
		enable_timing=False,
		temperature=1,
		top_k=1,
		top_p=1.0,
		min_p=0.,
		repetition_penalty=1,
	)
	print('Prediction:', output.sequences)
	print(' '.join(''.join(model.tokenizer.batch_decode(output.sequences, clean_up_tokenization_spaces=True, skip_special_tokens=False)).split(' ')))
	print(''.join(''.join(model.tokenizer.batch_decode(output.sequences, clean_up_tokenization_spaces=True, skip_special_tokens=False)).split(' ')))

if __name__ == '__main__':
	main()