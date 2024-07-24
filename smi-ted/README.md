This repository provides pytorch source code associated with our publication, "A Large Encoder-Decoder Family of Foundation Models For Chemical Language".

Paper: [Arxiv Link]()

# materials.smi-ted (smi-ted)

Here, we introduce our large encoder-decoder chemical foundation models pre-trained on a curated dataset of 91 million SMILES samples sourced from PubChem, which is equivalent to 4 billion of molecular tokens. The proposed foundation model supports different complex tasks, including quantum property prediction, and offer flexibility with two main variants ($289M$ and $8\times289M$). Our experiments across multiple benchmark datasets validate the capacity of the proposed model in providing state-of-the-art results for different tasks.

1. [Getting Started](#getting-started)
    1. [Pretrained Models and training logs](#pretrained-models-and-training-logs)
    2. [Replicating Conda Environment](#replicating-conda-environment)
2. [Pretraining](#pretraining)
3. [Finetuning](#finetuning)
4. [Feature extraction](#feature-extraction)
5. [Citations](#citatiobs)


## Getting Started

**This Code and Environment have been tested on Nvidia V100s and Nvidia A100s**

### Pretrained Models and training logs
We are providing checkpoints of a smi-ted model pre-trained on a dataset of ~91M molecules sourced and curated from PubChem. The pre-trained model shows competitive performance on classification and regression benchmarks from MoleculeNet.

Add the smi-ted `pre-trained weights.pt` to the `inference/` or `finetune/` directory according to your necessities.
The hierarchy should look like the following:

```
inference/
├── smi-ted_light
│   ├── smi-ted_light.pt
│   ├── bert_vocab_curated.txt
│   └── load.py
```
and/or:

```
finetune/
├── smi-ted_light
│   ├── smi-ted_light.pt
│   ├── bert_vocab_curated.txt
│   └── load.py
```

### Replicating Conda Environment

Here, is the step-by-step directions to replicate our Conda environment and install the necesary libraries.

## Conda Create and Activate Environment
```
conda create --name smi-ted-env python=3.8.18
conda activate smi-ted-env
```

## Conda Install Packages
```
conda install pytorch=1.13.1 cudatoolkit=11.4 -c pytorch
conda install numpy=1.23.5 pandas=2.0.3
conda install rdkit=2021.03.5 -c conda-forge
```

## Pip install Packages
```
pip install transformers==4.6.0 pytorch-fast-transformers==0.4.0 torch-optimizer==0.3.0 datasets==1.6.2 scikit-learn==1.3.2 scipy==1.12.0 tqdm==4.66.1
```
## Pretraining
For pre-training we use two different strategies: masked language model method to train the encoder part of the model. And an encoder-decoder strategy to refine the reconstruction of the SMILES, and improve the generated latent space.

smi-ted is pre-trained on canonicalized and curated 91M SMILES of PubChem with the following constraints:

During pre-processing, the compounds are filtered to keep a maximum length of 202 tokens. A 95/5/0 split was used for the enconder training. Where 5\% of the data was used to pre-train the decoder part. We also used a 100/0/0 split to train the encoder and decoder directly, improving the performance of the model. 

The pre-training code provides an example of data processing and training of a model trained on a smaller pre-training dataset size, which requires 8 A100 GPUs.

To pre-train the two variants of ted-smi model run:

> bash run_model_light_training.sh
> bash run_model_large_training.sh

You can choose `train_model_D.py` to train just the Decoder part or `train_model_ED.py` to train both, encoder and decoder.

## Finetuning

The finetuning related dataset and environment can be found in [finetune](finetune/). Once you have the environment set up, you can run a fine-tune task by running

> bash smi-ted_light/run_finetune_esol.sh

Finetuning training/checkpointing resources will be available in the diretory named ```checkpoint_<measure_name>```. 

## Feature Extraction

The example notebooks [smi-ted_encoder_decoder_example.ipynb](notebooks/smi-ted_encoder_decoder_example.ipynb) contains code needed to load the checkpoint files and use the pre-trained model for encoder and decoder tasks. You will also find examples of classification and regression tasks.

## Citations
```
to include
```