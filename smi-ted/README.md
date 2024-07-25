# SMILES-based Transformer Encoder-Decoder (SMI-TED)

This repository provides PyTorch source code associated with our publication, "A Large Encoder-Decoder Family of Foundation Models for Chemical Language".

Paper: [Arxiv Link](paper/smi_ted_preprint.pdf)
For model weights contact: eduardo.soares@ibm.com or evital@br.ibm.com .

![ted-smi](images/smi-ted.png)

## Introduction

We present a large encoder-decoder chemical foundation model, SMILES-based Transformer Encoder-Decoder (SMI-TED), pre-trained on a curated dataset of 91 million SMILES samples sourced from PubChem, equivalent to 4 billion molecular tokens. SMI-TED supports various complex tasks, including quantum property prediction, with two main variants ($289M$ and $8 \times 289M$). Our experiments across multiple benchmark datasets demonstrate state-of-the-art performance for various tasks. For model weights contact: eduardo.soares@ibm.com or evital@br.ibm.com .

## Table of Contents

1. [Getting Started](#getting-started)
    1. [Pretrained Models and Training Logs](#pretrained-models-and-training-logs)
    2. [Replicating Conda Environment](#replicating-conda-environment)
2. [Pretraining](#pretraining)
3. [Finetuning](#finetuning)
4. [Feature Extraction](#feature-extraction)
5. [Citations](#citations)

## Getting Started

**This code and environment have been tested on Nvidia V100s and Nvidia A100s**

### Pretrained Models and Training Logs

We provide checkpoints of the SMI-TED model pre-trained on a dataset of ~91M molecules curated from PubChem. The pre-trained model shows competitive performance on classification and regression benchmarks from MoleculeNet. For model weights contact: eduardo.soares@ibm.com or evital@br.ibm.com .

Add the SMI-TED `pre-trained weights.pt` to the `inference/` or `finetune/` directory according to your needs. The directory structure should look like the following:

```
inference/
├── smi_ted_light
│   ├── smi_ted_light.pt
│   ├── bert_vocab_curated.txt
│   └── load.py
```
and/or:

```
finetune/
├── smi_ted_light
│   ├── smi_ted_light.pt
│   ├── bert_vocab_curated.txt
│   └── load.py
```

### Replicating Conda Environment

Follow these steps to replicate our Conda environment and install the necessary libraries:

#### Create and Activate Conda Environment

```
conda create --name smi-ted-env python=3.8.18
conda activate smi-ted-env
```

#### Install Packages with Conda

```
conda install pytorch=1.13.1 cudatoolkit=11.4 -c pytorch
conda install numpy=1.23.5 pandas=2.0.3
conda install rdkit=2021.03.5 -c conda-forge
```

#### Install Packages with Pip

```
pip install transformers==4.6.0 pytorch-fast-transformers==0.4.0 torch-optimizer==0.3.0 datasets==1.6.2 scikit-learn==1.3.2 scipy==1.12.0 tqdm==4.66.1
```

## Pretraining

For pretraining, we use two strategies: the masked language model method to train the encoder part and an encoder-decoder strategy to refine SMILES reconstruction and improve the generated latent space.

SMI-TED is pre-trained on canonicalized and curated 91M SMILES from PubChem with the following constraints:

- Compounds are filtered to a maximum length of 202 tokens during preprocessing.
- A 95/5/0 split is used for encoder training, with 5% of the data for decoder pretraining.
- A 100/0/0 split is also used to train the encoder and decoder directly, enhancing model performance.

The pretraining code provides examples of data processing and model training on a smaller dataset, requiring 8 A100 GPUs.

To pre-train the two variants of the SMI-TED model, run:

```
bash run_model_light_training.sh
```
or
```
bash run_model_large_training.sh
```

Use `train_model_D.py` to train only the decoder or `train_model_ED.py` to train both the encoder and decoder.

## Finetuning

The finetuning datasets and environment can be found in the [finetune](finetune/) directory. After setting up the environment, you can run a finetuning task with:

```
bash smi_ted_light/run_finetune_esol.sh
```

Finetuning training/checkpointing resources will be available in directories named `checkpoint_<measure_name>`.

## Feature Extraction

The example notebook [smi_ted_encoder_decoder_example.ipynb](notebooks/smi_ted_encoder_decoder_example.ipynb) contains code to load checkpoint files and use the pre-trained model for encoder and decoder tasks. It also includes examples of classification and regression tasks. For model weights contact: eduardo.soares@ibm.com or evital@br.ibm.com.

To load smi-ted, you can simply use:

```python
model = load_smi_ted(
    folder='../inference/smi_ted_light',
    ckpt_filename='smi_ted_light.pt'
)
```

To encode SMILES into embeddings, you can use:

```python
with torch.no_grad():
    encoded_embeddings = model.encode(df['SMILES'], return_torch=True)
```
For decoder, you can use the function, so you can return from embeddings to SMILES strings:

```python
with torch.no_grad():
    decoded_smiles = model.decode(encoded_embeddings)
```


## Citations

```
to include
```