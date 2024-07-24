# materials.smi-ted (smi-ted)
Here, we introduce our large encoder-decoder chemical foundation models pre-trained on a curated dataset of 91 million SMILES samples sourced from PubChem, which is equivalent to 4 billion of molecular tokens. The proposed foundation model supports different complex tasks, including quantum property prediction, and offer flexibility with two main variants (289M and $8\times289M$). Our experiments across multiple benchmark datasets validate the capacity of the proposed model in providing state-of-the-art results for different tasks.
# Python Environment

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
