# materials.smi-ted (smi-ted)
Foundation Model for materials SMILES

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
