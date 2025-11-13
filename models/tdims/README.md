# TDiMS (Topological Distance of intraMolecular Substructures)

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Setting Up the Development Environment](#setting-up-the-development-environment)
4. [Example Code](#example-code)
   - [Retrieving Full Embeddings](#retrieving-full-embeddings)
   - [Retrieving Feature-Selected Embeddings](#retrieving-feature-selected-embeddings)
   - [Parameters](#parameters)

## Overview
TDiMS is a novel molecular descriptor designed to capture non-local interactions of molecules. Unlike conventional descriptors that either focus solely on local features or struggle to effectively learn long-distance intramolecular interactions, TDiMS overcomes these limitations by effectively summarizing enumerated pairwise topological distances between molecular substructures.

The `tdims` Python package provides molecular embeddings using the TDiMS algorithm. The `tdims_ext` module extends this functionality by offering embeddings with or without feature selection, as well as hyperparameter optimization.

## System Requirements
This project requires the following libraries:

- numpy>=1.26.1, <2.0.0
- pandas>=1.5.3, <2.2.0
- rdkit-pypi>=2022.9.4, <2023.9.6
- scikit-learn>=1.2.0, <1.5.0
- shap>=0.42.1, <0.43.0
- matplotlib>=3.10.1, <3.11.0

## Setting Up the Development Environment
Follow these steps to set up the development environment:

1. Create a new conda virtual environment named `tdims`
*(Typical Install Time: ~15 seconds)*
   ```sh
   conda create -n tdims python=3.10
   ```

2. Activate the virtual environment
   ```sh
   conda activate tdims
   ```

3. Navigate to the project directory
   ```sh
   cd TDiMS
   ```

4. Install the required libraries from `requirements.txt`
*(Typical Install Time: ~30 seconds)*
   ```sh
   pip install -r requirements.txt
   ```

## Example Code
For basic operations, please refer to `example_notebook.ipynb`. For SHAP-based analysis used in the paper, please refer to `SHAP_analitics.ipynb`.

> *Note: To run the example notebooks, please ensure you have a Jupyter environment such as [Jupyter Notebook](https://jupyter.org/) or [JupyterLab](https://jupyterlab.readthedocs.io/). These are not included in the default requirements and should be installed separately if needed.*

### Retrieving Full Embeddings
To generate full embeddings for a given list of SMILES strings:
```python
emb, key_all = tdims_ext.get_representation(sm_list, radius=2, func_dis=-2, func_merge=max, fragment_set=True, atom_set=True, fingerprint_set=True)
```

### Retrieving Feature-Selected Embeddings
To apply feature selection and retrieve optimized embeddings:
```python
x_slc, key_slc, key_all, optimized_param = tdims_ext.get_representation_with_fs_selection(sm_list, y, reg_model="Lasso", radius=1, func_dis=-1, func_merge=sum, fragment_set=False, atom_set=True, fingerprint_set=True)
```

### Parameters
- `reg_model`: Choose from "Lasso", "Ridge", "ElasticNet", "RandomForest"
- `radius`: Integer (≥1), sets the radius for MorganFingerprint substructures
- `func_dis`: Method for calculating feature values from topological distances (e.g., `-1` results in `x^(-1)`)
- `func_merge`: Aggregation method for identical pair distances at different locations (e.g., `sum`, `max`, `min`)
- `fragment_set`: Boolean, whether to include CEP fragments
- `fingerprint_set`: Boolean, whether to include circular fingerprints from Morgan Fingerprints
- `atom_set`: Boolean, whether to include heteroatoms

