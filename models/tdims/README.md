# TDiMS

TDiMS (Topological Distance of intraMolecular Substructures) is a molecular descriptor designed to capture non-local intramolecular interactions by effectively summarizing enumerated pairwise topological distances between molecular substructures.

This repository includes:

- the TDiMS descriptor implementation
- a minimal Python example
- a Jupyter notebook example
- experiment code for nested cross-validation used in the study

## Repository structure

```text
tdims/
├── README.md
├── requirements.txt
├── requirements-notebook.txt
├── data/
│   ├── cmpCl3_200.csv
├── examples/
│   ├── example_basic.py
│   └── example_notebook.ipynb
├── experiments/
│   └── run_nested_cv_experiment.py
└── src/
    └── tdims/
        ├── __init__.py
        ├── tdims_ext.py
        ├── load.py
        ├── sparse_transformers.py
        └── ChemGenerator/
            ├── __init__.py
            └── ChemGraph.py
```

## Installation

Python 3.9-3.11 is required. Python 3.12 and later are not supported, because
the pinned `pandas` and `rdkit-pypi` versions provide no wheels for them.

### Minimal setup

For the minimal descriptor code and Python example, install the required packages with:

```bash
pip install -r requirements.txt
```

The minimal dependency set is intended for descriptor generation and core functionality.

### Notebook setup

The notebook example uses SHAP. Depending on the platform, installing SHAP with `pip` may trigger `numba` / `llvmlite` build issues. A more stable approach is to install SHAP and its low-level dependencies with conda-forge first, and then install the remaining notebook dependencies.

```bash
conda create -n tdims python=3.10 -y
conda activate tdims
conda install -c conda-forge numba llvmlite shap
pip install -r requirements-notebook.txt
```

## Quick start

Run the basic example:

```bash
python examples/example_basic.py
```

This script generates TDiMS descriptors for a small set of SMILES strings and prints:

- the descriptor matrix shape
- example feature names

## Usage

The main entry point for descriptor generation is `tdims_ext.get_representation()`.

```python
from tdims import tdims_ext

sm_list = ["CCO", "c1ccccc1", "CC(=O)O"]

emb, key_all = tdims_ext.get_representation(
    sm_list,
    radius=1,
    func_dis=-2,
    func_merge=max,
    fragment_set=False,
    atom_set=True,
    fingerprint_set=True,
    display=True,
)
```

To generate descriptors with feature selection, pass the target property
values (the regression target) alongside the SMILES list:

```python
# Measured property value for each molecule in sm_list
prop = [78.4, 80.1, 118.1]

x_slc, key_slc, key_all = tdims_ext.get_representation_with_fs_selection(
    sm_list,
    prop,
    radius=1,
    func_dis=-2,
    func_merge=max,
    fragment_set=False,
    atom_set=True,
    fingerprint_set=True,
    display=True,
)
```

## Using this repository in Jupyter Notebook

If you are using the repository directly in a notebook without package installation, add `src` to `sys.path` before importing `tdims`:

```python
import sys
from pathlib import Path

src_path = (Path.cwd().resolve().parent / "src").resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from tdims import tdims_ext
```

For a notebook-based example, see:

- `examples/example_notebook.ipynb`

## Main parameters

Key arguments of `get_representation()` include:

- `radius`: radius used for substructure extraction
- `func_dis`: transformation applied to topological distance values
- `func_merge`: aggregation function for repeated substructure-pair distances
- `fragment_set`: whether fragment-based substructures are included
- `fingerprint_set`: whether fingerprint-derived substructures are included
- `atom_set`: whether atom-based substructures are included
- `display`: if `True`, prints descriptor shape and elapsed time

## Experiment code

The main experiment script is:

```bash
python experiments/run_nested_cv_experiment.py
```

This script is intended for nested cross-validation experiments used in the study. It is separate from the minimal examples above and is provided for experiment-level reproduction.

- `quick`: reduced search space for a first test run or lightweight debugging.
  Depending on the environment, this can still take several tens of minutes.
- `full`: used for the main journal-paper experiments. Considerably slower than
  `quick`; running it on a compute node is recommended.

## Notes

- Internal imports inside `src/tdims/` are package-relative.
- External usage from notebooks or scripts should use `from tdims import ...`.
- If the repository is not package-installed, add `src/` to `sys.path` before import.
- `requirements.txt` is intended for minimal descriptor usage.
- `requirements-notebook.txt` is intended for the notebook example and related plotting dependencies.

## Citation

If you use this repository in academic work, please cite the corresponding paper.

## License

This code is distributed under the Apache License 2.0 as part of the `IBM/materials` repository. See the `LICENSE` file in the root of the repository for details.
