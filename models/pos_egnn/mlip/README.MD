# <span style="font-size:larger;">Geodite</span>

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0_or_later-green)](https://pytorch.org/)

## Table of contents

- [About Geodite](#about-geodite)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Inference](#inference)
  - [Available Models](#available-models)
- [Citation](#citation)


## About Geodite

**Geodite** is an equivariant message-passing architecture for universal machine-learned interatomic potentials (MLIPs) that eliminates expensive Clebsch-Gordan tensor products while incorporating physical constraints. By avoiding tensor products, Geodite achieves computational efficiency without sacrificing accuracy or physical consistency.

For more details, please refer to our [preprint](#).

## Getting started

### Installation

Install Geodite and its dependencies using `uv`:

```bash
git clone https://github.com/IBM/materials.git
cd materials/models/pos_egnn/mlip
uv pip install -e .
```

### Inference

To perform inference with a pre-trained Geodite model using ASE:

```python
from geodite.calculator import GeoditeCalculator
from ase.build import bulk, make_supercell
import numpy as np

# Load the Geodite-MP model
geodite = GeoditeCalculator(
    "/path/to/checkpoint.ckpt",  
    device="cuda:0", 
    compute_stress=True
)

# Create an atomic structure
atoms = bulk("Al", "fcc", a=4.063)
atoms = make_supercell(atoms, np.eye(3) * 2)
atoms.calc = geodite

# Calculate properties
print("Potential Energy:", atoms.get_potential_energy())
print("Forces:\n", atoms.get_forces())
print("Stress:\n", atoms.get_stress())
```

### Available Models

Currently, we provide one model:

| Model | Parameters | Characteristics | Download |
| :--- | :--- | :--- | :--- |
| **Geodite-MP** | ~3.6M | Trained on the [MPtrj dataset](https://huggingface.co/datasets/nimashoghi/mptrj) covering ~160,000 diverse inorganic crystal structures. High speed (53 μs/atom/step). Ideal for large-scale simulations, high-throughput materials screening, and system-specific fine-tuning. | [🤗 Hugging Face](https://huggingface.co/ibm-research/materials.geodite) |

## How to contribute

If you find a bug or have a proposal for a feature, please post it in the [Issues](https://github.com/IBM/geodite/issues).

Please feel free to reach out to [thiagoreschutzegger@gmail.com](mailto:thiagoreschutzegger@gmail.com) or [fabian.thiemann@ibm.com](mailto:fabian.thiemann@ibm.com) for any questions.

## Citation

TODO

## License

Geodite is released under the [Apache 2.0 License](LICENSE).
