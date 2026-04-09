from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def add_src_to_path() -> Path:
    """
    Add <repo_root>/src to sys.path so this script can be run without
    `pip install -e .`.

    Expected location of this file:
        <repo_root>/examples/example_basic.py
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    return src_path


SRC_PATH = add_src_to_path()

from tdims import tdims_ext  # noqa: E402


def main() -> None:
    # A tiny example set of molecules
    smiles = [
        "COC(N)=O",           
        "Oc1cocn1",      
        "C#Cc1ncc[nH]1"           
    ]

    print(f"Using src path: {SRC_PATH}")
    print(f"Number of molecules: {len(smiles)}")

    # Generate TDiMS descriptors
    X, feature_names = tdims_ext.get_representation(
        smiles,
        model="tdims",
        radius=1,
        func_dis=-2,
        func_merge=max,
        fragment_set=False,
        atom_set=True,
        fingerprint_set=True,
        display=True,
    )
    
    if len(feature_names) > 0:
        print("\nFirst 5 feature names:")
        for name in feature_names[:5]:
            print(f"- {name}")

if __name__ == "__main__":
    main()