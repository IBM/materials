from setuptools import find_packages, setup

setup(
    name="geodite",
    packages=find_packages(include=["geodite"]),
    version="1.0.0",
    description="",
    setup_requires=["pytest-runner"],
    tests_require=[
        "coverage-badge==1.1.0",
        "coverage==7.4.3",
        "pytest",
        "torch_geometric==2.5.2",
        "torch==2.1.2",
    ],
    test_suite="tests",
    python_requires=">=3.11.0",
    install_requires=[
        "aim==3.26.1",
        "h5py==3.12.1",
        "ijson==3.3.0",
        "joblib==1.4.0",
        "lmdb==1.4.1",
        "numpy==1.26.4",
        "pymatgen==2024.4.13",
        "pytorch_lightning==2.5.5",
        "torch_geometric==2.5.3",
        "torch_nl==0.3",
        "torch==2.8.0",
        "tqdm==4.66.1",
        "wget==3.2",
    ],
    extras_require={
        "dev": [
            "ruff",
        ],
    },
)
