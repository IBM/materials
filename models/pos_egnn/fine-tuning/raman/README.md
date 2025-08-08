🧪 Raman Spectrum Inference using crystallographic information
This repository provides Python code for inferring Raman spectra using an AI model fine-tuned from the Position-based Equivariant Graph Neural Network (POS-EGNN)—a foundation model designed for chemistry and materials science.

🔍 Overview
The model predicts Raman spectra using crystallographic information of a target material. It outputs a unit vector of 4,000 elements, where each position corresponds to a Raman intensity value. The predicted spectrum has a frequency resolution of 1 cm⁻¹, covering the range from 0 to 4000 cm⁻¹.

🧠 Model Details
•	Base Model: POS-EGNN (pre-trained on 150,000 materials from the Materials Project Trajectory (MPtrj) dataset)
•	Fine-Tuning Task: Adapted to predict Raman spectra
•	Training Data:
    o	~5,400 materials
    o	Experimental data from the Raman Open Database (ROD)
    o	DFT-generated spectra from the Computational Raman Database (CRD)

📥 Input Requirements
To run inference, provide the following crystallographic data:
•	Atomic positions
•	Unit cell representation (lattice)
•	Atomic numbers

📤 Output
•	A 4,000-element vector representing Raman intensity values
•	Frequency resolution: 1 cm⁻¹
•	Range: 0–4000 cm⁻¹

Code (GitHub): https://github.com/IBM/materials/tree/main/models/pos_egnn
Model (HuggingFace): https://huggingface.co/ibm-research/materials.pos-egnn/blob/main/pos-egnn_ft-raman.v3.ckpt
For more information, please reach out to ademir.ferreira@br.ibm.com.

Getting Started
Make sure to have Python 3.12 installed.
Create a project folder. 
Copy the folder Morningstar, requirements.txt and inference.ipynb, available on Github, to the project folder. 
Then, follow these steps below to replicate our environment and install the necessary libraries:
•	Python3.12 -m venv env
•	source env/bin/activate
•	pip install -r requirements.txt

Example
Please refer to the inference.ipynb for a step-by-step demonstration on how to perform the Raman spectrum inference with the model.