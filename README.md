# EGAT
This repository contains the official implementation of our paper *"EGAT: **E**dge Aggregated **G**raph **At**tention Networks and Transfer Learning Improve Protein-Protein Interaction Site Prediction"*

If you use any part of this repository, we shall be obliged if you site our paper.

# Usage
## Pytorch and DGL installation
We implemented our method using PyTorch and Deep Graph Library (DGL). Please install these two for successfully running our code. The installation instructions are available at the following links-
1. [PyTorch](https://pytorch.org/get-started/locally/#start-locally)
2. [Deep Graph Library](https://www.dgl.ai/pages/start.html)

## Download pretrained-model weights:
### ProtBERT model weight
1. Please download the pretrained model weight-file "pytorch_model.bin" from [here](https://drive.google.com/file/d/10MLado6OTLtQ_RWbBEyZNaPCVXaCf73z/view?usp=sharing).
2. Place this weight-file in the folder "EGAT/inputs/ProtBert_model".
If you use this pretrained model for your paper, please cite the paper [ProtTrans: Towards Cracking the Language of Life’s Code Through Self-Supervised Deep Learning and High Performance Computing](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v2)
### EGAT model weight
1. Please download the pretrained model weight-file "egat_model_weight.dat" from [here](https://drive.google.com/file/d/1KsbJ6x8y_8YneO29d81khIhXwt57SWwz/view?usp=sharing).
2. Place this weight-file in the folder "EGAT/models".

## Input Features
To store input-features, navigate to the folder "EGAT/inputs". In this folder, follow any of the following steps:
1. Store the PDB files of the isolated proteins that shall be used for prediction in the folder "pdb_files". Rename the PDB files in the format: <an arbritary name>_<chain IDs>. Please see the example PDB files provided in this folder. Please provide the real chain IDs (as available in the PDB file) after the underscore ("_") correctly. (In the provided examples <an arbritary name> is the PDB ID of a complex in which this input protein is one of the subunits. It is not mendatory.)
2. List all the protein-names in the file "protein_list.txt"

## Run inference to predict numeric propensity (of each of the residues) for interaction
1. From command line cd to "EGAT" folder (where the file "run_egat.py" is situated).
2. Please run the following command:
  > python run_egat.py
3. The command above will generate the results in the "EGAT/outputs" folder.

# Output format

