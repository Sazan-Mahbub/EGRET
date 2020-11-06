# EGAT
This repository contains the official implementation of our paper *"EGAT: **E**dge Aggregated **G**raph **At**tention Networks and Transfer Learning Improve Protein-Protein Interaction Site Prediction"*

If you use any part of this repository, we shall be obliged if you site our paper.

# Usage
## Pytorch and DGL installation
We implemented our method using PyTorch and Deep Graph Library (DGL). Please install these two for successfully running our code. The installation instructions are available at the following links-
1. [PyTorch](https://pytorch.org/get-started/locally/#start-locally))
2. [Deep Graph Library](https://www.dgl.ai/pages/start.html)

## Download pretrained-model weights:
### ProtBERT model weight
1. Please download the pretrained model weight-file "pytorch_model.bin" from [here](https://drive.google.com/file/d/10MLado6OTLtQ_RWbBEyZNaPCVXaCf73z/view?usp=sharing).
2. Place this weight-file in the folder "".
3. If you use this pretrained model, please cite the paper [ProtTrans]()
### EGAT model weight
1. Please download the pretrained model weight-file "egat_model_weight.dat" from [here](https://drive.google.com/file/d/1KsbJ6x8y_8YneO29d81khIhXwt57SWwz/view?usp=sharing).
2. Place this weight-file in the folder "".

## Input Features
To store input-features, navigate to the folder EGAT. Then follow any of the following steps:
1. Store the PDB files of the proteins that shall be used for prediction in the folder "PDBs". Rename the PDB files in the format: <complex name>_<chain names>. Please see the example PDB files provided in this folder.
2. List all the protein-names in the file named "protein_list.txt"

## Run inference to predict secondary structures
1. From command line cd to "" folder (where run_egat.py is situated).
2. Please run the following command:
  > python run_egat.py
3. The command above will generate the results in the "outputs" folder.
  
# Output format
