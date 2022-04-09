# Requirments
## Python 3.6
## Pytorch 1.7.0
## RDkit 2019.03.4

# Conda Environemt Setup
```
conda install -c rdkit rdkit=2019.03.4 -y
conda install -c pytorch pytorch=1.7.0 -y
```

# Dataset
The data for training and vaildation of Diels-Alder reaction generation are provided in ```dataset``` file. 

# Quickstart
# Step 1: Preprocess the data
Use make_data.ipynb to create the data to be used for SMILES-MaskGAN and save it as a pickle.
## After running the preprocessing, the following files are generated:
```vocab.pt```:the vocab of reaction.

# Step 2: Train the model
## After running the training, the following files are generated:
```maskgan.pt```
## Train
Command line: python run_maskgan.py
