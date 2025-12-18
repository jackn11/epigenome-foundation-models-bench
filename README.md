[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jZYLDMog)

# Investigating the Zero-Shot Capabilities of Single-Cell Epigenomic Foundation Models

## Installation

Both EpiAgent and ChromFound require a unique set of dependencies. We advise you to create two separate `conda` environments as shown in the steps below.

We have tested the installation with
- Python 3.10.19
- CUDA Toolkit 12.1.1 (for GPU support)

###EpiAgent environment preparation

Note: please use the combination of `conda` and `pip` installs as shown below, other combinations are not guaranteed to work.

```bash
# Create the new Conda environment
conda create -n EpiAgentBench python=3.11
conda activate EpiAgentBench

# Install cuda-toolkit using conda (building it gives the incompatible version at least at our machine)
# Note: installing form conda-forge downloads a CPU version
conda install -c nvidia cuda-toolkit=11.7

# Install required additional libraries
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
conda install faiss

# Install EpiAgent from local folder (will install other packages)
cd EpiAgent
pip install -e .
```

### ChromFound environment preparation

Note: please use the combination of `conda` and `pip` installs as shown below, other combinations are not guaranteed to work.

```bash
cd ChromFound
conda env create -f environment.yml
conda activate ChromFoundBench

# Install cuda-toolkit using conda (building it gives the incompatible version at least at our machine)
# Note: installing form conda-forge downloads a CPU version
conda install -c nvidia cuda-toolkit=12.1.1

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Note: add --no-build-isolation otherwase the packages fail because of version mismatches at buildtime
pip install mamba-ssm==2.2.4 --no-build-isolation
pip install flash-attn==2.5.8 --no-build-isolation

pip install 'scib>=1.1.7'
```
