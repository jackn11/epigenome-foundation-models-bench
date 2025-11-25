[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jZYLDMog)

# Investigating the zero-shot performance of scATAC cell embedding models

```bash
# Create the new environment with necessary installs to run EpiAgent benchmarks

conda create -n EpiAgentBench python=3.11
conda activate EpiAgentBench

conda install -c nvidia cuda-toolkit=11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
conda install faiss


# cd into the EpiAgent folder

pip install -e .
```