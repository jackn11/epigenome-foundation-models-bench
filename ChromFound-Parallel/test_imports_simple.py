#!/usr/bin/env python
"""Quick import test - all imports from cell_embedding.ipynb"""

# Standard library
import os
import subprocess
import sys

# Scientific computing
import numpy as np
import pandas as pd

# Scanpy
import scanpy as sc

# Scikit-learn
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score

# ChromFound local imports
from src.data.atac_preprocess import quality_control
from src.data.atac_preprocess import deepen_atac_data

print("✓ All imports successful!")

