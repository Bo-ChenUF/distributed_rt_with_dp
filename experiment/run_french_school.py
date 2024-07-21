import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.central_authority import central_authority

# Load data
df = pd.read_csv("data/reproduction_number_matrix.csv", header=None)
matrix = df.to_numpy()
dist_repro_num = central_authority(matrix)

print(f"The overall reproduction number is {dist_repro_num.get_overall_repro_number()}.")

