import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.central_authority import central_authority
from experiment.config import exp_setting_fs

# Load data
df = pd.read_csv("data/reproduction_number_matrix.csv", header=None)
matrix = df.to_numpy()
dist_repro_num = central_authority(matrix)
total_repro_num = dist_repro_num.get_overall_repro_number()

print(f"The overall reproduction number is {total_repro_num}.")

# update exp_setting
exp_setting_fs['total_repro_num'] = total_repro_num

# Unpack privacy settings
epsilon_start = exp_setting_fs['privacy_setting']['epsilon_start']
epsilon_end = exp_setting_fs['privacy_setting']['epsilon_end']
epsilon_step = exp_setting_fs['privacy_setting']['epsilon_step']
sensitivity = exp_setting_fs['privacy_setting']['sensitivity']
sample_size = exp_setting_fs['privacy_setting']['sample_size']
lb = exp_setting_fs['privacy_setting']['lb']
ub = exp_setting_fs['privacy_setting']['ub']


print(f"Epsilon starts from {epsilon_start} to {epsilon_end} with step sizes {epsilon_step}. \nFor each epsilon, we will draw {sample_size} private samples in total.")

# Start looping epsilon
res = []
for eps in np.arange(epsilon_start, epsilon_end+epsilon_step, epsilon_step):
    private_samples = dist_repro_num.add_dp(epsilon=eps,
                                            sensitivity=sensitivity,
                                            lb=lb,
                                            ub=ub,
                                            num_samples=sample_size)
    abs_diff = [abs(sample.get_overall_repro_number() - total_repro_num) for sample in private_samples]
    res.append(abs_diff)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.boxplot(res[0], 
            labels=list(np.arange(epsilon_start, epsilon_end+epsilon_step, epsilon_step)),
            showfliers=False)
plt.savefig("mygraph.png")