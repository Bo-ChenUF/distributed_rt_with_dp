import numpy as np
import pandas as pd
import h5py
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(
    os.path.join('__file__', '..')
))
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from experiment.config import exp_setting_fs
from model.differential_privacy.mechanisms import privacyPara, SamplesContainer
from model.central_authority import dataCollection, distRepNum, sampleCollection
from util.notebook_util import get_portion_time_series, get_rep_num_time_series
from util.sim_util import ensemble_eff_rep
from util.file_util import check_exist
####################################################################
##### In this file we add privacy to distributed reproduction number
##### We use epidemiology flux model
####################################################################

###################
##### File Settings
###################
# result directory
res_dir = os.path.abspath(
    os.path.join(os.path.curdir, 'results')
)

# figure directory
fig_dir = os.path.abspath(
    os.path.join(os.path.curdir, 'figures')
)

# epidemiology result file
epi_resfile = os.path.join(
    res_dir, 'epi_analysis.hdf5'
)

##############################
##### First clustering results
##############################
# basic information
complevel=7
complib='zlib'
key_base = Path("/first_clustering")
with pd.HDFStore(epi_resfile, complevel=complevel, complib=complib) as store:
    print(f"File {epi_resfile} has {len(store.keys())} entries.")

    ########################################################
    ##### Load clustering and the corresponding csse results
    ########################################################
    # cluster information
    key = str(key_base / 'basic' / "cluster")
    df_clusters_first = store[key]

    #############################
    ##### population related data
    #############################
    # infected portion
    key = str(key_base / "population" / "infected_portion")
    df_inf_portion_first = store[key]

# reproduction numbers
base_path = os.path.join(os.path.abspath(os.path.join("__file__", '..','results')))
save_file_name = 'rep_num_exptest.pkl'
save_str = os.path.abspath(os.path.join(base_path, 
                                            save_file_name))   

with open(save_str, 'rb') as f:
    rep_num_logger = pickle.load(f)
    dataCollection_1023 = rep_num_logger["effective_reproduction_number_matrix"]

###############################
##### Second clustering results
###############################
# basic information
complevel=7
complib='zlib'
key_base = Path("/second_clustering")
with pd.HDFStore(epi_resfile, complevel=complevel, complib=complib) as store:
    print(f"File {epi_resfile} has {len(store.keys())} entries.")

    ########################################################
    ##### Load clustering and the corresponding csse results
    ########################################################
    # cluster information
    key = str(key_base / 'basic' / "cluster")
    df_clusters_second = store[key]

    # confirmed cases information
    key = str(key_base / 'basic' / "confirmed_cases")
    df_confirmed_case_second = store[key]

    #############################
    ##### population related data
    #############################
    # infected portion
    key = str(key_base / "population" / "infected_portion")
    df_inf_portion_second = store[key]

# reproduction numbers
save_file_name = 'simplied_eff_matrixtest.pkl'
save_str = os.path.abspath(os.path.join(base_path, 
                                            save_file_name))   
with open(save_str, 'rb') as f:
    logger = pickle.load(f)
    n_cluster_list = logger['node_id_map_to']
    n_list = logger['node_id_map_from'] 
    simpId2cluId = logger['simpId2cluId']
    dCollection_simplified_1023 = logger['simplified_data_collection_1023']
    dCollection_simplified_1023_w_col_sum = logger['simplified_data_collection_1023_w_col_sum']
    population = logger['population']
    inf_p = logger['infected_portion'] 
    firstClu2SecondClu = logger['firstClu2SecondClu']
    dataCollection_3 = logger['simplifed_effective_reproduction_number_matrix_100']

#########################
##### Experiment Settings
#########################
epsilon_start = exp_setting_fs['privacy_setting']['epsilon_start']
epsilon_end = exp_setting_fs['privacy_setting']['epsilon_end']
epsilon_step = exp_setting_fs['privacy_setting']['epsilon_step']
delta = exp_setting_fs['privacy_setting']['delta']
sensitivity = exp_setting_fs['privacy_setting']['sensitivity']
sample_size = exp_setting_fs['privacy_setting']['sample_size']
lb = exp_setting_fs['privacy_setting']['lb']
ub = exp_setting_fs['privacy_setting']['ub']
start_index = exp_setting_fs['sim_setting']['start_index']

end_index = exp_setting_fs['sim_setting']['end_index']
end_index = end_index if end_index else len(dCollection_simplified_1023.samples)-1

linewidth = exp_setting_fs['plot_setting']['linewidth']
markersize = exp_setting_fs['plot_setting']['markersize']
fontsize = exp_setting_fs['plot_setting']['fontsize']

###################
##### Plot Settings
###################
date = df_confirmed_case_second.columns.to_list()
days = [datetime.datetime.strptime(date[i], '%m/%d/%y') for i in range(len(date))]
node1 = 24
node2 = 39
node3 = 89

##################################################
##### Simulate the ensemable rep number procedures
##################################################
res = []
inf_portion_first = inf_p
epsilons = np.arange(start=epsilon_start, stop=epsilon_end, step=epsilon_step)
for k, epsilon in enumerate(epsilons):
    print(f"Current Epsilon value: {epsilon}")

    ############################
    ##### 0. For each epsilon do
    ############################
    prvPara = privacyPara(epsilon=epsilon,
                          delta=delta,
                          sensitivity=sensitivity,
                          lb=lb,
                          ub=ub)
    
    ####################
    ##### 1. Add privacy
    ####################
    samples = dCollection_simplified_1023_w_col_sum.add_dp(privacyPara=prvPara,
                                                num_samples=sample_size,
                                                start_index=start_index,
                                                end_index=end_index)

    #################
    ##### 2. Ensemble
    #################
    # We start drawing private samples when index is after "start_index"
    sCollector = sampleCollection()
    #for i, sContainer in enumerate(samples):

    for i in range(start_index, end_index+1):
        sContainer = samples[i - start_index]
        s_list = []
        for s in sContainer:
            s_list.append(ensemble_eff_rep(M=len(n_cluster_list),
                                           population=population,
                                           firstClu2SecondClu=firstClu2SecondClu,
                                           simpId2cluId=simpId2cluId,
                                           inf_portion=inf_portion_first[:,i],
                                           matrix=s,
                                           n_cluster_list = n_cluster_list))
        
        sCollector.add(SamplesContainer.from_list(s_list))
    
    res.append(sCollector)

    #############
    ##### 3. Save
    #############
    exp_setting_fs["private_sample_collection"] = sCollector
    save_folder_str = os.path.join(res_dir, 'result_' + datetime.datetime.today().strftime('%m_%d'))
    check_exist(save_folder_str, 'dir')
    save_file_name = exp_setting_fs['exp_name'] + '_' + str(k) + '.pkl'
    save_str = os.path.abspath(os.path.join(save_folder_str, 
                                                save_file_name))   
    with open(save_str, 'wb') as f:
        pickle.dump(exp_setting_fs, f, protocol=pickle.HIGHEST_PROTOCOL)
