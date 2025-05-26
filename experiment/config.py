import numpy as np

exp_setting_fs = {
    'exp_name' : 'distributed_r0_exp',
    
    'total_repro_num' : None,
    
    'privacy_setting' : {
        'epsilon_start' : 1,
        'epsilon_end' : 3,
        'epsilon_step' : 0.2,
        'sensitivity' : 0.00001,
        'sample_size' : 100,
        'lb' : [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 10, 20, 50, 100, 250, 500],
        'ub' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 10, 20, 50, 100, 250, 500, 1000],
        'delta' : 0
    },

    'sim_setting' : {
        'start_index' : 200,
        'end_index' : None
    },

    'plot_setting' : {
        'linewidth' : 3,
        'markersize' : 6,
        'fontsize' : 12
    }
}

