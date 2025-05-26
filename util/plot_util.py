import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import math
from typing import Union
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.central_authority import dataCollection, distRepNum, sampleCollection
from util.notebook_util import get_rep_num_time_series
from experiment.config import exp_setting_fs


node2Info = {
    24: {
        'id_sim' : 0,
        'color' : '#134A8E',
        'label': 'Detroit Area'
    },

    39: {
        'id_sim' : 1,
        'color' : '#0DB14B',
        'label': 'Miami Area'
    },

    89: {
        'id_sim' : 2,
        'color' : '#FFB612',
        'label': 'Delta Junction Area'
    }
}

def plot_time_series_private_r0(sCollector: Union[sampleCollection, dataCollection], node1: int, node2: int, days: list, eff_rep_num_list):
    """
        Plot private r0.

        Inputs:
            - sCollector: sample collection object. 
            - node1: the id of node 1 after the second clustering.
            - node2: the id of node 2 after the second clustering.
            - days: a list of dates
    """
    if (len(days) != len(sCollector)):
        raise ValueError("The dimension of days and sampleCollection must be the same!")

    if type(sCollector) == sampleCollection:
        private = 1
    else:
        private = 0

    ###################
    ##### Plot settings
    ###################
    # basic
    linewidth = exp_setting_fs['plot_setting']['linewidth']
    markersize = exp_setting_fs['plot_setting']['markersize']
    fontsize = exp_setting_fs['plot_setting']['fontsize']

    if private == 1:
        # data settings
        id_sim_1 = node2Info[node1]['id_sim']
        id_sim_2 = node2Info[node2]['id_sim']
        edge21_mean = get_rep_num_time_series(sCollector.get_mean(), id_sim_1, id_sim_2)            # type:ignore
        edge12_mean = get_rep_num_time_series(sCollector.get_mean(), id_sim_2, id_sim_1)            # type:ignore
        edge21_sigma = get_rep_num_time_series(sCollector.get_std(), id_sim_1, id_sim_2)            # type:ignore
        edge12_sigma = get_rep_num_time_series(sCollector.get_std(), id_sim_2, id_sim_1)            # type:ignore
    else:
        # data settings
        id_sim_1 = node1
        id_sim_2 = node2
        edge21_mean = get_rep_num_time_series(sCollector, id_sim_1, id_sim_2)            # type:ignore
        edge12_mean = get_rep_num_time_series(sCollector, id_sim_2, id_sim_1)            # type:ignore

    color1 = node2Info[node1]['color']
    color2 = node2Info[node2]['color']
    label_str1 = node2Info[node1]['label']
    label_str2 = node2Info[node2]['label']

    fig1 = plt.figure(figsize=(7,2))
    ax1 = fig1.add_subplot(111)
    ax1.plot(days, edge21_mean, marker='.', color = color1, linewidth=linewidth, markersize=markersize, label=label_str2 + " to " + label_str1)

    fig2 = plt.figure(figsize=(7,2))
    ax2 = fig2.add_subplot(111)
    ax2.plot(days, edge12_mean, marker='.', color = color2, linewidth=linewidth, markersize=markersize, label=label_str1 + " to " + label_str2)



    if private == 1:

        id_sim_1 = node1
        id_sim_2 = node2
        edge21_true = get_rep_num_time_series(eff_rep_num_list, id_sim_1, id_sim_2)            # type:ignore
        edge12_true = get_rep_num_time_series(eff_rep_num_list, id_sim_2, id_sim_1)            # type:ignore
        ax1.plot(days, edge21_true, linestyle='dashed', color = 'black', label="True Values")
        ax2.plot(days, edge12_true, linestyle='dashed', color = 'black', label="True Values")

        ax1.fill_between(days, 
                        [edge21_mean[i]+edge21_sigma[i] for i in range(len(edge21_mean))],     # type:ignore
                        [edge21_mean[i]-edge21_sigma[i] for i in range(len(edge21_mean))],     # type:ignore
                        alpha = 0.5, 
                        color = color1)
        ax2.fill_between(days, 
                        [edge12_mean[i]+edge12_sigma[i] for i in range(len(edge12_mean))],     # type:ignore
                        [edge12_mean[i]-edge12_sigma[i] for i in range(len(edge12_mean))],     # type:ignore
                        alpha = 0.5, 
                        color = color2)
        
    ax1.set_xlabel('Date', fontsize=fontsize)
    ax1.set_ylabel('Distributed Reproduction Number', fontsize=fontsize)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=50))

    ax2.set_xlabel('Date', fontsize=fontsize)
    ax2.set_ylabel('Distributed Reproduction Number', fontsize=fontsize)
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=50))

    return fig1, fig2


def plot_time_series_diff_private_original(sCollector: sampleCollection, 
                                           dCollector: dataCollection,
                                           node1: int, 
                                           node2: int, 
                                           days: list):
    """
        Plot private r0.

        Inputs:
            - sCollector: sample collection object. Private samples.
            - dCollector: data collection object. A collection of original distributed r0.
            - node1: the id of node 1 after the second clustering.
            - node2: the id of node 2 after the second clustering.
            - days: a list of dates
    """
    if (len(days) != len(sCollector)):
        raise ValueError("The dimension of days and sampleCollection must be the same!")

    ###################
    ##### Plot settings
    ###################
    # basic
    linewidth = exp_setting_fs['plot_setting']['linewidth']
    markersize = exp_setting_fs['plot_setting']['markersize']
    fontsize = exp_setting_fs['plot_setting']['fontsize']

    #####################
    ##### private samples
    #####################
    # data settings
    id_sim_1 = node2Info[node1]['id_sim']
    id_sim_2 = node2Info[node2]['id_sim']
    edge21_mean = get_rep_num_time_series(sCollector.get_mean(), id_sim_1, id_sim_2)            # type:ignore
    edge12_mean = get_rep_num_time_series(sCollector.get_mean(), id_sim_2, id_sim_1)            # type:ignore
    edge21_sigma = get_rep_num_time_series(sCollector.get_std(), id_sim_1, id_sim_2)            # type:ignore
    edge12_sigma = get_rep_num_time_series(sCollector.get_std(), id_sim_2, id_sim_1)            # type:ignore

    #################
    ##### Original r0
    #################
    # data settings
    id_sim_1 = node1
    id_sim_2 = node2
    edge21 = get_rep_num_time_series(dCollector, id_sim_1, id_sim_2)            # type:ignore
    edge12 = get_rep_num_time_series(dCollector, id_sim_2, id_sim_1)            # type:ignore

    edge12_diff = [np.abs(edge12[i] - edge12_mean[i]) for i in range(len(edge12))]
    edge21_diff = [np.abs(edge21[i] - edge21_mean[i]) for i in range(len(edge21))]

    color1 = node2Info[node1]['color']
    color2 = node2Info[node2]['color']
    label_str1 = node2Info[node1]['label']
    label_str2 = node2Info[node2]['label']

    fig1 = plt.figure(figsize=(7,4))
    ax1 = fig1.add_subplot(111)
    ax1.plot(days, edge21_diff, marker='.', color = color1, linewidth=linewidth, markersize=markersize, label=label_str2 + " to " + label_str1)
    ax1.plot(days, edge12_diff, marker='.', color = color2, linewidth=linewidth, markersize=markersize, label=label_str1 + " to " + label_str2)

    ax1.fill_between(days, 
                    [edge21_diff[i]+edge21_sigma[i] for i in range(len(edge21_mean))],     # type:ignore
                    [max(edge21_diff[i]-edge21_sigma[i], 0) for i in range(len(edge21_mean))],     # type:ignore
                    alpha = 0.5, 
                    color = color1)
    ax1.fill_between(days, 
                    [edge12_diff[i]+edge12_sigma[i] for i in range(len(edge12_mean))],     # type:ignore
                    [max(edge12_diff[i]-edge12_sigma[i], 0) for i in range(len(edge12_mean))],     # type:ignore
                    alpha = 0.5, 
                    color = color2)
        
    ax1.set_xlabel('Date', fontsize=fontsize)
    ax1.set_ylabel('Distributed Reproduction Number', fontsize=fontsize)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=50))

def save_fig(fig: Figure,
             fname: str,
             fig_dir: str):
    """
    Save figure.
    
    Input:
        - fig: Figure object.
        - fname: file name.
        - fig_dir: The directory that the figure would save to.
    """
    exts = ['.png', '.svg']
    dpi=300
    
    for ext in exts:
        filepath = os.path.join(fig_dir, fname+ext)
        fig.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=dpi)
        print("Written file: {:s}".format(str(filepath)))