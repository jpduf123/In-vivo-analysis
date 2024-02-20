# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:20:36 2023

@author: Mann Lab
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from matplotlib.pyplot import cm
import neo
import quantities as pq
import elephant
import scipy
import scipy.signal
from scipy import stats
import os
import copy
import pickle
import natsort
from statistics import mean
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import itertools
import math 
import matplotlib.colors as colors


overall_path = r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS'


days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

numb = len(days)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


channelMapArray = np.array([[30, 46, 31, 47,  1, 49,  0, 48],
                            [28, 44, 29, 45,  3, 51,  2, 50],
                            [26, 42, 27, 43,  5, 53,  4, 52],
                            [24, 40, 25, 41,  7, 55,  6, 54],
                            [22, 38, 23, 39,  9, 57,  8, 56],
                            [20, 36, 21, 37, 11, 59, 10, 58],
                            [18, 34, 19, 35, 13, 61, 12, 60],
                            [16, 32, 17, 33, 15, 63, 14, 62]])

chanMap = channelMapArray.flatten()

def cl():
    plt.close('all')

sweeps_for_SW_compare = [3]


#%% ------------------------------------------------- LOAD UP GROUP ARRAYS FOR PAIRED AND UNPAIRED RECORDINGS
to_plot_1_LFP = [0,1,2,3]
to_plot_2_LFP = [4,5,6,7,8,9]


lfp_cutoff_resp_channels = 200

os.chdir(os.path.join(overall_path, r'UNPAIRED_whisker'))
days_2whisk = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

#groups arrays for all mice with unpaired whisker stims:
LFP_min_2whisk_ALL = np.zeros([len(days_2whisk),10,64])
LFP_min_rel_2whisk_ALL = np.zeros([len(days_2whisk),10,64])
LFP_min_rel_change_2whisk_ALL = np.zeros([len(days_2whisk),64])

PSTH_peak_2whisk_ALL = np.zeros([len(days_2whisk),10,64])
PSTH_peak_rel_2whisk_ALL = np.zeros([len(days_2whisk),10,64])
PSTH_peak_rel_change_2whisk_ALL = np.zeros([len(days_2whisk),64])
PSTH_magn_2whisk_ALL = np.zeros([len(days_2whisk),10,64])
PSTH_magn_rel_2whisk_ALL = np.zeros([len(days_2whisk),10,64])
PSTH_magn_rel_change_2whisk_ALL = np.zeros([len(days_2whisk),64])

delta_2whisk_ALL = np.zeros([len(days_2whisk),10,64])
delta_rel_2whisk_ALL = np.zeros([len(days_2whisk),10,64])
delta_rel_change_2whisk_ALL = np.zeros([len(days_2whisk),64])

# baseline SW characteristics (to compare to same channels with other whisker deflected)
SW_freq_2whisk_ALL = np.zeros([len(days_2whisk),64])
SW_dur_2whisk_ALL = np.zeros([len(days_2whisk),64])
SW_famp_2whisk_ALL = np.zeros([len(days_2whisk),64])
SW_samp_2whisk_ALL = np.zeros([len(days_2whisk),64])
SW_fslope_2whisk_ALL = np.zeros([len(days_2whisk),64])
SW_sslope_2whisk_ALL = np.zeros([len(days_2whisk),64])
SW_waveform_2whisk_ALL = np.zeros([len(days_2whisk),64,1000])

LFP_resp_channels_cutoff_2whisk_list = []
PSTH_resp_channels_2whisk_list = []

# first unpaired whisker
for day_ind, day in enumerate(days_2whisk):
    os.chdir(day)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    #get channels
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_resp_channels_cutoff = np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
    PSTH_resp_channels = np.loadtxt('PSTH_resp_channels.csv', delimiter = ',', dtype = int)
    
    # if day == '160414_B2':
    #     chans_to_append = [35,37,11] 
    #     chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels_cutoff]
    #     for chan in chans_to_append:
    #         LFP_resp_channels_cutoff = np.append(LFP_resp_channels_cutoff, chan)
        # PSTH_bad = [56,61]  # increases by >200% during baseline
        # for chan in PSTH_bad:  
        #     PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == chan)[0]) 
    if day == '160426_B2':
    #     chans_to_append = [9,11,18,20,23,25,34,36,38] 
    #     chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels_cutoff]
    #     for chan in chans_to_append:
    #         LFP_resp_channels_cutoff = np.append(LFP_resp_channels_cutoff, chan)
        PSTH_bad = [33,37]  # decreases by >200% during baseline or changes shape
        for chan in PSTH_bad:  
            PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == chan)[0]) 
    if day == '160519_D1':
        # chans_to_append = [19,26] 
        # chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels_cutoff]
        # for chan in chans_to_append:
        #     LFP_resp_channels_cutoff = np.append(LFP_resp_channels_cutoff, chan)
        PSTH_bad = [42,38,23,11,18]  # increase by >200% and change shape
        for chan in PSTH_bad:  
            PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == chan)[0]) 
    # if day == '160624_D1':
    #     chans_to_append = [9,11,13,25,61] 
    #     chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels_cutoff]
    #     for chan in chans_to_append:
    #         LFP_resp_channels_cutoff = np.append(LFP_resp_channels_cutoff, chan)  
    # if day == '160628_B2':
    #     pass
    
    
    LFP_resp_channels_cutoff_2whisk_list.append(LFP_resp_channels_cutoff)
    PSTH_resp_channels_2whisk_list.append(PSTH_resp_channels)

    LFP_min_2whisk_ALL[day_ind, :, :] = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_min_rel_2whisk_ALL[day_ind, :, :] = LFP_min_2whisk_ALL[day_ind, :, :]/np.nanmean(LFP_min_2whisk_ALL[day_ind, [0,1,2,3], :], axis = 0)
    LFP_min_rel_change_2whisk_ALL[day_ind, :] = np.loadtxt('LFP_min_rel_change.csv', delimiter = ',')

    PSTH_peak_2whisk_ALL[day_ind, :, :] = np.loadtxt('PSTH_resp_peak.csv', delimiter = ',')
    PSTH_peak_rel_2whisk_ALL[day_ind, :, :] = PSTH_peak_2whisk_ALL[day_ind, :, :]/np.nanmean(PSTH_peak_2whisk_ALL[day_ind, [0,1,2,3], :], axis = 0)
    PSTH_peak_rel_change_2whisk_ALL[day_ind, :] = np.loadtxt('PSTH_resp_peak_rel_change.csv', delimiter = ',')
    
    PSTH_magn_2whisk_ALL[day_ind, :, :] = np.loadtxt('PSTH_resp_magn.csv', delimiter = ',')
    PSTH_magn_rel_2whisk_ALL[day_ind, :, :] = PSTH_magn_2whisk_ALL[day_ind, :, :]/np.nanmean(PSTH_magn_2whisk_ALL[day_ind, [0,1,2,3], :], axis = 0)
    PSTH_magn_rel_change_2whisk_ALL[day_ind, :] = np.loadtxt('PSTH_resp_magn_rel_change.csv', delimiter = ',')
        
    delta_2whisk_ALL[day_ind, :, :] = np.loadtxt('delta_power.csv', delimiter = ',')
    delta_rel_2whisk_ALL[day_ind, :, :] = delta_2whisk_ALL[day_ind, :, :]/np.nanmean(delta_2whisk_ALL[day_ind, [0,1,2,3], :], axis = 0)
    delta_rel_change_2whisk_ALL[day_ind, :] = np.loadtxt('delta_power_rel_change.csv', delimiter = ',')
    
    SW_freq_2whisk_ALL[day_ind, :] = np.mean(np.load('SW_frequency_sweeps_avg.npy')[sweeps_for_SW_compare,:], axis = 0)
    SW_dur_2whisk_ALL[day_ind, :] = np.mean(np.load('Peak_dur_sweeps_avg_overall.npy')[sweeps_for_SW_compare,:], axis = 0)
    SW_famp_2whisk_ALL[day_ind, :] = np.mean(np.load('SW_famp_sweeps_avg_overall.npy')[sweeps_for_SW_compare,:], axis = 0)
    SW_samp_2whisk_ALL[day_ind, :] = np.mean(np.load('SW_samp_sweeps_avg_overall.npy')[sweeps_for_SW_compare,:], axis = 0)
    SW_fslope_2whisk_ALL[day_ind, :] = np.mean(np.load('SW_fslope_sweeps_avg_overall.npy')[sweeps_for_SW_compare,:], axis = 0)
    SW_sslope_2whisk_ALL[day_ind, :] = np.mean(np.load('SW_sslope_sweeps_avg_overall.npy')[sweeps_for_SW_compare,:], axis = 0)
    
    SW_waveform_2whisk_ALL[day_ind, :] = np.mean(np.load('SW_waveform_sweeps_avg.npy')[sweeps_for_SW_compare,:,:], axis = 0)
    
    os.chdir('..')
    
    # # heatmap figure
    # # os.chdir(home_directory)
    # #  colorplot of LFP min magnitude
    # fig, ax = plt.subplots()
    # plot = ax.imshow(np.reshape(LFP_min[1,chanMap], (8, 8)), cmap = 'Blues')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.tight_layout()
    # plt.savefig('LFP min colormap.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('LFP min colormap.jpg', dpi = 1000, format = 'jpg')
    # # plt.colorbar(plot)
    
    # fig, ax = plt.subplots(figsize = (0.1,5))
    # cmap = cm.Blues
    # norm = colors.Normalize(vmin=0, vmax=1)
    # fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
    #               cax=ax, ticks = [0,0.33,0.66,1])
    # ax.set_yticklabels(list(map(str, np.linspace(-0, -3, 4).astype(int))), size = 18)
    # ax.set_ylabel('LFP response peak (mVolt)', size = 16)
    # plt.tight_layout()
    # plt.savefig('LFP peak colormap legend.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('LFP peak colormap legend.jpg', dpi = 1000, format = 'jpg')


    os.chdir('..')



# paired whisker:
os.chdir(os.path.join(overall_path, r'UP_pairing'))
days_1whisk = ['160414_D1', '160426_D1', '160519_B2', '160624_B2', '160628_D1']
LFP_min_1whisk_ALL = np.zeros([len(days_1whisk),10,64])
LFP_min_rel_1whisk_ALL = np.zeros([len(days_1whisk),10,64])
LFP_min_rel_change_1whisk_ALL = np.zeros([len(days_1whisk),64])
PSTH_peak_1whisk_ALL = np.zeros([len(days_2whisk),10,64])
PSTH_peak_rel_1whisk_ALL = np.zeros([len(days_2whisk),10,64])
PSTH_peak_rel_change_1whisk_ALL = np.zeros([len(days_2whisk),64])
PSTH_magn_1whisk_ALL = np.zeros([len(days_2whisk),10,64])
PSTH_magn_rel_1whisk_ALL = np.zeros([len(days_2whisk),10,64])
PSTH_magn_rel_change_1whisk_ALL = np.zeros([len(days_2whisk),64])
delta_1whisk_ALL = np.zeros([len(days_2whisk),10,64])
delta_rel_1whisk_ALL = np.zeros([len(days_2whisk),10,64])
delta_rel_change_1whisk_ALL = np.zeros([len(days_2whisk),64])

# baseline SW characteristics (to compare to same channels with other whisker deflected)
SW_freq_1whisk_ALL = np.zeros([len(days_2whisk),64])
SW_dur_1whisk_ALL = np.zeros([len(days_2whisk),64])
SW_famp_1whisk_ALL = np.zeros([len(days_2whisk),64])
SW_samp_1whisk_ALL = np.zeros([len(days_2whisk),64])
SW_fslope_1whisk_ALL = np.zeros([len(days_2whisk),64])
SW_sslope_1whisk_ALL = np.zeros([len(days_2whisk),64])
SW_waveform_1whisk_ALL = np.zeros([len(days_2whisk),64,1000])

LFP_resp_channels_cutoff_1whisk_list = []
PSTH_resp_channels_1whisk_list = []

for day_ind, day in enumerate(days_1whisk):
    os.chdir(day)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_resp_channels_cutoff = np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
    # LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
    PSTH_resp_channels = np.loadtxt('PSTH_resp_channels.csv', delimiter = ',', dtype = int)

    # if day == '160426_1':
    #     LFP_bad = [36,20] # noisy
    #     for chan in LFP_bad:  
    #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
    # if day == '160519_B2':
    #     LFP_bad = [27,20] # noisy
    #     for chan in LFP_bad:  
    #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
    if day == '160624_B2':
            LFP_resp_channels_cutoff = np.append(LFP_resp_channels_cutoff, 17)                 
        # LFP_bad = [21,19] # noisy
        # for chan in LFP_bad:  
        #     LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
    # if day == '160628_D1':
    #     LFP_bad = [37,35,19,17] # noisy
    #     for chan in LFP_bad:  
    #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 

    
    LFP_resp_channels_cutoff_1whisk_list.append(LFP_resp_channels_cutoff)
    PSTH_resp_channels_1whisk_list.append(PSTH_resp_channels)
    
    LFP_min_1whisk_ALL[day_ind, :, :] = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_min_rel_1whisk_ALL[day_ind, :, :] = LFP_min_1whisk_ALL[day_ind, :, :]/np.nanmean(LFP_min_1whisk_ALL[day_ind, [0,1,2,3], :], axis = 0)
    LFP_min_rel_change_1whisk_ALL[day_ind, :] = np.loadtxt('LFP_min_rel_change.csv', delimiter = ',')

    PSTH_peak_1whisk_ALL[day_ind, :, :] = np.loadtxt('PSTH_resp_peak.csv', delimiter = ',')
    PSTH_peak_rel_1whisk_ALL[day_ind, :, :] = PSTH_peak_1whisk_ALL[day_ind, :, :]/np.nanmean(PSTH_peak_1whisk_ALL[day_ind, [0,1,2,3], :], axis = 0)
    PSTH_peak_rel_change_1whisk_ALL[day_ind, :] = np.loadtxt('PSTH_resp_peak_rel_change.csv', delimiter = ',')
    
    PSTH_magn_1whisk_ALL[day_ind, :, :] = np.loadtxt('PSTH_resp_magn.csv', delimiter = ',')
    PSTH_magn_rel_1whisk_ALL[day_ind, :, :] = PSTH_magn_1whisk_ALL[day_ind, :, :]/np.nanmean(PSTH_magn_1whisk_ALL[day_ind, [0,1,2,3], :], axis = 0)
    PSTH_magn_rel_change_1whisk_ALL[day_ind, :] = np.loadtxt('PSTH_resp_magn_rel_change.csv', delimiter = ',')
        
    delta_1whisk_ALL[day_ind, :, :] = np.loadtxt('delta_power.csv', delimiter = ',')
    delta_rel_1whisk_ALL[day_ind, :, :] = delta_1whisk_ALL[day_ind, :, :]/np.nanmean(delta_1whisk_ALL[day_ind, [0,1,2,3], :], axis = 0)
    delta_rel_change_1whisk_ALL[day_ind, :] = np.loadtxt('delta_power_rel_change.csv', delimiter = ',')
    
    SW_freq_1whisk_ALL[day_ind, :] = np.mean(np.load('SW_frequency_sweeps_avg.npy')[sweeps_for_SW_compare,:], axis = 0)
    SW_dur_1whisk_ALL[day_ind, :] = np.mean(np.load('Peak_dur_sweeps_avg_overall.npy')[sweeps_for_SW_compare,:], axis = 0)
    SW_famp_1whisk_ALL[day_ind, :] = np.mean(np.load('SW_famp_sweeps_avg_overall.npy')[sweeps_for_SW_compare,:], axis = 0)
    SW_samp_1whisk_ALL[day_ind, :] = np.mean(np.load('SW_samp_sweeps_avg_overall.npy')[sweeps_for_SW_compare,:], axis = 0)
    SW_fslope_1whisk_ALL[day_ind, :] = np.mean(np.load('SW_fslope_sweeps_avg_overall.npy')[sweeps_for_SW_compare,:], axis = 0)
    SW_sslope_1whisk_ALL[day_ind, :] = np.mean(np.load('SW_sslope_sweeps_avg_overall.npy')[sweeps_for_SW_compare,:], axis = 0)
    
    SW_waveform_1whisk_ALL[day_ind, :] = np.mean(np.load('SW_waveform_sweeps_avg.npy')[sweeps_for_SW_compare,:,:], axis = 0)

    os.chdir('..')
    os.chdir('..')

# channels responsive to any, both, or only unpaired
LFP_resp_channels_unique_whisk = [np.unique(np.hstack((LFP_resp_channels_cutoff_2whisk_list[i], LFP_resp_channels_cutoff_1whisk_list[i]))) for i in range(5)] # responsive to all
LFP_resp_channels_overlap_whisk = [np.intersect1d(LFP_resp_channels_cutoff_2whisk_list[i], LFP_resp_channels_cutoff_1whisk_list[i]) for i in range(5)] # overlap
LFP_resp_channels_nonoverlap_2whisk = [LFP_resp_channels_cutoff_2whisk_list[i][~np.isin(LFP_resp_channels_cutoff_2whisk_list[i], LFP_resp_channels_cutoff_1whisk_list[i])] for i in range(5)]

PSTH_resp_channels_unique_whisk = [np.unique(np.hstack((PSTH_resp_channels_2whisk_list[i], PSTH_resp_channels_1whisk_list[i]))) for i in range(5)]
PSTH_resp_channels_overlap_whisk = [np.intersect1d(PSTH_resp_channels_2whisk_list[i], PSTH_resp_channels_1whisk_list[i]) for i in range(5)]
PSTH_resp_channels_nonoverlap_2whisk = [PSTH_resp_channels_2whisk_list[i][~np.isin(PSTH_resp_channels_2whisk_list[i], PSTH_resp_channels_1whisk_list[i])] for i in range(5)]



#%% 1. ---------------------------------------------- LFP and spike depression paired vs unpaired, timecourse

# mean change per mouse
paired_LFP_change = [np.mean(LFP_min_rel_change_1whisk_ALL[i,LFP_resp_channels_cutoff_1whisk_list[i]]) for i in range(5)]
unpaired_LFP_change = [np.mean(LFP_min_rel_change_2whisk_ALL[i,LFP_resp_channels_cutoff_2whisk_list[i]]) for i in range(5)]

paired_MUA_change = [np.mean(PSTH_magn_rel_change_1whisk_ALL[i,PSTH_resp_channels_1whisk_list[i]]) for i in range(5)]
unpaired_MUA_change = [np.mean(PSTH_magn_rel_change_2whisk_ALL[i,PSTH_resp_channels_2whisk_list[i]]) for i in range(5)]

paired_delta_change = [np.mean(delta_rel_change_1whisk_ALL[i,LFP_resp_channels_cutoff_1whisk_list[i]]) for i in range(5)]
unpaired_delta_change = [np.mean(delta_rel_change_2whisk_ALL[i,LFP_resp_channels_cutoff_2whisk_list[i]]) for i in range(5)]


scipy.stats.shapiro(unpaired_MUA_change)
scipy.stats.shapiro(paired_MUA_change)
# t test, do unpaired and paired depress after pairing?
print(f'paired LFP: {1 + np.mean(paired_LFP_change)}/{np.std(paired_LFP_change, ddof = 1)}, p = {stats.ttest_1samp(paired_LFP_change, 0)[1]}')
print(f'unpaired LFP: {1 + np.mean(unpaired_LFP_change)}/{np.std(unpaired_LFP_change, ddof = 1)}, p = {stats.ttest_1samp(unpaired_LFP_change, 0)[1]}')

print(f'paired MUA: {1 + np.mean(paired_MUA_change)}/{np.std(paired_MUA_change, ddof = 1)}, p = {stats.ttest_1samp(paired_MUA_change, 0)[1]}')
print(f'unpaired MUA: {1 + np.mean(unpaired_MUA_change)}/{np.std(unpaired_MUA_change, ddof = 1)}, p = {stats.ttest_1samp(unpaired_MUA_change, 0)[1]}')

# significant difference in depression? this is be the same as interaction in 2x2 mixed ANOVA with before/after and paired/unpaired. I do ANOVA on timecourse so not really that useful.
# print(stats.ttest_rel(paired_LFP_change, unpaired_LFP_change))
# print(stats.ttest_rel(paired_MUA_change, unpaired_MUA_change))


# plotbox and whisker plot of before/after change
flierprops = dict(marker='x', markerfacecolor='k')
fig, ax = plt.subplots(figsize = (5,3))
ax.boxplot([np.asarray(paired_LFP_change)*100, np.asarray(unpaired_LFP_change)*100], flierprops=flierprops, widths = 0.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1,2])
ax.set_xticklabels(['paired whisker', 'unpaired whisker'], size = 16)
ax.set_ylabel('LFP plasticity (%)', size = 16)
ax.set_yticks([-50,-25,0])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 14)
ax.set_ylim([-70,25])
# ax.set_yticklabels(['-40', '-20', '20', '40', '60'], size = 16)
plt.tight_layout()
plt.savefig('unpaired vs paired LFP boxplot.pdf', dpi = 1000, format = 'pdf')
plt.savefig('unpaired vs paired LFP boxplot.jpg', dpi = 1000, format = 'jpg')


fig, ax = plt.subplots(figsize = (5,3))
ax.boxplot([np.asarray(paired_MUA_change)*100, np.asarray(unpaired_MUA_change)*100], flierprops=flierprops, widths = 0.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([1,2])
ax.set_xticklabels(['paired whisker', 'unpaired whisker'], size = 16)
ax.set_ylabel('PSTH plasticity (%)', size = 16)
ax.set_yticks([-50,-25,0])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 14)
ax.set_ylim([-70,25])
#statistical annotation
x1, x2 = 1, 2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = (np.asarray(unpaired_MUA_change)*100).max() + 10, 5, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
plt.tight_layout()
plt.savefig('unpaired vs paired LFP boxplot.pdf', dpi = 1000, format = 'pdf')
plt.savefig('unpaired vs paired PSTH boxplot.jpg', dpi = 1000, format = 'jpg')



# prepare for ANOVA on timecourse
np.savetxt('LFP_min_rel_unpaired.csv', np.asarray([np.mean(LFP_min_rel_2whisk_ALL[i,:,LFP_resp_channels_cutoff_2whisk_list[i]], axis = 0) for i in range(5)]), delimiter = ',')
np.savetxt('LFP_min_rel_paired.csv', np.asarray([np.mean(LFP_min_rel_1whisk_ALL[i,:,LFP_resp_channels_cutoff_1whisk_list[i]], axis = 0) for i in range(5)]), delimiter = ',')

np.savetxt('delta_rel_unpaired.csv', np.asarray([np.mean(delta_rel_2whisk_ALL[i,:,LFP_resp_channels_cutoff_2whisk_list[i]], axis = 0) for i in range(5)]), delimiter = ',')
np.savetxt('delta_rel_paired.csv', np.asarray([np.mean(delta_rel_1whisk_ALL[i,:,LFP_resp_channels_cutoff_1whisk_list[i]], axis = 0) for i in range(5)]), delimiter = ',')

np.savetxt('PSTH_magn_rel_unpaired.csv', np.asarray([np.mean(PSTH_magn_rel_2whisk_ALL[i,:,PSTH_resp_channels_2whisk_list[i]], axis = 0) for i in range(5)]), delimiter = ',')
PSTH_magn_rel_paired = []
# one paired mouse with only one good PSTH channel
for i in range(5):
    if PSTH_resp_channels_1whisk_list[i].size == 1:
        PSTH_magn_rel_paired.append(PSTH_magn_rel_1whisk_ALL[i,:,PSTH_resp_channels_1whisk_list[i]])
    else:
        PSTH_magn_rel_paired.append(np.mean(PSTH_magn_rel_1whisk_ALL[i,:,PSTH_resp_channels_1whisk_list[i]], axis = 0))
np.savetxt('PSTH_magn_rel_paired.csv', np.asarray(PSTH_magn_rel_paired), delimiter = ',')





#plot LFP timecourse
patch = 1
fig, ax = plt.subplots(figsize = (10,4))
to_plot_1 = np.asarray([np.mean(LFP_min_rel_1whisk_ALL[i,:,LFP_resp_channels_cutoff_1whisk_list[i]], axis = 0) for i in range(5)])*100
ax.plot(np.nanmean(to_plot_1, axis = 0), color = 'r')
ax.fill_between(list(range(10)), np.nanmean(to_plot_1, axis = 0) + patch*np.nanstd(to_plot_1, axis = 0)/np.sqrt(to_plot_1.shape[0]), np.nanmean(to_plot_1, axis = 0) - patch*np.nanstd(to_plot_1, axis = 0)/np.sqrt(to_plot_1.shape[0]), alpha = 0.1, color = 'r')
to_plot_2 = np.asarray([np.mean(LFP_min_rel_2whisk_ALL[i,:,LFP_resp_channels_cutoff_2whisk_list[i]], axis = 0) for i in range(5)])*100
ax.plot(np.nanmean(to_plot_2, axis = 0), c = 'k')
ax.fill_between(list(range(10)), np.nanmean(to_plot_2, axis = 0) + patch*np.nanstd(to_plot_2, axis = 0)/np.sqrt(to_plot_2.shape[0]), np.nanmean(to_plot_2, axis = 0) - patch*np.nanstd(to_plot_2, axis = 0)/np.sqrt(to_plot_2.shape[0]), alpha = 0.1, color = 'k')
# ax.set_ylim([30, 160])
ax.set_xlabel('time from pairing (min)', size = 20)
ax.set_ylabel('LFP response (% of baseline', size = 20)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 20)
ax.tick_params(axis="y", labelsize=20)    
ax.axvline(3.5, linestyle = '--', color = 'k')
# ax.set_yticks([50,100,150])
# ax.set_yticklabels(list(map(str,[50,100,150])), size = 16)
plt.tight_layout()
plt.savefig('LFP paired vs unpaired.pdf', dpi = 1000, format = 'pdf')
plt.savefig('LFP paired vs unpaired.jpg', dpi = 1000, format = 'jpg')

# stats.ttest_1samp((np.mean(to_plot_1[:,[0,1,2,3]], axis = 1) - np.mean(to_plot_1[:,[4,5,6,7,8,9]], axis = 1)),0)
# stats.ttest_1samp((np.mean(to_plot_2[:,[0,1,2,3]], axis = 1) - np.mean(to_plot_2[:,[4,5,6,7,8,9]], axis = 1)),0)



# plot PSTH timecourse
patch = 1
fig, ax = plt.subplots(figsize = (10,4))
to_plot_1 = np.asarray(PSTH_magn_rel_paired)*100
ax.plot(np.nanmean(to_plot_1, axis = 0), color = 'r')
ax.fill_between(list(range(10)), np.nanmean(to_plot_1, axis = 0) + patch*np.nanstd(to_plot_1, axis = 0)/np.sqrt(to_plot_1.shape[0]), np.nanmean(to_plot_1, axis = 0) - patch*np.nanstd(to_plot_1, axis = 0)/np.sqrt(to_plot_1.shape[0]), alpha = 0.1, color = 'r')
to_plot_2 = np.asarray([np.mean(PSTH_magn_rel_2whisk_ALL[i,:,PSTH_resp_channels_2whisk_list[i]], axis = 0) for i in range(5)])*100
ax.plot(np.nanmean(to_plot_2, axis = 0), color = 'k')
ax.fill_between(list(range(10)), np.nanmean(to_plot_2, axis = 0) + patch*np.nanstd(to_plot_2, axis = 0)/np.sqrt(to_plot_2.shape[0]), np.nanmean(to_plot_2, axis = 0) - patch*np.nanstd(to_plot_2, axis = 0)/np.sqrt(to_plot_2.shape[0]), alpha = 0.1, color = 'k')
# ax.set_ylim([30, 160])
ax.set_xlabel('time from pairing (min)', size = 20)
ax.set_ylabel('MUA response (% of baseline', size = 20)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 20)
ax.tick_params(axis="y", labelsize=20)   
ax.axvline(3.5, linestyle = '--', color = 'k') 
# ax.set_yticks([50,100,150])
# ax.set_yticklabels(list(map(str,[50,100,150])), size = 16)
plt.tight_layout()
plt.savefig('MUA paired vs unpaired.pdf', dpi = 1000, format = 'pdf')
plt.savefig('MUA paired vs unpaired.jpg', dpi = 1000, format = 'jpg')
# np.savetxt('LFP_min_rel_paired.csv', np.concatenate([LFP_min_rel_change_1whisk_ALL[i,LFP_resp_channels_cutoff_1whisk_list[i]] for i in range(5)]), delimiter = ',')




#plot delta timecourse
fig, ax = plt.subplots()
to_plot_1 = np.asarray([np.mean(delta_rel_1whisk_ALL[i,:,LFP_resp_channels_cutoff_1whisk_list[i]], axis = 0) for i in range(5)])*100
ax.plot(np.nanmean(to_plot_1, axis = 0), color = 'r')
ax.fill_between(list(range(10)), np.nanmean(to_plot_1, axis = 0) + patch*np.nanstd(to_plot_1, axis = 0)/np.sqrt(to_plot_1.shape[0]), np.nanmean(to_plot_1, axis = 0) - patch*np.nanstd(to_plot_1, axis = 0)/np.sqrt(to_plot_1.shape[0]), alpha = 0.1, color = 'r')
to_plot_2 = np.asarray([np.mean(delta_rel_2whisk_ALL[i,:,LFP_resp_channels_cutoff_2whisk_list[i]], axis = 0) for i in range(5)])*100
ax.plot(np.nanmean(to_plot_2, axis = 0), c = 'k')
ax.fill_between(list(range(10)), np.nanmean(to_plot_2, axis = 0) + patch*np.nanstd(to_plot_2, axis = 0)/np.sqrt(to_plot_2.shape[0]), np.nanmean(to_plot_2, axis = 0) - patch*np.nanstd(to_plot_2, axis = 0)/np.sqrt(to_plot_2.shape[0]), alpha = 0.1, color = 'k')
# ax.set_ylim([30, 160])
ax.set_xlabel('time from pairing (min)', size = 16)
ax.set_ylabel('delta change (% of baseline', size = 16)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 16)
# ax.set_yticks([50,100,150])
# ax.set_yticklabels(list(map(str,[50,100,150])), size = 16)
plt.tight_layout()
plt.savefig('delta paired vs unpaired.pdf', dpi = 1000, format = 'pdf')
plt.savefig('delta paired vs unpaired.jpg', dpi = 1000, format = 'jpg')




#%% 2. ---------------------------------------------- significant difference in depression between overlapping and non overlapping unpaired channels?? NO
# NO, not significant, plus overlapping ones are less depressed than non overlapping ones...

LFP_plast_in_overlap = []
LFP_plast_in_nonoverlap = []

PSTH_plast_in_overlap = []
PSTH_plast_in_nonoverlap = []

for i in range(5):
    LFP_plast_in_overlap.append(LFP_min_rel_change_2whisk_ALL[i,LFP_resp_channels_overlap_whisk[i]])
    LFP_plast_in_nonoverlap.append(LFP_min_rel_change_2whisk_ALL[i,LFP_resp_channels_nonoverlap_2whisk[i]])
    
    PSTH_plast_in_overlap.append(PSTH_peak_rel_change_2whisk_ALL[i,PSTH_resp_channels_overlap_whisk[i]])
    PSTH_plast_in_nonoverlap.append(PSTH_peak_rel_change_2whisk_ALL[i,PSTH_resp_channels_nonoverlap_2whisk[i]])
    
# for the average of every mouse
print(stats.ttest_rel([np.mean(LFP_plast_in_overlap[i]) for i in range(5)], [np.mean(LFP_plast_in_nonoverlap[i]) for i in range(5)]))
# stats.ttest_rel([np.nanmean(PSTH_plast_in_overlap[i]) for i in range(5) if not np.isnan(np.nanmean(PSTH_plast_in_overlap[i]))], [np.nanmean(PSTH_plast_in_nonoverlap[i]) for i in range(5)])

# all channels
print(scipy.stats.shapiro([np.mean(LFP_plast_in_nonoverlap[i]) for i in range(5)]))
print(scipy.stats.ttest_ind(np.concatenate(LFP_plast_in_overlap), np.concatenate(LFP_plast_in_nonoverlap)))
print(f'overlap: {np.mean(np.concatenate(LFP_plast_in_overlap))}, non overlap {np.mean(np.concatenate(LFP_plast_in_nonoverlap))}')


# scipy.stats.shapiro(np.concatenate(PSTH_plast_in_nonoverlap))
# scipy.stats.ttest_ind(np.concatenate(PSTH_plast_in_overlap), np.concatenate(PSTH_plast_in_nonoverlap))

np.savetxt('overlap non overlap plast.csv', np.concatenate([np.concatenate(LFP_plast_in_overlap), np.concatenate(LFP_plast_in_nonoverlap)]), delimiter = ',')




#%% 3. ---------------------------------------------- whisker specificity vs depression in LFP and MUA response, for channels responsive to any or both

# ------------------------------------------------- 3.1) all channels for both whiskers (not very useful)
# spec_unique_LFP = []
# spec_unique_PSTH = []
# plast_LFP_1whisk_unique = []
# plast_LFP_2whisk_unique = []

# plast_PSTHpeak_1whisk_unique = []
# plast_PSTHpeak_2whisk_unique = []

# plast_PSTHmagn_1whisk_unique = []
# plast_PSTHmagn_2whisk_unique = []


# for i in range(5):
#     print(i)
#     # spec_unique.append(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_unique_whisk[i]]/(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_unique_whisk[i]] + np.mean(LFP_min_2whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_unique_whisk[i]]))
#     spec_unique_LFP.append(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_unique_whisk[i]])
#     plast_LFP_1whisk_unique.append(LFP_min_rel_change_1whisk_ALL[i,[LFP_resp_channels_unique_whisk[i]]])
#     plast_LFP_2whisk_unique.append(LFP_min_rel_change_2whisk_ALL[i,[LFP_resp_channels_unique_whisk[i]]])
    
#     spec_unique_PSTH.append(np.mean(PSTH_peak_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[PSTH_resp_channels_unique_whisk[i]])
#     plast_PSTHpeak_1whisk_unique.append(PSTH_peak_rel_change_1whisk_ALL[i,[PSTH_resp_channels_unique_whisk[i]]])
#     plast_PSTHpeak_2whisk_unique.append(PSTH_peak_rel_change_2whisk_ALL[i,[PSTH_resp_channels_unique_whisk[i]]])
#     plast_PSTHmagn_1whisk_unique.append(PSTH_magn_rel_change_1whisk_ALL[i,[PSTH_resp_channels_unique_whisk[i]]])
#     plast_PSTHmagn_2whisk_unique.append(PSTH_magn_rel_change_2whisk_ALL[i,[PSTH_resp_channels_unique_whisk[i]]])
    
#     # fig, ax = plt.subplots()
#     # ax.scatter(spec_unique[i], plast_unique[i])
    
# fig, ax = plt.subplots()
# ax.scatter(np.hstack(spec_unique_LFP), np.hstack(plast_LFP_2whisk_unique))
# slope, intercept, r, p, std_err = stats.linregress(np.hstack(spec_unique_LFP), np.hstack(plast_LFP_2whisk_unique))
# print(f'{r} and {p} for {len(np.hstack(spec_unique_LFP))} channels')


# ------------------------------------------------- 3.2) overlapping channels 
spec_overlap_LFP = []
spec_overlap_PSTH = []

plast_LFP_1whisk_overlap = []
plast_LFP_2whisk_overlap = []

plast_PSTHpeak_1whisk_overlap = []
plast_PSTHpeak_2whisk_overlap = []

plast_PSTHmagn_1whisk_overlap = []
plast_PSTHmagn_2whisk_overlap = []

for i in range(5):
    print(i)
    # spec_overlap_LFP.append(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_overlap_whisk[i]]/(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_overlap_whisk[i]] + np.mean(LFP_min_2whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_overlap_whisk[i]]))
    
    # it's not the ratio of whisker 1/2 that's important! it's the strength of the whisker 1 response in that channel! 
    # --> strength of the response to the paired whisker as a propotion of the strongest response across channels
    # spec_overlap_LFP.append(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_overlap_whisk[i]]/np.max(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_overlap_whisk[i]]))
    spec_overlap_LFP.append(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_overlap_whisk[i]]/np.max(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)))

    # --> absolute strength of the response to the paired whisker 
    # spec_overlap_LFP.append(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_overlap_whisk[i]])
    
    plast_LFP_1whisk_overlap.append(LFP_min_rel_change_1whisk_ALL[i,[LFP_resp_channels_overlap_whisk[i]]])
    plast_LFP_2whisk_overlap.append(LFP_min_rel_change_2whisk_ALL[i,[LFP_resp_channels_overlap_whisk[i]]])
    
    if PSTH_resp_channels_overlap_whisk[i].size == 0:
        continue
    else:
        # spec_overlap_PSTH.append(np.mean(PSTH_peak_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[PSTH_resp_channels_overlap_whisk[i]])
        spec_overlap_PSTH.append(np.mean(PSTH_magn_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[PSTH_resp_channels_overlap_whisk[i]]/np.max(np.mean(PSTH_magn_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[PSTH_resp_channels_overlap_whisk[i]]))
        plast_PSTHpeak_1whisk_overlap.append(PSTH_peak_rel_change_1whisk_ALL[i,[PSTH_resp_channels_overlap_whisk[i]]])
        plast_PSTHpeak_2whisk_overlap.append(PSTH_peak_rel_change_2whisk_ALL[i,[PSTH_resp_channels_overlap_whisk[i]]])
        plast_PSTHmagn_1whisk_overlap.append(PSTH_magn_rel_change_1whisk_ALL[i,[PSTH_resp_channels_overlap_whisk[i]]])
        plast_PSTHmagn_2whisk_overlap.append(PSTH_magn_rel_change_2whisk_ALL[i,[PSTH_resp_channels_overlap_whisk[i]]])
    
    # plot for every mouse:
    # fig, ax = plt.subplots()
    # ax.scatter(spec_overlap_LFP[i], plast_LFP_2whisk_overlap[i])

# LFP 
exclude_outliers = False
fig, ax = plt.subplots()
X = np.hstack(spec_overlap_LFP)*100
Y = np.hstack(plast_LFP_2whisk_overlap)[0]*100
X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
if exclude_outliers:
    X = np.delete(X, outliers)
    Y = np.delete(Y, outliers)
ax.scatter(X, Y, color = 'k')
slope, intercept, r, p, std_err = stats.linregress(X, Y)
print(f'{r} and {p} for {len(Y)} channels')
ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('paired whisker baseline response \n (% of principal channel)', size = 16)
ax.set_xticks([0,25,50,75,100])
ax.set_xticklabels(list(map(str, list(ax.get_xticks()))), size = 16)
ax.set_ylabel('unpaired whisker LFP plasticity \n(% of baseline)', size = 16)
ax.set_yticks([-75,-50,-25,0,25,50])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
plt.tight_layout()
# plt.savefig('unpaired vs paired baseline OVERLAP.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('unpaired vs paired baseline OVERLAP.jpg', dpi = 1000, format = 'jpg')


# # # spike magn
# exclude_outliers = True
# fig, ax = plt.subplots()
# X = np.hstack(spec_overlap_PSTH)/10
# Y = np.hstack(plast_PSTHmagn_2whisk_overlap)[0]*100 + 100
# X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
# Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
# outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
# if exclude_outliers:
#     X = np.delete(X, outliers)
#     Y = np.delete(Y, outliers)
# ax.scatter(X, Y)
# slope, intercept, r, p, std_err = stats.linregress(X, Y)
# print(f'{r} and {p} for {len(Y)} channels')
# ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)])



# ------------------------------------------------- 3.3) all whisker 2 channels:
spec_whisk2_LFP = []
spec_whisk2_LFP_overlap = []
spec_whisk2_LFP_nonoverlap = []
spec_whisk2_PSTH = []
plast_LFP_2whisk_whisk2 = []
plast_LFP_2whisk_whisk2_overlap = []
plast_LFP_2whisk_whisk2_nonoverlap = []
plast_PSTHpeak_2whisk_whisk2 = []
plast_PSTHmagn_2whisk_whisk2 = []

for i in range(5):
    print(i)
    # spec_whisk2_LFP.append(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_cutoff_2whisk_list[i]]/(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_cutoff_2whisk_list[i]] + np.mean(LFP_min_2whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_cutoff_2whisk_list[i]]))
    
    # option 1: set all the non overlapped channels to zero
    # LFP_min_1whisk_ALL_temp = copy.deepcopy(LFP_min_1whisk_ALL)
    # LFP_min_1whisk_ALL_temp[i,:,~np.isin(np.asarray(list(range(64))), LFP_resp_channels_overlap_whisk[i])] = 0
    # spec_whisk2_LFP.append(np.mean(LFP_min_1whisk_ALL_temp[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_cutoff_2whisk_list[i]])/np.max(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_cutoff_2whisk_list[i]])
    
    # option 2: take min of whisk 1 regardless of if it's been responsive channels or no
    spec_whisk2_LFP.append(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_cutoff_2whisk_list[i]]/np.max(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)))
    # spec_whisk2_LFP.append(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_cutoff_2whisk_list[i]])
    spec_whisk2_LFP_overlap.append(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_overlap_whisk[i]]/np.max(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)))
    spec_whisk2_LFP_nonoverlap.append(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[LFP_resp_channels_cutoff_2whisk_list[i][~np.isin(LFP_resp_channels_cutoff_2whisk_list[i], LFP_resp_channels_overlap_whisk[i])]]/np.max(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)))

    # option 3: set all non overlapped channels to zero on x axis
    # spec_whisk2_LFP.append(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)/np.max(np.mean(LFP_min_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)))
    # spec_whisk2_LFP[i][~np.isin(np.asarray(list(range(64))), LFP_resp_channels_overlap_whisk[i])] = 0
    # spec_whisk2_LFP[i] = spec_whisk2_LFP[i][LFP_resp_channels_cutoff_2whisk_list[i]]
    
    plast_LFP_2whisk_whisk2.append(LFP_min_rel_change_2whisk_ALL[i,[LFP_resp_channels_cutoff_2whisk_list[i]]])
    plast_LFP_2whisk_whisk2_overlap.append(LFP_min_rel_change_2whisk_ALL[i,[LFP_resp_channels_overlap_whisk[i]]])
    plast_LFP_2whisk_whisk2_nonoverlap.append(LFP_min_rel_change_2whisk_ALL[i,LFP_resp_channels_cutoff_2whisk_list[i][~np.isin(LFP_resp_channels_cutoff_2whisk_list[i], LFP_resp_channels_overlap_whisk[i])]])

    PSTH_peak_1whisk_ALL_temp = copy.deepcopy(PSTH_peak_1whisk_ALL)
    PSTH_peak_1whisk_ALL_temp[i,:,~np.isin(np.asarray(list(range(64))), PSTH_resp_channels_overlap_whisk[i])] = 0
    PSTH_magn_1whisk_ALL_temp = copy.deepcopy(PSTH_magn_1whisk_ALL)
    PSTH_magn_1whisk_ALL_temp[i,:,~np.isin(np.asarray(list(range(64))), PSTH_resp_channels_overlap_whisk[i])] = 0
    
    spec_whisk2_PSTH.append(np.mean(PSTH_magn_1whisk_ALL_temp[i,to_plot_1_LFP,:], axis = 0)[PSTH_resp_channels_2whisk_list[i]]/np.max(np.mean(PSTH_magn_1whisk_ALL[i,to_plot_1_LFP,:], axis = 0)[PSTH_resp_channels_2whisk_list[i]]))
    plast_PSTHpeak_2whisk_whisk2.append(PSTH_peak_rel_change_2whisk_ALL[i,[PSTH_resp_channels_2whisk_list[i]]])
    plast_PSTHmagn_2whisk_whisk2.append(PSTH_magn_rel_change_2whisk_ALL[i,[PSTH_resp_channels_2whisk_list[i]]])


# --------------------------------------------- all with correlation LFP 
exclude_outliers = False
fig, ax = plt.subplots()
X = np.hstack(spec_whisk2_LFP)*100
Y = np.hstack(plast_LFP_2whisk_whisk2)[0]*100
# X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
# Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
# outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
# if exclude_outliers:
#     X = np.delete(X, outliers)
#     Y = np.delete(Y, outliers)
ax.scatter(X, Y, color = 'k')
slope, intercept, r, p, std_err = stats.linregress(X, Y)
print(f'{r} and {p} for {len(Y)} channels')
ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('paired whisker baseline response \n (% of principal channel)', size = 16)
ax.set_xticks([0,25,50,75,100])
ax.set_xticklabels(list(map(str, list(ax.get_xticks()))), size = 16)
ax.set_ylabel('unpaired whisker LFP plasticity \n(% of baseline)', size = 16)
ax.set_yticks([-50,-25,0,25])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
plt.tight_layout()
# plt.savefig('unpaired vs paired baseline ALL.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('unpaired vs paired baseline ALL.jpg', dpi = 1000, format = 'jpg')
# ax.set_xticklabels()
# ax.set_ylim([35,135])


# # --------------------------------------------- all overlapped vs non overlapped in different colors correlation LFP 
# LFP 
exclude_outliers = False
fig, ax = plt.subplots()
X = np.hstack(spec_whisk2_LFP_overlap)*100
Y = np.hstack(plast_LFP_2whisk_whisk2_overlap)*100
ax.scatter(X, Y, color = 'k')
slope, intercept, r, p, std_err = stats.linregress(X, Y)
print(f'{r} and {p} for {len(Y)} channels')
ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
X = np.hstack(spec_whisk2_LFP_nonoverlap)*100
Y = np.hstack(plast_LFP_2whisk_whisk2_nonoverlap)*100
ax.scatter(X, Y, color = 'r')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('paired whisker baseline response \n (% of principal channel)', size = 16)
ax.set_xticks([0,25,50,75,100])
ax.set_xticklabels(list(map(str, list(ax.get_xticks()))), size = 16)
ax.set_ylabel('unpaired whisker LFP plasticity \n(% of baseline)', size = 16)
ax.set_yticks([-75,-50,-25,0,25,50])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
plt.tight_layout()
plt.savefig('unpaired vs paired baseline ALL.pdf', dpi = 1000, format = 'pdf')
plt.savefig('unpaired vs paired baseline ALL.jpg', dpi = 1000, format = 'jpg')


# # # spike magn
# exclude_outliers = False
# fig, ax = plt.subplots()
# X = np.hstack(spec_whisk2_PSTH)/10
# Y = np.hstack(plast_PSTHmagn_2whisk_whisk2)[0]*100 + 100
# X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
# Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
# outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
# if exclude_outliers:
#     X = np.delete(X, outliers)
#     Y = np.delete(Y, outliers)
# ax.scatter(X, Y)
# slope, intercept, r, p, std_err = stats.linregress(X, Y)
# print(f'{r} and {p} for {len(Y)} channels')
# ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)])




#%% 4. ---------------------------------------------- depression figure as in Antonia, depression whisker 1 vs whisker 2 in overlapping channels

# LFP
exclude_outliers = False    
fig, ax = plt.subplots()
# fig.suptitle('LFP')
X = np.hstack(plast_LFP_1whisk_overlap)[0]*100 + 100
Y = np.hstack(plast_LFP_2whisk_overlap)[0]*100 + 100
X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
if exclude_outliers:
    X = np.delete(X, outliers)
    Y = np.delete(Y, outliers)
ax.scatter(X, Y, color = 'k')
slope, intercept, r, p, std_err = stats.linregress(X, Y)
print(f'{r} and {p} for {len(Y)} channels')
ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], c = 'r')
ax.plot([40,140], [40,140], linestyle = '--', color = 'k')
ax.set_xlim([40,125])
ax.set_ylim([40,125])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('paired whisker LFP response \n (% of baseline)', size = 18)
ax.set_ylabel('unpaired whisker LFP response \n (% of baseline)', size = 18)
ax.set_yticks([40,60,80,100,120])
ax.set_xticks([40,60,80,100,120])
ax.set_xticklabels(list(map(str, ax.get_xticks())), size = 18)
ax.set_yticklabels(list(map(str, ax.get_xticks())), size = 18)
plt.tight_layout()
plt.savefig('LFP change paired vs unpaired whisker.pdf', dpi = 1000, format = 'pdf')
plt.savefig('LFP change paired vs unpaired whisker.jpg', dpi = 1000, format = 'jpg')



exclude_outliers = False    
fig, ax = plt.subplots()
# fig.suptitle('PSTH magn')
X = np.hstack(plast_PSTHmagn_1whisk_overlap)[0]*100 + 100
Y = np.hstack(plast_PSTHmagn_2whisk_overlap)[0]*100 + 100
X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
if exclude_outliers:
    X = np.delete(X, outliers)
    Y = np.delete(Y, outliers)
ax.scatter(X, Y, color = 'k')
slope, intercept, r, p, std_err = stats.linregress(X, Y)
print(f'{r} and {p} for {len(Y)} channels')
ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], c = 'r')
ax.plot([30,140], [30,140], linestyle = '--', color = 'k')
ax.set_xlim([30,125])
ax.set_ylim([30,125])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('paired whisker MUA response \n (% of baseline)', size = 18)
ax.set_ylabel('unpaired whisker MUA response \n (% of baseline)', size = 18)
ax.set_yticks([40,60,80,100,120])
ax.set_xticks([40,60,80,100,120])
ax.set_xticklabels(list(map(str, ax.get_xticks())), size = 18)
ax.set_yticklabels(list(map(str, ax.get_xticks())), size = 18)
plt.tight_layout()
plt.savefig('MUA change paired vs unpaired whisker.pdf', dpi = 1000, format = 'pdf')
plt.savefig('MUA change paired vs unpaired whisker.jpg', dpi = 1000, format = 'jpg')

# exclude_outliers = True    
# fig, ax = plt.subplots()
# fig.suptitle('PSTH magn')
# X = np.hstack(plast_PSTHmagn_1whisk_overlap)[0]*100 + 100
# Y = np.hstack(plast_PSTHmagn_2whisk_overlap)[0]*100 + 100
# X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
# Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
# outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
# if exclude_outliers:
#     X = np.delete(X, outliers)
#     Y = np.delete(Y, outliers)
# ax.scatter(X, Y)
# slope, intercept, r, p, std_err = stats.linregress(X, Y)
# print(f'{r} and {p} for {len(Y)} channels')
# ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], c = 'r')
# ax.plot([20,150], [20,150])
# ax.set_xlim([20,150])
# ax.set_ylim([20,150])


#%% 5. ---------------------------------------------- SW and delta power in unpaired vs paired whisker channels during baseline

sweeps_for_SW_compare = [0,1,2,3]

# update overlapping channels with some good ones that are just below detection threshold (or else one bad delta power channel can fuck up the whole average)
# LFP_resp_channels_overlap_whisk_for_delta[3] = np.append(LFP_resp_channels_overlap_whisk_for_delta[3], [15,33,35,17,61])
# LFP_resp_channels_overlap_whisk_for_delta[4] = np.append(LFP_resp_channels_overlap_whisk_for_delta[4], [23,25,29])   

# take out noisy channels
LFP_resp_channels_overlap_whisk_for_delta = copy.deepcopy(LFP_resp_channels_overlap_whisk)
LFP_resp_channels_overlap_whisk_for_delta[2] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[2], np.where(LFP_resp_channels_overlap_whisk_for_delta[2] == 27)[0]) 
LFP_resp_channels_overlap_whisk_for_delta[2] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[2], np.where(LFP_resp_channels_overlap_whisk_for_delta[2] == 20)[0]) 

LFP_resp_channels_overlap_whisk_for_delta[3] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[3], np.where(LFP_resp_channels_overlap_whisk_for_delta[3] == 21)[0]) 
LFP_resp_channels_overlap_whisk_for_delta[3] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[3], np.where(LFP_resp_channels_overlap_whisk_for_delta[3] == 19)[0]) 

LFP_resp_channels_overlap_whisk_for_delta[4] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[4], np.where(LFP_resp_channels_overlap_whisk_for_delta[4] == 39)[0])   
LFP_resp_channels_overlap_whisk_for_delta[4] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[4], np.where(LFP_resp_channels_overlap_whisk_for_delta[4] == 37)[0])   
LFP_resp_channels_overlap_whisk_for_delta[4] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[4], np.where(LFP_resp_channels_overlap_whisk_for_delta[4] == 35)[0])   
# LFP_resp_channels_overlap_whisk_for_delta[4] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[4], np.where(LFP_resp_channels_overlap_whisk_for_delta[4] == 19)[0])   
# LFP_resp_channels_overlap_whisk_for_delta[4] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[4], np.where(LFP_resp_channels_overlap_whisk_for_delta[4] == 17)[0])   



LFP_resp_channels_unique_whisk_for_delta = copy.deepcopy(LFP_resp_channels_unique_whisk)
LFP_resp_channels_unique_whisk_for_delta[0] = np.delete(LFP_resp_channels_unique_whisk_for_delta[0], np.where(LFP_resp_channels_unique_whisk_for_delta[0] == 7)[0]) 
LFP_resp_channels_unique_whisk_for_delta[0] = np.delete(LFP_resp_channels_unique_whisk_for_delta[0], np.where(LFP_resp_channels_unique_whisk_for_delta[0] == 11)[0]) 
LFP_resp_channels_unique_whisk_for_delta[0] = np.delete(LFP_resp_channels_unique_whisk_for_delta[0], np.where(LFP_resp_channels_unique_whisk_for_delta[0] == 39)[0]) 

LFP_resp_channels_unique_whisk_for_delta[1] = np.delete(LFP_resp_channels_unique_whisk_for_delta[1], np.where(LFP_resp_channels_unique_whisk_for_delta[1] == 33)[0]) 
LFP_resp_channels_unique_whisk_for_delta[1] = np.delete(LFP_resp_channels_unique_whisk_for_delta[1], np.where(LFP_resp_channels_unique_whisk_for_delta[1] == 14)[0]) 
LFP_resp_channels_unique_whisk_for_delta[1] = np.delete(LFP_resp_channels_unique_whisk_for_delta[1], np.where(LFP_resp_channels_unique_whisk_for_delta[1] == 36)[0]) 
LFP_resp_channels_unique_whisk_for_delta[1] = np.delete(LFP_resp_channels_unique_whisk_for_delta[1], np.where(LFP_resp_channels_unique_whisk_for_delta[1] == 20)[0]) 

LFP_resp_channels_unique_whisk_for_delta[2] = np.delete(LFP_resp_channels_unique_whisk_for_delta[2], np.where(LFP_resp_channels_unique_whisk_for_delta[2] == 55)[0]) 
LFP_resp_channels_unique_whisk_for_delta[2] = np.delete(LFP_resp_channels_unique_whisk_for_delta[2], np.where(LFP_resp_channels_unique_whisk_for_delta[2] == 58)[0]) 
LFP_resp_channels_unique_whisk_for_delta[2] = np.delete(LFP_resp_channels_unique_whisk_for_delta[2], np.where(LFP_resp_channels_unique_whisk_for_delta[2] == 20)[0]) 
LFP_resp_channels_unique_whisk_for_delta[2] = np.delete(LFP_resp_channels_unique_whisk_for_delta[2], np.where(LFP_resp_channels_unique_whisk_for_delta[2] == 27)[0]) 
LFP_resp_channels_unique_whisk_for_delta[2] = np.delete(LFP_resp_channels_unique_whisk_for_delta[2], np.where(LFP_resp_channels_unique_whisk_for_delta[2] == 36)[0]) 
LFP_resp_channels_unique_whisk_for_delta[2] = np.delete(LFP_resp_channels_unique_whisk_for_delta[2], np.where(LFP_resp_channels_unique_whisk_for_delta[2] == 46)[0]) 

LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 15)[0]) 
LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 17)[0]) 
LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 19)[0]) 
LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 21)[0]) 
LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 35)[0]) 
LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 37)[0]) 
LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 39)[0]) 

LFP_resp_channels_unique_whisk_for_delta[4] = np.delete(LFP_resp_channels_unique_whisk_for_delta[4], np.where(LFP_resp_channels_unique_whisk_for_delta[4] == 39)[0])   
LFP_resp_channels_unique_whisk_for_delta[4] = np.delete(LFP_resp_channels_unique_whisk_for_delta[4], np.where(LFP_resp_channels_unique_whisk_for_delta[4] == 37)[0])   
LFP_resp_channels_unique_whisk_for_delta[4] = np.delete(LFP_resp_channels_unique_whisk_for_delta[4], np.where(LFP_resp_channels_unique_whisk_for_delta[4] == 35)[0])   
LFP_resp_channels_unique_whisk_for_delta[4] = np.delete(LFP_resp_channels_unique_whisk_for_delta[4], np.where(LFP_resp_channels_unique_whisk_for_delta[4] == 19)[0])   
LFP_resp_channels_unique_whisk_for_delta[4] = np.delete(LFP_resp_channels_unique_whisk_for_delta[4], np.where(LFP_resp_channels_unique_whisk_for_delta[4] == 17)[0])   





channels = LFP_resp_channels_unique_whisk_for_delta
# change within each mouse, average of change in overlapping channels
delta_1vs2 = (np.mean(delta_1whisk_ALL[:,sweeps_for_SW_compare,:], axis = 1) - np.mean(delta_2whisk_ALL[:,sweeps_for_SW_compare,:], axis = 1))/np.mean(delta_2whisk_ALL[:,sweeps_for_SW_compare,:], axis = 1)
freq_1vs2 = (SW_freq_1whisk_ALL - SW_freq_2whisk_ALL)/SW_freq_1whisk_ALL
dur_1vs2 = (SW_dur_1whisk_ALL - SW_dur_2whisk_ALL)/SW_dur_1whisk_ALL
famp_1vs2 = (SW_famp_1whisk_ALL - SW_famp_2whisk_ALL)/SW_famp_1whisk_ALL
samp_1vs2 = (SW_samp_1whisk_ALL - SW_samp_2whisk_ALL)/SW_samp_1whisk_ALL
fslope_1vs2 = (SW_fslope_1whisk_ALL - SW_fslope_2whisk_ALL)/SW_fslope_1whisk_ALL
sslope_1vs2 = (SW_sslope_1whisk_ALL - SW_sslope_2whisk_ALL)/SW_sslope_1whisk_ALL

delta_1vs2_overlap = [delta_1vs2[i,channels[i]] for i in range(5)]
freq_1vs2_overlap = [freq_1vs2[i,channels[i]] for i in range(5)]
dur_1vs2_overlap = [dur_1vs2[i,channels[i]] for i in range(5)]
famp_1vs2_overlap = [famp_1vs2[i,channels[i]] for i in range(5)]
samp_1vs2_overlap = [samp_1vs2[i,channels[i]] for i in range(5)]
fslope_1vs2_overlap = [fslope_1vs2[i,channels[i]] for i in range(5)]
sslope_1vs2_overlap = [sslope_1vs2[i,channels[i]] for i in range(5)]


# average of average wavefom of channels
dur_1vs2_avg_overlap = np.zeros([5])
famp_1vs2_avg_overlap = np.zeros([5])
samp_1vs2_avg_overlap = np.zeros([5])
fslope_1vs2_avg_overlap = np.zeros([5])
sslope_1vs2_avg_overlap = np.zeros([5])

# dur_1vs2_avg_overlap = np.zeros([5])
# famp_1vs2_avg_overlap = np.zeros([5])
# samp_1vs2_avg_overlap = np.zeros([5])
# fslope_1vs2_avg_overlap = np.zeros([5])
# sslope_1vs2_avg_overlap = np.zeros([5])


for mouse in range(5):
    SW_avg_waveform_unpaired = np.mean(SW_waveform_2whisk_ALL[mouse,channels[mouse],:], axis = 0)
    UP_peak_unpaired = scipy.signal.find_peaks(-SW_avg_waveform_unpaired)[0][0]
    DOWN_peak_unpaired = scipy.signal.find_peaks(SW_avg_waveform_unpaired)[0]
    DOWN_peak_unpaired = DOWN_peak_unpaired[DOWN_peak_unpaired > UP_peak_unpaired][0]
    SW_avg_waveform_paired = np.mean(SW_waveform_1whisk_ALL[mouse,channels[mouse],:], axis = 0)
    UP_peak_paired = scipy.signal.find_peaks(-SW_avg_waveform_paired)[0][0]
    DOWN_peak_paired = scipy.signal.find_peaks(SW_avg_waveform_paired)[0]
    DOWN_peak_paired = DOWN_peak_paired[DOWN_peak_paired > UP_peak_paired][0]
    
    fig, ax = plt.subplots()
    ax.plot(SW_avg_waveform_unpaired, color = 'r')
    ax.plot(SW_avg_waveform_paired, color = 'k')
    
    dur_1vs2_avg_overlap[mouse] = ((UP_peak_unpaired - DOWN_peak_unpaired) - (UP_peak_paired - DOWN_peak_paired))/(UP_peak_paired - DOWN_peak_paired)
    famp_1vs2_avg_overlap[mouse] = ((SW_avg_waveform_unpaired[UP_peak_unpaired] - SW_avg_waveform_unpaired[0]) - (SW_avg_waveform_paired[UP_peak_paired] - SW_avg_waveform_paired[0]))/(SW_avg_waveform_paired[UP_peak_paired] -  SW_avg_waveform_paired[0])
    samp_1vs2_avg_overlap[mouse] = ((SW_avg_waveform_unpaired[DOWN_peak_unpaired] - SW_avg_waveform_unpaired[0]) - (SW_avg_waveform_paired[DOWN_peak_paired] - SW_avg_waveform_paired[0]))/(SW_avg_waveform_paired[DOWN_peak_paired] - SW_avg_waveform_paired[0])
    fslope_1vs2_avg_overlap[mouse] = (np.nanmean(np.diff(SW_avg_waveform_unpaired[0:UP_peak_unpaired])) - np.nanmean(np.diff(SW_avg_waveform_paired[0:UP_peak_paired])))/np.nanmean(np.diff(SW_avg_waveform_paired[0:UP_peak_paired]))
    sslope_1vs2_avg_overlap[mouse] = (np.nanmean(np.diff(SW_avg_waveform_unpaired[UP_peak_unpaired:DOWN_peak_unpaired])) - np.nanmean(np.diff(SW_avg_waveform_paired[UP_peak_paired:DOWN_peak_paired])))/np.nanmean(np.diff(SW_avg_waveform_paired[UP_peak_paired:DOWN_peak_paired]))
    
    
# #average change in slow wave characteristics across mice
# # if average change in channels
# to_plot_mice = [delta_1vs2_overlap, freq_1vs2_overlap, dur_1vs2_overlap, fslope_1vs2_overlap, sslope_1vs2_overlap, famp_1vs2_overlap, samp_1vs2_overlap]
# #average within each mouse in percentage
# to_plot_mice = [[np.nanmean(j)*100 for j in i] for i in to_plot_mice]

# change of average waveform
to_plot_mice = [delta_1vs2_overlap, freq_1vs2_overlap, dur_1vs2_avg_overlap, fslope_1vs2_avg_overlap, sslope_1vs2_avg_overlap, famp_1vs2_avg_overlap, samp_1vs2_avg_overlap]
to_plot_mice = [[np.nanmean(j)*100 for j in i] for i in to_plot_mice]

fig, ax = plt.subplots()
ax.boxplot(to_plot_mice, showfliers = True, notch = True)
ax.set_xticklabels(['dpower','freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'], size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([-25,0,25])
# ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 14)

fig, ax = plt.subplots()
ax.boxplot(to_plot_mice, showfliers = True, notch = True)
ax.set_xticklabels(['dpower','freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'], size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([-25,0,25])



for i in range(7):
    measure_to_test = to_plot_mice[i]
    print(scipy.stats.shapiro(measure_to_test))
    np.mean(measure_to_test)
    #multiple comparisons
    print(scipy.stats.ttest_1samp(measure_to_test,0)[1]*6)


# # plot SW waveform of paired vs unpaired
# mouse_to_plot = 2
# fig, ax = plt.subplots(8, 8, sharey=True)
# for ind in range(64):
#     ax.flatten()[ind].plot(SW_waveform_1whisk_ALL[mouse_to_plot,chanMap[ind],:], c = 'r')
#     ax.flatten()[ind].plot(SW_waveform_2whisk_ALL[mouse_to_plot,chanMap[ind],:], c = 'k')
#     ax.flatten()[ind].set_title(str(chanMap[ind]))
#     if chanMap[ind] in channels[mouse_to_plot]:
#         ax.flatten()[ind].set_facecolor("y")   


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# correlation LFP change in response to unpaired/paired vs delta power change during baseline
take_out_outliers = False
channels = LFP_resp_channels_overlap_whisk_for_delta
LFP_resp_difference_overlap = np.mean(LFP_min_2whisk_ALL[:,sweeps_for_SW_compare,:], axis = 1)/((np.mean(LFP_min_1whisk_ALL[:,sweeps_for_SW_compare,:], axis = 1)) + np.mean(LFP_min_2whisk_ALL[:,sweeps_for_SW_compare,:], axis = 1))
# LFP_resp_difference_overlap = (np.mean(LFP_min_1whisk_ALL[:,[0,1,2,3],:], axis = 1) - np.mean(LFP_min_2whisk_ALL[:,[0,1,2,3],:], axis = 1))/np.mean(LFP_min_2whisk_ALL[:,[0,1,2,3],:], axis = 1)
# LFP_resp_difference_overlap = (np.mean(LFP_min_2whisk_ALL[:,[0,1,2,3],:], axis = 1)/np.mean(LFP_min_1whisk_ALL[:,[0,1,2,3],:], axis = 1))
LFP_resp_difference_overlap = [LFP_resp_difference_overlap[i,channels[i]] for i in range(5)]

X = np.concatenate(LFP_resp_difference_overlap)
Y = np.concatenate([delta_1vs2[i,channels[i]] for i in range(5)])*100
if take_out_outliers == True:
    outliers_mask = np.any(np.vstack([np.logical_or((i > (np.percentile(i, 75) + 1.5*(np.abs(np.percentile(i, 75) - np.percentile(i, 25))))), (i < (np.percentile(i, 75) - 1.5*(np.abs(np.percentile(i, 75) - np.percentile(i, 25)))))) for i in [Y]]) == True, axis = 0)
    X = X[~outliers_mask]
    Y = Y[~outliers_mask]
    
fig,ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(X, Y)
print(f'{r} and {p} for {len(X)} channels')
ax.scatter(X,Y, color = 'k')
ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
ax.set_xlabel('B2 whisker response (% D1 whisker response)', size = 16)
ax.set_ylabel('delta power during B2 deflections \n (% during D1 deflections)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('delta vs unpaired whisker.pdf', dpi = 1000, format = 'pdf')
plt.savefig('delta vs unpaired whiske.jpg', dpi = 1000, format = 'jpg')




