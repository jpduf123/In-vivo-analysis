# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:00:18 2023

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


overall_path = r'C:\One_Drive\OneDrive\Dokumente\SWS\FOR_ANALYSIS'


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


#%% ----------------------------------------------- Save values from all mice into arrays

to_plot_1_LFP = [0,1,2,3]
to_plot_2_LFP = [4,5,6,7,8,9]
baseline_sweeps = [0,1,2,3]

lfp_cutoff_resp_channels = 200

mean_or_median = 'median'

# ------------------------------------------------------------------------ unpaired whisker:

os.chdir(os.path.join(overall_path, r'UNPAIRED_whisker'))
days_2whisk = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

#groups arrays for all mice with unpaired whisker stims:
LFP_min_2whisk_ALL = np.zeros([len(days_2whisk),10,64])
LFP_min_rel_2whisk_ALL = np.zeros([len(days_2whisk),10,64])
LFP_min_rel_change_2whisk_ALL = np.zeros([len(days_2whisk),64])

delta_2whisk_ALL = np.zeros([len(days_2whisk),10,64])
delta_2whisk_median_ALL = np.zeros([len(days_2whisk),10,64])
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
PSD_2whisk_ALL = []

LFP_resp_channels_cutoff_2whisk_list = []

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

    LFP_min_2whisk_ALL[day_ind, :, :] = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_min_rel_2whisk_ALL[day_ind, :, :] = LFP_min_2whisk_ALL[day_ind, :, :]/np.nanmean(LFP_min_2whisk_ALL[day_ind, baseline_sweeps, :], axis = 0)
    LFP_min_rel_change_2whisk_ALL[day_ind, :] = np.loadtxt('LFP_min_rel_change.csv', delimiter = ',')

    delta_2whisk_ALL[day_ind, :, :] = np.loadtxt('delta_power_auto_outliers.csv', delimiter = ',')
    delta_2whisk_median_ALL[day_ind, :, :] = np.loadtxt('delta_power_median_auto_outliers.csv', delimiter = ',')
    delta_rel_2whisk_ALL[day_ind, :, :] = delta_2whisk_ALL[day_ind, :, :]/np.nanmean(delta_2whisk_ALL[day_ind, baseline_sweeps, :], axis = 0)
    delta_rel_change_2whisk_ALL[day_ind, :] = np.loadtxt('delta_power_rel_change.csv', delimiter = ',')
    
    SW_freq_2whisk_ALL[day_ind, :] = np.mean(np.load('SW_frequency_sweeps_avg.npy')[baseline_sweeps,:], axis = 0)
    SW_dur_2whisk_ALL[day_ind, :] = np.mean(np.load(f'Peak_dur_sweeps_{mean_or_median}.npy')[baseline_sweeps,:], axis = 0)
    SW_famp_2whisk_ALL[day_ind, :] = np.mean(np.load(f'SW_famp_sweeps_{mean_or_median}.npy')[baseline_sweeps,:], axis = 0)
    SW_samp_2whisk_ALL[day_ind, :] = np.mean(np.load(f'SW_samp_sweeps_{mean_or_median}.npy')[baseline_sweeps,:], axis = 0)
    SW_fslope_2whisk_ALL[day_ind, :] = np.mean(np.load(f'SW_fslope_sweeps_{mean_or_median}.npy')[baseline_sweeps,:], axis = 0)
    SW_sslope_2whisk_ALL[day_ind, :] = np.mean(np.load(f'SW_sslope_sweeps_{mean_or_median}.npy')[baseline_sweeps,:], axis = 0)
    
    SW_waveform_2whisk_ALL[day_ind, :] = np.mean(np.load(f'SW_waveform_sweeps_{mean_or_median}.npy')[baseline_sweeps,:,:], axis = 0)
    
    # extract baseline PSD up to 50Hz 
    PSD_2whisk_ALL.append(np.load('PSD.npy')[[0,1,2,3],:,:][:,:,np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]])
    
    os.chdir('..')
    os.chdir('..')




# ------------------------------------------------------------------------ paired whisker:
os.chdir(os.path.join(overall_path, r'UP_pairing'))
days_1whisk = ['160414_D1', '160426_D1', '160519_B2', '160624_B2', '160628_D1']

LFP_min_1whisk_ALL = np.zeros([len(days_1whisk),10,64])
LFP_min_rel_1whisk_ALL = np.zeros([len(days_1whisk),10,64])
LFP_min_rel_change_1whisk_ALL = np.zeros([len(days_1whisk),64])

delta_1whisk_ALL = np.zeros([len(days_2whisk),10,64])
delta_1whisk_median_ALL = np.zeros([len(days_2whisk),10,64])
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
PSD_1whisk_ALL = []

LFP_resp_channels_cutoff_1whisk_list = []

for day_ind, day in enumerate(days_1whisk):
    os.chdir(day)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_resp_channels_cutoff = np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
    # LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',').astype(int)
    PSTH_resp_channels = np.loadtxt('PSTH_resp_channels.csv', delimiter = ',', dtype = int)

    # if day == '160426_1':
    #     LFP_bad = [36,20] # noisy
    #     for chan in LFP_bad:  
    #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
    # if day == '160519_B2':
    #     LFP_bad = [27,20] # noisy
    #     for chan in LFP_bad:  
    #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
    # if day == '160624_B2':
    #     LFP_bad = [21,19] # noisy
    #     for chan in LFP_bad:  
    #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
    # if day == '160628_D1':
    #     LFP_bad = [37,35,19,17] # noisy
    #     for chan in LFP_bad:  
    #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 

    LFP_resp_channels_cutoff_1whisk_list.append(LFP_resp_channels_cutoff)
    
    LFP_min_1whisk_ALL[day_ind, :, :] = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_min_rel_1whisk_ALL[day_ind, :, :] = LFP_min_1whisk_ALL[day_ind, :, :]/np.nanmean(LFP_min_1whisk_ALL[day_ind, baseline_sweeps, :], axis = 0)
    LFP_min_rel_change_1whisk_ALL[day_ind, :] = np.loadtxt('LFP_min_rel_change.csv', delimiter = ',')
        
    delta_1whisk_ALL[day_ind, :, :] = np.loadtxt('delta_power_auto_outliers.csv', delimiter = ',')
    delta_1whisk_median_ALL[day_ind, :, :] = np.loadtxt('delta_power_median_auto_outliers.csv', delimiter = ',')
    delta_rel_1whisk_ALL[day_ind, :, :] = delta_1whisk_ALL[day_ind, :, :]/np.nanmean(delta_1whisk_ALL[day_ind, baseline_sweeps, :], axis = 0)
    delta_rel_change_1whisk_ALL[day_ind, :] = np.loadtxt('delta_power_rel_change.csv', delimiter = ',')
    
    SW_freq_1whisk_ALL[day_ind, :] = np.mean(np.load('SW_frequency_sweeps_avg.npy')[baseline_sweeps,:], axis = 0)
    SW_dur_1whisk_ALL[day_ind, :] = np.mean(np.load(f'Peak_dur_sweeps_{mean_or_median}.npy')[baseline_sweeps,:], axis = 0)
    SW_famp_1whisk_ALL[day_ind, :] = np.mean(np.load(f'SW_famp_sweeps_{mean_or_median}.npy')[baseline_sweeps,:], axis = 0)
    SW_samp_1whisk_ALL[day_ind, :] = np.mean(np.load(f'SW_samp_sweeps_{mean_or_median}.npy')[baseline_sweeps,:], axis = 0)
    SW_fslope_1whisk_ALL[day_ind, :] = np.mean(np.load(f'SW_fslope_sweeps_{mean_or_median}.npy')[baseline_sweeps,:], axis = 0)
    SW_sslope_1whisk_ALL[day_ind, :] = np.mean(np.load(f'SW_sslope_sweeps_{mean_or_median}.npy')[baseline_sweeps,:], axis = 0)
    
    SW_waveform_1whisk_ALL[day_ind, :] = np.mean(np.load(f'SW_waveform_sweeps_{mean_or_median}.npy')[baseline_sweeps,:,:], axis = 0)

    PSD_1whisk_ALL.append(np.load('PSD.npy')[[0,1,2,3],:,:][:,:,np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]])

    os.chdir('..')
    os.chdir('..')

# channels responsive to any, both, or only unpaired
LFP_resp_channels_unique_whisk = [np.unique(np.hstack((LFP_resp_channels_cutoff_2whisk_list[i], LFP_resp_channels_cutoff_1whisk_list[i]))) for i in range(5)] # responsive to all
LFP_resp_channels_overlap_whisk = [np.intersect1d(LFP_resp_channels_cutoff_2whisk_list[i], LFP_resp_channels_cutoff_1whisk_list[i]) for i in range(5)] # overlap
LFP_resp_channels_nonoverlap_2whisk = [LFP_resp_channels_cutoff_2whisk_list[i][~np.isin(LFP_resp_channels_cutoff_2whisk_list[i], LFP_resp_channels_cutoff_1whisk_list[i])] for i in range(5)]



#%%
sweeps_for_SW_compare = [0,1,2,3]

# update overlapping channels with some good ones that are just below detection threshold (or else one bad delta power channel can fuck up the whole average)
# LFP_resp_channels_overlap_whisk_for_delta[3] = np.append(LFP_resp_channels_overlap_whisk_for_delta[3], [15,33,35,17,61])
# LFP_resp_channels_overlap_whisk_for_delta[4] = np.append(LFP_resp_channels_overlap_whisk_for_delta[4], [23,25,29])   

# take out noisy channels
LFP_resp_channels_overlap_whisk_for_delta = copy.deepcopy(LFP_resp_channels_overlap_whisk)
# LFP_resp_channels_overlap_whisk_for_delta[2] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[2], np.where(LFP_resp_channels_overlap_whisk_for_delta[2] == 27)[0]) 
# LFP_resp_channels_overlap_whisk_for_delta[2] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[2], np.where(LFP_resp_channels_overlap_whisk_for_delta[2] == 20)[0]) 

# LFP_resp_channels_overlap_whisk_for_delta[3] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[3], np.where(LFP_resp_channels_overlap_whisk_for_delta[3] == 21)[0]) 
# LFP_resp_channels_overlap_whisk_for_delta[3] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[3], np.where(LFP_resp_channels_overlap_whisk_for_delta[3] == 19)[0]) 

# LFP_resp_channels_overlap_whisk_for_delta[4] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[4], np.where(LFP_resp_channels_overlap_whisk_for_delta[4] == 39)[0])   
# LFP_resp_channels_overlap_whisk_for_delta[4] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[4], np.where(LFP_resp_channels_overlap_whisk_for_delta[4] == 37)[0])   
# LFP_resp_channels_overlap_whisk_for_delta[4] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[4], np.where(LFP_resp_channels_overlap_whisk_for_delta[4] == 35)[0])   
# # LFP_resp_channels_overlap_whisk_for_delta[4] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[4], np.where(LFP_resp_channels_overlap_whisk_for_delta[4] == 19)[0])   
# # LFP_resp_channels_overlap_whisk_for_delta[4] = np.delete(LFP_resp_channels_overlap_whisk_for_delta[4], np.where(LFP_resp_channels_overlap_whisk_for_delta[4] == 17)[0])   



LFP_resp_channels_unique_whisk_for_delta = copy.deepcopy(LFP_resp_channels_unique_whisk)
# LFP_resp_channels_unique_whisk_for_delta[0] = np.delete(LFP_resp_channels_unique_whisk_for_delta[0], np.where(LFP_resp_channels_unique_whisk_for_delta[0] == 7)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[0] = np.delete(LFP_resp_channels_unique_whisk_for_delta[0], np.where(LFP_resp_channels_unique_whisk_for_delta[0] == 11)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[0] = np.delete(LFP_resp_channels_unique_whisk_for_delta[0], np.where(LFP_resp_channels_unique_whisk_for_delta[0] == 39)[0]) 

# LFP_resp_channels_unique_whisk_for_delta[1] = np.delete(LFP_resp_channels_unique_whisk_for_delta[1], np.where(LFP_resp_channels_unique_whisk_for_delta[1] == 33)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[1] = np.delete(LFP_resp_channels_unique_whisk_for_delta[1], np.where(LFP_resp_channels_unique_whisk_for_delta[1] == 14)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[1] = np.delete(LFP_resp_channels_unique_whisk_for_delta[1], np.where(LFP_resp_channels_unique_whisk_for_delta[1] == 36)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[1] = np.delete(LFP_resp_channels_unique_whisk_for_delta[1], np.where(LFP_resp_channels_unique_whisk_for_delta[1] == 20)[0]) 

# LFP_resp_channels_unique_whisk_for_delta[2] = np.delete(LFP_resp_channels_unique_whisk_for_delta[2], np.where(LFP_resp_channels_unique_whisk_for_delta[2] == 55)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[2] = np.delete(LFP_resp_channels_unique_whisk_for_delta[2], np.where(LFP_resp_channels_unique_whisk_for_delta[2] == 58)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[2] = np.delete(LFP_resp_channels_unique_whisk_for_delta[2], np.where(LFP_resp_channels_unique_whisk_for_delta[2] == 20)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[2] = np.delete(LFP_resp_channels_unique_whisk_for_delta[2], np.where(LFP_resp_channels_unique_whisk_for_delta[2] == 27)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[2] = np.delete(LFP_resp_channels_unique_whisk_for_delta[2], np.where(LFP_resp_channels_unique_whisk_for_delta[2] == 36)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[2] = np.delete(LFP_resp_channels_unique_whisk_for_delta[2], np.where(LFP_resp_channels_unique_whisk_for_delta[2] == 46)[0]) 

# LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 15)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 17)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 19)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 21)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 35)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 37)[0]) 
# LFP_resp_channels_unique_whisk_for_delta[3] = np.delete(LFP_resp_channels_unique_whisk_for_delta[3], np.where(LFP_resp_channels_unique_whisk_for_delta[3] == 39)[0]) 

# LFP_resp_channels_unique_whisk_for_delta[4] = np.delete(LFP_resp_channels_unique_whisk_for_delta[4], np.where(LFP_resp_channels_unique_whisk_for_delta[4] == 39)[0])   
# LFP_resp_channels_unique_whisk_for_delta[4] = np.delete(LFP_resp_channels_unique_whisk_for_delta[4], np.where(LFP_resp_channels_unique_whisk_for_delta[4] == 37)[0])   
# LFP_resp_channels_unique_whisk_for_delta[4] = np.delete(LFP_resp_channels_unique_whisk_for_delta[4], np.where(LFP_resp_channels_unique_whisk_for_delta[4] == 35)[0])   
# LFP_resp_channels_unique_whisk_for_delta[4] = np.delete(LFP_resp_channels_unique_whisk_for_delta[4], np.where(LFP_resp_channels_unique_whisk_for_delta[4] == 19)[0])   
# LFP_resp_channels_unique_whisk_for_delta[4] = np.delete(LFP_resp_channels_unique_whisk_for_delta[4], np.where(LFP_resp_channels_unique_whisk_for_delta[4] == 17)[0])   



channels = LFP_resp_channels_overlap_whisk_for_delta

# change within each mouse, average of change in overlapping channels
delta_1vs2 = (np.mean(delta_1whisk_median_ALL[:,sweeps_for_SW_compare,:], axis = 1) - np.mean(delta_2whisk_median_ALL[:,sweeps_for_SW_compare,:], axis = 1))/np.mean(delta_2whisk_median_ALL[:,sweeps_for_SW_compare,:], axis = 1)
freq_1vs2 = (SW_freq_1whisk_ALL - SW_freq_2whisk_ALL)/SW_freq_2whisk_ALL
dur_1vs2 = (SW_dur_1whisk_ALL - SW_dur_2whisk_ALL)/SW_dur_2whisk_ALL
famp_1vs2 = (SW_famp_1whisk_ALL - SW_famp_2whisk_ALL)/SW_famp_2whisk_ALL
samp_1vs2 = (SW_samp_1whisk_ALL - SW_samp_2whisk_ALL)/SW_samp_2whisk_ALL
fslope_1vs2 = (SW_fslope_1whisk_ALL - SW_fslope_2whisk_ALL)/SW_fslope_2whisk_ALL
sslope_1vs2 = (SW_sslope_1whisk_ALL - SW_sslope_2whisk_ALL)/SW_sslope_2whisk_ALL

delta_1vs2_overlap = [delta_1vs2[i,channels[i]] for i in range(5)]
freq_1vs2_overlap = [freq_1vs2[i,channels[i]] for i in range(5)]
dur_1vs2_overlap = [dur_1vs2[i,channels[i]] for i in range(5)]
famp_1vs2_overlap = [famp_1vs2[i,channels[i]] for i in range(5)]
samp_1vs2_overlap = [samp_1vs2[i,channels[i]] for i in range(5)]
fslope_1vs2_overlap = [fslope_1vs2[i,channels[i]] for i in range(5)]
sslope_1vs2_overlap = [sslope_1vs2[i,channels[i]] for i in range(5)]



# # average of average wavefom of channels
# dur_1vs2_avg_overlap = np.zeros([5])
# famp_1vs2_avg_overlap = np.zeros([5])
# samp_1vs2_avg_overlap = np.zeros([5])
# fslope_1vs2_avg_overlap = np.zeros([5])
# sslope_1vs2_avg_overlap = np.zeros([5])

# # dur_1vs2_avg_overlap = np.zeros([5])
# # famp_1vs2_avg_overlap = np.zeros([5])
# # samp_1vs2_avg_overlap = np.zeros([5])
# # fslope_1vs2_avg_overlap = np.zeros([5])
# # sslope_1vs2_avg_overlap = np.zeros([5])


# for mouse in range(5):
#     SW_avg_waveform_unpaired = np.mean(SW_waveform_2whisk_ALL[mouse,channels[mouse],:], axis = 0)
#     UP_peak_unpaired = scipy.signal.find_peaks(-SW_avg_waveform_unpaired)[0][0]
#     DOWN_peak_unpaired = scipy.signal.find_peaks(SW_avg_waveform_unpaired)[0]
#     DOWN_peak_unpaired = DOWN_peak_unpaired[DOWN_peak_unpaired > UP_peak_unpaired][0]
#     SW_avg_waveform_paired = np.mean(SW_waveform_1whisk_ALL[mouse,channels[mouse],:], axis = 0)
#     UP_peak_paired = scipy.signal.find_peaks(-SW_avg_waveform_paired)[0][0]
#     DOWN_peak_paired = scipy.signal.find_peaks(SW_avg_waveform_paired)[0]
#     DOWN_peak_paired = DOWN_peak_paired[DOWN_peak_paired > UP_peak_paired][0]
    
#     fig, ax = plt.subplots()
#     ax.plot(SW_avg_waveform_unpaired, color = 'r')
#     ax.plot(SW_avg_waveform_paired, color = 'k')
    
#     dur_1vs2_avg_overlap[mouse] = ((UP_peak_unpaired - DOWN_peak_unpaired) - (UP_peak_paired - DOWN_peak_paired))/(UP_peak_paired - DOWN_peak_paired)
#     famp_1vs2_avg_overlap[mouse] = ((SW_avg_waveform_unpaired[UP_peak_unpaired] - SW_avg_waveform_unpaired[0]) - (SW_avg_waveform_paired[UP_peak_paired] - SW_avg_waveform_paired[0]))/(SW_avg_waveform_paired[UP_peak_paired] -  SW_avg_waveform_paired[0])
#     samp_1vs2_avg_overlap[mouse] = ((SW_avg_waveform_unpaired[DOWN_peak_unpaired] - SW_avg_waveform_unpaired[0]) - (SW_avg_waveform_paired[DOWN_peak_paired] - SW_avg_waveform_paired[0]))/(SW_avg_waveform_paired[DOWN_peak_paired] - SW_avg_waveform_paired[0])
#     fslope_1vs2_avg_overlap[mouse] = (np.nanmean(np.diff(SW_avg_waveform_unpaired[0:UP_peak_unpaired])) - np.nanmean(np.diff(SW_avg_waveform_paired[0:UP_peak_paired])))/np.nanmean(np.diff(SW_avg_waveform_paired[0:UP_peak_paired]))
#     sslope_1vs2_avg_overlap[mouse] = (np.nanmean(np.diff(SW_avg_waveform_unpaired[UP_peak_unpaired:DOWN_peak_unpaired])) - np.nanmean(np.diff(SW_avg_waveform_paired[UP_peak_paired:DOWN_peak_paired])))/np.nanmean(np.diff(SW_avg_waveform_paired[UP_peak_paired:DOWN_peak_paired]))
    
# # #average change in slow wave characteristics across mice
# # # if average change in channels
# # to_plot_mice = [delta_1vs2_overlap, freq_1vs2_overlap, dur_1vs2_overlap, fslope_1vs2_overlap, sslope_1vs2_overlap, famp_1vs2_overlap, samp_1vs2_overlap]
# # #average within each mouse in percentage
# # to_plot_mice = [[np.nanmean(j)*100 for j in i] for i in to_plot_mice]

# # change of average waveform
# to_plot_mice = [delta_1vs2_overlap, freq_1vs2_overlap, dur_1vs2_avg_overlap, fslope_1vs2_avg_overlap, sslope_1vs2_avg_overlap, famp_1vs2_avg_overlap, samp_1vs2_avg_overlap]
# to_plot_mice = [[np.nanmean(j)*100 for j in i] for i in to_plot_mice]


to_plot_mice = [delta_1vs2_overlap, freq_1vs2_overlap, dur_1vs2_overlap, fslope_1vs2_overlap, sslope_1vs2_overlap, famp_1vs2_overlap, samp_1vs2_overlap]
to_plot_mice = [[np.nanmedian(j)*100 for j in i] for i in to_plot_mice] #average across channels
for ind, curr_param in enumerate(to_plot_mice):
    outliers = (np.logical_or(curr_param > (np.percentile(curr_param, 75) + 1.5*(np.abs(np.percentile(curr_param, 75) - np.percentile(curr_param, 25)))), curr_param < (np.percentile(curr_param, 25) - 1.5*(np.abs(np.percentile(curr_param, 75) - np.percentile(curr_param, 25))))))
    # print(outliers)
    to_plot_mice[ind] = np.delete(curr_param, outliers)
fig, ax = plt.subplots()
ax.boxplot(to_plot_mice[1:], showfliers = True, notch = False)
ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'], size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([-10,0,10])
ax.set_ylim([-15,15])
ax.tick_params(axis="y", labelsize=16)    
ax.set_ylabel(' change during B2 vs D1 \n deflections at baseline (%)', size = 16)
plt.tight_layout()
plt.savefig('SW params paired vs unpaired.pdf', dpi = 1000, format = 'pdf')
plt.savefig('SW params paired vs unpaired.jpg', dpi = 1000, format = 'jpg')

# ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 14)

# fig, ax = plt.subplots()
# ax.boxplot(to_plot_mice, showfliers = True, notch = True)
# ax.set_xticklabels(['dpower','freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'], size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_yticks([-25,0,25])

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


#%%
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# correlation LFP change in response to unpaired/paired vs delta power change during baseline
take_out_outliers = True
channels = LFP_resp_channels_overlap_whisk_for_delta

LFP_resp_difference_overlap = np.mean(LFP_min_2whisk_ALL[:,sweeps_for_SW_compare,:], axis = 1)/((np.mean(LFP_min_1whisk_ALL[:,sweeps_for_SW_compare,:], axis = 1)) + np.mean(LFP_min_2whisk_ALL[:,sweeps_for_SW_compare,:], axis = 1))*100
# LFP_resp_difference_overlap = (np.mean(LFP_min_1whisk_ALL[:,[0,1,2,3],:], axis = 1) - np.mean(LFP_min_2whisk_ALL[:,[0,1,2,3],:], axis = 1))/np.mean(LFP_min_2whisk_ALL[:,[0,1,2,3],:], axis = 1)
# LFP_resp_difference_overlap = (np.mean(LFP_min_2whisk_ALL[:,[0,1,2,3],:], axis = 1)/np.mean(LFP_min_1whisk_ALL[:,[0,1,2,3],:], axis = 1))
LFP_resp_difference_overlap = [LFP_resp_difference_overlap[i,channels[i]] for i in range(5)]

X = np.concatenate(LFP_resp_difference_overlap)
Y = np.concatenate([delta_1vs2[i,channels[i]] for i in range(5)])*100
if take_out_outliers == True:
    outliers_mask = np.any(np.vstack([np.logical_or((i > (np.percentile(i, 75) + 1.5*(np.abs(np.percentile(i, 75) - np.percentile(i, 25))))), (i < (np.percentile(i, 25) - 1.5*(np.abs(np.percentile(i, 75) - np.percentile(i, 25)))))) for i in [Y]]) == True, axis = 0)
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
ax.set_yticks([-30,-20,-10,0,10,20,30])
ax.set_ylim([-31,31])
ax.tick_params(axis="x", labelsize=16)  
ax.tick_params(axis="y", labelsize=16)    
plt.tight_layout()
plt.savefig('delta vs unpaired whisker.pdf', dpi = 1000, format = 'pdf')
plt.savefig('delta vs unpaired whisker.jpg', dpi = 1000, format = 'jpg')
