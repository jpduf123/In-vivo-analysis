# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:48:09 2022

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

overall_path = r'C:\One_Drive\OneDrive\Dokumente\SWS\FOR_ANALYSIS'
figure_path = r'C:\One_Drive\OneDrive\Dokumente\SWS\Figures'


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


def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = mean(d1), mean(d2)
	# calculate the effect size
	return (u1 - u2) / s



#%% LOAD DATA IN ARRAYS FOR UP AND DOWN CONDITIONS AND SAVE (old)

lfp_cutoff_resp_channels = 200

for cond in ['UP_pairing', 'DOWN_pairing']:
# for cond in ['UP_pairing']:

    os.chdir(os.path.join(overall_path, cond))
    days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
    numb = len(days)
    
    # average timecourse across channels in each mouse
    LFP_min_ALL, LFP_min_rel_ALL, LFP_slope_rel_ALL, LFP_time_to_peak_ALL = (np.zeros([numb,10]) for i in range(4))
    PSTH_resp_magn_ALL, PSTH_resp_magn_rel_ALL, PSTH_resp_peak_ALL, PSTH_resp_peak_rel_ALL = (np.zeros([numb,10]) for i in range(4))
    delta_power_ALL, delta_power_rel_ALL, delta_power_auto_outliers_ALL, delta_power_auto_outliers_rel_ALL = (np.zeros([numb,10]) for i in range(4))
    # delta_power_median_auto_outliers_ALL, delta_power_median_auto_outliers_rel_ALL = (np.zeros([numb,10]) for i in range(2))

    # median timecourse across channels in each mouse
    LFP_min_median_ALL, LFP_min_rel_median_ALL, LFP_slope_rel_median_ALL, LFP_slope_rel_not_outliers_median_ALL = (np.zeros([numb,10]) for i in range(4)) # LFP time to peak already medianed
    PSTH_resp_magn_median_ALL, PSTH_resp_magn_rel_median_ALL, PSTH_resp_peak_median_ALL, PSTH_resp_peak_rel_median_ALL = (np.zeros([numb,10]) for i in range(4))
    delta_power_median_ALL, delta_power_rel_median_ALL, delta_power_auto_outliers_median_ALL, delta_power_auto_outliers_rel_median_ALL = (np.zeros([numb,10]) for i in range(4))
    # delta_power_median_auto_outliers_median_ALL, delta_power_median_auto_outliers_rel_median_ALL = (np.zeros([numb,10]) for i in range(2))

    #list of PSD for every sweep and mouse
    PSD_ALL = [] # mean across sweeps
    PSD_median_ALL = [] # median across sweeps
    
    # responsive channels
    LFP_resp_channels_cutoff_ALLCHANS, PSTH_resp_channels_ALLCHANS, SW_spiking_channels_ALLCHANS = (np.zeros([numb*64]) for i in range(3))
    
    # change in all chans of every mouse in one long array
    LFP_min_rel_change_ALLCHANS, PSTH_resp_magn_rel_change_ALLCHANS, PSTH_resp_peak_rel_change_ALLCHANS, LFP_slope_rel_change_ALLCHANS, LFP_time_to_peak_change_ALLCHANS, delta_power_rel_change_ALLCHANS, delta_power_auto_outliers_rel_change_ALLCHANS, delta_power_median_auto_outliers_rel_change_ALLCHANS  = (np.zeros([numb*64]) for i in range(8))
    Freq_change_ALLCHANS, Peak_dur_change_mean_ALLCHANS, Fslope_change_mean_ALLCHANS, Sslope_change_mean_ALLCHANS, Famp_change_mean_ALLCHANS, Samp_change_mean_ALLCHANS = (np.zeros([numb*64]) for i in range(6))
    Peak_dur_change_median_ALLCHANS, Fslope_change_median_ALLCHANS, Sslope_change_median_ALLCHANS, Famp_change_median_ALLCHANS, Samp_change_median_ALLCHANS = (np.zeros([numb*64]) for i in range(5))
    Peak_dur_overall_change_ALLCHANS, Fslope_overall_change_ALLCHANS, Sslope_overall_change_ALLCHANS, Famp_overall_change_ALLCHANS, Samp_overall_change_ALLCHANS = (np.zeros([numb*64]) for i in range(5))
    SW_spiking_area_change_ALLCHANS, SW_spiking_peak_change_ALLCHANS = (np.zeros([numb*64]) for i in range(2))
    AUC_first_stims_ALLCHANS, AUC_all_stims_ALLCHANS, pairing_UP_freq_first_stims_ALLCHANS = (np.zeros([numb*64]) for i in range(3))
    
    # UP vs DOWN stim delivery
    LFP_min_UP_ALL, LFP_min_DOWN_ALL, LFP_min_UP_rel_ALL, LFP_min_DOWN_rel_ALL = (np.zeros([numb,10]) for i in range(4))
    PSTH_magn_UP_ALL, PSTH_magn_DOWN_ALL, PSTH_magn_UP_rel_ALL, PSTH_magn_DOWN_rel_ALL, PSTH_peak_UP_ALL, PSTH_peak_DOWN_ALL, PSTH_peak_UP_rel_ALL, PSTH_peak_DOWN_rel_ALL = (np.zeros([numb,10]) for i in range(8))
    LFP_min_UP_median_ALL, LFP_min_DOWN_median_ALL, LFP_min_UP_rel_median_ALL, LFP_min_DOWN_rel_median_ALL = (np.zeros([numb,10]) for i in range(4))
    PSTH_magn_UP_median_ALL, PSTH_magn_DOWN_median_ALL, PSTH_magn_UP_rel_median_ALL, PSTH_magn_DOWN_rel_median_ALL, PSTH_peak_UP_median_ALL, PSTH_peak_DOWN_median_ALL, PSTH_peak_UP_rel_median_ALL, PSTH_peak_DOWN_rel_median_ALL = (np.zeros([numb,10]) for i in range(8))


    
    # all timecourse arrays with one value per mouse, and change arrays with one value per channel: set to NaN so you can exclude mice/channel if needed (e.g. spikes)
    overall_arrays = [i for i in list(globals().keys()) if 'ALL' in i and 'distance_vs_plasticity' not in i and 'nostim' not in i and 'PSD_lastpre' not in i]
    for array in overall_arrays:
        if array in ['PSD_ALL', 'PSD_median_ALL', 'LFP_resp_channels_cutoff_ALLCHANS', 'PSTH_resp_channels_ALLCHANS', 'SW_spiking_channels_ALLCHANS']:
            print(f'{array}')
            continue
        globals()[array][:] = np.NaN
    
    distance_vs_plasticity_LFP_ALL = []
    distance_vs_plasticity_LFP_slope_ALL = []
    distance_vs_plasticity_spike_peak_ALL = []
    distance_vs_plasticity_spike_magn_ALL = []
    
    delta_power_vs_nostim_ALL = np.zeros([numb,64,2]) 
    SW_waveform_vs_nostim_ALL = np.zeros([numb,64,1000,2])
    Peak_dur_vs_nostim = np.zeros([numb,64,2])
    Freq_vs_nostim = np.zeros([numb,64,2])
    Fslope_vs_nostim = np.zeros([numb,64,2])
    Sslope_vs_nostim = np.zeros([numb,64,2])
    Famp_vs_nostim = np.zeros([numb,64,2])
    Samp_vs_nostim = np.zeros([numb,64,2])    
    delta_power_vs_nostim_ALL[:] = np.NaN
    SW_waveform_vs_nostim_ALL[:] = np.NaN
    Peak_dur_vs_nostim[:] = np.NaN
    Fslope_vs_nostim[:] = np.NaN 
    Sslope_vs_nostim[:] = np.NaN
    Famp_vs_nostim[:] = np.NaN
    Samp_vs_nostim[:] = np.NaN
    PSD_lastpre_ALL = []
    PSD_nostim_ALL = []


    
    for day_ind, day in enumerate(days):
        os.chdir(day)
        print(day)
        
        os.chdir([i for i in os.listdir() if 'analysis' in i][0])
        
        # load up all the results
        for file in [i for i in os.listdir() if 'STTC' not in i and 'all_stims_' not in i and 'first_stims_' not in i and 'DESKTOP' not in i]: #just save time by not loading up everything, somethings you don't need
            if '.csv' in file:
                if 'channels' in file or 'to_plot' in file:
                    globals()[f'{file[:-4]}'] = np.loadtxt(f'{file}', dtype = int, delimiter = ',')
                else:
                    globals()[f'{file[:-4]}'] = np.loadtxt(f'{file}', delimiter = ',')
                
            elif 'npy' in file:
                globals()[f'{file[:-4]}'] = np.load(f'{file}')
        
        
        # recalculating relative values for the whisker reponses (in mice where some of the pre-pairing recording blocks hadn't reached a stable response baseline yet)
        if day == '061221':
            to_plot_1_PSTH = [1,2,3]
            to_plot_1_LFP = [1,2,3]
        if day == '160308':
            to_plot_1_PSTH = [3]
            to_plot_1_LFP = [3]
        if day == '121121':
            to_plot_1_PSTH = [1,2,3]
            to_plot_1_LFP = [1,2,3]
        if day == '160426_D1':
            to_plot_1_PSTH = [1,2,3]
            to_plot_1_LFP = [1,2,3]
        if day == '281021':
            to_plot_1_PSTH = [1,2,3]
            to_plot_1_LFP = [1,2,3]
        if day == '221212':
            to_plot_1_LFP = [3]

        LFP_min_rel = LFP_min/np.nanmean(LFP_min[to_plot_1_LFP,:], axis = 0)
        LFP_min_rel_change = np.mean(LFP_min_rel[to_plot_2_LFP,:], axis = 0) - np.mean(LFP_min_rel[to_plot_1_LFP,:], axis = 0)
        LFP_slope_rel = LFP_slope_rel/np.nanmean(LFP_slope_rel[to_plot_1_LFP,:], axis = 0)
        LFP_slope_rel_change = np.nanmean(LFP_slope_rel[to_plot_2_LFP,:], axis = 0) - np.nanmean(LFP_slope_rel[to_plot_1_LFP,:], axis = 0)
        PSTH_resp_magn_rel = PSTH_resp_magn/np.nanmean(PSTH_resp_magn[to_plot_1_PSTH,:], axis = 0)
        PSTH_resp_peak_rel = PSTH_resp_peak/np.nanmean(PSTH_resp_peak[to_plot_1_PSTH,:], axis = 0)
        PSTH_resp_magn_rel_change = np.nanmean(PSTH_resp_magn_rel[to_plot_2_PSTH,:], axis = 0) - np.nanmean(PSTH_resp_magn_rel[to_plot_1_PSTH,:], axis = 0)
        PSTH_resp_peak_rel_change = np.nanmean(PSTH_resp_peak_rel[to_plot_2_PSTH,:], axis = 0) - np.nanmean(PSTH_resp_peak_rel[to_plot_1_PSTH,:], axis = 0)

        # calculate relative values of UP vs DOWN stim deliveries 
        LFP_min_UP_rel = LFP_min_UP/np.nanmean(LFP_min_UP[to_plot_1_LFP,:], axis = 0)
        LFP_min_DOWN_rel = LFP_min_rel_DOWN/np.nanmean(LFP_min_rel_DOWN[to_plot_1_LFP,:], axis = 0)
        PSTH_magn_UP_rel = PSTH_magn_UP/np.nanmean(PSTH_magn_UP[to_plot_1_PSTH,:], axis = 0)
        PSTH_magn_DOWN_rel = PSTH_magn_DOWN/np.nanmean(PSTH_magn_DOWN[to_plot_1_PSTH,:], axis = 0)
        PSTH_peak_UP_rel = PSTH_peak_UP/np.nanmean(PSTH_peak_UP[to_plot_1_PSTH,:], axis = 0)
        PSTH_peak_DOWN_rel = PSTH_peak_DOWN/np.nanmean(PSTH_peak_DOWN[to_plot_1_PSTH,:], axis = 0)

        
        # ------------------------------------------------------ EXCLUDING NOISY OR DRIFTING CHANNELS
        # take out some channels that get noisy / change response shape or duration drastically (electrode drift)

        # automatically detect LFP responsive channels using an LFP cutoff threshold
        LFP_resp_channels_cutoff =  np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
        
        
        # --------------------- DOWN - Pairing
        if day == '061221': # the spiking response in this mouse is a bit buggy - it changes shape drastically in several channels. There is a case for excluding it from the spiking response analysis.
            LFP_bad = [14] # noisy
            for chan in LFP_bad:
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0]) 
            PSTH_bad = [49,53] # increase >200% or change shape drastically between recording blocks
            for chan in PSTH_bad:
                PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == chan)[0]) 
            # PSTH_resp_channels = np.asarray([])
        # if day == '160218':
            # pass
        if day == '160308':
            # all responses get unreasonably big after pairing - electrode drift during baseline. I set the relative values to the last baseline recording block - this also biases the analysis against my own hypothesis.
            LFP_bad = [37,35] # noisy
            for chan in LFP_bad:
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
            # PSTH_resp_channels = np.asarray([])
        if day == '160322':
            LFP_bad = [48,54,56,60] # noisy
            for chan in LFP_bad:  
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
        if day == '160331':
            LFP_bad = [54,56,58,60] # noisy
            for chan in LFP_bad:  
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
            PSTH_bad = [9] # increases >200%
            for chan in PSTH_bad:  
                PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == chan)[0]) 
        if day == '160427':
            # LFP_bad = [17,51,53,55,7] # increase >200%
            # for chan in LFP_bad:  
            #     LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
            PSTH_bad = [1,56,54] # increase >200%
            for chan in PSTH_bad:  
                PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == chan)[0]) 
        if day == '221208':
            LFP_bad = [18] # increases >200% during baseline
            for chan in LFP_bad:  
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
        # if day == '221212': # LFP responses strange... increase in length in depth all around
                #     pass
        if day == '221213':
            PSTH_bad = [11] # noisy
            for chan in PSTH_bad:  
                PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == chan)[0])         
        # if day == '221216':
        #     pass
        if day == '221219_1':
            PSTH_bad = [51] # increase >200%
            for chan in PSTH_bad:  
                PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == chan)[0]) 

            
        # ------------------------------------- UP - Pairing
        if day == '121121':
            LFP_bad = [8,9,13] # noisy channels/increase >100%
            for chan in LFP_bad:  
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
            PSTH_bad = [9,13] # increase > 100% during baseline or noisy
            for chan in PSTH_bad:  
                PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == chan)[0]) 
        # if day == '131121':
        #     LFP_bad = [8] # noisy channel
        #     for chan in LFP_bad:  
        #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
        #     PSTH_bad = [17,29,27] # increase > 200% or change shape
        #     for chan in PSTH_bad:  
        #         PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == chan)[0]) 
        if day == '160310':
            LFP_bad = [2,41,6] # noisy
            for chan in LFP_bad:  
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
        if day == '160426_D1':
            LFP_bad = [36,20] # noisy
            for chan in LFP_bad:  
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
        if day == '160519_B2':
            LFP_bad = [27,20,16,20,22,24,32,38,42,44] # noisy or inverted LFP responses (most likely in layer 1/on dura)
            for chan in LFP_bad:  
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
        if day == '160624_B2':
            LFP_bad = [21,19] # noisy
            for chan in LFP_bad:  
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
        if day == '160628_D1':
            LFP_bad = [37,35,19,17] # noisy
            for chan in LFP_bad:  
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
        #     PSTH_bad = [43,21]
        #     for chan in PSTH_bad:  
        #         PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == chan)[0]) 
        if day == '191121':
            PSTH_bad = [10,35,63] # increase >200%
            for chan in PSTH_bad:  
                PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == chan)[0]) 
        if day == '201121':
            LFP_bad = [22,20] # weird response shape
            for chan in LFP_bad:  
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
            PSTH_bad = [22] # increases >200%
            for chan in PSTH_bad:  
                PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == chan)[0]) 
        if day == '221220_3':
            chans_to_append = [9,57] # doesnt change significance of results but clearly good channels: HF background noise in the last recording block prevents them from crossing 99.9% CI threshold...
            chans_to_append = [i for i in chans_to_append if i not in PSTH_resp_channels]
            for chan in chans_to_append:
                PSTH_resp_channels = np.append(PSTH_resp_channels, chan)
        if day == '281021':
            LFP_bad = [23,8] # noisy
            for chan in LFP_bad:  
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
        if day == '291021':
            LFP_bad = [38] # noisy
            for chan in LFP_bad:  
                LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 

        # resave under that mouse's folder
        np.savetxt('LFP_resp_channels_cutoff.csv', LFP_resp_channels_cutoff, delimiter = ',')
        if PSTH_resp_channels.shape == ():
            np.savetxt('PSTH_resp_channels_cutoff.csv', np.array([PSTH_resp_channels]), delimiter = ',')    
        else:    
            np.savetxt('PSTH_resp_channels_cutoff.csv', PSTH_resp_channels, delimiter = ',')    
        
        
        
        # ----------------------------------------------- FILLING IN THE GROUP MATRICES
        
        for matrix in overall_arrays:
            if any(i in matrix for i in ['distance_vs_plasticity_LFP_ALL', 'distance_vs_plasticity_spike_peak_ALL', 'distance_vs_plasticity_spike_magn_ALL', 'distance_vs_plasticity_LFP_slope_ALL', 'delta_power_nostim_ALL']):
                continue
            
            elif 'ALLCHANS' in matrix and 'channels' in matrix:
                # big array of all resp channels
                if LFP_resp_channels_cutoff.size > 0:
                    LFP_resp_channels_cutoff_ALLCHANS[LFP_resp_channels_cutoff + day_ind*64] = 1
                if PSTH_resp_channels.size > 0:
                    PSTH_resp_channels_ALLCHANS[PSTH_resp_channels + day_ind*64] = 1
                if SW_spiking_channels.size > 0:
                    SW_spiking_channels_ALLCHANS[SW_spiking_channels + day_ind*64] = 1
    
            elif 'ALLCHANS' in matrix:
                # LFP, PSTH, delta, SW params change in all channels
                globals()[matrix][day_ind*64:(day_ind + 1)*64] = globals()[f'{matrix[:-9]}'] 
                
            # LFP timecourse                    
            elif 'LFP' in matrix:
                if LFP_resp_channels_cutoff.size > 0:
                    if LFP_resp_channels_cutoff.size == 1:
                        if 'median' not in matrix: 
                            globals()[matrix][day_ind,:] = np.squeeze(globals()[f'{matrix[:-4]}'][:,LFP_resp_channels_cutoff])
                        else:
                            globals()[matrix][day_ind,:] = np.squeeze(globals()[f'{matrix[:-11]}'][:,LFP_resp_channels_cutoff])
                    elif 'time_to_peak' in matrix:
                        globals()[matrix][day_ind,:] = np.nanmedian(globals()[f'{matrix[:-4]}'][:,LFP_resp_channels_cutoff], axis = 1)
                    elif 'median' in matrix:
                        globals()[matrix][day_ind,:] = np.nanmedian(globals()[f'{matrix[:-11]}'][:,LFP_resp_channels_cutoff], axis = 1)
                    else:
                        globals()[matrix][day_ind,:] = np.nanmean(globals()[f'{matrix[:-4]}'][:,LFP_resp_channels_cutoff], axis = 1)
            
            # spiking timecourse                    
            elif 'PSTH' in matrix:
                if PSTH_resp_channels.size > 0:
                    if PSTH_resp_channels.size == 1:
                        if 'median' not in matrix: 
                            globals()[matrix][day_ind,:] = np.squeeze(globals()[f'{matrix[:-4]}'][:,PSTH_resp_channels])
                        else:
                            globals()[matrix][day_ind,:] = np.squeeze(globals()[f'{matrix[:-11]}'][:,PSTH_resp_channels])
                    elif 'median'in matrix:
                        globals()[matrix][day_ind,:] = np.median(globals()[f'{matrix[:-11]}'][:,PSTH_resp_channels], axis = 1)
                    else:                       
                        globals()[matrix][day_ind,:] = np.mean(globals()[f'{matrix[:-4]}'][:,PSTH_resp_channels], axis = 1)
            
            # delta timecourse                          
            elif 'delta' in matrix:
                if LFP_resp_channels_cutoff.size > 0:
                    if matrix == 'delta_power_vs_nostim_ALL':
                        continue
                    if LFP_resp_channels_cutoff.size == 1: 
                        if 'median' not in matrix: 
                            globals()[matrix][day_ind,:] = np.squeeze(globals()[f'{matrix[:-4]}'][:,LFP_resp_channels_cutoff])
                        else:
                            globals()[matrix][day_ind,:] = np.squeeze(globals()[f'{matrix[:-11]}'][:,LFP_resp_channels_cutoff])
                    elif 'median'in matrix:
                        globals()[matrix][day_ind,:] = np.median(globals()[f'{matrix[:-11]}'][:,LFP_resp_channels_cutoff], axis = 1)
                    else:                       
                        globals()[matrix][day_ind,:] = np.mean(globals()[f'{matrix[:-4]}'][:,LFP_resp_channels_cutoff], axis = 1)
    
        PSD_ALL.append(PSD[:,:,np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]])
        PSD_median_ALL.append(PSD_median[:,:,np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]])

        
        
        # delta and slow waves no stim vs last/first of pre pairing
        if os.path.isfile('PSD_nostim.npy'):
            if day == '160218': #  no stim was done too early before other recordings, not comparable for slow waves/delta
                os.chdir('..')
                os.chdir('..')
                continue
            else:
                delta_power_vs_nostim_ALL[day_ind,:,0] = delta_power_auto_outliers_nostim   
                SW_waveform_vs_nostim_ALL[day_ind,:,:,0] = SW_waveform_sweeps_avg_nostim
                Peak_dur_vs_nostim[day_ind,:,0] = Peak_dur_sweeps_avg_overall_nostim
                Freq_vs_nostim[day_ind,:,0] = SW_frequency_sweeps_avg_nostim       
                Fslope_vs_nostim[day_ind,:,0] = SW_famp_sweeps_avg_overall_nostim
                Sslope_vs_nostim[day_ind,:,0] = SW_samp_sweeps_avg_overall_nostim
                Famp_vs_nostim[day_ind,:,0] = SW_fslope_sweeps_avg_overall_nostim
                Samp_vs_nostim[day_ind,:,0] = SW_sslope_sweeps_avg_overall_nostim
                
                if day == '221220_3' or day == '221213' or day == '221216' or day == '221219_1': # in these mice the nostim sweep was done after the last baseline
                    delta_power_vs_nostim_ALL[day_ind,:,1] = delta_power_auto_outliers[3,:] 
                    SW_waveform_vs_nostim_ALL[day_ind,:,:,1] = SW_waveform_sweeps_avg[3,:,:]
                    Peak_dur_vs_nostim[day_ind,:,1] = Peak_dur_sweeps_avg_overall[3,:]
                    Freq_vs_nostim[day_ind,:,1] = SW_frequency_sweeps_avg[3,:]      
                    Fslope_vs_nostim[day_ind,:,1] = SW_famp_sweeps_avg_overall[3,:]
                    Sslope_vs_nostim[day_ind,:,1] = SW_samp_sweeps_avg_overall[3,:]
                    Famp_vs_nostim[day_ind,:,1] = SW_fslope_sweeps_avg_overall[3,:]
                    Samp_vs_nostim[day_ind,:,1] = SW_sslope_sweeps_avg_overall[3,:]
                else: # in the other mice the nostim sweep was done before the first baseline
                    delta_power_vs_nostim_ALL[day_ind,:,1] = delta_power_auto_outliers[0,:] 
                    SW_waveform_vs_nostim_ALL[day_ind,:,:,1] = SW_waveform_sweeps_avg[0,:,:]
                    Peak_dur_vs_nostim[day_ind,:,1] = Peak_dur_sweeps_avg_overall[0,:]
                    Freq_vs_nostim[day_ind,:,1] = SW_frequency_sweeps_avg[0,:]      
                    Fslope_vs_nostim[day_ind,:,1] = SW_famp_sweeps_avg_overall[0,:]
                    Sslope_vs_nostim[day_ind,:,1] = SW_samp_sweeps_avg_overall[0,:]
                    Famp_vs_nostim[day_ind,:,1] = SW_fslope_sweeps_avg_overall[0,:]
                    Samp_vs_nostim[day_ind,:,1] = SW_sslope_sweeps_avg_overall[0,:]
        
                PSD_lastpre_ALL.append(PSD[3,:,np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]])
                PSD_nostim_ALL.append(PSD_nostim[:,np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]])
    
        
    
    
        
        #distance vs plasticity 
        distance_vs_plasticity_LFP = np.zeros([2,len(LFP_resp_channels_cutoff)])
        distance_vs_plasticity_LFP_slope = np.zeros([2,len(LFP_resp_channels_cutoff)])
        distance_vs_plasticity_spike_peak = np.zeros([2,PSTH_resp_channels.size])
        distance_vs_plasticity_spike_magn = np.zeros([2,PSTH_resp_channels.size])
        
        i = 0
        for chan_ind, chan in enumerate(list(LFP_resp_channels_cutoff)):
            electrode_distance_in_indices = np.squeeze(np.argwhere(channelMapArray == princ_channel) - np.argwhere(channelMapArray == chan))
            distance_vs_plasticity_LFP[0,chan_ind] = np.sqrt(electrode_distance_in_indices[0]**2 + electrode_distance_in_indices[1]**2)
            distance_vs_plasticity_LFP_slope[0,chan_ind] = np.sqrt(electrode_distance_in_indices[0]**2 + electrode_distance_in_indices[1]**2)
            distance_vs_plasticity_LFP[1,chan_ind] = LFP_min_rel_change[chan]
            distance_vs_plasticity_LFP_slope[1,chan_ind] = LFP_slope_rel_change[chan]
    
            if chan in PSTH_resp_channels:
                distance_vs_plasticity_spike_peak[0,i] = np.sqrt(electrode_distance_in_indices[0]**2 + electrode_distance_in_indices[1]**2)
                distance_vs_plasticity_spike_magn[0,i] = np.sqrt(electrode_distance_in_indices[0]**2 + electrode_distance_in_indices[1]**2)          
                distance_vs_plasticity_spike_peak[1,i] = PSTH_resp_peak_rel_change[chan]
                distance_vs_plasticity_spike_magn[1,i] = PSTH_resp_magn_rel_change[chan]
                i+=1
        distance_vs_plasticity_LFP_ALL.append(distance_vs_plasticity_LFP)
        distance_vs_plasticity_LFP_slope_ALL.append(distance_vs_plasticity_LFP_slope)
        distance_vs_plasticity_spike_peak_ALL.append(distance_vs_plasticity_spike_peak)
        distance_vs_plasticity_spike_magn_ALL.append(distance_vs_plasticity_spike_magn)
            
        os.chdir('..')
        os.chdir('..')
    
    
    
    
    # save average timecourse per mouse     
    np.savetxt('LFP_min_ALL.csv', LFP_min_ALL, delimiter = ',')
    np.savetxt('LFP_min_rel_ALL.csv', LFP_min_rel_ALL, delimiter = ',')
    np.savetxt('LFP_min_median_ALL.csv', LFP_min_median_ALL, delimiter = ',')
    np.savetxt('LFP_min_rel_median_ALL.csv', LFP_min_rel_median_ALL, delimiter = ',')

    np.savetxt('PSTH_resp_magn_ALL.csv', PSTH_resp_magn_ALL, delimiter = ',')
    np.savetxt('PSTH_resp_peak_ALL.csv', PSTH_resp_peak_ALL, delimiter = ',')
    np.savetxt('PSTH_resp_magn_rel_ALL.csv', PSTH_resp_magn_rel_ALL, delimiter = ',')
    np.savetxt('PSTH_resp_peak_rel_ALL.csv', PSTH_resp_peak_rel_ALL, delimiter = ',')
    np.savetxt('PSTH_resp_magn_median_ALL.csv', PSTH_resp_magn_median_ALL, delimiter = ',')
    np.savetxt('PSTH_resp_peak_median_ALL.csv', PSTH_resp_peak_median_ALL, delimiter = ',')
    np.savetxt('PSTH_resp_magn_rel_median_ALL.csv', PSTH_resp_magn_rel_median_ALL, delimiter = ',')
    np.savetxt('PSTH_resp_peak_rel_median_ALL.csv', PSTH_resp_peak_rel_median_ALL, delimiter = ',')

    np.savetxt('delta_power_ALL.csv', delta_power_ALL, delimiter = ',')
    np.savetxt('delta_power_rel_ALL.csv', delta_power_rel_ALL, delimiter = ',')
    np.savetxt('delta_power_median_ALL.csv', delta_power_median_ALL, delimiter = ',')
    np.savetxt('delta_power_rel_median_ALL.csv', delta_power_rel_median_ALL, delimiter = ',')
    np.savetxt('delta_power_auto_outliers_ALL.csv', delta_power_auto_outliers_ALL, delimiter = ',')
    np.savetxt('delta_power_auto_outliers_rel_ALL.csv', delta_power_auto_outliers_rel_ALL, delimiter = ',')
    np.savetxt('delta_power_auto_outliers_median_ALL.csv', delta_power_auto_outliers_median_ALL, delimiter = ',')
    np.savetxt('delta_power_auto_outliers_rel_median_ALL.csv', delta_power_auto_outliers_rel_median_ALL, delimiter = ',')
    # np.savetxt('delta_power_median_auto_outliers_ALL.csv', delta_power_median_auto_outliers_ALL, delimiter = ',')
    # np.savetxt('delta_power_median_auto_outliers_rel_ALL.csv', delta_power_median_auto_outliers_rel_ALL, delimiter = ',')
    # np.savetxt('delta_power_median_auto_outliers_median_ALL.csv', delta_power_median_auto_outliers_median_ALL, delimiter = ',')
    # np.savetxt('delta_power_median_auto_outliers_rel_median_ALL.csv', delta_power_median_auto_outliers_rel_median_ALL, delimiter = ',')

    pickle.dump(PSD_ALL, open('PSD_ALL.pkl', 'wb'))
    
    np.savetxt('LFP_slope_rel_ALL.csv', LFP_slope_rel_ALL, delimiter = ',')
    np.savetxt('LFP_slope_rel_median_ALL.csv', LFP_slope_rel_median_ALL, delimiter = ',')
    
    np.savetxt('LFP_min_UP_ALL.csv', LFP_min_UP_ALL, delimiter = ',')
    np.savetxt('LFP_min_DOWN_ALL.csv', LFP_min_DOWN_ALL, delimiter = ',')
    np.savetxt('LFP_min_UP_rel_ALL.csv', LFP_min_UP_rel_ALL, delimiter = ',')
    np.savetxt('LFP_min_DOWN_rel_ALL.csv', LFP_min_DOWN_rel_ALL, delimiter = ',')
    np.savetxt('PSTH_peak_UP_ALL.csv', PSTH_peak_UP_ALL, delimiter = ',')
    np.savetxt('PSTH_peak_DOWN_ALL.csv', PSTH_peak_DOWN_ALL, delimiter = ',')
    np.savetxt('PSTH_peak_UP_rel_ALL.csv', PSTH_peak_UP_rel_ALL, delimiter = ',')
    np.savetxt('PSTH_peak_DOWN_rel_ALL.csv', PSTH_peak_DOWN_rel_ALL, delimiter = ',')
    np.savetxt('PSTH_magn_UP_ALL.csv', PSTH_magn_UP_ALL, delimiter = ',')
    np.savetxt('PSTH_magn_DOWN_ALL.csv', PSTH_magn_DOWN_ALL, delimiter = ',')
    np.savetxt('PSTH_magn_UP_rel_ALL.csv', PSTH_magn_UP_rel_ALL, delimiter = ',')
    np.savetxt('PSTH_magn_DOWN_rel_ALL.csv', PSTH_magn_DOWN_rel_ALL, delimiter = ',')
    
    np.savetxt('LFP_min_UP_median_ALL.csv', LFP_min_UP_median_ALL, delimiter = ',')
    np.savetxt('LFP_min_DOWN_median_ALL.csv', LFP_min_DOWN_median_ALL, delimiter = ',')
    np.savetxt('LFP_min_UP_rel_median_ALL.csv', LFP_min_UP_rel_median_ALL, delimiter = ',')
    np.savetxt('LFP_min_DOWN_rel_median_ALL.csv', LFP_min_DOWN_rel_median_ALL, delimiter = ',')
    np.savetxt('PSTH_peak_UP_median_ALL.csv', PSTH_peak_UP_median_ALL, delimiter = ',')
    np.savetxt('PSTH_peak_DOWN_median_ALL.csv', PSTH_peak_DOWN_median_ALL, delimiter = ',')
    np.savetxt('PSTH_peak_UP_rel_median_ALL.csv', PSTH_peak_UP_rel_median_ALL, delimiter = ',')
    np.savetxt('PSTH_peak_DOWN_rel_median_ALL.csv', PSTH_peak_DOWN_rel_median_ALL, delimiter = ',')
    np.savetxt('PSTH_magn_UP_median_ALL.csv', PSTH_magn_UP_median_ALL, delimiter = ',')
    np.savetxt('PSTH_magn_DOWN_median_ALL.csv', PSTH_magn_DOWN_median_ALL, delimiter = ',')
    np.savetxt('PSTH_magn_UP_rel_median_ALL.csv', PSTH_magn_UP_rel_median_ALL, delimiter = ',')
    np.savetxt('PSTH_magn_DOWN_rel_median_ALL.csv', PSTH_magn_DOWN_rel_median_ALL, delimiter = ',')



    # save change in all channels
    np.savetxt('LFP_min_rel_change_ALLCHANS.csv', LFP_min_rel_change_ALLCHANS, delimiter = ',')
    np.savetxt('LFP_slope_rel_change_ALLCHANS.csv', LFP_slope_rel_change_ALLCHANS, delimiter = ',')
    np.savetxt('LFP_time_to_peak_change_ALLCHANS.csv', LFP_time_to_peak_change_ALLCHANS, delimiter = ',')
    
    np.savetxt('PSTH_resp_magn_rel_change_ALLCHANS.csv', PSTH_resp_magn_rel_change_ALLCHANS, delimiter = ',')
    np.savetxt('PSTH_resp_peak_rel_change_ALLCHANS.csv', PSTH_resp_peak_rel_change_ALLCHANS, delimiter = ',')

    np.savetxt('delta_power_rel_change_ALLCHANS.csv', delta_power_rel_change_ALLCHANS, delimiter = ',')
    np.savetxt('delta_power_auto_outliers_rel_change_ALLCHANS.csv', delta_power_auto_outliers_rel_change_ALLCHANS, delimiter = ',')
    np.savetxt('delta_power_median_auto_outliers_rel_change_ALLCHANS.csv', delta_power_median_auto_outliers_rel_change_ALLCHANS, delimiter = ',')
    
    np.savetxt('Freq_change_ALLCHANS.csv', Freq_change_ALLCHANS, delimiter = ',')
    np.savetxt('Peak_dur_overall_change_ALLCHANS.csv', Peak_dur_overall_change_ALLCHANS, delimiter = ',')
    np.savetxt('Fslope_overall_change_ALLCHANS.csv', Fslope_overall_change_ALLCHANS, delimiter = ',')
    np.savetxt('Sslope_overall_change_ALLCHANS.csv', Sslope_overall_change_ALLCHANS, delimiter = ',')
    np.savetxt('Famp_overall_change_ALLCHANS.csv', Famp_overall_change_ALLCHANS, delimiter = ',')
    np.savetxt('Samp_overall_change_ALLCHANS.csv', Samp_overall_change_ALLCHANS, delimiter = ',')
    np.savetxt('SW_spiking_area_change_ALLCHANS.csv', SW_spiking_area_change_ALLCHANS, delimiter = ',')
    np.savetxt('SW_spiking_peak_change_ALLCHANS.csv', SW_spiking_peak_change_ALLCHANS, delimiter = ',')
    
    np.savetxt('Peak_dur_change_mean_ALLCHANS.csv', Peak_dur_change_mean_ALLCHANS, delimiter = ',')
    np.savetxt('Fslope_change_mean_ALLCHANS.csv', Fslope_change_mean_ALLCHANS, delimiter = ',')
    np.savetxt('Sslope_change_mean_ALLCHANS.csv', Sslope_change_mean_ALLCHANS, delimiter = ',')
    np.savetxt('Famp_change_mean_ALLCHANS.csv', Famp_change_mean_ALLCHANS, delimiter = ',')
    np.savetxt('Samp_change_mean_ALLCHANS.csv', Samp_change_mean_ALLCHANS, delimiter = ',')

    np.savetxt('Peak_dur_change_median_ALLCHANS.csv', Peak_dur_change_median_ALLCHANS, delimiter = ',')
    np.savetxt('Fslope_change_median_ALLCHANS.csv', Fslope_change_median_ALLCHANS, delimiter = ',')
    np.savetxt('Sslope_change_median_ALLCHANS.csv', Sslope_change_median_ALLCHANS, delimiter = ',')
    np.savetxt('Famp_change_median_ALLCHANS.csv', Famp_change_median_ALLCHANS, delimiter = ',')
    np.savetxt('Samp_change_median_ALLCHANS.csv', Samp_change_median_ALLCHANS, delimiter = ',')

    np.savetxt('AUC_all_stims_ALLCHANS.csv', AUC_all_stims_ALLCHANS, delimiter = ',')
    np.savetxt('AUC_first_stims_ALLCHANS.csv', AUC_first_stims_ALLCHANS, delimiter = ',')
    np.savetxt('pairing_UP_freq_first_stims_ALLCHANS.csv', pairing_UP_freq_first_stims_ALLCHANS, delimiter = ',')

    # save selected channnels
    np.savetxt('LFP_resp_channels_cutoff_ALLCHANS.csv', LFP_resp_channels_cutoff_ALLCHANS, delimiter = ',')
    np.savetxt('PSTH_resp_channels_ALLCHANS.csv', PSTH_resp_channels_ALLCHANS, delimiter = ',')
    np.savetxt('SW_spiking_channels_ALLCHANS.csv', SW_spiking_channels_ALLCHANS, delimiter = ',')
    
    
    pickle.dump(distance_vs_plasticity_LFP_ALL, open('distance_vs_plasticity_LFP_ALL.pkl', 'wb'))
    pickle.dump(distance_vs_plasticity_LFP_slope_ALL, open('distance_vs_plasticity_LFP_slope_ALL.pkl', 'wb'))
    pickle.dump(distance_vs_plasticity_spike_peak_ALL, open('distance_vs_plasticity_spike_peak_ALL.pkl', 'wb'))
    pickle.dump(distance_vs_plasticity_spike_magn_ALL, open('distance_vs_plasticity_spike_magn_ALL.pkl', 'wb'))
    
    pickle.dump(delta_power_vs_nostim_ALL, open('delta_power_vs_nostim_ALL.pkl', 'wb'))
    pickle.dump(SW_waveform_vs_nostim_ALL, open('SW_waveform_vs_nostim_ALL.pkl', 'wb'))
    pickle.dump(Peak_dur_vs_nostim, open('Peak_dur_vs_nostim.pkl', 'wb'))
    pickle.dump(Freq_vs_nostim, open('Freq_vs_nostim.pkl', 'wb'))
    pickle.dump(Fslope_vs_nostim, open('Fslope_vs_nostim.pkl', 'wb'))
    pickle.dump(Sslope_vs_nostim, open('Sslope_vs_nostim.pkl', 'wb'))
    pickle.dump(Famp_vs_nostim, open('Famp_vs_nostim.pkl', 'wb'))
    pickle.dump(Samp_vs_nostim, open('Samp_vs_nostim.pkl', 'wb'))
    pickle.dump(PSD_lastpre_ALL, open('PSD_lastpre_ALL.pkl', 'wb'))
    pickle.dump(PSD_nostim_ALL, open('PSD_nostim_ALL.pkl', 'wb'))



#%% NUMBER OF RESPONSIVE CHANNELS
os.chdir(os.path.join(overall_path, r'UP_pairing'))
channels_UP = np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool)
n_chans_UP = np.sum(np.reshape(channels_UP, (-1,64)),1)
print(f'UP mean LFP channels {np.mean(n_chans_UP)}, std {np.std(n_chans_UP)}')
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
channels_DOWN = np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool)
n_chans_DOWN = np.sum(np.reshape(channels_DOWN, (-1,64)),1)
print(f'DOWN mean LFP channels {np.mean(n_chans_DOWN)}, std {np.std(n_chans_DOWN)}')
print(f'total number of LFP channels: {np.mean(np.concatenate((n_chans_UP, n_chans_DOWN)))}, std {np.std(np.concatenate((n_chans_UP, n_chans_DOWN)))}')

os.chdir(os.path.join(overall_path, r'UP_pairing'))
channels_UP = np.loadtxt('PSTH_resp_channels_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool)
n_chans_UP = np.sum(np.reshape(channels_UP, (-1,64)),1)
print(f'UP mean LFP channels {np.mean(n_chans_UP)}, std {np.std(n_chans_UP)}')
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
channels_DOWN = np.loadtxt('PSTH_resp_channels_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool)
n_chans_DOWN = np.sum(np.reshape(channels_DOWN, (-1,64)),1)
print(f'DOWN mean LFP channels {np.mean(n_chans_DOWN)}, std {np.std(n_chans_DOWN)}')
print(f'total number of LFP channels: {np.mean(np.concatenate((n_chans_UP, n_chans_DOWN)))}, std {np.std(np.concatenate((n_chans_UP, n_chans_DOWN)))}')



#%% UP vs DOWN LFP, PSTH and delta timecourse on same plot, mean and median across channels

os.chdir(overall_path)

patch = 1 #(how many times SEM to plot as patch)



 # ----------------------------------------------------------------- LFP 
fig, ax = plt.subplots(figsize = (10,4))
os.chdir(os.path.join(overall_path, r'UP_pairing'))
LFP_min_UP = np.loadtxt('LFP_min_rel_ALL.csv', delimiter = ',')*100
take_out = [[0,0], [10,0]]
for sweep in take_out:
    LFP_min_UP[sweep[0], sweep[1]] = np.NaN
ax.plot(np.nanmean(LFP_min_UP, axis = 0), color = 'r')
ax.fill_between(list(range(10)), np.nanmean(LFP_min_UP, axis = 0) + patch*np.nanstd(LFP_min_UP, axis = 0)/np.sqrt(LFP_min_UP.shape[0]), np.nanmean(LFP_min_UP, axis = 0) - patch*np.nanstd(LFP_min_UP, axis = 0)/np.sqrt(LFP_min_UP.shape[0]), alpha = 0.1, color = 'r')
os.chdir('..')
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
LFP_min_DOWN = np.loadtxt('LFP_min_rel_ALL.csv', delimiter = ',')*100
take_out = [[2,0], [2,1], [2,2], [8,0], [8,1], [8,2]] # hadn't reached baseline yet
for sweep in take_out:
    LFP_min_DOWN[sweep[0], sweep[1]] = np.NaN
ax.plot(np.nanmean(LFP_min_DOWN, axis = 0), c = 'k')
ax.fill_between(list(range(10)), np.nanmean(LFP_min_DOWN, axis = 0) + patch*np.nanstd(LFP_min_DOWN, axis = 0)/np.sqrt(LFP_min_DOWN.shape[0]), np.nanmean(LFP_min_DOWN, axis = 0) - patch*np.nanstd(LFP_min_DOWN, axis = 0)/np.sqrt(LFP_min_DOWN.shape[0]), alpha = 0.1, color = 'k')
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_ylim([30, 160])
ax.set_xlabel('time from pairing (min)', size = 16)
ax.set_ylabel('LFP response \n (% of baseline)', size = 16)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 16)
ax.set_yticks([50,100,150])
ax.set_yticklabels(list(map(str,[50,100,150])), size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
os.chdir(figure_path)
plt.savefig('LFP UP vs DOWN MEAN.pdf', dpi = 1000, format = 'pdf')
plt.savefig('LFP UP vs DOWN MEAN.jpg', dpi = 1000, format = 'jpg')

fig, ax = plt.subplots(figsize = (10,4))
os.chdir(os.path.join(overall_path, r'UP_pairing'))
LFP_min_median_UP = np.loadtxt('LFP_min_rel_median_ALL.csv', delimiter = ',')*100
take_out = [[0,0], [10,0]]
for sweep in take_out:
    LFP_min_median_UP[sweep[0], sweep[1]] = np.NaN
ax.plot(np.nanmean(LFP_min_median_UP, axis = 0), color = 'r')
ax.fill_between(list(range(10)), np.nanmean(LFP_min_median_UP, axis = 0) + patch*np.nanstd(LFP_min_median_UP, axis = 0)/np.sqrt(LFP_min_median_UP.shape[0]), np.nanmean(LFP_min_median_UP, axis = 0) - patch*np.nanstd(LFP_min_median_UP, axis = 0)/np.sqrt(LFP_min_median_UP.shape[0]), alpha = 0.1, color = 'r')
os.chdir('..')
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
LFP_min_median_DOWN = np.loadtxt('LFP_min_rel_median_ALL.csv', delimiter = ',')*100
take_out = [[2,0], [2,1], [2,2], [8,0], [8,1], [8,2]] # hadn't reached baseline yet
for sweep in take_out:
    LFP_min_DOWN[sweep[0], sweep[1]] = np.NaN
ax.plot(np.nanmean(LFP_min_median_DOWN, axis = 0), c = 'k')
ax.fill_between(list(range(10)), np.nanmean(LFP_min_median_DOWN, axis = 0) + patch*np.nanstd(LFP_min_median_DOWN, axis = 0)/np.sqrt(LFP_min_median_DOWN.shape[0]), np.nanmean(LFP_min_median_DOWN, axis = 0) - patch*np.nanstd(LFP_min_median_DOWN, axis = 0)/np.sqrt(LFP_min_median_DOWN.shape[0]), alpha = 0.1, color = 'k')
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_ylim([30, 160])
ax.set_xlabel('time from pairing (min)', size = 16)
ax.set_ylabel('LFP median response \n (% of baseline)', size = 16)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 16)
ax.set_yticks([50,100,150])
ax.set_yticklabels(list(map(str,[50,100,150])), size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
os.chdir(figure_path)
plt.savefig('LFP UP vs DOWN MEDIAN.pdf', dpi = 1000, format = 'pdf')
plt.savefig('LFP UP vs DOWN MEDIAN.jpg', dpi = 1000, format = 'jpg')

scipy.stats.ttest_1samp(np.mean(LFP_min_UP[:,[4,5,6,7,8,9]], axis = 1),100)
print(f'LFP UP mean {np.mean(np.mean(LFP_min_UP[:,[4,5,6,7,8,9]], axis = 1))}')
print(f'LFP UP std {np.std(np.mean(LFP_min_UP[:,[4,5,6,7,8,9]], axis = 1), ddof = 1)}') 
scipy.stats.shapiro(np.mean(LFP_min_UP[:,[4,5,6,7,8,9]], axis = 1))

scipy.stats.ttest_1samp(np.mean(LFP_min_DOWN[:,[4,5,6,7,8,9]], axis = 1),100)
print(f'LFP DOWN mean {np.mean(np.mean(LFP_min_DOWN[:,[4,5,6,7,8,9]], axis = 1))}')
print(f'LFP DOWN std {np.std(np.mean(LFP_min_DOWN[:,[4,5,6,7,8,9]], axis = 1), ddof = 1)}') 
scipy.stats.shapiro(np.mean(LFP_min_DOWN[:,[4,5,6,7,8,9]], axis = 1))




 # ----------------------------------------------------------------- spikes 

fig, ax = plt.subplots(figsize = (10,4))
os.chdir(os.path.join(overall_path, r'UP_pairing'))
PSTH_magn_UP = np.loadtxt('PSTH_resp_magn_rel_ALL.csv', delimiter = ',')*100
take_out = [[0,0], [10,0]] # hadn't reached baseline yet
for sweep in take_out:
    PSTH_magn_UP[sweep[0], sweep[1]] = np.NaN
ax.plot(np.nanmean(PSTH_magn_UP, axis = 0), color = 'r')
ax.fill_between(list(range(10)), np.nanmean(PSTH_magn_UP, axis = 0) + patch*np.nanstd(PSTH_magn_UP, axis = 0)/np.sqrt(PSTH_magn_UP.shape[0]), np.nanmean(PSTH_magn_UP, axis = 0) - patch*np.nanstd(PSTH_magn_UP, axis = 0)/np.sqrt(PSTH_magn_UP.shape[0]), alpha = 0.1, color = 'r')
os.chdir('..')
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
PSTH_magn_DOWN = np.loadtxt('PSTH_resp_magn_rel_ALL.csv', delimiter = ',')*100
ax.plot(np.nanmean(PSTH_magn_DOWN, axis = 0), c = 'k')
ax.fill_between(list(range(10)), np.nanmean(PSTH_magn_DOWN, axis = 0) + patch*np.nanstd(PSTH_magn_DOWN, axis = 0)/np.sqrt(PSTH_magn_DOWN.shape[0]), np.nanmean(PSTH_magn_DOWN, axis = 0) - patch*np.nanstd(PSTH_magn_DOWN, axis = 0)/np.sqrt(PSTH_magn_DOWN.shape[0]), alpha = 0.1, color = 'k')
ax.set_ylim([30, 160])
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 16)
ax.set_ylabel('PSTH response \n (% of baseline)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 16)
ax.set_yticks([50,100,150])
ax.set_yticklabels(list(map(str,[50,100,150])), size = 16)
plt.tight_layout()
os.chdir(figure_path)
plt.savefig('PSTH magn UP vs DOWN MEAN.pdf', dpi = 1000, format = 'pdf')
plt.savefig('PSTH magn UP vs DOWN MEAN.jpg', dpi = 1000, format = 'jpg')

fig, ax = plt.subplots(figsize = (10,4))
os.chdir(os.path.join(overall_path, r'UP_pairing'))
PSTH_magn_median_UP = np.loadtxt('PSTH_resp_magn_rel_median_ALL.csv', delimiter = ',')*100
take_out = [[0,0], [10,0]] # hadn't reached baseline yet
for sweep in take_out:
    PSTH_magn_median_UP[sweep[0], sweep[1]] = np.NaN
ax.plot(np.nanmean(PSTH_magn_UP, axis = 0), color = 'r')
ax.fill_between(list(range(10)), np.nanmean(PSTH_magn_median_UP, axis = 0) + patch*np.nanstd(PSTH_magn_median_UP, axis = 0)/np.sqrt(PSTH_magn_median_UP.shape[0]), np.nanmean(PSTH_magn_median_UP, axis = 0) - patch*np.nanstd(PSTH_magn_median_UP, axis = 0)/np.sqrt(PSTH_magn_median_UP.shape[0]), alpha = 0.1, color = 'r')
os.chdir('..')
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
PSTH_magn_median_DOWN = np.loadtxt('PSTH_resp_magn_rel_median_ALL.csv', delimiter = ',')*100
ax.plot(np.nanmean(PSTH_magn_median_DOWN, axis = 0), c = 'k')
ax.fill_between(list(range(10)), np.nanmean(PSTH_magn_median_DOWN, axis = 0) + patch*np.nanstd(PSTH_magn_median_DOWN, axis = 0)/np.sqrt(PSTH_magn_median_DOWN.shape[0]), np.nanmean(PSTH_magn_median_DOWN, axis = 0) - patch*np.nanstd(PSTH_magn_median_DOWN, axis = 0)/np.sqrt(PSTH_magn_median_DOWN.shape[0]), alpha = 0.1, color = 'k')
ax.set_ylim([30, 160])
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 16)
ax.set_ylabel('PSTH median response \n (% of baseline)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 16)
ax.set_yticks([50,100,150])
ax.set_yticklabels(list(map(str,[50,100,150])), size = 16)
plt.tight_layout()
os.chdir(figure_path)
plt.savefig('PSTH magn UP vs DOWN MEDIAN.pdf', dpi = 1000, format = 'pdf')
plt.savefig('PSTH magn UP vs DOWN MEDIAN.jpg', dpi = 1000, format = 'jpg')

scipy.stats.ttest_1samp(np.mean(PSTH_magn_UP[:,[4,5,6,7,8,9]], axis = 1),100)
print(f'PSTH UP mean {np.mean(np.mean(PSTH_magn_UP[:,[4,5,6,7,8,9]], axis = 1))}')
print(f'PSTH UP std {np.std(np.mean(PSTH_magn_UP[:,[4,5,6,7,8,9]], axis = 1), ddof = 1)}') 
scipy.stats.shapiro(np.mean(PSTH_magn_UP[:,[4,5,6,7,8,9]], axis = 1))

scipy.stats.ttest_1samp(np.mean(PSTH_magn_DOWN[:,[4,5,6,7,8,9]], axis = 1),100)
print(f'PSTH DOWN mean {np.mean(np.mean(PSTH_magn_DOWN[:,[4,5,6,7,8,9]], axis = 1))}')
print(f'PSTH DOWN std {np.std(np.mean(PSTH_magn_DOWN[:,[4,5,6,7,8,9]], axis = 1), ddof = 1)}') 
scipy.stats.shapiro(np.mean(PSTH_magn_DOWN[:,[4,5,6,7,8,9]], axis = 1))







 # ----------------------------------------------------------------- delta 

patch = 1 #(how many times SEM to plot as patch)
fig, ax = plt.subplots(figsize = (10,4))
os.chdir(os.path.join(overall_path, r'UP_pairing'))
delta_UP = np.loadtxt('delta_power_auto_outliers_rel_ALL.csv', delimiter = ',')*100
ax.plot(np.nanmean(delta_UP, axis = 0), color = 'r')
ax.fill_between(list(range(10)), np.nanmean(delta_UP, axis = 0) + patch*np.nanstd(delta_UP, axis = 0)/np.sqrt(delta_UP.shape[0]), np.nanmean(delta_UP, axis = 0) - patch*np.nanstd(delta_UP, axis = 0)/np.sqrt(delta_UP.shape[0]), alpha = 0.1, color = 'r')
os.chdir('..')
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
delta_DOWN = np.loadtxt('delta_power_auto_outliers_rel_ALL.csv', delimiter = ',')*100
# delta_DOWN[1,:] = np.NaN
ax.plot(np.nanmean(delta_DOWN, axis = 0), c = 'k')
ax.fill_between(list(range(10)), np.nanmean(delta_DOWN, axis = 0) + patch*np.nanstd(delta_DOWN, axis = 0)/np.sqrt(delta_DOWN.shape[0]), np.nanmean(delta_DOWN, axis = 0) - patch*np.nanstd(delta_DOWN, axis = 0)/np.sqrt(delta_DOWN.shape[0]), alpha = 0.1, color = 'k')
ax.set_ylim([30, 160])
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 16)
ax.set_ylabel('delta power \n (% of baseline)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 16)
ax.set_yticks([50,100,150])
ax.set_yticklabels(list(map(str,[50,100,150])), size = 16)
plt.tight_layout()
os.chdir(figure_path)
plt.savefig('delta UP vs DOWN.pdf', dpi = 1000, format = 'pdf')
plt.savefig('delta UP vs DOWN.jpg', dpi = 1000, format = 'jpg')

patch = 1 #(how many times SEM to plot as patch)
fig, ax = plt.subplots(figsize = (10,4))
os.chdir(os.path.join(overall_path, r'UP_pairing'))
delta_median_UP = np.loadtxt('delta_power_auto_outliers_rel_median_ALL.csv', delimiter = ',')*100
ax.plot(np.nanmean(delta_median_UP, axis = 0), color = 'r')
ax.fill_between(list(range(10)), np.nanmean(delta_median_UP, axis = 0) + patch*np.nanstd(delta_median_UP, axis = 0)/np.sqrt(delta_median_UP.shape[0]), np.nanmean(delta_median_UP, axis = 0) - patch*np.nanstd(delta_median_UP, axis = 0)/np.sqrt(delta_median_UP.shape[0]), alpha = 0.1, color = 'r')
os.chdir('..')
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
delta_median_DOWN = np.loadtxt('delta_power_auto_outliers_rel_median_ALL.csv', delimiter = ',')*100
# delta_DOWN[1,:] = np.NaN
ax.plot(np.nanmean(delta_median_DOWN, axis = 0), c = 'k')
ax.fill_between(list(range(10)), np.nanmean(delta_median_DOWN, axis = 0) + patch*np.nanstd(delta_median_DOWN, axis = 0)/np.sqrt(delta_median_DOWN.shape[0]), np.nanmean(delta_median_DOWN, axis = 0) - patch*np.nanstd(delta_median_DOWN, axis = 0)/np.sqrt(delta_median_DOWN.shape[0]), alpha = 0.1, color = 'k')
ax.set_ylim([30, 160])
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 16)
ax.set_ylabel('delta power median \n (% of baseline)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 16)
ax.set_yticks([50,100,150])
ax.set_yticklabels(list(map(str,[50,100,150])), size = 16)
plt.tight_layout()
os.chdir(figure_path)
plt.savefig('delta UP vs DOWN MEDIAN.pdf', dpi = 1000, format = 'pdf')
plt.savefig('delta UP vs DOWN MEDIAN.jpg', dpi = 1000, format = 'jpg')

scipy.stats.ttest_1samp(np.mean(delta_UP[:,[4,5,6,7,8,9]], axis = 1),100)
print(f'delta UP mean {np.mean(np.mean(delta_UP[:,[4,5,6,7,8,9]], axis = 1))}')
print(f'delta UP std {np.std(np.mean(delta_UP[:,[4,5,6,7,8,9]], axis = 1), ddof = 1)}') 
scipy.stats.shapiro(np.mean(delta_UP[:,[4,5,6,7,8,9]], axis = 1))

scipy.stats.ttest_1samp(np.mean(delta_DOWN[:,[4,5,6,7,8,9]], axis = 1),100)
print(f'delta DOWN mean {np.mean(np.mean(delta_DOWN[:,[4,5,6,7,8,9]], axis = 1))}')
print(f'delta DOWN std {np.std(np.mean(delta_DOWN[:,[4,5,6,7,8,9]], axis = 1), ddof = 1)}') 
scipy.stats.shapiro(np.mean(delta_DOWN[:,[4,5,6,7,8,9]], axis = 1))







 # ----------------------------------------------------------------- slope 

patch = 1 #(how many times SEM to plot as patch)
fig, ax = plt.subplots(figsize = (10,4))
os.chdir(os.path.join(overall_path, r'UP_pairing'))
slope_UP = np.loadtxt('LFP_slope_rel_ALL.csv', delimiter = ',')*100
ax.plot(np.nanmean(slope_UP, axis = 0), color = 'r')
ax.fill_between(list(range(10)), np.nanmean(slope_UP, axis = 0) + patch*np.nanstd(slope_UP, axis = 0)/np.sqrt(slope_UP.shape[0]), np.nanmean(slope_UP, axis = 0) - patch*np.nanstd(slope_UP, axis = 0)/np.sqrt(slope_UP.shape[0]), alpha = 0.1, color = 'r')
os.chdir('..')
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
slope_DOWN = np.loadtxt('LFP_slope_rel_ALL.csv', delimiter = ',')*100
# delta_DOWN[1,:] = np.NaN
ax.plot(np.nanmean(slope_DOWN, axis = 0), c = 'k')
ax.fill_between(list(range(10)), np.nanmean(slope_DOWN, axis = 0) + patch*np.nanstd(slope_DOWN, axis = 0)/np.sqrt(slope_DOWN.shape[0]), np.nanmean(slope_DOWN, axis = 0) - patch*np.nanstd(slope_DOWN, axis = 0)/np.sqrt(slope_DOWN.shape[0]), alpha = 0.1, color = 'k')
ax.set_ylim([30, 160])
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 16)
ax.set_ylabel('LFP slope \n (% of baseline)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 16)
ax.set_yticks([50,100,150])
ax.set_yticklabels(list(map(str,[50,100,150])), size = 16)
plt.tight_layout()
os.chdir(figure_path)
plt.savefig('LFP slope UP vs DOWN.pdf', dpi = 1000, format = 'pdf')
plt.savefig('LFP slope UP vs DOWN.jpg', dpi = 1000, format = 'jpg')

scipy.stats.ttest_1samp(np.mean(slope_UP[:,[4,5,6,7,8,9]], axis = 1),100)
print(f'slope UP mean {np.mean(np.mean(slope_UP[:,[4,5,6,7,8,9]], axis = 1))}')
print(f'slope UP std {np.std(np.mean(slope_UP[:,[4,5,6,7,8,9]], axis = 1), ddof = 1)}') 
scipy.stats.shapiro(np.mean(slope_UP[:,[4,5,6,7,8,9]], axis = 1))

scipy.stats.ttest_1samp(np.mean(slope_DOWN[:,[4,5,6,7,8,9]], axis = 1),100)
print(f'slope DOWN mean {np.mean(np.mean(slope_DOWN[:,[4,5,6,7,8,9]], axis = 1))}')
print(f'slope DOWN std {np.std(np.mean(slope_DOWN[:,[4,5,6,7,8,9]], axis = 1), ddof = 1)}') 
scipy.stats.shapiro(np.mean(slope_DOWN[:,[4,5,6,7,8,9]], axis = 1))



#%% LFP, MUA and LFP slope change across all channels with non parametric test (not normally distributed across channels)
# Basically the distribution is not normal so account for that. Make sure it is normal across mice though!

os.chdir(os.path.join(overall_path, r'UP_pairing'))
allchans_LFP_UP = np.hstack(pickle.load(open('distance_vs_plasticity_LFP_ALL.pkl','rb')))#
allchans_LFP_slope_UP = np.hstack(pickle.load(open('distance_vs_plasticity_LFP_slope_ALL.pkl','rb')))#
allchans_PSTH_magn_UP = np.hstack(pickle.load(open('distance_vs_plasticity_spike_magn_ALL.pkl','rb')))
os.chdir('..')
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
allchans_LFP_DOWN = np.hstack(pickle.load(open('distance_vs_plasticity_LFP_ALL.pkl','rb')))#
allchans_LFP_slope_DOWN = np.hstack(pickle.load(open('distance_vs_plasticity_LFP_slope_ALL.pkl','rb')))#
allchans_PSTH_magn_DOWN = np.hstack(pickle.load(open('distance_vs_plasticity_spike_magn_ALL.pkl','rb')))
os.chdir('..')

# bins = 100
# plt.bar(np.histogram(allchans_LFP_DOWN, bins = bins)[1][:-1], np.histogram(allchans_LFP_DOWN, bins = bins)[0], width = 0.005)

# none of the channel groups are normally distributed
scipy.stats.shapiro(allchans_LFP_UP[1,:])

# one sample against 0:
scipy.stats.wilcoxon(allchans_LFP_DOWN[1,:])

# non parametric tests against each other
print(f'LFP change across channels: {scipy.stats.mannwhitneyu(allchans_LFP_UP[1,:], allchans_LFP_DOWN[1,:])}, UP/DOWN n {allchans_LFP_UP.shape[1]}, {allchans_LFP_DOWN.shape[1]}')
print(f'PSTH change across channels: {scipy.stats.mannwhitneyu(allchans_PSTH_magn_UP[1,:], allchans_PSTH_magn_DOWN[1,:])}, UP/DOWN n {allchans_PSTH_magn_UP.shape[1]}, {allchans_PSTH_magn_DOWN.shape[1]}')
print(f'slope change across channels: {scipy.stats.mannwhitneyu(allchans_LFP_slope_UP[1,:], allchans_LFP_slope_DOWN[1,:][~np.isnan(allchans_LFP_slope_DOWN[1,:])])}')

# box and whisker plot LFP and PSTH
fig, ax = plt.subplots(figsize = (4,4))
plot = ax.boxplot((allchans_LFP_UP[1,:]*100, allchans_LFP_DOWN[1,:]*100), notch = True, showfliers = True, widths = 0.25)
plt.axhline(0, color = 'k', linestyle = '--')
ax.set_xticklabels(['UP-pairing', 'DOWN-pairing'], size = 16)
ax.set_ylabel('LFP plasticity (%)', size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
ax.set_yticks([-80,-40,0,40, 80])
ax.set_ylim([-90, 110])
ax.set_yticklabels(list(map(str,ax.get_yticks())), size = 16)
plt.tight_layout()
plt.savefig('LFP UP vs DOWN all channels.jpg', dpi = 1000, format = 'jpg')
plt.savefig('LFP UP vs DOWN all channels.pdf', dpi = 1000, format = 'pdf')

fig, ax = plt.subplots(figsize = (4,4))
ax.boxplot((allchans_PSTH_magn_UP[1,:]*100, allchans_PSTH_magn_DOWN[1,:]*100), notch = True, showfliers = True, widths = 0.25)
plt.axhline(0, color = 'k', linestyle = '--')
plt.axhline(0, color = 'k', linestyle = '--')
ax.set_xticklabels(['UP-pairing', 'DOWN-pairing'], size = 16)
ax.set_ylabel('LFP plasticity (%)', size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
ax.set_yticks([-80,-40,0,40,80,120])
ax.set_ylim([-90, 140])
ax.set_yticklabels(list(map(str,ax.get_yticks())), size = 16)
plt.tight_layout()
plt.savefig('PSTH UP vs DOWN all channels.jpg', dpi = 1000, format = 'jpg')
plt.savefig('PSTH UP vs DOWN all channels.pdf', dpi = 1000, format = 'pdf')

# fig, ax = plt.subplots()
# ax.boxplot((allchans_LFP_slope_UP[1,:], allchans_LFP_slope_DOWN[1,:][~np.isnan(allchans_LFP_slope_DOWN[1,:])]), notch = True, showfliers = False)
# plt.axhline(0, color = 'k', linestyle = '--')



#%% distance from principal channel vs plasticity

os.chdir(os.path.join(overall_path, r'UP_pairing'))
distance_vs_plasticity_LFP_ALL = pickle.load(open('distance_vs_plasticity_LFP_ALL.pkl','rb'))
distance_vs_plasticity_LFP_slope_ALL = pickle.load(open('distance_vs_plasticity_LFP_slope_ALL.pkl','rb'))
distance_vs_plasticity_spike_peak_ALL = pickle.load(open('distance_vs_plasticity_spike_peak_ALL.pkl','rb'))
distance_vs_plasticity_spike_magn_ALL = pickle.load(open('distance_vs_plasticity_spike_magn_ALL.pkl','rb'))

# for measure in [distance_vs_plasticity_LFP_ALL, distance_vs_plasticity_LFP_slope_ALL, distance_vs_plasticity_spike_peak_ALL, distance_vs_plasticity_spike_magn_ALL]:
for measure in [distance_vs_plasticity_LFP_ALL]:
    # fig.suptitle(f'')
    to_plot = np.hstack(measure)
    to_plot[0,:] = to_plot[0,:]*200
    to_plot[1,:] = to_plot[1,:]*100
    
    # take out outliers
    to_plot = np.delete(to_plot, np.where(to_plot[1,:] > (np.percentile(to_plot[1,:], 75) + 1.2*(np.abs(np.percentile(to_plot[1,:], 75) - np.percentile(to_plot[1,:], 25)))))[0], axis = 1)
    
    # only up to 1mm distance
    # to_plot = np.delete(to_plot, np.where(to_plot[0,:] > 1000), axis = 1)
    
    # only channels showing depression
    # to_plot = np.delete(to_plot, np.where(to_plot[1,:] > 0), axis = 1)
    
    slope, intercept, r, p, std_err = stats.linregress(to_plot[0,:], to_plot[1,:])
    
    fig, ax = plt.subplots(figsize = (6,3.7))
    ax.scatter(to_plot[0,:], to_plot[1,:], color = 'k', s = 6)
    ax.plot([np.min(to_plot[0,:]), np.max(to_plot[0,:])], [(slope*np.min(to_plot[0,:]) + intercept), (slope*np.max(to_plot[0,:]) + intercept)], color = 'k')
    ax.set_xlabel('distance from principal barrel (ym)', size = 16)
    ax.set_ylabel('LFP plasticity (%)', size = 16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.set_xticks([0,500,1000,1500])
    ax.set_xticklabels(list(map(str,[0,500,1000,1500])), size = 16)
    ax.set_yticks([-80,-60,-40,-20,0,20])
    ax.set_yticklabels(list(map(str,[-80,-60,-40,-20,0,20])), size = 16)
    plt.tight_layout()
    # plt.savefig('LFP vs distance.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('LFP vs distance.jpg', dpi = 1000, format = 'jpg')
    print(f'{r} and {p} for {len(to_plot[0,:])} channels')
    # plt.savefig('LFP plasticity versus distance from principal channel')
    
    
#%% LFP plasticity vs slope change

os.chdir(os.path.join(overall_path, r'UP_pairing'))
channels = np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool)
LFP_min_rel_change_ALLCHANS = np.loadtxt('LFP_min_rel_change_ALLCHANS.csv', delimiter = ',')
LFP_slope_rel_change_ALLCHANS = np.loadtxt('LFP_slope_rel_change_ALLCHANS.csv', delimiter = ',')


exclude_outliers = True
# X = copy.deepcopy(delta_power_rel_change_ALLCHANS)
X = copy.deepcopy(LFP_slope_rel_change_ALLCHANS[channels])*100
# X_outliers = np.where(X > (np.median(X) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]

# Y = copy.deepcopy(LFP_min_rel_change_ALLCHANS)
Y = copy.deepcopy(LFP_min_rel_change_ALLCHANS[channels])*100
# Y_outliers = np.where(Y > (np.median(Y) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]

Y = Y[np.where(~np.isnan(X))]
X = X[np.where(~np.isnan(X))]

X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
if exclude_outliers:
    X = np.delete(X, outliers)
    Y = np.delete(Y, outliers)

fig, ax = plt.subplots(figsize = (6,3.7))
slope, intercept, r, p, std_err = stats.linregress(X, Y)
print(f'{r} and {p} for {len(X)} channels')
ax.scatter(X,Y, color = 'k', s = 6)
ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
ax.set_xlabel('LFP slope change (%)', size = 16)
ax.set_ylabel('LFP magnitude change (%)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
ax.set_xticks([-80,-60,-40,-20,0,20,40])
ax.set_xticklabels(list(map(str, ax.get_xticks())), size = 16)
ax.set_yticks([-80,-60,-40,-20,0,20])
ax.set_yticklabels(list(map(str, [-80,-60,-40,-20,0,20])), size = 16)
plt.tight_layout()
plt.savefig('LFP vs slope.pdf', dpi = 1000, format = 'pdf')
plt.savefig('LFP vs slope.jpg', dpi = 1000, format = 'jpg')



#%% pairing success vs depression (inconclusive)

os.chdir(os.path.join(overall_path, r'UP_pairing'))
channels = ((np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int) + np.loadtxt('SW_spiking_channels_ALLCHANS.csv', delimiter = ',', dtype = int)) == 2)
# channels = np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int)

exclude_outliers = True
normalize_within_mouse = True

if normalize_within_mouse == True:
    X = np.loadtxt('pairing_UP_freq_first_stims_ALLCHANS.csv', delimiter = ',')
    Y = np.loadtxt('LFP_min_rel_change_ALLCHANS.csv', delimiter = ',')
    X_norm = []
    Y_norm = []
    for mouse in range(13):
        curr_X = X.reshape(13,64)[mouse,:][channels.reshape(13,64)[mouse, :]]
        X_norm.append(curr_X/np.max(curr_X))
        curr_Y = Y.reshape(13,64)[mouse,:][channels.reshape(13,64)[mouse, :]]
        Y_norm.append(curr_Y/np.min(curr_Y))
    X = np.concatenate(X_norm)
    Y = np.concatenate(Y_norm)

else:
    X = np.loadtxt('pairing_UP_freq_first_stims_ALLCHANS.csv', delimiter = ',')[channels]
    Y = np.loadtxt('LFP_min_rel_change_ALLCHANS.csv', delimiter = ',')[channels]

Y = Y[np.where(~np.isnan(X))]
X = X[np.where(~np.isnan(X))]

X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
# Y_outliers = np.where(Y < (np.percentile(Y, 75) - 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
Y_outliers = np.where(Y < 0)[0]
outliers = np.unique(np.concatenate((X_outliers, Y_outliers))).astype(int)
if exclude_outliers:
    X = np.delete(X, outliers)
    Y = np.delete(Y, outliers)


fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(X, Y)
print(f'{r} and {p} for {len(X)} channels')
ax.scatter(X,Y)
ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)])
ax.set_xlabel('pairing')
ax.set_ylabel('LFP change')

print(f'{r} and {p} for {len(X)} channels')

# ax.set_xlabel('distance from principal barrel (ym)', size = 16)
# ax.set_ylabel('LFP plasticity (%)', size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xticks([0,500,1000,1500])
# ax.set_xticklabels(list(map(str,[0,500,1000,1500])), size = 16)
# ax.set_yticks([-80,-60,-40,-20,0,20])
# ax.set_yticklabels(list(map(str,[-80,-60,-40,-20,0,20])), size = 16)
# plt.tight_layout()
# plt.savefig('LFP vs distance.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('LFP vs distance.jpg', dpi = 1000, format = 'jpg')



#%% UP vs DOWN delivery: no change in frequency of delivery, LFP and spiking response: no interaction between group and time

# distribution of frequencies of UP/DOWN deliveries over time across all UP mice with one way ANOVA
os.chdir(os.path.join(overall_path, r'UP_pairing'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
numb = len(days)
UP_vs_DOWN_deliveries_ALL = np.zeros([numb, 10])
UP_vs_DOWN_deliveries_ALL[:] = np.NaN

for day_ind, day in enumerate(days):
    os.chdir(day)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    UP_vs_DOWN_deliveries_ALL[day_ind, :] = np.loadtxt('UP_stims_freq.csv', delimiter = ',')*100
    os.chdir('..')
    os.chdir('..')

os.chdir(overall_path)
#plot frequency of deliveries
fig, ax = plt.subplots()
yerror = [np.std(UP_vs_DOWN_deliveries_ALL[:,i]) for i in range(10)]/np.sqrt(numb)
ax.bar(np.linspace(1,10,10), [np.mean(UP_vs_DOWN_deliveries_ALL[:,i]) for i in range(10)], yerr = yerror, capsize = 5)
ax.set_ylim([0,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axvline(4.5, linestyle = '--')
ax.set_xlabel('time from pairing (min)', size = 16)
ax.set_xticks([1,3,6,8,10])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 16)
ax.set_ylabel('proportion of stimulations \n delivered during UP states (%)', size = 16)
ax.set_yticks([0,20,40,60])
ax.set_yticklabels(list(map(str,ax.get_yticks())), size = 16)
plt.tight_layout()
plt.savefig('UP vs DOWN deliveries proportions.pdf', dpi = 1000, format = 'pdf')
plt.savefig('UP vs DOWN deliveries proportions.jpg', dpi = 1000, format = 'jpg')

print(f'{stats.f_oneway(*[UP_vs_DOWN_deliveries_ALL[:,i] for i in range(10)])}')

# get data about stims delivered during UP vs DOWN states
os.chdir(os.path.join(overall_path, r'UP_pairing'))
LFP_min_UP_median_ALL = np.loadtxt('LFP_min_UP_median_ALL.csv', delimiter = ',')
LFP_min_DOWN_median_ALL = np.loadtxt('LFP_min_DOWN_median_ALL.csv', delimiter = ',')
LFP_min_UP_rel_median_ALL = np.loadtxt('LFP_min_UP_rel_median_ALL.csv', delimiter = ',')*100
LFP_min_DOWN_rel_median_ALL = np.loadtxt('LFP_min_DOWN_rel_median_ALL.csv', delimiter = ',')*100
PSTH_magn_UP_median_ALL = np.loadtxt('PSTH_magn_UP_median_ALL.csv', delimiter = ',')
PSTH_magn_DOWN_median_ALL = np.loadtxt('PSTH_magn_DOWN_median_ALL.csv', delimiter = ',')
PSTH_magn_UP_rel_median_ALL = np.loadtxt('PSTH_magn_UP_rel_median_ALL.csv', delimiter = ',')*100
PSTH_magn_DOWN_rel_median_ALL = np.loadtxt('PSTH_magn_DOWN_rel_median_ALL.csv', delimiter = ',')*100
PSTH_peak_UP_median_ALL = np.loadtxt('PSTH_peak_UP_median_ALL.csv', delimiter = ',')
PSTH_peak_DOWN_median_ALL = np.loadtxt('PSTH_peak_DOWN_median_ALL.csv', delimiter = ',')
PSTH_peak_UP_rel_median_ALL = np.loadtxt('PSTH_peak_UP_rel_median_ALL.csv', delimiter = ',')*100
PSTH_peak_DOWN_rel_median_ALL = np.loadtxt('PSTH_peak_DOWN_rel_median_ALL.csv', delimiter = ',')*100

os.chdir(overall_path)
# LFP
patch = 1
fig, ax = plt.subplots(figsize = (10,4))
ax.plot([np.nanmean(LFP_min_UP_rel_median_ALL[:,i]) for i in range(10)], c = 'r')
ax.fill_between(list(range(10)), np.nanmean(LFP_min_UP_rel_median_ALL, axis = 0) + patch*np.nanstd(LFP_min_UP_rel_median_ALL, axis = 0)/np.sqrt(LFP_min_UP_rel_median_ALL.shape[0]), np.nanmean(LFP_min_UP_rel_median_ALL, axis = 0) - patch*np.nanstd(LFP_min_UP_rel_median_ALL, axis = 0)/np.sqrt(LFP_min_UP_rel_median_ALL.shape[0]), alpha = 0.1, color = 'r')
ax.plot([np.nanmean(LFP_min_DOWN_rel_median_ALL[:,i]) for i in range(10)], c = 'k')
ax.fill_between(list(range(10)), np.nanmean(LFP_min_DOWN_rel_median_ALL, axis = 0) + patch*np.nanstd(LFP_min_DOWN_rel_median_ALL, axis = 0)/np.sqrt(LFP_min_DOWN_rel_median_ALL.shape[0]), np.nanmean(LFP_min_DOWN_rel_median_ALL, axis = 0) - patch*np.nanstd(LFP_min_DOWN_rel_median_ALL, axis = 0)/np.sqrt(LFP_min_DOWN_rel_median_ALL.shape[0]), alpha = 0.1, color = 'k')
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 18)
ax.set_ylabel('LFP response \n (% of baseline)', size = 18)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 18)
ax.set_yticks([60,80,100,120])
ax.set_yticklabels(list(map(str,ax.get_yticks())), size = 18)
ax.set_ylim([55, 130])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('LFP stims UP vs DOWN delivery.pdf', dpi = 1000, format = 'pdf')
plt.savefig('LFP stims UP vs DOWN delivery.jpg', dpi = 1000, format = 'jpg')

# MUA magnitude
patch = 1
fig, ax = plt.subplots(figsize = (10,4))
to_plot_1 = copy.deepcopy(PSTH_magn_UP_rel_median_ALL)
to_plot_2 = copy.deepcopy(PSTH_magn_DOWN_rel_median_ALL)
take_out = [[0,0], [5,0], [10,0]] # spikes did not reach baseline yet or noise (221220_3)
for sweep in take_out:
    to_plot_1[sweep[0], sweep[1]] = np.NaN
    to_plot_2[sweep[0], sweep[1]] = np.NaN
ax.plot([np.nanmean(to_plot_1[:,i]) for i in range(10)], c = 'r')
ax.fill_between(list(range(10)), np.nanmean(to_plot_1, axis = 0) + patch*np.nanstd(to_plot_1, axis = 0)/np.sqrt(to_plot_1.shape[0]), np.nanmean(to_plot_1, axis = 0) - patch*np.nanstd(to_plot_1, axis = 0)/np.sqrt(to_plot_1.shape[0]), alpha = 0.1, color = 'r')
ax.plot([np.nanmean(to_plot_2[:,i]) for i in range(10)], c = 'k')
ax.fill_between(list(range(10)), np.nanmean(to_plot_2, axis = 0) + patch*np.nanstd(to_plot_2, axis = 0)/np.sqrt(to_plot_2.shape[0]), np.nanmean(to_plot_2, axis = 0) - patch*np.nanstd(to_plot_2, axis = 0)/np.sqrt(to_plot_2.shape[0]), alpha = 0.1, color = 'k')
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 18)
ax.set_ylabel('MUA response \n (% of baseline)', size = 18)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 18)
ax.set_yticks([60,80,100,120])
ax.set_yticklabels(list(map(str,ax.get_yticks())), size = 18)
ax.set_ylim([50, 130])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('PSTH magn stims UP vs DOWN delivery.pdf', dpi = 1000, format = 'pdf')
plt.savefig('PSTH magn stims UP vs DOWN delivery.jpg', dpi = 1000, format = 'jpg')

# difference UP vs DOWN deliveries in magnitude during baseline:
differences_LFP = (np.nanmean(LFP_min_UP_median_ALL[:,[1,2,3]], axis = 1) - np.nanmean(LFP_min_DOWN_median_ALL[:,[1,2,3]], axis = 1))/np.nanmean(LFP_min_DOWN_median_ALL[:,[1,2,3]], axis = 1)
differences_PSTH = (np.nanmean(PSTH_magn_UP_median_ALL[:,[1,2,3]], axis = 1) - np.nanmean(PSTH_magn_DOWN_median_ALL[:,[1,2,3]], axis = 1))/np.nanmean(PSTH_magn_DOWN_median_ALL[:,[1,2,3]], axis = 1)
#normally distributed?
scipy.stats.shapiro(differences_LFP)
# one sample against 0:
scipy.stats.ttest_1samp(differences_PSTH,0)

print(1 + np.mean(differences_LFP))    
print(np.std(differences_LFP, ddof = 1)) # report sample std so ddof of 1

print(1 + np.mean(differences_PSTH))    
print(np.std(differences_PSTH, ddof = 1)) # report sample std so ddof of 1


#%% LFP VS DELTA CHANGE

os.chdir(os.path.join(overall_path, r'UP_pairing'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
numb = len(days)

channels = np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool)
# channels[9*63:10*64] = False
# channels[:] = True

exclude_outliers = True
normalize_within_mouse = False
only_LFP_depression = False
only_delta_depression = False

if normalize_within_mouse == True:
    # X = np.loadtxt('delta_power_rel_change_ALLCHANS.csv', delimiter = ',')*100
    X = np.loadtxt('delta_power_auto_outliers_rel_change_ALLCHANS.csv', delimiter = ',')*100
    Y = np.loadtxt('LFP_min_rel_change_ALLCHANS.csv', delimiter = ',')*100
    X_norm = []
    Y_norm = []
    for mouse in range(numb):
        curr_X = X.reshape(numb,64)[mouse,:][channels.reshape(numb,64)[mouse, :]]
        X_norm.append(curr_X/np.min(curr_X))
        curr_Y = Y.reshape(numb,64)[mouse,:][channels.reshape(numb,64)[mouse, :]]
        Y_norm.append(curr_Y/np.min(curr_Y))
    X = np.concatenate(X_norm)
    Y = np.concatenate(Y_norm)
    
else:
    # X = copy.deepcopy(delta_power_rel_change_ALLCHANS)
    # X = copy.deepcopy(np.loadtxt('delta_power_rel_change_ALLCHANS.csv', delimiter = ',')[channels])*100
    X = np.loadtxt('delta_power_auto_outliers_rel_change_ALLCHANS.csv', delimiter = ',')[channels]*100
    # X = copy.deepcopy(np.loadtxt('delta_power_rel_change_ALLCHANS.csv', delimiter = ',')[:])*100
    
    # Y = copy.deepcopy(LFP_min_rel_change_ALLCHANS)
    Y = np.loadtxt('LFP_min_rel_change_ALLCHANS.csv', delimiter = ',')[channels]*100
    # Y = copy.deepcopy(np.loadtxt('LFP_min_rel_change_ALLCHANS.csv', delimiter = ',')[:])*100

mask = ~np.isnan(X) & ~np.isnan(Y)
X = X[mask]
Y = Y[mask]

if only_LFP_depression:
    X = X[Y<0]
    Y = Y[Y<0]

if only_delta_depression:
    Y = Y[X<0]
    X = X[X<0]

# X_outliers = np.where(X > (np.median(X) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
# X_outliers = np.where(X < (np.percentile(X, 75) - 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]

# X_outliers = np.where(X > 0)[0]

# Y_outliers = np.where(Y > (np.median(Y) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
# Y_outliers = np.where(Y < (np.percentile(Y, 75) - 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]

# Y_outliers = np.where(Y > 0)[0]

outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))

if exclude_outliers:
    X = np.delete(X, outliers)
    Y = np.delete(Y, outliers)

fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(X, Y)
print(f'{r**2} and {p} for {len(X)} channels')
ax.scatter(X,Y, color = 'k')
ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
# ax.axhline(np.mean(Y))
ax.set_xlabel('delta power change (% baseline)', size = 16)
ax.set_ylabel('LFP change (% baseline)', size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
ax.tick_params(axis="x", labelsize=16)    
ax.tick_params(axis="y", labelsize=16) 
plt.tight_layout()
plt.savefig('delta vs LFP.pdf', dpi = 1000, format = 'pdf')
plt.savefig('delta vs LFP.jpg', dpi = 1000, format = 'jpg')

# # plot with color code for each mouse
# mice = [0,1,2,3,4,5,6,7,8,9,10,11]
# color = cm.gist_rainbow(np.linspace(0, 1, len(mice)))
# X = np.loadtxt('delta_power_rel_change_ALLCHANS.csv', delimiter = ',')*100
# Y = np.loadtxt('LFP_min_rel_change_ALLCHANS.csv', delimiter = ',')*100
# fig, ax = plt.subplots()
# ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
# # for mouse in range(numb):
# for mouse_ind, mouse in enumerate(mice):
#     curr_X = X.reshape(numb,64)[mouse,:][channels.reshape(numb,64)[mouse, :]]
#     curr_Y = Y.reshape(numb,64)[mouse,:][channels.reshape(numb,64)[mouse, :]]
#     ax.scatter(curr_X,curr_Y, label = mouse, color = color[mouse_ind])
# ax.set_xlabel('delta power change (% baseline)', size = 16)
# ax.set_ylabel('LFP change (% baseline)', size = 16)
# plt.legend()
# ax.set_xlim([-100,100])
# ax.set_ylim([-80,100])

# interactive scatter plot
# os.chdir(os.path.join(overall_path, r'UP_pairing'))
# days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# numb = len(days)
# channels = np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool)
# X = np.loadtxt('delta_power_auto_outliers_rel_change_ALLCHANS.csv', delimiter = ',')[channels]*100
# Y = np.loadtxt('LFP_min_rel_change_ALLCHANS.csv', delimiter = ',')[channels]*100

# channel_indices = np.argwhere(channels.reshape(numb,64) == True)
# chans_cumsum = np.sum(channels.reshape(numb,64), axis = 1)
# chans_cumsum = np.cumsum(chans_cumsum)
# chans_cumsum = np.insert(chans_cumsum, 0, 0)
# channels_cumsum = [np.arange(chans_cumsum[i], chans_cumsum[i+1]) for i in range(len(chans_cumsum) -1)]

# fig, ax = plt.subplots()
# slope, intercept, r, p, std_err = stats.linregress(X, Y)
# ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')[0]
# b = ax.scatter(X,Y, color = 'k')
# c = ax.axhline(np.mean(Y))
# ax.set_xlabel('delta power change (% baseline)', size = 16)
# ax.set_ylabel('LFP change (% baseline)', size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# reds = []
# print(f'{r**2} and {p} for {len(X)} channels')
# def add_or_remove_point(event):
#     # global channels
#     # global channels_cumsum
#     global reds
#     global b
#     xydata_b = b.get_offsets()
#     xdata_b = b.get_offsets()[:,0]
#     ydata_b = b.get_offsets()[:,1]
#     # print(xdata_b)
    
#     #click value
#     xdata_click = event.xdata
#     ydata_click = event.ydata

#     if event.button == 1:
#         plt.clf()                         

#         #index of nearest value (euclidean distance)
#         xdata_nearest_index_b = np.sqrt((xdata_click - np.asarray([i[0] for i in reds]))**2 + (ydata_click - np.asarray([i[1] for i in reds]))**2).argmin()
#         # xdata_nearest_index_b = event.ind[0]
#         print(xdata_nearest_index_b)
#         new_xydata_b = np.vstack([xydata_b, reds[xdata_nearest_index_b]])
#         new_X = new_xydata_b[:,0]
#         new_Y = new_xydata_b[:,1]

#         #update scatter plot
#         b = plt.scatter(new_X, new_Y, color = 'k')
        
#         #update regression
#         slope, intercept, r, p, std_err = stats.linregress(new_X, new_Y)
#         print(f'{r**2} and {p} for {len(new_X)} channels')
#         plt.plot([np.min(new_X), np.max(new_X)], [(slope*np.min(new_X) + intercept), (slope*np.max(new_X) + intercept)], color = 'k')
#         plt.axhline(np.mean(new_Y))        
        
#         #update reds
#         # print(reds[xdata_nearest_index_b])
#         reds.remove(reds[xdata_nearest_index_b])
#         for red in reds:
#             plt.plot(red[0], red[1], 'ro', color = 'red')

#     if event.button == 3:
#         plt.clf()                         

#         #index of nearest value (euclidean distance)
#         xdata_nearest_index_b = np.sqrt((xdata_b-xdata_click)**2 + (ydata_b-ydata_click)**2).argmin()
#         print(xdata_nearest_index_b)
#         # xdata_nearest_index_b = event.ind[0]

#         #shade xdata point and put it back in as an individual point
#         reds.append([xydata_b[xdata_nearest_index_b,0], xydata_b[xdata_nearest_index_b,1]])
#         for red in reds:
#             plt.plot(red[0], red[1], 'ro', color = 'red')
        
#         #remove xdata point and redo regression
#         new_xydata_b = np.delete(xydata_b, xdata_nearest_index_b, axis=0)
#         new_X = new_xydata_b[:,0]
#         new_Y = new_xydata_b[:,1]

#         #update scatter plot
#         b = plt.scatter(new_X, new_Y, color = 'k')
        
#         #update regression
#         slope, intercept, r, p, std_err = stats.linregress(new_X, new_Y)
#         print(f'{r**2} and {p} for {len(new_X)} channels')
#         plt.plot([np.min(new_X), np.max(new_X)], [(slope*np.min(new_X) + intercept), (slope*np.max(new_X) + intercept)], color = 'k')
#         plt.axhline(np.mean(new_Y))            
        
#         # print mouse and channel number CAVE only works on not outliers taken out
#         channel_ind = np.argwhere(X == xydata_b[xdata_nearest_index_b, 0])[0][0]
#         mouse_ind = [i for i in range(len(channels_cumsum)) if channel_ind in channels_cumsum[i]][0]
#         print(f'mouse {days[mouse_ind]}, channel {channel_indices[channel_ind][1]}')
        
# fig.canvas.mpl_connect('button_press_event',add_or_remove_point)

#%% PSD before and after UP and DOWN pairing

# frequencies on 3.5 second interstim interval
fftfreq = np.fft.fftfreq(3500, d = (1/1000))
fftfreq_to_plot = fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]][:]

patch = 1
os.chdir(os.path.join(overall_path, r'UP_pairing'))
chans_to_plot = np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool).reshape(12,64)
PSD_UP = pickle.load(open('PSD_ALL.pkl', 'rb'))
PSD_UP = np.log(np.asarray([np.squeeze(np.median(i[:,chans_to_plot[ind,:],:], axis = 1)) for ind, i in enumerate(PSD_UP)])/3.5/100000) # average across channels, divide by time window
PSD_UP_before_mean = np.mean(PSD_UP[:,[0,1,2,3],:], axis = 1)
PSD_UP_after_mean = np.mean(PSD_UP[:,[4,5,6,7,8,9],:], axis = 1)

os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
chans_to_plot = np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool).reshape(12,64)
PSD_DOWN = pickle.load(open('PSD_ALL.pkl', 'rb'))
PSD_DOWN = np.log(np.asarray([np.squeeze(np.median(i[:,chans_to_plot[ind,:],:], axis = 1)) for ind, i in enumerate(PSD_DOWN)])/3.5/100000) # average across channels
PSD_DOWN_before_mean = np.mean(PSD_DOWN[:,[0,1,2,3],:], axis = 1)
PSD_DOWN_after_mean = np.mean(PSD_DOWN[:,[4,5,6,7,8,9],:], axis = 1)
os.chdir('..')


mice_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11]
fig, ax = plt.subplots() 
# UP
#before
mice_avg = smooth(np.mean(PSD_UP_before_mean[mice_to_plot,:], axis = 0),3) # average and SEM across mice.
mice_std = smooth(np.std(PSD_UP_before_mean[mice_to_plot,:], axis = 0),3)
ax.plot(fftfreq_to_plot, mice_avg, c = 'r')
ax.fill_between(fftfreq_to_plot, mice_avg + patch*mice_std/np.sqrt(len(mice_to_plot)), mice_avg - patch*mice_std/np.sqrt(len(mice_to_plot)), alpha = 0.1, color = 'r')
#after
mice_avg = smooth(np.mean(PSD_UP_after_mean[mice_to_plot,:], axis = 0),3)
mice_std = smooth(np.std(PSD_UP_after_mean[mice_to_plot,:], axis = 0),3)
ax.plot(fftfreq_to_plot,mice_avg, c = 'r', linestyle = '--')
ax.fill_between(fftfreq_to_plot, mice_avg + patch*mice_std/np.sqrt(len(mice_to_plot)), mice_avg - patch*mice_std/np.sqrt(len(mice_to_plot)), alpha = 0.1, color = 'r')


fig, ax = plt.subplots() 
#DOWN
#before
# ax.plot(fftfreq_to_plot, PSD_DOWN_before_mean.T)
mice_avg = smooth(np.mean(PSD_DOWN_before_mean[mice_to_plot,:], axis = 0),3)
mice_std = smooth(np.std(PSD_DOWN_before_mean[mice_to_plot,:], axis = 0),3)
ax.plot(fftfreq_to_plot, mice_avg, c = 'k') 
ax.fill_between(fftfreq_to_plot, mice_avg + patch*mice_std/np.sqrt(len(mice_to_plot)), mice_avg - patch*mice_std/np.sqrt(len(mice_to_plot)), alpha = 0.1, color = 'k')
#after
mice_avg = smooth(np.mean(PSD_DOWN_after_mean[mice_to_plot,:], axis = 0),3)
mice_std = smooth(np.std(PSD_DOWN_after_mean[mice_to_plot,:], axis = 0),3)
ax.plot(fftfreq_to_plot,mice_avg, c = 'k', linestyle = '--')
ax.fill_between(fftfreq_to_plot, mice_avg + patch*mice_std/np.sqrt(len(mice_to_plot)), mice_avg - patch*mice_std/np.sqrt(len(mice_to_plot)), alpha = 0.1, color = 'k')
# ax.set_yscale('log')
# ax.set_ylim([])
ax.set_xlim([0.25,6])
ax.set_ylim([6,12])



                      
#%% DELTA IN LFP RESPONSIVE VS SW SPIKING BUT NOT LFP RESPONSIVE (inconclusive)

os.chdir(os.path.join(overall_path, r'UP_pairing'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
numb = len(days)
delta_power_auto_outliers_rel_ALL = np.loadtxt('delta_power_auto_outliers_rel_ALL.csv', delimiter = ',')
delta_power_auto_outliers_rel_change_ALLCHANS = np.loadtxt('delta_power_auto_outliers_rel_change_ALLCHANS.csv', delimiter = ',')

delta_change_LFP_group = []
delta_change_not_LFP_group = []
for day_ind, day in enumerate(days):
    os.chdir(day)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    # SW spiking channels that aren't LFP responsive channels
    LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
    SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',', dtype = int)
    not_LFP_resp_channels_cutoff = np.asarray([i for i in SW_spiking_channels if i not in LFP_resp_channels_cutoff])
    
    # # take out noisy non-LFP responsive channels    
    # # ------------------------------------- UP - Pairing
    if day == '121121':
        LFP_bad = [21,25] # noisy channels
        for chan in LFP_bad:  
            not_LFP_resp_channels_cutoff = np.delete(not_LFP_resp_channels_cutoff, np.where(not_LFP_resp_channels_cutoff == chan)[0])                 
    if day == '160310':
        LFP_bad = [41,9,57,8,11,59,58,16,32,17] # noisy
        for chan in LFP_bad:  
            not_LFP_resp_channels_cutoff = np.delete(not_LFP_resp_channels_cutoff, np.where(not_LFP_resp_channels_cutoff == chan)[0])                 
    if day == '160426_D1':
        LFP_bad = [45,43,57] # noisy
        for chan in LFP_bad:  
            not_LFP_resp_channels_cutoff = np.delete(not_LFP_resp_channels_cutoff, np.where(not_LFP_resp_channels_cutoff == chan)[0])         
    if day == '160426_D1':
        LFP_bad = [9,57,25,14,17,13,11] # noisy or inverted LFP responses (most likely in layer 1/on dura)
        for chan in LFP_bad:  
            not_LFP_resp_channels_cutoff = np.delete(not_LFP_resp_channels_cutoff, np.where(not_LFP_resp_channels_cutoff == chan)[0])                 
    if day == '160519_B2':
        LFP_bad = [36,33,56,53,4] # noisy or inverted LFP responses (most likely in layer 1/on dura)
        for chan in LFP_bad:  
            not_LFP_resp_channels_cutoff = np.delete(not_LFP_resp_channels_cutoff, np.where(not_LFP_resp_channels_cutoff == chan)[0])                 
    if day == '160624_B2':
        LFP_bad = [37,35,17,19] # noisy
        for chan in LFP_bad:  
            not_LFP_resp_channels_cutoff = np.delete(not_LFP_resp_channels_cutoff, np.where(not_LFP_resp_channels_cutoff == chan)[0])                 
    # if day == '160628_D1':
    # if day == '191121':
    if day == '201121':
        LFP_bad = [30,5,] # weird response shape
        for chan in LFP_bad:  
            not_LFP_resp_channels_cutoff = np.delete(not_LFP_resp_channels_cutoff, np.where(not_LFP_resp_channels_cutoff == chan)[0])                 
    # if day == '221220_3':
    if day == '281021':
        LFP_bad = [30,18,1,49,45,25] # noisy
        for chan in LFP_bad:  
            not_LFP_resp_channels_cutoff = np.delete(not_LFP_resp_channels_cutoff, np.where(not_LFP_resp_channels_cutoff == chan)[0]) 
    # if day == '291021':
    #     LFP_bad = [] # noisy
    #     for chan in LFP_bad:  
    #         not_LFP_resp_channels_cutoff = np.delete(not_LFP_resp_channels_cutoff, np.where(not_LFP_resp_channels_cutoff == chan)[0]) 
      
    if len(not_LFP_resp_channels_cutoff) > 0:
        delta_change_LFP = np.mean(delta_power_auto_outliers_rel_change_ALLCHANS[day_ind*64 + LFP_resp_channels_cutoff])
        delta_change_LFP_group.append(delta_change_LFP)
        delta_change_not_LFP = np.mean(delta_power_auto_outliers_rel_change_ALLCHANS[day_ind*64 + not_LFP_resp_channels_cutoff]) 
        delta_change_not_LFP_group.append(delta_change_not_LFP)
        print(f'{day} delta change LFP vs not LFP: {delta_change_LFP} vs {delta_change_not_LFP} in {LFP_resp_channels_cutoff.size} vs {not_LFP_resp_channels_cutoff.size}')
    
    os.chdir('..')
    os.chdir('..')

print(np.mean(delta_change_LFP_group))
print(np.mean(delta_change_not_LFP_group))

    # if day == '291021':

    # # --------------------- DOWN - Pairing
    # if day == '061221': # the spiking response in this mouse is a bit buggy - it changes shape drastically in several channels. There is a case for excluding it from the spiking response analysis.
    #     LFP_bad = [14] # noisy
    #     for chan in LFP_bad:
    #         not_LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0]) 
    # if day == '160128':
    #     #weird LFP responses that get bigger and longer
    #     LFP_bad = [31,47,29,27,45,43] # increase >200% and change shape
    #     for chan in LFP_bad:  
    #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0]) 
    # if day == '160202':
    #     LFP_bad = [14,59,61] # increase >200%
    #     for chan in LFP_bad:  
    #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                     
    # # if day == '160218':
    #     # pass
    # if day == '160308':
    #     # all responses get unreasonably big after pairing - electrode drift during baseline. I set the relative values to the last baseline recording block - this also biases the analysis against my own hypothesis.
    #     LFP_bad = [37,35] # noisy
    #     for chan in LFP_bad:
    #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
    #     # PSTH_resp_channels = np.asarray([])
    # if day == '160322':
    #     LFP_bad = [48,54,56,60] # noisy
    #     for chan in LFP_bad:  
    #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
    # if day == '160331':
    #     LFP_bad = [54,56,58,60] # noisy
    #     for chan in LFP_bad:  
    #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
    # if day == '160427':
    #     # LFP_bad = [17,51,53,55,7] # increase >200%
    #     # for chan in LFP_bad:  
    #     #     LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
    # if day == '221208':
    #     LFP_bad = [18] # increases >200% during baseline
    #     for chan in LFP_bad:  
    #         LFP_resp_channels_cutoff = np.delete(LFP_resp_channels_cutoff, np.where(LFP_resp_channels_cutoff == chan)[0])                 
    # # if day == '221212': # LFP responses strange... increase in length in depth all around
    #         #     pass
    # # if day == '221213':
    # # if day == '221216':
    # #     pass
    # # if day == '221219_1':


#%% SW params change

delta_power_rel_change_ALLCHANS, Freq_change_ALLCHANS, Peak_dur_change_mean_ALLCHANS, Fslope_change_mean_ALLCHANS, Sslope_change_mean_ALLCHANS, Famp_change_mean_ALLCHANS, Samp_change_mean_ALLCHANS = (np.zeros([numb*64]) for i in range(7))
Peak_dur_change_median_ALLCHANS, Fslope_change_median_ALLCHANS, Sslope_change_median_ALLCHANS, Famp_change_median_ALLCHANS, Samp_change_median_ALLCHANS = (np.zeros([numb*64]) for i in range(5))
Peak_dur_overall_change_ALLCHANS, Fslope_overall_change_ALLCHANS, Sslope_overall_change_ALLCHANS, Famp_overall_change_ALLCHANS, Samp_overall_change_ALLCHANS = (np.zeros([numb*64]) for i in range(5))

os.chdir(os.path.join(overall_path, r'UP_pairing'))
chans_to_plot = np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool)

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
numb = len(days)

for day_ind, day in enumerate(days):
    os.chdir(day)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    delta_power_rel_change_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('delta_power_rel_change.csv', delimiter = ',')*100
    Freq_change_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Freq_change.csv', delimiter = ',')*100
    
    Peak_dur_overall_change_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Peak_dur_overall_change.csv', delimiter = ',')*100
    Fslope_overall_change_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Fslope_overall_change.csv', delimiter = ',')*100
    Sslope_overall_change_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Sslope_overall_change.csv', delimiter = ',')*100
    Famp_overall_change_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Famp_overall_change.csv', delimiter = ',')*100
    Samp_overall_change_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Samp_overall_change.csv', delimiter = ',')*100
    
    Peak_dur_change_mean_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Peak_dur_change_mean.csv', delimiter = ',')*100
    Fslope_change_mean_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Fslope_change_mean.csv', delimiter = ',')*100
    Sslope_change_mean_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Sslope_change_mean.csv', delimiter = ',')*100
    Famp_change_mean_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Famp_change_mean.csv', delimiter = ',')*100
    Samp_change_mean_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Samp_change_mean.csv', delimiter = ',')*100
    
    Peak_dur_change_median_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Peak_dur_change_median.csv', delimiter = ',')*100
    Fslope_change_median_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Fslope_change_median.csv', delimiter = ',')*100
    Sslope_change_median_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Sslope_change_median.csv', delimiter = ',')*100
    Famp_change_median_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Famp_change_median.csv', delimiter = ',')*100
    Samp_change_median_ALLCHANS[day_ind*64:(day_ind + 1)*64] = np.loadtxt('Samp_change_median.csv', delimiter = ',')*100

    os.chdir('..')
    os.chdir('..')

# #average change in slow wave characteristics in ALL CHANNELS. 
# delta_power_rel_change_ALLCHANS = np.loadtxt('delta_power_rel_change_ALLCHANS.csv', delimiter = ',')*100
# Freq_change_ALLCHANS = np.loadtxt('Freq_change_ALLCHANS.csv', delimiter = ',')*100

# Peak_dur_overall_change_ALLCHANS = np.loadtxt('Peak_dur_overall_change_ALLCHANS.csv', delimiter = ',')*100
# Fslope_overall_change_ALLCHANS = np.loadtxt('Fslope_overall_change_ALLCHANS.csv', delimiter = ',')*100
# Sslope_overall_change_ALLCHANS = np.loadtxt('Sslope_overall_change_ALLCHANS.csv', delimiter = ',')*100
# Famp_overall_change_ALLCHANS = np.loadtxt('Famp_overall_change_ALLCHANS.csv', delimiter = ',')*100
# Samp_overall_change_ALLCHANS = np.loadtxt('Samp_overall_change_ALLCHANS.csv', delimiter = ',')*100

# Peak_dur_change_mean_ALLCHANS = np.loadtxt('Peak_dur_change_mean_ALLCHANS.csv', delimiter = ',')*100
# Fslope_change_mean_ALLCHANS = np.loadtxt('Fslope_change_mean_ALLCHANS.csv', delimiter = ',')*100
# Sslope_change_mean_ALLCHANS = np.loadtxt('Sslope_change_mean_ALLCHANS.csv', delimiter = ',')*100
# Famp_change_mean_ALLCHANS = np.loadtxt('Famp_change_mean_ALLCHANS.csv', delimiter = ',')*100
# Samp_change_mean_ALLCHANS = np.loadtxt('Samp_change_mean_ALLCHANS.csv', delimiter = ',')*100

# Peak_dur_change_median_ALLCHANS = np.loadtxt('Peak_dur_change_median_ALLCHANS.csv', delimiter = ',')*100
# Fslope_change_median_ALLCHANS = np.loadtxt('Fslope_change_median_ALLCHANS.csv', delimiter = ',')*100
# Sslope_change_median_ALLCHANS = np.loadtxt('Sslope_change_median_ALLCHANS.csv', delimiter = ',')*100
# Famp_change_median_ALLCHANS = np.loadtxt('Famp_change_median_ALLCHANS.csv', delimiter = ',')*100
# Samp_change_median_ALLCHANS = np.loadtxt('Samp_change_median_ALLCHANS.csv', delimiter = ',')*100


# ---------------------------------------- CHANGE ACROSS CHANNELS
to_plot_chans_overall = [Freq_change_ALLCHANS[chans_to_plot], Peak_dur_overall_change_ALLCHANS[chans_to_plot], Fslope_overall_change_ALLCHANS[chans_to_plot], Sslope_overall_change_ALLCHANS[chans_to_plot], Famp_overall_change_ALLCHANS[chans_to_plot], Samp_overall_change_ALLCHANS[chans_to_plot]]
# tkae out NaN values
to_plot_chans_overall = [i[~np.isnan(i)] for i in to_plot_chans_overall]
fig, ax = plt.subplots()
fig.suptitle('CHANGE IN AVERAGE SLOW WAVE WAVEFORM')
ax.boxplot(to_plot_chans_overall, showfliers = True, notch = True)
ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'], size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([-100,-75,-50,-25,0,25,50,75])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 14)
ax.set_ylabel('change (% baseline)', size = 16)
ax.set_ylim([-100,100])
plt.tight_layout()
# plt.savefig('SW params across channels OVERALL.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('SW params across channels OVERALL.jpg', dpi = 1000, format = 'jpg')

# mean values per sweep
to_plot_chans_mean = [Freq_change_ALLCHANS[chans_to_plot], Peak_dur_change_mean_ALLCHANS[chans_to_plot], Fslope_change_mean_ALLCHANS[chans_to_plot], Sslope_change_mean_ALLCHANS[chans_to_plot], Famp_change_mean_ALLCHANS[chans_to_plot], Samp_change_mean_ALLCHANS[chans_to_plot]]
# tkae out NaN values
to_plot_chans_mean = [i[~np.isnan(i)] for i in to_plot_chans_mean]
fig, ax = plt.subplots()
fig.suptitle('MEAN CHANGE PER SWEEP')
ax.boxplot(to_plot_chans_mean, showfliers = True, notch = True)
ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'], size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([-100,-75,-50,-25,0,25,50,75])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 14)
ax.set_ylabel('change (% baseline)', size = 16)
ax.set_ylim([-100,100])
plt.tight_layout()
# plt.savefig('SW params across channels MEAN per sweep.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('SW params across channels MEAN per sweep.jpg', dpi = 1000, format = 'jpg')

# median values per sweep
to_plot_chans_median = [Freq_change_ALLCHANS[chans_to_plot], Peak_dur_change_median_ALLCHANS[chans_to_plot], Fslope_change_median_ALLCHANS[chans_to_plot], Sslope_change_median_ALLCHANS[chans_to_plot], Famp_change_median_ALLCHANS[chans_to_plot], Samp_change_median_ALLCHANS[chans_to_plot]]
# tkae out NaN values
to_plot_chans_median = [i[~np.isnan(i)] for i in to_plot_chans_median]
fig, ax = plt.subplots()
fig.suptitle('MEDIAN CHANGE PER SWEEP')
ax.boxplot(to_plot_chans_median, showfliers = True, notch = True)
ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'], size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([-100,100])
ax.tick_params(axis="y", labelsize=14)   
# ax.set_yticks([-100,-75,-50,-25,0,25,50,75])
# ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 14)
ax.set_ylabel('change (% baseline)', size = 16)
# ax.set_ylim([-100,100])
plt.tight_layout()
# plt.savefig('SW params across channels MEDIAN per sweep.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('SW params across channels MEDIAN per sweep.jpg', dpi = 1000, format = 'jpg')

# significant across channels?
for_stats = to_plot_chans_median[1]
scipy.stats.shapiro(for_stats)
scipy.stats.anderson(for_stats)
scipy.stats.ttest_1samp(for_stats,0)

print('across channels, non parametric test, Bonferroni corrected')
for i in range(6): 
    curr_param = to_plot_chans_median[i]
    outliers = (np.logical_or(curr_param > (np.percentile(curr_param, 75) + 1.5*(np.abs(np.percentile(curr_param, 75) - np.percentile(curr_param, 25)))), curr_param < (np.percentile(curr_param, 25) - 1.5*(np.abs(np.percentile(curr_param, 75) - np.percentile(curr_param, 25))))))
    curr_param = curr_param[~outliers]
    print(scipy.stats.shapiro(curr_param))
    # t, p = scipy.stats.ttest_1samp(curr_param, 0)
    t, p = scipy.stats.mannwhitneyu(curr_param, np.zeros(len(curr_param)))
    print(t, p*6, cohend(curr_param, np.zeros([len(curr_param)])), len(curr_param))


# Fslope_change_median_ALLCHANS_1 = copy.deepcopy(Fslope_change_median_ALLCHANS)
# Fslope_change_median_ALLCHANS_1[~chans_to_plot] = np.NaN
# a = np.reshape(Fslope_change_median_ALLCHANS_1, (-1,64)) 

# fig, ax = plt.subplots()
# ax.bar(np.histogram(for_stats, bins = 200)[1][:-1], np.histogram(for_stats, bins = 200)[0][:])

# ------------------------------------------- CHANGE ACROSS MICE
to_plot_mice = [Freq_change_ALLCHANS, Peak_dur_overall_change_ALLCHANS, Fslope_overall_change_ALLCHANS, Sslope_overall_change_ALLCHANS, Famp_overall_change_ALLCHANS, Samp_overall_change_ALLCHANS]
# to_plot_mice = [Freq_change_ALLCHANS, Peak_dur_change_median_ALLCHANS, Fslope_change_median_ALLCHANS, Sslope_change_median_ALLCHANS, Famp_change_median_ALLCHANS, Samp_change_median_ALLCHANS]
# replace non LFP channels with NaN and average across channels
for ind, param in enumerate(to_plot_mice):
    param[~chans_to_plot] = np.NaN
to_plot_mice = [np.nanmedian(np.reshape(i, (-1,64)), axis = 1) for i in to_plot_mice]        
# exclude outliers:
for param_ind, param in enumerate(to_plot_mice):
    outliers = (param > (np.percentile(param, 75) + 1.5*(np.abs(np.percentile(param, 75) - np.percentile(param, 25)))))
    to_plot_mice[param_ind] = to_plot_mice[param_ind][~outliers]
fig, ax = plt.subplots()
ax.boxplot(to_plot_mice, showfliers = True, notch = False)
ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'], size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([-50,-25,0,25])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 14)
ax.set_ylabel('change (% baseline)', size = 16)
plt.savefig('SW params across mice.pdf', dpi = 1000, format = 'pdf')
plt.savefig('SW params across mice.jpg', dpi = 1000, format = 'jpg')

print('across mice, parametric test, Bonferroni corrected')
for i in range(6):
    t, p = scipy.stats.ttest_1samp(to_plot_mice[i], 0)
    print(t, p*6) #bonferroni correction



# correlation matrix across channels before vs after
take_out_outliers = True
take_out_bottom_outliers = True
take_out_top_outliers = False
to_plot_chans = [delta_power_rel_change_ALLCHANS[chans_to_plot], Fslope_overall_change_ALLCHANS[chans_to_plot], Sslope_overall_change_ALLCHANS[chans_to_plot], Famp_overall_change_ALLCHANS[chans_to_plot], Samp_overall_change_ALLCHANS[chans_to_plot], Freq_change_ALLCHANS[chans_to_plot], Peak_dur_overall_change_ALLCHANS[chans_to_plot]]
# take out nans in every array
nans_mask = np.any(np.vstack([np.isnan(i) for i in to_plot_chans]) == True, axis = 0)
to_plot_chans = [i[~nans_mask] for i in to_plot_chans]
#take out outliers
if take_out_outliers == True and take_out_bottom_outliers == True:
    outliers_mask = np.any(np.vstack([np.logical_or((i > (np.percentile(i, 75) + 1.5*(np.abs(np.percentile(i, 75) - np.percentile(i, 25))))), (i < (np.percentile(i, 75) - 1.5*(np.abs(np.percentile(i, 75) - np.percentile(i, 25)))))) for i in to_plot_chans]) == True, axis = 0)
    to_plot_chans = [i[~outliers_mask] for i in to_plot_chans]
    predictors = to_plot_chans[1:]
    dependent = to_plot_chans[0]
elif take_out_outliers == True and take_out_bottom_outliers == False:
    outliers_mask = np.any(np.vstack([(i > (np.percentile(i, 75) + 1.5*(np.abs(np.percentile(i, 75) - np.percentile(i, 25))))) for i in to_plot_chans]) == True, axis = 0)
    to_plot_chans = [i[~outliers_mask] for i in to_plot_chans]
    predictors = to_plot_chans[1:]
    dependent = to_plot_chans[0]
else:
    predictors = to_plot_chans[1:]
    dependent = to_plot_chans[0]

corr_matrix = np.zeros([7,7])
for param_ind, param_for_corr in enumerate(to_plot_chans):
    for param2_ind, param2_for_corr in enumerate(to_plot_chans):
        if np.isnan(param_for_corr).any() == False and np.isnan(param2_for_corr).any() == False:
            slope, intercept, r, p, std_err = stats.linregress(param_for_corr, param2_for_corr)
        else:
            slope, intercept, r, p, std_err = stats.linregress(param_for_corr[~np.isnan(param_for_corr) & ~np.isnan(param2_for_corr)], param2_for_corr[~np.isnan(param_for_corr) & ~np.isnan(param2_for_corr)])
        corr_matrix[param_ind, param2_ind] = r
        
fig, ax = plt.subplots()
cmap = cm.seismic
cmap.set_bad(color = 'white')
to_plot = copy.deepcopy(corr_matrix)
to_plot[np.triu_indices(to_plot.shape[0], k = 1)] = np.NaN
im = ax.imshow(to_plot, cmap = 'Reds')
ax.set_xticks([0,1,2,3,4,5,6])
ax.set_xticklabels(['Dpower', 'Fslp', 'Sslp', 'Famp', 'Samp', 'Freq',  'Dur'], size = 14, rotation = 45)
ax.set_yticks([0,1,2,3,4,5,6])
ax.set_yticklabels(['Dpower', 'Fslp', 'Sslp', 'Famp', 'Samp', 'Freq',  'Dur'], size = 14)
plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.75])
fig.colorbar(im, cax=cbar_ax)
# plt.savefig('SW params corr.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('SW params corr.jpg', dpi = 1000, format = 'jpg')

  
LR = LinearRegression()   
LR.fit(np.transpose(np.vstack(predictors)), dependent)
prediction = LR.predict(np.transpose(np.vstack(predictors)))

# predicting the accuracy score of the total model -- check the decomposed R2 values add up to this value 
score = r2_score(dependent,prediction)
print(score)

# shapley-Owen R2 decomposition giving the relative importance of every predictor to delta power change
def ShapleyOwen(Dep, Indep):
    '''
    Parameters
    ----------
    Dep : Array 
        dependent value (X).
    Indep : List of Arrays
        Predictors.

    Returns
    -------
    Explained variances of each predictor.
    '''
    # get all combinations of indices
    k = len(Indep)
    a = list(range(k))
    ind_comb = []
    for L in range(1,k+1):  
        if L == 1:
            ind_comb = list(itertools.combinations(a, L))
        else:
            ind_comb = ind_comb + list(itertools.combinations(a, L))
    # convert to list of lists (not needed actually)
    ind_comb = [list(i) for i in ind_comb]
    
    var_explained = []
    
    for predict_ind, predictor in enumerate(Indep):
        # print(predict_ind)
        R2_weighted_list = []
        # all models with this predictor in
        models = [i for i in ind_comb if predict_ind in i]
        for model_ind, model in enumerate(models):
            # print(model)
            weight = math.factorial(len(model) -1) * math.factorial(k - len(model))/math.factorial(k)
            # model with predictor
            independent = np.transpose(np.vstack(Indep)[model])
            LR = LinearRegression()  
            LR.fit(independent, Dep)
            prediction = LR.predict(independent)
            score_with = r2_score(dependent,prediction)            
            # model without
            if len(model) == 1: # only one predictor (itself)
                score_without = 0
            else:
                model_without = np.asarray(model)[np.asarray(model) != predict_ind]
                independent = np.transpose(np.vstack(Indep)[model_without])
                LR = LinearRegression()  
                LR.fit(independent, Dep)
                prediction = LR.predict(independent)
                score_without = r2_score(dependent,prediction) 
            # print(f'{weight} {score_with} {score_without}')
            R2_weighted_list.append(weight * (score_with - score_without))
            
            if model_ind == len(models) - 1:
                var_explained.append(sum(R2_weighted_list))
    return var_explained

# ShapleyOwen to explain delta change
var_explained = ShapleyOwen(dependent, predictors)
print(var_explained)
params = ['Fslp', 'Sslp', 'Famp', 'Samp', 'Freq',  'Dur']
fig, ax = plt.subplots()
ax.barh(np.arange(len(var_explained)) + 1, [i*100 for i in var_explained], color = 'k')
ax.set_yticks(np.arange(len(var_explained)) + 1)
ax.set_yticklabels(params, size = 18)
ax.invert_yaxis()
# ax.set_ylabel('Slow oscillation parameter', size = 18)
ax.set_xticks([0,10,20,30])
ax.set_xticklabels(list(map(str, ax.get_xticks())), size = 18)
ax.set_xlabel('delta power change explained (%)', size = 18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# plt.savefig('delta explained variance.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('delta explained variance.jpg', dpi = 1000, format = 'jpg')


#%% change in amount of wave peaks

os.chdir(os.path.join(overall_path, r'UP_pairing'))
chans_to_plot = np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool).reshape(12,64)
SW_peaks_down_ALLCHANS = []
SW_peaks_up_ALLCHANS = []
Mulitpeak_waves_prct_down_ALLCHANS = []
Mulitpeak_waves_prct_up_ALLCHANS = []

for day_ind, day in enumerate(days):
    os.chdir(day)
    print(day)   
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    SW_peaks_down = np.load('SW_peaks_sweeps_avg_down.npy')[:,chans_to_plot[day_ind,:]]
    SW_peaks_down_ALLCHANS.append(SW_peaks_down)
    SW_peaks_up = np.load('SW_peaks_sweeps_avg_up.npy')[:,chans_to_plot[day_ind,:]]
    SW_peaks_up_ALLCHANS.append(SW_peaks_up)
    All_SW_peaks_down = [[i[chan] for chan in range(64) if chans_to_plot[day_ind,chan] == True] for i in pickle.load(open('SW_peaks_sweeps_down', 'rb'))]
    All_SW_peaks_up = [[i[chan] for chan in range(64) if chans_to_plot[day_ind,chan] == True] for i in pickle.load(open('SW_peaks_sweeps_up', 'rb'))]
    Mulitpeak_waves_prct_down_ALLCHANS.append(np.asarray([[np.sum(np.asarray(j) > 1)/len(j) for j in i] for i in All_SW_peaks_down]))
    Mulitpeak_waves_prct_up_ALLCHANS.append(np.asarray([[np.sum(np.asarray(j) > 1)/len(j) for j in i] for i in All_SW_peaks_up]))
    os.chdir('..')
    os.chdir('..')

#average for each channel before and after pairing
SW_peaks_down_before_ALLCHANS = [np.mean(i[[0,1,2,3],:], axis = 0) for i in SW_peaks_down_ALLCHANS] # average number of peaks per channel
SW_peaks_down_after_ALLCHANS = [np.mean(i[[4,5,6,7,8,9],:], axis = 0) for i in SW_peaks_down_ALLCHANS]
SW_peaks_up_before_ALLCHANS = [np.mean(i[[0,1,2,3],:], axis = 0) for i in SW_peaks_up_ALLCHANS]
SW_peaks_up_after_ALLCHANS = [np.mean(i[[4,5,6,7,8,9],:], axis = 0) for i in SW_peaks_up_ALLCHANS]
Mulitpeak_waves_prct_before_down_ALLCHANS = [np.mean(i[[0,1,2,3],:], axis = 0) for i in Mulitpeak_waves_prct_down_ALLCHANS] # percentage of waves that are multipeak
Mulitpeak_waves_prct_after_down_ALLCHANS = [np.mean(i[[4,5,6,7,8,9],:], axis = 0) for i in Mulitpeak_waves_prct_down_ALLCHANS]
Mulitpeak_waves_prct_before_up_ALLCHANS = [np.mean(i[[0,1,2,3],:], axis = 0) for i in Mulitpeak_waves_prct_up_ALLCHANS]
Mulitpeak_waves_prct_after_up_ALLCHANS = [np.mean(i[[4,5,6,7,8,9],:], axis = 0) for i in Mulitpeak_waves_prct_up_ALLCHANS]

# boxplots across channels
# average number of peaks per wave
# fig, ax = plt.subplots()
# ax.plot([np.repeat(1,len(np.concatenate(SW_peaks_down_before_ALLCHANS))), np.repeat(2,len(np.concatenate(SW_peaks_down_before_ALLCHANS)))], [np.concatenate(SW_peaks_down_before_ALLCHANS), np.concatenate(SW_peaks_down_after_ALLCHANS)], color = 'k', linewidth = 0.25)
fig, ax = plt.subplots(figsize = (2,4))
ax.boxplot([np.concatenate(SW_peaks_down_before_ALLCHANS), np.concatenate(SW_peaks_down_after_ALLCHANS)], widths = 0.25)
scipy.stats.shapiro(np.concatenate(SW_peaks_down_before_ALLCHANS))
ax.set_xticks([1,2])
ax.set_xticklabels(['before', 'after'], size = 16)
t, p = scipy.stats.mannwhitneyu(np.concatenate(SW_peaks_down_before_ALLCHANS), np.concatenate(SW_peaks_down_after_ALLCHANS))
print(t, p)
# t, p = scipy.stats.ttest_rel(np.concatenate(SW_peaks_down_before_ALLCHANS), np.concatenate(SW_peaks_down_after_ALLCHANS))
# print(t, p)

# percentage of waves that are multipeaks
# fig, ax = plt.subplots()
# ax.plot([np.repeat(1,len(np.concatenate(Mulitpeak_waves_prct_before_down_ALLCHANS))), np.repeat(2,len(np.concatenate(Mulitpeak_waves_prct_before_down_ALLCHANS)))], [np.concatenate(Mulitpeak_waves_prct_before_down_ALLCHANS), np.concatenate(Mulitpeak_waves_prct_after_down_ALLCHANS)], color = 'k', linewidth = 0.25)
fig, ax = plt.subplots(figsize = (2,4))
ax.boxplot([np.concatenate(Mulitpeak_waves_prct_before_down_ALLCHANS)*100, np.concatenate(Mulitpeak_waves_prct_after_down_ALLCHANS)*100], widths = 0.35)
ax.set_xticks([1,2])
ax.set_xticklabels(['before', 'after'], size = 14)
ax.tick_params(axis="y", labelsize=14)  
ax.set_ylim([15,85])  
plt.tight_layout()
# plt.savefig('Multipeak waves prct all chans.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('Multipeak waves prct all chans.pdf', dpi = 1000, format = 'pdf')
# scipy.stats.shapiro(np.concatenate(Mulitpeak_waves_prct_before_down_ALLCHANS))
t, p = scipy.stats.mannwhitneyu(np.concatenate(Mulitpeak_waves_prct_before_down_ALLCHANS), np.concatenate(Mulitpeak_waves_prct_after_down_ALLCHANS))
print(t, p)
# t, p = scipy.stats.ttest_rel(np.concatenate(Mulitpeak_waves_prct_before_down_ALLCHANS), np.concatenate(Mulitpeak_waves_prct_before_down_ALLCHANS))
# print(t, p)


# boxplot average in mice, percentage of multipeak waves DOWN
# fig, ax = plt.subplots()
# ax.plot([np.repeat(1, 12), np.repeat(2, 12)], [[np.mean(i) for i in Mulitpeak_waves_prct_before_down_ALLCHANS], [np.mean(i) for i in Mulitpeak_waves_prct_after_down_ALLCHANS]])
fig, ax = plt.subplots(figsize = (2,4))
ax.boxplot([[np.mean(i*100) for i in Mulitpeak_waves_prct_before_down_ALLCHANS], [np.mean(i*100) for i in Mulitpeak_waves_prct_after_down_ALLCHANS]], widths = 0.35)
ax.set_xticks([1,2])
ax.set_xticklabels(['before', 'after'], size = 14)
ax.tick_params(axis="y", labelsize=14)  
ax.set_ylim([20,65])  
plt.tight_layout()
# plt.savefig('Multipeak waves prct all mice.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('Multipeak waves prct all mice.pdf', dpi = 1000, format = 'pdf')
t, p = scipy.stats.ttest_rel([np.mean(i) for i in Mulitpeak_waves_prct_before_down_ALLCHANS], [np.mean(i) for i in Mulitpeak_waves_prct_after_down_ALLCHANS])
print(t, p)

# boxplot average in mice, percentage of multipeak waves UP
fig, ax = plt.subplots()
ax.plot([np.repeat(1, 12), np.repeat(2, 12)], [[np.mean(i) for i in Mulitpeak_waves_prct_before_up_ALLCHANS], [np.mean(i) for i in Mulitpeak_waves_prct_after_up_ALLCHANS]])
fig, ax = plt.subplots()
ax.boxplot([[np.mean(i) for i in Mulitpeak_waves_prct_before_up_ALLCHANS], [np.mean(i) for i in Mulitpeak_waves_prct_after_up_ALLCHANS]])
t, p = scipy.stats.ttest_rel([np.mean(i) for i in Mulitpeak_waves_prct_before_up_ALLCHANS], [np.mean(i) for i in Mulitpeak_waves_prct_after_up_ALLCHANS])
print(t, p)


# boxplot of average in mice, average number of DOWN peak
fig, ax = plt.subplots()
ax.plot([np.repeat(1, 12), np.repeat(2, 12)], [[np.mean(i) for i in SW_peaks_down_before_ALLCHANS], [np.mean(i) for i in SW_peaks_down_after_ALLCHANS]])
fig, ax = plt.subplots()
ax.boxplot([[np.mean(i) for i in SW_peaks_down_before_ALLCHANS], [np.mean(i) for i in SW_peaks_down_after_ALLCHANS]])
t, p = scipy.stats.ttest_rel([np.mean(i) for i in SW_peaks_down_before_ALLCHANS], [np.mean(i) for i in SW_peaks_down_after_ALLCHANS])
print(t, p)

# boxplot of average in mice, average number UP peak
fig, ax = plt.subplots()
ax.plot([np.repeat(1, 12), np.repeat(2, 12)], [[np.mean(i) for i in SW_peaks_up_before_ALLCHANS], [np.mean(i) for i in SW_peaks_up_after_ALLCHANS]])
fig, ax = plt.subplots()
ax.boxplot([[np.mean(i) for i in SW_peaks_down_before_ALLCHANS], [np.mean(i) for i in SW_peaks_down_after_ALLCHANS]])
np.asarray([[np.mean(i) for i in SW_peaks_down_before_ALLCHANS], [np.mean(i) for i in SW_peaks_down_after_ALLCHANS]]).T
t, p = scipy.stats.ttest_rel([np.mean(i) for i in SW_peaks_up_before_ALLCHANS], [np.mean(i) for i in SW_peaks_up_after_ALLCHANS])
print(t, p)


#%% SW spiking peak and magn, spontaneous spike rate before and after.
to_plot_1_SW = [0,1,2,3]
to_plot_2_SW = [4,5,6,7,8,9]

intersect = False

plot_mice_UP = True
plot_mice_DOWN = True

# ------------------------------------------------------------------ UP ---------------------------------------------------------------
os.chdir(os.path.join(overall_path, r'UP_pairing'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
mice = len(days)

channels_LFP = np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool)
channels_SW_spiking = np.loadtxt('SW_spiking_channels_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool)
if intersect == True:
    channels = np.logical_and(channels_LFP, channels_SW_spiking)
else:
    channels = channels_SW_spiking
    
# average of overall spiking averaged across channels
SW_spiking_area_change_overall_UP_ANOVA = np.zeros([mice,10])
SW_spiking_area_change_overall_UP_ANOVA[:] = np.NaN
SW_spiking_area_change_overall_UP = np.zeros([mice])
SW_spiking_area_change_overall_UP[:] = np.NaN
SW_spiking_peak_change_overall_UP = np.zeros([mice])
SW_spiking_peak_change_overall_UP[:] = np.NaN

spont_activity_change_UP = np.zeros([mice])
spont_activity_change_UP[:] = np.NaN
spont_activity_UP_ANOVA = np.zeros([mice,10])
spont_activity_UP_ANOVA[:] = np.NaN
spont_activity_UP_rel_ANOVA = np.zeros([mice,10])
spont_activity_UP_rel_ANOVA[:] = np.NaN

for day_ind, day in enumerate(days):
    os.chdir(day)
    print(day)   
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    curr_SW_spiking = np.load('SW_spiking_sweeps_avg.npy')
    curr_SW_spiking_avg = np.mean(curr_SW_spiking[:,np.reshape(channels, (mice,64))[day_ind,:],:], axis = 1) #average spiking waveform across channels
    SW_spiking_area_change_overall_UP[day_ind] = (np.sum(np.mean(curr_SW_spiking_avg[to_plot_2_SW,:], axis = 0)) - np.sum(np.mean(curr_SW_spiking_avg[to_plot_1_SW,:], axis = 0)))/np.sum(np.mean(curr_SW_spiking_avg[to_plot_1_SW,:], axis = 0))
    SW_spiking_peak_change_overall_UP[day_ind] = (np.max(smooth(np.mean(curr_SW_spiking_avg[to_plot_2_SW,:], axis = 0), 25)) - np.max(smooth(np.mean(curr_SW_spiking_avg[to_plot_1_SW,:], axis = 0), 10)))/np.max(smooth(np.mean(curr_SW_spiking_avg[to_plot_1_SW,:], axis = 0), 10))
    SW_spiking_area_change_overall_UP_ANOVA[day_ind,:] = np.sum(curr_SW_spiking_avg, axis = 1)/np.sum(np.nanmean(curr_SW_spiking_avg[to_plot_1_SW,:], axis = 0))
    
    if plot_mice_UP:
        fig, ax = plt.subplots()
        fig.suptitle(f'{day} UP')
        ax.plot(smooth(np.nanmean(curr_SW_spiking_avg[to_plot_1_SW,:], axis = 0),25), 'b')
        ax.plot(smooth(np.nanmean(curr_SW_spiking_avg[to_plot_2_SW,:], axis = 0),25), 'r')
    
    #spontaneous activity
    curr_spont_activity = np.load('spont_spiking.npy')
    #average across channels
    spont_activity_UP_ANOVA[day_ind,:] = np.mean(curr_spont_activity[:,np.reshape(channels, (mice,64))[day_ind,:]], axis = 1)
    #baseline averaged
    spont_activity_UP_rel_ANOVA[day_ind,:] = spont_activity_UP_ANOVA[day_ind,:]/np.mean(spont_activity_UP_ANOVA[day_ind, [0,1,2,3]])
    spont_activity_change_UP[day_ind] = np.mean(spont_activity_UP_rel_ANOVA[day_ind,[4,5,6,7,8,9]]) - 1
    os.chdir('..')
    os.chdir('..')

np.savetxt('SW_spiking_area_change_overall_UP_ANOVA.csv', SW_spiking_area_change_overall_UP_ANOVA, delimiter = ',')
np.savetxt('spont_activity_UP_ANOVA.csv', spont_activity_UP_ANOVA, delimiter = ',')
np.savetxt('spont_activity_UP_rel_ANOVA.csv', spont_activity_UP_rel_ANOVA, delimiter = ',')



# ------------------------------------------------------------------ DOWN ---------------------------------------------------------------
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
mice= len(days)
channels_LFP = np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool)
channels_SW_spiking = np.loadtxt('SW_spiking_channels_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool)
if intersect == True:
    channels = np.logical_and(channels_LFP, channels_SW_spiking)
else:
    channels = channels_SW_spiking

# average of overall spiking averaged across channels
SW_spiking_area_change_overall_DOWN_ANOVA = np.zeros([mice, 10])
SW_spiking_area_change_overall_DOWN_ANOVA[:] = np.NaN
SW_spiking_area_change_overall_DOWN = np.zeros([mice])
SW_spiking_area_change_overall_DOWN[:] = np.NaN
SW_spiking_peak_change_overall_DOWN = np.zeros([mice])
SW_spiking_peak_change_overall_DOWN[:] = np.NaN

spont_activity_change_DOWN = np.zeros([mice])
spont_activity_change_DOWN[:] = np.NaN
spont_activity_DOWN_ANOVA = np.zeros([mice,10])
spont_activity_DOWN_ANOVA[:] = np.NaN
spont_activity_DOWN_rel_ANOVA = np.zeros([mice,10])
spont_activity_DOWN_rel_ANOVA[:] = np.NaN

for day_ind, day in enumerate(days):
    os.chdir(day)
    print(day)   
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    curr_SW_spiking = np.load('SW_spiking_sweeps_avg.npy')
    curr_SW_spiking_avg = np.mean(curr_SW_spiking[:,np.reshape(channels, (mice,64))[day_ind,:],:], axis = 1) #average spiking waveform across channels
    SW_spiking_area_change_overall_DOWN[day_ind] = (np.sum(np.nanmean(curr_SW_spiking_avg[to_plot_2_SW,:], axis = 0)) - np.sum(np.nanmean(curr_SW_spiking_avg[to_plot_1_SW,:], axis = 0)))/np.sum(np.nanmean(curr_SW_spiking_avg[to_plot_1_SW,:], axis = 0))
    SW_spiking_peak_change_overall_DOWN[day_ind] = (np.max(smooth(np.nanmean(curr_SW_spiking_avg[to_plot_2_SW,:], axis = 0), 25)) - np.max(smooth(np.nanmean(curr_SW_spiking_avg[to_plot_1_SW,:], axis = 0), 10)))/np.max(smooth(np.nanmean(curr_SW_spiking_avg[to_plot_1_SW,:], axis = 0), 10))
    SW_spiking_area_change_overall_DOWN_ANOVA[day_ind,:] = np.sum(curr_SW_spiking_avg, axis = 1)/np.sum(np.nanmean(curr_SW_spiking_avg[to_plot_1_SW,:], axis = 0))

    if plot_mice_DOWN:
        fig, ax = plt.subplots()
        fig.suptitle(f'{day} DOWN')
        ax.plot(smooth(np.nanmean(curr_SW_spiking_avg[to_plot_1_SW,:], axis = 0),25), 'b')
        ax.plot(smooth(np.nanmean(curr_SW_spiking_avg[to_plot_2_SW,:], axis = 0),25), 'r')

    #spontaneous activity
    curr_spont_activity = np.load('spont_spiking.npy')
    #average across channels
    spont_activity_DOWN_ANOVA[day_ind,:] = np.mean(curr_spont_activity[:,np.reshape(channels, (mice,64))[day_ind,:]], axis = 1)
    #baseline averaged
    spont_activity_DOWN_rel_ANOVA[day_ind,:] = spont_activity_DOWN_ANOVA[day_ind,:]/np.mean(spont_activity_DOWN_ANOVA[day_ind, [0,1,2,3]])
    spont_activity_change_DOWN[day_ind] = np.mean(spont_activity_DOWN_rel_ANOVA[day_ind,[4,5,6,7,8,9]]) - 1

    os.chdir('..')
    os.chdir('..')

np.savetxt('SW_spiking_area_change_overall_DOWN_ANOVA.csv', SW_spiking_area_change_overall_DOWN_ANOVA, delimiter = ',')
np.savetxt('spont_activity_DOWN_ANOVA.csv', spont_activity_DOWN_ANOVA, delimiter = ',')
np.savetxt('spont_activity_DOWN_rel_ANOVA.csv', spont_activity_DOWN_rel_ANOVA, delimiter = ',')

print(f'spontaneous spiking: {1 + np.mean(spont_activity_change_UP), np.std(spont_activity_change_UP), 1 + np.mean(spont_activity_change_DOWN), np.std(spont_activity_change_DOWN)}')
print(f'spiking area: {1 + np.mean(SW_spiking_area_change_overall_UP), np.std(SW_spiking_area_change_overall_UP), 1 + np.mean(SW_spiking_area_change_overall_DOWN), np.std(SW_spiking_area_change_overall_DOWN)}')
# print(f'spiking peak: {np.mean(SW_spiking_peak_change_overall_UP), np.mean(SW_spiking_peak_change_overall_DOWN)}')

print(f'spiking area: {scipy.stats.ttest_ind(SW_spiking_area_change_overall_UP, SW_spiking_area_change_overall_DOWN, nan_policy = "omit")}')
# print(f'spiking peak: {scipy.stats.ttest_ind(SW_spiking_peak_change_overall_UP, SW_spiking_peak_change_overall_DOWN, nan_policy = "omit")}')

# np.nanmean(SW_spiking_peak_change_overall_DOWN)
# np.nanmean(SW_spiking_peak_change_overall_UP)

# np.nanmean(SW_spiking_area_change_overall_UP)*100 + 100
# np.nanmean(SW_spiking_area_change_overall_DOWN)*100 + 100
# np.nanstd(SW_spiking_area_change_overall_UP)*100
# np.nanstd(SW_spiking_area_change_overall_DOWN)*100

# np.nanmean(spont_activity_DOWN_rel_ANOVA[:,[4,5,6,7,8,9]])*100
# np.nanmean(spont_activity_UP_rel_ANOVA[:,[4,5,6,7,8,9]])*100
# np.nanstd(spont_activity_DOWN_rel_ANOVA[:,[4,5,6,7,8,9]])*100
# np.nanstd(spont_activity_UP_rel_ANOVA[:,[4,5,6,7,8,9]])*100


# PLOT
patch = 1 #(how many times SEM to plot as patch)
fig, ax = plt.subplots(figsize = (10,4))
os.chdir(os.path.join(overall_path, r'UP_pairing'))
SW_magn_UP = np.loadtxt('SW_spiking_area_change_overall_UP_ANOVA.csv', delimiter = ',')*100
ax.plot(np.nanmean(SW_magn_UP, axis = 0), color = 'r')
ax.fill_between(list(range(10)), np.nanmean(SW_magn_UP, axis = 0) + patch*np.nanstd(SW_magn_UP, axis = 0)/np.sqrt(SW_magn_UP.shape[0]), np.nanmean(SW_magn_UP, axis = 0) - patch*np.nanstd(SW_magn_UP, axis = 0)/np.sqrt(SW_magn_UP.shape[0]), alpha = 0.1, color = 'r')
os.chdir('..')
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
SW_magn_DOWN = np.loadtxt('SW_spiking_area_change_overall_DOWN_ANOVA.csv', delimiter = ',')*100
ax.plot(np.nanmean(SW_magn_DOWN, axis = 0), c = 'k')
ax.fill_between(list(range(10)), np.nanmean(SW_magn_DOWN, axis = 0) + patch*np.nanstd(SW_magn_DOWN, axis = 0)/np.sqrt(SW_magn_DOWN.shape[0]), np.nanmean(SW_magn_DOWN, axis = 0) - patch*np.nanstd(SW_magn_DOWN, axis = 0)/np.sqrt(SW_magn_DOWN.shape[0]), alpha = 0.1, color = 'k')
ax.set_ylim([30, 160])
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 16)
ax.set_ylabel('SW spiking area (% of baseline)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 16)
ax.set_yticks([50,100,150])
ax.set_yticklabels(list(map(str,[50,100,150])), size = 16)
plt.tight_layout()
os.chdir(overall_path)
plt.savefig('SW spiking area UP vs DOWN.pdf', dpi = 1000, format = 'pdf')
plt.savefig('SW spiking area UP vs DOWN.jpg', dpi = 1000, format = 'jpg')


patch = 1 #(how many times SEM to plot as patch)
fig, ax = plt.subplots(figsize = (10,4))
os.chdir(os.path.join(overall_path, r'UP_pairing'))
spont_spik_UP = np.loadtxt('spont_activity_UP_rel_ANOVA.csv', delimiter = ',')*100
ax.plot(np.nanmean(spont_spik_UP, axis = 0), color = 'r')
ax.fill_between(list(range(10)), np.nanmean(spont_spik_UP, axis = 0) + patch*np.nanstd(spont_spik_UP, axis = 0)/np.sqrt(spont_spik_UP.shape[0]), np.nanmean(spont_spik_UP, axis = 0) - patch*np.nanstd(spont_spik_UP, axis = 0)/np.sqrt(spont_spik_UP.shape[0]), alpha = 0.1, color = 'r')
os.chdir('..')
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
spont_spik_DOWN = np.loadtxt('spont_activity_DOWN_rel_ANOVA.csv', delimiter = ',')*100
ax.plot(np.nanmean(spont_spik_DOWN, axis = 0), c = 'k')
ax.fill_between(list(range(10)), np.nanmean(spont_spik_DOWN, axis = 0) + patch*np.nanstd(spont_spik_DOWN, axis = 0)/np.sqrt(spont_spik_DOWN.shape[0]), np.nanmean(spont_spik_DOWN, axis = 0) - patch*np.nanstd(spont_spik_DOWN, axis = 0)/np.sqrt(spont_spik_DOWN.shape[0]), alpha = 0.1, color = 'k')
ax.set_ylim([30, 160])
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 16)
ax.set_ylabel('spont spiking activity (% of baseline)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 16)
ax.set_yticks([50,100,150])
ax.set_yticklabels(list(map(str,[50,100,150])), size = 16)
plt.tight_layout()
os.chdir(overall_path)
plt.savefig('spontaneous spiking UP vs DOWN.pdf', dpi = 1000, format = 'pdf')
plt.savefig('spontaneous spiking UP vs DOWN.jpg', dpi = 1000, format = 'jpg')


#%% correlation between delta power change and LFP synchronization change

os.chdir(os.path.join(overall_path, r'UP_pairing'))
LFP_cross_corr_all = np.loadtxt('channel_peak_nearest_difference_mean_rel.csv', delimiter = ',')
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
numb = len(days)
delta_power_auto_outliers_rel_ALL = np.loadtxt('delta_power_auto_outliers_rel_ALL.csv', delimiter = ',')
delta_power_auto_outliers_rel_change_ALLCHANS = np.loadtxt('delta_power_auto_outliers_rel_change_ALLCHANS.csv', delimiter = ',')
delta_change_LFP_group = []

for day_ind, day in enumerate(days):
    os.chdir(day)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
     
    delta_change_LFP = np.mean(delta_power_auto_outliers_rel_change_ALLCHANS[day_ind*64 + LFP_resp_channels_cutoff])
    delta_change_LFP_group.append(delta_change_LFP)
    
    os.chdir('..')
    os.chdir('..')

print(np.mean(delta_change_LFP_group))

# take out mice without complete set of channel pairs
LFP_cross_corr_all_clean = LFP_cross_corr_all[:,:12]
delta_change_LFP_group_clean = np.asarray(delta_change_LFP_group)[~np.any(np.isnan(LFP_cross_corr_all_clean), axis=1)]
LFP_cross_corr_all_clean = LFP_cross_corr_all_clean[~np.any(np.isnan(LFP_cross_corr_all_clean), axis=1),:]

X = delta_change_LFP_group_clean*100
Y = np.mean(LFP_cross_corr_all_clean, axis = 1)*100 # average across channel distances
fig, ax = plt.subplots(figsize = (8,4))
slope, intercept, r, p, std_err = stats.linregress(X, Y)
print(f'{r**2} and {p} for {len(X)} channels')
ax.scatter(X,Y, color = 'k')
ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
ax.set_xlabel('delta power change (% baseline)', size = 16)
ax.set_ylabel('LFP synchronization change \n (% baseline)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis="x", labelsize=16)
ax.set_yticks([-20,-15,-10,-5,0])
ax.tick_params(axis="y", labelsize=16)
ax.set_ylim([-22,2])
plt.tight_layout()
plt.savefig('delta vs cross corr.pdf', dpi = 1000, format = 'pdf')
plt.savefig('delta vs cross corr.jpg', dpi = 1000, format = 'jpg')



#%% first/last pre sweep vs nostim sweep for delta power and slow wave parameters
overall_path = r'C:\One_Drive\OneDrive\Dokumente\SWS\FOR_ANALYSIS'
fftfreq = np.fft.fftfreq(3500, d = (1/1000))

median_or_avg = 'median'

delta_power_vs_nostim_ALL = [[] for i in range(2)] # two lists, one baseline one nostim
SW_waveform_vs_nostim_ALL = [[] for i in range(2)]
Peak_dur_vs_nostim_ALL = [[] for i in range(2)]
Freq_vs_nostim_ALL = [[] for i in range(2)]
Fslope_vs_nostim_ALL = [[] for i in range(2)]
Sslope_vs_nostim_ALL = [[] for i in range(2)]
Famp_vs_nostim_ALL = [[] for i in range(2)]
Samp_vs_nostim_ALL = [[] for i in range(2)] 

delta_power_baseline_diffs_ALL = [] 
SW_waveform_baseline_diffs_ALL = []
Peak_dur_baseline_diffs_ALL = []
Freq_baseline_diffs_ALL = []
Fslope_baseline_diffs_ALL = []
Sslope_baseline_diffs_ALL = []
Famp_baseline_diffs_ALL = []
Samp_baseline_diffs_ALL = []
delta_power_baseline_diffs_ALL_rel = [] 
SW_waveform_baseline_diffs_ALL_rel = []
Peak_dur_baseline_diffs_ALL_rel = []
Freq_baseline_diffs_ALL_rel = []
Fslope_baseline_diffs_ALL_rel = []
Sslope_baseline_diffs_ALL_rel = []
Famp_baseline_diffs_ALL_rel = []
Samp_baseline_diffs_ALL_rel = []


PSD_baseline_ALL = []
PSD_lastpre_ALL = []
PSD_nostim_ALL = []


for cond in ['UP_pairing', 'DOWN_pairing']:
    os.chdir(os.path.join(overall_path, cond))
    LFP_resp_channels_cutoff_ALLCHANS = np.loadtxt('LFP_resp_channels_cutoff_ALLCHANS.csv', delimiter = ',', dtype = int).astype(bool)

    days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
    numb = len(days)
    for day in days:
        os.chdir(day)
        
        if [i for i in os.listdir() if 'pairing_nowhisker' in i].__len__() == 0:
            print(f'{day} doesnt have nowhisker')
            os.chdir('..')
            continue
    
        os.chdir([i for i in os.listdir() if 'analysis' in i][0])
        if os.path.isfile('PSD_nostim.npy'):
            print(day)
            if day == '160218' or day == '221216': #  no stim was done too early before other recordings, not comparable for slow waves/delta, or nostim recording too short.
                os.chdir('..')
                os.chdir('..')
                continue
            else:
                LFP_resp_channels = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
                delta_power_vs_nostim_ALL[0].append(np.load('delta_power_auto_outliers_nostim.npy')[LFP_resp_channels])
                SW_waveform_vs_nostim_ALL[0].append(np.load('SW_waveform_sweeps_avg_nostim.npy')[LFP_resp_channels,:])
                Peak_dur_vs_nostim_ALL[0].append(np.load('Peak_dur_sweeps_median_nostim.npy')[LFP_resp_channels])
                Freq_vs_nostim_ALL[0].append(np.load('SW_frequency_sweeps_avg_nostim.npy')[LFP_resp_channels])
                Fslope_vs_nostim_ALL[0].append(np.load('SW_fslope_sweeps_median_nostim.npy')[LFP_resp_channels])
                Sslope_vs_nostim_ALL[0].append(np.load('SW_sslope_sweeps_median_nostim.npy')[LFP_resp_channels])
                Famp_vs_nostim_ALL[0].append(np.load('SW_famp_sweeps_median_nostim.npy')[LFP_resp_channels])
                Samp_vs_nostim_ALL[0].append(np.load('SW_samp_sweeps_median_nostim.npy')[LFP_resp_channels])
                
                # take the average of the whole baseline (better, averages out)
                # delta_power_vs_nostim_ALL[1].append(np.mean(np.loadtxt('delta_power_median_auto_outliers.csv', delimiter = ',')[[0,1,2,3],:], axis = 0)[LFP_resp_channels])
                # SW_waveform_vs_nostim_ALL[1].append(np.mean(np.load('SW_waveform_sweeps_avg.npy')[[0,1,2,3],:], axis = 0)[LFP_resp_channels])
                # Peak_dur_vs_nostim_ALL[1].append(np.mean(np.load('Peak_dur_sweeps_median.npy')[[0,1,2,3],:], axis = 0)[LFP_resp_channels])
                # Freq_vs_nostim_ALL[1].append(np.mean(np.load('SW_frequency_sweeps_avg.npy')[[0,1,2,3],:], axis = 0)[LFP_resp_channels])
                # Fslope_vs_nostim_ALL[1].append(np.mean(np.load('SW_fslope_sweeps_median.npy')[[0,1,2,3],:], axis = 0)[LFP_resp_channels])
                # Sslope_vs_nostim_ALL[1].append(np.mean(np.load('SW_sslope_sweeps_median.npy')[[0,1,2,3],:], axis = 0)[LFP_resp_channels])
                # Famp_vs_nostim_ALL[1].append(np.mean(np.load('SW_famp_sweeps_median.npy')[[0,1,2,3],:], axis = 0)[LFP_resp_channels])
                # Samp_vs_nostim_ALL[1].append(np.mean(np.load('SW_samp_sweeps_median.npy')[[0,1,2,3],:], axis = 0)[LFP_resp_channels])
                
                
                # take first or last baseline stim 
                if day == '221220_3' or day == '221213' or day == '221216' or day == '221219_1': # in these mice the nostim sweep was done after the last baseline
                    delta_power_vs_nostim_ALL[1].append(np.loadtxt('delta_power_auto_outliers.csv', delimiter = ',')[3,LFP_resp_channels])
                    SW_waveform_vs_nostim_ALL[1].append(np.load('SW_waveform_sweeps_avg.npy')[3,LFP_resp_channels,:])
                    Peak_dur_vs_nostim_ALL[1].append(np.load('Peak_dur_sweeps_median.npy')[3,LFP_resp_channels])
                    Freq_vs_nostim_ALL[1].append(np.load('SW_frequency_sweeps_avg.npy')[3,LFP_resp_channels])
                    Fslope_vs_nostim_ALL[1].append(np.load('SW_fslope_sweeps_median.npy')[3,LFP_resp_channels])
                    Sslope_vs_nostim_ALL[1].append(np.load('SW_sslope_sweeps_median.npy')[3,LFP_resp_channels])
                    Famp_vs_nostim_ALL[1].append(np.load('SW_famp_sweeps_median.npy')[3,LFP_resp_channels])
                    Samp_vs_nostim_ALL[1].append(np.load('SW_samp_sweeps_median.npy')[3,LFP_resp_channels])
                else: # in the other mice the nostim sweep was done before the first baseline
                    delta_power_vs_nostim_ALL[1].append(np.loadtxt('delta_power_auto_outliers.csv', delimiter = ',')[0,LFP_resp_channels])
                    SW_waveform_vs_nostim_ALL[1].append(np.load('SW_waveform_sweeps_avg.npy')[0,LFP_resp_channels,:])
                    Peak_dur_vs_nostim_ALL[1].append(np.load('Peak_dur_sweeps_median.npy')[0,LFP_resp_channels])
                    Freq_vs_nostim_ALL[1].append(np.load('SW_frequency_sweeps_avg.npy')[0,LFP_resp_channels])
                    Fslope_vs_nostim_ALL[1].append(np.load('SW_fslope_sweeps_median.npy')[0,LFP_resp_channels])
                    Sslope_vs_nostim_ALL[1].append(np.load('SW_sslope_sweeps_median.npy')[0,LFP_resp_channels])
                    Famp_vs_nostim_ALL[1].append(np.load('SW_famp_sweeps_median.npy')[3,LFP_resp_channels])
                    Samp_vs_nostim_ALL[1].append(np.load('SW_samp_sweeps_median.npy')[3,LFP_resp_channels])
        
        
                # average change between sweeps during baseline for every channel
                delta_power_baseline_diffs_ALL.append(np.mean(np.abs(np.diff(np.loadtxt('delta_power_median_auto_outliers.csv', delimiter = ',')[[0,1,2,3],:], axis = 0)), axis = 0)[LFP_resp_channels])
                SW_waveform_baseline_diffs_ALL.append(np.mean(np.abs(np.diff(np.load('SW_waveform_sweeps_avg.npy')[[0,1,2,3],:], axis = 0)), axis = 0)[LFP_resp_channels])
                Peak_dur_baseline_diffs_ALL.append(np.mean(np.abs(np.diff(np.load('Peak_dur_sweeps_median.npy')[[0,1,2,3],:], axis = 0)), axis = 0)[LFP_resp_channels])
                Freq_baseline_diffs_ALL.append(np.mean(np.abs(np.diff(np.load('SW_frequency_sweeps_avg.npy')[[0,1,2,3],:], axis = 0)), axis = 0)[LFP_resp_channels])
                Fslope_baseline_diffs_ALL.append(np.mean(np.abs(np.diff(np.load('SW_fslope_sweeps_median.npy')[[0,1,2,3],:], axis = 0)), axis = 0)[LFP_resp_channels])
                Sslope_baseline_diffs_ALL.append(np.mean(np.abs(np.diff(np.load('SW_sslope_sweeps_median.npy')[[0,1,2,3],:], axis = 0)), axis = 0)[LFP_resp_channels])
                Famp_baseline_diffs_ALL.append(np.mean(np.abs(np.diff(np.load('SW_famp_sweeps_median.npy')[[0,1,2,3],:], axis = 0)), axis = 0)[LFP_resp_channels])
                Samp_baseline_diffs_ALL.append(np.mean(np.abs(np.diff(np.load('SW_samp_sweeps_median.npy')[[0,1,2,3],:], axis = 0)), axis = 0)[LFP_resp_channels])
                
                # average relative change between sweeps during baseline for every channel
                delta_power_baseline_diffs_ALL_rel.append(np.mean(np.abs(np.diff(np.loadtxt('delta_power_median_auto_outliers.csv', delimiter = ',')[[0,1,2,3],:], axis = 0)/np.loadtxt('delta_power_median_auto_outliers.csv', delimiter = ',')[[0,1,2],:]), axis = 0)[LFP_resp_channels])
                SW_waveform_baseline_diffs_ALL_rel.append(np.mean(np.abs(np.diff(np.load('SW_waveform_sweeps_avg.npy')[[0,1,2,3],:], axis = 0)/np.load('SW_waveform_sweeps_avg.npy')[[0,1,2],:]), axis = 0)[LFP_resp_channels])
                Peak_dur_baseline_diffs_ALL_rel.append(np.mean(np.abs(np.diff(np.load('Peak_dur_sweeps_median.npy')[[0,1,2,3],:], axis = 0)/np.load('Peak_dur_sweeps_median.npy')[[0,1,2],:]), axis = 0)[LFP_resp_channels])
                Freq_baseline_diffs_ALL_rel.append(np.mean(np.abs(np.diff(np.load('SW_frequency_sweeps_avg.npy')[[0,1,2,3],:], axis = 0)/np.load('SW_frequency_sweeps_avg.npy')[[0,1,2],:]), axis = 0)[LFP_resp_channels])
                Fslope_baseline_diffs_ALL_rel.append(np.mean(np.abs(np.diff(np.load('SW_fslope_sweeps_median.npy')[[0,1,2,3],:], axis = 0)/np.load('SW_fslope_sweeps_median.npy')[[0,1,2],:]), axis = 0)[LFP_resp_channels])
                Sslope_baseline_diffs_ALL_rel.append(np.mean(np.abs(np.diff(np.load('SW_sslope_sweeps_median.npy')[[0,1,2,3],:], axis = 0)/np.load('SW_sslope_sweeps_median.npy')[[0,1,2],:]), axis = 0)[LFP_resp_channels])
                Famp_baseline_diffs_ALL_rel.append(np.mean(np.abs(np.diff(np.load('SW_famp_sweeps_median.npy')[[0,1,2,3],:], axis = 0)/np.load('SW_famp_sweeps_median.npy')[[0,1,2],:]), axis = 0)[LFP_resp_channels])
                Samp_baseline_diffs_ALL_rel.append(np.mean(np.abs(np.diff(np.load('SW_samp_sweeps_median.npy')[[0,1,2,3],:], axis = 0)/np.load('SW_samp_sweeps_median.npy')[[0,1,2],:]), axis = 0)[LFP_resp_channels])

                # power spectrum densities
                PSD_baseline_ALL.append(np.mean(np.load('PSD.npy')[[0,1,2,3],:,:], axis = 0)[:,np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]][LFP_resp_channels,:])
                PSD_lastpre_ALL.append(np.load('PSD.npy')[3,:,np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]][LFP_resp_channels,:])
                PSD_nostim_ALL.append(np.load('PSD_nostim.npy')[:,np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]][LFP_resp_channels,:])
                os.chdir('..')
                os.chdir('..')
    os.chdir('..')
    

        # delta_power_vs_nostim_ALL = pickle.load(open('delta_power_vs_nostim_ALL.pkl', 'rb'))
        # SW_waveform_vs_nostim_ALL = pickle.load(open('SW_waveform_vs_nostim_ALL.pkl', 'rb'))
        # Peak_dur_vs_nostim = pickle.load(open('Peak_dur_vs_nostim.pkl', 'rb'))
        # Freq_vs_nostim = pickle.load(open('Freq_vs_nostim.pkl', 'rb'))        
        # Fslope_vs_nostim = pickle.load(open('Fslope_vs_nostim.pkl', 'rb'))
        # Sslope_vs_nostim = pickle.load(open('Sslope_vs_nostim.pkl', 'rb'))
        # Famp_vs_nostim = pickle.load(open('Famp_vs_nostim.pkl', 'rb'))   
        # Samp_vs_nostim = pickle.load(open('Samp_vs_nostim.pkl', 'rb'))
        # PSD_lastpre_ALL = pickle.load(open('PSD_lastpre_ALL.pkl', 'rb'))
        # PSD_nostim_ALL = pickle.load(open('PSD_nostim_ALL.pkl', 'rb'))

# -------------- average change in slow wave characteristics between baseline vs nostim across channels
to_plot_chans = [delta_power_vs_nostim_ALL, Freq_vs_nostim_ALL, Peak_dur_vs_nostim_ALL, Fslope_vs_nostim_ALL, Sslope_vs_nostim_ALL, Famp_vs_nostim_ALL, Samp_vs_nostim_ALL]
to_plot_chans = [(np.hstack(i[1]) - np.hstack(i[0]))/np.hstack(i[0])*100 for i in to_plot_chans]
for ind, curr_param in enumerate(to_plot_chans):
    outliers = (np.logical_or(curr_param > (np.percentile(curr_param, 75) + 1.5*(np.abs(np.percentile(curr_param, 75) - np.percentile(curr_param, 25)))), curr_param < (np.percentile(curr_param, 25) - 1.5*(np.abs(np.percentile(curr_param, 75) - np.percentile(curr_param, 25))))))
    # print(outliers)
    to_plot_chans[ind] = np.delete(curr_param, outliers)
fig, ax = plt.subplots()
ax.boxplot(to_plot_chans[1:], showfliers = True, notch = True)
ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'], size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([-100,-50,0,50,100])
ax.set_ylim([-100,100])
# ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 14)
print('across channels, non-parametric test')
for i in range(6):
    t, p = scipy.stats.mannwhitneyu(to_plot_chans[i], np.zeros(len(to_plot_chans[i])))
    # t, p = scipy.stats.ttest_1samp(to_plot_chans[i], 0)
    print(t, p*6) #bonferroni correction




# average change in slow wave characteristics between baseline vs nostim across mice
delta_power_vs_nostim_allmice = np.asarray([[np.median(i) for i in j] for j in delta_power_vs_nostim_ALL])
Freq_vs_nostim_allmice = np.asarray([[np.median(i) for i in j] for j in Freq_vs_nostim_ALL])
peakdur_vs_nostim_allmice = np.asarray([[np.median(i) for i in j] for j in Peak_dur_vs_nostim_ALL])
famp_vs_nostim_allmice = np.asarray([[np.median(i) for i in j] for j in Famp_vs_nostim_ALL])
samp_vs_nostim_allmice = np.asarray([[np.median(i) for i in j] for j in Samp_vs_nostim_ALL])
fslope_vs_nostim_allmice = np.asarray([[np.median(i) for i in j] for j in Fslope_vs_nostim_ALL])
sslope_vs_nostim_allmice = np.asarray([[np.median(i) for i in j] for j in Sslope_vs_nostim_ALL])

to_plot_mice = [delta_power_vs_nostim_allmice, Freq_vs_nostim_allmice, peakdur_vs_nostim_allmice, fslope_vs_nostim_allmice, sslope_vs_nostim_allmice, famp_vs_nostim_allmice, samp_vs_nostim_allmice]
to_plot_mice = [(i[1] - i[0])/i[0]*100 for i in to_plot_mice]
for ind, curr_param in enumerate(to_plot_mice):
    outliers = (np.logical_or(curr_param > (np.percentile(curr_param, 75) + 1.5*(np.abs(np.percentile(curr_param, 75) - np.percentile(curr_param, 25)))), curr_param < (np.percentile(curr_param, 25) - 1.5*(np.abs(np.percentile(curr_param, 75) - np.percentile(curr_param, 25))))))
    # print(outliers)
    curr_param = np.delete(curr_param, outliers)
fig, ax = plt.subplots(figsize = (6,2))
ax.boxplot(to_plot_mice[1:], showfliers = True, notch = False)
ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'], size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([-50,-25,0,25,50])
ax.set_ylim([-55, 55])
ax.tick_params(axis="y", labelsize=16)  
ax.set_ylabel('mock pairing period \n vs baseline (%)', size=16)
plt.tight_layout()
plt.savefig('SW params nostim vs baseline.pdf', dpi = 1000, format = 'pdf')
plt.savefig('SW params nostim vs baseline.jpg', dpi = 1000, format = 'jpg')
# ax.set_ylim([-50,25])

# measure_to_test = to_plot_mice[0]
# scipy.stats.shapiro(measure_to_test)
# np.mean(measure_to_test)
# scipy.stats.ttest_1samp(measure_to_test,0)

print('across mice, parametric test')
for i in range(6):
    t, p = scipy.stats.ttest_1samp(to_plot_mice[i], 0)
    print(t, p*6) #bonferroni correction



# average change across mice, normalized to average change during baseline sweeps
delta_power_baseline_diffs_allmice = np.asarray([np.median(i) for i in delta_power_baseline_diffs_ALL]) # average across channels
Freq_baseline_diffs_allmice = np.asarray([np.median(i) for i in Freq_baseline_diffs_ALL])
Peak_dur_baseline_diffs_allmice = np.asarray([np.median(i) for i in Peak_dur_baseline_diffs_ALL])
Famp_baseline_diffs_allmice = np.asarray([np.median(i) for i in Famp_baseline_diffs_ALL])
Samp_baseline_diffs_allmice = np.asarray([np.median(i) for i in Samp_baseline_diffs_ALL])
Fslope_baseline_diffs_allmice = np.asarray([np.median(i) for i in Fslope_baseline_diffs_ALL])
Sslope_baseline_diffs_allmice = np.asarray([np.median(i) for i in Sslope_baseline_diffs_ALL])

delta_power_baseline_diffs_allmice_rel = np.asarray([np.median(i) for i in delta_power_baseline_diffs_ALL_rel])
Freq_baseline_diffs_allmice_rel = np.asarray([np.median(i) for i in Freq_baseline_diffs_ALL_rel])
Peak_dur_baseline_diffs_allmice_rel = np.asarray([np.median(i) for i in Peak_dur_baseline_diffs_ALL_rel])
Famp_baseline_diffs_allmice_rel = np.asarray([np.median(i) for i in Famp_baseline_diffs_ALL_rel])
Samp_baseline_diffs_allmice_rel = np.asarray([np.median(i) for i in Samp_baseline_diffs_ALL_rel])
Fslope_baseline_diffs_allmice_rel = np.asarray([np.median(i) for i in Fslope_baseline_diffs_ALL_rel])
Sslope_baseline_diffs_allmice_rel = np.asarray([np.median(i) for i in Sslope_baseline_diffs_ALL_rel])

baseline_vs_nostim = [delta_power_vs_nostim_allmice, Freq_vs_nostim_allmice, peakdur_vs_nostim_allmice, fslope_vs_nostim_allmice, sslope_vs_nostim_allmice, famp_vs_nostim_allmice, samp_vs_nostim_allmice]
baseline_vs_nostim_abs = [(i[1] - i[0]) for i in baseline_vs_nostim]
baseline_vs_nostim_rel = [(i[1] - i[0])/i[0] for i in baseline_vs_nostim]

baseline_diff_avg = [delta_power_baseline_diffs_allmice, Freq_baseline_diffs_allmice, Peak_dur_baseline_diffs_allmice, Fslope_baseline_diffs_allmice, Sslope_baseline_diffs_allmice, Famp_baseline_diffs_allmice, Samp_baseline_diffs_allmice]
baseline_diff_avg_rel = [delta_power_baseline_diffs_allmice_rel, Freq_baseline_diffs_allmice_rel, Peak_dur_baseline_diffs_allmice_rel, Fslope_baseline_diffs_allmice_rel, Sslope_baseline_diffs_allmice_rel, Famp_baseline_diffs_allmice_rel, Samp_baseline_diffs_allmice_rel]

# relative baseline vs nostim change normalized to average relative change during baseline sweeps
baseline_vs_nostim_norm = [np.abs(i)/j*100 for i, j in zip(baseline_vs_nostim_rel, baseline_diff_avg_rel)]
baseline_vs_nostim_diff_to_avg_baseline = [np.abs(i)*100-j*100 for i, j in zip(baseline_vs_nostim_rel, baseline_diff_avg_rel)]

to_plot = baseline_vs_nostim_norm
for ind, curr_param in enumerate(to_plot):
    outliers = (np.logical_or(curr_param > (np.percentile(curr_param, 75) + 1.5*(np.abs(np.percentile(curr_param, 75) - np.percentile(curr_param, 25)))), curr_param < (np.percentile(curr_param, 25) - 1.5*(np.abs(np.percentile(curr_param, 75) - np.percentile(curr_param, 25))))))
    # print(outliers)
    to_plot[ind] = np.delete(curr_param, outliers)

fig, ax = plt.subplots(figsize = (6,2))
mean_across_mice = [np.mean(i) for i in to_plot]
err_across_mice = [np.std(i)/np.sqrt(len(i)) for i in to_plot]
ax.bar(np.linspace(1,7,6), [np.mean(i) for i in mean_across_mice[1:]], yerr = err_across_mice[1:], color = 'k')
ax.set_xticks(np.linspace(1,7,6))
ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'], size = 16)
ax.set_yticks([0,50,100,150,200])
ax.set_ylim([0,200])
ax.tick_params(axis="y", labelsize=16)  
plt.tight_layout()
# plt.savefig('SW params nostim change vs baseline diff.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('SW params nostim change vs baseline diff.jpg', dpi = 1000, format = 'jpg')

print('across mice, parametric test')
for i in range(7):
    t, p = scipy.stats.ttest_1samp(to_plot[i], 100)
    print(t, p*6) #bonferroni correction



# ax.boxplot(baseline_vs_nostim_norm[1:], showfliers = True, notch = True)
# ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'], size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_yticks([0,1,2])
# ax.tick_params(axis="y", labelsize=16)  
# ax.set_ylabel('mock pairing period vs baseline', size=16)
# plt.tight_layout()
# plt.savefig('SW params nostim vs baseline.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('SW params nostim vs baseline.jpg', dpi = 1000, format = 'jpg')






#%%
# -------------------------------------------------------------------- PSD nostim vs baseline --------------------------------------------------------------

PSD_baseline_all_allmice = []
for i in PSD_baseline_ALL:
    if len(i.shape) > 1:
        PSD_baseline_all_allmice.append(np.mean(i, axis = 0))
    else:
        PSD_baseline_all_allmice.append(i)
PSD_baseline_all_allmice = np.asarray(PSD_baseline_all_allmice)/3500/1000

PSD_nostim_allmice = []
for i in PSD_nostim_ALL:
    if len(i.shape) > 1:
        PSD_nostim_allmice.append(np.mean(i, axis = 0))
    else:
        PSD_nostim_allmice.append(i)
PSD_nostim_allmice = np.asarray(PSD_nostim_allmice)/3500/1000

# # unpaired whisker PSD (doesn't work can't directly compare, not all the same mice...)
# PSD_unpaired_whisker_baseline_ALL = []
# os.chdir(os.path.join(overall_path, 'UNPAIRED_whisker'))
# days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# numb = len(days)
# for day in days:
#     os.chdir(day)    
#     os.chdir([i for i in os.listdir() if 'analysis' in i][0])
#     PSD_unpaired_whisker_baseline_ALL.append(np.mean(np.load('PSD.npy')[[0,1,2,3],:,:], axis = 0)[:,np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]][LFP_resp_channels,:])
#     os.chdir('..')
#     os.chdir('..')
# os.chdir('..')
# PSD_unpaired_whisker_baseline_all_allmice = []
# for i in PSD_unpaired_whisker_baseline_ALL:
#     if len(i.shape) > 1:
#         PSD_unpaired_whisker_baseline_all_allmice.append(np.mean(i, axis = 0))
#     else:
#         PSD_unpaired_whisker_baseline_all_allmice.append(i)
# PSD_unpaired_whisker_baseline_all_allmice = np.log(np.asarray(PSD_unpaired_whisker_baseline_all_allmice)/3500/1000)


mice_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11,12]
fftfreq = np.fft.fftfreq(3500, d = (1/1000))
fftfreq_to_plot = fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]][:]
patch = 1
#PSD baseline
fig, ax = plt.subplots(figsize = (7,4))
mice_avg = smooth(np.nanmean(PSD_baseline_all_allmice[mice_to_plot,:], axis = 0),3) # average and SEM across mice.
mice_std = smooth(np.nanstd(PSD_baseline_all_allmice[mice_to_plot,:], axis = 0),3)
ax.semilogy(fftfreq_to_plot, mice_avg, c = 'k')
ax.fill_between(fftfreq_to_plot, mice_avg + patch*mice_std/np.sqrt(len(mice_to_plot)), mice_avg - patch*mice_std/np.sqrt(len(mice_to_plot)), alpha = 0.1, color = 'k')
# nostim
mice_avg = smooth(np.nanmean(PSD_nostim_allmice[mice_to_plot,:], axis = 0),3)
mice_std = smooth(np.nanstd(PSD_nostim_allmice[mice_to_plot,:], axis = 0),3)
ax.plot(fftfreq_to_plot,mice_avg, c = 'purple')
ax.fill_between(fftfreq_to_plot, mice_avg + patch*mice_std/np.sqrt(len(mice_to_plot)), mice_avg - patch*mice_std/np.sqrt(len(mice_to_plot)), alpha = 0.1, color = 'purple')
# # unpaired whisker (doesn't work as not the same mice)
# mice_avg = smooth(np.nanmean(PSD_unpaired_whisker_baseline_all_allmice, axis = 0),3)
# mice_std = smooth(np.nanstd(PSD_unpaired_whisker_baseline_all_allmice, axis = 0),3)
# ax.plot(fftfreq_to_plot,mice_avg, c = 'blue')
# ax.fill_between(fftfreq_to_plot, mice_avg + patch*mice_std/np.sqrt(len(mice_to_plot)), mice_avg - patch*mice_std/np.sqrt(len(mice_to_plot)), alpha = 0.1, color = 'blue')
ax.set_xlim([0,45])
ax.set_ylim([2,12**4])
ax.tick_params(axis="x", labelsize=20)    
ax.tick_params(axis="y", labelsize=20, size = 12)   
ax.tick_params(which = 'minor', axis="y", size = 7)    
ax.set_xlabel('frequency (Hz)', size=20)
ax.set_ylabel('dB ($\mathregular{mV^2}$/Hz)', size=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# plt.savefig('PSD nostim vs baseline.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('PSD nostim vs baseline.pdf', dpi = 1000, format = 'pdf')




