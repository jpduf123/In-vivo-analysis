#-*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:58:39 2023

@author: Mann Lab
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.pyplot import cm
import neo
import quantities as pq
import elephant
import scipy
import scipy.signal
import os
import copy
import pickle
import natsort
from statistics import mean
import xml.etree.ElementTree as ET
# from load_intan_rhd_format import *
from operator import itemgetter
import pandas as pd
from scipy.linalg import lstsq

# home_directory = r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS\DOWN_pairing\160202'
# os.chdir(home_directory)

# day = os.getcwd()[-6:]


# os.chdir(home_directory)
# if os.path.exists(f'analysis_{day}') == False:
#     os.mkdir(f'analysis_{day}')

fs = 30000
resample_factor = 30
new_fs = fs/resample_factor

channelMapArray = np.array([[30, 46, 31, 47,  1, 49,  0, 48],
                            [28, 44, 29, 45,  3, 51,  2, 50],
                            [26, 42, 27, 43,  5, 53,  4, 52],
                            [24, 40, 25, 41,  7, 55,  6, 54],
                            [22, 38, 23, 39,  9, 57,  8, 56],
                            [20, 36, 21, 37, 11, 59, 10, 58],
                            [18, 34, 19, 35, 13, 61, 12, 60],
                            [16, 32, 17, 33, 15, 63, 14, 62]])

chanMap = channelMapArray.flatten()

def smooth(y, box_pts, axis = 0):
    box = np.ones(box_pts)/box_pts
    if len(y.shape) == 1:
        y_smooth = np.convolve(y, box, mode='same')
    else:
        y_smooth = np.apply_along_axis(lambda m : np.convolve(m, box, mode = 'same'), axis = axis, arr = y)
    return y_smooth

def cl():
    plt.close('all')

reanalyze = True

plot = True

highpass_cutoff = 4

b_notch, a_notch = scipy.signal.iirnotch(50, 100, 1000)


#%% ------------------------------------------------------------------- - delta power in each interstim interval ---------

reanalyze = True
plot = False 
new_fs = 1000

# days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i) and 'old' not in i]
for day in ['160310']:
# for day in days:
    os.chdir(day)
    print(day)
    
    to_plot_1_LFP = [0,1,2,3]
    to_plot_2_LFP = list(np.linspace(4,len(LFP_all_sweeps) - 1, len(LFP_all_sweeps) - 4, dtype = int))
    
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    

    #stims I exclude myself (I opt for an automatic exclusion based on outliers so not relevant)
    stims_for_delta = copy.deepcopy(stim_times)
    if '131121' in os.getcwd():
        stims_for_delta[4][9] = 0    
        stims_for_delta[8][41] = 0    
        stims_for_delta[8][55] = 0
    if '160218' in os.getcwd():
        stims_for_delta[2][39] = 0    
    if '160322' in os.getcwd():
        stims_for_delta[0][10] = 0    
        stims_for_delta[2][12] = 0    
    if '201121' in os.getcwd():
        stims_for_delta[9][2] = 0
        stims_for_delta[9][3] = 0
        stims_for_delta[5][15] = 0
    if '221213' in os.getcwd():
        stims_for_delta[4][6] = 0    
    if '221218' in os.getcwd():
        stims_for_delta[0][49] = 0
        stims_for_delta[0][51] = 0
        stims_for_delta[2][37] = 0
        stims_for_delta[3][33] = 0                
        stims_for_delta[4][51] = 0    
    if '221219_1' in os.getcwd():
        stims_for_delta[1][32] = 0
        stims_for_delta[3][30] = 0
        stims_for_delta[3][29] = 0   
        stims_for_delta[3][1] = 0   
        stims_for_delta[5][53] = 0   
    # if '160128' in os.getcwd():
    #     stims_for_delta[3][12] = 0    
    #     stims_for_delta[3][15] = 0    
    #     stims_for_delta[3][19] = 0    
    #     stims_for_delta[3][25] = 0    
    #     stims_for_delta[3][39] = 0
    #     stims_for_delta[6][1] = 0
    #     stims_for_delta[6][2] = 0
    #     stims_for_delta[7][1] = 0
    #     stims_for_delta[8][1] = 0
        
        
    if '121121' in os.getcwd():
        stims_for_delta[4][2] = 0    
        stims_for_delta[4][32] = 0    
        stims_for_delta[4][56] = 0    
        stims_for_delta[5][49] = 0    
        stims_for_delta[5][65] = 0    
        stims_for_delta[6][48] = 0    
        stims_for_delta[7][3] = 0    
        stims_for_delta[9][34] = 0
    if '160310' in os.getcwd():
        stims_for_delta[4][6] = 0      
        stims_for_delta[4][12] = 0      
        stims_for_delta[4][51] = 0   
        stims_for_delta[6][5] = 0      
        stims_for_delta[6][23] = 0      
        stims_for_delta[6][22] = 0      
        stims_for_delta[6][25] = 0      
        stims_for_delta[6][37] = 0     
        stims_for_delta[6][41] = 0      
        stims_for_delta[6][45] = 0      
        stims_for_delta[6][50:52] = 0      
        stims_for_delta[7][3] = 0      
        stims_for_delta[8][34] = 0      
    if '160414_D1' in os.getcwd():
        stims_for_delta[2][2] = 0      
        stims_for_delta[5][52] = 0      
        stims_for_delta[7][1:6] = 0      
    if '160426_D1' in os.getcwd():
        stims_for_delta[4][2] = 0    
        stims_for_delta[4][3] = 0    
        stims_for_delta[6][53] = 0  
        stims_for_delta[7][5] = 0  
        stims_for_delta[7][30] = 0  
        stims_for_delta[8][3] = 0    
        stims_for_delta[8][50] = 0    
        stims_for_delta[9][12] = 0    
        stims_for_delta[9][53] = 0    
    if '160519_B2' in os.getcwd():
        stims_for_delta[8][34] = 0    
        stims_for_delta[3][31] = 0    
        stims_for_delta[2][44] = 0    
        stims_for_delta[4][54] = 0    
        stims_for_delta[5][3] = 0
        stims_for_delta[6][57] = 0    
        stims_for_delta[7][2] = 0    
    if '160624_B2' in os.getcwd():
        stims_for_delta[0][40] = 0    
        stims_for_delta[1][40] = 0 
        stims_for_delta[4][51] = 0    
        stims_for_delta[5][4] = 0    
        stims_for_delta[5][14] = 0    
        stims_for_delta[6][51] = 0    
        stims_for_delta[7][4] = 0    
        stims_for_delta[7][42] = 0    
        stims_for_delta[7][56] = 0    
        stims_for_delta[8][2] = 0    
        stims_for_delta[8][15] = 0    
        stims_for_delta[8][40] = 0    
        stims_for_delta[8][45] = 0    
        stims_for_delta[8][47] = 0    
        stims_for_delta[9][5] = 0    
        stims_for_delta[9][14] = 0    
        stims_for_delta[9][26] = 0    
        stims_for_delta[9][50] = 0    
    if '160628_D1' in os.getcwd():
        stims_for_delta[0][10] = 0    
        stims_for_delta[1][14] = 0    
        stims_for_delta[4][47] = 0    
        stims_for_delta[4][51:58] = 0  
        stims_for_delta[5][4] = 0
        stims_for_delta[5][18] = 0
        stims_for_delta[5][41] = 0
        stims_for_delta[6][41] = 0    
        stims_for_delta[6][44:46] = 0    
        stims_for_delta[6][29] = 0    
        stims_for_delta[6][25:27] = 0    
        stims_for_delta[6][33] = 0    
        stims_for_delta[6][39] = 0    
        stims_for_delta[7][27] = 0    
        stims_for_delta[7][47] = 0   
        stims_for_delta[8][22] = 0 
        stims_for_delta[8][57] = 0 
        stims_for_delta[8][15:17] = 0    
        stims_for_delta[8][19:22] = 0    
        stims_for_delta[9][22:24] = 0    
        stims_for_delta[9][34] = 0    
        stims_for_delta[9][38] = 0    
    if '191121' in os.getcwd():
        stims_for_delta[4][16:18] = 0    
        stims_for_delta[3][6] = 0
        stims_for_delta[5][38] = 0    
        stims_for_delta[7][6] = 0 
        stims_for_delta[7][15] = 0
        stims_for_delta[7][41] = 0    
        stims_for_delta[9][31] = 0    
    if '201121' in os.getcwd():
        stims_for_delta[1][28] = 0    
        stims_for_delta[2][22] = 0    
        stims_for_delta[3][1] = 0    
        stims_for_delta[3][19] = 0    
        stims_for_delta[3][59] = 0    
        stims_for_delta[7][12] = 0    
        stims_for_delta[7][30] = 0    
        stims_for_delta[8][22] = 0    
    if '221220_3' in os.getcwd():
        stims_for_delta[0][59] = 0    
        stims_for_delta[3][16] = 0    
        stims_for_delta[3][21] = 0    
        stims_for_delta[4][24] = 0    
        stims_for_delta[5][9:11] = 0    
        stims_for_delta[5][48] = 0    
        stims_for_delta[6][18:20] = 0    
        stims_for_delta[6][31] = 0    
        stims_for_delta[6][34] = 0    
        stims_for_delta[6][35] = 0    
        stims_for_delta[6][68] = 0    
        stims_for_delta[7][14] = 0    
        stims_for_delta[7][22] = 0    
        stims_for_delta[7][37:39] = 0    
        stims_for_delta[7][56] = 0    
        stims_for_delta[7][59] = 0    
        stims_for_delta[8][7] = 0    
        stims_for_delta[8][39] = 0    
        stims_for_delta[9][2:4] = 0    
        stims_for_delta[9][7] = 0    
        stims_for_delta[9][10] = 0    
        stims_for_delta[9][43:48] = 0    
        stims_for_delta[9][55] = 0    
        stims_for_delta[9][58:61] = 0    
    if '281021' in os.getcwd():
        stims_for_delta[0][41] = 0 
        stims_for_delta[1][20] = 0 
        stims_for_delta[2][49] = 0 
        stims_for_delta[3][47] = 0 
        stims_for_delta[3][51] = 0 
        stims_for_delta[3][52] = 0 
        stims_for_delta[3][56] = 0 
        stims_for_delta[5][38] = 0
        stims_for_delta[5][15] = 0 
        stims_for_delta[5][28] = 0 
        stims_for_delta[5][33] = 0 
        stims_for_delta[6][14] = 0 
        stims_for_delta[6][18] = 0 
        stims_for_delta[6][35] = 0 
        stims_for_delta[7][8] = 0 
        stims_for_delta[7][36] = 0 
        stims_for_delta[8][1] = 0 
        stims_for_delta[9][49] = 0 
        stims_for_delta[9][54] = 0 
    if '291021' in os.getcwd():
        stims_for_delta[0][44] = 0 
        stims_for_delta[4][10] = 0 
        stims_for_delta[4][20] = 0 
        stims_for_delta[5][5] = 0 
        stims_for_delta[7][8] = 0    
        stims_for_delta[7][9] = 0    
        stims_for_delta[8][24] = 0    
        stims_for_delta[8][26] = 0    
        stims_for_delta[8][28] = 0  
        stims_for_delta[8][44] = 0   
        stims_for_delta[9][12] = 0    
        stims_for_delta[9][46] = 0    
        stims_for_delta[9][52] = 0    

    pickle.dump(stims_for_delta, open('stims_for_delta','wb'))

    stim_cumsum = np.cumsum(np.asarray([len(stims_for_delta[i]) for i in range(len(stims_for_delta))]))
    stim_cumsum = np.insert(stim_cumsum, 0, 0)
    all_stims_delta = np.zeros([64, sum([len(stims_for_delta[i]) for i in range(len(stims_for_delta))])])
    all_stims_delta_outliers = np.zeros([64, sum([len(stims_for_delta[i]) for i in range(len(stims_for_delta))])])
    all_stims_delta_auto_outliers = np.zeros([64, sum([len(stims_for_delta[i]) for i in range(len(stims_for_delta))])])

    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_min_rel_change = np.loadtxt('LFP_min_rel_change.csv', delimiter = ',')
    lfp_cutoff_resp_channels = 200
    LFP_resp_channels_cutoff = np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
    LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',').astype(int)
    os.chdir('..')
    
    def find_stim(stim, stim_cumsum= stim_cumsum):
        where_sweep = np.argwhere(stim_cumsum <= stim)[-1][0]
        print(f'sweep {where_sweep}, stim {stim - stim_cumsum[where_sweep]+ 1}')
        
    if reanalyze == False:
        os.chdir([i for i in os.listdir() if 'analysis' in i][0])
        # delta_channels = np.loadtxt('delta_channels.csv', delimiter = ',', dtype = int)
        fftfreq = np.loadtxt('fftfreq.csv', delimiter = ',')
        delta_power = np.loadtxt('delta_power.csv', delimiter = ',')
        delta_power_rel = np.loadtxt('delta_power_rel.csv', delimiter = ',')
        delta_power_rel_change = np.loadtxt('delta_power_rel_change.csv', delimiter = ',')
        delta_power_auto_outliers_rel = np.loadtxt('delta_power_auto_outliers_rel.csv', delimiter = ',')
        delta_power_auto_outliers_rel_change = np.loadtxt('delta_power_auto_outliers_rel_change.csv', delimiter = ',')
        delta_power_median_auto_outliers_rel = np.loadtxt('delta_power_median_auto_outliers_rel.csv', delimiter = ',')
        delta_power_median_auto_outliers_rel_change = np.loadtxt('delta_power_median_auto_outliers_rel_change.csv', delimiter = ',')
        PSD = np.load('PSD.npy')
        delta_lower = np.load('delta_lower.npy')
        delta_upper = np.load('delta_upper.npy')
        to_plot_1_delta = np.loadtxt('to_plot_1_delta.csv', delimiter = ',', dtype = int)
        to_plot_2_delta = np.loadtxt('to_plot_2_delta.csv', delimiter = ',', dtype = int)
        os.chdir('..')

    else: 
        to_plot_1_delta = [0,1,2,3]    
        to_plot_2_delta = list(np.linspace(4,len(LFP_all_sweeps) - 1, len(LFP_all_sweeps) - 4, dtype = int))
                
        exclude_before = 0.1
        exclude_after = 1.4
        
        delta_lower = 0.5
        delta_upper = 4
        
        fftfreq = np.fft.fftfreq(int((5 - exclude_before - exclude_after)*new_fs), d = (1/new_fs))
        hanning_window = np.tile(np.hanning((5 - exclude_before - exclude_after)*new_fs), (64, 1))
        hamming_window = np.tile(np.hamming((5 - exclude_before - exclude_after)*new_fs), (64, 1))
        
        PSD = np.empty([len(LFP_all_sweeps), LFP_all_sweeps[1].shape[0], int((5 - exclude_before - exclude_after)*new_fs)])
        PSD[:] = np.NaN
        PSD_median = np.empty([len(LFP_all_sweeps), LFP_all_sweeps[1].shape[0], int((5 - exclude_before - exclude_after)*new_fs)])
        PSD_median[:] = np.NaN

        delta_power = np.empty([10, 64])
        delta_power[:] = np.NaN
        delta_power_rel = np.empty([10, 64])
        delta_power_rel[:] = np.NaN
        
        delta_power_auto_outliers = np.empty([10, 64])
        delta_power_auto_outliers[:] = np.NaN
        delta_power_auto_outliers_rel = np.empty([10, 64])
        delta_power_auto_outliers_rel[:] = np.NaN
        delta_power_median_auto_outliers = np.empty([10, 64])
        delta_power_median_auto_outliers[:] = np.NaN
        delta_power_median_auto_outliers_rel = np.empty([10, 64])
        delta_power_median_auto_outliers_rel[:] = np.NaN

        auto_outlier_stims = [[] for i in range(10)]
        auto_outlier_stims_indices = [[] for i in range(10)]

        for ind_sweep, LFP in enumerate(LFP_all_sweeps):
            if LFP_all_sweeps[ind_sweep].size == 0:
                continue
            print(ind_sweep)
            
            FFT_current_sweep = np.zeros([len(stim_times[ind_sweep]), LFP_all_sweeps[ind_sweep].shape[0], int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
            FFT_current_sweep[:] = np.NaN
            FFT_current_sweep_outliers = np.zeros([len(stim_times[ind_sweep]), LFP_all_sweeps[ind_sweep].shape[0], int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
            FFT_current_sweep_outliers[:] = np.NaN
            FFT_current_sweep_auto_outliers = np.zeros([len(stim_times[ind_sweep]), LFP_all_sweeps[ind_sweep].shape[0], int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
            FFT_current_sweep_auto_outliers[:] = np.NaN

            for ind_stim, stim in enumerate(list(stim_times[ind_sweep])):
            
                #EXCLUDE first and last stim just in case there isnt enough time, makes it easier
                if ind_stim == 0 or ind_stim == len(stim_times[ind_sweep]) - 1:
                    continue
                
                # apply hamming window do FFT and calculate delta power in that interstim interval
                FFT_current_sweep[ind_stim,:,:] = np.fft.fft(hanning_window*LFP[:, int(stim+exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs)], axis = 1)
                FFT_current_sweep_outliers[ind_stim,:,:] = np.fft.fft(hanning_window*LFP[:, int(stim+exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs)], axis = 1)
                FFT_current_sweep_auto_outliers[ind_stim,:,:] = np.fft.fft(hanning_window*LFP[:, int(stim+exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs)], axis = 1)
                all_stims_delta[:,stim_cumsum[ind_sweep]+ind_stim] = np.transpose(np.nanmean(np.abs(FFT_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]]**2), axis = 0))
                all_stims_delta_outliers[:,stim_cumsum[ind_sweep]+ind_stim] = all_stims_delta[:,stim_cumsum[ind_sweep]+ind_stim]
                all_stims_delta_auto_outliers[:,stim_cumsum[ind_sweep]+ind_stim] = all_stims_delta[:,stim_cumsum[ind_sweep]+ind_stim]
                
                # if outlier defined by me, delete:
                if stims_for_delta[ind_sweep][ind_stim] == 0:
                    # print(f'{ind_stim}: continue')
                    FFT_current_sweep_outliers[ind_stim,:,:] = np.NaN
                    all_stims_delta_outliers[:,stim_cumsum[ind_sweep]+ind_stim] = 0

            # define auto outlier periods as exceeding statistical outlier threshold within each sweep in each LFP responsive channel
            for chan in range(64):
                curr_delta = all_stims_delta[chan, stim_cumsum[ind_sweep]:stim_cumsum[ind_sweep + 1]]
                outliers = (curr_delta > (np.percentile(curr_delta, 75) + 1.5*(np.abs(np.percentile(curr_delta, 75) - np.percentile(curr_delta, 25)))))
                if len(np.where(outliers == True)[0]) > 0:
                    all_stims_delta_auto_outliers[chan, np.where(outliers == True)[0] + stim_cumsum[ind_sweep]] = 0
                    FFT_current_sweep_auto_outliers[np.where(outliers == True)[0],:,:] = np.NaN
                delta_power_auto_outliers[ind_sweep, chan] = np.nanmean(curr_delta[~outliers])
                delta_power_median_auto_outliers[ind_sweep, chan] = np.nanmedian(curr_delta[~outliers])
                auto_outlier_stims[ind_sweep].append(outliers)
                auto_outlier_stims_indices[ind_sweep].append(np.where(outliers == True)[0])
                
            PSD[ind_sweep,:,:] = np.nanmean(np.abs(FFT_current_sweep_auto_outliers)**2, axis = 0)
            PSD_median[ind_sweep,:,:] = np.nanmedian(np.abs(FFT_current_sweep_auto_outliers)**2, axis = 0) # median across stims
            delta_power[ind_sweep,:] = np.nanmean(PSD[ind_sweep,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 0)
            # delta_power_median[ind_sweep,:] = np.nanmean(PSD_median[ind_sweep,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 0)

        delta_power_rel = delta_power/np.nanmean(delta_power[to_plot_1_delta,:], axis = 0)
        delta_power_rel_change = np.nanmean(delta_power_rel[to_plot_2_delta,:], axis = 0) - np.mean(delta_power_rel[to_plot_1_delta,:], axis = 0)
        
        delta_power_auto_outliers_rel = delta_power_auto_outliers/np.nanmean(delta_power_auto_outliers[to_plot_1_delta,:], axis = 0)
        delta_power_auto_outliers_rel_change = np.nanmean(delta_power_auto_outliers_rel[to_plot_2_delta,:], axis = 0) - np.mean(delta_power_auto_outliers_rel[to_plot_1_delta,:], axis = 0)
        delta_power_median_auto_outliers_rel = delta_power_median_auto_outliers/np.nanmean(delta_power_median_auto_outliers[to_plot_1_delta,:], axis = 0)
        delta_power_median_auto_outliers_rel_change = np.nanmean(delta_power_median_auto_outliers_rel[to_plot_2_delta,:], axis = 0) - np.nanmean(delta_power_median_auto_outliers_rel[to_plot_1_delta,:], axis = 0)

    
    # if plot:   
        
        # #delta power timecourse over whole recording
        # fig, ax = plt.subplots(8,8, figsize = (12,10)) 
        # fig.suptitle(f'delta in all stims {day}')
        # for ind, ax1 in enumerate(list(ax.flatten())):                        
        #     ax1.plot(all_stims_delta[chanMap[ind],:], linewidth = 1)
        #     # ax1.axhline(450000, linestyle = '--')
        #     if chanMap[ind] in LFP_resp_channels_cutoff:
        #         ax1.set_facecolor("y")
        #     ax1.set_yticks([])
        #     ax1.set_xticks([])
        #     ax1.set_title(str(chanMap[ind]), size = 4)
        # # plt.tight_layout()
        # plt.savefig('delta power whole recording no y-share', dpi = 1000)

        # #delta power timecourse over whole recording
        # fig, ax = plt.subplots(8,8, figsize = (12,10)) 
        # fig.suptitle(f'delta in all stims {day}')
        # for ind, ax1 in enumerate(list(ax.flatten())):                        
        #     ax1.plot(all_stims_delta_outliers[chanMap[ind],:], linewidth = 1)
        #     # ax1.axhline(450000, linestyle = '--')
        #     if chanMap[ind] in LFP_resp_channels_cutoff:
        #         ax1.set_facecolor("y")
        #     ax1.set_yticks([])
        #     ax1.set_xticks([])
        #     ax1.set_title(str(chanMap[ind]), size = 4)
        # # plt.tight_layout()
        # plt.savefig('delta power whole recording outliers no y-share', dpi = 1000)

        # #delta power timecourse over whole recording
        # fig, ax = plt.subplots(8,8, figsize = (12,10)) 
        # fig.suptitle(f'delta in all stims {day}')
        # for ind, ax1 in enumerate(list(ax.flatten())):                        
        #     ax1.plot(all_stims_delta_auto_outliers[chanMap[ind],:], linewidth = 1)
        #     # ax1.axhline(450000, linestyle = '--')
        #     if chanMap[ind] in LFP_resp_channels_cutoff:
        #         ax1.set_facecolor("y")
        #     ax1.set_yticks([])
        #     ax1.set_xticks([])
        #     ax1.set_title(str(chanMap[ind]), size = 4)
        # # plt.tight_layout()
        # plt.savefig('delta power whole recording auto outliers no y-share', dpi = 1000)
        
        # #delta power timecourse in each channel
        # fig, ax = plt.subplots(8,8, figsize = (12,10))
        # fig.suptitle(f'delta in all chans {day}')
        # for ind, ax1 in enumerate(list(ax.flatten())):
        #     ax1.plot(delta_power_rel[:,chanMap[ind]])
        #     if chanMap[ind] in LFP_resp_channels_cutoff:
        #         ax1.set_facecolor("y")
        #     # ax1.set_yticks([])
        #     ax1.set_xticks([])
        #     ax1.set_title(str(chanMap[chan]), size = 4)
        #     ax1.axvline(3.5)
        # plt.savefig('delta power in all chans', dpi = 1000)


        # #delta power timecourse in each channel
        # fig, ax = plt.subplots(8,8, figsize = (12,10))
        # fig.suptitle(f'delta in all chans {day}')
        # for ind, ax1 in enumerate(list(ax.flatten())):
        #     ax1.plot(delta_power_auto_outliers_rel[:,chanMap[ind]])
        #     if chanMap[ind] in LFP_resp_channels_cutoff:
        #         ax1.set_facecolor("y")
        #     # ax1.set_yticks([])
        #     ax1.set_xticks([])
        #     ax1.set_title(str(chanMap[chan]), size = 4)
        #     ax1.axvline(3.5)
        # plt.savefig('delta power in all chans auto outliers', dpi = 1000)

        
    PSD_before_LFP_resp = np.mean(np.mean(PSD[np.asarray(to_plot_1_delta)[:,None], LFP_resp_channels_cutoff, :], axis = 0),axis = 0)
    PSD_after_LFP_resp = np.mean(np.mean(PSD[np.asarray(to_plot_2_delta)[:,None], LFP_resp_channels_cutoff, :], axis = 0),axis = 0)
    PSD_after_relative_LFP_resp = PSD_after_LFP_resp/PSD_before_LFP_resp
    
    # if plot:    
        # fig, ax = plt.subplots()
        # fig.suptitle(f'LFP_resp {day}')
        # ax.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]], np.abs(PSD_before_LFP_resp[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]]), 'b')
        # ax.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]], np.abs(PSD_after_LFP_resp[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]]), 'r')
        # plt.savefig(f'PSD_{to_plot_1_delta}_vs_{to_plot_2_delta}', dpi = 1000)
            
        # #average before and after as bar plots
        # fig, ax = plt.subplots(8,8)
        # for chan in range(64):
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].bar(0,np.mean(delta_power[to_plot_1_delta,chan]))
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].bar(1,np.mean(delta_power[to_plot_2_delta,chan]))
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_yticklabels([])
        
        
        # # #correlation with change in LFP
        # fig, ax = plt.subplots(2,1)
        # fig.suptitle('delta power vs LFP, ALL vs lfp resp')
        # #take out outliers??
        # ax[0].scatter(LFP_min_rel_change, delta_power_rel_change)
        # ax[0].set_xlabel('relative LFP change')
        # ax[0].set_ylabel('relative delta change')
        # ax[1].scatter(LFP_min_rel_change[LFP_resp_channels_cutoff], delta_power_rel_change[LFP_resp_channels_cutoff])
        # ax[1].set_xlabel('relative LFP change')
        # ax[1].set_ylabel('relative delta change')
        # # # ax.set_xlim(right = 0)
        # # # ax.set_ylim(bottom = -60000)
        # plt.savefig('LFP change vs delta change', dpi = 1000)
    
        # # #correlation with change in LFP
        # fig, ax = plt.subplots(2,1)
        # fig.suptitle('delta power vs LFP, ALL vs lfp resp')
        # #take out outliers??
        # ax[0].scatter(LFP_min_rel_change, delta_power_auto_outliers_rel_change)
        # ax[0].set_xlabel('relative LFP change')
        # ax[0].set_ylabel('relative delta change')
        # ax[1].scatter(LFP_min_rel_change[LFP_resp_channels_cutoff], delta_power_auto_outliers_rel_change[LFP_resp_channels_cutoff])
        # ax[1].set_xlabel('relative LFP change')
        # ax[1].set_ylabel('relative delta change')
        # # # ax.set_xlim(right = 0)
        # # # ax.set_ylim(bottom = -60000)
        # plt.tight_layout()
        # plt.savefig('LFP change vs delta change auto outliers', dpi = 1000)


    # overall change in LFP resp channels:
    # overall_delta_change_LFP_resp = np.mean(delta_power_rel_change[LFP_resp_channels_cutoff])
    # overall_delta_change_all = np.mean(delta_power_rel_change)
    # print(f'overall relative delta change in LFP resp channels: {overall_delta_change_LFP_resp}')
    # print(f'overall relative delta change in all channels: {overall_delta_change_all}')
    
    # channel_avg_delta_change = (np.mean(PSD_after_LFP_resp[np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]]) - np.mean(PSD_before_LFP_resp[np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]]))/np.mean(PSD_before_LFP_resp[np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]])
    # print(f'overall relative delta change averaging PSD in all LFP resp channels: {channel_avg_delta_change}')
    
        
    
# -------------------------------------------------------- heatmap of LFP change and delta change

    # # -------------------- heatmap of LFP min rel change,only LFP responsive channels
    # fig, ax = plt.subplots()
    # to_plot = copy.deepcopy(LFP_min_rel_change)
    # to_plot[~np.asarray([i in LFP_resp_channels_cutoff for i in np.linspace(0,63,64).astype(int)])] = np.NaN
    # heatmap = ax.imshow(np.reshape(to_plot[chanMap], (8, 8))*100, cmap = 'bwr', vmin = -50, vmax = 50)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # # ax.spines['top'].set_visible(False)
    # # ax.spines['right'].set_visible(False)
    # # ax.spines['left'].set_visible(False)
    # # ax.spines['bottom'].set_visible(False)
    # cb = plt.colorbar(heatmap) #, boundaries=np.linspace(-55,0,100).astype(int)
    # cb.set_label(label='LFP response change (%)',size=15)   
    # cb.ax.set_ylim([-100,0])
    # cb.ax.tick_params(labelsize=14)
    # plt.tight_layout()
    # plt.savefig('LFP rel change colormap LFP responsive.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('LFP rel change colormap LFP responsive.jpg', dpi = 1000, format = 'jpg')
    
    # #  --------- heatmap of LFP min rel change all channels
    fig, ax = plt.subplots()
    to_plot = copy.deepcopy(LFP_min_rel_change)[chanMap]
    to_plot[27] = np.NaN
    to_plot[30] = np.NaN
    to_plot[35] = np.NaN
    to_plot[36] = np.NaN
    to_plot[43] = np.NaN
    to_plot[46] = np.NaN
    to_plot[57] = np.NaN
    to_plot[58] = np.NaN
    to_plot[63] = np.NaN    
    heatmap = ax.imshow(np.reshape(to_plot, (8, 8))*100, cmap = 'bwr', vmin = -100, vmax = 100)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    cb = plt.colorbar(heatmap)
    cb.set_label(label='LFP response change (%)',size=15)   
    cb.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig('LFP rel change colormap ALL.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('LFP rel change colormap ALL.jpg', dpi = 1000, format = 'jpg')

    # # fig, ax = plt.subplots(figsize = (2,5))
    # # cmap = cm.jet
    # # norm = colors.Normalize(vmin=-1, vmax=1)
    # # fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
    # #               cax=ax, ticks = [-1,-0.5,0,0.5,1])
    # # ax.set_yticklabels(list(map(str, np.linspace(100, -100, 5).astype(int))), size = 18)
    # # ax.set_ylabel('LFP response change (%)', size = 16)
    # # plt.tight_layout()
    # # plt.savefig('LFP rel change colormap legend.pdf', dpi = 1000, format = 'pdf')
    # # plt.savefig('LFP rel change colormap legend.jpg', dpi = 1000, format = 'jpg')




    # # # ---------------------------------------------------------------------------- heatmap of delta power rel change
    # #  --------- heatmap of LFP min rel change responsive channels
    # fig, ax = plt.subplots()
    # to_plot = copy.deepcopy(delta_power_auto_outliers_rel_change)
    # to_plot[~np.asarray([i in LFP_resp_channels_cutoff for i in np.linspace(0,63,64).astype(int)])] = np.NaN
    # heatmap = ax.imshow(np.reshape(to_plot[chanMap], (8, 8))*100, cmap = 'bwr', vmin = -70, vmax = 70)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # # ax.spines['top'].set_visible(False)
    # # ax.spines['right'].set_visible(False)
    # # ax.spines['left'].set_visible(False)
    # # ax.spines['bottom'].set_visible(False)
    # cb = plt.colorbar(heatmap)
    # cb.set_label(label='delta power change (%)',size=15)   
    # cb.ax.tick_params(labelsize=14)
    # plt.tight_layout()
    # plt.savefig('delta power rel change colormap LFP responsive.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('delta power rel change colormap LFP responsive.jpg', dpi = 1000, format = 'jpg')

    fig, ax = plt.subplots()
    to_plot = copy.deepcopy(delta_power_auto_outliers_rel_change)[chanMap]
    # to_plot[chanMap][27] = 0
    to_plot[27] = np.NaN
    to_plot[30] = np.NaN
    to_plot[35] = np.NaN
    to_plot[36] = np.NaN
    to_plot[43] = np.NaN
    to_plot[46] = np.NaN
    to_plot[57] = np.NaN
    to_plot[58] = np.NaN
    to_plot[63] = np.NaN

    heatmap = ax.imshow(np.reshape(to_plot, (8, 8))*100, cmap = 'bwr', vmin = -100, vmax = 100)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    cb = plt.colorbar(heatmap)
    cb.set_label(label='delta power change (%)',size=15)   
    cb.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig('delta power rel change colormap ALL.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('delta power rel change colormap ALL.jpg', dpi = 1000, format = 'jpg')
    
    # # heatmap of delta power rel change MEDIAN ACROSS STIMS
    # fig, ax = plt.subplots()
    # to_plot = copy.deepcopy(delta_power_median_auto_outliers_rel_change)
    # to_plot[~np.asarray([i in LFP_resp_channels_cutoff for i in np.linspace(0,63,64).astype(int)])] = np.NaN
    # heatmap = ax.imshow(np.reshape(to_plot[chanMap], (8, 8))*100, cmap = 'bwr', vmin = -70, vmax = 70)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # # ax.spines['top'].set_visible(False)
    # # ax.spines['right'].set_visible(False)
    # # ax.spines['left'].set_visible(False)
    # # ax.spines['bottom'].set_visible(False)
    # cb = plt.colorbar(heatmap)
    # cb.set_label(label='delta power change (%)',size=15)   
    # cb.ax.tick_params(labelsize=14)
    # plt.tight_layout()
    # plt.savefig('delta power median across stims rel change colormap LFP responsive.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('delta power median across stims rel change colormap LFP responsive.jpg', dpi = 1000, format = 'jpg')

    # fig, ax = plt.subplots()
    # to_plot = copy.deepcopy(delta_power_median_auto_outliers_rel_change)
    # heatmap = ax.imshow(np.reshape(to_plot[chanMap], (8, 8))*100, cmap = 'bwr', vmax = 100)
    # # Minor ticks
    # ax.set_xticks(np.arange(.5, 7.5, 1), minor=True)
    # ax.set_yticks(np.arange(.5, 7.5, 1), minor=True)
    # # Gridlines based on minor ticks
    # ax.grid(which='minor', color='k', linestyle='-', linewidth=0.1)
    # ax.tick_params(which='minor', bottom=False, left=False)
    # ax.grid(color='w', linestyle='-', linewidth=2)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # # ax.spines['top'].set_visible(False)
    # # ax.spines['right'].set_visible(False)
    # # ax.spines['left'].set_visible(False)
    # # ax.spines['bottom'].set_visible(False)
    # cb = plt.colorbar(heatmap)
    # cb.set_label(label='delta power change (%)',size=15)   
    # cb.ax.tick_params(labelsize=14)
    # plt.tight_layout()
    # plt.savefig('delta power median across stims rel change colormap ALL.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('delta power median across stims rel change colormap ALL.jpg', dpi = 1000, format = 'jpg')

    # # fig, ax = plt.subplots(figsize = (0.1,5))
    # # cmap = cm.jet
    # # norm = colors.Normalize(vmin=-1, vmax=1)
    # # fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
    # #               cax=ax, ticks = [-1,-0.5,0,0.5,1])
    # # ax.set_yticklabels(list(map(str, np.linspace(100, -100, 5).astype(int))), size = 18)
    # # ax.set_ylabel('delta power change (%)', size = 16)
    # # plt.tight_layout()
    # # plt.savefig('delta power rel change colormap legend.pdf', dpi = 1000, format = 'pdf')
    # # plt.savefig('delta power rel change colormap legend.jpg', dpi = 1000, format = 'jpg')



    if reanalyze == True:
        os.chdir([i for i in os.listdir() if 'analysis' in i][0])
        np.savetxt('fftfreq.csv', fftfreq, delimiter = ',')
        np.savetxt('delta_power.csv', delta_power, delimiter = ',')
        np.savetxt('delta_power.csv', delta_power, delimiter = ',')
        np.savetxt('delta_power_rel.csv', delta_power_rel, delimiter = ',')
        np.savetxt('delta_power_rel_change.csv', delta_power_rel_change, delimiter = ',')
        np.savetxt('delta_power_auto_outliers.csv', delta_power_auto_outliers, delimiter = ',')
        np.savetxt('delta_power_auto_outliers_rel.csv', delta_power_auto_outliers_rel, delimiter = ',')
        np.savetxt('delta_power_auto_outliers_rel_change.csv', delta_power_auto_outliers_rel_change, delimiter = ',')
        np.savetxt('delta_power_median_auto_outliers.csv', delta_power_median_auto_outliers, delimiter = ',')
        np.savetxt('delta_power_median_auto_outliers_rel.csv', delta_power_median_auto_outliers_rel, delimiter = ',')
        np.savetxt('delta_power_median_auto_outliers_rel_change.csv', delta_power_median_auto_outliers_rel_change, delimiter = ',')
        np.save('PSD.npy', PSD)
        np.save('PSD_median.npy', PSD_median)
        np.savetxt('to_plot_1_delta.csv', to_plot_1_delta, delimiter = ',')
        np.savetxt('to_plot_2_delta.csv', to_plot_2_delta, delimiter = ',')
        np.save('delta_lower.npy', delta_lower)
        np.save('delta_upper.npy', delta_upper)
        pickle.dump(auto_outlier_stims_indices, open('auto_outlier_stims_indices','wb'))
        os.chdir('..')
        
    os.chdir('..')

    # cl()
    
    
#%% how many stims are taken out automatically, average per sweep and channel
total_avg = []
for cond in ['UP_pairing', 'DOWN_pairing']:
    os.chdir(cond)
    days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i) and 'old' not in i]
    for day in days:
        os.chdir(day)
        os.chdir([i for i in os.listdir() if 'analysis' in i][0])
        auto_outlier_stims_indices = pickle.load(open('auto_outlier_stims_indices','rb'))
        LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
        
        outliers_per_channel = [[len(i[j]) for i in auto_outlier_stims_indices] for j in range(64)]
        avg_outliers = np.asarray([sum(i)/10 for i in outliers_per_channel])
        print(f'{day}, average auto outlier stims per channel and sweep: {np.mean(avg_outliers[LFP_resp_channels_cutoff])}')
        total_avg.append(np.asarray(avg_outliers[LFP_resp_channels_cutoff]))
        
        os.chdir('..')
        os.chdir('..')
    os.chdir('..')

all_chans_avg = np.hstack(total_avg)
print(f'average across all channels in all mice: {np.mean(all_chans_avg)}')


#%% LFP vs delta all mice
exclude_outliers = True
normalize_within_mouse = False
only_LFP_depression = False
only_delta_depression = False

# os.chdir(os.path.join(overall_path, r'UP_pairing'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
numb = len(days)

X = []
Y = []
for day in days:
    os.chdir(day)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    channels = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
    curr_X = np.loadtxt('delta_power_auto_outliers_rel_change.csv', delimiter = ',')[channels]*100
    curr_Y = np.loadtxt('LFP_min_rel_change.csv', delimiter = ',')[channels]*100
    if normalize_within_mouse == True:
        curr_X = curr_X/np.min(curr_X)
        curr_Y = curr_Y/np.min(curr_Y)
    X.append(curr_X)
    Y.append(curr_Y)
    os.chdir('..')
    os.chdir('..')
X = np.hstack(X)
Y = np.hstack(Y)

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
slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
print(f'{r**2} and {p} for {len(X)} channels')
ax.scatter(X,Y, color = 'k')
ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
# ax.axhline(np.mean(Y))
ax.set_xlabel('delta power change (% baseline)', size = 16)
ax.set_ylabel('LFP change (% baseline)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis="x", labelsize=16)    
ax.tick_params(axis="y", labelsize=16) 
plt.tight_layout()
# plt.savefig('delta vs LFP.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('delta vs LFP.jpg', dpi = 1000, format = 'jpg')



#%% # PSD before vs after all mice
# frequencies on 3.5 second interstim interval
overall_path = r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS'


fftfreq = np.fft.fftfreq(3500, d = (1/1000))
fftfreq_to_plot = fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 100 >= fftfreq))[0]][:]

patch = 1
os.chdir(os.path.join(overall_path, r'UP_pairing'))
# os.chdir(os.path.join(overall_path, r'UP_pairing'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
numb = len(days)

PSD_UP_all = []
PSD_median_UP_all = []
for day in days:
    os.chdir(day)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    channels = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
    if channels.size == 1:
        PSD_UP_all.append(np.squeeze(np.load('PSD.npy')[:,channels,:])) # mean over trials
        PSD_median_UP_all.append(np.squeeze(np.load('PSD_median.npy')[:,channels,:])) # median over trials
    else:
        PSD_UP_all.append(np.squeeze(np.median(np.load('PSD.npy')[:,channels,:], axis = 1))) # average across channels
        PSD_median_UP_all.append(np.squeeze(np.median(np.load('PSD_median.npy')[:,channels,:], axis = 1))) # average across channels
    os.chdir('..') 
    os.chdir('..')
    
PSD_UP = np.asarray(PSD_UP_all)[:,:,np.where(np.logical_and(0.1 <= fftfreq , 100 >= fftfreq))[0]]/3500/1000 # divide by sampling frequency and number of samples
PSD_median_UP = np.asarray(PSD_median_UP_all)[:,:,np.where(np.logical_and(0.1 <= fftfreq , 100 >= fftfreq))[0]]/3500/1000
PSD_UP_before_mean = np.nanmean(PSD_UP[:,[0,1,2,3],:], axis = 1)
PSD_UP_after_mean = np.nanmean(PSD_UP[:,[4,5,6,7,8,9],:], axis = 1)

fig, ax = plt.subplots(figsize = (7,4))
# ax.plot(PSD_DOWN_before_mean.T)
# UP
mice_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11]
smooth_over_freq = 2
#before
mice_avg = scipy.ndimage.gaussian_filter(np.nanmean(PSD_UP_before_mean[mice_to_plot,:], axis = 0), smooth_over_freq) # average and SEM across mice.
mice_std = scipy.ndimage.gaussian_filter(np.nanstd(PSD_UP_before_mean[mice_to_plot,:], axis = 0),smooth_over_freq)
ax.semilogy(fftfreq_to_plot, mice_avg, c = 'r')
ax.fill_between(fftfreq_to_plot, mice_avg + patch*mice_std/np.sqrt(len(mice_to_plot)), mice_avg - patch*mice_std/np.sqrt(len(mice_to_plot)), alpha = 0.1, color = 'r')
#after
mice_avg = scipy.ndimage.gaussian_filter(np.nanmean(PSD_UP_after_mean[mice_to_plot,:], axis = 0),smooth_over_freq)
mice_std = scipy.ndimage.gaussian_filter(np.nanstd(PSD_UP_after_mean[mice_to_plot,:], axis = 0),smooth_over_freq)
ax.plot(fftfreq_to_plot,mice_avg, c = 'r', linestyle = '--')
ax.fill_between(fftfreq_to_plot, mice_avg + patch*mice_std/np.sqrt(len(mice_to_plot)), mice_avg - patch*mice_std/np.sqrt(len(mice_to_plot)), alpha = 0.1, color = 'r')
ax.set_xlim([0.5,45])
ax.set_ylim([2,12**4])
ax.fill_between([0.5,4], [100000,100000], alpha = 0.1, color = 'k')
ax.tick_params(axis="x", labelsize=20)    
ax.tick_params(axis="y", labelsize=20, size = 12)   
ax.tick_params(which = 'minor', axis="y", size = 7)    
ax.set_xlabel('frequency (Hz)', size=20)
ax.set_ylabel('LFP power density ($\mathregular{mV^2}$/Hz)', size=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('PSD all UP.jpg', dpi = 1000, format = 'jpg')
plt.savefig('PSD all UP.pdf', dpi = 1000, format = 'pdf')
np.savetxt('PSD before UP log.csv', np.log(PSD_UP_before_mean), delimiter = ',')
np.savetxt('PSD after UP log.csv', np.log(PSD_UP_after_mean), delimiter = ',')
np.savetxt('PSD before UP.csv', PSD_UP_before_mean, delimiter = ',')
np.savetxt('PSD after UP.csv', PSD_UP_after_mean, delimiter = ',')



patch = 1
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
# os.chdir(os.path.join(overall_path, r'UP_pairing'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
numb = len(days)

PSD_DOWN = []
PSD_median_DOWN = []
for day in days:
    os.chdir(day)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    channels = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
    if channels.size == 1:
        PSD_DOWN.append(np.squeeze(np.load('PSD.npy')[:,channels,:]))
        PSD_median_DOWN.append(np.squeeze(np.load('PSD_median.npy')[:,channels,:]))
    else:
        PSD_DOWN.append(np.squeeze(np.median(np.load('PSD.npy')[:,channels,:], axis = 1)))
        PSD_median_DOWN.append(np.squeeze(np.median(np.load('PSD_median.npy')[:,channels,:], axis = 1)))
    os.chdir('..')
    os.chdir('..')
    
PSD_DOWN = np.asarray(PSD_DOWN)[:,:,np.where(np.logical_and(0.1 <= fftfreq , 100 >= fftfreq))[0]]/3500/1000
PSD_median_DOWN = np.asarray(PSD_median_DOWN)[:,:,np.where(np.logical_and(0.1 <= fftfreq , 100 >= fftfreq))[0]]/3500/1000
PSD_DOWN_before_mean = np.mean(PSD_DOWN[:,[0,1,2,3],:], axis = 1)
PSD_DOWN_after_mean = np.mean(PSD_DOWN[:,[4,5,6,7,8,9],:], axis = 1)

fig, ax = plt.subplots(figsize = (7,4)) 
#DOWN
#before
mice_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11]
smooth_over_freq = 2
# ax.plot(fftfreq_to_plot, PSD_DOWN_before_mean.T)
mice_avg = scipy.ndimage.gaussian_filter(np.mean(PSD_DOWN_before_mean[mice_to_plot,:], axis = 0),smooth_over_freq)
mice_std = scipy.ndimage.gaussian_filter(np.std(PSD_DOWN_before_mean[mice_to_plot,:], axis = 0),smooth_over_freq)
ax.semilogy(fftfreq_to_plot, mice_avg, c = 'k') 
ax.fill_between(fftfreq_to_plot, mice_avg + patch*mice_std/np.sqrt(len(mice_to_plot)), mice_avg - patch*mice_std/np.sqrt(len(mice_to_plot)), alpha = 0.1, color = 'k')
#after
mice_avg = scipy.ndimage.gaussian_filter(np.mean(PSD_DOWN_after_mean[mice_to_plot,:], axis = 0),smooth_over_freq)
mice_std = scipy.ndimage.gaussian_filter(np.std(PSD_DOWN_after_mean[mice_to_plot,:], axis = 0),smooth_over_freq)
ax.plot(fftfreq_to_plot,mice_avg, c = 'k', linestyle = '--')
ax.fill_between(fftfreq_to_plot, mice_avg + patch*mice_std/np.sqrt(len(mice_to_plot)), mice_avg - patch*mice_std/np.sqrt(len(mice_to_plot)), alpha = 0.1, color = 'k')
ax.set_xlim([0,45])
ax.set_ylim([2,12**4])
ax.tick_params(axis="x", labelsize=20)    
ax.tick_params(axis="y", labelsize=20, size = 12)   
ax.tick_params(which = 'minor', axis="y", size = 7)    
ax.set_xlabel('frequency (Hz)', size=20)
ax.set_ylabel('LFP power density ($\mathregular{mV^2}$/Hz)', size=20)
ax.fill_between([0.5,4], [100000,100000], alpha = 0.1, color = 'k')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('PSD all DOWN.jpg', dpi = 1000, format = 'jpg')
plt.savefig('PSD all DOWN.pdf', dpi = 1000, format = 'pdf')
np.savetxt('PSD before DOWN log.csv', np.log(PSD_DOWN_before_mean), delimiter = ',')
np.savetxt('PSD after DOWN log.csv', np.log(PSD_DOWN_after_mean), delimiter = ',')
np.savetxt('PSD before DOWN.csv', PSD_DOWN_before_mean, delimiter = ',')
np.savetxt('PSD after DOWN.csv', PSD_DOWN_after_mean, delimiter = ',')
os.chdir('..')

 
# # relative power change after pairing as a function of frequency
# smooth_over_freq = 0.5
# import scipy.ndimage as si

# freq_change_rel_UP = (si.gaussian_filter(PSD_UP_after_mean, smooth_over_freq) - si.gaussian_filter(PSD_UP_before_mean, smooth_over_freq))/si.gaussian_filter(PSD_UP_before_mean, smooth_over_freq)
# freq_change_rel_DOWN = (si.gaussian_filter(PSD_DOWN_after_mean, smooth_over_freq) - si.gaussian_filter(PSD_DOWN_before_mean, smooth_over_freq))/si.gaussian_filter(PSD_DOWN_before_mean, smooth_over_freq)

# freq_change_UP = (si.gaussian_filter(PSD_UP_after_mean, smooth_over_freq) - si.gaussian_filter(PSD_UP_before_mean, smooth_over_freq))
# freq_change_DOWN = (si.gaussian_filter(PSD_DOWN_after_mean, smooth_over_freq) - si.gaussian_filter(PSD_DOWN_before_mean, smooth_over_freq))


# #UP
# mice_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11]
# fig, ax = plt.subplots(figsize = (5,3)) 
# to_plot = si.gaussian_filter(freq_change_rel_UP*100, (0,smooth_over_freq))
# mice_avg = np.mean(to_plot[mice_to_plot,:], axis = 0)
# mice_std = np.std(to_plot[mice_to_plot,:], axis = 0)
# ax.plot(fftfreq_to_plot, mice_avg, c = 'r') 
# ax.fill_between(fftfreq_to_plot, mice_avg + patch*mice_std/np.sqrt(len(mice_to_plot)), mice_avg - patch*mice_std/np.sqrt(len(mice_to_plot)), alpha = 0.1, color = 'r')
# #DOWN
# mice_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11]
# to_plot = si.gaussian_filter(freq_change_rel_DOWN*100, (0,smooth_over_freq))
# mice_avg = np.mean(to_plot[mice_to_plot,:], axis = 0)
# mice_std = np.std(to_plot[mice_to_plot,:], axis = 0)
# ax.plot(fftfreq_to_plot,mice_avg, c = 'k', linestyle = '--')
# ax.fill_between(fftfreq_to_plot, mice_avg + patch*mice_std/np.sqrt(len(mice_to_plot)), mice_avg - patch*mice_std/np.sqrt(len(mice_to_plot)), alpha = 0.1, color = 'k')
# ax.set_xlim([0,45])
# ax.set_ylim([-60,20])
# ax.tick_params(axis="x", labelsize=16)    
# ax.tick_params(axis="y", labelsize=16)    
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xlabel('frequency (Hz)', size=18)
# ax.set_ylabel('relative change (%)', size=18)
# plt.tight_layout()
# np.savetxt('rel change per freq UP.csv', freq_change_rel_UP, delimiter = ',')
# np.savetxt('rel change per freq DOWN.csv', freq_change_rel_DOWN, delimiter = ',')


# fig, ax = plt.subplots()
# avg_change = (scipy.ndimage.gaussian_filter(np.mean(PSD_DOWN_after_mean[mice_to_plot,:], axis = 0),3) - scipy.ndimage.gaussian_filter(np.mean(PSD_DOWN_before_mean[mice_to_plot,:], axis = 0),3))/scipy.ndimage.gaussian_filter(np.mean(PSD_DOWN_before_mean[mice_to_plot,:], axis = 0),3)
# ax.plot(fftfreq_to_plot, avg_change)
# avg_change = (scipy.ndimage.gaussian_filter(np.mean(PSD_UP_after_mean[mice_to_plot,:], axis = 0),3) - scipy.ndimage.gaussian_filter(np.mean(PSD_UP_before_mean[mice_to_plot,:], axis = 0),3))/scipy.ndimage.gaussian_filter(np.mean(PSD_UP_before_mean[mice_to_plot,:], axis = 0),3)
# ax.plot(fftfreq_to_plot, avg_change)



#%% Wavelet transform of SOs

# frequencies on 3.5 second interstim interval
overall_path = r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS'

# parameters for CWT
freq_min = 0.1 # low spindle band
freq_max = 45 # high first harmonic
width_min = 5*new_fs/(2*freq_min*np.pi) # from scipy documentation, way of getting width of wavelet for specific fundamental frequency
width_max = 5*new_fs/(2*freq_max*np.pi)
width_nr = freq_max*2 # number of wavelets widths to convolve
widths = 5*new_fs/(2*np.linspace(freq_min, freq_max, width_nr)*np.pi) # in ms

patch = 1
os.chdir(os.path.join(overall_path, r'UP_pairing'))
# os.chdir(os.path.join(overall_path, r'UP_pairing'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
numb = len(days)
# for day in days[1:3]:
for day in ['160310']:

    os.chdir(day)
    print(day)
    
    to_plot_1_LFP = [0,1,2,3]
    to_plot_2_LFP = [4,5,6,7,8,9]
    
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    stim_times = pickle.load(open('stim_times','rb'))

    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    channels = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
    UP_Cross_sweeps = pickle.load(open('UP_Cross_sweeps', 'rb'))

    os.chdir('..')
    
    for chan in channels[[34]]:
        fig, ax = plt.subplots(2,3,sharex = True, figsize = (15,5))
        for sweep_ind, sweep in enumerate([3,9]):
            curr_LFP = scipy.signal.filtfilt(b_notch, a_notch, LFP_all_sweeps[sweep][chan])
            #continuous complex morlet wavelet transform
            CWT = scipy.signal.cwt(curr_LFP, scipy.signal.morlet2, widths, dtype = 'complex128')        
            # STFT = scipy.signal.spectrogram(LFP_all_sweeps[sweep][channels[2]], fs = 1000)
            
            # average of LFP and CWT on the peak of UP state (within 1 sec preceding the UP crossing)
            UP_LFP_average = []
            UP_CWT_average = []
            for UP_state_ind, UP_state in enumerate(UP_Cross_sweeps[sweep][chan]):
                UP_peak = np.argmin(curr_LFP[UP_state - 1000:UP_state]) + UP_state - 1000
                UP_LFP_average.append(curr_LFP[UP_peak - 1000:UP_peak + 1000])
                UP_CWT_average.append(CWT[:,UP_peak - 1000:UP_peak + 1000])
            
            if sweep_ind == 0:
                vmax = np.max(np.abs(CWT[:]))/1.5
            y_tick_scaling = 8
            
            if sweep_ind == 0:
                # plot SOs extracted in LFP pairing with CWT of average LFP and average CWT
                # fig.suptitle(f'{day}, {chan}, {delta power}')
                ax.flatten()[0].plot(np.asarray(UP_LFP_average).T/1000, linewidth = 0.1)
                ax.flatten()[0].plot(np.mean(np.asarray(UP_LFP_average).T/1000, axis = 1), color = 'k')
                ax.flatten()[0].set_ylim([-2,1])
                ax.flatten()[0].tick_params(axis = 'y', labelsize = 18)
                ax.flatten()[0].set_ylabel('LFP (mV)', size = 18)
                ax.flatten()[0].tick_params(axis = 'x', labelsize = 18)

                if sweep_ind == 0:
                    vmax = np.max(np.abs(scipy.signal.cwt(np.mean(np.asarray(UP_LFP_average).T, axis = 1), scipy.signal.morlet2, widths, dtype = 'complex128')))/1.3
                ax.flatten()[1].imshow(np.abs(scipy.signal.cwt(np.mean(np.asarray(UP_LFP_average).T, axis = 1), scipy.signal.morlet2, widths, dtype = 'complex128')), aspect = 'auto', cmap = 'jet', vmax = vmax)
                ax.flatten()[1].set_yticks(np.arange(0, width_nr - 1, 10*(width_nr/freq_max)))
                ax.flatten()[1].set_yticklabels(list(map(str, np.arange(0, freq_max, 10))), size = 18)
                ax.flatten()[1].set_xlim([580,1420])
                ax.flatten()[1].set_xticks([600,800,1000,1200,1400])            
                ax.flatten()[1].set_xticklabels(['-200', '-100', '0', '100', '200'], size = 18)
                ax.flatten()[1].set_xlabel('time from ON peak (ms)', size = 18)
                ax.flatten()[1].set_ylabel('Freq (Hz)', size = 18)

                if sweep_ind == 0:
                    vmax = np.max(np.abs(np.mean(np.asarray(UP_CWT_average), axis = 0 )))/1.3
                ax.flatten()[2].imshow(np.abs(np.mean(np.asarray(UP_CWT_average), axis = 0 )), aspect = 'auto', cmap = 'jet', vmax = vmax)
                ax.flatten()[2].set_yticks(np.arange(0, width_nr - 1, 10*(width_nr/freq_max)))
                ax.flatten()[2].set_yticklabels(list(map(str, np.arange(0, freq_max, 10))), size = 18)
                ax.flatten()[2].set_xlim([580,1420])
                ax.flatten()[2].set_xticks([600,800,1000,1200,1400])            
                ax.flatten()[2].set_xticklabels(['-200', '-100', '0', '100', '200'], size = 18)
                ax.flatten()[2].set_xlabel('time from ON peak (ms)', size = 18)
                ax.flatten()[2].set_ylabel('Freq (Hz)', size = 18)

            elif sweep_ind == 1:
                # plot SOs extracted in LFP pairing with CWT of average LFP and average CWT
                # fig.suptitle(f'{day}, {chan}, {delta power}')
                ax.flatten()[3].plot(np.asarray(UP_LFP_average).T/1000, linewidth = 0.1)
                ax.flatten()[3].plot(np.mean(np.asarray(UP_LFP_average).T/1000, axis = 1), color = 'k')
                ax.flatten()[3].set_ylim([-2,1])
                ax.flatten()[3].tick_params(axis = 'y', labelsize = 18)
                ax.flatten()[3].set_ylabel('LFP (mV)', size = 18)
                ax.flatten()[3].tick_params(axis = 'x', labelsize = 18)

                ax.flatten()[4].imshow(np.abs(scipy.signal.cwt(np.mean(np.asarray(UP_LFP_average).T, axis = 1), scipy.signal.morlet2, widths, dtype = 'complex128')), aspect = 'auto', cmap = 'jet', vmax = vmax)
                ax.flatten()[4].set_yticks(np.arange(0, width_nr - 1, 10*(width_nr/freq_max)))
                ax.flatten()[4].set_yticklabels(list(map(str, np.arange(0, freq_max, 10))), size = 18)
                ax.flatten()[4].set_xlim([580,1420])
                ax.flatten()[4].set_xticks([600,800,1000,1200,1400])            
                ax.flatten()[4].set_xticklabels(['-200', '-100', '0', '100', '200'], size = 18)
                ax.flatten()[4].set_xlabel('time from ON peak (ms)', size = 18)
                ax.flatten()[4].set_ylabel('Freq (Hz)', size = 18)

                ax.flatten()[5].imshow(np.abs(np.mean(np.asarray(UP_CWT_average), axis = 0 )), aspect = 'auto', cmap = 'jet', vmax = vmax)
                ax.flatten()[5].set_yticks(np.arange(0, width_nr - 1, 10*(width_nr/freq_max)))
                ax.flatten()[5].set_yticklabels(list(map(str, np.arange(0, freq_max, 10))), size = 18)
                ax.flatten()[5].set_xlim([580,1420])
                ax.flatten()[5].set_xticks([600,800,1000,1200,1400])            
                ax.flatten()[5].set_xticklabels(['-200', '-100', '0', '100', '200'], size = 18)
                ax.flatten()[5].set_xlabel('time from ON peak (ms)', size = 18)
                ax.flatten()[5].set_ylabel('Freq (Hz)', size = 18)

            # if sweep_ind == 0:
            #     vmax = np.max(np.abs(np.mean(np.asarray(UP_CWT_average), axis = 0)))/1.5
            # ax[2].imshow(np.abs(np.mean(np.asarray(UP_CWT_average), axis = 0)), aspect = 'auto', cmap = 'jet', vmax = 1000)
            # ax[2].set_yticks(np.linspace(0, width_nr - 1, int(width_nr/y_tick_scaling)))
            # ax[2].set_yticklabels(list(map(str, np.linspace(freq_min, freq_max, int(width_nr/y_tick_scaling)).astype(int))))

            plt.tight_layout()
    plt.savefig('ON state CWT before vs after.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('ON state CWT before vs after.pdf', dpi = 1000, format = 'pdf')

        # y_tick_scaling = 3
        # fig, ax = plt.subplots(3,1,figsize = (8,8), sharex = True)
        # ax[0].imshow(np.abs(CWT[:]), cmap = 'jet', aspect = 'auto')
        # ax[0].set_yticks(np.linspace(0, width_nr - 1, int(width_nr/y_tick_scaling)))
        # ax[0].set_yticklabels(list(map(str, np.linspace(freq_min, freq_max, int(width_nr/y_tick_scaling)).astype(int))))
        # # ax[1].imshow(STFT[2][STFT[0]<45,:], cmap = 'jet', aspect = 'auto')
        # # ax[0].set_yticks(np.linspace(0, width_nr - 1, int(width_nr/y_tick_scaling)))
        # # ax[0].set_yticklabels(list(map(str, np.linspace(freq_min, freq_max, int(width_nr/y_tick_scaling)).astype(int))))
        # ax[2].plot(LFP_all_sweeps[sweep][chan])
        
    
    os.chdir('..') 
    # os.chdir('..')


# # plot example traces, LFP with CWT
# y_tick_scaling = 10
# for chan in channels[[2]]:
#     for sweep_ind, sweep in enumerate([4]):
#         curr_LFP = LFP_all_sweeps[sweep][chan][30000:40000]
#         #continuous complex morlet wavelet transform
#         CWT = scipy.signal.cwt(curr_LFP, scipy.signal.morlet2, widths, dtype = 'complex128')        
#         # STFT = scipy.signal.spectrogram(LFP_all_sweeps[sweep][channels[2]], fs = 1000)
        
#         # plot example
#         fig, ax = plt.subplots(2,1, sharex = True, figsize = (10,5))
#         ax[0].plot(curr_LFP/1000, color = 'k')
#         # ax[0].set_xticks([])
#         ax[0].set_ylim([-2.5,1.5])
#         ax[0].tick_params(axis = 'y', labelsize = 18)
#         ax[0].spines['top'].set_visible(False)
#         ax[0].spines['right'].set_visible(False)
#         # ax[0].set_yticks([])
        
#         if sweep_ind == 0:
#             vmax = np.max(np.abs(CWT[:]))/1.5
#             vmin = np.min(np.abs(CWT[:]))
#         image = ax[1].imshow(np.abs(CWT[:]), cmap = 'jet', aspect = 'auto', vmax = vmax)
#         ax[1].set_yticks(np.arange(0, width_nr - 1, 10*(width_nr/freq_max)))
#         ax[1].set_yticklabels(list(map(str, np.arange(0, freq_max, 10))), size = 18)
#         ax[1].tick_params(axis = 'x', labelsize = 18)
#         # ax[1].set_xlim([30000,40000])
#         # cb = plt.colorbar(image) #, boundaries=np.linspace(-55,0,100).astype(int)
#         # cb.set_label(label='LFP response change (%)',size=15)   
#         # cb.ax.tick_params(labelsize=14)
#         plt.tight_layout()
        
# plt.savefig('CWT example.jpg', format = 'jpg', dpi = 1000)
# plt.savefig('CWT example.pdf', format = 'pdf', dpi = 1000)



