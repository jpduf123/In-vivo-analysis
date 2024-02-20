# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 00:43:10 2022

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
from sklearn.metrics import auc
from sys import getsizeof
import pandas as pd
# import line_profiler
import matplotlib.colors as colors


# os.chdir(r'D:\One_Drive\OneDrive\Dokumente\SWS\JP good recordings\DOWNpairing\new')

overall_path = r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS'

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

electrode_distance_matrix = np.empty([64, 64])
for chan1_ind, chan1 in enumerate(list(range(64))): 
    for chan2_ind, chan2 in enumerate(list(range(64))): 
        electrode_distance_in_indices = np.squeeze(np.argwhere(channelMapArray == chan1) - np.argwhere(channelMapArray == chan2))
        electrode_distance_matrix[chan1_ind, chan2_ind] = (np.sqrt(electrode_distance_in_indices[0]**2 + electrode_distance_in_indices[1]**2))*200
        # electrode_distance_matrix[np.tril(electrode_distance_matrix)]
electrode_distances = np.unique(electrode_distance_matrix)
electrode_distances = np.delete(electrode_distances, 0)     


exclude_before = 0.1
# maybe better to take 1 second after stim for slow waves as high change they get fucked up by the stim otherwise?
exclude_after = 1.9

time_steps = [1,2,5,10,20,50,100,200,500,1000,2000]
  
def cl():
    plt.close('all')



def distance_plot(Matrix, channels = list(range(64))):
    #return a list of arrays with equidistant channel values in there:
    distance_corr_list = []
    for distance_ind, distance in enumerate(list(electrode_distances)):
        # you only select channels that are in SW_spiking channels
        chan_pairs = [[np.argwhere(electrode_distance_matrix == distance)[i][0],np.argwhere(electrode_distance_matrix == distance)[i][1]] for i in range(len(np.argwhere(electrode_distance_matrix == distance)))]
        chan_pairs = [i for i in chan_pairs if (i[0] in channels and i[1] in channels)]
        
        curr_dist = [Matrix[chan_pairs[i][0], chan_pairs[i][1]] for i in range(len(chan_pairs))]
        #take out nan values (redundant channel pairs)
        distance_corr_list.append([i for i in curr_dist if ~np.isnan(i)])
        
    return distance_corr_list


def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b


#%% DO STTC
highpass_cutoff = 4
time_steps = [1,2,5,10,20,50,100,200,500,1000,2000]
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i) and 'old' not in i]
# for day in days:
for day in ['221216']:
    os.chdir(day)
    print(day)
    
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    try:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}_cleaned','rb'))
        print('found cleaned file')
    except FileNotFoundError:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    if os.path.exists('stims_for_delta'):
        stims_for_delta = pickle.load(open('stims_for_delta','rb'))
    else:
        stims_for_delta = copy.deepcopy(stim_times)
        pickle.dump(stims_for_delta, open('stims_for_delta', 'wb'))
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    # if os.path.exists('STTC_list_per_stim'):
    #     os.chdir('..')
    #     os.chdir('..')
    #     continue
    
    # do STTC for every stim
    STTC_list_per_stim = []
    
    STTC_matrix = np.empty([len(time_steps), 64, 64]) #sweep, timestep, then matrix of nchanxnchan with correlation between channels.  have all 64 channels (in 1-64 order, NOT chanMap order) matrix for all mice. makes it easier to later only take out channel pairings that I want.
    STTC_matrix[:] = np.NaN
    
    chan_names = np.asarray(list(map(int, list(spikes_allsweeps[1].keys())))) # as a list of integers to access spikes_allsweeps which is a dict with string keys... a bit retarded maybe
    
    exclude_no_spikes = False
    
    for ind_sweep, spikes in enumerate(spikes_allsweeps):
        curr_sweep = np.empty([len(stims_for_delta[ind_sweep][1:-1]), len(time_steps), 64, 64])
        curr_sweep[:] = np.NaN
        for chan1_ind, chan1 in enumerate(list(range(64))): 
            print(f'{day}, sweep {ind_sweep}, chan number {chan1_ind}')
            # print(f'sweep {ind_sweep}, chan number {chan1_ind}')
            #chan name for accessing dict
            chan_name1 = list(spikes_allsweeps[ind_sweep].keys())[np.where(chan_names == chan1)[0][0]]
            for chan2_ind, chan2 in enumerate(list(range(64))):
                # only fill one triangle of the matrix (don't do the STTC again for the same channel pair)
                if chan2_ind >= chan1_ind:
                    continue
                chan_name2 = list(spikes_allsweeps[ind_sweep].keys())[np.where(chan_names == chan2)[0][0]]
                
                spikes_1 = neo.core.SpikeTrain(list(spikes_allsweeps[ind_sweep][chan_name1]), t_stop = LFP_all_sweeps[ind_sweep].shape[1], units = 'ms')
                spikes_2 = neo.core.SpikeTrain(list(spikes_allsweeps[ind_sweep][chan_name2]), t_stop = LFP_all_sweeps[ind_sweep].shape[1], units = 'ms')
                
                for ind_time_step, time_step in enumerate(time_steps):
                    # curr_sweep = np.empty([len(stims_for_delta[ind_sweep][1:-1])])
                    # curr_sweep[:] = np.NaN
                    
                    for ind_stim, stim in enumerate(list(stims_for_delta[ind_sweep][1:-1])):
                        if stim == 0:
                            # print(f'{ind_stim}: continue')
                            continue
                            
                        curr_spikes_1 = spikes_1[np.where((stim + exclude_after*new_fs < spikes_1.times) & (spikes_1.times < stim + (5 - exclude_before)*new_fs))[0]]
                        curr_spikes_1.t_start = (stim + exclude_after*new_fs)*pq.ms
                        curr_spikes_1.t_stop = (stim + (5 - exclude_before)*new_fs)*pq.ms
                        
                        
                        curr_spikes_2 = spikes_2[np.where((stim + exclude_after*new_fs < spikes_2.times) & (spikes_2.times < stim + (5 - exclude_before)*new_fs))[0]]
                        curr_spikes_2.t_start = (stim + exclude_after*new_fs)*pq.ms
                        curr_spikes_2.t_stop = (stim + (5 - exclude_before)*new_fs)*pq.ms
    
                        if exclude_no_spikes and (curr_spikes_1.size == 0 or curr_spikes_2.size == 0):
                            # print('no spikes together')
                            continue
                        else:
                            curr_sweep[ind_stim, ind_time_step, chan1, chan2] = elephant.spike_train_correlation.spike_time_tiling_coefficient(curr_spikes_1, curr_spikes_2, dt = time_step * pq.ms)
                    
                    # now average it across stims during a sweep (one value per time step and channel pair and sweep)                                 
                    # STTC_matrix[ind_time_step, chan1, chan2] = np.nanmean(curr_sweep, axis = 0)
        STTC_list_per_stim.append(curr_sweep)
    
    pickle.dump(STTC_list_per_stim, open('STTC_list_per_stim_cleaned','wb'))

    os.chdir('..')
    os.chdir('..')



#% do STTC on the whole sweeps, with stims set at 0.
# STTC_list_per_sweep = []

# #STTC matrix for every sweep (then have sweeps as lists, so I can inspect in variable inspector. 4D numpy matrices cant be inspected in spyder...)
# # STTC_matrix = np.empty([len(time_steps), 64, 64]) #sweep, timestep, then matrix of nchanxnchan with correlation between channels.  have all 64 channels (in 1-64 order, NOT chanMap order) matrix for all mice. makes it easier to later only take out channel pairings that I want.
# # STTC_matrix[:] = np.NaN

# chan_names = np.asarray(list(map(int, list(spikes_allsweeps[0].keys())))) # as a list of integers to access spikes_allsweeps which is a dict with string keys... a bit retarded maybe

# exclude_no_spikes = False

# for ind_sweep, spikes in enumerate(spikes_allsweeps):
#     curr_sweep = np.empty([len(time_steps), 64, 64])
#     curr_sweep[:] = np.NaN
#     for chan1_ind, chan1 in enumerate(list(range(64))): 
#         print(f'sweep {ind_sweep}, chan number {chan1_ind}')
#         chan_name1 = list(spikes_allsweeps[ind_sweep].keys())[np.where(chan_names == chan1)[0][0]]
#         for chan2_ind, chan2 in enumerate(list(range(64))):
#             # only fill one triangle of the matrix (don't do the STTC again for the same channel pair)
#             if chan2_ind >= chan1_ind:
#                 continue
#             chan_name2 = list(spikes_allsweeps[ind_sweep].keys())[np.where(chan_names == chan2)[0][0]]
            
#             spikes_1 = neo.core.SpikeTrain(list(spikes_allsweeps[ind_sweep][chan_name1]), t_stop = LFP_all_sweeps[ind_sweep].shape[1], units = 'ms')
#             spikes_2 = neo.core.SpikeTrain(list(spikes_allsweeps[ind_sweep][chan_name2]), t_stop = LFP_all_sweeps[ind_sweep].shape[1], units = 'ms')
            
#             # take out stims:
#             spikes_1_nostims = spikes_1[np.where(np.searchsorted(stim_times[ind_sweep] - exclude_before, spikes_1.times) - 1 != np.searchsorted(stim_times[ind_sweep] + exclude_after, spikes_1.times))[0]]
#             spikes_2_nostims = spikes_2[np.where(np.searchsorted(stim_times[ind_sweep] - exclude_before, spikes_2.times) - 1 != np.searchsorted(stim_times[ind_sweep] + exclude_after, spikes_2.times))[0]]

                                
#             for ind_time_step, time_step in enumerate(time_steps):
#                 # curr_sweep = np.empty([len(stims_for_delta[ind_sweep][1:-1])])
#                 # curr_sweep[:] = np.NaN
                                    
#                 if exclude_no_spikes and (spikes_1.size == 0 or spikes_2.size == 0):
#                     # print('no spikes together')
#                     continue
#                 else:
#                     curr_sweep[ind_time_step, chan1, chan2] = elephant.spike_train_correlation.spike_time_tiling_coefficient(spikes_1_nostims, spikes_2_nostims, dt = time_step * pq.ms)
                
#                 # now average it across stims during a sweep (one value per time step and channel pair and sweep)                                 
#                 # STTC_matrix[ind_time_step, chan1, chan2] = np.nanmean(curr_sweep, axis = 0)
#     STTC_list_per_sweep.append(curr_sweep)

# pickle.dump(STTC_list_per_sweep, open('STTC_list_per_sweep','wb'))



#%% DO LFP CROSS CORRELATION
lowpass = 2

# for day_ind, day in enumerate([i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]):
for day_ind, day in enumerate(['221220_2']):
    os.chdir(day)
    
    # if os.path.exists('corr_peak_norm_per_stim'):
    #     os.chdir('..')
    #     continue
    
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    # spikes_allsweeps = pickle.load(open('spikes_allsweeps','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    if os.path.exists('stims_for_delta'):
        stims_for_delta = pickle.load(open('stims_for_delta','rb'))
    else:
        stims_for_delta = copy.deepcopy(stim_times)
        pickle.dump(stims_for_delta, open('stims_for_delta', 'wb'))
    
    
    # corr_list_per_stim = [] #CAVE that doenst work, this list wold take up way too much memory. It's also 5 GB per sweep so not really feasible.
    corr_peak_per_stim = []
    corr_peak_norm_per_stim = []
    corr_peak_norm_nearest_per_stim = []
    corr_lag_per_stim = []
    corr_lag_nearest_per_stim = []

    for ind_sweep, LFP in enumerate(LFP_all_sweeps):
        if LFP_all_sweeps[ind_sweep].size == 0:
            corr_peak_per_stim.append(np.array([]))
            corr_peak_norm_per_stim.append(np.array([]))
            corr_lag_per_stim.append(np.array([]))
            continue
        LFP_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = lowpass*pq.Hz).as_array()
        
        # curr_sweep = np.empty([5999, len(stims_for_delta[ind_sweep][1:-1]), 64, 64], dtype = np.float32)
        # curr_sweep[:] = np.NaN
        curr_peak = np.empty([len(stims_for_delta[ind_sweep][1:-1]), 64, 64]) # peak cross correlation value
        curr_peak[:] = np.NaN
        curr_peak_norm = np.empty([len(stims_for_delta[ind_sweep][1:-1]), 64, 64]) # peak cross correlation value
        curr_peak_norm[:] = np.NaN
        curr_peak_norm_nearest = np.empty([len(stims_for_delta[ind_sweep][1:-1]), 64, 64]) # peak cross correlation value
        curr_peak_norm_nearest[:] = np.NaN
        curr_lag = np.empty([len(stims_for_delta[ind_sweep][1:-1]), 64, 64]) # lag of peak cross correlation value
        curr_lag[:] = np.NaN
        curr_lag_nearest = np.empty([len(stims_for_delta[ind_sweep][1:-1]), 64, 64]) # lag of peak cross correlation value
        curr_lag_nearest[:] = np.NaN

        for ind_stim, stim in enumerate(list(stims_for_delta[ind_sweep][1:-1])):
            print(f'day {day} sweep {ind_sweep}, stim {ind_stim}')
            if stim == 0:
                # print(f'{ind_stim}: continue')
                continue                   
            else:
                curr_LFP_filt_total = LFP_filt[int(stim):int(stim + 5*new_fs), :]
                curr_LFP_filt = LFP_filt[int(stim + exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs), :]
    
            for chan1_ind, chan1 in enumerate(list(range(64))): 
                for chan2_ind, chan2 in enumerate(list(range(64))):
                    # only fill one triangle of the matrix (don't do the corr again for the same channel pair)
                    if chan2_ind >= chan1_ind:
                        continue                 
                    else:
                        curr_sweep = scipy.signal.correlate(curr_LFP_filt[:,chan1_ind], curr_LFP_filt[:,chan2_ind])
                        curr_sweep_norm = curr_sweep/np.sqrt(np.sum(curr_LFP_filt[:,chan1_ind]**2)*np.sum(curr_LFP_filt[:,chan2_ind]**2))
                        
                        curr_peak[ind_stim, chan1, chan2] = np.max(curr_sweep)
                        curr_peak_norm[ind_stim, chan1, chan2] = np.max(curr_sweep_norm)
                        
                        if (scipy.signal.find_peaks(curr_sweep_norm)[0]).size == 0:
                            continue
                        else:                            
                            nearest_peak_indx = scipy.signal.find_peaks(curr_sweep_norm)[0][np.argmin(np.abs(scipy.signal.find_peaks(curr_sweep_norm)[0] - 3000))]
                            curr_peak_norm_nearest[ind_stim, chan1, chan2] = curr_sweep_norm[nearest_peak_indx]
                        
                        curr_lag[ind_stim, chan1, chan2] = np.argmax(curr_sweep) - 3000
                        curr_lag_nearest[ind_stim, chan1, chan2] = nearest_peak_indx - 3000
                        
                        # curr_sweep[:,ind_stim, chan1, chan2] = scipy.signal.correlate(curr_LFP_filt[:,chan1_ind], curr_LFP_filt[:,chan2_ind])
                        # curr_peak[ind_stim, chan1, chan2] = np.max(curr_sweep[:,ind_stim, chan1, chan2])
                        # curr_lag[ind_stim, chan1, chan2] = np.argmax(curr_sweep[:,ind_stim, chan1, chan2]) - 3000
                    # now average it across stims during a sweep (one value per time step and channel pair and sweep)                                 
                    # STTC_matrix[ind_time_step, chan1, chan2] = np.nanmean(curr_sweep, axis = 0)
                    
        # pickle.dump(curr_sweep, open(f'corr_list_{ind_sweep}','wb'))
        corr_peak_per_stim.append(curr_peak)
        corr_peak_norm_per_stim.append(curr_peak_norm)
        corr_peak_norm_nearest_per_stim.append(curr_peak_norm_nearest)
        corr_lag_per_stim.append(curr_lag)
        corr_lag_nearest_per_stim.append(curr_lag_nearest)
        
        # for chan1_ind, chan1 in enumerate(list(range(64))): 
        #     for chan2_ind, chan2 in enumerate(list(range(64))):
        #         if chan2_ind >= chan1_ind:
        #             for sweep in range(len(corr_peak_per_stim)):
        #                 corr_peak_per_stim[sweep][:,chan1_ind, chan2_ind] = np.NaN                 
        #                 corr_lag_per_stim[sweep][:,chan1_ind, chan2_ind] = np.NaN                 
    
    pickle.dump(corr_peak_per_stim, open('corr_peak_per_stim','wb'))
    pickle.dump(corr_peak_norm_per_stim, open('corr_peak_norm_per_stim','wb'))
    pickle.dump(corr_peak_norm_nearest_per_stim, open('corr_peak_norm_nearest_per_stim','wb'))
    pickle.dump(corr_lag_per_stim, open('corr_lag_per_stim','wb'))
    pickle.dump(corr_lag_nearest_per_stim, open('corr_lag_nearest_per_stim','wb'))

    os.chdir('..')

# corr = pickle.load(open('corr_list_1', 'rb'))

# plt.plot(corr[:,:,35,25])

#%% plot STTC before and after one mouse

# AUC_before = np.empty([11, len(time_steps)])
# AUC_before[:] = np.NaN
# AUC_after = np.empty([11, len(time_steps)])
# AUC_after[:] = np.NaN

# params_before = np.empty([11, len(time_steps), 3])
# params_after = np.empty([11, len(time_steps), 3])


os.chdir([i for i in os.listdir() if 'analysis' in i][0])

# if os.path.exists('STTC_list_per_stim') == False:
#     os.chdir('..')
#     os.chdir('..')
#     continue
# else:
if os.path.exists('STTC_list_per_stim_avg') == False:          
    STTC_all_stims = pickle.load(open('STTC_list_per_stim','rb'))
    STTC_list = [np.nanmean(i, axis = 0) for i in STTC_all_stims]
    pickle.dump(STTC_list, open('STTC_list_per_stim_avg','wb'))

#     os.chdir('..')
#     os.chdir('..')
# os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
else:
    STTC_list = pickle.load(open('STTC_list_per_stim_avg','rb'))
    
SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',')
LFP_resp_channels = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',') 

# if SW_spiking_channels.size < 3:
#     os.chdir('..')
#     os.chdir('..')
#     continue

only_intersect_chans = False
if only_intersect_chans:    
    SW_spiking_channels = np.intersect1d(SW_spiking_channels,LFP_resp_channels)
 
# to_plot_1_STTC = np.loadtxt('to_plot_1_delta.csv', delimiter = ',', dtype = int)
to_plot_1_STTC = [0,1,2,3]
to_plot_2_STTC = [4,5,6,7,8,9]
# to_plot_2_STTC = np.loadtxt('to_plot_2_delta.csv', delimiter = ',', dtype = int)
# to_plot_2_STTC = [4,5]


def distance_plot(Matrix, channels = SW_spiking_channels, intersect_with_SW_spiking = False):
    #return a list of arrays with correlations of each distance in there:
    distance_corr_list = []
    for distance_ind, distance in enumerate(list(electrode_distances)):
        # you only select channels that are in SW_spiking channels
        chan_pairs = [[np.argwhere(electrode_distance_matrix == distance)[i][0], np.argwhere(electrode_distance_matrix == distance)[i][1]] for i in range(len(np.argwhere(electrode_distance_matrix == distance)))]
        if intersect_with_SW_spiking:
            chan_pairs = [i for i in chan_pairs if (i[0] in SW_spiking_channels and i[0] in channels and i[1] in SW_spiking_channels and i[1] in channels)]
        else:
            chan_pairs = [i for i in chan_pairs if (i[0] in channels and i[1] in channels)]
        # make sure there is a value of STTC in all sweeps   
        curr_dist = [Matrix[chan_pairs[i][0], chan_pairs[i][1]] for i in range(len(chan_pairs))]
        #take out nan values (redundant channel pairs)
        distance_corr_list.append([i for i in curr_dist if ~np.isnan(i)])
        
    return distance_corr_list


def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

# #plot exponential curve for every sweep before with data points
p0 = (0.5, 0.001, 0.01) # start with value for the curve fit near those we expect

fig, ax = plt.subplots(4,3,sharey = True, figsize = (15,10))
color = ['k', 'r']
for t_ind, t in enumerate(time_steps):
    for cond_ind, cond in enumerate(['before', 'after']):
        if cond_ind == 0:
            a = distance_plot(np.nanmean(np.array([STTC_list[i] for i in to_plot_1_STTC]), axis = 0)[t_ind,:,:], channels = LFP_resp_channels, intersect_with_SW_spiking = only_intersect_chans)
            STTC_before = a
            # medians_before[day_ind, :, t_ind] = np.asarray(list(map(np.median, a)))
        else:
            if len(to_plot_2_STTC) <2:
                a = distance_plot(STTC_list[to_plot_2_STTC][t_ind,:,:])  
            else:                        
                a = distance_plot(np.nanmean(np.array([STTC_list[i] for i in to_plot_2_STTC]), axis = 0)[t_ind,:,:], channels = LFP_resp_channels, intersect_with_SW_spiking = only_intersect_chans)   
            STTC_after = a
            # medians_after[day_ind, :, t_ind] = np.asarray(list(map(np.median, a)))            
        pickle.dump(a, open(f'distance_plot_{cond}_intersect_{only_intersect_chans}','wb'))
        ydata = np.concatenate(a)
        xdata = np.repeat(electrode_distances, [len(i) for i in a])
        params, cv = scipy.optimize.curve_fit(monoExp, xdata, ydata, p0, maxfev = 100000000)
        # print(params[0], params[1], params[2])
        
        # if cond_ind == 0:
        #     params_before[day_ind, t_ind, :] = params[:]
        #     AUC_before[day_ind, t_ind] = auc(electrode_distances, monoExp(electrode_distances, params[0], params[1], params[2]))
        # else: 
        #     params_after[day_ind, t_ind, :] = params[:]
        #     AUC_after[day_ind, t_ind] = auc(electrode_distances, monoExp(electrode_distances, params[0], params[1], params[2]))

        ax.flatten()[t_ind].plot(electrode_distances, monoExp(electrode_distances, params[0], params[1], params[2]), c = color[cond_ind])
        ax.flatten()[t_ind].scatter(xdata, ydata, c = color[cond_ind], s=1)
        ax.flatten()[t_ind].set_xlim([0,2000])
        
    # os.chdir('..')
    # os.chdir('..')


#%% plot individual STTC delta t timesteps as examples in one mouse (281021 in UP pairing group)

p0 = (0.5, 0.001, 0.01) # start with value for the curve fit near those we expect

time_steps_examples = [3,6,9]
distances_to_plot = 20
electrode_distances_to_plot = electrode_distances[:distances_to_plot] 

os.chdir([i for i in os.listdir() if 'analysis' in i][0])

SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',')
LFP_resp_channels = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',') 

def distance_plot(Matrix, channels = SW_spiking_channels, intersect_with_SW_spiking = False):
    #return a list of arrays with correlations of each distance in there:
    distance_corr_list = []
    for distance_ind, distance in enumerate(list(electrode_distances)):
        # you only select channels that are in SW_spiking channels
        chan_pairs = [[np.argwhere(electrode_distance_matrix == distance)[i][0],np.argwhere(electrode_distance_matrix == distance)[i][1]] for i in range(len(np.argwhere(electrode_distance_matrix == distance)))]
        if intersect_with_SW_spiking:
            chan_pairs = [i for i in chan_pairs if (i[0] in SW_spiking_channels and i[0] in channels and i[1] in SW_spiking_channels and i[1] in channels)]
        else:
            chan_pairs = [i for i in chan_pairs if (i[0] in channels and i[1] in channels)]
        # make sure there is a value of STTC in all sweeps   
        curr_dist = [Matrix[chan_pairs[i][0], chan_pairs[i][1]] for i in range(len(chan_pairs))]
        #take out nan values (redundant channel pairs)
        distance_corr_list.append([i for i in curr_dist if ~np.isnan(i)])
        
    return distance_corr_list


def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b


# if os.path.exists('STTC_list_per_stim') == False:
#     os.chdir('..')
#     os.chdir('..')
#     continue
# else:
if os.path.exists('STTC_list_per_stim_avg') == False:          
    STTC_all_stims = pickle.load(open('STTC_list_per_stim','rb'))
    STTC_list = [np.nanmean(i, axis = 0) for i in STTC_all_stims]
    pickle.dump(STTC_list, open('STTC_list_per_stim_avg','wb'))

#     os.chdir('..')
#     os.chdir('..')
# os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
else:
    STTC_list = pickle.load(open('STTC_list_per_stim_avg','rb'))
    

# if SW_spiking_channels.size < 3:
#     os.chdir('..')
#     os.chdir('..')
#     continue

only_intersect_chans = False
if only_intersect_chans:    
    SW_spiking_channels = np.intersect1d(SW_spiking_channels,LFP_resp_channels)
 
# to_plot_1_STTC = np.loadtxt('to_plot_1_delta.csv', delimiter = ',', dtype = int)
fig, ax = plt.subplots(figsize = (6,3))
to_plot_1_STTC = [0,1,2,3]
to_plot_2_STTC = [4,5,6,7,8,9]
color = ['k', 'y', 'g']
for t_inddd, t_ind in enumerate(time_steps_examples):
    # for cond_ind, cond in enumerate(['before', 'after']):
    for cond_ind, cond in enumerate(['before']):
        if cond_ind == 0:
            a = distance_plot(np.nanmean(np.array([STTC_list[i] for i in to_plot_1_STTC]), axis = 0)[t_ind,:,:], channels = LFP_resp_channels, intersect_with_SW_spiking = only_intersect_chans)[:distances_to_plot]
            STTC_before = a
            # medians_before[day_ind, :, t_ind] = np.asarray(list(map(np.median, a)))
        else:
            if len(to_plot_2_STTC) <2:
                a = distance_plot(STTC_list[to_plot_2_STTC][t_ind,:,:])[:distances_to_plot]  
            else:                        
                a = distance_plot(np.nanmean(np.array([STTC_list[i] for i in to_plot_2_STTC]), axis = 0)[t_ind,:,:], channels = LFP_resp_channels, intersect_with_SW_spiking = only_intersect_chans)[:distances_to_plot]   
            STTC_after = a
            # medians_after[day_ind, :, t_ind] = np.asarray(list(map(np.median, a)))            
        ydata = np.concatenate(a)
        xdata = np.repeat(electrode_distances_to_plot, [len(i) for i in a])
        params, cv = scipy.optimize.curve_fit(monoExp, xdata, ydata, p0, maxfev = 100000000)
        # print(params[0], params[1], params[2])
        
        # if cond_ind == 0:
        #     params_before[day_ind, t_ind, :] = params[:]
        #     AUC_before[day_ind, t_ind] = auc(electrode_distances, monoExp(electrode_distances, params[0], params[1], params[2]))
        # else: 
        #     params_after[day_ind, t_ind, :] = params[:]
        #     AUC_after[day_ind, t_ind] = auc(electrode_distances, monoExp(electrode_distances, params[0], params[1], params[2]))

        ax.plot(electrode_distances_to_plot, monoExp(electrode_distances_to_plot, params[0], params[1], params[2]), c = color[t_inddd])
        ax.scatter(xdata, ydata, c = color[t_inddd], s=1)
        ax.set_ylim([0,1])
        ax.set_xlabel('distance between channels (ym)', size = 16)
        ax.set_ylabel('STTC', size = 16)
        ax.set_xticks(np.linspace(200,1200,6))
        ax.set_xticklabels(['200', '400', '600', '800', '1000', '1200'], size = 16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([0,0.25,0.5,0.75,1])
        ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
        plt.tight_layout()

        # ax.set_xlim([0,2000])
        
os.chdir('..')
plt.savefig('STTC_time_steps_example.pdf', dpi = 1000, format = 'pdf')
plt.savefig('STTC_time_steps_example.jpg', dpi = 1000, format = 'jpg')

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('STTC_time_steps_example no lables.pdf', dpi = 1000, format = 'pdf')
plt.savefig('STTC_time_steps_example no labels.jpg', dpi = 1000, format = 'jpg')

# fig, ax = plt.subplots(1,2)
# ax[0].imshow(np.nanmean(medians_before, axis = 0), fit = 'jet')
# ax[1].imshow(np.nanmean(medians_after, axis = 0), cmap = 'jet')

# # medians_rel_difference = 

# np.save(f'medians_before_intersect_{only_intersect_chans}.npy', medians_before)
# np.save(f'medians_after_intersect_{only_intersect_chans}.npy', medians_after)
# np.savetxt(f'AUC_before_intersect_{only_intersect_chans}.csv', AUC_before, delimiter = ',')
# np.savetxt(f'AUC_after_intersect_{only_intersect_chans}.csv', AUC_after, delimiter = ',')

# rel_AUC_change = (AUC_after - AUC_before)/AUC_before
# np.savetxt(f'rel_AUC_change_intersect_{only_intersect_chans}.csv', rel_AUC_change, delimiter = ',')


# fig, ax = plt.subplots(4,4,sharey = True)
# for t_ind, t in enumerate(time_steps):  
#     ax.flatten()[t_ind].boxplot(distance_plot(STTC_list[0][t_ind,:,:], channels = list(range(64))), whis = 1, showfliers = False)

# fig, ax = plt.subplots(4,4,sharey = True)
# for t_ind, t in enumerate(time_steps):
#     ax.flatten()[t_ind].boxplot(distance_plot(STTC_list[4][t_ind,:,:], channels = list(range(64))), whis = 1, showfliers = False)
    
    
# fig, ax = plt.subplots(4,3,sharey = True)
# color = cm.rainbow(np.linspace(0, 1, 6))
# for t_ind, t in enumerate(time_steps):  
#     for sweep in [0,1,2,3]:
#         a = distance_plot(STTC_list[sweep][t_ind,:,:])
#         ydata = np.concatenate(a)
#         xdata = np.repeat(electrode_distances, [len(i) for i in a])
#         params, cv = scipy.optimize.curve_fit(monoExp, xdata, ydata, p0, maxfev = 100000000)
#         ax.flatten()[t_ind].plot(xdata, monoExp(xdata, params[0], params[1], params[2]), c = color[sweep], label = f'{sweep + 1}')
#         handles, labels = ax.flatten()[t_ind].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center')

# #plot exponential curve for every sweep after with data points
# fig, ax = plt.subplots(4,3,sharey = True)
# color = cm.rainbow(np.linspace(0, 1, 6))
# for t_ind, t in enumerate(time_steps):  
#     for ind_sweep, sweep in enumerate(to_plot_2_STTC):
#         a = distance_plot(STTC_list[sweep][t_ind,:,:])
#         ydata = np.concatenate(a)
#         xdata = np.repeat(electrode_distances, [len(i) for i in a])
#         params, cv = scipy.optimize.curve_fit(monoExp, xdata, ydata, p0, maxfev = 100000000)
#         ax.flatten()[t_ind].plot(xdata, monoExp(xdata, params[0], params[1], params[2]), c = color[ind_sweep], label = f'{sweep + 1}')
#         handles, labels = ax.flatten()[t_ind].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center')

# SW_spiking_channels = list(map(int, [45., 43., 55.,  7., 57., 10., 13., 12., 60., 63., 14., 62.]))
# SW_spiking_channels = SW_spiking_channels + [53,39,15]
# np.savetxt('SW_spiking_channels.csv', SW_spiking_channels, delimiter = ',')

# determine quality of the fit





#%% plot STTC before and after all mice and prepare for ANOVA (median before-after difference across equidistant channels, average across before sweeps and after sweeps)

time_steps = [1,2,5,10,20,50,100,200,500,1000,2000]
# time_steps = [2]

plot = True
intersect_channels = False # only intersect SW spiking and LFP resp channels?

prepare_for_ANOVA = False

lfp_cutoff_resp_channels = 200 
to_plot_1_LFP = [0,1,2,3]
to_plot_1_corr = [0,1,2,3]# to_plot_2_corr = np.loadtxt('to_plot_2_delta.csv', delimiter = ',', dtype = int)
to_plot_2_corr = [4,5,6,7,8,9]
to_plot_2_LFP = [4,5,6,7,8,9]

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

if prepare_for_ANOVA:
    STTC_medians_diff_rel = np.empty([len(days), len(time_steps), len(electrode_distances)])
    STTC_medians_diff_rel[:] = np.NaN
    STTC_medians_diff = np.empty([len(days), len(time_steps), len(electrode_distances)])
    STTC_medians_diff[:] = np.NaN

# AUC_before = np.empty([11, len(time_steps)])
# AUC_before[:] = np.NaN
# AUC_after = np.empty([11, len(time_steps)])
# AUC_after[:] = np.NaN

# params_before = np.empty([11, len(time_steps), 3])
# params_after = np.empty([11, len(time_steps), 3])

# for day_ind, day in enumerate(days):
for day in ['160427']:
    os.chdir(day)
    # if '221212' in day:
    #     plot = True
    # else:
    #     plot = False
    print(f'{day}, plot {plot}')
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    # if os.path.exists('STTC_list_per_stim') == False:
    #     os.chdir('..')
    #     os.chdir('..')
    #     continue
    # else:
    
    # if os.path.exists('STTC_list_per_stim_avg') == False:          
    STTC_all_stims = pickle.load(open('STTC_list_per_stim','rb'))
    STTC_list = [np.nanmean(i, axis = 0) for i in STTC_all_stims] # AVERAGE ACROSS STIMS FOR EVERY SWEEP
    pickle.dump(STTC_list, open('STTC_list_per_stim_avg','wb'))
    #     os.chdir('..')
    #     os.chdir('..')    
    # else:
    #     STTC_list = pickle.load(open('STTC_list_per_stim_avg','rb'))
        
    SW_spiking_channels = []
    # SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',')
    # if SW_spiking_channels.size < 3:
    #     os.chdir('..')
    #     os.chdir('..')
    #     continue
        
        
    # LFP_resp_channels = np.loadtxt('LFP_resp_channels.csv', delimiter = ',') 
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_resp_channels =  np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
    os.chdir('..')
    
    # maximize number of channels for correlation analysis (include good channels with slightly smaller LFP response below the cutoff too)
    # DOWN
    if '160308' in os.getcwd():
        chans_to_append = [55,10,12,14,17,19,21,23,25,53,51]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
    if '160420' in os.getcwd():
        chans_to_append = [9,59,61,15,14,62,60,58,52,50,63,0,2,4,6,8,10]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)     
    # if '160427' in os.getcwd():
        # chans_to_append = [1,5,3,13,12,2,4,56,26]
        # chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        # for chan in chans_to_append:
        #     LFP_resp_channels = np.append(LFP_resp_channels, chan)
        #     LFP_resp_channels = np.append(LFP_resp_channels, chan)     
    if '221212' in os.getcwd():
        chans_to_append = [22,1,49,0,51,2,4,6,15]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)    
        #some channels have small spiking artifacts not tied to SW activity which artificially increase STTC metric in the lower timesteps (taking these out biases against own hypothesis)
        # chans_to_delete = [25,33,35,21,39,37,35,33,21]
        # for chan in chans_to_delete:
        #     LFP_resp_channels = np.delete(LFP_resp_channels, np.where(LFP_resp_channels == chan)[0])
    if '221216' in os.getcwd():
        to_plot_1_corr = [3] # noisy background high-frequency before
        chans_to_append = [30,28]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)    
        #some channels have small spiking artifacts not tied to SW activity which artificially increase STTC metric in the lower timesteps (taking these out biases against own hypothesis)
        chans_to_delete = [9,57]
        for chan in chans_to_delete:
            LFP_resp_channels = np.delete(LFP_resp_channels, np.where(LFP_resp_channels == chan)[0])

    # UP
    if '160414' in os.getcwd():
        chans_to_append = [41,43,9,45,27,58]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)     
    if '160426' in os.getcwd():
        chans_to_append = [33,63,14,22,23,25,9,7,11,41,5,53,55,1]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)            
    if '121121' in os.getcwd():
        chans_to_append = [46,44,42,62,25]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
    if '221220_3' in os.getcwd():
        to_plot_2_corr = [4,5,6,7,8] # noisy background high-frequency afterwards
        chans_to_append = [57,8,23]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)

    print(LFP_resp_channels)

    STTC_list_copy = copy.deepcopy(STTC_list)
    STTC_copy_avg = np.ones([11,10,64]) # average value in all channels over time
    for sweep in range(10):
        STTC_list_copy[sweep][:,:,[i for i in range(64) if i not in LFP_resp_channels]] = np.NaN
        STTC_list_copy[sweep][:,[i for i in range(64) if i not in LFP_resp_channels],:] = np.NaN
        for chan in range(64):
            STTC_copy_avg[:,sweep, chan] = np.nanmean(np.hstack([STTC_list_copy[sweep][:,chan,:], STTC_list_copy[sweep][:,:,chan]]), axis = 1)
    
    if plot:
        #change over time in all channels (average STTC with all other channels)
        fig, ax = plt.subplots(8,8,sharey = True) 
        for ind, ax1 in enumerate(list(ax.flatten())):                        
            ax1.plot(STTC_copy_avg[6,:,chanMap[ind]])
            if chanMap[ind] in LFP_resp_channels:
                ax1.set_facecolor("y")
            ax1.set_title(str(chanMap[ind]), size = 6)
            ax1.axvline(3.5)


    def distance_plot(Matrix, channels = SW_spiking_channels, intersect_with_SW_spiking = False):
        #return a list of arrays with correlations of each distance in there:
        distance_corr_list = []
        for distance_ind, distance in enumerate(list(electrode_distances)):
            # you only select channels that are in SW_spiking channels
            chan_pairs = [[np.argwhere(electrode_distance_matrix == distance)[i][0],np.argwhere(electrode_distance_matrix == distance)[i][1]] for i in range(len(np.argwhere(electrode_distance_matrix == distance)))]
            if intersect_with_SW_spiking:
                chan_pairs = [i for i in chan_pairs if (i[0] in SW_spiking_channels and i[0] in channels and i[1] in SW_spiking_channels and i[1] in channels)]
            else:
                chan_pairs = [i for i in chan_pairs if (i[0] in channels and i[1] in channels)]  
                
            # make sure there is at least one STTC value in before AND after
            chan_pairs = [i for i in chan_pairs if np.logical_or(np.isnan([STTC_list[j][:,i[0],i[1]] for j in to_plot_1_corr]).all(), np.isnan([STTC_list[j][:,i[0],i[1]] for j in to_plot_2_corr]).all()) == False]   

            curr_dist = [Matrix[chan_pairs[i][0], chan_pairs[i][1]] for i in range(len(chan_pairs))]
            #take out nan values (redundant channel pairs)
            distance_corr_list.append([i for i in curr_dist if ~np.isnan(i)])
            
        return distance_corr_list
    
    
    def monoExp(x, m, t, b):
        return m * np.exp(-t * x) + b
    p0 = (0.5, 0.001, 0.01) # start with value for the curve fit near those we expect
    
    
    if plot:
        fig, ax = plt.subplots(4,3,sharey = True, figsize = (15,10))
        fig.suptitle(f'{day}')
        color = ['k', 'r']
        
    #plot exponential curve before vs after and save data
    for t_ind, t in enumerate(time_steps):
        for cond_ind, cond in enumerate(['before', 'after']):
            if cond_ind == 0:
                a = distance_plot(np.nanmean(np.array([STTC_list[i] for i in to_plot_1_corr]), axis = 0)[t_ind,:,:], channels = LFP_resp_channels, intersect_with_SW_spiking = intersect_channels)
                STTC_before = a
                # medians_before[day_ind, :, t_ind] = np.asarray(list(map(np.median, a)))
            else:
                if len(to_plot_2_corr)<2:
                    a = distance_plot(STTC_list[to_plot_2_corr[0]][t_ind,:,:], channels = LFP_resp_channels, intersect_with_SW_spiking = intersect_channels)  
                else:                        
                    a = distance_plot(np.nanmean(np.array([STTC_list[i] for i in to_plot_2_corr]), axis = 0)[t_ind,:,:], channels = LFP_resp_channels, intersect_with_SW_spiking = intersect_channels)   
                STTC_after = a
                # medians_after[day_ind, :, t_ind] = np.asarray(list(map(np.median, a)))            
            pickle.dump(a, open(f'distance_plot_{cond}','wb'))
            
            if plot:
                ydata = np.concatenate(a)
                xdata = np.repeat(electrode_distances, [len(i) for i in a])
                params, cv = scipy.optimize.curve_fit(monoExp, xdata, ydata, p0, maxfev = 100000000)
                # print(params[0], params[1], params[2])
                
                # if cond_ind == 0:
                #     params_before[day_ind, t_ind, :] = params[:]
                #     AUC_before[day_ind, t_ind] = auc(electrode_distances, monoExp(electrode_distances, params[0], params[1], params[2]))
                # else: 
                #     params_after[day_ind, t_ind, :] = params[:]
                #     AUC_after[day_ind, t_ind] = auc(electrode_distances, monoExp(electrode_distances, params[0], params[1], params[2]))
        
                ax.flatten()[t_ind].plot(electrode_distances, monoExp(electrode_distances, params[0], params[1], params[2]), c = color[cond_ind])
                ax.flatten()[t_ind].scatter(xdata, ydata, c = color[cond_ind], s=1)
                ax.flatten()[t_ind].set_xlim([0,2000])
            # ax.flatten()[t_ind].set_ylim([-0.1,1.1])
            
        STTC_channel_difference_rel = [(np.asarray(STTC_after[i]) - np.asarray(STTC_before[i]))/np.asarray(STTC_before[i]) for i in range(len(STTC_before))]
        STTC_channel_difference = [(np.asarray(STTC_after[i]) - np.asarray(STTC_before[i])) for i in range(len(STTC_before))]
        
        if prepare_for_ANOVA: # median across 
            STTC_medians_diff_rel[day_ind,t_ind,:] = np.asarray(list(map(np.nanmedian, STTC_channel_difference_rel)))
            STTC_medians_diff[day_ind,t_ind,:] = np.asarray(list(map(np.nanmedian, STTC_channel_difference)))
        

    pickle.dump(STTC_channel_difference_rel, open('STTC_channel_difference_rel','wb'))
    pickle.dump(STTC_channel_difference, open('STTC_channel_difference','wb'))
    
    if plot:
        plt.savefig('STTC before vs after', dpi = 1000)
         
    os.chdir('..')
    # cl()
    # os.chdir('..')


if prepare_for_ANOVA:
    pickle.dump(STTC_medians_diff_rel, open(f'STTC_medians_diff_rel_overlap_{intersect_channels}', 'wb'))
    pickle.dump(STTC_medians_diff, open(f'STTC_medians_diff_overlap_{intersect_channels}', 'wb'))
    
    STTC_medians_diff_rel_ANOVA = np.reshape(STTC_medians_diff_rel, (STTC_medians_diff_rel.shape[0], STTC_medians_diff_rel.shape[1]*STTC_medians_diff_rel.shape[2]))
    STTC_medians_diff_ANOVA = np.reshape(STTC_medians_diff, (STTC_medians_diff.shape[0], STTC_medians_diff.shape[1]*STTC_medians_diff.shape[2]))
    pickle.dump(STTC_medians_diff_rel_ANOVA, open(f'STTC_medians_diff_rel_ANOVA_overlap_{intersect_channels}', 'wb'))
    pickle.dump(STTC_medians_diff_ANOVA, open(f'STTC_medians_diff_ANOVA_overlap_{intersect_channels}', 'wb'))
    
    
    # save in SPSS ANOVA format for three-way ANOVA, with only complete mice (mice x timestep(distance))
    for_anova = 13 #how many electrode distances for anova
    for_anova_indices = list(np.hstack((np.repeat(True, for_anova), np.repeat(False, len(electrode_distances) - for_anova))))*len(time_steps)
    
    #relative difference
    a = pickle.load(open(f'STTC_medians_diff_rel_ANOVA_overlap_{intersect_channels}', 'rb'))
    a = a[:,for_anova_indices]
    # take out mice with incomplete distance sets
    for day in range(sum(np.any(np.isnan(a),axis = 1))):
        print(f'incomplete mouse: {days[np.argwhere(np.any(np.isnan(a),axis = 1) == True)[day][0]]}')
    a = a[~np.any(np.isnan(a),axis = 1),:]
    np.savetxt(f'STTC_medians_diff_rel_ANOVA_overlap_{intersect_channels}.csv', a, delimiter = ',')
    
    #absolute difference
    a = pickle.load(open(f'STTC_medians_diff_ANOVA_overlap_{intersect_channels}', 'rb'))
    a = a[:,for_anova_indices]
    # take out mice with incomplete distance sets
    a = a[~np.any(np.isnan(a),axis = 1),:]
    np.savetxt(f'STTC_medians_diff_ANOVA_overlap_{intersect_channels}.csv', a, delimiter = ',')











#%% plot LFP cross correlation for every experiment with curve fitting

to_plot_1_LFP = [0,1,2,3]
to_plot_2_LFP = [4,5,6,7,8,9]
lfp_cutoff_resp_channels = 200

p0 = (0.5, 0.001, 0.01) # start with value for the curve fit near those we expect
color = ['k', '#00CCFF']

nearest = True

# for day_ind, day in enumerate([i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]):
for day_ind, day in enumerate(['281021']):

    os.chdir(day)
        
    corr_peak_per_stim = pickle.load(open('corr_peak_per_stim', 'rb'))
    if nearest == False:
        corr_peak_norm_per_stim = pickle.load(open('corr_peak_norm_per_stim', 'rb'))
        corr_lag_per_stim = pickle.load(open('corr_lag_per_stim', 'rb'))
    else:
        corr_peak_norm_per_stim = pickle.load(open('corr_peak_norm_nearest_per_stim', 'rb'))
        corr_lag_per_stim = pickle.load(open('corr_lag_nearest_per_stim', 'rb'))

    # corr_list_1 = pickle.load(open('corr_list_1', 'rb'))
    
    # fig, ax = plt.subplots(8,8,sharey = True)
    # for chan in range(64):
    #     ax.flatten()[chan].plot(corr_list_1[:,:,61,chan])
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    # SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',')
    # LFP_resp_channels = np.loadtxt('LFP_resp_channels.csv', delimiter = ',') 
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_resp_channels =  np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
    
    # maximize number of channels for correlation analysis (include good channels with slightly smaller LFP response below the cutoff too)
    # DOWN
    if '160308' in os.getcwd():
        chans_to_append = [55,10,12,14,17,19,21,23,25,53,51]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
    if '160420' in os.getcwd():
        chans_to_append = [9,59,61,15,14,62,60,58,52,50,63,0,2,4,6,8,10]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)     
    if '160427' in os.getcwd():
        chans_to_append = [1,5,3,13,12,2,4,56,26]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)     
    if '221212' in os.getcwd():
        chans_to_append = [22,1,49,0,51,2,4,6,15]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)    
        #some channels have small spiking artifacts not tied to SW activity which artificially increase STTC metric in the lower timesteps (taking these out biases against own hypothesis)
        # chans_to_delete = [25,33,35,21,39,37,35,33,21]
        # for chan in chans_to_delete:
        #     LFP_resp_channels = np.delete(LFP_resp_channels, np.where(LFP_resp_channels == chan)[0])
    if '221216' in os.getcwd():
        to_plot_1_corr = [3]
        chans_to_append = [30,28]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)    
        #some channels have small spiking artifacts not tied to SW activity which artificially increase STTC metric in the lower timesteps (taking these out biases against own hypothesis)
        chans_to_delete = [9,57]
        for chan in chans_to_delete:
            LFP_resp_channels = np.delete(LFP_resp_channels, np.where(LFP_resp_channels == chan)[0])
    # if '221219_1' in os.getcwd():
    #     chans_to_append = [30,16,332,17,33,15,63]
    #     chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
    #     for chan in chans_to_append:
    #         LFP_resp_channels = np.append(LFP_resp_channels, chan)    

    # UP
    if '160414' in os.getcwd():
        chans_to_append = [41,43,9,45,27,58]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)     
    if '160426' in os.getcwd():
        chans_to_append = [33,63,14,22,23,25,9,7,11,41,5,53,55,1]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)            
    if '121121' in os.getcwd():
        chans_to_append = [46,44,42,62,25]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
    if '221220_3' in os.getcwd():
        to_plot_2_corr = [4,5,6,7,8]
        chans_to_append = [57,8,23]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
    # only_intersect_chans = True    
    # if only_intersect_chans:
    #     SW_spiking_channels = np.intersect1d(SW_spiking_channels,LFP_resp_channels)
    
        
    
    # sweeps_to_plot = [0,1,2,3]
    # sweeps_to_plot = [4,5,6,7,8,9]
    # sweeps_to_plot = [4]
    
    # #EVERY CHANNEL
    # measure_to_plot = corr_peak_per_stim
    # fig, ax = plt.subplots()
    # fig.suptitle('peak')
    # ax.boxplot(distance_plot(np.median(np.concatenate([measure_to_plot[i] for i in sweeps_to_plot], axis = 0), axis = 0), channels = range(64)), whis = 1, showfliers = False, positions = (electrode_distances/50).astype(np.int16))
    
    # measure_to_plot = list(map(np.abs, corr_lag_per_stim))
    # fig, ax = plt.subplots()
    # fig.suptitle('lag')
    # ax.boxplot(distance_plot(np.median(np.concatenate([measure_to_plot[i] for i in sweeps_to_plot], axis = 0), axis = 0), channels = range(64)), whis = 1, showfliers = False, positions = (electrode_distances/50).astype(np.int16))
    
    #ONLY LFP channels
    # measure_to_plot = corr_peak_per_stim
    # fig, ax = plt.subplots()
    # fig.suptitle('peak')
    # ax.boxplot(distance_plot(np.median(np.concatenate([measure_to_plot[i] for i in sweeps_to_plot], axis = 0), axis = 0), channels = LFP_resp_channels), whis = 1, showfliers = False, positions = (electrode_distances/50).astype(np.int16))
    
    # measure_to_plot = list(map(np.abs, corr_lag_per_stim))
    # fig, ax = plt.subplots()
    # fig.suptitle('lag')
    # ax.boxplot(distance_plot(np.median(np.concatenate([measure_to_plot[i] for i in sweeps_to_plot], axis = 0), axis = 0), channels = LFP_resp_channels), whis = 1, showfliers = False, positions = (electrode_distances/50).astype(np.int16))
    
    
    os.chdir('..')
    
    
    distances_to_plot = 33
    #before vs after
    fig1, ax1 = plt.subplots(figsize = (6,3))
    # fig1.suptitle('peak')
    
    fig2, ax2 = plt.subplots(figsize = (6,3))
    # fig2.suptitle('lag')
    
    to_plot_1_corr = [0,1,2,3]
    to_plot_2_corr = [4,5,6,7,8,9]
        
    channels_to_plot = LFP_resp_channels
    
    corr_lag_per_stim_abs = list(map(np.abs, corr_lag_per_stim))
    
    for cond_ind, cond in enumerate(['before', 'after']):
        if cond_ind == 0:
            peak = distance_plot(np.nanmedian(np.concatenate([corr_peak_norm_per_stim[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)
            lag = distance_plot(np.nanmedian(np.concatenate([corr_lag_per_stim_abs[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)
            # medians_before[day_ind, :] = np.asarray(list(map(np.median, a)))
        else:
            if len(to_plot_2_corr) == 1:
                peak = distance_plot(np.nanmedian(corr_peak_norm_per_stim[to_plot_2_corr], axis = 0), channels = channels_to_plot)
                lag = distance_plot(np.nanmedian(corr_lag_per_stim_abs[to_plot_2_corr], axis = 0), channels = channels_to_plot)               
            else:
                peak = distance_plot(np.nanmedian(np.concatenate([corr_peak_norm_per_stim[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)
                lag = distance_plot(np.nanmedian(np.concatenate([corr_lag_per_stim_abs[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)
            # medians_after[day_ind, :] = np.asarray(list(map(np.median, a)))            
        
        ydata_peak = np.concatenate(peak[:distances_to_plot])
        xdata_peak = np.repeat(electrode_distances[:distances_to_plot], [len(i) for i in peak[:distances_to_plot]])
        params_peak, cv_peak = scipy.optimize.curve_fit(monoExp, xdata_peak, ydata_peak, p0, maxfev = 100000000)
        
        ydata_lag = np.concatenate(lag[:distances_to_plot])
        xdata_lag = np.repeat(electrode_distances[:distances_to_plot], [len(i) for i in lag[:distances_to_plot]])
        params_lag, cv_lag = scipy.optimize.curve_fit(monoExp, xdata_lag, ydata_lag, p0, maxfev = 100000000)
    
        # print(params[0], params[1], params[2])
        
        # if cond_ind == 0:
        #     params_before[day_ind, t_ind, :] = params[:]
        #     AUC_before[day_ind, t_ind] = auc(electrode_distances, monoExp(electrode_distances, params[0], params[1], params[2]))
        # else: 
        #     params_after[day_ind, t_ind, :] = params[:]
        #     AUC_after[day_ind, t_ind] = auc(electrode_distances, monoExp(electrode_distances, params[0], params[1], params[2]))
    
        ax1.plot(electrode_distances, monoExp(electrode_distances, params_peak[0], params_peak[1], params_peak[2]), c = color[cond_ind], linewidth = 1.5)
        ax1.scatter(xdata_peak, ydata_peak, c = color[cond_ind], s=2, marker = 'o')
        ax1.set_xlim([100,1400])
        ax1.set_ylim([0.15,1])
        ax1.set_yticks([0.2,0.4,0.6,0.8,1])
    
        
        ax2.plot(electrode_distances, monoExp(electrode_distances, params_lag[0], params_lag[1], params_lag[2]), c = color[cond_ind], linewidth = 1.5)
        ax2.scatter(xdata_lag, ydata_lag, c = color[cond_ind], s=2, marker = 'o')
        ax2.set_xlim([100,1400])
        ax2.set_ylim([0,210])
        # ax2.set_title(day)
    
    # fig1.savefig('corr peak.jpg', dpi = 1000, format = 'jpg')
    # fig1.savefig('corr peak.pdf', dpi = 1000, format = 'pdf')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # # ax1.set_title(day)
    # fig1.savefig('corr peak for thesis.jpg', dpi = 1000, format = 'jpg')

    # fig2.savefig('corr lag.jpg', dpi = 1000, format = 'jpg')
    # fig2.savefig('corr lag.pdf', dpi = 1000, format = 'pdf')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_yticks([0,50,100,150,200])
    fig2.savefig('corr lag for thesis.jpg', dpi = 1000, format = 'jpg')

    os.chdir('..')
    # cl()
    
    
#%% plot LFP cross correlation for every experiment as boxplot

fig1, ax1 = plt.subplots(4,3)
fig1.suptitle('peak LFP resp')

fig2, ax2 = plt.subplots(4,3)
fig2.suptitle('peak all')


fig3, ax3 = plt.subplots(4,3)
fig3.suptitle('lag LFP resp')

fig4, ax4 = plt.subplots(4,3)
fig4.suptitle('lag all')

sweeps_to_plot = [0,1,2,3]
# sweeps_to_plot = [4,5,6,7,8,9]


for day_ind, day in enumerate([i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]):
    os.chdir(day)
    
    corr_peak_per_stim = pickle.load(open('corr_peak_per_stim', 'rb'))
    corr_peak_norm_per_stim = pickle.load(open('corr_peak_norm_per_stim', 'rb'))
    corr_lag_per_stim = pickle.load(open('corr_lag_per_stim', 'rb'))

    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
     
    SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',')
    LFP_resp_channels = np.loadtxt('LFP_resp_channels.csv', delimiter = ',') 
    
    measure_to_plot = corr_peak_norm_per_stim
    ax1.flatten()[day_ind].boxplot(distance_plot(np.median(np.concatenate([measure_to_plot[i] for i in sweeps_to_plot], axis = 0), axis = 0), channels = LFP_resp_channels), whis = 1, showfliers = False, positions = (electrode_distances/50).astype(np.int16))
    ax1.flatten()[day_ind].set_title(day)
    
    ax2.flatten()[day_ind].boxplot(distance_plot(np.median(np.concatenate([measure_to_plot[i] for i in sweeps_to_plot], axis = 0), axis = 0), channels = list(range(64))), whis = 1, showfliers = False, positions = (electrode_distances/50).astype(np.int16))
    ax2.flatten()[day_ind].set_title(day)
   
    measure_to_plot = list(map(np.abs, corr_lag_per_stim))
    ax3.flatten()[day_ind].boxplot(distance_plot(np.median(np.concatenate([measure_to_plot[i] for i in sweeps_to_plot], axis = 0), axis = 0), channels = LFP_resp_channels), whis = 1, showfliers = False, positions = (electrode_distances/50).astype(np.int16))
    ax3.flatten()[day_ind].set_title(day)
    
    ax4.flatten()[day_ind].boxplot(distance_plot(np.median(np.concatenate([measure_to_plot[i] for i in sweeps_to_plot], axis = 0), axis = 0), channels = list(range(64))), whis = 1, showfliers = False, positions = (electrode_distances/50).astype(np.int16))
    ax4.flatten()[day_ind].set_title(day)
    
    os.chdir('..')
    os.chdir('..')








#%% PREPARE LFP CROSS CORR ANOVA: CALCULATE MEDIAN/MEAN BEFORE AND AFTER FOR EVERY CHANNEL PAIR, THEN TAKE MEDIAN/MEAN OVER EQUIDISTANT CHANNELS AND SAVE FOR ANOVA AND PLOTTING

lfp_cutoff_resp_channels = 200 # maximize number of channels. #CAVE one of the DOWN pairings has huge outlier channel pair and it's the only one at that distance... need to exclude mouse for this analysis. Sad because all else is good in that mouse

to_plot_1_LFP = [0,1,2,3]
to_plot_2_LFP = [4,5,6,7,8,9]


#TAKE OUT UNREALISTIALLY LONG LAGS?
take_out_long_lags = False
lag_limit = 500

# median or mean of interstim intervals:
take_median = False


for group in ['UP_pairing', 'DOWN_pairing']:
    os.chdir(group)
    days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

    # arrays with before/after difference for every channel distance in every mouse
    channel_peak_difference_median = np.empty([len(days), 33])
    channel_lag_difference_median = np.empty([len(days), 33])
    channel_peak_difference_median[:] = np.NaN
    channel_lag_difference_median[:] = np.NaN
    
    channel_peak_difference_median_rel = np.empty([len(days), 33])
    channel_lag_difference_median_rel = np.empty([len(days), 33])
    channel_peak_difference_median_rel[:] = np.NaN
    channel_lag_difference_median_rel[:] = np.NaN
    
    channel_peak_nearest_difference_median = np.empty([len(days), 33])
    channel_lag_nearest_difference_median = np.empty([len(days), 33])
    channel_peak_nearest_difference_median[:] = np.NaN
    channel_lag_nearest_difference_median[:] = np.NaN
    
    channel_peak_nearest_difference_median_rel = np.empty([len(days), 33])
    channel_lag_nearest_difference_median_rel = np.empty([len(days), 33])
    channel_peak_nearest_difference_median_rel[:] = np.NaN
    channel_lag_nearest_difference_median_rel[:] = np.NaN
    
    channel_peak_difference_mean = np.empty([len(days), 33])
    channel_lag_difference_mean = np.empty([len(days), 33])
    channel_peak_difference_mean[:] = np.NaN
    channel_lag_difference_mean[:] = np.NaN
    
    channel_peak_difference_mean_rel = np.empty([len(days), 33])
    channel_lag_difference_mean_rel = np.empty([len(days), 33])
    channel_peak_difference_mean_rel[:] = np.NaN
    channel_lag_difference_mean_rel[:] = np.NaN
    
    channel_peak_nearest_difference_mean = np.empty([len(days), 33])
    channel_lag_nearest_difference_mean = np.empty([len(days), 33])
    channel_peak_nearest_difference_mean[:] = np.NaN
    channel_lag_nearest_difference_mean[:] = np.NaN
    
    channel_peak_nearest_difference_mean_rel = np.empty([len(days), 33])
    channel_lag_nearest_difference_mean_rel = np.empty([len(days), 33])
    channel_peak_nearest_difference_mean_rel[:] = np.NaN
    channel_lag_nearest_difference_mean_rel[:] = np.NaN
    
    channel_lag_nonabs_nearest_difference_median = np.empty([len(days), 33])
    channel_lag_nonabs_nearest_std_difference_median = np.empty([len(days), 33])
    channel_lag_nonabs_nearest_std_difference_median_rel = np.empty([len(days), 33])
    channel_lag_nonabs_nearest_difference_median[:] = np.NaN
    channel_lag_nonabs_nearest_std_difference_median[:] = np.NaN
    channel_lag_nonabs_nearest_std_difference_median_rel[:] = np.NaN
    
    channel_lag_cov_nearest_difference_mean = np.empty([len(days), 33])
    channel_lag_cov_nearest_difference_mean[:] = np.NaN
    channel_lag_cov_nearest_difference_mean_rel = np.empty([len(days), 33])
    channel_lag_cov_nearest_difference_mean_rel[:] = np.NaN
    
    channel_lag_cov_nearest_difference_median = np.empty([len(days), 33])
    channel_lag_cov_nearest_difference_median[:] = np.NaN
    channel_lag_cov_nearest_difference_median_rel = np.empty([len(days), 33])
    channel_lag_cov_nearest_difference_median_rel[:] = np.NaN
    
    
    to_plot_1_corr = [0,1,2,3]
    to_plot_2_corr = [4,5,6,7,8,9]
    
    color = ['k', 'r']
    
    def monoExp(x, m, t, b):
        return m * np.exp(-t * x) + b
    
    p0 = (0.5, 0.001, 0.01) # start with value for the curve fit near those we expect
    
    for day_ind, day in enumerate(days):
        os.chdir(day)
            
        corr_peak_per_stim = pickle.load(open('corr_peak_per_stim', 'rb'))
        corr_peak_norm_per_stim = pickle.load(open('corr_peak_norm_per_stim', 'rb'))
        corr_peak_norm_nearest_per_stim = pickle.load(open('corr_peak_norm_nearest_per_stim', 'rb'))
    
        corr_lag_per_stim = pickle.load(open('corr_lag_per_stim', 'rb'))
        corr_lag_per_stim_abs = list(map(np.abs, corr_lag_per_stim))
    
        corr_lag_nearest_per_stim = pickle.load(open('corr_lag_nearest_per_stim', 'rb'))
        corr_lag_nearest_per_stim_abs = list(map(np.abs, corr_lag_nearest_per_stim))
    
    
        os.chdir([i for i in os.listdir() if 'analysis' in i][0])
        # SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',')
        SW_spiking_channels = []
        # LFP_resp_channels = np.loadtxt('LFP_resp_channels.csv', delimiter = ',') 
        LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
        LFP_resp_channels =  np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
        
        # maximize number of channels for correlation analysis (include good channels with slightly smaller LFP response below the cutoff too)
        # DOWN
        if '160308' in os.getcwd():
            chans_to_append = [55,10,12,14,17,19,21,23,25,53,51]
            chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
            for chan in chans_to_append:
                LFP_resp_channels = np.append(LFP_resp_channels, chan)
        if '160420' in os.getcwd():
            chans_to_append = [9,59,61,15,14,62,60,58,52,50,63,0,2,4,6,8,10]
            chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
            for chan in chans_to_append:
                LFP_resp_channels = np.append(LFP_resp_channels, chan)
                LFP_resp_channels = np.append(LFP_resp_channels, chan)     
        if '160427' in os.getcwd():
            chans_to_append = [1,5,3,13,12,2,4,56,26]
            chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
            for chan in chans_to_append:
                LFP_resp_channels = np.append(LFP_resp_channels, chan)
                LFP_resp_channels = np.append(LFP_resp_channels, chan)     
        if '221212' in os.getcwd():
            chans_to_append = [22,1,49,0,51,2,4,6,15]
            chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
            for chan in chans_to_append:
                LFP_resp_channels = np.append(LFP_resp_channels, chan)    
            #some channels have small spiking artifacts not tied to SW activity which artificially increase STTC metric in the lower timesteps (taking these out biases against own hypothesis)
            # chans_to_delete = [25,33,35,21,39,37,35,33,21]
            # for chan in chans_to_delete:
            #     LFP_resp_channels = np.delete(LFP_resp_channels, np.where(LFP_resp_channels == chan)[0])
        if '221216' in os.getcwd():
            to_plot_1_corr = [3]
            chans_to_append = [30,28]
            chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
            for chan in chans_to_append:
                LFP_resp_channels = np.append(LFP_resp_channels, chan)    
            #some channels have small spiking artifacts not tied to SW activity which artificially increase STTC metric in the lower timesteps (taking these out biases against own hypothesis)
            chans_to_delete = [9,57]
            for chan in chans_to_delete:
                LFP_resp_channels = np.delete(LFP_resp_channels, np.where(LFP_resp_channels == chan)[0])
    
        # UP
        if '160414' in os.getcwd():
            chans_to_append = [41,43,9,45,27,58]
            chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
            for chan in chans_to_append:
                LFP_resp_channels = np.append(LFP_resp_channels, chan)
                LFP_resp_channels = np.append(LFP_resp_channels, chan)     
        if '160426' in os.getcwd():
            chans_to_append = [33,63,14,22,23,25,9,7,11,41,5,53,55,1]
            chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
            for chan in chans_to_append:
                LFP_resp_channels = np.append(LFP_resp_channels, chan)
                LFP_resp_channels = np.append(LFP_resp_channels, chan)            
        if '121121' in os.getcwd():
            chans_to_append = [46,44,42,62,25]
            chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
            for chan in chans_to_append:
                LFP_resp_channels = np.append(LFP_resp_channels, chan)
        if '221220_3' in os.getcwd():
            to_plot_2_corr = [4,5,6,7,8]
            chans_to_append = [57,8,23]
            chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
            for chan in chans_to_append:
                LFP_resp_channels = np.append(LFP_resp_channels, chan)
                
    
        # LFP_resp_channels = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',') 
        channels_to_plot = LFP_resp_channels
        
        #-------------------------------------------------------------------------- median/mean peak and lag values across interstim intervals efore and after for every channel pair
        for cond_ind, cond in enumerate(['before', 'after']):
            if cond_ind == 0: # before pairing
                lag_before_nearest_std = distance_plot(np.nanstd(np.concatenate([corr_lag_nearest_per_stim[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)
                cov_before_nearest = distance_plot(np.nanstd(np.concatenate([corr_lag_nearest_per_stim_abs[i] for i in to_plot_1_corr], axis = 0), axis = 0)/np.nanmean(np.concatenate([corr_lag_nearest_per_stim_abs[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)
    
                if take_median: # median across interstim intervals
                    peak_before = distance_plot(np.nanmedian(np.concatenate([corr_peak_norm_per_stim[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    lag_before = distance_plot(np.nanmedian(np.concatenate([corr_lag_per_stim_abs[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    
                    peak_before_nearest = distance_plot(np.nanmedian(np.concatenate([corr_peak_norm_nearest_per_stim[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    lag_before_nearest = distance_plot(np.nanmedian(np.concatenate([corr_lag_nearest_per_stim_abs[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)
                     
                    # take median and std of NOT ABSOLUTE value
                    lag_before_nearest_nonabs = distance_plot(np.nanmedian(np.concatenate([corr_lag_nearest_per_stim[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)  
                
                else: #mean across interstim intervals
                    peak_before = distance_plot(np.nanmean(np.concatenate([corr_peak_norm_per_stim[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    lag_before = distance_plot(np.nanmean(np.concatenate([corr_lag_per_stim_abs[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    
                    peak_before_nearest = distance_plot(np.nanmean(np.concatenate([corr_peak_norm_nearest_per_stim[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    lag_before_nearest = distance_plot(np.nanmean(np.concatenate([corr_lag_nearest_per_stim_abs[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)
        
                    # take mean and std of NOT ABSOLUTE value
                    lag_before_nearest_nonabs = distance_plot(np.nanmean(np.concatenate([corr_lag_nearest_per_stim[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)                
                    
            else: # after pairing
                lag_after_nearest_std = distance_plot(np.nanstd(np.concatenate([corr_lag_nearest_per_stim[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)
                cov_after_nearest = distance_plot(np.nanstd(np.concatenate([corr_lag_nearest_per_stim_abs[i] for i in to_plot_2_corr], axis = 0), axis = 0)/np.nanmean(np.concatenate([corr_lag_nearest_per_stim_abs[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)
    
                if take_median: # median across interstim intervals
                    peak_after = distance_plot(np.nanmedian(np.concatenate([corr_peak_norm_per_stim[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    lag_after = distance_plot(np.nanmedian(np.concatenate([corr_lag_per_stim_abs[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    
                    peak_after_nearest = distance_plot(np.nanmedian(np.concatenate([corr_peak_norm_nearest_per_stim[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    lag_after_nearest = distance_plot(np.nanmedian(np.concatenate([corr_lag_nearest_per_stim_abs[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    
                    lag_after_nearest_nonabs = distance_plot(np.nanmedian(np.concatenate([corr_lag_nearest_per_stim[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)                
    
                else: # mean across interstim intervals
                    peak_after = distance_plot(np.nanmean(np.concatenate([corr_peak_norm_per_stim[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    lag_after = distance_plot(np.nanmean(np.concatenate([corr_lag_per_stim_abs[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    
                    peak_after_nearest = distance_plot(np.nanmean(np.concatenate([corr_peak_norm_nearest_per_stim[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    lag_after_nearest = distance_plot(np.nanmean(np.concatenate([corr_lag_nearest_per_stim_abs[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)
                    
                    lag_after_nearest_nonabs = distance_plot(np.nanmean(np.concatenate([corr_lag_nearest_per_stim[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)                
    
    
    
        
        # ---------------------------------------------------------------------- difference before after for every channel pair, relative to baseline and absolute
        if take_out_long_lags == False:           
            channel_peak_difference_rel = [(np.asarray(peak_after[i]) - np.asarray(peak_before[i]))/np.asarray(peak_before[i]) for i in range(len(peak_after))]
            channel_lag_difference_rel = [(np.asarray(lag_after[i]) - np.asarray(lag_before[i]))/np.asarray(lag_before[i]) for i in range(len(peak_after))]
            channel_peak_difference = [(np.asarray(peak_after[i]) - np.asarray(peak_before[i])) for i in range(len(peak_after))]
            channel_lag_difference = [(np.asarray(lag_after[i]) - np.asarray(lag_before[i])) for i in range(len(peak_after))]
            
            channel_peak_nearest_difference_rel = [(np.asarray(peak_after_nearest[i]) - np.asarray(peak_before_nearest[i]))/np.asarray(peak_before_nearest[i]) for i in range(len(peak_after_nearest))]
            channel_lag_nearest_difference_rel = [(np.asarray(lag_after_nearest[i]) - np.asarray(lag_before_nearest[i]))/np.asarray(lag_before_nearest[i]) for i in range(len(peak_after_nearest))]
            channel_peak_nearest_difference = [(np.asarray(peak_after_nearest[i]) - np.asarray(peak_before_nearest[i])) for i in range(len(peak_after_nearest))]
            channel_lag_nearest_difference = [(np.asarray(lag_after_nearest[i]) - np.asarray(lag_before_nearest[i])) for i in range(len(peak_after_nearest))]
    
            channel_lag_nonabs_nearest_difference = [(np.asarray(lag_after_nearest_nonabs[i]) - np.asarray(lag_before_nearest_nonabs[i])) for i in range(len(peak_after_nearest))]
            channel_lag_nonabs_nearest_std_difference = [(np.asarray(lag_after_nearest_std[i]) - np.asarray(lag_before_nearest_std[i])) for i in range(len(peak_after_nearest))]
            channel_lag_nonabs_nearest_std_difference_rel = [(np.asarray(lag_after_nearest_std[i]) - np.asarray(lag_before_nearest_std[i]))/np.asarray(lag_before_nearest_std[i]) for i in range(len(peak_after_nearest))]
    
            channel_lag_cov_nearest_difference = [(np.asarray(cov_after_nearest[i]) - np.asarray(cov_before_nearest[i])) for i in range(len(cov_after_nearest))]
            channel_lag_cov_nearest_difference_rel = [(np.asarray(cov_after_nearest[i]) - np.asarray(cov_before_nearest[i]))/np.asarray(cov_before_nearest[i]) for i in range(len(cov_after_nearest))]
    
        else:
            masks = [(np.asarray(lag_after[i]) < lag_limit) & (np.asarray(lag_before[i]) < lag_limit) for i in range(len(peak_after))]
            channel_lag_difference_rel = [(np.asarray(lag_after[i])[masks[i]] - np.asarray(lag_before[i])[masks[i]]) / np.asarray(lag_before[i])[masks[i]] for i in range(len(peak_after))]
            channel_lag_difference = [(np.asarray(lag_after[i])[masks[i]] - np.asarray(lag_before[i])[masks[i]]) for i in range(len(peak_after))]
    
            masks = [(np.asarray(lag_after_nearest[i]) < lag_limit) & (np.asarray(lag_before_nearest[i]) < lag_limit) for i in range(len(peak_after))]        
            channel_lag_nearest_difference_rel = [(np.asarray(lag_after_nearest[i])[masks[i]] - np.asarray(lag_before_nearest[i])[masks[i]])/np.asarray(lag_before_nearest[i])[masks[i]] for i in range(len(peak_after_nearest))]
            channel_lag_nearest_difference = [(np.asarray(lag_after_nearest[i])[masks[i]] - np.asarray(lag_before_nearest[i])[masks[i]]) for i in range(len(peak_after_nearest))]
    
            channel_lag_cov_nearest_difference = [(np.asarray(cov_after_nearest[i])[masks[i]] - np.asarray(cov_before_nearest[i]))[masks[i]] for i in range(len(cov_after_nearest))]
            channel_lag_cov_nearest_difference_rel = [(np.asarray(cov_after_nearest[i])[masks[i]] - np.asarray(cov_before_nearest[i]))[masks[i]]/np.asarray(cov_before_nearest[i])[masks[i]]  for i in range(len(cov_after_nearest))]
    
    
        os.chdir('..')
        
        pickle.dump(channel_peak_difference_rel, open('channel_peak_difference_rel','wb'))
        pickle.dump(channel_lag_difference_rel, open('channel_lag_difference_rel','wb'))
        
        pickle.dump(channel_peak_difference, open('channel_peak_difference','wb'))
        pickle.dump(channel_lag_difference, open('channel_lag_difference','wb'))
        
        pickle.dump(channel_peak_nearest_difference_rel, open('channel_peak_nearest_difference_rel','wb'))
        pickle.dump(channel_lag_nearest_difference_rel, open('channel_lag_nearest_difference_rel','wb'))
        
        pickle.dump(channel_peak_nearest_difference, open('channel_peak_nearest_difference','wb'))
        pickle.dump(channel_lag_nearest_difference, open('channel_lag_nearest_difference','wb'))
        
        pickle.dump(channel_lag_nonabs_nearest_difference, open('channel_lag_nonabs_nearest_difference','wb'))
        pickle.dump(channel_lag_nonabs_nearest_std_difference, open('channel_lag_nonabs_nearest_std_difference','wb'))
    
    
    
        # --------------------------------------------------------------- median of equidistant channel pairs within each mouse
        channel_peak_difference_median_rel[day_ind,:] = np.asarray(list(map(np.median, channel_peak_difference_rel)))
        channel_lag_difference_median_rel[day_ind,:] = np.asarray(list(map(np.median, channel_lag_difference_rel)))
        channel_peak_difference_median[day_ind,:] = np.asarray(list(map(np.median, channel_peak_difference)))
        channel_lag_difference_median[day_ind,:] = np.asarray(list(map(np.median, channel_lag_difference)))
        
        channel_peak_nearest_difference_median_rel[day_ind,:] = np.asarray(list(map(np.median, channel_peak_nearest_difference_rel)))
        channel_lag_nearest_difference_median_rel[day_ind,:] = np.asarray(list(map(np.median, channel_lag_nearest_difference_rel)))
        channel_peak_nearest_difference_median[day_ind,:] = np.asarray(list(map(np.median, channel_peak_nearest_difference)))
        channel_lag_nearest_difference_median[day_ind,:] = np.asarray(list(map(np.median, channel_lag_nearest_difference)))
        
        channel_lag_nonabs_nearest_difference_median[day_ind,:] = np.asarray(list(map(np.median, channel_lag_nonabs_nearest_difference)))
        channel_lag_nonabs_nearest_std_difference_median[day_ind,:] = np.asarray(list(map(np.median, channel_lag_nonabs_nearest_std_difference)))
        channel_lag_nonabs_nearest_std_difference_median_rel[day_ind,:] = np.asarray(list(map(np.median, channel_lag_nonabs_nearest_std_difference_rel)))
        
        channel_lag_cov_nearest_difference_median[day_ind,:] = np.asarray(list(map(np.median, channel_lag_cov_nearest_difference)))
        channel_lag_cov_nearest_difference_median_rel[day_ind,:] = np.asarray(list(map(np.median, channel_lag_cov_nearest_difference_rel)))
    
    
    
        # --------------------------------------------------------------- mean of equidistant channel pairs within each mouse
        channel_peak_difference_mean_rel[day_ind,:] = np.asarray(list(map(np.mean, channel_peak_difference_rel)))
        channel_lag_difference_mean_rel[day_ind,:] = np.asarray(list(map(np.mean, channel_lag_difference_rel)))
        channel_peak_difference_mean[day_ind,:] = np.asarray(list(map(np.mean, channel_peak_difference)))
        channel_lag_difference_mean[day_ind,:] = np.asarray(list(map(np.mean, channel_lag_difference)))
        
        channel_peak_nearest_difference_mean_rel[day_ind,:] = np.asarray(list(map(np.mean, channel_peak_nearest_difference_rel)))
        channel_lag_nearest_difference_mean_rel[day_ind,:] = np.asarray(list(map(np.mean, channel_lag_nearest_difference_rel)))
        channel_peak_nearest_difference_mean[day_ind,:] = np.asarray(list(map(np.mean, channel_peak_nearest_difference)))
        channel_lag_nearest_difference_mean[day_ind,:] = np.asarray(list(map(np.mean, channel_lag_nearest_difference)))
    
        # mean of equidistant channel pairs, NON Absolute value    
        channel_lag_nearest_difference_mean_rel[day_ind,:] = np.asarray(list(map(np.mean, channel_lag_nearest_difference_rel)))
        channel_lag_nearest_difference_mean[day_ind,:] = np.asarray(list(map(np.mean, channel_lag_nearest_difference)))
    
        channel_lag_cov_nearest_difference_mean[day_ind,:] = np.asarray(list(map(np.mean, channel_lag_cov_nearest_difference)))
        channel_lag_cov_nearest_difference_mean_rel[day_ind,:] = np.asarray(list(map(np.mean, channel_lag_cov_nearest_difference_rel)))
    
        os.chdir('..')
       
        
       
    np.savetxt('channel_peak_difference_median_rel.csv', channel_peak_difference_median_rel, delimiter = ',')
    np.savetxt('channel_lag_difference_median_rel.csv', channel_lag_difference_median_rel, delimiter = ',')
    np.savetxt('channel_peak_difference_median.csv', channel_peak_difference_median, delimiter = ',')
    np.savetxt('channel_lag_difference_median.csv', channel_lag_difference_median, delimiter = ',')
    
    np.savetxt('channel_peak_nearest_difference_median_rel.csv', channel_peak_nearest_difference_median_rel, delimiter = ',')
    np.savetxt('channel_lag_nearest_difference_median_rel.csv', channel_lag_nearest_difference_median_rel, delimiter = ',')
    np.savetxt('channel_peak_nearest_difference_median.csv', channel_peak_nearest_difference_median, delimiter = ',')
    np.savetxt('channel_lag_nearest_difference_median.csv', channel_lag_nearest_difference_median, delimiter = ',')
    
    np.savetxt('channel_lag_nonabs_nearest_difference_median.csv', channel_lag_nonabs_nearest_difference_median, delimiter = ',')
    np.savetxt('channel_lag_nonabs_nearest_std_difference_median.csv', channel_lag_nonabs_nearest_std_difference_median, delimiter = ',')
    np.savetxt('channel_lag_nonabs_nearest_std_difference_median_rel.csv', channel_lag_nonabs_nearest_std_difference_median_rel, delimiter = ',')
    
    np.savetxt('channel_lag_cov_nearest_difference_median.csv', channel_lag_cov_nearest_difference_median, delimiter = ',')
    np.savetxt('channel_lag_cov_nearest_difference_median_rel.csv', channel_lag_cov_nearest_difference_median_rel, delimiter = ',')
    
    
    np.savetxt('channel_peak_difference_mean_rel.csv', channel_peak_difference_mean_rel, delimiter = ',')
    np.savetxt('channel_lag_difference_mean_rel.csv', channel_lag_difference_mean_rel, delimiter = ',')
    np.savetxt('channel_peak_difference_mean.csv', channel_peak_difference_mean, delimiter = ',')
    np.savetxt('channel_lag_difference_mean.csv', channel_lag_difference_mean, delimiter = ',')
    
    np.savetxt('channel_peak_nearest_difference_mean_rel.csv', channel_peak_nearest_difference_mean_rel, delimiter = ',')
    np.savetxt('channel_lag_nearest_difference_mean_rel.csv', channel_lag_nearest_difference_mean_rel, delimiter = ',')
    np.savetxt('channel_peak_nearest_difference_mean.csv', channel_peak_nearest_difference_mean, delimiter = ',')
    np.savetxt('channel_lag_nearest_difference_mean.csv', channel_lag_nearest_difference_mean, delimiter = ',')
        
    np.savetxt('channel_lag_cov_nearest_difference_mean.csv', channel_lag_cov_nearest_difference_mean, delimiter = ',')
    np.savetxt('channel_lag_cov_nearest_difference_mean_rel.csv', channel_lag_cov_nearest_difference_mean_rel, delimiter = ',')
    
    os.chdir('..')

# fig1.savefig('corr peak', dpi = 1000)
# fig2.savefig('corr lag', dpi = 1000)
# fig3.savefig('corr peak nearest', dpi = 1000)
# fig4.savefig('corr lag nearest', dpi = 1000)


#%% plot LFP before vs after for one experiment with exp curve fitting
fig1, ax1 = plt.subplots()
fig1.set_tight_layout(True)


fig2, ax2 = plt.subplots()
fig2.set_tight_layout(True)


to_plot_1_corr = [0,1,2,3]
# to_plot_2_corr = [4,5,6,7,8,9]

color = ['b', 'r']

def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

p0 = (0.5, 0.001, 0.01) # start with value for the curve fit near those we expect

distances_to_plot = 12
electrode_distances_to_plot = electrode_distances[:distances_to_plot] 

corr_peak_per_stim = pickle.load(open('corr_peak_per_stim', 'rb'))
corr_peak_norm_per_stim = pickle.load(open('corr_peak_norm_per_stim', 'rb'))
corr_lag_per_stim = pickle.load(open('corr_lag_per_stim', 'rb'))
corr_lag_per_stim_abs = list(map(np.abs, corr_lag_per_stim))

os.chdir([i for i in os.listdir() if 'analysis' in i][0])
 
SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',')
LFP_resp_channels = np.loadtxt('LFP_resp_channels.csv', delimiter = ',') 

# to_plot_2_corr = np.loadtxt('to_plot_2_delta.csv', delimiter = ',', dtype = int)
if '021221' in os.getcwd() or '131221' in os.getcwd():
    to_plot_2_corr = [4,5]
elif '111121' in os.getcwd():
    to_plot_2_corr = [4]
else:
    to_plot_2_corr = [4,5,6,7,8,9]
channels_to_plot = LFP_resp_channels


for cond_ind, cond in enumerate(['before', 'after']):
    if cond_ind == 0:
        peak_before = distance_plot(np.nanmedian(np.concatenate([corr_peak_norm_per_stim[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)[:distances_to_plot]
        lag_before = distance_plot(np.nanmedian(np.concatenate([corr_lag_per_stim_abs[i] for i in to_plot_1_corr], axis = 0), axis = 0), channels = channels_to_plot)[:distances_to_plot]
        # medians_before[day_ind, :] = np.asarray(list(map(np.median, a)))
    else:
        if len(to_plot_2_corr) == 1:
            peak_after = distance_plot(np.nanmedian(corr_peak_norm_per_stim[to_plot_2_corr[0]], axis = 0), channels = channels_to_plot)[:distances_to_plot]
            lag_after = distance_plot(np.nanmedian(corr_lag_per_stim_abs[to_plot_2_corr[0]], axis = 0), channels = channels_to_plot)[:distances_to_plot]           
        else:
            peak_after = distance_plot(np.nanmedian(np.concatenate([corr_peak_norm_per_stim[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)[:distances_to_plot]
            lag_after = distance_plot(np.nanmedian(np.concatenate([corr_lag_per_stim_abs[i] for i in to_plot_2_corr], axis = 0), axis = 0), channels = channels_to_plot)[:distances_to_plot]
        # medians_after[day_ind, :] = np.asarray(list(map(np.median, a)))            
    
    if cond_ind == 0:
        ydata_peak = np.concatenate(peak_before)
        xdata_peak = np.repeat(electrode_distances_to_plot[:12], [len(i) for i in peak_before])
        params_peak, cv_peak = scipy.optimize.curve_fit(monoExp, xdata_peak, ydata_peak, p0, maxfev = 100000000)
        
        ydata_lag = np.concatenate(lag_before)
        xdata_lag = np.repeat(electrode_distances_to_plot, [len(i) for i in lag_before])
        params_lag, cv_lag = scipy.optimize.curve_fit(monoExp, xdata_lag, ydata_lag, p0, maxfev = 100000000)
    else:
        ydata_peak = np.concatenate(peak_after)
        xdata_peak = np.repeat(electrode_distances_to_plot, [len(i) for i in peak_after])
        params_peak, cv_peak = scipy.optimize.curve_fit(monoExp, xdata_peak, ydata_peak, p0, maxfev = 100000000)
        
        ydata_lag = np.concatenate(lag_after)
        xdata_lag = np.repeat(electrode_distances_to_plot, [len(i) for i in lag_after])
        params_lag, cv_lag = scipy.optimize.curve_fit(monoExp, xdata_lag, ydata_lag, p0, maxfev = 100000000)
    # print(params[0], params[1], params[2])
    
    # if cond_ind == 0:
    #     params_before[day_ind, t_ind, :] = params[:]
    #     AUC_before[day_ind, t_ind] = auc(electrode_distances, monoExp(electrode_distances, params[0], params[1], params[2]))
    # else: 
    #     params_after[day_ind, t_ind, :] = params[:]
    #     AUC_after[day_ind, t_ind] = auc(electrode_distances, monoExp(electrode_distances, params[0], params[1], params[2]))
    
    ax1.plot(electrode_distances_to_plot, monoExp(electrode_distances_to_plot, params_peak[0], params_peak[1], params_peak[2]), c = color[cond_ind])
    ax1.scatter(xdata_peak, ydata_peak, c = color[cond_ind], s=1)
    ax1.set_xlabel('distance between channels (ym)', size = 16)
    ax1.set_ylabel('correlation peak', size = 16)
    ax1.set_xticks(np.linspace(200,electrode_distances_to_plot[-1],5))
    ax1.set_xticklabels(['200', '400', '600', '800', '1000'], size = 16)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_yticks([0.25,0.5,0.75,1])
    ax1.set_yticklabels(list(map(str, list(ax1.get_yticks()))), size = 16)
    # plt.tight_layout()
    if cond_ind == 1:
        fig1.savefig('LFP corr peak before vs after.pdf', dpi = 1000, format = 'pdf')
        fig1.savefig('LFP corr peak before vs after.jpg', dpi = 1000, format = 'jpg')

    ax2.plot(electrode_distances_to_plot, monoExp(electrode_distances_to_plot, params_lag[0], params_lag[1], params_lag[2]), c = color[cond_ind])
    ax2.scatter(xdata_lag, ydata_lag, c = color[cond_ind], s=1)
    ax2.set_xlabel('distance between channels (ym)', size = 16)
    ax2.set_ylabel('correlation lag (ms)', size = 16)
    ax2.set_xticks(np.linspace(200,electrode_distances_to_plot[-1],5))
    ax2.set_xticklabels(['200', '400', '600', '800', '1000'], size = 16)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_yticks([0,100,200,300,400])
    ax2.set_yticklabels(list(map(str, list(ax2.get_yticks()))), size = 16)
    # plt.tight_layout()
    if cond_ind == 1:
        fig2.savefig('LFP corr lag before vs after.pdf', dpi = 1000, format = 'pdf')
        fig2.savefig('LFP corr lag before vs after.jpg', dpi = 1000, format = 'jpg')

os.chdir('..')


#%% GRAND LFP cross correlation UP vs DOWN

patch = 1 #(how many times SEM to plot as patch)
plot = 'SEM' # 'SEM' or 'IQR'

distance_to_plot = 13

#nearest peak and lag or not
nearest = True


# relative peak
fig, ax = plt.subplots(figsize = (7,3))
fig.suptitle('CORR PEAK REL')
os.chdir(os.path.join(overall_path, r'UP_pairing'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i) and 'old' not in i]
if nearest == True:
    to_plot = np.loadtxt('channel_peak_nearest_difference_median_rel.csv', delimiter = ',')[:,0:distance_to_plot]
else:
    to_plot = np.loadtxt('channel_peak_difference_median_rel.csv', delimiter = ',')[:,0:distance_to_plot]
# select mice to plot:
print(f'incomplete UP mice: {[days[ind] for ind,i in enumerate(np.any(np.isnan(to_plot), axis=1)) if i == True]}')
to_plot = to_plot[~np.any(np.isnan(to_plot), axis=1),:]*100
print(f'complete UP mice: {to_plot.shape[0]}')
if plot == 'SEM':
    ax.plot(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0), color = 'r')
    ax.fill_between(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'r')
elif plot == 'IQR':
    ax.plot(electrode_distances[:distance_to_plot], np.nanmedian(to_plot, axis = 0), color = 'r')
    ax.fill_between(electrode_distances[:distance_to_plot], patch*np.percentile(to_plot, 25, axis = 0), patch*np.percentile(to_plot, 75, axis = 0), alpha = 0.1, color = 'r')  
os.chdir('..')

os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
if nearest == True:
    to_plot = np.loadtxt('channel_peak_nearest_difference_median_rel.csv', delimiter = ',')[:,0:distance_to_plot]
else:
    to_plot = np.loadtxt('channel_peak_difference_median_rel.csv', delimiter = ',')[:,0:distance_to_plot]
# select mice to plot:
print(f'incomplete DOWN mice: {[days[ind] for ind,i in enumerate(np.any(np.isnan(to_plot), axis=1)) if i == True]}')
to_plot = to_plot[~np.any(np.isnan(to_plot), axis=1),:]*100
print(f'complete DOWN mice: {to_plot.shape[0]}')
if plot == 'SEM':
    ax.plot(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0), color = 'k')
    ax.fill_between(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'k')
elif plot == 'IQR':
    ax.plot(electrode_distances[:distance_to_plot], np.nanmedian(to_plot, axis = 0), color = 'k')
    ax.fill_between(electrode_distances[:distance_to_plot], patch*np.percentile(to_plot, 25, axis = 0), patch*np.percentile(to_plot, 75, axis = 0), alpha = 0.1, color = 'k')  
# ax.set_ylim([40, 160])
ax.set_xlabel('inter-channel distance (mm)', size = 16)
ax.set_ylabel('correlation coefficient \n post-pairing \n (% of baseline)', size = 16)
ax.set_xticks(np.linspace(200,1000,5))
ax.set_xticklabels(['0.2', '0.4', '0.6', '0.8', '1'], size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([-15, -10,-5,0, 5])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
plt.tight_layout()
os.chdir(overall_path)
# plt.savefig('LFP rel corr UP vs DOWN.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('LFP rel corr UP vs DOWN.jpg', dpi = 1000, format = 'jpg')




# relative lag
os.chdir(overall_path)
patch = 1 #(how many times SEM to plot as patch)
fig, ax = plt.subplots(figsize = (7,3))
# fig.suptitle('CORR LAG REL')
os.chdir(os.path.join(overall_path, r'UP_pairing'))
if nearest == True:
    to_plot = np.loadtxt('channel_lag_nearest_difference_median_rel.csv', delimiter = ',')[:,0:distance_to_plot]
else:
    to_plot = np.loadtxt('channel_lag_difference_median_rel.csv', delimiter = ',')[:,0:distance_to_plot]
to_plot = to_plot[~np.any(np.isnan(to_plot), axis=1),:]*100
if plot == 'SEM':
    ax.plot(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0), color = 'r')
    ax.fill_between(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'r')
elif plot == 'IQR':
    ax.plot(electrode_distances[:distance_to_plot], np.nanmedian(to_plot, axis = 0), color = 'r')
    ax.fill_between(electrode_distances[:distance_to_plot], patch*np.percentile(to_plot, 25, axis = 0), patch*np.percentile(to_plot, 75, axis = 0), alpha = 0.1, color = 'r')  
os.chdir('..')

os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
if nearest == True:
    to_plot = np.loadtxt('channel_lag_nearest_difference_median_rel.csv', delimiter = ',')[:,0:distance_to_plot]
else:
    to_plot = np.loadtxt('channel_lag_difference_median_rel.csv', delimiter = ',')[:,0:distance_to_plot]
to_plot = to_plot[~np.any(np.isnan(to_plot), axis=1),:]*100
if plot == 'SEM':
    ax.plot(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0), color = 'k')
    ax.fill_between(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'k')
elif plot == 'IQR':
    ax.plot(electrode_distances[:distance_to_plot], np.nanmedian(to_plot, axis = 0), color = 'k')
    ax.fill_between(electrode_distances[:distance_to_plot], patch*np.percentile(to_plot, 25, axis = 0), patch*np.percentile(to_plot, 75, axis = 0), alpha = 0.1, color = 'k')  
# ax.set_ylim([40, 160])
ax.set_xlabel('inter-channel distance (mm)', size = 16)
ax.set_ylabel('peak lag \n post-pairing \n (% of baseline)', size = 16)
ax.set_xticks(np.linspace(200,1000,5))
ax.set_xticklabels(['0.2', '0.4', '0.6', '0.8', '1'], size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([-20,-10,0,10,20,30])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
plt.tight_layout()
os.chdir(overall_path)
# plt.savefig('LFP rel corr lag UP vs DOWN.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('LFP rel corr lag UP vs DOWN.jpg', dpi = 1000, format = 'jpg')




# # Absolute peak
# os.chdir(overall_path)
# patch = 1 #(how many times SEM to plot as patch)
# fig, ax = plt.subplots(figsize = (7,3))
# fig.suptitle('CORR PEAK ABS')
# os.chdir(os.path.join(overall_path, r'UP_pairing'))
# if nearest == True:
#     to_plot = np.loadtxt('channel_peak_nearest_difference_median.csv', delimiter = ',')[:,0:distance_to_plot]
# else:
#     to_plot = np.loadtxt('channel_peak_difference_median.csv', delimiter = ',')[:,0:distance_to_plot]
# to_plot = to_plot[~np.any(np.isnan(to_plot), axis=1),:]
# if plot == 'SEM':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0), color = 'r')
#     ax.fill_between(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'r')
# elif plot == 'IQR':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmedian(to_plot, axis = 0), color = 'r')
#     ax.fill_between(electrode_distances[:distance_to_plot], patch*np.percentile(to_plot, 25, axis = 0), patch*np.percentile(to_plot, 75, axis = 0), alpha = 0.1, color = 'r')  
# os.chdir('..')

# os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
# if nearest == True:
#     to_plot = np.loadtxt('channel_peak_nearest_difference_median.csv', delimiter = ',')[:,0:distance_to_plot]
# else:
#     to_plot = np.loadtxt('channel_peak_difference_median.csv', delimiter = ',')[:,0:distance_to_plot]
# to_plot = to_plot[~np.any(np.isnan(to_plot), axis=1),:]
# if plot == 'SEM':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0), color = 'k')
#     ax.fill_between(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'k')
# elif plot == 'IQR':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmedian(to_plot, axis = 0), color = 'k')
#     ax.fill_between(electrode_distances[:distance_to_plot], patch*np.percentile(to_plot, 25, axis = 0), patch*np.percentile(to_plot, 75, axis = 0), alpha = 0.1, color = 'k')  
# # ax.set_ylim([40, 160])
# ax.set_xlabel('inter-channel distance (mm)', size = 16)
# ax.set_ylabel('correlation peak difference', size = 16)
# ax.set_xticks(np.linspace(200,1000,5))
# ax.set_xticklabels(['0.2', '0.4', '0.6', '0.8', '1'], size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # ax.set_yticks([50,75,100,125,150])
# # ax.set_yticklabels(list(map(str, [50,75,100,125,150])), size = 16)
# plt.tight_layout()
# os.chdir(overall_path)




# # absolute lag
# os.chdir(overall_path)
# patch = 1 #(how many times SEM to plot as patch)
# fig, ax = plt.subplots(figsize = (7,3))
# # fig.suptitle('CORR LAG ABS')
# os.chdir(os.path.join(overall_path, r'UP_pairing'))
# if nearest == True:
#     to_plot = np.loadtxt('channel_lag_nearest_difference_median.csv', delimiter = ',')[:,0:distance_to_plot]
# else:
#     to_plot = np.loadtxt('channel_lag_difference_median.csv', delimiter = ',')[:,0:distance_to_plot]
# to_plot = to_plot[~np.any(np.isnan(to_plot), axis=1),:]
# if plot == 'SEM':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0), color = 'r')
#     ax.fill_between(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'r')
# elif plot == 'IQR':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmedian(to_plot, axis = 0), color = 'r')
#     ax.fill_between(electrode_distances[:distance_to_plot], patch*np.percentile(to_plot, 25, axis = 0), patch*np.percentile(to_plot, 75, axis = 0), alpha = 0.1, color = 'r')  
# os.chdir('..')

# os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
# if nearest == True:
#     to_plot = np.loadtxt('channel_lag_nearest_difference_median.csv', delimiter = ',')[:,0:distance_to_plot]
# else:
#     to_plot = np.loadtxt('channel_lag_difference_median.csv', delimiter = ',')[:,0:distance_to_plot]
# to_plot = to_plot[~np.any(np.isnan(to_plot), axis=1),:]
# if plot == 'SEM':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0), color = 'k')
#     ax.fill_between(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'k')
# elif plot == 'IQR':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmedian(to_plot, axis = 0), color = 'k')
#     ax.fill_between(electrode_distances[:distance_to_plot], patch*np.percentile(to_plot, 25, axis = 0), patch*np.percentile(to_plot, 75, axis = 0), alpha = 0.1, color = 'k')  
# # ax.set_ylim([40, 160])
# ax.set_xlabel('inter-channel distance (mm)', size = 16)
# ax.set_ylabel('correlation lag difference (ms)', size = 16)
# ax.set_xticks(np.linspace(200,1000,5))
# ax.set_xticklabels(['0.2', '0.4', '0.6', '0.8', '1'], size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_yticks([-5,0,5,10,15,20])
# ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
# plt.tight_layout()
# os.chdir(overall_path)
# plt.savefig('LFP abs corr lag UP vs DOWN.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('LFP abs corr lag UP vs DOWN.jpg', dpi = 1000, format = 'jpg')



#  # plt.savefig('LFP non abs corr lag UP vs DOWN.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('LFP non abs corr lag UP vs DOWN.jpg', dpi = 1000, format = 'jpg')


# lag standard deviation
# os.chdir(overall_path)
# patch = 1 #(how many times SEM to plot as patch)
# fig, ax = plt.subplots(figsize = (7,3))
# # fig.suptitle('CORR LAG STD')
# os.chdir(os.path.join(overall_path, r'UP_pairing'))
# to_plot = np.loadtxt('channel_lag_nonabs_nearest_std_difference_median_rel.csv', delimiter = ',')[:,0:distance_to_plot]
# to_plot = to_plot[~np.any(np.isnan(to_plot), axis=1),:]*100
# if plot == 'SEM':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0), color = 'r')
#     ax.fill_between(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'r')
# elif plot == 'IQR':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmedian(to_plot, axis = 0), color = 'r')
#     ax.fill_between(electrode_distances[:distance_to_plot], patch*np.percentile(to_plot, 25, axis = 0), patch*np.percentile(to_plot, 75, axis = 0), alpha = 0.1, color = 'r')  
# os.chdir('..')

# os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
# to_plot = np.loadtxt('channel_lag_nonabs_nearest_std_difference_median_rel.csv', delimiter = ',')[:,0:distance_to_plot]
# to_plot = to_plot[~np.any(np.isnan(to_plot), axis=1),:]*100
# if plot == 'SEM':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0), color = 'k')
#     ax.fill_between(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'k')
# elif plot == 'IQR':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmedian(to_plot, axis = 0), color = 'k')
#     ax.fill_between(electrode_distances[:distance_to_plot], patch*np.percentile(to_plot, 25, axis = 0), patch*np.percentile(to_plot, 75, axis = 0), alpha = 0.1, color = 'k')  
# # ax.set_ylim([40, 160])
# ax.set_xlabel('inter-channel distance (mm)', size = 16)
# ax.set_ylabel('peak lag SD \n post-pairing \n (% of baseline)', size = 16)
# ax.set_xticks(np.linspace(200,1000,5))
# ax.set_xticklabels(['0.2', '0.4', '0.6', '0.8', '1'], size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_yticks([-10,0,10,20,30,40])
# ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
# plt.tight_layout()
# os.chdir(overall_path)
# plt.savefig('LFP non abs corr lag std UP vs DOWN.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('LFP non abs corr lag std UP vs DOWN.jpg', dpi = 1000, format = 'jpg')


# coefficient of variation
# os.chdir(overall_path)
# patch = 1 #(how many times SEM to plot as patch)
# fig, ax = plt.subplots(figsize = (7,3))
# # fig.suptitle('CORR LAG NON ABS')
# os.chdir(os.path.join(overall_path, r'UP_pairing'))
# to_plot = np.loadtxt('channel_lag_cov_nearest_difference_median.csv', delimiter = ',')[:,0:distance_to_plot]
# to_plot = to_plot[~np.any(np.isnan(to_plot), axis=1),:]
# if plot == 'SEM':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0), color = 'r')
#     ax.fill_between(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'r')
# elif plot == 'IQR':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmedian(to_plot, axis = 0), color = 'r')
#     ax.fill_between(electrode_distances[:distance_to_plot], patch*np.percentile(to_plot, 25, axis = 0), patch*np.percentile(to_plot, 75, axis = 0), alpha = 0.1, color = 'r')  
# os.chdir('..')

# os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
# to_plot = np.loadtxt('channel_lag_cov_nearest_difference_median.csv', delimiter = ',')[:,0:distance_to_plot]
# to_plot = to_plot[~np.any(np.isnan(to_plot), axis=1),:]
# if plot == 'SEM':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0), color = 'k')
#     ax.fill_between(electrode_distances[:distance_to_plot], np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'k')
# elif plot == 'IQR':
#     ax.plot(electrode_distances[:distance_to_plot], np.nanmedian(to_plot, axis = 0), color = 'k')
#     ax.fill_between(electrode_distances[:distance_to_plot], patch*np.percentile(to_plot, 25, axis = 0), patch*np.percentile(to_plot, 75, axis = 0), alpha = 0.1, color = 'k')  
# # ax.set_ylim([40, 160])
# ax.set_yticks([-0.1, 0, 0.1])
# ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
# ax.set_xlabel('inter-channel distance (mm)', size = 16)
# ax.set_ylabel('coefficient of variation', size = 16)
# ax.set_xticks(np.linspace(200,1000,5))
# ax.set_xticklabels(['0.2', '0.4', '0.6', '0.8', '1'], size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # ax.set_yticks([-5,0,5,10,15])
# # ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
# plt.tight_layout()
# os.chdir(overall_path)
# plt.savefig('LFP lag coeff of variation UP vs DOWN.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('LFP lag coeff of variation UP vs DOWN.jpg', dpi = 1000, format = 'jpg')



#%% STTC UP vs DOWN, medianed over equidistant channels. 1) averaged over all distances 2) 2D colorplot with distances vs timesteps?
distance_to_plot = 13 # 
time_steps = [1,2,5,10,20,50,100,200,500,1000,2000]

patch = 1


def interpolate_grid(array, space_interp = 25):
    # interpolate in space, for better visualization
    #you have to flatten the array and then define X and Y coords for every point
    flat = array.flatten()
    grid_x = np.tile(np.log(time_steps), array.shape[0]) #
    grid_y = np.repeat(electrode_distances[:distance_to_plot], array.shape[1]) #
    grid_x_int, grid_y_int = np.meshgrid(np.linspace(np.log(time_steps)[0], np.log(time_steps)[-1], space_interp), np.linspace(electrode_distances[0], electrode_distances[distance_to_plot - 1], space_interp)) # i.e. the grid you want to interpolate to
    flat_spatial_interpolated = scipy.interpolate.griddata((grid_x, grid_y), flat, (grid_x_int, grid_y_int), method='cubic')
    return flat_spatial_interpolated


#absolute STTC difference in UP pairing with timesteps and distances:
os.chdir(os.path.join(overall_path, r'UP_pairing'))
to_plot = np.loadtxt('STTC_medians_diff_ANOVA_overlap_False.csv', delimiter = ',')
to_plot = np.mean(np.reshape(to_plot, (to_plot.shape[0], len(time_steps), distance_to_plot)), axis = 0) # average across mice
to_plot_UP = copy.deepcopy(to_plot)
fig, ax = plt.subplots()
y = electrode_distances[:distance_to_plot]
x = np.log(time_steps)
X,Y = np.meshgrid(x,y)
ax.pcolormesh(X, Y, -to_plot_UP.T, cmap = 'jet')
ax.set_xlabel('tiling window (ms)', size = 16)
ax.set_ylabel('distance between channels (mm)', size = 16)

space_interp = 100
a = interpolate_grid(-to_plot_UP.T, space_interp = space_interp)
fig, ax = plt.subplots()
ax.imshow(np.flip(a, axis = 0), cmap = 'jet')
ax.set_xlabel('tiling window (ms)', size = 16)
ax.set_ylabel('distance between channels (mm)', size = 16)
ax.set_xticks((np.log(time_steps)*(space_interp/np.log(time_steps)[-1])))
ax.set_xticklabels(['1','','', '10','','', '100','','', '1000',''], rotation = 45, size = 14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks(np.linspace(0,100,5))
ax.set_yticklabels(['1', '0.8', '0.6', '0.4', '0.2'], size = 14)
plt.tight_layout()
os.chdir(overall_path)
plt.savefig('STTC UP heatmap.pdf', dpi = 1000, format = 'pdf')
plt.savefig('STTC UP heatmap.jpg', dpi = 1000, format = 'jpg')

fig, ax = plt.subplots(figsize = (2,5))
norm = colors.Normalize(vmin=0, vmax=0.17)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'),
              cax=ax, ticks = [0,0.05,0.1,0.15])
ax.set_yticklabels(list(map(str, [0,-0.05,-0.1,-0.15])), size = 18)
ax.set_ylabel('STTC difference', size = 16)
plt.tight_layout()
plt.savefig('STTC difference colormap legend.pdf', dpi = 1000, format = 'pdf')
plt.savefig('STTC difference colormap legend.jpg', dpi = 1000, format = 'jpg')


#absolute
fig, ax = plt.subplots(figsize = (6,4))
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
to_plot = np.loadtxt('STTC_medians_diff_ANOVA_overlap_False.csv', delimiter = ',')
to_plot = np.nanmean(np.reshape(to_plot, (to_plot.shape[0], len(time_steps), distance_to_plot)), axis = 2) # average across electrode distances within  (as there is no significant effect of distance)
to_plot_DOWN = copy.deepcopy(to_plot)
# to_plot = np.loadtxt('STTC_medians_diff_ANOVA_overlap_False_OLd.csv', delimiter = ',')
# to_plot = np.nanmean(np.reshape(to_plot, (7,12,11)), axis = 2)
# to_plot_DOWN_old = copy.deepcopy(to_plot)
ax.plot(np.log(time_steps), np.nanmean(to_plot, axis = 0), color = 'k')
ax.fill_between(np.log(time_steps), np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'k')
os.chdir(os.path.join(overall_path, r'UP_pairing'))
to_plot = np.loadtxt('STTC_medians_diff_ANOVA_overlap_False.csv', delimiter = ',')
to_plot = np.nanmean(np.reshape(to_plot, (to_plot.shape[0], len(time_steps), distance_to_plot)), axis = 2)
to_plot_UP = copy.deepcopy(to_plot)
# to_plot = np.loadtxt('STTC_medians_diff_ANOVA_overlap_False_OLd.csv', delimiter = ',')
# to_plot = np.nanmean(np.reshape(to_plot, (9,12,11)), axis = 2)
# to_plot_UP_old = copy.deepcopy(to_plot)
ax.plot(np.log(time_steps), np.nanmean(to_plot, axis = 0), color = 'r')
ax.fill_between(np.log(time_steps), np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'r')
ax.set_xlabel('tiling window (ms)', size = 16)
ax.set_ylabel('STTC difference', size = 16)
ax.set_xticks(np.log(time_steps))
ax.set_xticklabels(list(map(str, time_steps)), rotation = 45, size = 14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([-0.15,-0.1,-0.05,0, 0.05])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 14)
# ax.set_xlim([0,12])
plt.tight_layout()
os.chdir(overall_path)
plt.savefig('STTC UP vs DOWN.pdf', dpi = 1000, format = 'pdf')
plt.savefig('STTC UP vs DOWN.jpg', dpi = 1000, format = 'jpg')


# relative
fig, ax = plt.subplots()
os.chdir(os.path.join(overall_path, r'DOWN_pairing'))
to_plot = np.loadtxt('STTC_medians_diff_rel_ANOVA_overlap_False.csv', delimiter = ',')
to_plot = np.nanmean(np.reshape(to_plot, (to_plot.shape[0], len(time_steps), distance_to_plot)), axis = 2) # average across electrode distances (as there is no significant effect of distance)
to_plot_DOWN = copy.deepcopy(to_plot)
# to_plot = np.loadtxt('STTC_medians_diff_ANOVA_overlap_False_OLd.csv', delimiter = ',')
# to_plot = np.nanmean(np.reshape(to_plot, (7,12,11)), axis = 2)
# to_plot_DOWN_old = copy.deepcopy(to_plot)
ax.plot(np.log(time_steps), np.nanmean(to_plot, axis = 0), color = 'k')
ax.fill_between(np.log(time_steps), np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'k')
os.chdir(os.path.join(overall_path, r'UP_pairing'))
to_plot = np.loadtxt('STTC_medians_diff_rel_ANOVA_overlap_False.csv', delimiter = ',')
to_plot = np.nanmean(np.reshape(to_plot, (to_plot.shape[0], len(time_steps), distance_to_plot)), axis = 2)
to_plot_UP = copy.deepcopy(to_plot)
# to_plot = np.loadtxt('STTC_medians_diff_ANOVA_overlap_False_OLd.csv', delimiter = ',')
# to_plot = np.nanmean(np.reshape(to_plot, (9,12,11)), axis = 2)
# to_plot_UP_old = copy.deepcopy(to_plot)
ax.plot(np.log(time_steps), np.nanmean(to_plot, axis = 0), color = 'r')
ax.fill_between(np.log(time_steps), np.nanmean(to_plot, axis = 0) + patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot, axis = 0) - patch*np.nanstd(to_plot, axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.1, color = 'r')
ax.set_xlabel('tiling window (ms)', size = 16)
ax.set_ylabel('STTC difference', size = 16)
ax.set_xticks(np.log(time_steps))
ax.set_xticklabels(list(map(str, time_steps)), rotation = 45, size = 14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([-0.15,-0.1,-0.05,0, 0.05])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 14)
# ax.set_xlim([0,12])
plt.tight_layout()
os.chdir(overall_path)
# plt.savefig('STTC UP vs DOWN rel.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('STTC UP vs DOWN rel.jpg', dpi = 1000, format = 'jpg')



