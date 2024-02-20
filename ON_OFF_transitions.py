# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:13:20 2023

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


# exclude_before = 0.1
# # maybe better to take 1 second after stim for slow waves as high change they get fucked up by the stim otherwise?
# exclude_after = 1.9


def cl():
    plt.close('all')

def smooth(y, box_pts, axis = 0):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.apply_along_axis(lambda m: np.convolve(m, box, mode='same'), axis = axis, arr = y)
    return y_smooth  

#%% ----------------------------------------------- define population ON OFF states and extract transition synchrony

# OFF state is >99 percentile of gamma fit of ISIs Vlad 2011/Rodriguez 2016, or no population activity for 50ms or longer, Vlad 2009
# I define OFF states as no activity in >90% of channels
# calculate latency of other electrodes to first or last spike in that transition (only use ON states where >50% of channels for these transitions)
# os.chdir(os.path.join(overall_path, r'UP_pairing'))

intersect = False
unique = False

channel_proportion_off = 0.9 # proportion of channels that have to be silent for an OFF state to count:
OFF_duration_threshold = 100
ON_duration_threshold = 250
safety_buffer = 400 # how much safety buffer to end of inter stim time window on both sides (has to be bigger than duration thresholds to exclude lone spikes at the beginning and end)
spike_minimum_on = 5 # how many spikes at a minimum in an UP state
transition_window = 250 # how long to look for spikes within each transition


lfp_cutoff_resp_channels = 200
to_plot_1_LFP = [0,1,2,3]
to_plot_2_LFP = [4,5,6,7,8,9]  
to_plot_1_SW = [0,1,2,3]
to_plot_2_SW = [4,5,6,7,8,9]  

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
#only use mice with a full set of inter channel pairs
# for day_ind, day in enumerate(['121121', '160310', '160414_D1', '160426_D1', '160519_B2', '201121', '221220_3', '281021', '291021']):
# for day_ind, day in enumerate(['121121', '160414_D1', '160426_D1', '160519_B2', '201121', '221220_3', '281021', '291021']):
# for day_ind, day in enumerate(['160310']):
for day_ind, day in enumerate(days):
    
    if day == '160310':
        highpass_cutoff = 7
    else:
        highpass_cutoff = 4
    
    os.chdir(day)
    print(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    
    try:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}_cleaned','rb'))
    except FileNotFoundError:
        try:
            spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
        except FileNotFoundError:
            print(f'spikes with highpass {highpass_cutoff} not found')
            spikes_allsweeps = pickle.load(open([i for i in os.listdir() if 'spikes_allsweeps' in i][0],'rb'))
    
    stim_times = pickle.load(open('stim_times','rb'))
    if os.path.exists('stims_for_delta'):
        stims_for_delta = pickle.load(open('stims_for_delta','rb'))
    else:
        stims_for_delta = copy.deepcopy(stim_times)
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_resp_channels_cutoff =  np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
    SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',')
    LFP_SW_channels = LFP_resp_channels_cutoff
    if intersect:
        LFP_SW_channels = np.intersect1d(LFP_resp_channels_cutoff, SW_spiking_channels)
    elif unique:
        LFP_SW_channels = np.unique(np.concatenate([LFP_resp_channels_cutoff, SW_spiking_channels]))
     
        
     # ---------------------------------------------------------- select channels -----------------------------------------------------------------------
    # DOWN
    if '160308' in os.getcwd():
        chans_to_append = [55,10,12,14,17,19,21,23,25,53,51]
        chans_to_append = [i for i in chans_to_append if i not in LFP_SW_channels]
        for chan in chans_to_append:
            LFP_SW_channels = np.append(LFP_SW_channels, chan)
    if '160420' in os.getcwd():
        chans_to_append = [9,59,61,15,14,62,60,58,52,50,63,0,2,4,6,8,10]
        chans_to_append = [i for i in chans_to_append if i not in LFP_SW_channels]
        for chan in chans_to_append:
            LFP_SW_channels = np.append(LFP_SW_channels, chan)
            LFP_SW_channels = np.append(LFP_SW_channels, chan)     
    # if '160427' in os.getcwd():
        # chans_to_append = [1,5,3,13,12,2,4,56,26]
        # chans_to_append = [i for i in chans_to_append if i not in LFP_SW_channels]
        # for chan in chans_to_append:
        #     LFP_SW_channels = np.append(LFP_SW_channels, chan)
        #     LFP_SW_channels = np.append(LFP_SW_channels, chan)     
    if '221212' in os.getcwd():
        chans_to_append = [22,1,49,0,51,2,4,6,15]
        chans_to_append = [i for i in chans_to_append if i not in LFP_SW_channels]
        for chan in chans_to_append:
            LFP_SW_channels = np.append(LFP_SW_channels, chan)    
        #some channels have small spiking artifacts not tied to SW activity which artificially increase STTC metric in the lower timesteps (taking these out biases against own hypothesis)
        # chans_to_delete = [25,33,35,21,39,37,35,33,21]
        # for chan in chans_to_delete:
        #     LFP_SW_channels = np.delete(LFP_SW_channels, np.where(LFP_SW_channels == chan)[0])
    if '221216' in os.getcwd():
        to_plot_1_corr = [3] # noisy background high-frequency before
        chans_to_append = [30,28]
        chans_to_append = [i for i in chans_to_append if i not in LFP_SW_channels]
        for chan in chans_to_append:
            LFP_SW_channels = np.append(LFP_SW_channels, chan)    
        #some channels have small spiking artifacts not tied to SW activity which artificially increase STTC metric in the lower timesteps (taking these out biases against own hypothesis)
        chans_to_delete = [9,57]
        for chan in chans_to_delete:
            LFP_SW_channels = np.delete(LFP_SW_channels, np.where(LFP_SW_channels == chan)[0])
    # UP
    if '160414' in os.getcwd():
        chans_to_append = [41,43,9,45,27,58]
        chans_to_append = [i for i in chans_to_append if i not in LFP_SW_channels]
        for chan in chans_to_append:
            LFP_SW_channels = np.append(LFP_SW_channels, chan)
            LFP_SW_channels = np.append(LFP_SW_channels, chan)     
    if '160426' in os.getcwd():
        chans_to_append = [33,63,14,22,23,25,9,7,11,41,5,53,55,1]
        chans_to_append = [i for i in chans_to_append if i not in LFP_SW_channels]
        for chan in chans_to_append:
            LFP_SW_channels = np.append(LFP_SW_channels, chan)
            LFP_SW_channels = np.append(LFP_SW_channels, chan)            
    if '121121' in os.getcwd():
        chans_to_append = [46,44,42,62,25]
        chans_to_append = [i for i in chans_to_append if i not in LFP_SW_channels]
        for chan in chans_to_append:
            LFP_SW_channels = np.append(LFP_SW_channels, chan)
    if '221220_3' in os.getcwd():
        to_plot_2_corr = [4,5,6,7,8] # noisy background high-frequency afterwards
        chans_to_append = [57,8,23]
        chans_to_append = [i for i in chans_to_append if i not in LFP_SW_channels]
        for chan in chans_to_append:
            LFP_SW_channels = np.append(LFP_SW_channels, chan)

    auto_outlier_stims_indices = pickle.load(open('auto_outlier_stims_indices','rb'))
    os.chdir('..')
    
         
    # how long to exlude from stim
    exclude_before = 0.2
    exclude_after = 1
    
    ON_start_spike_rate = [[] for i in range(10)]
    ON_stop_spike_rate = [[] for i in range(10)]

    OFF_ON_transitions = [[] for i in range(10)]
    ON_OFF_transitions = [[] for i in range(10)]



    for ind_sweep, LFP in enumerate(LFP_all_sweeps):   
        if day == '221220_3' and ind_sweep == 9: # noisy high frequency background
            continue
        
        
        spiking = [list(spikes_allsweeps[ind_sweep].values())[ind] for ind in range(64) if ind in LFP_SW_channels]
        all_chan_spiking_histogram = np.zeros([len(LFP_SW_channels), LFP.shape[1]])
        for chan in range(len(spiking)):
            all_chan_spiking_histogram[chan, spiking[chan].astype(int)] = 1
            # print(np.sum(all_chan_spiking_histogram[chan,:]))
        
        #smooth over OFF_period_duration (i.e. will be nonzero if spikes during that period)
        all_chan_spiking_histogram_smoothed = smooth(all_chan_spiking_histogram, OFF_duration_threshold, axis = 1)
        # find periods where > 90% of channels are silent for the defined amount of time: OFF periods
        population_offs = np.sum(all_chan_spiking_histogram_smoothed == 0, axis = 0)
        OFF_periods = population_offs > int(channel_proportion_off*len(spiking))
        # fig, ax = plt.subplots()
        # ax.plot(OFF_periods)
        # ax.plot(np.mean(LFP, axis = 0)/-np.min(np.mean(LFP, axis = 0)) + 1, linewidth = 0.5)
        # ax.set_ylim([0,2])
        
        ON_states_start_all = np.where(np.diff(OFF_periods.astype(int)) < 0)[0]
        OFF_states_start_all = np.where(np.diff(OFF_periods.astype(int)) > 0)[0]     
        # make sure you start with ON_start and finish with OFF start
        OFF_states_start_all = OFF_states_start_all[OFF_states_start_all>ON_states_start_all[0]]
        ON_states_start_all = ON_states_start_all[ON_states_start_all<OFF_states_start_all[-1]]
        
        # ON states have to be longer than ON duration threshold
        mask = (OFF_states_start_all - ON_states_start_all) > ON_duration_threshold
        OFF_states_start_all = OFF_states_start_all[mask]
        ON_states_start_all = ON_states_start_all[mask]

        
        # # rolling average over window for minimum OFF duration. Do it looking backwards and forwards, a point is in a channel OFF is there is no spiking either XXms afterwards or XXms before
        # rolling_avg_window_back = np.concatenate([np.ones([OFF_duration_threshold]), np.zeros([OFF_duration_threshold])])
        # rolling_avg_window_forward = np.concatenate([np.zeros([OFF_duration_threshold-1]), np.ones([OFF_duration_threshold])])
        # all_chan_spiking_histogram_smoothed_back = np.apply_along_axis(lambda m: np.convolve(m, rolling_avg_window_back, mode='same'), axis = 1, arr = all_chan_spiking_histogram)
        # all_chan_spiking_histogram_smoothed_forward = np.apply_along_axis(lambda m: np.convolve(m, rolling_avg_window_forward, mode='same'), axis = 1, arr = all_chan_spiking_histogram)
        # all_chan_spiking_histogram_smoothed = np.any(np.stack((all_chan_spiking_histogram_smoothed_back == 0, all_chan_spiking_histogram_smoothed_forward == 0)), axis = 0)
        # plt.scatter(np.arange(LFP.shape[1]), all_chan_spiking_histogram[0,:], marker = '|', color = 'k')
            
        
        # #define OFF starts and stops as no spiking at all during a certain duration:
        # all_chan_spiking = np.sort(np.concatenate([list(spikes_allsweeps[ind_sweep].values())[ind] for ind in range(64) if ind in LFP_SW_channels]))
        # OFF_states_start_all = all_chan_spiking[np.where(np.diff(all_chan_spiking) > OFF_duration_threshold)]
        # ON_states_start_all = all_chan_spiking[np.where(np.diff(all_chan_spiking) > OFF_duration_threshold)[0] + 1]           
        # OFF_states_start_all = OFF_states_start_all[1:]
        # ON_states_start_all = ON_states_start_all[:-1]
        
        # # at least 5 spikes during the ON state
        # mask = []
        # for ON_state_start, ON_state_stop in zip(ON_states_start_all, OFF_states_start_all):
        #     if len(all_chan_spiking[np.searchsorted(all_chan_spiking, ON_state_start):np.searchsorted(all_chan_spiking, ON_state_stop)]) < spike_minimum_on:
        #         mask.append(False)
        #     else:
        #         mask.append(True)
        # OFF_states_start_all = OFF_states_start_all[mask]
        # ON_states_start_all = ON_states_start_all[mask]
        
        # plot population OFF states across the array
        fig, ax = plt.subplots(5,1,figsize = (20,10)) # plot one minute at a time
        time_bins = [0,60000,120000,180000,240000,300000]
        for i, spikes in enumerate([list(spikes_allsweeps[ind_sweep].values())[ind] for ind in range(64) if ind in LFP_SW_channels]):
            for time_bin in range(5):
                curr_spikes = spikes[np.searchsorted(spikes, time_bins[time_bin]):np.searchsorted(spikes, time_bins[time_bin + 1])] - time_bins[time_bin]
                ax.flatten()[time_bin].plot(curr_spikes, i/1.5 * np.ones_like(curr_spikes), 'k.', '.', markersize = .025)
                
                curr_stims = stim_times[ind_sweep][np.searchsorted(stim_times[ind_sweep], time_bins[time_bin]):np.searchsorted(stim_times[ind_sweep], time_bins[time_bin + 1])] - time_bins[time_bin]
                for stim in curr_stims:
                    ax.flatten()[time_bin].axvspan(stim + exclude_after*1000, stim + (5 - exclude_before*1000), color='blue', alpha=0.15)

                if np.searchsorted(OFF_states_start_all, time_bins[time_bin]) != np.searchsorted(OFF_states_start_all, time_bins[time_bin + 1]):
                    curr_OFF_states_start_all = OFF_states_start_all[np.searchsorted(OFF_states_start_all, time_bins[time_bin]):np.searchsorted(OFF_states_start_all, time_bins[time_bin + 1])] - time_bins[time_bin]
                    curr_ON_states_start_all = ON_states_start_all[np.searchsorted(ON_states_start_all, time_bins[time_bin]):np.searchsorted(ON_states_start_all, time_bins[time_bin + 1])] - time_bins[time_bin]
                    curr_ON_states_start_all = curr_ON_states_start_all[curr_ON_states_start_all > curr_OFF_states_start_all[0]]
                    # if curr_ON_states_start_all[0] < curr_OFF_states_start_all[0]:
                    for OFF_start, OFF_stop in zip(curr_OFF_states_start_all, curr_ON_states_start_all):
                        ax.flatten()[time_bin].axvspan(OFF_start, OFF_stop, color='red', alpha=0.01)
                        if np.sum(np.asarray([(stim + exclude_after*1000 + safety_buffer) < OFF_start < (stim + (5000 - exclude_before*1000)) for stim in curr_stims])) == 1:
                            ax.flatten()[time_bin].axvline(OFF_start, color = 'purple', linewidth = 0.05)
                        if np.sum(np.asarray([(stim + exclude_after*1000) < OFF_stop < (stim + (5000 - exclude_before*1000 - safety_buffer)) for stim in curr_stims])) == 1:
                            ax.flatten()[time_bin].axvline(OFF_stop, color = 'green', linewidth = 0.05)                
                ax.flatten()[time_bin].set_xticks(np.linspace(time_bins[time_bin], time_bins[time_bin+1], 61) - time_bins[time_bin])
                ax.flatten()[time_bin].set_xticklabels(list(map(str,(np.linspace(time_bins[time_bin], time_bins[time_bin+1], 61)/1000).astype(int))), size = 5, rotation = 45)
                ax.flatten()[time_bin].set_yticks(np.arange(len(LFP_SW_channels)))
                ax.flatten()[time_bin].set_yticklabels(list(map(str,LFP_SW_channels.astype(int))), size = 4.5)
                ax.flatten()[time_bin].set_xlim([0,60000])
        plt.tight_layout()
        plt.savefig(f'MUA OFF periods sweep {ind_sweep + 1}', dpi = 1500)
        cl()
        
        # calculate transition latencies and spike rates
        # makes sure to start with an OFF_ON transition, makes life easier
        # OFF_states_start_all = OFF_states_start_all[OFF_states_start_all > ON_states_start_all[0]]
        
        OFF_ON_total = 0
        ON_OFF_total = 0
        
        # at least half the channels have to spike within transition period
        spikes_during_transition_cutoff = len(spiking)/1.5

        for OFF_ON, ON_OFF in zip(ON_states_start_all, OFF_states_start_all):
            on_spikes = [i[np.searchsorted(i, OFF_ON) : np.searchsorted(i, ON_OFF)] for i in spiking]
            # on_spikes_concat = np.sort(np.concatenate(on_spikes) - np.min(np.concatenate(on_spikes))) # all MUA from all channels collapsed, starting at 0 (first spike time = 0)
            
            # calculate the std of first and last spikes of the UP states
            first_spikes = np.asarray([i[0]for i in on_spikes if len(i) != 0]) # fist spike of each channel
            last_spikes = np.asarray([i[-1]for i in on_spikes if len(i) != 0]) # last spike of each channel
            
            # exclude spikes that aren't within the transition period defined above (i.e. if have a super long UP state for some reason don't count spike from a second earlier from low-spiking channel)
            first_spikes = first_spikes[first_spikes - np.min(first_spikes) < transition_window]
            last_spikes = last_spikes[np.max(last_spikes) - last_spikes < transition_window]
            
            # only count that transition metric if far enough from whisker stims and enough spikes during transition period
            if len(first_spikes) > spikes_during_transition_cutoff and np.sum(np.asarray([(stim + exclude_after*1000) < OFF_ON < (stim + (5000 - exclude_before*1000 - safety_buffer)) for stim in stim_times[ind_sweep]])) == 1:
                OFF_ON_transition = np.nanstd(first_spikes)
                OFF_ON_transitions[ind_sweep].append(OFF_ON_transition)
                OFF_ON_total+=1
            if len(last_spikes) > spikes_during_transition_cutoff and np.sum(np.asarray([(stim + exclude_after*1000 + safety_buffer) < ON_OFF < (stim + (5000 - exclude_before*1000)) for stim in stim_times[ind_sweep]])) == 1:
                ON_OFF_transition = np.nanstd(last_spikes)
                ON_OFF_transitions[ind_sweep].append(ON_OFF_transition)
                ON_OFF_total+=1
        print(f'{OFF_ON_total}, {ON_OFF_total}')
    


    # plot std of transition latency 
    fig, ax = plt.subplots(2,1, sharey = True)
    fig.suptitle(f'{day}')
    ax[0].plot([np.nanmedian(i) for i in ON_OFF_transitions])
    ax[0].set_title('ON_OFF')
    ax[1].plot([np.nanmedian(i) for i in OFF_ON_transitions])
    ax[1].set_title('OFF_ON')
    plt.tight_layout
    plt.savefig('State transition synchrony', dpi = 1000)
    # cl()
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    pickle.dump(OFF_ON_transitions, open('OFF_ON_transitions', 'wb'))
    pickle.dump(ON_OFF_transitions, open('ON_OFF_transitions', 'wb'))
    
    os.chdir('..')
    os.chdir('..')




#%% Synchrony group analysis

os.chdir(os.path.join(overall_path, r'UP_pairing'))
ON_OFF_transitions_ALL_mean = []
ON_OFF_transitions_ALL_median = []
OFF_ON_transitions_ALL_mean = []
OFF_ON_transitions_ALL_median = []
delta_power_auto_outliers_rel_ALL = np.loadtxt('delta_power_auto_outliers_rel_ALL.csv', delimiter = ',')
delta_power_auto_outliers_rel_change_ALLCHANS = np.loadtxt('delta_power_auto_outliers_rel_change_ALLCHANS.csv', delimiter = ',')
delta_change_LFP_group = []

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for day_ind, day in enumerate(days):
    
    # only days used for LFP cross correlation and STTC?
    if day == '160624_B2' or day == '160628_D1' or day == '191121' or day == '061221' or day == '160218':
        continue
    # if day == '160624_B2':
    #     continue
    
    os.chdir(day)
    print(day)   
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    if os.path.exists('OFF_ON_transitions'):
        OFF_ON_transitions = pickle.load(open('OFF_ON_transitions', 'rb'))
        ON_OFF_transitions = pickle.load(open('ON_OFF_transitions', 'rb'))
    
        ON_OFF_transitions_ALL_mean.append([np.nanmean(i) for i in ON_OFF_transitions])
        ON_OFF_transitions_ALL_median.append([np.nanmedian(i) for i in ON_OFF_transitions])
        OFF_ON_transitions_ALL_mean.append([np.nanmean(i) for i in OFF_ON_transitions])
        OFF_ON_transitions_ALL_median.append([np.nanmedian(i) for i in OFF_ON_transitions])
    
        LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
        delta_change_LFP = np.mean(np.loadtxt('delta_power_auto_outliers_rel_change.csv', delimiter = ',')[LFP_resp_channels_cutoff])
        # delta_change_LFP = np.mean(delta_power_auto_outliers_rel_change_ALLCHANS[day_ind*64 + LFP_resp_channels_cutoff])
        delta_change_LFP_group.append(delta_change_LFP)

    os.chdir('..')
    os.chdir('..')

ON_OFF_transitions_ALL_mean = np.asarray(ON_OFF_transitions_ALL_mean).T
ON_OFF_transitions_ALL_median = np.asarray(ON_OFF_transitions_ALL_median).T
OFF_ON_transitions_ALL_mean = np.asarray(OFF_ON_transitions_ALL_mean).T
OFF_ON_transitions_ALL_median = np.asarray(OFF_ON_transitions_ALL_median).T

ON_OFF_transitions_ALL_mean_rel = ON_OFF_transitions_ALL_mean/np.mean(ON_OFF_transitions_ALL_mean[[0,1,2,3,],:], axis = 0)
ON_OFF_transitions_ALL_median_rel = ON_OFF_transitions_ALL_median/np.mean(ON_OFF_transitions_ALL_median[[0,1,2,3,],:], axis = 0)
OFF_ON_transitions_ALL_mean_rel = OFF_ON_transitions_ALL_mean/np.mean(OFF_ON_transitions_ALL_mean[[0,1,2,3,],:], axis = 0)
OFF_ON_transitions_ALL_median_rel = OFF_ON_transitions_ALL_median/np.mean(OFF_ON_transitions_ALL_median[[0,1,2,3,],:], axis = 0)

# ON_OFF
measure_to_plot = ON_OFF_transitions_ALL_mean
fig, ax = plt.subplots(figsize = (3,4.5))
ax.plot([np.repeat(1,measure_to_plot.shape[1]), np.repeat(2,measure_to_plot.shape[1])], [np.nanmean(measure_to_plot[[0,1,2,3],:], axis = 0), np.nanmean(measure_to_plot[[4,5,6,7,8,9],:], axis = 0)], color = 'k', linewidth = 1)
ax.scatter(np.concatenate([np.repeat(1,measure_to_plot.shape[1]), np.repeat(2,measure_to_plot.shape[1])]), np.concatenate([np.nanmean(measure_to_plot[[0,1,2,3],:], axis = 0), np.nanmean(measure_to_plot[[4,5,6,7,8,9],:], axis = 0)]), color = 'k')
ax.set_xticks([1,2])
ax.set_xticklabels(['before \n UP-pairing', 'after \n UP-pairing'], size = 14)
ax.set_xlim([0.75, 2.25])
ax.set_yticks([90,100,110,120])
ax.set_yticklabels(list(map(str, ax.get_yticks())), size = 14)
ax.set_ylabel('latency standard deviation (ms)', size = 14)
plt.tight_layout()
# plt.savefig('ON_OFF transition before vs after.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('ON_OFF transition before vs after.pdf', dpi = 1000, format = 'pdf')
# fig, ax = plt.subplots()
# ax.boxplot([np.nanmean(measure_to_plot[[0,1,2,3],:], axis = 0), np.nanmean(measure_to_plot[[4,5,6,7,8,9],:], axis = 0)])
t, p = scipy.stats.ttest_rel(np.nanmean(measure_to_plot[[0,1,2,3],:], axis = 0), np.nanmean(measure_to_plot[[4,5,6,7,8,9],:], axis = 0))
print(t, p)


#OFF_ON
measure_to_plot = OFF_ON_transitions_ALL_mean
fig, ax = plt.subplots(figsize = (3,4.5))
ax.plot([np.repeat(1,measure_to_plot.shape[1]), np.repeat(2,measure_to_plot.shape[1])], [np.nanmean(measure_to_plot[[0,1,2,3],:], axis = 0), np.nanmean(measure_to_plot[[4,5,6,7,8,9],:], axis = 0)], color = 'k', linewidth = 1)
ax.scatter(np.concatenate([np.repeat(1,measure_to_plot.shape[1]), np.repeat(2,measure_to_plot.shape[1])]), np.concatenate([np.nanmean(measure_to_plot[[0,1,2,3],:], axis = 0), np.nanmean(measure_to_plot[[4,5,6,7,8,9],:], axis = 0)]), color = 'k')
ax.set_xticks([1,2])
ax.set_xticklabels(['before \n UP-pairing', 'after \n UP-pairing'], size = 14)
ax.set_xlim([0.75, 2.25])
ax.set_yticks([80,100,120])
ax.set_yticklabels(list(map(str, ax.get_yticks())), size = 14)
ax.set_ylabel('latency standard deviation (ms)', size = 14)
plt.tight_layout()
# plt.savefig('OFF_ON transition before vs after.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('OFF_ON transition before vs after.pdf', dpi = 1000, format = 'pdf')
# fig, ax = plt.subplots()
# ax.boxplot([np.nanmean(measure_to_plot[[0,1,2,3],:], axis = 0), np.nanmean(measure_to_plot[[4,5,6,7,8,9],:], axis = 0)])
t, p = scipy.stats.ttest_rel(np.nanmean(measure_to_plot[[0,1,2,3],:], axis = 0), np.nanmean(measure_to_plot[[4,5,6,7,8,9],:], axis = 0))
print(t, p)



#%% does that desynchronization correlate with delta power change? NO
to_correlate = ON_OFF_transitions_ALL_mean
X = delta_change_LFP_group
Y = np.nanmean(to_correlate[[4,5,6,7,8,9],:], axis = 0)/np.nanmean(to_correlate[[0,1,2,3],:], axis = 0)
fig, ax = plt.subplots()
slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
print(f'{r**2} and {p} for {len(X)} channels')
ax.scatter(X,Y, color = 'k')
ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
ax.set_xlabel('delta power change (% baseline)', size = 16)
ax.set_ylabel('LFP synchronization change (% baseline)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()



#%% Example MUA Synchrony for thesis (160427)

LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))

try:
    spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}_cleaned','rb'))
except FileNotFoundError:
    try:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    except FileNotFoundError:
        print(f'spikes with highpass {highpass_cutoff} not found')
        spikes_allsweeps = pickle.load(open([i for i in os.listdir() if 'spikes_allsweeps' in i][0],'rb'))

stim_times = pickle.load(open('stim_times','rb'))
os.chdir([i for i in os.listdir() if 'analysis' in i][0])
LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
LFP_resp_channels_cutoff =  np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',')
os.chdir('..')

#4 24 770
sweep = 4
xlim1 = stim_times[sweep][24] + 770
xlim2 = xlim1 + 890

i = 1
fig, ax = plt.subplots(figsize = (3,5))
for chan, value in enumerate(list(spikes_allsweeps[sweep].values())):
    if chan in SW_spiking_channels:
        ax.plot(value, i * np.ones_like(value), 'k|', markersize = 4)
        ax.set_xlim(xlim1,xlim2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        i += 1 
plt.savefig('ON state for transition example.jpg', dpi = 1000, format = 'jpg')
plt.savefig('ON state for transition example.pdf', dpi = 1000, format = 'pdf')




