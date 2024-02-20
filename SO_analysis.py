# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 20:33:00 2023

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



#%% SLOW WAVES extracting

zero_mean = False

 # -------------------------------------------------------------------------------------- SW --------------------------------------------------------------------------------------------
# extract slow waves. You want waveform, firstamp, secondamp, firstslope, secondslope and duration for every sweep. as a list because number chagnes for every sweep and channel
#each list here is another 64 lists one for each channel, with the 
# os.chdir(home_directory)
lfp_cutoff_resp_channels = 200
to_plot_1_LFP = [0,1,2,3]
to_plot_2_LFP = [4,5,6,7,8,9]  
to_plot_1_SW = [0,1,2,3]
to_plot_2_SW = [4,5,6,7,8,9]  

UP_std_cutoff = 1.75

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for day in days:
# for day in ['160310']:

    if day == '160310':
        highpass_cutoff = 8
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
    auto_outlier_stims_indices = pickle.load(open('auto_outlier_stims_indices','rb'))

    spont_spiking = np.zeros([10,64])

    SW_waveform_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))] # lowpass 2
    SW_waveform_sweeps_4 = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))] # lowpass 4
    SW_spiking_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    Peak_dur_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    SW_fslope_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    SW_sslope_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    SW_famp_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    SW_samp_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    SW_peaks_sweeps_up = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    SW_peaks_sweeps_down = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    UP_Cross_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    
    
    # mean value within sweep
    SW_frequency_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64])
    SW_waveform_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64, 1000])
    SW_spiking_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64, 1000])
    Peak_dur_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64])
    SW_fslope_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64])
    SW_sslope_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64])
    SW_famp_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64])
    SW_samp_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64])
    SW_peaks_sweeps_avg_up = np.zeros([len(LFP_all_sweeps), 64])
    SW_peaks_sweeps_avg_down = np.zeros([len(LFP_all_sweeps), 64])
    SW_frequency_sweeps_avg[:] = np.NaN
    SW_waveform_sweeps_avg[:] = np.NaN
    SW_spiking_sweeps_avg[:] = np.NaN
    Peak_dur_sweeps_avg[:] = np.NaN
    SW_fslope_sweeps_avg[:] = np.NaN
    SW_sslope_sweeps_avg[:] = np.NaN
    SW_famp_sweeps_avg[:] = np.NaN
    SW_samp_sweeps_avg[:] = np.NaN
    SW_peaks_sweeps_avg_up[:] = np.NaN
    SW_peaks_sweeps_avg_down[:] = np.NaN

    # median value within sweep
    SW_waveform_sweeps_median = np.zeros([len(LFP_all_sweeps), 64, 1000])
    SW_spiking_sweeps_median = np.zeros([len(LFP_all_sweeps), 64, 1000])
    Peak_dur_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
    SW_fslope_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
    SW_sslope_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
    SW_famp_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
    SW_samp_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
    SW_peaks_sweeps_median_up = np.zeros([len(LFP_all_sweeps), 64])
    SW_peaks_sweeps_median_down = np.zeros([len(LFP_all_sweeps), 64])
    SW_waveform_sweeps_median[:] = np.NaN
    SW_spiking_sweeps_median[:] = np.NaN
    Peak_dur_sweeps_median[:] = np.NaN
    SW_fslope_sweeps_median[:] = np.NaN
    SW_sslope_sweeps_median[:] = np.NaN
    SW_famp_sweeps_median[:] = np.NaN
    SW_samp_sweeps_median[:] = np.NaN
    SW_peaks_sweeps_median_up[:] = np.NaN
    SW_peaks_sweeps_median_down[:] = np.NaN

    exclude_before = 0.1
    exclude_after = 1.4
    duration_criteria = 100 # how long must the UP state be?
    
    # filter in slow wave range, then find every time it goes under threshold x SD
    for ind_sweep, LFP in enumerate(LFP_all_sweeps):
        if LFP_all_sweeps[ind_sweep].size == 0:
            continue
        LFP_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP.astype('float32')), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 2*pq.Hz).as_array()
        LFP_filt_4 = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP.astype('float32')), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 4*pq.Hz).as_array()

        for ind_stim, stim in enumerate(list(stim_times[ind_sweep])):
            if ind_stim == 0 or ind_stim == len(stim_times[ind_sweep]) - 1:
                continue
            
        # for ind_stim, stim in enumerate(list(stims_for_delta[ind_sweep][1:-1])):
        #     if stim == 0:
        #         continue

            print(ind_sweep, ind_stim)
            curr_LFP_filt_total = LFP_filt[int(stim):int(stim + 5*new_fs), :]
            curr_LFP_filt_total_4 = LFP_filt_4[int(stim):int(stim + 5*new_fs), :]
            curr_LFP_filt = LFP_filt[int(stim + exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs), :]
            curr_LFP_filt_4 = LFP_filt_4[int(stim + exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs), :]

            if zero_mean:
                curr_LFP_filt = curr_LFP_filt - np.mean(curr_LFP_filt, axis = 0) # zero mean

            for chan in range(64):
                
                # if detected as outlier in delta power analysis
                if ind_stim in auto_outlier_stims_indices[ind_sweep][chan]:
                    continue

                # because spiking is saved as dict of channels need to convert it to list to be able to access channels
                chan_spiking = list(spikes_allsweeps[ind_sweep].values())[chan]
                curr_spike_number = np.diff(np.searchsorted(chan_spiking, [stim + exclude_after*new_fs, stim + (5 - exclude_before)*new_fs]))[0]
                #add number of spikes during that time for spontaneous spiking
                spont_spiking[ind_sweep,chan] = spont_spiking[ind_sweep,chan] + np.diff(np.searchsorted(chan_spiking, [stim + exclude_after*new_fs, stim + (5 - exclude_before)*new_fs]))[0]
                
                
                # print(chan)
                DOWN_Cross = np.where(np.diff((curr_LFP_filt[:,chan] < 0).astype(int)) == 1)[0]
                UP_Cross = np.where(np.diff((curr_LFP_filt[:,chan] < 0).astype(int)) == -1)[0]
                
                if DOWN_Cross.size == 0:
                    continue
                
                #if no Down crossing before or after:
                UP_Cross = np.delete(UP_Cross, UP_Cross < DOWN_Cross[0])
                UP_Cross = np.delete(UP_Cross, UP_Cross > DOWN_Cross[-1])
                
                # If upwards crossing too close to end of trial (need to be able to take out 500ms after for waveform)
                UP_Cross = np.delete(UP_Cross, UP_Cross > (4.5-exclude_before-exclude_after)*new_fs)
                UP_Cross = np.delete(UP_Cross, UP_Cross < (0.5*new_fs))

                if UP_Cross.size == 0:
                    continue
                
                UP_LFP = np.where(curr_LFP_filt[:,chan] < -UP_std_cutoff*np.std(curr_LFP_filt[:,chan]))[0]
                
                # If no UP crossing after
                UP_LFP = np.delete(UP_LFP, UP_LFP > UP_Cross[-1])
                
                # only LFP points under threshold within 250ms of a UP Crossing
                for i in range(len(UP_LFP)):
                   diff_to_crossing = UP_Cross - UP_LFP[i]
                   if min(diff_to_crossing[diff_to_crossing > 0]) > 249:
                       UP_LFP[i] = 0
                UP_LFP = np.delete(UP_LFP, UP_LFP == 0)
                
                #if no DOWN crossing before
                UP_LFP = np.delete(UP_LFP, UP_LFP < DOWN_Cross[0])
                
                #take out continuous numbers, so just left with first one that fits all the criteria
                UP_LFP = np.delete(UP_LFP, np.where(np.diff(UP_LFP) == 1)[0] + 1)
    
                if UP_LFP.size == 0:
                    continue
                
                DOWN_Cross_before = []
                UP_Cross_after = []
                DOWN_Cross_after = []
                
                #find the Crossings before and after. Here also apply the duration criteria: Down before and UP after must be separated by at least 100ms (so 100ms UP state duration)
                for i in range(len(UP_LFP)):
                    idx_down = np.argmin(np.abs(UP_LFP[i] - DOWN_Cross))
                    idx_up = np.argmin(np.abs(UP_LFP[i] - UP_Cross))
                    
                    if DOWN_Cross[idx_down] < UP_LFP [i]:
                        curr_DOWN_Cross_before = DOWN_Cross[idx_down]
                        curr_DOWN_Cross_after = DOWN_Cross[idx_down + 1]
    
                    elif DOWN_Cross[idx_down] > UP_LFP [i]:
                        curr_DOWN_Cross_before = DOWN_Cross[idx_down - 1]
                        curr_DOWN_Cross_after = DOWN_Cross[idx_down]
                        
                    if UP_Cross[idx_up] > UP_LFP[i]:
                        curr_UP_Cross_after = UP_Cross[idx_up]
                    elif UP_Cross[idx_up] < UP_LFP[i]:
                        curr_UP_Cross_after = UP_Cross[idx_up + 1]
                    
                    # duration criteria:
                    if curr_UP_Cross_after - curr_DOWN_Cross_before < duration_criteria:
                        continue
                    else:
                        DOWN_Cross_before.append(curr_DOWN_Cross_before)
                        DOWN_Cross_after.append(curr_DOWN_Cross_after)
                        UP_Cross_after.append(curr_UP_Cross_after)
                    
                    
                    # save UP_Cross_after in list of lists to get the spiking with the slow wave. remember UP_cross after is indexed with a 0.5s offset from stim start.
                    UP_Cross_sweeps[ind_sweep][chan].append(UP_Cross_after[i] + int(stim + exclude_after*new_fs))
                    
                    #peak duration
                    Peak_dur_sweeps[ind_sweep][chan].append(DOWN_Cross_after[i] - DOWN_Cross_before[i])
                    
                    #save filtered LFP
                    SW_waveform_sweeps[ind_sweep][chan].append(curr_LFP_filt[int(UP_Cross_after[i] - 0.5*new_fs) : int(UP_Cross_after[i] + 0.5*new_fs), chan])
                    SW_waveform_sweeps_4[ind_sweep][chan].append(curr_LFP_filt_4[int(UP_Cross_after[i] - 0.5*new_fs) : int(UP_Cross_after[i] + 0.5*new_fs), chan])

                    #how many peaks
                    SW_peaks_sweeps_up[ind_sweep][chan].append(len(scipy.signal.find_peaks(-curr_LFP_filt_4[int(DOWN_Cross_before[i]) : int(UP_Cross_after[i]), chan])[0]))
                    SW_peaks_sweeps_down[ind_sweep][chan].append(len(scipy.signal.find_peaks(curr_LFP_filt_4[int(UP_Cross_after[i]) : int(DOWN_Cross_after[i]), chan])[0]))

                    #save spiking (as 1ms bins)
                    temp_spiking = np.zeros(1000)
                    # set all spikes there as 1. So take out spikes within 500ms of UP crossing, then subtract 500ms before UP crossing to start at 0
                    temp_spiking[np.round(chan_spiking[np.logical_and(int(UP_Cross_after[i] + exclude_after*new_fs + stim - 0.5*new_fs) < chan_spiking, int(UP_Cross_after[i] + exclude_after*new_fs + stim + 0.5*new_fs) > chan_spiking)] - int(UP_Cross_after[i] + exclude_after*new_fs + stim - 0.5*new_fs) - 1).astype(int)] = 1
                    SW_spiking_sweeps[ind_sweep][chan].append(temp_spiking)
                    
                    idx_peak = np.argmax(curr_LFP_filt[UP_Cross_after[i]:DOWN_Cross_after[i],chan])
                    idx_trough = np.argmin(curr_LFP_filt[DOWN_Cross_before[i]:UP_Cross_after[i],chan])

                    SW_fslope_sweeps[ind_sweep][chan].append(np.mean(np.diff(curr_LFP_filt[DOWN_Cross_before[i]:DOWN_Cross_before[i] + idx_trough, chan])))
                    SW_sslope_sweeps[ind_sweep][chan].append(np.mean(np.diff(curr_LFP_filt[DOWN_Cross_before[i] + idx_trough:UP_Cross_after[i]+idx_peak, chan])))
                    
                    SW_famp_sweeps[ind_sweep][chan].append(np.abs(min(curr_LFP_filt[DOWN_Cross_before[i]:UP_Cross_after[i],chan])))
                    SW_samp_sweeps[ind_sweep][chan].append(np.abs(max(curr_LFP_filt[UP_Cross_after[i]:DOWN_Cross_after[i],chan])))
        
        #convert spontaneous spiking in Hz by dividing by seconds (number of non-outlier stims x inter stimulus interval)
        for chan in range(64):
            spont_spiking[ind_sweep,chan] = spont_spiking[ind_sweep,chan]/((5 - exclude_before - exclude_after)*(len(stim_times[ind_sweep]) - 2 - len(auto_outlier_stims_indices[ind_sweep][chan])))
        
        
    np.save('spont_spiking.npy', spont_spiking)
    
    # median and mean values over whole sweep.
    for ind_sweep in range(len(LFP_all_sweeps)):
        if LFP_all_sweeps[ind_sweep].size == 0:
            continue
        for chan in range(64):
            SW_frequency_sweeps_avg[ind_sweep,chan] = len(Peak_dur_sweeps[ind_sweep][chan])/(len(stim_times[ind_sweep]) - 2 - len(auto_outlier_stims_indices[ind_sweep][chan])) # -2 because exclude first and last stim

            SW_waveform_sweeps_avg[ind_sweep,chan,:] = np.mean(np.asarray(SW_waveform_sweeps[ind_sweep][chan]), axis = 0)
            SW_spiking_sweeps_avg[ind_sweep,chan,:] = np.mean(np.asarray(SW_spiking_sweeps[ind_sweep][chan]), axis = 0)
            Peak_dur_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(Peak_dur_sweeps[ind_sweep][chan]))
            SW_fslope_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_fslope_sweeps[ind_sweep][chan]))
            SW_sslope_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_sslope_sweeps[ind_sweep][chan]))
            SW_famp_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_famp_sweeps[ind_sweep][chan]))
            SW_samp_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_samp_sweeps[ind_sweep][chan]))
            SW_peaks_sweeps_avg_up[ind_sweep,chan] = np.mean(np.asarray(SW_peaks_sweeps_up[ind_sweep][chan]))
            SW_peaks_sweeps_avg_down[ind_sweep,chan] = np.mean(np.asarray(SW_peaks_sweeps_down[ind_sweep][chan]))

            SW_waveform_sweeps_median[ind_sweep,chan,:] = np.median(np.asarray(SW_waveform_sweeps[ind_sweep][chan]), axis = 0)
            SW_spiking_sweeps_median[ind_sweep,chan,:] = np.median(np.asarray(SW_spiking_sweeps[ind_sweep][chan]), axis = 0)
            Peak_dur_sweeps_median[ind_sweep,chan] = np.median(np.asarray(Peak_dur_sweeps[ind_sweep][chan]))
            SW_fslope_sweeps_median[ind_sweep,chan] = np.median(np.asarray(SW_fslope_sweeps[ind_sweep][chan]))
            SW_sslope_sweeps_median[ind_sweep,chan] = np.median(np.asarray(SW_sslope_sweeps[ind_sweep][chan]))
            SW_famp_sweeps_median[ind_sweep,chan] = np.median(np.asarray(SW_famp_sweeps[ind_sweep][chan]))
            SW_samp_sweeps_median[ind_sweep,chan] = np.median(np.asarray(SW_samp_sweeps[ind_sweep][chan]))
            SW_peaks_sweeps_median_up[ind_sweep,chan] = np.median(np.asarray(SW_peaks_sweeps_up[ind_sweep][chan]))
            SW_peaks_sweeps_median_down[ind_sweep,chan] = np.median(np.asarray(SW_peaks_sweeps_down[ind_sweep][chan]))

    
    pickle.dump(UP_Cross_sweeps, open('UP_Cross_sweeps', 'wb')) # UP cross times
    pickle.dump(SW_waveform_sweeps, open('SW_waveform_sweeps', 'wb'))
    pickle.dump(SW_waveform_sweeps_4, open('SW_waveform_sweeps_4', 'wb'))
    pickle.dump(Peak_dur_sweeps, open('Peak_dur_sweeps', 'wb'))
    pickle.dump(SW_spiking_sweeps, open('SW_spiking_sweeps', 'wb'))
    pickle.dump(SW_fslope_sweeps, open('SW_fslope_sweeps', 'wb'))
    pickle.dump(SW_sslope_sweeps, open('SW_sslope_sweeps', 'wb'))
    pickle.dump(SW_famp_sweeps, open('SW_famp_sweeps', 'wb'))
    pickle.dump(SW_samp_sweeps, open('SW_samp_sweeps', 'wb'))      
    pickle.dump(SW_peaks_sweeps_up, open('SW_peaks_sweeps_up', 'wb'))      
    pickle.dump(SW_peaks_sweeps_down, open('SW_peaks_sweeps_down', 'wb'))      

    # save mean and median across all SW in a sweep in each channel and sweep
    np.save('SW_waveform_sweeps_avg.npy', SW_waveform_sweeps_avg)
    np.save('SW_frequency_sweeps_avg.npy', SW_frequency_sweeps_avg)
    np.save('SW_spiking_sweeps_avg.npy', SW_spiking_sweeps_avg)
    np.save('Peak_dur_sweeps_avg.npy', Peak_dur_sweeps_avg)
    np.save('SW_fslope_sweeps_avg.npy', SW_fslope_sweeps_avg)
    np.save('SW_sslope_sweeps_avg.npy', SW_sslope_sweeps_avg)
    np.save('SW_famp_sweeps_avg.npy', SW_famp_sweeps_avg)
    np.save('SW_samp_sweeps_avg.npy', SW_samp_sweeps_avg)
    np.save('SW_peaks_sweeps_avg_up.npy', SW_peaks_sweeps_avg_up)
    np.save('SW_peaks_sweeps_avg_down.npy', SW_peaks_sweeps_avg_down)

    np.save('SW_waveform_sweeps_median.npy', SW_waveform_sweeps_median)
    np.save('SW_spiking_sweeps_median.npy', SW_spiking_sweeps_median)
    np.save('Peak_dur_sweeps_median.npy', Peak_dur_sweeps_median)
    np.save('SW_fslope_sweeps_median.npy', SW_fslope_sweeps_median)
    np.save('SW_sslope_sweeps_median.npy', SW_sslope_sweeps_median)
    np.save('SW_famp_sweeps_median.npy', SW_famp_sweeps_median)
    np.save('SW_samp_sweeps_median.npy', SW_samp_sweeps_median)
    np.save('SW_peaks_sweeps_median_up.npy', SW_peaks_sweeps_median_up)
    np.save('SW_peaks_sweeps_median_down.npy', SW_peaks_sweeps_median_down)

    #relative change in individual params before vs after in all channels, of mean and median across all SW in a sweep
    Freq_change = (np.mean(SW_frequency_sweeps_avg[to_plot_2_SW,:], axis = 0) - np.mean(SW_frequency_sweeps_avg[to_plot_1_SW,:], axis = 0))/np.mean(SW_frequency_sweeps_avg[to_plot_1_SW,:], axis = 0)
    
    Peak_dur_change_mean = (np.mean(Peak_dur_sweeps_avg[to_plot_2_SW,:], axis = 0) - np.mean(Peak_dur_sweeps_avg[to_plot_1_SW,:], axis = 0))/np.mean(Peak_dur_sweeps_avg[to_plot_1_SW,:], axis = 0)
    Fslope_change_mean = (np.mean(SW_fslope_sweeps_avg[to_plot_2_SW,:], axis = 0) - np.mean(SW_fslope_sweeps_avg[to_plot_1_SW,:], axis = 0))/np.mean(SW_fslope_sweeps_avg[to_plot_1_SW,:], axis = 0)
    Sslope_change_mean = (np.mean(SW_sslope_sweeps_avg[to_plot_2_SW,:], axis = 0) - np.mean(SW_sslope_sweeps_avg[to_plot_1_SW,:], axis = 0))/np.mean(SW_sslope_sweeps_avg[to_plot_1_SW,:], axis = 0)
    Famp_change_mean = (np.mean(SW_famp_sweeps_avg[to_plot_2_SW,:], axis = 0) - np.mean(SW_famp_sweeps_avg[to_plot_1_SW,:], axis = 0))/np.mean(SW_famp_sweeps_avg[to_plot_1_SW,:], axis = 0)
    Samp_change_mean = (np.mean(SW_samp_sweeps_avg[to_plot_2_SW,:], axis = 0) - np.mean(SW_samp_sweeps_avg[to_plot_1_SW,:], axis = 0))/np.mean(SW_samp_sweeps_avg[to_plot_1_SW,:], axis = 0)
    Peaks_up_change_mean = (np.mean(SW_peaks_sweeps_avg_up[to_plot_2_SW,:], axis = 0) - np.mean(SW_peaks_sweeps_avg_up[to_plot_1_SW,:], axis = 0))/np.mean(SW_peaks_sweeps_avg_up[to_plot_1_SW,:], axis = 0)
    Peaks_down_change_mean = (np.mean(SW_peaks_sweeps_avg_down[to_plot_2_SW,:], axis = 0) - np.mean(SW_peaks_sweeps_avg_down[to_plot_1_SW,:], axis = 0))/np.mean(SW_peaks_sweeps_avg_down[to_plot_1_SW,:], axis = 0)

    Peak_dur_change_median = (np.mean(Peak_dur_sweeps_median[to_plot_2_SW,:], axis = 0) - np.mean(Peak_dur_sweeps_median[to_plot_1_SW,:], axis = 0))/np.mean(Peak_dur_sweeps_median[to_plot_1_SW,:], axis = 0)
    Fslope_change_median = (np.mean(SW_fslope_sweeps_median[to_plot_2_SW,:], axis = 0) - np.mean(SW_fslope_sweeps_median[to_plot_1_SW,:], axis = 0))/np.mean(SW_fslope_sweeps_median[to_plot_1_SW,:], axis = 0)
    Sslope_change_median = (np.mean(SW_sslope_sweeps_median[to_plot_2_SW,:], axis = 0) - np.mean(SW_sslope_sweeps_median[to_plot_1_SW,:], axis = 0))/np.mean(SW_sslope_sweeps_median[to_plot_1_SW,:], axis = 0)
    Famp_change_median = (np.mean(SW_famp_sweeps_median[to_plot_2_SW,:], axis = 0) - np.mean(SW_famp_sweeps_median[to_plot_1_SW,:], axis = 0))/np.mean(SW_famp_sweeps_median[to_plot_1_SW,:], axis = 0)
    Samp_change_median = (np.mean(SW_samp_sweeps_median[to_plot_2_SW,:], axis = 0) - np.mean(SW_samp_sweeps_median[to_plot_1_SW,:], axis = 0))/np.mean(SW_samp_sweeps_median[to_plot_1_SW,:], axis = 0)
    Peaks_up_change_median = (np.mean(SW_peaks_sweeps_median_up[to_plot_2_SW,:], axis = 0) - np.mean(SW_peaks_sweeps_median_up[to_plot_1_SW,:], axis = 0))/np.mean(SW_peaks_sweeps_median_up[to_plot_1_SW,:], axis = 0)
    Peaks_down_change_median = (np.mean(SW_peaks_sweeps_median_down[to_plot_2_SW,:], axis = 0) - np.mean(SW_peaks_sweeps_median_down[to_plot_1_SW,:], axis = 0))/np.mean(SW_peaks_sweeps_median_down[to_plot_1_SW,:], axis = 0)

    np.savetxt('Freq_change.csv', Freq_change, delimiter = ',')
    
    np.savetxt('Peak_dur_change_mean.csv', Peak_dur_change_mean, delimiter = ',')
    np.savetxt('Fslope_change_mean.csv', Fslope_change_mean, delimiter = ',')
    np.savetxt('Sslope_change_mean.csv', Sslope_change_mean, delimiter = ',')
    np.savetxt('Famp_change_mean.csv', Famp_change_mean, delimiter = ',')
    np.savetxt('Samp_change_mean.csv', Samp_change_mean, delimiter = ',')
    np.savetxt('Peaks_up_change_mean.csv', Peaks_up_change_mean, delimiter = ',')
    np.savetxt('Peaks_down_change_mean.csv', Peaks_down_change_mean, delimiter = ',')

    np.savetxt('Peak_dur_change_median.csv', Peak_dur_change_median, delimiter = ',')
    np.savetxt('Fslope_change_median.csv', Fslope_change_median, delimiter = ',')
    np.savetxt('Sslope_change_median.csv', Sslope_change_median, delimiter = ',')
    np.savetxt('Famp_change_median.csv', Famp_change_median, delimiter = ',')
    np.savetxt('Samp_change_median.csv', Samp_change_median, delimiter = ',')
    np.savetxt('Peaks_up_change_median.csv', Peaks_up_change_median, delimiter = ',')
    np.savetxt('Peaks_down_change_median.csv', Peaks_down_change_median, delimiter = ',')



    # SW param values on the mean waveforms:
    Peak_dur_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), 64])
    SW_fslope_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), 64])
    SW_sslope_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), 64])
    SW_famp_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), 64])
    SW_samp_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), 64])
    
    Peak_dur_sweeps_avg_overall[:] = np.NaN
    SW_fslope_sweeps_avg_overall[:] = np.NaN
    SW_sslope_sweeps_avg_overall[:] = np.NaN
    SW_famp_sweeps_avg_overall[:] = np.NaN
    SW_samp_sweeps_avg_overall[:] = np.NaN
    
    for ind_sweep in range(len(LFP_all_sweeps)):
        if LFP_all_sweeps[ind_sweep].size == 0:
            continue
        for chan in range(64):
            Peak_dur_sweeps_avg_overall[ind_sweep,chan] = np.argmax(SW_waveform_sweeps_avg[ind_sweep,chan,:]) - np.argmin(SW_waveform_sweeps_avg[ind_sweep,chan,:])
            SW_fslope_sweeps_avg_overall[ind_sweep,chan] = np.nanmean(np.diff(SW_waveform_sweeps_avg[ind_sweep,chan,250:np.argmin(SW_waveform_sweeps_avg[ind_sweep,chan,:])]))
            SW_sslope_sweeps_avg_overall[ind_sweep,chan] = np.nanmean(np.diff(SW_waveform_sweeps_avg[ind_sweep,chan,np.argmin(SW_waveform_sweeps_avg[ind_sweep,chan,:500]):np.argmax(SW_waveform_sweeps_avg[ind_sweep,chan,500:])+500]))
            SW_famp_sweeps_avg_overall[ind_sweep,chan] = np.min(SW_waveform_sweeps_avg[ind_sweep,chan,:])
            SW_samp_sweeps_avg_overall[ind_sweep,chan] = np.max(SW_waveform_sweeps_avg[ind_sweep,chan,:])
    
    np.save('Peak_dur_sweeps_avg_overall.npy', Peak_dur_sweeps_avg_overall)
    np.save('SW_fslope_sweeps_avg_overall.npy', SW_fslope_sweeps_avg_overall)
    np.save('SW_sslope_sweeps_avg_overall.npy', SW_sslope_sweeps_avg_overall)
    np.save('SW_famp_sweeps_avg_overall.npy', SW_famp_sweeps_avg_overall)
    np.save('SW_samp_sweeps_avg_overall.npy', SW_samp_sweeps_avg_overall)
    
    # relative changes in overall slow wave
    Peak_dur_overall_change = (np.nanmean(np.abs(Peak_dur_sweeps_avg_overall[to_plot_2_SW,:]), axis = 0) - np.nanmean(np.abs(Peak_dur_sweeps_avg_overall[to_plot_1_SW,:]), axis = 0))/np.nanmean(np.abs(Peak_dur_sweeps_avg_overall[to_plot_1_SW,:]), axis = 0)
    Fslope_overall_change = (np.nanmean(SW_fslope_sweeps_avg_overall[to_plot_2_SW,:], axis = 0) - np.nanmean(SW_fslope_sweeps_avg_overall[to_plot_1_SW,:], axis = 0))/np.nanmean(SW_fslope_sweeps_avg_overall[to_plot_1_SW,:], axis = 0)
    Sslope_overall_change = (np.nanmean(SW_sslope_sweeps_avg_overall[to_plot_2_SW,:], axis = 0) - np.nanmean(SW_sslope_sweeps_avg_overall[to_plot_1_SW,:], axis = 0))/np.nanmean(SW_sslope_sweeps_avg_overall[to_plot_1_SW,:], axis = 0)
    Famp_overall_change = (np.nanmean(SW_famp_sweeps_avg_overall[to_plot_2_SW,:], axis = 0) - np.nanmean(SW_famp_sweeps_avg_overall[to_plot_1_SW,:], axis = 0))/np.nanmean(SW_famp_sweeps_avg_overall[to_plot_1_SW,:], axis = 0)
    Samp_overall_change = (np.nanmean(SW_samp_sweeps_avg_overall[to_plot_2_SW,:], axis = 0) - np.nanmean(SW_samp_sweeps_avg_overall[to_plot_1_SW,:], axis = 0))/np.nanmean(SW_samp_sweeps_avg_overall[to_plot_1_SW,:], axis = 0)
        
    np.savetxt('Peak_dur_overall_change.csv', Peak_dur_overall_change, delimiter = ',')
    np.savetxt('Fslope_overall_change.csv', Fslope_overall_change, delimiter = ',')
    np.savetxt('Sslope_overall_change.csv', Sslope_overall_change, delimiter = ',')
    np.savetxt('Famp_overall_change.csv', Famp_overall_change, delimiter = ',')
    np.savetxt('Samp_overall_change.csv', Samp_overall_change, delimiter = ',')




    # get unique SW UP cross times across LFP responsive channels
    tolerance = 500
    unique_times = []
    for ind_sweep in range(len(LFP_all_sweeps)):
        all_times = np.sort(np.concatenate([np.asarray(UP_Cross_sweeps[ind_sweep][i]) for i in LFP_resp_channels_cutoff]))
        unique_times.append(list(all_times[~(np.triu(np.abs(all_times[:,None] - all_times) <= tolerance,1)).any(0)]))  # now you have only the unique SW times with a tolerance of x msec
    pickle.dump(unique_times, open(f'unique_times_tol_{tolerance}_chans_{lfp_cutoff_resp_channels}', 'wb')) # UP cross times

    # SW occurrences: matrix of true or false if that channel has a SW at that time
    SW_occurrence_cutoff = 500 #how many ms for a SW to be considered co-occurrent
    
    SW_occurence_list = [[] for j in range(len(LFP_all_sweeps))]
    # SW_peak_list = [[] for j in range(len(LFP_all_sweeps))]
    # SW_onset_list = [[] for j in range(len(LFP_all_sweeps))]
    
    for ind_sweep, LFP in enumerate(LFP_all_sweeps):
        print(f'MATRIX {ind_sweep}')
        LFP_delta_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 2*pq.Hz).as_array()
        
        for SW_ind, SW in enumerate(unique_times[ind_sweep]):
        
            occurrence_matrix = np.zeros([8,8])
            # peak_matrix = np.zeros([8,8])
            # peak_matrix[:] = np.NaN
            # start_matrix = np.zeros([8,8])
            # start_matrix[:] = np.NaN
            for chan in range(64):
                if len(UP_Cross_sweeps[ind_sweep][chan]) == 0: 
                    continue
                else:
                    # boolean with true or false if this channel has a SW then.
                    occurrence_matrix[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]] = np.min(np.abs(np.asarray(UP_Cross_sweeps[ind_sweep][chan]) - SW)) < SW_occurrence_cutoff
                    # time of UP peak occurrence, disregard as I recalculate it below correctly
                    # peak_matrix[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]] = np.argmin(LFP_delta_filt[SW - 500:SW + 500, chan])
            
            # SW_peak_list[ind_sweep].append(peak_matrix)
            SW_occurence_list[ind_sweep].append(occurrence_matrix)
    
    pickle.dump(SW_occurence_list, open('SW_occurence_list', 'wb'))
    # pickle.dump(SW_peak_list, open('SW_peak_list', 'wb'))
    
    os.chdir('..')
    
    
    # # PLOT INDIVIDUAL SLOW WAVES
    # for sweep in range(10):
    #     fig, ax = plt.subplots(8,8, figsize = (15,12))
    #     fig.suptitle(f'Slow Waves sweep {sweep + 1}')
    #     for ind, ax1 in enumerate(list(ax.flatten())):  
    #         ax1.tick_params(axis='both', which='minor', labelsize=4)
    #         ax1.tick_params(axis='both', which='major', labelsize=4)
    #         ax1.set_xticks([])
    #         if chanMap[ind] in LFP_resp_channels_cutoff:                     
    #             ax1.plot(np.asarray(SW_waveform_sweeps[sweep][chanMap[ind]]).T, linewidth = 0.45)
    #             ax1.set_title(str(chanMap[ind]), size = 5)
    #     plt.tight_layout()
    #     plt.savefig(f'Slow Waves sweep {sweep + 1}', dpi = 1000)
    #     cl()
        
    # # PLOT INDIVIDUAL SLOW WAVES lowpass 4
    # for sweep in range(10):
    #     fig, ax = plt.subplots(8,8, figsize = (15,12))
    #     fig.suptitle(f'Slow Waves sweep {sweep + 1}')
    #     for ind, ax1 in enumerate(list(ax.flatten())):  
    #         ax1.tick_params(axis='both', which='minor', labelsize=4)
    #         ax1.tick_params(axis='both', which='major', labelsize=4)
    #         ax1.set_xticks([])
    #         if chanMap[ind] in LFP_resp_channels_cutoff:                     
    #             ax1.plot(np.asarray(SW_waveform_sweeps_4[sweep][chanMap[ind]]).T, linewidth = 0.45)
    #             ax1.set_title(str(chanMap[ind]), size = 5)
    #     plt.tight_layout()
    #     plt.savefig(f'Slow Waves sweep lowpass 4 {sweep + 1}', dpi = 1000)
    #     cl()

    # # changes of parameters, median within each sweep
    # fig, ax = plt.subplots(8,8)
    # fig.suptitle('MEDIAN: Freq, Dur, Fslope, Sslope, Famp, Samp')
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     chan = chanMap[ind]
    #     ax1.bar(range(6), [Freq_change[chan], Peak_dur_change_median[chan], Fslope_change_median[chan], Sslope_change_median[chan], Famp_change_median[chan], Samp_change_median[chan]])
    #     ax1.set_title(str(chan), size = 5)
    #     ax1.set_yticklabels([])
    #     ax1.set_ylim([-1,1])
    #     if chan in LFP_resp_channels_cutoff:
    #         ax1.set_facecolor("y")
    # plt.savefig(f'Slow waves params median', dpi = 1000)

    # # median change across channels of median parameter within each sweep
    # fig, ax = plt.subplots()
    # fig.suptitle('MEDIAN of medians: Freq, Dur, Fslope, Sslope, Famp, Samp')
    # ax.bar(range(6), list(map(np.median, [Freq_change[LFP_resp_channels_cutoff], Peak_dur_change_median[LFP_resp_channels_cutoff], Fslope_change_median[LFP_resp_channels_cutoff], Sslope_change_median[LFP_resp_channels_cutoff], Famp_change_median[LFP_resp_channels_cutoff], Samp_change_median[LFP_resp_channels_cutoff]])))
    # ax1.set_ylim([-1,1])
    # plt.savefig('Slow waves params median MEDIAN', dpi = 1000)


    # # multipeak waves
    # fig, ax = plt.subplots(8,8)
    # fig.suptitle('average peaks UP before and after, DOWN before and after')
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     chan = chanMap[ind]
    #     ax1.bar(range(4), [np.mean(SW_peaks_sweeps_avg_up[[0,1,2,3],chan], axis= 0), np.mean(SW_peaks_sweeps_avg_up[[4,5,6,7,8,9],chan], axis= 0), np.mean(SW_peaks_sweeps_avg_down[[0,1,2,3],chan], axis= 0), np.mean(SW_peaks_sweeps_avg_down[[4,5,6,7,8,9],chan], axis= 0)])
    #     ax1.set_title(str(chan), size = 5)
    #     ax1.set_yticklabels([])
    #     # ax1.set_ylim([-1,1])
    #     if chan in LFP_resp_channels_cutoff:
    #         ax1.set_facecolor("y")
    # plt.savefig(f'Slow waves average peaks', dpi = 1000)

    os.chdir('..')







#%%
# #% SLOW WAVES plotting.
# # os.chdir(home_directory)
# lfp_cutoff_resp_channels = 200

# days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# # for day in days:
# for day in ['160310']:
    
    os.chdir(day)
    print(day)
    
    # LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    # try:
    #     spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    # except FileNotFoundError:
    #     print(f'spikes with highpass {highpass_cutoff} not found')
    #     spikes_allsweeps = pickle.load(open('spikes_allsweeps','rb'))
    # stim_times = pickle.load(open('stim_times','rb'))

    # if os.path.exists('stims_for_delta'):
    #     stims_for_delta = pickle.load(open('stims_for_delta','rb'))
    # else:
    #     stims_for_delta = copy.deepcopy(stim_times)
    #     # stims_for_delta[5][50:] = 0    
    #     pickle.dump(stims_for_delta, open('stims_for_delta','wb'))

    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    to_plot_1_SW = [0,1,2,3]
    to_plot_2_SW = [4,5,6,7,8,9]    
    LFP_resp_channels_cutoff =  np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_SW, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_SW,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
      
    SW_spiking_sweeps_avg = np.load('SW_spiking_sweeps_avg.npy')
    SW_waveform_sweeps_avg = np.load('SW_waveform_sweeps_avg.npy')
    
    SW_frequency_sweeps_avg = np.load('SW_frequency_sweeps_avg.npy')
    Peak_dur_sweeps_avg = np.load('Peak_dur_sweeps_avg.npy')
    SW_fslope_sweeps_avg = np.load('SW_fslope_sweeps_avg.npy')
    SW_sslope_sweeps_avg = np.load('SW_sslope_sweeps_avg.npy')
    SW_famp_sweeps_avg = np.load('SW_famp_sweeps_avg.npy')
    SW_samp_sweeps_avg = np.load('SW_samp_sweeps_avg.npy')
    
    Peak_dur_sweeps_avg_overall = np.load('Peak_dur_sweeps_avg_overall.npy')
    SW_fslope_sweeps_avg_overall = np.load('SW_fslope_sweeps_avg_overall.npy')
    SW_sslope_sweeps_avg_overall = np.load('SW_sslope_sweeps_avg_overall.npy')
    SW_famp_sweeps_avg_overall = np.load('SW_famp_sweeps_avg_overall.npy')
    SW_samp_sweeps_avg_overall = np.load('SW_samp_sweeps_avg_overall.npy')
    try:
        SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',')
    except OSError:
        SW_spiking_channels = np.loadtxt('PSTH_resp_channels.csv', delimiter = ',')
        pass
    os.chdir('..')
    
    
    #average waveforms before vs after on same axis
    fig, ax = plt.subplots(8,8, sharey = True)
    fig.suptitle(f'before vs after {day}')
    for chan in range(64):
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(np.mean(SW_waveform_sweeps_avg[to_plot_1_SW,chan,:], axis = 0), 'b', linewidth = 1)
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(np.mean(SW_waveform_sweeps_avg[to_plot_2_SW,chan,:], axis = 0), 'r', linewidth = 1)
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 4)
        # ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].axvline(x = 3)
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_yticklabels([])
        if chan in LFP_resp_channels_cutoff:
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
    plt.savefig(f'SW {to_plot_1_SW} vs {to_plot_2_SW}', dpi = 1000)
    


    # #check waveforms for every sweep individually to check for outliers.
    # # first baseline
    # fig, ax = plt.subplots(8,8)
    # fig.suptitle('before')
    # for chan in range(64):
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[0,chan,:], 'b', label = '1')
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[1,chan,:], 'r', label = '2')
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[2,chan,:], 'y', label = '3')
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[3,chan,:], 'c', label = '4')
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_yticklabels([])
    #     # if chan in LFP_resp_channels_cutoff:
    #     #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
    # plt.legend()
    
    # # check after pairing every sweep
    # fig, ax = plt.subplots(8,8)
    # fig.suptitle('after')
    # for chan in range(64):
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[4,chan,:], 'b', label = '5')
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[5,chan,:], 'r', label = '6')
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[6,chan,:], 'y', label = '7')
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[7,chan,:], 'c', label = '8')
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[8,chan,:], 'k', label = '9')
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[9,chan,:], 'm', label = '10')
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
    # #     if chan in LFP_resp_channels_cutoff:
    # #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
    # plt.legend()
    
    
    #     if chan in LFP_resp_channels:
    #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
            
    # # SW_params_channels = list(np.linspace(0,63,64, dtype = int))
    
    # #average change in slow wave characteristics in channels LFP channels
    # fig, ax = plt.subplots()
    # fig.suptitle('LFP resp channels')
    # ax.bar(range(6), list(map(np.nanmean, [Freq_change[LFP_resp_channels], Peak_dur_overall_change[LFP_resp_channels], Fslope_overall_change[LFP_resp_channels], Sslope_overall_change[LFP_resp_channels], Famp_overall_change[LFP_resp_channels], Samp_overall_change[LFP_resp_channels]])))
    # ax.set_xticks([0,1,2,3,4,5])
    # ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'])
    

    
    # #average change in slow wave characteristics in SW channels
    # fig, ax = plt.subplots()
    # fig.suptitle('SW resp channels MEDIAN')
    # ax.bar(np.linspace(0,5,6), list(map(np.nanmedian, [Freq_change[SW_params_channels], Peak_dur_overall_change[SW_params_channels], Fslope_overall_change[SW_params_channels], Sslope_overall_change[SW_params_channels], Famp_overall_change[SW_params_channels], Samp_overall_change[SW_params_channels]])))
    # ax.set_xticks([0,1,2,3,4,5])
    # ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'])
    # plt.savefig('SW resp channels median', dpi = 1000)
    
    # # individual params of slow waves over time                                    
    # SW_feature_to_plot = SW_famp_sweeps_avg
    # fig, ax = plt.subplots(8,8)
    # for chan in range(64):
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_feature_to_plot[:,chan])
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].axvline(x = 3)
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_yticklabels([])
    
    
    
    
    
    
    # SW_spiking_channels = PSTH_resp_channels
    # UP
    if day == '121121':
        SW_spiking_channels = [30,46,31,47,1,49,28,44,29,45,3,51,42,45,5,53,4,24,40,25,6,22,38,20,21,59,10,18,34,61,12,16,63,14,62]
    if day == '160310':
        SW_spiking_channels = [31,47,1,49,0,48,28,45,3,51,50,43,5,53,52,7,6,56]
    if day == '160414_D1':
        SW_spiking_channels = [45,43,53,7,55,39,57,37,10,58,35,13,61,12,60,15,63,14,62]
    if day == '160426_D1':
        SW_spiking_channels = [25,22,38,23,39,9,57,21,37,11,18,35,13,61,16,32,17,33,15,63,14,62]
    if day == '160519_B2':
        SW_spiking_channels = [29,45,43,53,4,24,40,25,41,7,55,22,38,23,39,9,57,8,56,20,36,21,37,11,10,58,18,19,35,13,61,60,16,32,33,15,63,62]
    if day == '160624_B2':
        SW_spiking_channels = [41,23,39,9,21,37,19,35,13,61,17,33,15]
    if day == '160628_D1':
        SW_spiking_channels = [29,45,27,43,4,25,41,23,39,21,37,11,10,19,35,13,61,17,33,15]
    if day == '191121':
        SW_spiking_channels = [46,47,44,3,5,24,22,8,20,10,34,16,14]
    if day == '201121':
        SW_spiking_channels = [30,31,47,1,49,48,28,3,43,5,4,24,22,23,20,21,59,10,18,34,35,61,12,32,17,33,63,14,62]
    if day == '221220_3':
        SW_spiking_channels = [30,46,31,47,1,49,48,28,44,29,45,3,51,2,26,42,27,43,53,4,24,25,55,22,39,20,36,18,34,19,16,32,17]
    if day == '281021':
        SW_spiking_channels = [30,46,31,47,1,49,28,44,29,45,3,51,26,42,27,43,5,53,4,24,40,25,7,55,6,22,38,23,9,8,20,36,21,37,10,18,34,35,13,61,12,16,32,17,33,63,14,62]
    if day == '291021':
        SW_spiking_channels = [30,46,31,47,1,49,28,44,45,3,51,26,42,43,5,53,4,24,40,25,7,6,23,8,59,10,34,13,61,12,17,33,63,14,62]
    
    # DOWN
    if day == '061221':
        SW_spiking_channels = [46,47,49,28,3,4,24,6,22,20,10,34,35,61,12,32,63,14]
    if day == '160218':
        SW_spiking_channels = [31,47,49,29,51,7,39,9,57,21,37,11,59,19,35,13,61,12,17,63,14]
    if day == '160308':
        SW_spiking_channels = [49,29,51,25,41,7,55,6,23,39,9,57,21,37,11,59,10,19,35,13,61,12,17,33,15,63,14]
    if day == '160322':
        SW_spiking_channels = [30,46,47,1,49,0,48,28,44,29,45,3,51,2,50,26,42,27,43,5,53,4,52,24,40,25,41,7,55,6,22,38,23,39,9,57,8,20,36,21,37,11,59,10,58,18,34,19,35,13,61,12,60,16,32,63,14,62]
    if day == '160331':
        SW_spiking_channels = [46,1,0,29,45,3,27,43,5,4,25,41,7,55,6,22,23,57,8,20,21,37,11,59,10,18,34,19,13,61,12,16,32,63,14,62]
    if day == '160420':
        SW_spiking_channels = [29,45,43,24,40,25,41,22,38,23,39,20,36,21,37,18,19,35,16,32]
    if day == '160427':
        SW_spiking_channels = [49,48,29,45,51,2,50,43,53,4,52,24,25,7,55,22,38,23,39,57,8,20,36,21,37,11,59,10,58,18,34,19,35,13,61,12,16,32,17,33,15,63,14,62]
    if day == '221208':
        SW_spiking_channels = [30,46,31,47,1,49,0,44,29,45,3,51,2,50,27,43,4,52,25,41,54,20,36,21,18,34,19,35,13,16,32,17,33,15,63]
    if day == '221212':
        SW_spiking_channels = [47,1,49,0,29,45,51,2,27,43,4,24,40,41,7,55,6,54,22,9,20,19,15,63]
    if day == '221213':
        SW_spiking_channels = [46,31,47,1,49,48,44,29,45,3,51,2,50,42,27,43,5,53,4,24,40,25,41,7,55,6,22,38,23,39,9,57,8,56,20,36,21,37,59,10,18,34,19,35,13,61,16,32,17,33,15,63,14]
    if day == '221216':
        SW_spiking_channels = [30,46,31,47,1,49,28,44,29,45,3,51,2,27,43,5,53,4,40,25,41,7,55,38,23,39,56,20,36,21,13,15,63]
    if day == '221219_1':
        SW_spiking_channels = [30,46,47,1,49,44,45,3,51,2,42,27,43,5,52,24,40,25,41,7,6,54,22,22,38,23,39,9,20,36,21,37,19,16,32,17,33,15,63,14]

    #slow-wave evoked spiking before vs after on same axis
    # for day in [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]:
    #     os.chdir(day)
    #     os.chdir([i for i in os.listdir() if 'analysis' in i][0])
        
    # to_plot_1_SW = [0,1,2,3]
    # to_plot_2_SW = list(np.linspace(4,len(LFP_all_sweeps) - 1, len(LFP_all_sweeps) - 4, dtype = int))
    
    #     # os.chdir(home_directory)
    #     # os.chdir(f'analysis_{day}')
    #     # SW_spiking_sweeps_avg = np.load('SW_spiking_sweeps_avg.npy')
    
    # # SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',')
    fig, ax = plt.subplots(8,8, sharey = True)
    for chan in range(64):
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(np.nanmean(SW_spiking_sweeps_avg[to_plot_1_SW,chan,:], axis = 0),8), 'b', linewidth = .5)
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(np.nanmean(SW_spiking_sweeps_avg[to_plot_2_SW,chan,:], axis = 0),8), 'r', linewidth = .5)
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 4)
        # ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].axvline(x = 3)
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_yticklabels([])
        if chan in SW_spiking_channels:
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")        
    plt.savefig(f'SW Spiking {to_plot_1_SW} vs {to_plot_2_SW}', dpi = 1000)
    
    # #change in slow-wave evoked spiking? do peak, area under the curve
    # #overall change in all channels:
    # fig, ax = plt.subplots()
    # ax.plot(np.mean(np.nanmean(SW_spiking_sweeps_avg[to_plot_1_SW,:,:], axis = 0), axis = 0), 'b')
    # ax.plot(np.mean(np.nanmean(SW_spiking_sweeps_avg[to_plot_2_SW,:,:], axis = 0), axis = 0), 'r')
    
    SW_spiking_peak_change = (np.max(smooth(np.nanmean(SW_spiking_sweeps_avg[to_plot_2_SW,:], axis = 0), 25, axis = 1), axis = 1) - np.max(smooth(np.nanmean(SW_spiking_sweeps_avg[to_plot_1_SW,:], axis = 0), 25, axis = 1), axis = 1))/np.max(smooth(np.nanmean(SW_spiking_sweeps_avg[to_plot_1_SW,:], axis = 0), 25, axis = 1), axis = 1)
    SW_spiking_area_change = (np.sum(smooth(np.nanmean(SW_spiking_sweeps_avg[to_plot_2_SW,:], axis = 0), 25, axis = 1), axis = 1) - np.sum(smooth(np.nanmean(SW_spiking_sweeps_avg[to_plot_1_SW,:], axis = 0), 25, axis = 1), axis = 1))/np.sum(smooth(np.nanmean(SW_spiking_sweeps_avg[to_plot_1_SW,:], axis = 0), 25, axis = 1), axis = 1)
    
    np.savetxt('SW_spiking_peak_change.csv', SW_spiking_peak_change, delimiter = ',')
    np.savetxt('SW_spiking_area_change.csv', SW_spiking_area_change, delimiter = ',')
    
    # os.chdir('..')
    
    # fig, ax = plt.subplots()
    # fig.suptitle('relative change in spiking peak')
    # im = ax.imshow(np.reshape(SW_spiking_peak_change[chanMap], (8, 8)), cmap = 'jet', vmin = -1, vmax = 1)
    # fig.colorbar(im)
    
    # fig, ax = plt.subplots()
    # fig.suptitle('relative change in spiking area')
    # im = ax.imshow(np.reshape(SW_spiking_area_change[chanMap], (8, 8)), cmap = 'jet', vmin = -1, vmax = 1)
    # fig.colorbar(im)


    # os.chdir('121121')
    # os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    # SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',')
    # SW_spiking_channels = np.append(SW_spiking_channels, [43])
    #%
    # os.chdir(home_directory)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    np.savetxt('SW_spiking_channels.csv', SW_spiking_channels, delimiter = ',')
    
    # np.savetxt('Freq_change.csv', Freq_change, delimiter = ',')
    # np.savetxt('Peak_dur_change.csv', Peak_dur_change, delimiter = ',')
    # np.savetxt('Fslope_change.csv', Fslope_change, delimiter = ',')
    # np.savetxt('Sslope_change.csv', Sslope_change, delimiter = ',')
    # np.savetxt('Famp_change.csv', Famp_change, delimiter = ',')
    # np.savetxt('Samp_change.csv', Samp_change, delimiter = ',')
    
    # np.savetxt('Peak_dur_overall_change.csv', Peak_dur_overall_change, delimiter = ',')
    # np.savetxt('Fslope_overall_change.csv', Fslope_overall_change, delimiter = ',')
    # np.savetxt('Sslope_overall_change.csv', Sslope_overall_change, delimiter = ',')
    # np.savetxt('Famp_overall_change.csv', Famp_overall_change, delimiter = ',')
    # np.savetxt('Samp_overall_change.csv', Samp_overall_change, delimiter = ',')
    
    # np.savetxt('SW_spiking_peak_change.csv', SW_spiking_peak_change, delimiter = ',')
    # np.savetxt('SW_spiking_area_change.csv', SW_spiking_area_change, delimiter = ',')
    
    # np.savetxt('to_plot_1_SW.csv', to_plot_1_SW, delimiter = ',')
    # np.savetxt('to_plot_2_SW.csv', to_plot_2_SW, delimiter = ',')

    os.chdir('..')
    
    
    #average waveforms for each channel before and after
    fig, ax = plt.subplots()
    for chan in LFP_resp_channels_cutoff:
        ax.plot(np.mean(SW_waveform_sweeps_avg[to_plot_1_SW,chan,:], axis = 0))
        ax.set_ylim([-600,600])
        ax.set_yticks([-500,0,500])
        ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
        ax.set_ylabel('Amplitude (yV)', size = 16)
        ax.set_xticks([0,500,1000])
        ax.set_xticklabels(list(map(str,[-500,0,500])), size = 16)
        ax.set_xlabel('time from SW onset (ms)', size = 16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
    plt.savefig('SW before.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('SW before.jpg', dpi = 1000, format = 'jpg')
    
    fig, ax = plt.subplots()
    for chan in LFP_resp_channels_cutoff:
        ax.plot(np.mean(SW_waveform_sweeps_avg[to_plot_2_SW,chan,:], axis = 0))
        ax.set_ylim([-600,600])
        ax.set_yticks([-500,0,500])
        ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
        ax.set_ylabel('Amplitude (yV)', size = 16)
        ax.set_xticks([0,500,1000])
        ax.set_xticklabels(list(map(str,[-500,0,500])), size = 16)
        ax.set_xlabel('time from SW onset (ms)', size = 16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
    plt.savefig('SW after.pdf', format = 'pdf')
    plt.savefig('SW after.jpg', format = 'jpg')
    
    
    #average SO-locked spiking across channels, before and after
    patch = 2 # how many times standard deviation
    # SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',', dtype = int)
    #average waveforms before vs after on same axis
    fig, ax = plt.subplots()
    to_plot_before = np.mean(SW_spiking_sweeps_avg[to_plot_1_SW,:,:], axis = 0)*1000
    ax.plot(np.mean(to_plot_before[SW_spiking_channels,:], axis = 0), color = 'k') # average across channels
    ax.fill_between(list(range(SW_spiking_sweeps_avg.shape[2])), np.mean(to_plot_before[SW_spiking_channels,:], axis = 0) + patch*np.nanstd(to_plot_before[SW_spiking_channels,:], axis = 0), np.mean(to_plot_before[SW_spiking_channels,:], axis = 0) - patch*np.nanstd(to_plot_before[SW_spiking_channels,:], axis = 0), alpha = 0.1, color = 'k')
    to_plot_after = np.mean(SW_spiking_sweeps_avg[to_plot_2_SW,:,:], axis = 0)*1000
    ax.plot(np.mean(to_plot_after[SW_spiking_channels,:], axis = 0), color = 'r')
    ax.fill_between(list(range(SW_spiking_sweeps_avg.shape[2])), np.mean(to_plot_after[SW_spiking_channels,:], axis = 0) + patch*np.nanstd(to_plot_after[SW_spiking_channels,:], axis = 0), np.mean(to_plot_after[SW_spiking_channels,:], axis = 0) - patch*np.nanstd(to_plot_after[SW_spiking_channels,:], axis = 0), alpha = 0.1, color = 'r')
    ax.set_ylim([0,80])
    ax.set_yticks([0,20,40])
    ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
    ax.set_ylabel('Spike rate (Hz)', size = 16)
    ax.set_xticks([0,500,1000])
    ax.set_xticklabels(list(map(str,[-500,0,500])), size = 16)
    ax.set_xlabel('time from SW onset (ms)', size = 16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('SW spiking before vs after.pdf', format = 'pdf')
    plt.savefig('SW spiking before vs after.jpg', format = 'jpg')
    
    # fig, ax = plt.subplots()
    # to_plot_after = np.mean(SW_spiking_sweeps_avg[to_plot_2_SW,:,:], axis = 0)*1000
    # ax.plot(np.mean(to_plot_after[SW_spiking_channels,:], axis = 0), color = 'r')
    # ax.fill_between(list(range(SW_spiking_sweeps_avg.shape[2])), np.mean(to_plot_after[SW_spiking_channels,:], axis = 0) + patch*np.nanstd(to_plot_after[SW_spiking_channels,:], axis = 0), np.mean(to_plot_after[SW_spiking_channels,:], axis = 0) - patch*np.nanstd(to_plot_after[SW_spiking_channels,:], axis = 0), alpha = 0.1, color = 'k')
    # ax.set_ylim([0,80])
    # ax.set_yticks([0,20,40])
    # ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
    # ax.set_ylabel('Spike rate (Hz)', size = 16)
    # ax.set_xticks([0,500,1000])
    # ax.set_xticklabels(list(map(str,[-500,0,500])), size = 16)
    # ax.set_xlabel('time from SW onset (ms)', size = 16)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tight_layout()
    # plt.savefig('SW spiking after.pdf', format = 'pdf')
    # plt.savefig('SW spiking after.jpg', format = 'jpg')

    
    
    os.chdir('..')
    
    cl()
    
    
    

#%% EXAMPLE slow waves before and after all chans for figure in just one mouse

to_plot_1_SW = [0,1,2,3]
to_plot_2_SW = [4,5,6,7,8,9]

os.chdir([i for i in os.listdir() if 'analysis' in i][0])
SW_waveform_sweeps_avg = np.load('SW_waveform_sweeps_avg.npy')
SW_waveform_sweeps_4 = pickle.load(open('SW_waveform_sweeps_4', 'rb'))
LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
SW_spiking_avg = np.load('SW_spiking_sweeps_avg.npy')
SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',', dtype = int)
os.chdir('..')


#average waveforms before on same axis for each channel
fig, ax = plt.subplots(figsize = (5,4))
ax.plot(np.mean(SW_waveform_sweeps_avg[to_plot_1_SW,:,:], axis = 0)[LFP_resp_channels_cutoff,:].T, linewidth = 0.5, alpha = 0.5)
ax.plot(np.mean(np.mean(SW_waveform_sweeps_avg[to_plot_1_SW,:,:], axis = 0)[LFP_resp_channels_cutoff,:].T,axis = 1), linewidth = 2, color = 'k')
ax.set_ylim([-600,600])
ax.set_yticks([-750,0,750])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
ax.set_ylabel('Filtered LFP (yV)', size = 16)
ax.set_xticks([0,500,1000])
ax.set_xticklabels(list(map(str,[-500,0,500])), size = 16)
ax.set_xlabel('time from zero-crossing (ms)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('SW before.pdf', dpi = 1000, format = 'pdf')
plt.savefig('SW before.jpg', dpi = 1000, format = 'jpg')

#average waveforms after on same axis
fig, ax = plt.subplots(figsize = (5,4))
ax.plot(np.mean(SW_waveform_sweeps_avg[to_plot_2_SW,:,:], axis = 0)[LFP_resp_channels_cutoff,:].T, linewidth = 0.5, alpha = 0.5)
ax.plot(np.mean(np.mean(SW_waveform_sweeps_avg[to_plot_2_SW,:,:], axis = 0)[LFP_resp_channels_cutoff,:].T,axis = 1), linewidth = 2, color = 'k')
ax.set_ylim([-600,600])
ax.set_yticks([-750,0,750])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
ax.set_ylabel('Filtered LFP (yV)', size = 16)
ax.set_xticks([0,500,1000])
ax.set_xticklabels(list(map(str,[-500,0,500])), size = 16)
ax.set_xlabel('time from zero-crossing (ms)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('SW after.pdf', format = 'pdf')
plt.savefig('SW after.jpg', format = 'jpg')




# example waveforms lowpass 4 for multipeak, here fro 160310
fig, ax = plt.subplots(figsize = (5,4))
ax.plot(np.asarray(SW_waveform_sweeps_4[6][LFP_resp_channels_cutoff[0]]).T[:,0], color = 'k')
ax.set_xlim([100,950])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig('SW multipeak example.pdf', dpi = 1000, format = 'pdf')
plt.savefig('SW multipeak example.jpg', dpi = 1000, format = 'jpg')

# lineObjects = ax.plot(np.asarray(SW_waveform_sweeps_4[6][LFP_resp_channels_cutoff[0]]).T, linewidth = 1, alpha = 0.5)
# plt.legend(iter(lineObjects), list(map(str, np.linspace(0,len(lineObjects)-1, len(lineObjects)))))# ax.plot(np.mean(np.mean(SW_waveform_sweeps_4[to_plot_1_SW,:,:], axis = 0)[LFP_resp_channels_cutoff,:].T,axis = 1), linewidth = 2, color = 'k')
# ax.set_ylim([-600,600])
# ax.set_yticks([-750,0,750])
# ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
# ax.set_ylabel('Filtered LFP (yV)', size = 16)
# ax.set_xticks([0,500,1000])
# ax.set_xticklabels(list(map(str,[-500,0,500])), size = 16)
# ax.set_xlabel('time from zero-crossing (ms)', size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.tight_layout()
# plt.savefig('SW before.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('SW before.jpg', dpi = 1000, format = 'jpg')

# fig, ax = plt.subplots(figsize = (5,4))
# ax.plot(np.asarray(SW_waveform_sweeps_4[1][LFP_resp_channels_cutoff[0]]).T, linewidth = 0.5, alpha = 0.5)
# ax.plot(np.mean(np.mean(SW_waveform_sweeps_4[to_plot_1_SW,:,:], axis = 0)[LFP_resp_channels_cutoff,:].T,axis = 1), linewidth = 2, color = 'k')
# ax.set_ylim([-600,600])
# ax.set_yticks([-750,0,750])
# ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
# ax.set_ylabel('Filtered LFP (yV)', size = 16)
# ax.set_xticks([0,500,1000])
# ax.set_xticklabels(list(map(str,[-500,0,500])), size = 16)
# ax.set_xlabel('time from zero-crossing (ms)', size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.tight_layout()
# plt.savefig('SW before.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('SW before.jpg', dpi = 1000, format = 'jpg')



# example SW spiking from 281021
patch = 1 # how many times standard deviation
chans_to_plot = np.intersect1d(SW_spiking_channels,LFP_resp_channels_cutoff)
#average waveforms before vs after on same axis
fig, ax = plt.subplots()
to_plot_before = np.mean(SW_spiking_avg[to_plot_1_SW,:,:], axis = 0)*1000
ax.plot(np.mean(np.mean(SW_waveform_sweeps_avg[to_plot_1_SW,:,:], axis = 0)[LFP_resp_channels_cutoff,:].T, axis = 1)/30 + 65, linewidth = 1, alpha = 1, color = 'k')
# ax.plot(scipy.ndimage.gaussian_filter(to_plot_before[SW_spiking_channels,:].T, (25,0)), linewidth = 0.5, alpha = 0.5)
ax.plot(np.mean(to_plot_before[chans_to_plot,:], axis = 0), color = 'k', linewidth = 0.75)
ax.fill_between(list(range(SW_spiking_avg.shape[2])), np.mean(to_plot_before[chans_to_plot,:], axis = 0) + patch*np.nanstd(to_plot_before[chans_to_plot,:], axis = 0), np.mean(to_plot_before[chans_to_plot,:], axis = 0) - patch*np.nanstd(to_plot_before[chans_to_plot,:], axis = 0), alpha = 0.1, color = 'k')
ax.set_ylim([0,90])
ax.set_yticks([0,20,40,60,80])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
ax.set_ylabel('Spike rate (Hz)', size = 16)
ax.set_xticks([0,500,1000])
ax.set_xticklabels(list(map(str,[-500,0,500])), size = 16)
ax.set_xlabel('time from UP-crossing (ms)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# plt.savefig('SW spiking before.pdf', format = 'pdf')
# plt.savefig('SW spiking before.jpg', format = 'jpg')
to_plot_after = np.mean(SW_spiking_avg[to_plot_2_SW,:,:], axis = 0)*1000
ax.plot(np.mean(to_plot_after[chans_to_plot,:], axis = 0), color = 'r', linewidth = 0.75)
ax.fill_between(list(range(SW_spiking_avg.shape[2])), np.mean(to_plot_after[chans_to_plot,:], axis = 0) + patch*np.nanstd(to_plot_after[chans_to_plot,:], axis = 0), np.mean(to_plot_after[chans_to_plot,:], axis = 0) - patch*np.nanstd(to_plot_after[chans_to_plot,:], axis = 0), alpha = 0.1, color = 'r')
# ax.set_ylim([0,80])
# ax.set_yticks([0,20,40])
# ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
# ax.set_ylabel('Spike rate (Hz)', size = 16)
# ax.set_xticks([0,500,1000])
# ax.set_xticklabels(list(map(str,[-500,0,500])), size = 16)
# ax.set_xlabel('time from SW onset (ms)', size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.tight_layout()
plt.savefig('SW spiking example before and after.pdf', format = 'pdf')
plt.savefig('SW spiking example before and after.jpg', format = 'jpg')





# #average SW waveform for slope and amplitude example
# fig, ax = plt.subplots(figsize = (8,3))
# ax.plot(np.mean(SW_waveform_sweeps_avg[1,LFP_resp_channels_cutoff,:], axis = 0), color = 'k')
# ax.set_ylim([-400,400])
# ax.set_yticks([-250,0,250])
# ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
# ax.set_ylabel('Amplitude (yV)', size = 16)
# ax.set_xticks([0,500,1000])
# ax.set_xticklabels(list(map(str,[-500,0,500])), size = 16)
# ax.set_xlabel('time from SW onset (ms)', size = 16)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.tight_layout()
# plt.savefig('SW example.pdf', format = 'pdf')
# plt.savefig('SW example.jpg', format = 'jpg')






