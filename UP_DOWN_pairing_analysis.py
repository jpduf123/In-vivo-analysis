# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:26:36 2022

@author: Mann Lab
"""

# pairing analysis and figures
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
    y_smooth = np.apply_along_axis(lambda m: np.convolve(m, box, mode='same'), axis = axis, arr = y)
    return y_smooth

def cl():
    plt.close('all')

chanMap_32 = np.array([31,32,30,33,27,36,26,37,16,47,18,38,17,46,29,39,19,45,28,40,20,44,22,41,21,34,24,42,23,43,25,35]) - 16
chanMap_16 = np.linspace(0, 15, 16).astype(int)

#coordinates of electrodes, to be given as a list of lists (in case of laminar 1 coord per point):
coordinates = [[i] for i in list(np.linspace(0, 1.55, 32))]*pq.mm


#%% extract state detection and all stims during mock period

# If UP paired: everytime it crosses the treshold, 5 stims at 10Hz.
# If DOWN paired: Everytime it doesn't cross the threshold for 200ms, 5 stims, then no stims until it crosses the threshold again.

lfp_cutoff_resp_channels = 200
plot = True

# LFP responsive channels to use for spike percentile analysis - intersect with channels that we know have good SW-locked spiking? 
intersect_channels = False

#extract LFP and spikes before and after each crossing
extract_before = 2000
extract_after = 2000

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for day in days:
# for day in days[10:]:
# for day in ['221212', '221216']:
    os.chdir(day)
    print(day)
    # if any(i in os.getcwd() for i in ['160128','160202','160420']): #'[160414', '160426', '160519', '160624', '160628', '160310', '160218', '160308', '160322', '160331', '160420', '160427']) == False:
    #     os.chdir('..')
    #     continue
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    PSTH_resp_channels = np.loadtxt('PSTH_resp_channels.csv', delimiter = ',')
    LFP_resp_channels_cutoff =  np.asarray([chan for chan in range(64) if (LFP_min[[0,1,2,3], chan] > lfp_cutoff_resp_channels).all() and (LFP_min[[4,5,6,7,8,9],chan] > lfp_cutoff_resp_channels).all()], dtype = int)
    SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',', dtype = int)

    os.chdir('..')
    
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
                
    
    if intersect_channels == False:
        channels_for_pairing_analysis = LFP_resp_channels_cutoff   
    else:
         channels_for_pairing_analysis = np.asarray([i for i in LFP_resp_channels_cutoff if i in SW_spiking_channels])

    # for mice with the nostim pairing sweep
    if [i for i in os.listdir() if 'pairing_nowhisker' in i or 'nostim' in i].__len__() > 0:
        os.chdir([i for i in os.listdir() if 'pairing_nowhisker' in i or 'nostim' in i][0])
        if os.path.exists('LFP_resampled'):
            pairing_LFP = pickle.load(open('LFP_resampled','rb'))
            pairing_spikes = pickle.load(open('pairing_spikes','rb'))
        else:
            channels = [s for s in os.listdir() if 'amp' in s and '.dat' in s]
            pairing_spikes = {}
            pairing_LFP = np.array([])
            for ind_channel, channel in enumerate(channels):
                print(channel)
                with open(channel,'rb') as f:
                    curr_LFP = np.fromfile(f, np.int16)
                    print(curr_LFP.size)
                    # take out spikes
                    highpass = elephant.signal_processing.butter(curr_LFP,highpass_frequency = 250, sampling_frequency=30000)
                    std = np.std(highpass)
                    crossings = np.argwhere(highpass<-5*std)
                    # take out values within half a second of each other
                    crossings = crossings[np.roll(crossings,-1) - crossings > 20]
                    pairing_spikes[channel[-6:-4]] = crossings/resample_factor
                    
                    #resample for LFP
                    curr_LFP_resampled = scipy.signal.resample(curr_LFP, int(np.ceil(len(curr_LFP)/resample_factor)))
                    
                if ind_channel == 0:
                    pairing_LFP = curr_LFP_resampled
                elif ind_channel > 0:                
                    pairing_LFP = np.vstack((pairing_LFP, curr_LFP_resampled))
                    
            pickle.dump(pairing_LFP, open('LFP_resampled','wb'))
            pickle.dump(pairing_spikes, open('pairing_spikes','wb'))
        
        # take biggest deflection channel as paired channel
        paired_channel = np.argmax(np.mean(LFP_min[0:4,:], axis = 0))
        pair_LFP = pairing_LFP[paired_channel]
        pair_spikes = list(pairing_spikes.values())[paired_channel]
        
        # concatenate spike from all  good channels
        # pair_spikes_all = np.sort(np.concatenate(list(pairing_spikes.values())))
        pair_spikes_all = np.sort(np.concatenate([list(pairing_spikes.values())[i] for i in list(channels_for_pairing_analysis.astype(int))]))
        
        paired_mean_firing_rate = len(pair_spikes)/(len(pair_LFP)/1000)
        all_mean_firing_rate = len(pair_spikes_all)/(len(pair_LFP)/1000)
    
        if os.path.isfile('board-DIN-00.dat'):
            stimfile = 'board-DIN-00.dat'   
        elif os.path.isfile('board-DIGITAL-IN-00.dat'):
            stimfile = 'board-DIGITAL-IN-00.dat'
        else:
            raise KeyError('no stim file')
        with open(stimfile,'rb') as f:
                crossings_all = np.fromfile(f, np.int16)
                crossings_all = np.where(np.diff(crossings_all) == 1)[0]/resample_factor
        
        crossings_all = np.delete(crossings_all, np.where(crossings_all < 2501))
        crossings_all = np.delete(crossings_all, np.where(crossings_all > len(pair_LFP) -2101))
        crossings_first = copy.deepcopy(crossings_all)
        diff = np.diff(crossings_first)
        for ind_stim, stim in enumerate(list(crossings_first)):
            if ind_stim == len(crossings_first) - 1:
                break
            if crossings_first[ind_stim + 1] - crossings_first[ind_stim] < 402:
                for i in range(1,100):
                    if ind_stim + i > len(crossings_first) - 1:
                        break
                    if crossings_first[ind_stim + i] - crossings_first[ind_stim] < 402:
                        crossings_first[ind_stim + i] = 0
                    elif crossings_first[ind_stim + i] - crossings_first[ind_stim] > 402:
                        break 
        
        crossings_first = list(np.delete(crossings_first, np.where(crossings_first < 1001)).astype(int))
        crossings_all = list(crossings_all.astype(int))
    
        first_stims_LFP_pair = np.asarray([pair_LFP[j - extract_before:j + extract_after] for j in crossings_first])
        first_stims_spikes_pair = np.asarray([np.histogram(pair_spikes[np.logical_and(pair_spikes > (j - extract_before), pair_spikes < (j + extract_before))] - (j - extract_before), bins = np.linspace(1,extract_after + extract_before,extract_after + extract_before))[0] for j in crossings_first])
        all_stims_LFP_pair = np.asarray([pair_LFP[j - extract_before:j + extract_after] for j in crossings_all])
        all_stims_spikes_pair = np.asarray([np.histogram(pair_spikes[np.logical_and(pair_spikes > (j - extract_before), pair_spikes < (j + extract_before))] - (j - extract_before), bins = np.linspace(1,extract_after + extract_before,extract_after + extract_before))[0] for j in crossings_all])
        
        # average of pairing signal (LFP and spiking) in all the good LFP channels
        first_stims_LFP_all = np.asarray([np.mean(pairing_LFP[channels_for_pairing_analysis,j - extract_before:j + extract_after], axis = 0) for j in crossings_first])
        first_stims_spikes_all = np.asarray([np.histogram(pair_spikes_all[np.logical_and(pair_spikes_all > (j - extract_before), pair_spikes_all < (j + extract_before))] - (j - extract_before), bins = np.linspace(1,extract_after + extract_before,extract_after + extract_before))[0] for j in crossings_first])
        all_stims_LFP_all = np.asarray([np.mean(pairing_LFP[channels_for_pairing_analysis,j - extract_before:j + extract_after], axis = 0) for j in crossings_all])
        all_stims_spikes_all = np.asarray([np.histogram(pair_spikes_all[np.logical_and(pair_spikes_all > (j - extract_before), pair_spikes_all < (j + extract_before))] - (j - extract_before), bins = np.linspace(1,extract_after + extract_before,extract_after + extract_before))[0] for j in crossings_all])
    
    
    
    
    # in mice with no nostim/nowhisker folder:
    else:   
        LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
        try:
            spikes_allsweeps = pickle.load(open('spikes_allsweeps_4','rb'))
        except FileNotFoundError:
            spikes_allsweeps = pickle.load(open('spikes_allsweeps_5','rb'))
        stim_times = pickle.load(open('stim_times','rb'))
        
        os.chdir([os.listdir()[i] for i in range(len(os.listdir())) if 'pairing' in os.listdir()[i]][0])
        settings = ET.parse('settings.xml')
        threshold = int(settings.getroot()[0].attrib['AnalogOut1ThresholdMicroVolts'])*10 # for some reason threshold is 10 times smaller than it actually is.
        paired_channel = settings.getroot()[0].attrib['AnalogOut1Channel']
        if settings.getroot()[0].attrib['AnalogOutHighpassFilterEnabled'] == 'False':
            high_filtered = False
        else:
            high_filtered = True
        
        # if didn't use spikes to pair, just use resampled LFP of last pre-pairing. if used spikes need to load the real thing and high pass filter
        if high_filtered == False:
            pair_LFP = LFP_all_sweeps[3][int(paired_channel[3:5]),:]
        elif high_filtered == True:
            pair_LFP = LFP_all_sweeps[3][int(paired_channel[3:5]),:]
        
        pair_spikes = spikes_allsweeps[3][paired_channel[3:5]] # note you might not want to take the spikes of that channel necessarily, what if it doesn't have spikes?
        pair_spikes_all = np.sort(np.concatenate([list(spikes_allsweeps[3].values())[i] for i in list(channels_for_pairing_analysis.astype(int))]))
        # get mean firing rate in pairing channel to make percentile figure.
        paired_mean_firing_rate = len(pair_spikes)/(len(pair_LFP)/1000)
        all_mean_firing_rate = len(pair_spikes_all)/(len(pair_LFP)/1000)
        
        if 'DOWN' in os.getcwd():
            pairing = 'DOWN'
        elif 'UP' in os.getcwd():
            pairing = 'UP'
        
        if pairing == 'UP':
            crossings = np.argwhere(pair_LFP < threshold)
        elif pairing == 'DOWN':    
            crossings = np.argwhere(pair_LFP > threshold)
            # needs to be consecutively above treshold for 200ms before it counts as a DOWN state. fist figure out consecutive DOWN times:
            # where index difference between threshold values was not 1 just before (i.e. it crossed threshold just before = start of DOWN state)
            DOWN_start = np.squeeze(crossings[np.squeeze(np.argwhere(np.squeeze(np.diff(crossings, axis = 0)) > 1)) + 1])
            #now where it doesn't cross threshold in the next 200ms, then add 200 as that is when you deliver stim
            crossings = DOWN_start[np.argwhere(np.diff(DOWN_start) > 200)] + 200
            
        
        #take out first 2 and last 2 seconds as need to extract LFP
        crossings = np.delete(crossings, np.where(crossings < 2000))
        crossings = np.delete(crossings, np.where(crossings > len(pair_LFP) - 2501))
        
        # now delete all crossings within XXms of the one before (the more, the more likely you'll have only one pairing between UP states)
        tolerance = 1000
        crossings_first = crossings[~(np.triu(np.abs(crossings[:,None] - crossings) <= tolerance,1)).any(0)]
        crossings_all = np.concatenate([np.linspace(i, i + 400, 5, dtype = int) for i in crossings_first])
        
        # take out crossings within XX of after a stim
        tolerance_stim = 2000
        crossings_first = list(crossings_first[~(np.triu(np.abs(stim_times[3][:,None] - np.asarray(crossings_first)) <= tolerance_stim, 1)).any(0)])
        crossings_all = list(crossings_all[~(np.triu(np.abs(stim_times[3][:,None] - np.asarray(crossings_all)) <= tolerance_stim, 1)).any(0)])
    
        
        first_stims_LFP_pair = np.asarray([pair_LFP[i - extract_before:i + extract_after] for i in crossings_first])
        first_stims_spikes_pair = np.asarray([np.histogram(pair_spikes[np.logical_and(pair_spikes > (i - extract_before), pair_spikes < (i + extract_after))] - (i - extract_before), bins = np.linspace(1,extract_after + extract_before,extract_after + extract_before))[0] for i in crossings_first])
        all_stims_LFP_pair = np.asarray([pair_LFP[i - extract_before:i + extract_after] for i in crossings_all])
        all_stims_spikes_pair = np.asarray([np.histogram(pair_spikes[np.logical_and(pair_spikes > (i - extract_before), pair_spikes < (i + extract_after))] - (i - extract_before), bins = np.linspace(1,extract_after + extract_before,extract_after + extract_before))[0] for i in crossings_all])
        
        first_stims_LFP_all = np.asarray([np.mean(LFP_all_sweeps[3][channels_for_pairing_analysis,i - extract_before:i + extract_after], axis = 0) for i in crossings_first])
        first_stims_spikes_all = np.asarray([np.histogram(pair_spikes_all[np.logical_and(pair_spikes_all > (i - extract_before), pair_spikes_all < (i + extract_after))] - (i - extract_before), bins = np.linspace(1,extract_after + extract_before,extract_after + extract_before))[0] for i in crossings_first])
        all_stims_LFP_all = np.asarray([np.mean(LFP_all_sweeps[3][channels_for_pairing_analysis,i - extract_before:i + extract_after], axis = 0) for i in crossings_all])
        all_stims_spikes_all = np.asarray([np.histogram(pair_spikes_all[np.logical_and(pair_spikes_all > (i - extract_before), pair_spikes_all < (i + extract_after))] - (i - extract_before), bins = np.linspace(1,extract_after + extract_before,extract_after + extract_before))[0] for i in crossings_all])   
    os.chdir('..')





    # SPIKE PERCENTILES OF ALL RESPONSIVE CHANNELS AS THE MEDIAN PERCENTILE SPIKE RATE ACROSS CHANNELS (INSTEAD OF THE PERCENTILE SPIKE RATE OF ALL SPIKES CONCATENATED)
    first_stims_percentiles_all_median = []
    all_stims_percentiles_all_median = []
    AUC_first_stims = [] # 'success' of pairing --> AUC of the pairing cumulative histogram (up to 1000 percent). The smaller the AUC, the more pairing success
    AUC_all_stims = []
    time_for_percentiles = 50
    if plot:
        fig, ax = plt.subplots(8,8,sharey = True)
        fig.suptitle(f'{day}')
    for chan in range(64):
        if [i for i in os.listdir() if 'pairing_nowhisker' in i or 'nostim' in i].__len__() > 0:
            curr_spikes = list(pairing_spikes.values())[chan]
        else:
            curr_spikes = list(spikes_allsweeps[3].values())[chan]
        mean_firing_rate = len(curr_spikes)/(len(pair_LFP)/1000)
        
        first_stims_spikes = np.asarray([np.histogram(curr_spikes[np.logical_and(curr_spikes > (i - extract_before), curr_spikes < (i + extract_after))] - (i - extract_before), bins = np.linspace(1,extract_after + extract_before,extract_after + extract_before))[0] for i in crossings_first])
        first_stims_percentiles = np.sum(first_stims_spikes[:, extract_before:extract_before + time_for_percentiles], axis = 1)/(time_for_percentiles/1000)/mean_firing_rate*100
        first_stims_percentiles_all_median.append(first_stims_percentiles)
        AUC_first_stims.append(np.sum((np.cumsum(np.histogram(first_stims_percentiles, bins = np.arange(0,4000,10))[0])/len(first_stims_percentiles))[:101]))
        
        all_stims_spikes = np.asarray([np.histogram(curr_spikes[np.logical_and(curr_spikes > (i - extract_before), curr_spikes < (i + extract_after))] - (i - extract_before), bins = np.linspace(1,extract_after + extract_before,extract_after + extract_before))[0] for i in crossings_all])
        all_stims_percentiles = np.sum(all_stims_spikes[:, extract_before:extract_before + time_for_percentiles], axis = 1)/(time_for_percentiles/1000)/mean_firing_rate*100
        all_stims_percentiles_all_median.append(all_stims_percentiles)
        AUC_all_stims.append(np.sum((np.cumsum(np.histogram(all_stims_percentiles, bins = np.arange(0,4000,10))[0])/len(all_stims_percentiles))[:101])) # SUM OF CUMULATIVE HISTOGRAM UNTIL 1000 PERCENT
        
        if plot:
            ax.flatten()[np.where(chanMap == chan)[0][0]].set_title(f'{chan}', size = 4)
            ax.flatten()[np.where(chanMap == chan)[0][0]].plot(np.histogram(first_stims_percentiles, bins = np.arange(0,4000,10))[1][:-1], np.cumsum(np.histogram(first_stims_percentiles, bins = np.arange(0,4000,10))[0])/len(first_stims_percentiles), color = 'k')
            ax.flatten()[np.where(chanMap == chan)[0][0]].set_xlim([0,1000])
            if chan in LFP_resp_channels_cutoff:
                ax.flatten()[np.where(chanMap == chan)[0][0]].set_facecolor('yellow')
            if chan in channels_for_pairing_analysis:
                ax.flatten()[np.where(chanMap == chan)[0][0]].set_facecolor('green')
            if chan == paired_channel:
                ax.flatten()[np.where(chanMap == chan)[0][0]].set_facecolor('red')
    if plot:
        plt.savefig('first stim all channel subplots')
    
    AUC_first_stims = np.asarray(AUC_first_stims)
    AUC_all_stims = np.asarray(AUC_all_stims)
    
    first_stims_percentiles_all_median = np.median(np.vstack([first_stims_percentiles_all_median[i] for i in channels_for_pairing_analysis]), axis = 0) # median across channels
    all_stims_percentiles_all_median = np.median(np.vstack([all_stims_percentiles_all_median[i] for i in channels_for_pairing_analysis]), axis = 0) # median across channels
    print(f'average percentile {day} first stims, median-channels for each stim: {np.mean(first_stims_percentiles_all_median)}')
    print(f'average percentile {day} all stims, median-channels for each stim: {np.mean(all_stims_percentiles_all_median)}')
    
                    
                    
    if plot:
        fig, ax = plt.subplots(2,1)
        fig.suptitle(day)
        fig.suptitle(f'first stim paired channel {day}')
        # ax[0].plot(np.mean(first_stims_LFP_pair, axis = 0))
        ax[0].plot(np.transpose(first_stims_LFP_pair))
        ax[1].plot(np.mean(first_stims_spikes_pair, axis = 0))
        plt.savefig('first stim paired channel')
        
        # fig, ax = plt.subplots(2,1)
        # fig.suptitle('all pair')
        # ax[0].plot(np.mean(all_stims_LFP_pair, axis = 0))
        # ax[1].plot(np.mean(all_stims_spikes_pair, axis = 0))
        
        fig, ax = plt.subplots(2,1)
        fig.suptitle(day)
        fig.suptitle(f'first stim all responsive channels {day}')
        # ax[0].plot(np.mean(first_stims_LFP_all, axis = 0))
        ax[0].plot(np.transpose(first_stims_LFP_all))
        ax[1].plot(np.mean(first_stims_spikes_all, axis = 0))
        plt.savefig('first stim all channels')

    # fig, ax = plt.subplots(2,1)
    # fig.suptitle('all all')
    # ax[0].plot(np.mean(all_stims_LFP_all, axis = 0))
    # ax[1].plot(np.mean(all_stims_spikes_all, axis = 0))
    
    
    # how long to take after stim for spike percentiles
    time_for_percentiles = 50
    first_stims_percentiles_pair = np.sum(first_stims_spikes_pair[:, extract_before:extract_before + time_for_percentiles], axis = 1)/(time_for_percentiles/1000)/paired_mean_firing_rate*100
    all_stims_percentiles_pair = np.sum(all_stims_spikes_pair[:, extract_before:extract_before + time_for_percentiles], axis = 1)/(time_for_percentiles/1000)/paired_mean_firing_rate*100
    first_stims_percentiles_all = np.sum(first_stims_spikes_all[:, extract_before:extract_before + time_for_percentiles], axis = 1)/(time_for_percentiles/1000)/all_mean_firing_rate*100
    all_stims_percentiles_all = np.sum(all_stims_spikes_all[:, extract_before:extract_before + time_for_percentiles], axis = 1)/(time_for_percentiles/1000)/all_mean_firing_rate*100
    
    if plot:
        #plot spike percentile
        fig, ax = plt.subplots(2,1)
        fig.suptitle(day)
        fig.suptitle(f'paired channel, first stim and all stims {day}')
        ax[0].plot(np.histogram(first_stims_percentiles_pair, bins = np.arange(0,4000,10))[1][:-1], np.cumsum(np.histogram(first_stims_percentiles_pair, bins = np.arange(0,4000,10))[0])/len(first_stims_percentiles_pair))
        ax[1].plot(np.histogram(all_stims_percentiles_pair, bins = np.arange(0,4000,10))[1][:-1], np.cumsum(np.histogram(all_stims_percentiles_pair, bins = np.arange(0,4000,10))[0])/len(all_stims_percentiles_pair))
        plt.savefig('pairing spike percentile paired channel')
    
        #plot spike percentile
        fig, ax = plt.subplots(2,1)
        fig.suptitle(day)
        fig.suptitle(f'all responsive channels, first stim and all stims {day}')
        ax[0].plot(np.histogram(first_stims_percentiles_all, bins = np.arange(0,4000,10))[1][:-1], np.cumsum(np.histogram(first_stims_percentiles_all, bins = np.arange(0,4000,10))[0])/len(first_stims_percentiles_all))
        ax[1].plot(np.histogram(all_stims_percentiles_all, bins = np.arange(0,4000,10))[1][:-1], np.cumsum(np.histogram(all_stims_percentiles_all, bins = np.arange(0,4000,10))[0])/len(all_stims_percentiles_all))
        plt.savefig('pairing spike percentile all channels')

    
    # os.chdir(home_directory)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    np.savetxt('first_stims_LFP_pair.csv', first_stims_LFP_pair, delimiter = ',')
    np.savetxt('first_stims_spikes_pair.csv', first_stims_spikes_pair, delimiter = ',')
    np.savetxt('all_stims_LFP_pair.csv', all_stims_LFP_pair, delimiter = ',')
    np.savetxt('all_stims_spikes_pair.csv', all_stims_spikes_pair, delimiter = ',')
    
    np.savetxt('first_stims_LFP_all.csv', first_stims_LFP_all, delimiter = ',')
    np.savetxt('first_stims_spikes_all.csv', first_stims_spikes_all, delimiter = ',')
    np.savetxt('all_stims_LFP_all.csv', all_stims_LFP_all, delimiter = ',')
    np.savetxt('all_stims_spikes_all.csv', all_stims_spikes_all, delimiter = ',')
    
    np.savetxt('first_stims_percentiles_pair.csv', first_stims_percentiles_pair, delimiter = ',')
    np.savetxt('all_stims_percentiles_pair.csv', all_stims_percentiles_pair, delimiter = ',')
    np.savetxt('first_stims_percentiles_all.csv', first_stims_percentiles_all, delimiter = ',')
    np.savetxt('all_stims_percentiles_all.csv', all_stims_percentiles_all, delimiter = ',')
    np.savetxt('first_stims_percentiles_all_median.csv', first_stims_percentiles_all_median, delimiter = ',')
    np.savetxt('all_stims_percentiles_all_median.csv', all_stims_percentiles_all_median, delimiter = ',')

    np.savetxt('AUC_first_stims.csv', AUC_first_stims, delimiter = ',')
    np.savetxt('AUC_all_stims.csv', AUC_all_stims, delimiter = ',')

    cl()
    
    os.chdir('..')
    os.chdir('..')
    
    
#%% FIGURES of state detection LFP and MUA all mice

reanalyze = True

b_notch, a_notch = scipy.signal.iirnotch(50, 100, 1000)

overall_path = r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS'

for UP_or_DOWN in ['DOWN', 'UP']:

    if UP_or_DOWN == 'UP':
        home_path = os.path.join(overall_path, r'UP_pairing')
    else:
        home_path = os.path.join(overall_path, r'DOWN_pairing')
    
    os.chdir(home_path)
    
    days = [i for i in os.listdir() if 'not_used' not in i and all(j not in i for j in ['160128','160202','160420']) and os.path.isdir(i)]
    
    numb = len(days)
    
    if reanalyze == False:
        first_stims_potential_all_ALL = np.loadtxt('first_stims_potential_all_ALL.csv', delimiter = ',')
        first_stims_potential_pair_ALL = np.loadtxt('first_stims_potential_pair_ALL.csv', delimiter = ',')
        first_stims_spikes_all_ALL = np.loadtxt('first_stims_spikes_all_ALL.csv', delimiter = ',')
        first_stims_spikes_pair_ALL = np.loadtxt('first_stims_spikes_pair_ALL.csv', delimiter = ',')
        first_stims_percentiles_all_ALL = pickle.load(open('first_stims_percentiles_all_ALL','rb'))
        all_stims_percentiles_all_ALL = pickle.load(open('all_stims_percentiles_all_ALL','rb'))
        first_stims_percentiles_all_median_ALL = pickle.load(open('first_stims_percentiles_all_median_ALL','rb'))
        all_stims_percentiles_all_median_ALL = pickle.load(open('all_stims_percentiles_all_median_ALL','rb'))

    
    else:
        first_stims_potential_all_ALL = np.zeros([numb,4000])
        first_stims_potential_pair_ALL = np.zeros([numb,4000])
        first_stims_spikes_all_ALL = np.zeros([numb,3999])
        first_stims_spikes_pair_ALL = np.zeros([numb,3999])
        first_stims_percentiles_all_ALL = []
        all_stims_percentiles_all_ALL = []
        first_stims_percentiles_all_median_ALL = []
        all_stims_percentiles_all_median_ALL = []

        lfp_cutoff_resp_channels = 200
        
        os.chdir(home_path)
        
        #AVERAGE OUT/MEDIAN LFP AND SPIKES ACROSS STIMS
        for day_ind, day in enumerate(days):
            os.chdir(day)
            print(day)  
            os.chdir([i for i in os.listdir() if 'analysis' in i][0])
            for file in os.listdir():
                if '.csv' in file:
                    if 'channels' in file or 'to_plot' in file:
                        globals()[f'{file[:-4]}'] = np.loadtxt(f'{file}', dtype = int, delimiter = ',')
                    
            first_stims_LFP_all = np.loadtxt('first_stims_LFP_all.csv', delimiter = ',')
            first_stims_LFP_pair = np.loadtxt('first_stims_LFP_pair.csv', delimiter = ',')
            first_stims_spikes_all = np.loadtxt('first_stims_spikes_all.csv', delimiter = ',')
            first_stims_spikes_pair = np.loadtxt('first_stims_spikes_pair.csv', delimiter = ',')
            first_stims_percentiles_all = np.loadtxt('first_stims_percentiles_all.csv', delimiter = ',')
            all_stims_percentiles_all = np.loadtxt('all_stims_percentiles_all.csv', delimiter = ',')
            first_stims_percentiles_all_median = np.loadtxt('first_stims_percentiles_all_median.csv', delimiter = ',')
            all_stims_percentiles_all_median = np.loadtxt('all_stims_percentiles_all_median.csv', delimiter = ',')

            first_stims_potential_all_ALL[day_ind, :] = np.mean(first_stims_LFP_all, axis = 0)
            first_stims_potential_pair_ALL[day_ind, :] = np.mean(first_stims_LFP_pair, axis = 0)
            first_stims_spikes_all_ALL[day_ind, :] = np.mean(first_stims_spikes_all, axis = 0)
            first_stims_spikes_pair_ALL[day_ind, :] = np.mean(first_stims_spikes_pair, axis = 0)
            first_stims_percentiles_all_ALL.append(first_stims_percentiles_all)
            all_stims_percentiles_all_ALL.append(all_stims_percentiles_all)
            first_stims_percentiles_all_median_ALL.append(first_stims_percentiles_all_median)
            all_stims_percentiles_all_median_ALL.append(all_stims_percentiles_all_median)
            os.chdir('..')
            os.chdir('..')
        
        np.savetxt('first_stims_potential_all_ALL.csv', first_stims_potential_all_ALL, delimiter = ',')
        np.savetxt('first_stims_potential_pair_ALL.csv', first_stims_potential_pair_ALL, delimiter = ',')
        np.savetxt('first_stims_spikes_all_ALL.csv', first_stims_spikes_all_ALL, delimiter = ',')
        np.savetxt('first_stims_spikes_pair_ALL.csv', first_stims_spikes_pair_ALL, delimiter = ',')
        pickle.dump(first_stims_percentiles_all_ALL, open('first_stims_percentiles_all_ALL', 'wb'))
        pickle.dump(all_stims_percentiles_all_ALL, open('all_stims_percentiles_all_ALL', 'wb'))
        pickle.dump(first_stims_percentiles_all_median_ALL, open('first_stims_percentiles_all_median_ALL', 'wb'))
        pickle.dump(all_stims_percentiles_all_median_ALL, open('all_stims_percentiles_all_median_ALL', 'wb'))


    fig, ax = plt.subplots(4,4, sharey = True)
    fig.suptitle('First all')
    for ax_ind, ax1 in enumerate(list(ax.flatten())):
        try:
            ax1.plot(first_stims_potential_all_ALL[ax_ind, :])
            ax1.set_title(days[ax_ind])
        except IndexError:
            continue
    
    fig, ax = plt.subplots(4,4, sharey = True)
    fig.suptitle('First pair')
    for ax_ind, ax1 in enumerate(list(ax.flatten())):
        try:
            ax1.plot(first_stims_potential_pair_ALL[ax_ind, :])#
            ax1.set_title(days[ax_ind])
        except IndexError:
            continue
    
    fig, ax = plt.subplots(4,4, sharey = True)
    fig.suptitle('first all')
    for ax_ind, ax1 in enumerate(list(ax.flatten())):
        try:
            ax1.plot(first_stims_spikes_all_ALL[ax_ind, :])
            ax1.set_title(days[ax_ind])
        except IndexError:
            continue
    
    fig, ax = plt.subplots(4,4, sharey = True)
    fig.suptitle('first pair')
    for ax_ind, ax1 in enumerate(list(ax.flatten())):
        try:
            ax1.plot(first_stims_spikes_pair_ALL[ax_ind, :])
            ax1.set_title(days[ax_ind])
        except IndexError:
            continue
    
    # fig, ax = plt.subplots()
    # ax.plot(smooth(np.median(first_stims_potential_all_ALL, axis = 0), 15))
    # # ax.set_ylim([-250,250])
    # fig, ax = plt.subplots()
    # ax.plot(np.median(scipy.signal.filtfilt(b_notch, a_notch, first_stims_potential_all_ALL[:,:], axis = 1), axis = 0))
    # ax.set_ylim([-250,250])
    
    # fig, ax = plt.subplots()
    # ax.plot(smooth(np.median(first_stims_potential_pair_ALL[[2,3,4,5,6,7],:], axis = 0), 1))
    # ax.set_ylim([-250,350])
    
    fig, ax = plt.subplots()
    # to_plot = scipy.signal.filtfilt(b_notch, a_notch, first_stims_potential_pair_ALL[[2,3,4,5,6,7],:], axis = 1)
    if UP_or_DOWN == 'UP':
        to_plot = scipy.signal.filtfilt(b_notch, a_notch, first_stims_potential_all_ALL, axis = 1)
    else:
        to_plot = scipy.signal.filtfilt(b_notch, a_notch, first_stims_potential_pair_ALL[:,:], axis = 1)
    # to_plot = scipy.signal.filtfilt(b_notch, a_notch, first_stims_potential_all_ALL, axis = 1)
    # to_plot = scipy.signal.filtfilt(b_notch, a_notch, first_stims_potential_pair_ALL[[0,1,2,3],:], axis = 1)
    ax.plot(np.median(to_plot, axis = 0), color = 'k')
    ax.fill_between(list(range(4000)), np.percentile(to_plot, 25, axis = 0), np.percentile(to_plot, 75, axis = 0), alpha = 0.2, color = 'k', edgecolor = 'w')
    # ax.set_ylim([-1500,500])
    # ax.set_xticks([0,999,1999])
    # ax.set_xticklabels(['-1', '0', '1'], size = 16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.set_yticks([-1500,-1000,-500,0,500])
    # ax.set_yticklabels(list(map(str, [-1500,-1000,-500,0,500])), size = 16)
    # ax.set_yticks([])
    plt.savefig('state detection LFP.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('state detection LFP.jpg', dpi = 1000, format = 'jpg')
    
    
    # fig, ax = plt.subplots()
    # ax.plot(np.mean(first_stims_spikes_all_ALL[[2,3,4,5,6,7],:], axis = 0))
    fig, ax = plt.subplots()
    if UP_or_DOWN == 'UP':
        to_plot = first_stims_spikes_all_ALL[:,:]
    else:
        to_plot = first_stims_spikes_all_ALL[:,:]
    # to_plot = first_stims_spikes_all_ALL[[1,2,3],:]
    # to_plot = first_stims_spikes_all_ALL[[3,4,5,6,7,8],:]
    # to_plot = first_stims_spikes_all_ALL[:,:]
    ax.plot(smooth(np.median(to_plot, axis = 0), 6), c = 'k')
    ax.fill_between(list(range(3999)), np.percentile(to_plot, 25, axis = 0), np.percentile(to_plot, 75, axis = 0), alpha = 0.2, color = 'k', edgecolor = 'w')
    ax.set_ylim([0,1.2])
    # ax.set_xticks([0,999,1999])
    # ax.set_xticklabels(['-1', '0', '1'], size = 16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.set_yticks([])
    plt.savefig('state detection spikes.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('state detection spikes.jpg', dpi = 1000, format = 'jpg')
    
    
    
    # plot percentile curves
    X = np.histogram(first_stims_percentiles_all_ALL[0], bins = np.arange(0,4000,10))[1][:-1]
    Y_first = np.asarray([np.cumsum(np.histogram(i, bins = np.arange(0,4000,10))[0])/len(i) for i in first_stims_percentiles_all_ALL])
    Y_all = np.asarray([np.cumsum(np.histogram(i, bins = np.arange(0,4000,10))[0])/len(i) for i in all_stims_percentiles_all_ALL])
    
    fig, ax = plt.subplots(4,4,sharey = True)
    fig.suptitle('first stim')
    for ax_ind, ax1 in enumerate(list(ax.flatten())):
        try:
            ax1.plot(X, np.transpose(Y_first)[:,ax_ind])
            ax1.set_title(days[ax_ind])
            ax1.set_xscale('log')
        except IndexError:
            continue
        
    fig, ax = plt.subplots(4,4,sharey = True)
    fig.suptitle('all stims')
    for ax_ind, ax1 in enumerate(list(ax.flatten())):
        try:
            ax1.plot(X, np.transpose(Y_all)[:,ax_ind])
            ax1.set_title(days[ax_ind])
            ax1.set_xscale('log')
        except IndexError:
            continue


#%% cumulative histogram state detection and all stims

# UP vs DOWN figure
os.chdir(overall_path)
bins_to_plot = np.arange(0,4000,1)
Y_first_UPDOWN = np.zeros([2,3999])
Y_all_UPDOWN = np.zeros([2,3999])
# plot percentile curves UP vs DOWN
# ax[0].plot(X, Y_all)
fig, ax = plt.subplots(figsize = (5,6.5))
# fig.suptitle('first stim')
for cond in ['UP_pairing', 'DOWN_pairing']:
    os.chdir(os.path.join(overall_path, cond))
    first_stims_percentiles_all_median_ALL = pickle.load(open('first_stims_percentiles_all_ALL','rb'))
    all_stims_percentiles_all_median_ALL = pickle.load(open('all_stims_percentiles_all_ALL','rb'))
    X = np.histogram(first_stims_percentiles_all_median_ALL[0], bins = bins_to_plot)[1][:-1]
    Y_first = np.asarray([np.cumsum(np.histogram(i, bins = bins_to_plot)[0])/len(i) for i in first_stims_percentiles_all_median_ALL])
    Y_all = np.asarray([np.cumsum(np.histogram(i, bins = bins_to_plot)[0])/len(i) for i in all_stims_percentiles_all_median_ALL])
    if 'UP' in cond:
        to_plot = Y_first*100
        Y_first_UPDOWN[0,:] = np.mean(to_plot, axis = 0)
        ax.plot(X, np.mean(to_plot, axis = 0), color = 'r')
        # ax.fill_between(X, np.percentile(to_plot, 25, axis = 0), np.percentile(to_plot, 75, axis = 0), alpha = 0.2, color = 'r', edgecolor = 'w')
        ax.fill_between(X, np.mean(to_plot, axis = 0) - np.std(to_plot,axis = 0)/np.sqrt(to_plot.shape[0]), np.mean(to_plot, axis = 0) + np.std(to_plot,axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.2, color = 'r', edgecolor = 'w')
    else:
        to_plot = Y_first[:,:]*100
        Y_first_UPDOWN[1,:] = np.mean(to_plot, axis = 0)
        ax.plot(X, np.mean(to_plot, axis = 0), color = 'k')
        # ax.fill_between(X, np.percentile(to_plot, 25, axis = 0), np.percentile(to_plot, 75, axis = 0), alpha = 0.2, color = 'k', edgecolor = 'w')
        ax.fill_between(X, np.mean(to_plot, axis = 0) - np.std(to_plot,axis = 0)/np.sqrt(to_plot.shape[0]), np.mean(to_plot, axis = 0) + np.std(to_plot,axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.2, color = 'k', edgecolor = 'w')
    ax.set_xscale('log')
    ax.set_xlim([1,1000])
    ax.set_xticks([10,100])
    ax.set_xticklabels(['10', '100'], size = 18)
    ax.set_xlabel('Spike rate (% baseline mean)', size = 18)
    ax.set_yticks([0,50,100])
    ax.set_yticklabels(['0','50', '100'], size = 18)
    ax.set_ylabel('% state detections', size = 18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    os.chdir('..')
    plt.savefig('First stim spikes cumulative.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('First stim spikes cumulative.jpg', dpi = 1000, format = 'jpg')


fig, ax = plt.subplots(figsize = (5,6.5))
# fig.suptitle('all stims')
for cond in ['UP_pairing', 'DOWN_pairing']:
    os.chdir(os.path.join(overall_path, cond))
    first_stims_percentiles_all_median_ALL = pickle.load(open('first_stims_percentiles_all_ALL','rb'))
    all_stims_percentiles_all_median_ALL = pickle.load(open('all_stims_percentiles_all_ALL','rb'))
    X = np.histogram(first_stims_percentiles_all_median_ALL[0], bins = bins_to_plot)[1][:-1]
    Y_first = np.asarray([np.cumsum(np.histogram(i, bins = bins_to_plot)[0])/len(i) for i in first_stims_percentiles_all_median_ALL])
    Y_all = np.asarray([np.cumsum(np.histogram(i, bins = bins_to_plot)[0])/len(i) for i in all_stims_percentiles_all_median_ALL])
    if 'UP' in cond:
        to_plot = Y_all*100
        Y_all_UPDOWN[0,:] = np.mean(to_plot, axis = 0) # average over mice for Anderson-Darling test
        ax.plot(X, np.mean(to_plot, axis = 0), color = 'r')
        # ax.fill_between(X, np.percentile(to_plot, 25, axis = 0), np.percentile(to_plot, 75, axis = 0), alpha = 0.2, color = 'r', edgecolor = 'w')
        ax.fill_between(X, np.mean(to_plot, axis = 0) - np.std(to_plot,axis = 0)/np.sqrt(to_plot.shape[0]), np.mean(to_plot, axis = 0) + np.std(to_plot,axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.2, color = 'r', edgecolor = 'w')

    else:
        # to_plot = Y_all[[0,1,2,3,4,5,6,7,8,9,10,12],:]*100
        to_plot = Y_all[:,:]*100
        Y_all_UPDOWN[1,:] = np.mean(to_plot, axis = 0)
        ax.plot(X, np.mean(to_plot, axis = 0), color = 'k')
        # ax.fill_between(X, np.percentile(to_plot, 25, axis = 0), np.percentile(to_plot, 75, axis = 0), alpha = 0.2, color = 'k', edgecolor = 'w')
        ax.fill_between(X, np.mean(to_plot, axis = 0) - np.std(to_plot,axis = 0)/np.sqrt(to_plot.shape[0]), np.mean(to_plot, axis = 0) + np.std(to_plot,axis = 0)/np.sqrt(to_plot.shape[0]), alpha = 0.2, color = 'k', edgecolor = 'w')
    ax.set_xscale('log')
    ax.set_xlim([1,1000])
    ax.set_xticks([10,100])
    ax.set_xticklabels(['10', '100'], size = 18)
    ax.set_xlabel('Spike rate (% baseline mean)', size = 18)
    ax.set_yticks([0,50,100])
    ax.set_yticklabels(['0','50', '100'], size = 18)
    ax.set_ylabel('% stimulations', size = 18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    os.chdir('..')
    plt.savefig('All stim spikes cumulative.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('All stim spikes cumulative.jpg', dpi = 1000, format = 'jpg')

# check overlap of average distributions UP vs DOWN
print(f'spiking distribution FIRST stim: {scipy.stats.anderson_ksamp((Y_first_UPDOWN[0,:], Y_first_UPDOWN[1,:]))}')
print(f'spiking distribution ALL stims: {scipy.stats.anderson_ksamp((Y_all_UPDOWN[0,:], Y_all_UPDOWN[1,:]))}')




# median or mean values across stims in each mouse s of spike percentile for t-test or Mann Whitney test
first_median = []
all_median = []
first_avg = []
all_avg = []
os.chdir(overall_path)
for cond in ['UP_pairing', 'DOWN_pairing']:
    os.chdir(os.path.join(overall_path, cond))
    first_stims_percentiles_all_median_ALL = pickle.load(open('first_stims_percentiles_all_median_ALL','rb'))
    all_stims_percentiles_all_median_ALL = pickle.load(open('all_stims_percentiles_all_median_ALL','rb'))
    first_median.append([np.median(i) for i in first_stims_percentiles_all_median_ALL])
    all_median.append([np.median(i) for i in all_stims_percentiles_all_median_ALL])
    first_avg.append([np.mean(i) for i in first_stims_percentiles_all_median_ALL])
    all_avg.append([np.mean(i) for i in all_stims_percentiles_all_median_ALL])

    os.chdir('..')
    
scipy.stats.shapiro(first_avg[0])   

# DOWN pairings not normally distributed (UP pairings are) so use non parametric test
scipy.stats.mannwhitneyu(first_avg[0], first_avg[1])
scipy.stats.mannwhitneyu(all_avg[0], all_avg[1])

print(mean(first_avg[0]), mean(first_avg[1]))
print(np.std(first_avg[0]), np.std(first_avg[1]))

print(mean(all_avg[0]), mean(all_avg[1]))
print(np.std(all_avg[0]), np.std(all_avg[1]))

#%% Pairing vs depression:
# 1) frequency of state detections that are UP states for each channel individually? no
# 2) how much smaller is the UP state response? If it's a lot smaller than DOWN state response (ratio between them) then more subthreshold which would be more depressed?

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i) and 'old' not in i]
for day in days:
# for day in ['281021']:
    os.chdir(day) 
    print(day)
    
    os.chdir('pairing')
    LFP = pickle.load(open('LFP_resampled','rb'))
    spikes = pickle.load(open('pairing_spikes','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    
    #clean up shitty stim times
    for ind_stim, stim in enumerate(list(stim_times)):
        if ind_stim == len(stim_times) - 1:
            break
        if stim_times[ind_stim + 1] - stim_times[ind_stim] < 99.9:
            for i in range(1,10):
                if ind_stim + i > len(stim_times) - 1:
                    break
                if stim_times[ind_stim + i] - stim_times[ind_stim] < 99.9:
                    stim_times[ind_stim + i] = 0
                elif stim_times[ind_stim + i] - stim_times[ind_stim] > 99.9:
                    break 
    stim_times = np.delete(stim_times, np.where(stim_times == 0))   
    pickle.dump(stim_times, open('stim_times','wb'))


    if np.all(np.diff(stim_times)[:4] <= 101):     
        first_stims = stim_times[0::5]
    
    tolerance_before_1 = 200
    tolerance_before_2 = 5
    
    UP_frequency = np.zeros([64])
    
    UP_PSTH = np.zeros([64,299])
    DOWN_PSTH = np.zeros([64,299])
    bins = np.linspace(1,300,300)

    for chan in range(64):
        curr_spikes = list(spikes.values())[chan]
        DOWN_stims = []
        UP_stims = []
        UP_spikes = []
        DOWN_spikes = []
        for stim in first_stims:
            curr_spiking_response = np.histogram((curr_spikes[(stim - 100 < curr_spikes) & (curr_spikes < stim + 200)] - (stim - 100)), bins)[0]
            if curr_spikes[np.logical_and((stim - tolerance_before_1) < curr_spikes, (stim - tolerance_before_2) > curr_spikes)].size == 0:
                DOWN_stims.append(stim)
                DOWN_spikes.append(curr_spiking_response)
            else:
                UP_stims.append(stim)
                UP_spikes.append(curr_spiking_response)
        UP_frequency[chan] = len(UP_stims)/(len(UP_stims) + len(DOWN_stims))
        # average PSTH over stims
        UP_PSTH[chan] = np.mean(np.asarray(UP_spikes), axis = 0)
        DOWN_PSTH[chan] = np.mean(np.asarray(DOWN_spikes), axis = 0)
    os.chdir('..')

    artifacts = list(np.linspace(97,123,27,dtype = int))     
    UP_PSTH[:,artifacts] = 0
    DOWN_PSTH[:,artifacts] = 0
    fig, ax = plt.subplots(8,8,sharey = True) 
    fig.suptitle(f'UP vs DOWN spiking pairing {day}')
    for ind, ax1 in enumerate(list(ax.flatten())):                        
        ax1.plot(UP_PSTH[chanMap[ind],80:195], color = 'r')
        ax1.plot(DOWN_PSTH[chanMap[ind],80:195], color = 'k')
        ax1.set_title(str(chanMap[ind]), size = 6)
    plt.savefig('pairing spiking response UP vs DOWN')
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    np.savetxt('pairing_UP_freq_first_stims.csv', UP_frequency, delimiter = ',')
    
    os.chdir('..')
    os.chdir('..')



#%% Actual pairing traces and examples, maybe also extract last vs first LFP/MUA peak
b_notch, a_notch = scipy.signal.iirnotch(50, 100, 1000)

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i) and 'old' not in i]
for day in days:
# for day in ['160427']:
    os.chdir(day) 
    print(day)
    if day == '160624_B2':
        os.chdir('..')
        continue
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
    PSTH_resp_channels = np.loadtxt('PSTH_resp_channels.csv', delimiter = ',', dtype = int)
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    principal_channel = np.argmin(LFP_min[3,:])
    os.chdir('..')
    
    # os.chdir([os.listdir()[i] for i in range(len(os.listdir())) if 'pairing' in os.listdir()[i] and 'nowhisker' not in os.listdir()[i]][0])
    os.chdir('pairing')
    LFP = pickle.load(open('LFP_resampled','rb')).astype('float32')
    # highpass_LFP = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP.astype('float32')), units = 'mV', sampling_rate = new_fs*pq.Hz), highpass_frequency = 150*pq.Hz).as_array().T
    spikes = pickle.load(open('pairing_spikes','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    if os.path.isfile('settings.xml'):
        settings = ET.parse('settings.xml')
        paired_channel = int(settings.getroot()[0].attrib['AnalogOut1Channel'][3:5])
        threshold = int(settings.getroot()[0].attrib['AnalogOut1ThresholdMicroVolts'])*10 # for some reason threshold is 10 times smaller than it actually is.
    # os.chdir('..')
    
    a = np.diff(stim_times)
    # clean up shitty stim times
    for ind_stim, stim in enumerate(list(stim_times)):
        if ind_stim == len(stim_times) - 1:
            break
        if stim_times[ind_stim + 1] - stim_times[ind_stim] < 99.89:
            for i in range(1,10):
                if ind_stim + i > len(stim_times) - 1:
                    break
                if stim_times[ind_stim + i] - stim_times[ind_stim] < 99.89:
                    stim_times[ind_stim + i] = 0
                elif stim_times[ind_stim + i] - stim_times[ind_stim] > 99.89:
                    break 
    stim_times_clean = np.delete(stim_times, np.where(stim_times == 0))   
    # pickle.dump(stim_times, open('stim_times','wb'))
    
    if np.all(np.diff(stim_times_clean)[:4] <= 101):     
        first_stims = stim_times_clean[0::5]
    else:
        first_stims = stim_times_clean[np.argwhere((np.diff(stim_times_clean)[:4] <= 101) == False)[0][0]+1::5]
    
    # b = []
    # for s in np.arange(0, len(stim_times_clean), 5)[:-1]:
    #     b.append(np.diff(stim_times_clean[[s, s+4]]))
    # b = np.asarray(b)
    
    # ------------------------------------------------------------------------- LFP pairings
    LFP_pairings = []
    for stim in first_stims[first_stims>1000][:-3]:
        # LFP_pairings.append(scipy.signal.filtfilt(b_notch, a_notch, LFP[:, int(stim-500):int(stim+1000)]))
        LFP_pairings.append(LFP[:, int(stim-500):int(stim+1000)])
    LFP_pairings = np.asarray(LFP_pairings)/1000 # convert to mV
    
    patch = 2
    fig, ax = plt.subplots(figsize = (5,2))
    mean_across_chans = np.mean(np.mean(LFP_pairings[:,LFP_resp_channels_cutoff,:], axis = 1).T, axis = 1)
    ax.plot(np.mean(LFP_pairings[:,LFP_resp_channels_cutoff,:].T, axis = 2), linewidth = 0.5) # plot all channels
    ax.plot(mean_across_chans, color = 'k')
    # yerror = np.std(np.mean(LFP_pairings[:,LFP_resp_channels_cutoff,:], axis = 1).T, axis = 1)/np.sqrt(LFP_pairings.shape[0])
    # ax.fill_between(np.linspace(0,1500,1500), mean_to_plot + patch*yerror, mean_to_plot - patch*yerror, alpha = 0.1, color = 'k')
    for stim in range(5):
        ax.axvline([500+stim*100], color = 'k', linestyle = '--', linewidth = 0.5)
    ax.set_xlim([200, 1250])
    ax.set_xticks([500,1000])
    ax.set_xticklabels(['0', '500'])
    ax.tick_params(axis="x", labelsize=14)    
    ax.tick_params(axis="y", labelsize=14) 
    ax.set_xlabel('time (ms)', size = 16)
    ax.set_ylabel('LFP (mV)', size = 16)
    plt.tight_layout()
    plt.savefig('pairing LFP average.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('pairing LFP average.pdf', format = 'pdf', dpi = 1000)
    
    # pairing responses first 5 minutes vs last 5 minutes
    fig, ax = plt.subplots(8,8, figsize = (15,10), constrained_layout = True, sharex = True) 
    first_pairings = np.mean(LFP_pairings[:100,:,:], axis = 0)
    last_pairings = np.mean(LFP_pairings[-100:,:,:], axis = 0)
    # fig.suptitle('LFP min in all stims')
    for ind, ax1 in enumerate(list(ax.flatten())):
        ax1.set_xlim([400,1100])
        ax1.set_xticks([])
        if chanMap[ind] in LFP_resp_channels_cutoff:             
            ax1.plot(smooth(first_pairings[chanMap[ind],:], 5), color = 'k', linewidth = 0.5)
            ax1.plot(smooth(last_pairings[chanMap[ind],:], 5), color = 'r', linewidth = 0.5)
            ax1.set_title(str(chanMap[ind]), size = 6)
            for stim in range(5):
                ax1.axvline([500+stim*100], color = 'k', linestyle = '--', linewidth = 0.25)
    plt.savefig('pairing LFP first vs last 5min average.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('pairing LFP first vs last 5min average.pdf', format = 'pdf', dpi = 1000)
    
    # ------------------------------------------------------------------------- spike pairings
    spike_pairings = [[] for i in range(64)]
    bins = np.linspace(1,1500,1500)
    for chan in range(64):
        curr_spikes = list(spikes.values())[chan]
        for stim in first_stims[first_stims>1000][:-1]:
            curr_spiking_response = (np.histogram((curr_spikes[(stim - 500 < curr_spikes) & (curr_spikes < stim + 1000)] - (stim - 500)), bins)[0]).astype('float32')
            #clean up artifacts
            if '191121' in os.getcwd() or '121121' in os.getcwd() or '291021' in os.getcwd():
                for stim in range(5):
                    curr_spiking_response[500+stim*100-2:500+stim*100+1] = 0
            if '281021' in os.getcwd():
                for stim in range(5):
                    curr_spiking_response[500+stim*100-2:500+stim*100+1] = 0
                    curr_spiking_response[500+stim*100+19] = 0
            if '061221' in os.getcwd():
                for stim in range(5):
                    curr_spiking_response[500+stim*100-3:500+stim*100+1] = 0
                    if chan == 44 or chan == 46:
                        curr_spiking_response[500+stim*100+6:500+stim*100+20] = 0
            if '221208' in os.getcwd():
                for stim in range(5):
                    curr_spiking_response[500+stim*100+6:500+stim*100+10] = 0
                    curr_spiking_response[500+stim*100+18:500+stim*100+20] = 0
                    if chan == 11:
                        curr_spiking_response[500+stim*100+31:500+stim*100+32] = 0
                        curr_spiking_response[500+stim*100+10:500+stim*100+11] = 0
            if '221213' in os.getcwd():
                for stim in range(5):
                    curr_spiking_response[500+stim*100-2:500+stim*100+1] = 0
                    curr_spiking_response[500+stim*100+19:500+stim*100+20] = 0
                    PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == 11)[0]) # artifacty channel

            spike_pairings[chan].append(curr_spiking_response)
    spike_pairings = np.asarray(spike_pairings).astype('float32')
    
    
    fig, ax = plt.subplots(figsize = (5,2))
    channel_avg_spike_pairings = np.mean(spike_pairings[PSTH_resp_channels,:,:], axis = 0).T*1000 #average across channels
    mean_to_plot = np.mean(channel_avg_spike_pairings, axis = 1) #average across pairings
    #clean up artifacts
    # if '121121' in os.getcwd():
    #     mean_to_plot[mean_to_plot>30] = 0
        
    ax.plot(smooth(np.mean(spike_pairings[PSTH_resp_channels,:,:], axis = 1).T*1000, 5, axis = 0), linewidth = 0.5) # all channels
    ax.plot(smooth(mean_to_plot, 1), color = 'k')
    # patch = 2
    # yerror = np.std(np.mean(spike_pairings[PSTH_resp_channels,:,:], axis = 0).T*1000/np.sqrt(spike_pairings.shape[2])
    # ax.fill_between(np.linspace(0,1500,1499), mean_to_plot + patch*yerror, mean_to_plot - patch*yerror, alpha = 0.1, color = 'k')
    for stim in range(5):
        ax.axvline([500+stim*100], color = 'k', linestyle = '--', linewidth = 0.25)
    plt.tight_layout()
    # ax.set_ylim([0,np.max(smooth(mean_to_plot,5)) + 1])
    ax.set_xlim([200, 1250])
    ax.set_xticks([500,1000])
    ax.set_xticklabels(['0', '500'])
    ax.tick_params(axis="x", labelsize=14)    
    ax.tick_params(axis="y", labelsize=14) 
    ax.set_xlabel('time (ms)', size = 16)
    ax.set_ylabel('MUA (Hz)', size = 16)
    plt.tight_layout()
    plt.savefig('pairing MUA average.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('pairing MUA average.pdf', format = 'pdf', dpi = 1000)

    # pairing responses first 5 minutes vs last 5 minutes
    fig, ax = plt.subplots(8,8, figsize = (15,10), constrained_layout = True, sharex = True) 
    first_pairings = np.mean(spike_pairings[:,:100,:], axis = 1)*1000
    last_pairings = np.mean(spike_pairings[:,-100:,:], axis = 1)*1000
    # fig.suptitle('LFP min in all stims')
    for ind, ax1 in enumerate(list(ax.flatten())):
        ax1.set_xlim([400,1100])
        ax1.set_xticks([])
        if chanMap[ind] in PSTH_resp_channels:             
            ax1.plot(smooth(first_pairings[chanMap[ind],:], 5), color = 'k', linewidth = 0.5)
            ax1.plot(smooth(last_pairings[chanMap[ind],:], 5), color = 'r', linewidth = 0.5)
            ax1.set_title(str(chanMap[ind]), size = 6)
            for stim in range(5):
                ax1.axvline([500+stim*100], color = 'k', linestyle = '--', linewidth = 0.25)
    plt.savefig('pairing MUA first vs last 5 min average.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('pairing MUA first vs last 5 min average.pdf', format = 'pdf', dpi = 1000)

            # ax1.axvline(3.5)

    
# ----------------------------------------------------------------------------------- example traces
    # example trace, 201121 stim 24 for UP pairing is ok, 221213 stim 103 for DOWN pairing
    stim_trace = np.zeros([LFP.shape[1]])
    stim_trace[stim_times_clean.astype(int)] = 1
    stim = 24
    before = 800
    after = 4200
    fig, ax = plt.subplots(figsize = (5,2))
    ax.plot(np.linspace(0, (before+after)/1000, before+after), LFP[paired_channel, int(first_stims[stim] - before):int(first_stims[stim] + after)], color = 'k', linewidth = 0.5)
    ax.axhline(-900 ,linestyle = '--', color = 'red')
    ax.plot(np.linspace(0, (before+after)/1000, before+after), stim_trace[int(first_stims[stim] - before):int(first_stims[stim] + after)]*500 - 2900, color = 'k')
    ax.set_xlabel('time (s)', size = 16)
    ax.set_ylabel('LFP (uV)', size = 16)
    ax.set_yticks([-2000,0])
    ax.set_yticklabels(['-2','0'])
    ax.tick_params(axis="x", labelsize=14)    
    ax.tick_params(axis="y", labelsize=14) 
    plt.tight_layout()# ax[1].plot(highpass_LFP[paired_channel, int(first_stims[stim] - 900):int(first_stims[stim] + 8000)])
    plt.savefig('pairing example.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('pairing example.pdf', format = 'pdf', dpi = 1000)
    
    os.chdir('..')
    os.chdir('..')

    cl()


#%% # pairing examples for laminar probes
b_notch, a_notch = scipy.signal.iirnotch(50, 100, 1000)

chanMap_32 = np.array([31,32,30,33,27,36,26,37,16,47,18,38,17,46,29,39,19,45,28,40,20,44,22,41,21,34,24,42,23,43,25,35]) - 16
chanMap_16 = np.linspace(0, 15, 16).astype(int)

mouse_1_1 = list(map(np.asarray, [[5], [9], [13], [22]]))
mouse_2_1 = list(map(np.asarray, [[12], [14], [20], [24]])) # layer 2/3 could also be 10
mouse_3_1 = list(map(np.asarray, [[7], [11], [17], [22]])) # layer 2/3 could also be 6
mouse_4_1 = list(map(np.asarray, [[7], [11], [16], [21]])) # layer 2/3 could also be 5,6,7
mouse_5_1 = list(map(np.asarray, [[5], [9], [14], [19]])) # layer 2/3 could also be 4
mouse_6_1 = list(map(np.asarray, [[5], [8], [14], [17]])) # layer 2/3 could be 4,5,6 SW CSD a bit unclear
mouse_7_1 = list(map(np.asarray, [[5], [7], [12], [14]])) # layer 5 could also be 11
mouse_8_1 = list(map(np.asarray, [[4], [7], [11], [14]]))
mouse_9_1 = list(map(np.asarray, [[3], [7], [11], [14]]))
mouse_10_1 = list(map(np.asarray, [[4], [8], [12], [14]])) #layer 2/3 could also be 4
mouse_11_1 = list(map(np.asarray, [[4], [7], [12], [14]]))
mouse_12_1 = list(map(np.asarray, [[3], [7], [11], [14]]))
mouse_13_1 = list(map(np.asarray, [[5], [8], [12], [15]]))
layer_dict_1 = {'160614' : [mouse_1_1]*10,               
            '160615' : [mouse_2_1]*4 + [[i + 1 for i in mouse_2_1]]*6,
            '160622' : [mouse_3_1]*10,
            '160728' : [mouse_4_1]*10,
            '160729' : [mouse_5_1]*5 + [[i + 1 for i in mouse_5_1]]*1 + [[i + 2 for i in mouse_5_1]]*1 + [[i + 3 for i in mouse_5_1]]*3,
            # '160810' : [mouse_6_1]*4 + [[i + 1 for i in mouse_6_1]]*6,
            '160810' : [mouse_6_1]*10,
            '220810_2' : [mouse_7_1]*10,
            '221018_1' : [mouse_8_1]*10,
            '221021_1' : [mouse_9_1]*10,
            '221024_1' : [mouse_10_1]*10,
            '221025_1' : [mouse_11_1]*10,
            '221026_1' : [mouse_12_1]*10,
            '221206_1' : [mouse_13_1]*10
            }


highpass_cutoff = 4
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

# for 32 channel days
# for day in ['160615']:
for day_ind, day in enumerate(days[0:6]):
    os.chdir(day)
    print(day)
    # os.chdir([os.listdir()[i] for i in range(len(os.listdir())) if 'pairing' in os.listdir()[i] and 'nowhisker' not in os.listdir()[i]][0])
    os.chdir('pairing')
    LFP = pickle.load(open('LFP_resampled','rb')).astype('float32')[chanMap_32,:]
    spikes = pickle.load(open(f'pairing_spikes_{highpass_cutoff}','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    # os.chdir('..')
    
    if LFP.shape[0] == 16:
        chanMap = chanMap_16
    elif LFP.shape[0] == 32:
        chanMap = chanMap_32
        
    #clean up shitty stim times
    for ind_stim, stim in enumerate(list(stim_times)):
        if ind_stim == len(stim_times) - 1:
            break
        if stim_times[ind_stim + 1] - stim_times[ind_stim] < 99.9:
            for i in range(1,10):
                if ind_stim + i > len(stim_times) - 1:
                    break
                if stim_times[ind_stim + i] - stim_times[ind_stim] < 99.9:
                    stim_times[ind_stim + i] = 0
                elif stim_times[ind_stim + i] - stim_times[ind_stim] > 99.9:
                    break 
    stim_times = np.delete(stim_times, np.where(stim_times == 0))   
    # pickle.dump(stim_times, open('stim_times','wb'))
    
    if np.all(np.diff(stim_times)[:4] <= 101):     
        first_stims = stim_times[0::5]
    else:
        first_stims = stim_times[np.argwhere((np.diff(stim_times)[:4] <= 101) == False)[0][0]+1::5]
    
    
    
    # # LFP pairings
    LFP_pairings = []
    for stim in first_stims[first_stims>1000][:-3]:
        LFP_pairings.append(LFP[:, int(stim-500):int(stim+1000)])
    LFP_pairings = np.asarray(LFP_pairings)
    
    # spacer = np.max(np.mean(LFP_pairings[:,3:,:], axis = 0))/1.2
    # fig, ax = plt.subplots(figsize = (3,10))
    # for ind in range(32):
    #     ax.plot(np.mean(LFP_pairings[:,ind,:], axis = 0) + ind * -spacer *np.ones(LFP_pairings.shape[2]), 'k', linewidth = 1)                 
    # for stim in range(5):
    #     ax.axvline([500+stim*100], color = 'k', linestyle = '--', linewidth = 0.5)
    # ax.set_xlim([400, 1100])
    # ax.set_xticks([500,1000])
    # ax.set_xticklabels(['0', '500'])
    # ax.tick_params(axis="x", labelsize=14)    
    # ax.set_yticks([])
    # # ax.tick_params(axis="y", labelsize=14) 
    # ax.set_xlabel('time (ms)', size = 16)
    # # ax.set_ylabel('LFP (uV)', size = 16)
    # plt.tight_layout()
    # # plt.savefig('pairing LFP average.jpg', format = 'jpg', dpi = 1000)
    # # plt.savefig('pairing LFP average.pdf', format = 'pdf', dpi = 1000)
    
    # fig, ax = plt.subplots(3, 1, sharey = True, figsize = (6,7))
    # for layer in [0,1,2]:
    #     ax[layer].plot(scipy.signal.filtfilt(b_notch, a_notch, np.mean(LFP_pairings[:,layer_dict_1[day][0][layer][0],:], axis = 0)), 'k')
    #     for stim in range(5):
    #         ax[layer].axvline([500+stim*100], color = 'k', linestyle = '--', linewidth = 0.5)
    #     ax[layer].set_xlim([400, 1100])
    #     ax[layer].set_xticks([500,1000])
    #     ax[layer].set_xticklabels(['0', '500'])
    #     ax[layer].tick_params(axis="x", labelsize=18)    
    #     # ax[layer].set_yticks([])
    #     ax[layer].tick_params(axis="y", labelsize=18) 
    #     # ax[layer].set_xlabel('time (ms)', size = 16)
    #     # ax.set_ylabel('LFP (uV)', size = 16)
    # plt.tight_layout()
    # plt.savefig('pairing LFP average per layer sharey.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('pairing LFP average per layer sharey.pdf', format = 'pdf', dpi = 1000)

    # fig, ax = plt.subplots(3, 1, figsize = (6,7))
    # for layer in [0,1,2]:
    #     ax[layer].plot(scipy.signal.filtfilt(b_notch, a_notch, np.mean(LFP_pairings[:,layer_dict_1[day][0][layer][0],:], axis = 0)), 'k')
    #     for stim in range(5):
    #         ax[layer].axvline([500+stim*100], color = 'k', linestyle = '--', linewidth = 0.5)
    #     ax[layer].set_xlim([400, 1100])
    #     ax[layer].set_xticks([500,1000])
    #     ax[layer].set_xticklabels(['0', '500'])
    #     ax[layer].tick_params(axis="x", labelsize=18)    
    #     # ax[layer].set_yticks([])
    #     ax[layer].tick_params(axis="y", labelsize=18) 
    #     # ax[layer].set_xlabel('time (ms)', size = 16)
    #     # ax.set_ylabel('LFP (uV)', size = 16)
    # plt.tight_layout()
    # plt.savefig('pairing LFP average per layer.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('pairing LFP average per layer.pdf', format = 'pdf', dpi = 1000)



    # # spike pairings
    spike_pairings = [[] for i in range(32)]
    bins = np.linspace(1,1500,1500)
    for chan in range(32):
        curr_spikes = list(spikes.values())[chan]
        for stim in first_stims[first_stims>1000][:-3]:
            curr_spiking_response = np.histogram((curr_spikes[(stim - 500 < curr_spikes) & (curr_spikes < stim + 1000)] - (stim - 500)), bins)[0]
            spike_pairings[chan].append(curr_spiking_response)
    spike_pairings = np.transpose(np.asarray(spike_pairings), (1,0,2))[:,chanMap,:]
    
    # spacer = np.max(np.mean(spike_pairings[:,3:,:], axis = 0))/1.2
    # fig, ax = plt.subplots(figsize = (3,10))
    # for ind in range(32):
    #     ax.plot(np.mean(spike_pairings[:,ind,:], axis = 0) + ind * -spacer *np.ones(spike_pairings.shape[2]), 'k', linewidth = 1)                 
    # for stim in range(5):
    #     ax.axvline([500+stim*100], color = 'k', linestyle = '--', linewidth = 0.5)
    # ax.set_xlim([400, 1100])
    # ax.set_xticks([500,1000])
    # ax.set_xticklabels(['0', '500'])
    # ax.tick_params(axis="x", labelsize=14)    
    # ax.set_yticks([])
    # # ax.tick_params(axis="y", labelsize=14) 
    # ax.set_xlabel('time (ms)', size = 16)
    # # ax.set_ylabel('LFP (uV)', size = 16)
    # plt.tight_layout()
    # # plt.savefig('pairing MUA average.jpg', format = 'jpg', dpi = 1000)
    # # plt.savefig('pairing MUA average.pdf', format = 'pdf', dpi = 1000)

    fig, ax = plt.subplots(3, 1, sharey = True, figsize = (6,7))
    for layer in [0,1,2]:
        ax[layer].plot(scipy.signal.filtfilt(b_notch, a_notch, np.mean(spike_pairings[:,layer_dict_1[day][0][layer][0],:], axis = 0))*1000, 'k')
        for stim in range(5):
            ax[layer].axvline([500+stim*100], color = 'k', linestyle = '--', linewidth = 0.5)
        ax[layer].set_xlim([400, 1100])
        ax[layer].set_xticks([500,1000])
        ax[layer].set_xticklabels(['0', '500'])
        ax[layer].tick_params(axis="x", labelsize=18)    
        # ax[layer].set_yticks([])
        ax[layer].tick_params(axis="y", labelsize=18) 
        # ax[layer].set_xlabel('time (ms)', size = 16)
        # ax.set_ylabel('LFP (uV)', size = 16)
    plt.tight_layout()
    plt.savefig('pairing MUA average per layer sharey.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('pairing MUA average per layer sharey.pdf', format = 'pdf', dpi = 1000)

    fig, ax = plt.subplots(3, 1, figsize = (6,7))
    for layer in [0,1,2]:
        ax[layer].plot(scipy.signal.filtfilt(b_notch, a_notch, np.mean(spike_pairings[:,layer_dict_1[day][0][layer][0],:], axis = 0))*1000, 'k')
        for stim in range(5):
            ax[layer].axvline([500+stim*100], color = 'k', linestyle = '--', linewidth = 0.5)
        ax[layer].set_xlim([400, 1100])
        ax[layer].set_xticks([500,1000])
        ax[layer].set_xticklabels(['0', '500'])
        ax[layer].tick_params(axis="x", labelsize=18)    
        # ax[layer].set_yticks([])
        ax[layer].tick_params(axis="y", labelsize=18) 
        # ax[layer].set_xlabel('time (ms)', size = 16)
        # ax.set_ylabel('LFP (uV)', size = 16)
    plt.tight_layout()
    plt.savefig('pairing MUA average per layer.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('pairing MUA average per layer.pdf', format = 'pdf', dpi = 1000)





    # # # CSD pairings     
    CSD_pairings = []
    CSD_matrix = -np.eye(LFP.shape[0]) # 
    for j in range(1, CSD_matrix.shape[0] - 1):
        CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
    for stim in range(LFP_pairings.shape[0]):
        CSD_pairings.append(- np.dot(CSD_matrix, scipy.ndimage.gaussian_filter(scipy.signal.filtfilt(b_notch, a_notch, LFP_pairings[stim,:,:]), (2, 0))))
    CSD_pairings = np.asarray(CSD_pairings)
    CSD_pairings[:,0,:] = 0
    CSD_pairings[:,-1,:] = 0

    # spacer = np.max(np.mean(CSD_pairings[:,3:,:], axis = 0))/1.2
    # fig, ax = plt.subplots(figsize = (3,10))
    # for ind in range(32):
    #     ax.plot(np.mean(CSD_pairings[:,ind,:], axis = 0) + ind * -spacer *np.ones(CSD_pairings.shape[2]), 'k', linewidth = 1)                 
    # for stim in range(5):
    #     ax.axvline([500+stim*100], color = 'k', linestyle = '--', linewidth = 0.5)
    # ax.set_xlim([400, 1100])
    # ax.set_xticks([500,1000])
    # ax.set_xticklabels(['0', '500'])
    # ax.tick_params(axis="x", labelsize=14)    
    # ax.set_yticks([])
    # # ax.tick_params(axis="y", labelsize=14) 
    # ax.set_xlabel('time (ms)', size = 16)
    # # ax.set_ylabel('LFP (uV)', size = 16)
    # plt.tight_layout()
    # # plt.savefig('pairing CSD average.jpg', format = 'jpg', dpi = 1000)
    # # plt.savefig('pairing CSD average.pdf', format = 'pdf', dpi = 1000)
    
    # fig, ax = plt.subplots(3, 1, sharey = True, figsize = (6,7))
    # for layer in [0,1,2]:
    #     ax[layer].plot(scipy.signal.filtfilt(b_notch, a_notch, np.mean(CSD_pairings[:,layer_dict_1[day][0][layer][0],:], axis = 0)), 'k')
    #     for stim in range(5):
    #         ax[layer].axvline([500+stim*100], color = 'k', linestyle = '--', linewidth = 0.5)
    #     ax[layer].set_xlim([400, 1100])
    #     ax[layer].set_xticks([500,1000])
    #     ax[layer].set_xticklabels(['0', '500'])
    #     ax[layer].tick_params(axis="x", labelsize=18)    
    #     # ax[layer].set_yticks([])
    #     ax[layer].tick_params(axis="y", labelsize=18) 
    #     # ax[layer].set_xlabel('time (ms)', size = 16)
    #     # ax.set_ylabel('LFP (uV)', size = 16)
    # plt.tight_layout()
    # plt.savefig('pairing CSD average per layer sharey.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('pairing CSD average per layer sharey.pdf', format = 'pdf', dpi = 1000)

    # fig, ax = plt.subplots(3, 1, figsize = (6,7))
    # for layer in [0,1,2]:
    #     ax[layer].plot(scipy.signal.filtfilt(b_notch, a_notch, np.mean(CSD_pairings[:,layer_dict_1[day][0][layer][0],:], axis = 0)), 'k')
    #     for stim in range(5):
    #         ax[layer].axvline([500+stim*100], color = 'k', linestyle = '--', linewidth = 0.5)
    #     ax[layer].set_xlim([400, 1100])
    #     ax[layer].set_xticks([500,1000])
    #     ax[layer].set_xticklabels(['0', '500'])
    #     ax[layer].tick_params(axis="x", labelsize=18)    
    #     # ax[layer].set_yticks([])
    #     ax[layer].tick_params(axis="y", labelsize=18) 
    #     # ax[layer].set_xlabel('time (ms)', size = 16)
    #     # ax.set_ylabel('LFP (uV)', size = 16)
    # plt.tight_layout()
    # plt.savefig('pairing CSD average per layer.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('pairing CSD average per layer.pdf', format = 'pdf', dpi = 1000)

    
    

    # cl()
    
    os.chdir('..')
    os.chdir('..')
