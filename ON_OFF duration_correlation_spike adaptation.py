# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 19:58:13 2023

@author: Mann Lab
"""

import numpy as np
import matplotlib.pyplot as plt
# import neo
import quantities as pq
import elephant
import scipy
import scipy.signal
import os
import copy
import pickle
# import natsort
from statistics import mean
# import xml.etree.ElementTree as ET
# from load_intan_rhd_format import *
from operator import itemgetter
import pandas as pd
import matplotlib.colors as colors
from matplotlib.pyplot import cm
from scipy import stats
import random

if os.path.isdir(r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS'):
    overall_path = r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS'
    figures_path = r'D:\JP OneDrive\OneDrive\Dokumente\SWS\Figures'

else:
    overall_path = r'C:\One_Drive\OneDrive\Dokumente\SWS\FOR_ANALYSIS'
    figures_path = r'C:\One_Drive\OneDrive\Dokumente\SWS\Figures'

# day = os.getcwd()[-6:]
# if os.path.exists(f'analysis_{day}') == False:
#     os.mkdir(f'analysis_{day}')

fs = 30000
resample_factor = 30
new_fs = fs/resample_factor

chanMap_32 = np.array([31,32,30,33,27,36,26,37,16,47,18,38,17,46,29,39,19,45,28,40,20,44,22,41,21,34,24,42,23,43,25,35]) - 16
chanMap_16 = np.linspace(0, 15, 16).astype(int)

#coordinates of electrodes, to be given as a list of lists (in case of laminar 1 coord per point):
coordinates = [[i] for i in list(np.linspace(0, 1.55, 32))]*pq.mm

def smooth(y, box_pts, axis = 0):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.apply_along_axis(lambda m: np.convolve(m, box, mode='same'), axis = axis, arr = y)
    return y_smooth

def cl():
    plt.close('all')

#%%
# list of lists: 10 sweeps with 5 layers each (1, 2/3, 4, 5, 6)
# layer map for every mouse (approximate)
mouse_1 = list(map(np.asarray, [[1,2], [3,4,5,6], [7,8,9,10,11,12], [13,14,15,16,17,18], [19,20,21,22,23,24,25,26]]))
mouse_2 = list(map(np.asarray, [[4,5,6,7,8], [9,10,11,12], [13,14,15,16,17], [18,19,20,21], [22,23,24,25]]))
mouse_3 = list(map(np.asarray, [[2,3,4], [5,6,7,8,9], [10,11,12,13], [14,15,16,17,18,19], [20,21,22,23,24]]))
mouse_4 = list(map(np.asarray, [[1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18], [19,20,21,22,23]]))
mouse_5 = list(map(np.asarray, [[1,2], [3,4,5,6,7,8], [9,10,11,12,13], [14,15,16,17], [18,19,20,21,22,23]]))
mouse_6 = list(map(np.asarray, [[1,2], [3,4,5,6,7,8,9], [10,11,12,13,14], [15,16,17,18], [19,20,21,22]]))
mouse_7 = list(map(np.asarray, [[1,2], [3,4,5], [6,7,8], [9,10,11,12,13], [14]]))
mouse_8 = list(map(np.asarray, [[1], [2,3,4,5], [6,7,8], [9,10,11,12,13], [14]]))
mouse_9 = list(map(np.asarray, [[1,2], [3,4,5], [6,7,8], [9,10,11,12,13], [14]]))
mouse_10 = list(map(np.asarray, [[1,2], [3,4,5,6], [7,8], [9,10,11,12,13], [14]]))
mouse_11 = list(map(np.asarray, [[1,2], [3,4,5], [6,7,8], [9,10,11,12], [13,14]]))
mouse_12 = list(map(np.asarray, [[1], [2,3,4,5], [6,7,8,9], [10,11,12,13], [14]]))
mouse_13 = list(map(np.asarray, [[1], [2,3,4,5], [6,7,8,9], [10,11,12,13], [14]]))
mouse_14 = list(map(np.asarray, [[1], [2,3,4,5], [6,7,8,9], [10,11,12,13], [14]]))

layer_dict = {'160614' : [mouse_1]*10,
                  
            # this mouse drifted one down after pairing
            '160615' : [mouse_2]*4 + [[i + 1 for i in mouse_2]]*6,
            
            '160622' : [mouse_3]*10,
                        
            '160728' : [mouse_4]*10,
            
            #this mouse drifted quite a bit
            '160729' : [mouse_5]*5 + [[i + 1 for i in mouse_5]]*1 + [[i + 2 for i in mouse_5]]*1 + [[i + 3 for i in mouse_5]]*3,
            
            # '160810' : [mouse_6]*4 + [[i + 1 for i in mouse_6]]*6,
            '160810' : [mouse_6]*10,

            '220810_2' : [mouse_7]*10,
            
            '221018_1' : [mouse_8]*10,
            
            '221021_1' : [mouse_9]*10,

            '221024_1' : [mouse_10]*10,

            '221025_1' : [mouse_11]*10,

            '221026_1' : [mouse_12]*10,
            
            '221206_2' : [mouse_13]*10,
            
            '221206_1' : [mouse_14]*10

            }

layer_list_LFP = list(layer_dict.values())
layer_list_CSD = copy.deepcopy(layer_list_LFP)



# ONE CHANNEL PER LAYER. 
#LAYER 2: BIGGEST CSD PEAK FROM WHISKER STIM. LAYER 4: EARLIEST CSD DEFLECTION FROM WHISKER STIM. LAYER 5: EARLIEST DEFLECTION FROM SW CSD.
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
mouse_13_1 = list(map(np.asarray, [[6], [10], [13], [15]]))
mouse_14_1 = list(map(np.asarray, [[5], [8], [12], [15]]))

layer_dict_1 = {'160614' : [mouse_1_1]*10,
                  
            # this mouse drifted one down after pairing
            '160615' : [mouse_2_1]*4 + [[i + 1 for i in mouse_2_1]]*6,
            
            '160622' : [mouse_3_1]*10,
                        
            '160728' : [mouse_4_1]*10,
            
            #this mouse drifted quite a bit
            '160729' : [mouse_5_1]*5 + [[i + 1 for i in mouse_5_1]]*1 + [[i + 2 for i in mouse_5_1]]*1 + [[i + 3 for i in mouse_5_1]]*3,
            
            # '160810' : [mouse_6_1]*4 + [[i + 1 for i in mouse_6_1]]*6,
            '160810' : [mouse_6_1]*10,

            '220810_2' : [mouse_7_1]*10,
            
            '221018_1' : [mouse_8_1]*10,
            
            '221021_1' : [mouse_9_1]*10,

            '221024_1' : [mouse_10_1]*10,

            '221025_1' : [mouse_11_1]*10,

            '221026_1' : [mouse_12_1]*10,
            
            '221206_2' : [mouse_13_1]*10,
            
            '221206_1' : [mouse_14_1]*10
            }

layer_list_LFP_1 = list(layer_dict_1.values())
layer_list_CSD_1 = copy.deepcopy(layer_list_LFP_1)


# PSTH channels for analysis
mouse_1_PSTH = list(map(np.asarray, [[4,5,6], [9,10,11], [16,17,18], [22]]))
mouse_2_PSTH = list(map(np.asarray, [[10,11,12], [13,14,15], [20,21,22], [24]])) # layer 2/3 could also be 10
mouse_3_PSTH = list(map(np.asarray, [[7,8], [10,11,12], [16,17,18], [22]]))
mouse_4_PSTH = list(map(np.asarray, [[6,7,8], [10,11,12], [15,16,17], [21]])) # layer 2/3 could also be 5,6,7
mouse_5_PSTH = list(map(np.asarray, [[4,5,6], [8,9,10], [13,14,15], [19]])) # layer 2/3 could also be 4
mouse_6_PSTH = list(map(np.asarray, [[6,7], [8,9], [13,14,15], [17]])) # layer 2/3 could be 4,5,6 SW CSD a bit unclear
mouse_7_PSTH = list(map(np.asarray, [[3,4,5], [6,7,8], [11,12,13], [14]])) # layer 5 could also be 11
mouse_8_PSTH = list(map(np.asarray, [[3,4,5], [6,7,8], [11,12,13], [14]]))
mouse_9_PSTH = list(map(np.asarray, [[2,3,4], [6,7,8], [10,11,12], [14]]))
mouse_10_PSTH = list(map(np.asarray, [[3,4,5], [6,7,8], [11,12], [14]])) #layer 2/3 could also be 4
mouse_11_PSTH = list(map(np.asarray, [[3,4,5], [6,7,8], [11,12], [14]]))
mouse_12_PSTH = list(map(np.asarray, [[3,4,5], [7,8], [10,11,12], [14]]))
mouse_13_PSTH = list(map(np.asarray, [[5,6,7], [9,10,11], [13,14,15], [15]]))
mouse_14_PSTH = list(map(np.asarray, [[4,5,6], [7,8,9], [11,12,13], [15]]))

layer_dict_PSTH = {'160614' : [mouse_1_PSTH]*10,
                  
            # this mouse drifted one down after pairing
            '160615' : [mouse_2_PSTH]*4 + [[i + 1 for i in mouse_2_PSTH]]*6,
            
            '160622' : [mouse_3_PSTH]*10,
                        
            '160728' : [mouse_4_PSTH]*10,
            
            #this mouse drifted quite a bit : next time i will put it on a lead <3 i love ALEXANDRA STANLEY. SHES THE BESTEST
            '160729' : [mouse_5_PSTH]*5 + [[i + 1 for i in mouse_5_PSTH]]*1 + [[i + 2 for i in mouse_5_PSTH]]*1 + [[i + 3 for i in mouse_5_PSTH]]*3,
            
            # '160810' : [mouse_6_PSTH]*4 + [[i + 1 for i in mouse_6_PSTH]]*6,
            '160810' : [mouse_6_PSTH]*10,

            '220810_2' : [mouse_7_PSTH]*10,
            
            '221018_1' : [mouse_8_PSTH]*10,
            
            '221021_1' : [mouse_9_PSTH]*10,

            '221024_1' : [mouse_10_PSTH]*10,

            '221025_1' : [mouse_11_PSTH]*10,

            '221026_1' : [mouse_12_PSTH]*10,
            
            '221206_2' : [mouse_13_PSTH]*10,
            
            '221206_1' : [mouse_14_PSTH]*10
            }

layer_list_PSTH = list(layer_dict_PSTH.values())

# fig, ax = plt.subplots()
# ax.plot(highpass[300000:600000])
# ax.axhline(-5*std)

use_kilosort = False
highpass_cutoff = 4


#%% extract spikes and resample nostim folder

for highpass_cutoff in [3]:
    os.chdir(r'G:\Antonia_in_vivo\Linear_probe\Up_state_paired')
    days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
    for day in days:
        os.chdir(r'G:\Antonia_in_vivo\Linear_probe\Up_state_paired')
    # for day_ind, day in enumerate(['160615']):
        os.chdir(day) 
        print(day)
        no_stim_folder = [s for s in os.listdir() if 'nowhisker' in s][0]
        os.chdir(no_stim_folder)
        # if os.path.isfile('MUA_power_binned'):
        #     os.chdir('..')
        #     os.chdir('..')
        #     continue
        channels = [s for s in os.listdir() if 'amp' in s and '.dat' in s]
        curr_spikes = {}
        for ind_channel, channel in enumerate(channels):
            print(channel)
            with open(channel,'rb') as f:
                curr_LFP = np.fromfile(f, np.int16)
                
                # # get highpass band
                MUA_power = elephant.signal_processing.butter(curr_LFP, highpass_frequency = 200, lowpass_frequency = 1500, sampling_frequency=30000)
                #binning
                timestamps = np.linspace(0, MUA_power.shape[0], MUA_power.shape[0])
                timeseries_df = pd.DataFrame({"Timestamps": timestamps, "Values": MUA_power})
                timeseries_df["Bins"] = pd.cut(timeseries_df["Timestamps"], int(MUA_power.shape[0]/30)) 
                curr_MUA_power_binned = timeseries_df.groupby("Bins").mean()["Values"].to_numpy()
    
                # # take out spikes
                # if day == '160615': # massive artifacts in the nostim after 4 minutes
                #     highpass = elephant.signal_processing.butter(curr_LFP[:218000*resample_factor], highpass_frequency = 250, sampling_frequency=30000)
                #     std = np.std(highpass)
                #     # ax.plot(highpass + ind_channel*1000)
                #     # ax.axhline(-highpass_cutoff*std + ind_channel*1000, linestyle = '--')
                # else:
                #     highpass = elephant.signal_processing.butter(curr_LFP, highpass_frequency = 250, sampling_frequency=30000)
                #     std = np.std(highpass)
                
                # #the highpass cutoff you choose is not that important, you'll just get more noise in the end but maybe not stupid for the slow wave spiking
                # crossings = np.argwhere(highpass<-highpass_cutoff*std)
                # # take out values within half a ms of each other
                # crossings = crossings[np.roll(crossings,-1) - crossings > 20]
                # curr_spikes[channel[-6:-4]] = crossings/resample_factor
                
                # #resample for LFP. SHOULD TRY DECIMATE AS WELL AND SEE IF ITS DIFFERENT. scipy resample just deletes all frequencies above Nyquist resample frequency, it doesnt apply a filter like decimate does
                # curr_LFP = scipy.signal.resample(curr_LFP, int(np.ceil(len(curr_LFP)/resample_factor)))
                
                # fig, ax = plt.subplots(3,1)
                # ax[0].plot(np.linspace(0, curr_MUA_power_binned.shape[0], curr_MUA_power_binned.shape[0])/1000, scipy.ndimage.gaussian_filter1d(log(curr_MUA_power_binned), 100))
                # ax[1].plot(np.linspace(0, curr_LFP.shape[0], curr_LFP.shape[0])/1000, curr_LFP)
                # ax[2].hist(scipy.ndimage.gaussian_filter1d(np.log(curr_MUA_power_binned), 80), bins = 1000)
    
                
            if ind_channel == 0:
                # LFP = curr_LFP
                MUA_power_binned = curr_MUA_power_binned
            elif ind_channel > 0:                
                # LFP = np.vstack((LFP,curr_LFP))
                MUA_power_binned = np.vstack((MUA_power_binned, curr_MUA_power_binned))
    
        # pickle.dump(LFP, open('LFP_resampled_nostim','wb'))
        pickle.dump(MUA_power_binned, open('MUA_power_binned','wb'))
        # pickle.dump(curr_spikes, open(f'spikes_nostim_{highpass_cutoff}','wb'))
        
        # save in the onedrive folder 
        # os.chdir(r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS\LAMINAR_UP')
        # os.chdir(day)
        # pickle.dump(curr_spikes, open(f'spikes_nostim_{highpass_cutoff}','wb'))

    # if os.path.isfile('board-DIN-00.dat'):
    #     stimfile = 'board-DIN-00.dat'   
    # elif os.path.isfile('board-DIGITAL-IN-00.dat'):
    #     stimfile = 'board-DIGITAL-IN-00.dat'
    # else:
    #     raise KeyError('no stim file')
    # with open(stimfile,'rb') as f:
    #         curr_stims = np.fromfile(f, np.int16)
    #         curr_stims = np.where(np.diff(curr_stims) == 1)[0]/resample_factor

    # pickle.dump(curr_stims, open('stim_times_nostim', 'wb'))
    
    # os.chdir('..')
    # os.chdir('..')

#%% define ON OFF states using log(MUA)

# highpass_cutoff = 5
smooth_over = 15

# save values for each animal in that list
ON_durations_ALL = []
OFF_durations_ALL = []
auto_corr_ALL = []

ON_start_spikes_ALL = []
ON_stop_spikes_ALL = []
ON_start_spikes_rel_ALL = [] # relative to average spike rate during recording
ON_stop_spikes_rel_ALL = [] # relative to average spike rate during recording

ON_start_number_for_adaptation_ALL = []

exclude_outliers = True # exclude outlier ON/OFF state for correlation of durations

ON_duration_minimum_for_spikerates = 400 # how long does ON state have to be for it to be included in spike adaptation analysis? how long is a piece of string jp?
ON_start_spikes_duration = 200 # how many ms to extract spike rates from at the beginning of ON states?
ON_stop_spikes_duration = 200
spike_number_cutoff_adaptation = 1
start_stop_margin = 50 #how much before and after ON states to take out with the spikes


# for cond in ['UP_pairing', 'DOWN_pairing']:
    # os.chdir(os.path.join(overall_path, fr'{cond}'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

# use days with a nowhisker stim
for day_ind, day in enumerate(days[0:6]):
# for day_ind, day in enumerate(['160614']):
    print(day)
    os.chdir(day)
    no_stim_folder = [s for s in os.listdir() if 'nowhisker' in s][0]
    os.chdir(no_stim_folder)
    
    if day == '160614':
        highpass_cutoff = 3
    elif day == '160514':
        highpass_cutoff = 4
    else:
        highpass_cutoff = 4

    if day == '160615': # massive artifacts after 4 minutes
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:218000]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:218000]**2
        spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 2
    else:
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]**2
        spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 1.3

    # take log and smooth MUA power across time
    MUA_power_binned_log_smoothed = scipy.ndimage.gaussian_filter(np.emath.logn(log_base,MUA_power_binned), (0,80))
    
    # normalize within each channel, take median value in each channel
    MUA_power_binned_rel = (MUA_power_binned.T/np.median(MUA_power_binned, axis = 1)).T
    MUA_power_binned_log_smoothed_rel = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_power_binned_rel), (0,80))
    
    # plot a snippet of MUA power of each channel
    # fig, ax = plt.subplots()
    # for i in range(MUA_power_binned_log_smoothed.shape[0]):
    #     ax.plot(MUA_power_binned_log_smoothed[i,0:20000] - i*np.ones_like(MUA_power_binned_log_smoothed[i,0:20000]), linewidth = 0.5)

    # plot MUA power histogram for each channel to decide which is best bimodal
    # fig, ax = plt.subplots(figsize = (2,12))
    # fig.suptitle(f'{day}')
    # spacer = 3000
    # tot_chans = MUA_power_binned_log_smoothed.shape[0]
    # for i in range(MUA_power_binned_log_smoothed.shape[0]):
    #     ax.plot(np.histogram(MUA_power_binned_log_smoothed[i,:], bins = 500)[0] - i*3000*np.ones_like(np.histogram(MUA_power_binned_log_smoothed[i,:], bins = 500)[0]), linewidth = 1)
    #     ax.set_yticks(np.linspace(-(spacer*((tot_chans - 1))), 0, tot_chans))
    #     ax.set_yticklabels(np.linspace(((tot_chans - 1)), 0, tot_chans).astype(int), size = 6)
    #     plt.tight_layout()
    
    # use a layer 5 channel log MUA to detect UP vs DOWN states. This is the number of channel from layer 1 onwards, not from the channel map!
    if day == '160614':
        chan_for_MUA = 17
    if day == '160615':
        chan_for_MUA = 24
    if day == '160622':
        chan_for_MUA = 21
    if day == '160728':
        chan_for_MUA = 21
    if day == '160729':
        chan_for_MUA = 19
    if day == '160810':
        chan_for_MUA = 16

    if day == '160614':
        SW_spiking_channels = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    if day == '160615':
        SW_spiking_channels = np.array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25])
    if day == '160622':
        SW_spiking_channels = np.array([6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    if day == '160728':
        SW_spiking_channels = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23])
    if day == '160729':
        SW_spiking_channels = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23])
    if day == '160810':
        SW_spiking_channels = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    
    if day == '160614':
        bimodal_channels = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    if day == '160615':
        bimodal_channels = [14,16,17,18,19,20,21,22,23,24,25,26,29,30,31]
    if day =='160622':
        bimodal_channels = [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
    if day == '160728':
        bimodal_channels = [12,14,15,16,17,18,19,20,21,22,23,24,25]
    if day == '160729':
        bimodal_channels = [8,10,11,12,13,14,15,16,17,18,19,21,22,23]
    if day == '160810':
        bimodal_channels = [9,10,11,12,13,14,15,16,17,18,19,20]
        
        
    # take peaks of bimodal MUA power distribution
    # OFF_peak = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(MUA_power_binned_log_smoothed[chan_for_MUA,:], bins = 1000)[0], 15))[0][0]
    # ON_peak = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(MUA_power_binned_log_smoothed[chan_for_MUA,:], bins = 1000)[0], 15))[0][1]
    # OFF_value = np.histogram(MUA_power_binned_log_smoothed[chan_for_MUA,:], bins = 1000)[1][OFF_peak]
    # ON_value = np.histogram(MUA_power_binned_log_smoothed[chan_for_MUA,:], bins = 1000)[1][ON_peak]
    bimodal_avg_rel_MUA = np.mean(MUA_power_binned_log_smoothed_rel[bimodal_channels,:], axis = 0)
    OFF_peak = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(bimodal_avg_rel_MUA, bins = 1000)[0], smooth_over), distance = 90)[0][0]
    OFF_value = np.histogram(bimodal_avg_rel_MUA, bins = 1000)[1][OFF_peak_avg]
    # if day == '160615' or day == '160729' or day == '160810':
    if day == '220810_2':
    #     pass
        OFF_ON_threshold = OFF_value + np.abs(1*(OFF_value-np.histogram(bimodal_avg_rel_MUA, bins = 1000)[1][0]))
    else:
        ON_peak = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(bimodal_avg_rel_MUA, bins = 1000)[0], smooth_over), distance = 90)[0][1]
        ON_value = np.histogram(bimodal_avg_rel_MUA, bins = 1000)[1][ON_peak]
        # OFF_ON_threshold is about halfway between both peaks (varies in Nghiem and Sanchez-vives)
        OFF_ON_threshold = OFF_value + 0.6*(ON_value - OFF_value)

    
    # # # plot bimodal distribution of that channel smoothed
    # fig, ax = plt.subplots(4,1)
    # fig.suptitle(f'{day}')
    # ax[0].plot(np.linspace(0, MUA_power_binned_log_smoothed.shape[1], MUA_power_binned_log_smoothed.shape[1])/1000, MUA_power_binned_log_smoothed[chan_for_MUA,:])
    # ax[1].plot(np.linspace(0, LFP.shape[1], LFP.shape[1])/1000, LFP[chan_for_MUA,:])
    # ax[2].hist(MUA_power_binned_log_smoothed[chan_for_MUA,:], bins = 1000)
    # ax[3].plot(np.histogram(MUA_power_binned_log_smoothed[chan_for_MUA,:], bins = 1000)[1][:-1], scipy.ndimage.gaussian_filter1d(np.histogram(MUA_power_binned_log_smoothed[chan_for_MUA,:], bins = 1000)[0], 20), color = 'k')    
    # ax[3].axvline(OFF_ON_threshold, color = 'red')    

    # # plot bimodal distribution
    fig, ax = plt.subplots(4,1)
    fig.suptitle(f'{day}')
    ax[0].plot(np.linspace(0, MUA_power_binned_log_smoothed.shape[1], MUA_power_binned_log_smoothed.shape[1])/1000, bimodal_avg_rel_MUA)
    ax[1].plot(np.linspace(0, LFP.shape[1], LFP.shape[1])/1000, LFP[chan_for_MUA,:])
    ax[2].hist(bimodal_avg_rel_MUA, bins = 1000)
    ax[3].plot(np.histogram(bimodal_avg_rel_MUA, bins = 1000)[1][:-1], scipy.ndimage.gaussian_filter1d(np.histogram(bimodal_avg_rel_MUA, bins = 1000)[0], smooth_over), color = 'k')    
    ax[3].axvline(OFF_ON_threshold, color = 'red')    


    OFF_duration_threshold = 50
    ON_duration_threshold = 50

    # ON_states_starts = np.where(np.diff((MUA_power_binned_log_smoothed[chan_for_MUA,:] < OFF_ON_threshold).astype(int)) == -1)[0]/new_fs
    # ON_states_stops = np.where(np.diff((MUA_power_binned_log_smoothed[chan_for_MUA,:] < OFF_ON_threshold).astype(int)) == 1)[0]/new_fs
    ON_states_starts = np.where(np.diff((bimodal_avg_rel_MUA < OFF_ON_threshold).astype(int)) == -1)[0]/new_fs
    ON_states_stops = np.where(np.diff((bimodal_avg_rel_MUA < OFF_ON_threshold).astype(int)) == 1)[0]/new_fs

    #nake sure the first ON_state start is before the first ON_state stop
    ON_states_starts = ON_states_starts[ON_states_starts<ON_states_stops[-1]]
    ON_states_stops = ON_states_stops[ON_states_stops>ON_states_starts[0]]
    
    # take out ON or OFF states that are too short
    shorts = []
    for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts, ON_states_stops)):
        if ON_stop - ON_start < ON_duration_threshold/1000:
            shorts.append(i)
    ON_states_starts = np.delete(ON_states_starts, shorts)
    ON_states_stops = np.delete(ON_states_stops, shorts)
    
    # # plot all ON states in red
    # fig, ax = plt.subplots(2,1, sharex = True)
    # fig.suptitle(f'{day}')
    # ax[0].plot(np.linspace(0, MUA_power_binned_log_smoothed.shape[1], MUA_power_binned_log_smoothed.shape[1])/1000, MUA_power_binned_log_smoothed[chan_for_MUA,:])
    # ax[1].plot(np.linspace(0, LFP.shape[1], LFP.shape[1])/1000, LFP[chan_for_MUA,:])
    # for ON_start, ON_stop in zip(ON_states_starts, ON_states_stops):
    #     ax[0].axvspan(ON_start, ON_stop, color = 'red', alpha = 0.1)
    #     ax[1].axvspan(ON_start, ON_stop, color = 'red', alpha = 0.1)
    
    # calculate ON and OFF durations. 
    # THE OFF DURATION HERE IS FIRST
    ON_durations = (ON_states_stops - ON_states_starts)[1:]
    OFF_durations = ON_states_starts[1:] - ON_states_stops[:-1]
    ON_durations_ALL.append(ON_durations)
    OFF_durations_ALL.append(OFF_durations)




    # spike adaptation for all channels separately
    ON_start_spikes_all_chans = []
    ON_stop_spikes_all_chans = []
    for chan in SW_spiking_channels:
        spikes_for_adaptation = list(spikes.values())[np.argwhere(chanMap_32 == chan)[0][0]] # MUA from the channel used for logMUA detection
        ON_start_spikes = []
        ON_stop_spikes = []
        for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts*1000, ON_states_stops*1000)):
            on_spikes = spikes_for_adaptation[np.searchsorted(spikes_for_adaptation, ON_start -start_stop_margin) : np.searchsorted(spikes_for_adaptation, ON_stop + start_stop_margin)] - ON_start #spikes during that ON state, with ON_start = time 0
            # print(on_spikes.size)
            # extract spike ON-start and ON-end PSTHs, only in ON states that are long enough and there is enough spiking during the ON state
            if ON_stop - ON_start > ON_duration_minimum_for_spikerates:
                ON_start_spikes.append(np.histogram(on_spikes[on_spikes < ON_start_spikes_duration], bins = np.linspace(-start_stop_margin, ON_start_spikes_duration, ON_start_spikes_duration + start_stop_margin))[0])
                ON_stop_spikes.append(np.histogram(on_spikes[on_spikes > int(ON_stop - ON_start) - ON_stop_spikes_duration], bins = np.linspace(int(ON_stop - ON_start) - ON_stop_spikes_duration, int(ON_stop - ON_start + start_stop_margin), ON_stop_spikes_duration + start_stop_margin))[0])
        ON_start_spikes_all_chans.append(ON_start_spikes)
        ON_stop_spikes_all_chans.append(ON_stop_spikes)
    ON_start_spikes_all_chans = np.asarray(ON_start_spikes_all_chans)
    ON_stop_spikes_all_chans = np.asarray(ON_stop_spikes_all_chans)


    # spike adaptation for compressed MUA across chans
    # spikes_for_adaptation = np.sort(np.concatenate(list(spikes.values()))) # all MUA across all channels of the laminar probe
    # spikes_for_adaptation = np.sort(np.concatenate([i for i_ind, i in enumerate(list(spikes.values())) if np.argwhere(chanMap_32 == i_ind)[0][0] in SW_spiking_channels])) # all MUA across all CORTICAL channels of the laminar probe
    spikes_for_adaptation = np.sort(np.concatenate([i for i_ind, i in enumerate(list(spikes.values())) if np.argwhere(chanMap_32 == i_ind)[0][0] in SW_spiking_channels[SW_spiking_channels > 10]])) # all MUA across all channels in layer 5/6
    avg_spike_rate = len(spikes_for_adaptation)/(LFP.shape[1])
    avg_spike_rate = np.mean(np.histogram(spikes_for_adaptation, bins = LFP.shape[1])[0])

    ON_start_number_for_adaptation = 0
    ON_start_spikes = []
    ON_stop_spikes = []
    for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts*1000, ON_states_stops*1000)):
        on_spikes = spikes_for_adaptation[np.searchsorted(spikes_for_adaptation, ON_start -start_stop_margin) : np.searchsorted(spikes_for_adaptation, ON_stop + start_stop_margin)] - ON_start #spikes during that ON state, with ON_start = time 0
        # print(on_spikes.size)
        # extract spike ON-start and ON-end PSTHs, only in ON states that are long enough and there is enough spiking during the ON state
        if ON_stop - ON_start > ON_duration_minimum_for_spikerates and on_spikes.size > spike_number_cutoff_adaptation:
            ON_start_spikes.append((np.histogram(on_spikes[on_spikes < ON_start_spikes_duration], bins = np.linspace(-start_stop_margin, ON_start_spikes_duration, ON_start_spikes_duration + start_stop_margin))[0]))
            ON_stop_spikes.append((np.histogram(on_spikes[on_spikes > int(ON_stop - ON_start) - ON_stop_spikes_duration], bins = np.linspace(int(ON_stop - ON_start) - ON_stop_spikes_duration, int(ON_stop - ON_start + start_stop_margin), ON_stop_spikes_duration + start_stop_margin))[0]))
            ON_start_number_for_adaptation += 1
        
    ON_start_spikes_ALL.append(np.asarray(ON_start_spikes))
    ON_stop_spikes_ALL.append(np.asarray(ON_stop_spikes))
    ON_start_spikes_rel_ALL.append(np.asarray(ON_start_spikes)/avg_spike_rate)
    ON_stop_spikes_rel_ALL.append(np.asarray(ON_stop_spikes)/avg_spike_rate)
    ON_start_number_for_adaptation_ALL.append(ON_start_number_for_adaptation)




    #  -------------------------------------------------------------- plotting spike frequency adaptation during ON states for individual mouse ------------------------------------------------------
    
    
    # fig, ax = plt.subplots(figsize = (4,4))
    # fig.suptitle(f'{day}')
    # ax.plot(np.arange(- start_stop_margin, ON_start_spikes_duration-1, 1), scipy.ndimage.gaussian_filter1d(np.nanmean(np.asarray(ON_start_spikes), axis = 0)*1000/len(SW_spiking_channels[SW_spiking_channels > 10]), 2), 'k')
    # ax.tick_params(axis="x", labelsize=16)    
    # ax.tick_params(axis="y", labelsize=16) 
    # ax.set_xlabel('time from ON start (ms)', size = 14)
    # ax.set_ylabel('mean instantaneous MUA in layer 5/6', size = 14)
    # ax.set_ylim(bottom = 0)
    # ax.set_ylim(top = np.max(scipy.ndimage.gaussian_filter1d(np.nanmean(np.asarray(ON_start_spikes), axis = 0)*1000, 2)) + 5)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.axhline(avg_spike_rate*1000)
    # plt.tight_layout()
    # plt.savefig('ON start spike rate based on log(MUA).jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('ON start start spike rate based on log(MUA).pdf', dpi = 1000, format = 'pdf')

    # fig, ax = plt.subplots(figsize = (4,4))
    # ax.plot(np.arange(- ON_stop_spikes_duration, start_stop_margin-1, 1), scipy.ndimage.gaussian_filter1d(np.nanmean(np.asarray(ON_stop_spikes), axis = 0)*1000/len(SW_spiking_channels[SW_spiking_channels > 10]), 2), 'k')
    # ax.tick_params(axis="x", labelsize=16)    
    # ax.tick_params(axis="y", labelsize=16) 
    # ax.set_xlabel('time from ON stop (ms)', size = 14)
    # ax.set_ylabel('mean instantaneous MUA in layer 5/6', size = 14)
    # ax.set_ylim(top = np.max(scipy.ndimage.gaussian_filter1d(np.nanmean(np.asarray(ON_start_spikes), axis = 0)*1000, 2)) + 5)
    # ax.set_ylim(bottom = 0)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.axhline(avg_spike_rate*1000)
    # plt.tight_layout()
    # plt.savefig('ON end spike rate based on log(MUA).jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('ON end spike rate based on log(MUA).pdf', dpi = 1000, format = 'pdf')
    # print(ON_start_number_for_adaptation)
    
    # # spike freq adaptation for each channel 
    # colors = cm.rainbow(np.linspace(0, 1, len(SW_spiking_channels)))
    # fig, ax = plt.subplots(2,1, sharey = True, figsize = (4,8))
    # fig.suptitle(f'{day}')
    # for chan in range(len(SW_spiking_channels)):
    #     ax[0].plot(np.arange(- start_stop_margin, ON_start_spikes_duration-1, 1), scipy.ndimage.gaussian_filter1d(np.nanmean(ON_start_spikes_all_chans[chan,:,:], axis = 0)*1000, 1), linewidth = 0.5, label = f'{SW_spiking_channels[chan]}', color = colors[chan])
    #     ax[1].plot(np.arange(- start_stop_margin, ON_start_spikes_duration-1, 1), scipy.ndimage.gaussian_filter1d(np.nanmean(ON_stop_spikes_all_chans[chan,:,:], axis = 0)*1000, 1), linewidth = 0.5, label = f'{SW_spiking_channels[chan]}', color = colors[chan])
    # ax[0].tick_params(axis="x", labelsize=16)    
    # ax[0].tick_params(axis="y", labelsize=16) 
    # ax[0].set_xlabel('time from ON start', size = 14)
    # ax[0].set_ylabel('mean instantaneous firing rate', size = 14)
    # ax[0].set_ylim(bottom = 0)
    # ax[0].legend()
    # ax[1].tick_params(axis="x", labelsize=16)    
    # ax[1].tick_params(axis="y", labelsize=16) 
    # ax[1].set_xlabel('time from ON stop', size = 14)
    # ax[1].set_ylabel('mean instantaneous firing rate', size = 14)
    # plt.tight_layout()
    # plt.savefig('ON start and end spike rate based on log(MUA)', dpi = 1000)
    # print(ON_start_number_for_adaptation)

    



    #  -------------------------------------------------------------- plotting ON OFF durations and correlations for individual mouse ------------------------------------------------------

    # # distribution of ON and OFF durations
    # fig, ax = plt.subplots(2,1)
    # fig.suptitle(f'{day}')
    # ax[0].hist(ON_durations, bins = 20)
    # ax[1].hist(OFF_durations, bins = 20)
    
    # # correlation between OFF and next UP duration
    # X = OFF_durations
    # Y = ON_durations
    # X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
    # Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
    # outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
    # if exclude_outliers:
    #     X = np.delete(X, outliers)
    #     Y = np.delete(Y, outliers)
    # fig, ax = plt.subplots(2,1)
    # fig.suptitle(f'{day}')
    # slope, intercept, r_zero_lag, p, std_err = scipy.stats.linregress(X, Y)
    # print(f'{r_zero_lag} and {p} for {len(X)} states')
    # ax[0].scatter(X,Y, color = 'k')
    # ax[0].plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
    
    # can't do cross correlate with scipy in case there are some outliers in duration..., so need to calculate pearson'r R manually and exclude outliers manually...
    # cross_corr = [[] for i in range(2)]
    # for offset in range(20):
    #     if offset == 0:
    #         cross_corr[1].append(r_zero_lag)
    #     else:
    #         #backwards
    #         X = OFF_durations[:-offset]
    #         Y = ON_durations[offset:]
    #         X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
    #         Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
    #         outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
    #         if exclude_outliers:
    #             X = np.delete(X, outliers)
    #             Y = np.delete(Y, outliers)
    #         slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
    #         cross_corr[0].append(r)

    #         #forwards
    #         X = OFF_durations[offset:]
    #         Y = ON_durations[:-offset]
    #         X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
    #         Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
    #         outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
    #         if exclude_outliers:
    #             X = np.delete(X, outliers)
    #             Y = np.delete(Y, outliers)
    #         slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
    #         cross_corr[1].append(r)
    
    # # ax[1].plot(scipy.signal.correlate(X-np.mean(X), Y - np.mean(Y))/np.sqrt(np.sum((X-np.mean(X))**2)*np.sum((Y-np.mean(Y))**2)))
    # ax[1].plot(np.concatenate(cross_corr))
    # # ax[1].set_xlim([len(X)-20,len(X)+20])
    # plt.tight_layout()
    # plt.savefig('State transition synchrony', dpi = 1000)
    # cl()
    
    
    
    
    
    #  -------------------------------------------------------------- data examples logMUA with threshold, ON states in red, LFP autocorrelation ------------------------------------------------------
    # os.chdir(figures_path)

    # # # plot bimodal distribution of that channel smoothed
    # fig, ax = plt.subplots(figsize = (6,3))
    # ax.hist(MUA_power_binned_log_smoothed[chan_for_MUA,:], bins = 1000, color = 'grey')
    # ax.plot(np.histogram(MUA_power_binned_log_smoothed[chan_for_MUA,:], bins = 1000)[1][:-1], scipy.ndimage.gaussian_filter1d(np.histogram(MUA_power_binned_log_smoothed[chan_for_MUA,:], bins = 1000)[0], 20), color = 'k')    
    # ax.axvline(OFF_ON_threshold, color = 'red')
    # ax.set_xlabel('log(MUA power)', size = 16)
    # ax.set_ylabel('# samples', size = 16)
    # ax.tick_params(axis="x", labelsize=14)    
    # ax.tick_params(axis="y", labelsize=14)
    # plt.tight_layout()
    # os.chdir(figures_path)
    # ax.set_xlim(right = 25)
    # plt.savefig('log(MUA) histogram with cutoff.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('log(MUA) histogram with cutoff.pdf', dpi = 1000, format = 'pdf')
    
    # # # plot all ON states in red
    # x_lim1 = 65
    # x_lim2 = 80
    # fig, ax = plt.subplots(2,1, sharex = True, figsize = (8,4))
    # ax[0].plot(np.linspace(0, MUA_power_binned_log_smoothed.shape[1], MUA_power_binned_log_smoothed.shape[1])/1000, MUA_power_binned_log_smoothed[chan_for_MUA,:], color = 'k')
    # ax[1].plot(np.linspace(0, LFP.shape[1], LFP.shape[1])/1000, LFP[chan_for_MUA,:], color = 'k')
    # for ON_start, ON_stop in zip(ON_states_starts, ON_states_stops):
    #     ax[0].axvspan(ON_start, ON_stop, color = 'red', alpha = 0.1)
    #     ax[1].axvspan(ON_start, ON_stop, color = 'red', alpha = 0.1)
    # ax[0].set_ylim([20, 25])
    # ax[1].set_ylim([-2500, 2500])
    # ax[1].set_xlim([x_lim1,x_lim2])
    # ax[1].set_xticks(np.linspace(65,80,4))
    # ax[1].set_xticklabels(list(map(str, ax[0].get_xticks().astype(int) - x_lim1)))
    # ax[0].spines['right'].set_visible(False)
    # ax[0].spines['left'].set_visible(False)
    # ax[1].spines['right'].set_visible(False)
    # ax[1].spines['left'].set_visible(False)
    # ax[0].set_yticks([])
    # ax[1].set_yticks([])
    # ax[1].tick_params(axis="x", labelsize=16)    
    # ax[1].set_xlabel('time (s)', size = 18)
    # plt.tight_layout()
    # plt.savefig('log(MUA) and LFP.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('log(MUA) and LFP.pdf', dpi = 1000, format = 'pdf')

    # correlation between OFF and next UP duration
    # X = OFF_durations
    # Y = ON_durations
    # X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
    # Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
    # outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
    # if exclude_outliers:
    #     X = np.delete(X, outliers)
    #     Y = np.delete(Y, outliers)
    # fig, ax = plt.subplots()
    # slope, intercept, r_zero_lag, p, std_err = scipy.stats.linregress(X, Y)
    # print(f'{r_zero_lag} and {p} for {len(X)} states')
    # ax.scatter(X,Y, color = 'k')
    # ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
    # ax.set_xlabel('OFF-state duration (seconds)', size = 16)
    # ax.set_ylabel('ON-state duration (seconds)', size = 16)
    # ax.tick_params(axis="x", labelsize=14)    
    # ax.tick_params(axis="y", labelsize=14)    
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tight_layout()
    # plt.savefig('OFF vs ON duration.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('OFF vs ON duration.pdf', dpi = 1000, format = 'pdf')
    
    
    # correlation betwen OFF and n + 1 UP state
    # cross_corr = [[] for i in range(2)]
    # for offset in range(20):
    #     if offset == 0:
    #         cross_corr[1].append(r_zero_lag)
    #     else:
    #         #backwards (previous OFFs)
    #         X = OFF_durations[:-offset]
    #         Y = ON_durations[offset:]
    #         X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
    #         Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
    #         outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
    #         if exclude_outliers:
    #             X = np.delete(X, outliers)
    #             Y = np.delete(Y, outliers)
    #         slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
    #         cross_corr[0].append(r)

    #         #forwards (OFFs after)
    #         X = OFF_durations[offset:]
    #         Y = ON_durations[:-offset]
    #         X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
    #         Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
    #         outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
    #         if exclude_outliers:
    #             X = np.delete(X, outliers)
    #             Y = np.delete(Y, outliers)
    #         slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
    #         cross_corr[1].append(r)
    
    # fig, ax = plt.subplots()
    # ax.plot(np.concatenate(cross_corr))
    # ax.set_xlabel('Offset (i)', size = 16)
    # ax.set_ylabel('R($\mathregular{OFF_{n+1}}$, $\mathregular{ON_{n}}$)', size = 16)
    # ax.tick_params(axis="x", labelsize=14)    
    # ax.tick_params(axis="y", labelsize=14) 
    # ax.set_ylim([-0.15, 0.25])
    # plt.tight_layout()
    
    
    
    # autocorrelation of LFP
    # b_notch, a_notch = scipy.signal.iirnotch(50, 100, 1000)
    # time_gap = 20000
    # auto_corr = []
    # for time in range(LFP.shape[1]//time_gap-2):
    #     curr_LFP = scipy.signal.filtfilt(b_notch, a_notch, LFP[chan_for_MUA,time*time_gap:time*time_gap+time_gap])
    #     auto_corr.append(scipy.signal.correlate(curr_LFP, curr_LFP)/np.sum((curr_LFP-np.mean(curr_LFP))**2))
    # auto_corr = np.asarray(auto_corr)
    # fig, ax = plt.subplots()
    # ax.plot(auto_corr.T, color = 'k', linewidth = 0.2, alpha = 0.5) # plot autocorr of all timesteps
    # ax.plot(np.mean(auto_corr, axis = 0), color = 'k') # plot mean autocorr
    # ax.set_xlim([15000,25000])
    # ax.set_xticks([16000,18000,20000,22000,24000])
    # ax.set_xticklabels(list(map(str, ax.get_xticks()/1000 - 20)), size = 14)
    # ax.set_xlabel('time (s)', size = 16)
    # ax.set_ylabel('LFP autocorrelation', size = 16)
    # ax.tick_params(axis="x", labelsize=14)    
    # ax.tick_params(axis="y", labelsize=14) 
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tight_layout()
    # plt.savefig('LFP autocorrelation.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('LFP autocorrelation.pdf', dpi = 1000, format = 'pdf')
    # auto_corr_ALL.append(auto_corr)
    
    
    
    
    # # # ---------------------------------------------------------------------------- example heatmap of all ON/OFF states ordered by length
    # # # 1) extract matrix of LFP, CSD and MUA values in each cortical channel around every ON crossing
    # avg_MUA = np.mean(MUA_power_binned_log_smoothed_rel[bimodal_channels,:], axis = 0)

    # OFF_peak = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(avg_MUA, bins = 1000)[0], 20))[0][0]
    # ON_peak = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(avg_MUA, bins = 1000)[0], 20))[0][1]
    # OFF_value = np.histogram(avg_MUA, bins = 1000)[1][OFF_peak]
    # ON_value = np.histogram(avg_MUA, bins = 1000)[1][ON_peak]
    
    # OFF_ON_threshold = OFF_value + 0.6*(ON_value - OFF_value)
    
    # OFF_duration_threshold = 50
    # ON_duration_threshold = 50

    # ON_states_starts_avg = np.where(np.diff((avg_MUA[1000:-2000] < OFF_ON_threshold).astype(int)) == -1)[0]/new_fs + 1
    # ON_states_stops_avg = np.where(np.diff((avg_MUA[1000:-2000] < OFF_ON_threshold).astype(int)) == 1)[0]/new_fs + 1
    # ON_states_starts_avg = ON_states_starts_avg[ON_states_starts_avg<ON_states_stops_avg[-1]]
    # ON_states_stops_avg = ON_states_stops_avg[ON_states_stops_avg>ON_states_starts_avg[0]]
    
    # # take out ON or OFF states that are too short
    # shorts = []
    # for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts_avg, ON_states_stops_avg)):
    #     if ON_stop - ON_start < ON_duration_threshold/1000:
    #         shorts.append(i)
    # ON_states_starts_avg = np.delete(ON_states_starts_avg, shorts)
    # ON_states_stops_avg = np.delete(ON_states_stops_avg, shorts)
    
    # ON_durations_avg = (ON_states_stops_avg - ON_states_starts_avg)[1:]
    # OFF_durations_avg = ON_states_starts_avg[1:] - ON_states_stops_avg[:-1]

    # MUA_matrix_start = []
    # MUA_matrix_end = []
    # if len(ON_states_starts_avg) > 0:
    #     for ON_start, ON_end in zip((ON_states_starts_avg[1:]*1000).astype(int), (ON_states_stops_avg[1:]*1000).astype(int)):
    #         MUA_matrix_start.append(MUA_power_binned_log_smoothed_rel[:,ON_start - 100:ON_start + 1500])
    #         MUA_matrix_end.append(MUA_power_binned_log_smoothed_rel[:,ON_end - 100:ON_end + 1500])
    #     MUA_matrix_start = np.asarray(MUA_matrix_start)
    #     MUA_matrix_end = np.asarray(MUA_matrix_end)
    
    # ON_dur_ind = np.argsort(ON_durations_avg)
    # OFF_dur_ind = np.argsort(OFF_durations_avg[1:])
    
    # fig, ax = plt.subplots(figsize = (4,4))
    # ax.imshow(np.squeeze(np.mean(MUA_matrix_start[:,bimodal_channels,:], axis = 1)[ON_dur_ind, :]), cmap = 'jet', vmax = 1.5, aspect = 3)
    # plt.tight_layout()
    # plt.savefig('ON durations heatmap.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('ON durations heatmap.pdf', format = 'pdf', dpi = 1000)

    
    # fig, ax = plt.subplots(figsize = (4,4))
    # ax.imshow(np.squeeze(np.mean(MUA_matrix_end[:,bimodal_channels,:], axis = 1)[OFF_dur_ind, :]), cmap = 'jet', vmax = 1.5, aspect = 3)
    # plt.tight_layout()
    # plt.savefig('OFF durations heatmap.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('OFF durations heatmap.pdf', format = 'pdf', dpi = 1000)

    
    # fig, ax = plt.subplots(figsize = (1.5,5))
    # norm = colors.Normalize(vmin=-1, vmax=1.5)
    # fig.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'),
    #               cax=ax, ticks = [-1, -0.5, 0, 0.5, 1, 1.5])
    # ax.set_yticklabels(list(map(str, np.linspace(-1, 1.5, 6))), size = 16)
    # ax.set_ylabel('log(MUA)', size = 18)
    # plt.tight_layout()
    # plt.savefig('ON duration colormap legend.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('ON duration colormap legend.jpg', dpi = 1000, format = 'jpg')

    
    os.chdir('..')
    os.chdir('..')
    
pickle.dump(ON_durations_ALL, open('ON_durations_ALL', 'wb'))
pickle.dump(OFF_durations_ALL, open('OFF_durations_ALL', 'wb'))
pickle.dump(auto_corr_ALL, open('auto_corr_ALL', 'wb'))

pickle.dump(ON_start_spikes_ALL, open('ON_start_spikes_ALL', 'wb'))
pickle.dump(ON_stop_spikes_ALL, open('ON_stop_spikes_ALL', 'wb'))
pickle.dump(ON_start_spikes_rel_ALL, open('ON_start_spikes_rel_ALL', 'wb'))
pickle.dump(ON_stop_spikes_rel_ALL, open('ON_stop_spikes_rel_ALL', 'wb'))

pickle.dump(ON_start_number_for_adaptation_ALL, open('ON_start_number_for_adaptation_ALL', 'wb'))
pickle.dump(ON_start_number_for_adaptation_ALL, open('ON_start_number_for_adaptation_ALL', 'wb'))




#%% group figures: ON OFF durations, cross correlation, autocorrelation of all, spike rate adaptation

exclude_outliers = True
# os.chdir(os.path.join(overall_path, 'LAMINAR_UP'))
ON_durations_ALL = pickle.load(open('ON_durations_ALL', 'rb'))
OFF_durations_ALL = pickle.load(open('OFF_durations_ALL', 'rb'))
auto_corr_ALL = pickle.load(open('auto_corr_ALL', 'rb'))

# os.chdir(figures_path)

# # correlation between OFF and next UP across all mice and ON states (doesn't really work)
# X = np.concatenate(OFF_durations_ALL)
# Y = np.concatenate(ON_durations_ALL)
# X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
# Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
# outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
# if exclude_outliers:
#     X = np.delete(X, outliers)
#     Y = np.delete(Y, outliers)
# fig, ax = plt.subplots()
# slope, intercept, r_zero_lag, p, std_err = scipy.stats.linregress(X, Y)
# print(f'{r_zero_lag} and {p} for {len(X)} states')
# ax.scatter(X,Y, color = 'k')
# ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')


# # lag correlation between OFF and next UP across all mice 
# # do shuffle to get pearson correlation confidence interval
# lag_correlation_ALL = []
# r_shuffle_mean_ALL = []
# r_shuffle_std_ALL = []
# for i, (ON_durations, OFF_durations) in enumerate(zip(ON_durations_ALL, OFF_durations_ALL)):
#     print(i)
#     # correlation betwen OFF and n + 1 UP state
#     cross_corr = [[] for i in range(2)]
#     for offset in range(20):
#         if offset == 0:
#             X = OFF_durations
#             Y = ON_durations
#             X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
#             Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
#             outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
#             if exclude_outliers:
#                 X = np.delete(X, outliers)
#                 Y = np.delete(Y, outliers)
#             slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
#             cross_corr[1].append(r)
#             print(r)

#             #shuffle
#             r_shuffle = []
#             for i in range(1000):
#                 X_shuffle = copy.deepcopy(X)
#                 np.random.shuffle(X_shuffle)
#                 slope, intercept, r, p, std_err = scipy.stats.linregress(X_shuffle, Y)
#                 r_shuffle.append(r)
#             r_shuffle_std = np.std(r_shuffle)
#             r_shuffle_mean = np.mean(r_shuffle)
#             r_shuffle_mean_ALL.append(r_shuffle_mean)
#             r_shuffle_std_ALL.append(r_shuffle_std)
#             print(r_shuffle_mean + 2*r_shuffle_std)
#         else:
#             #backwards (previous OFFs)
#             X = OFF_durations[:-offset]
#             Y = ON_durations[offset:]
#             X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
#             Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
#             outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
#             if exclude_outliers:
#                 X = np.delete(X, outliers)
#                 Y = np.delete(Y, outliers)
#             slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
#             cross_corr[0].append(r)

#             #forwards (OFFs after)
#             X = OFF_durations[offset:]
#             Y = ON_durations[:-offset]
#             X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
#             Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
#             outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
#             if exclude_outliers:
#                 X = np.delete(X, outliers)
#                 Y = np.delete(Y, outliers)
#             slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
#             cross_corr[1].append(r)
            
#     lag_correlation_ALL.append(np.concatenate(cross_corr))
# lag_correlation_ALL = np.asarray(lag_correlation_ALL)

# fig, ax = plt.subplots()
# ax.bar(np.linspace(-19, 19, 39), np.mean(lag_correlation_ALL, axis = 0), color = 'k', yerr=np.std(lag_correlation_ALL, axis = 0)/np.sqrt(lag_correlation_ALL.shape[0]))
# ax.set_ylim([-0.1, 0.35])
# ax.set_xlabel('Offset (i)', size = 16)
# ax.set_ylabel('R($\mathregular{OFF_{n+i}}$, $\mathregular{ON_{n}}$)', size = 16)
# ax.tick_params(axis="x", labelsize=14)    
# ax.tick_params(axis="y", labelsize=14) 
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.axhline(np.mean(r_shuffle_mean_ALL) + 2*(np.mean(r_shuffle_std_ALL)), linestyle = '--', color = 'k')
# plt.tight_layout()
# plt.savefig('OFF vs ON duration LAG.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('OFF vs ON duration LAG.pdf', dpi = 1000, format = 'pdf')



# ON and OFF duration histograms
# os.chdir(figures_path)
# fig, ax = plt.subplots(figsize = (8,4.5))
# ON_durations_ALL_conc = np.concatenate(ON_durations_ALL)
# to_plot = ON_durations_ALL_conc[ON_durations_ALL_conc < 3]
# ax.hist(to_plot, bins = 40, color = 'black', width = 0.05, weights=np.ones(len(to_plot)) / len(to_plot) * 100)
# ax.set_xlabel('ON duration (second)', size = 18)
# ax.set_ylabel('% of ON states', size = 18)
# ax.tick_params(axis="x", labelsize=16)    
# ax.tick_params(axis="y", labelsize=16) 
# ax.set_xlim([0,2.6])
# ax.set_ylim([0,13])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.tight_layout()
# plt.savefig('ON state duration.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('ON state duration.pdf', dpi = 1000, format = 'pdf')

# fig, ax = plt.subplots(figsize = (8,4.5))
# OFF_durations_ALL_conc = np.concatenate(OFF_durations_ALL)
# to_plot = OFF_durations_ALL_conc[OFF_durations_ALL_conc < 3]
# ax.hist(to_plot, bins = 40, color = 'black', width = 0.05, weights=np.ones(len(to_plot)) / len(to_plot) * 100)
# ax.set_xlabel('OFF duration (second)', size = 18)
# ax.set_ylabel('% of OFF states', size = 18)
# ax.tick_params(axis="x", labelsize=16)    
# ax.tick_params(axis="y", labelsize=16) 
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xlim([0,2.6])
# ax.set_ylim([0,13])
# plt.tight_layout()
# plt.savefig('OFF state duration.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('OFF state duration.pdf', dpi = 1000, format = 'pdf')


# fig, ax = plt.subplots(figsize = (1,1))
# ax.boxplot([np.std(i)/np.mean(i) for i in ON_durations_ALL], widths = 0.5)
# ax.set_yticks([0.5, 0.6])
# ax.set_ylim([0.49, 0.62])
# ax.tick_params(axis="x", labelsize=16)    
# ax.set_xticks([])
# plt.tight_layout()
# plt.savefig('CV ON state duration.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('CV ON state duration.pdf', dpi = 1000, format = 'pdf')

# fig, ax = plt.subplots(figsize = (1,1))
# ax.boxplot([np.std(i)/np.mean(i) for i in OFF_durations_ALL], widths = 0.5)
# ax.set_yticks([0.4, 0.7])
# ax.set_ylim([0.35, 0.75])
# ax.tick_params(axis="x", labelsize=16)    
# ax.set_xticks([])
# plt.tight_layout()
# plt.savefig('CV OFF state duration.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('CV OFF state duration.pdf', dpi = 1000, format = 'pdf')

# fig, ax = plt.subplots(figsize = (1,1))
# ax.boxplot([np.mean(i) for i in ON_durations_ALL], widths = 0.5)
# ax.set_yticks([0.4, 0.7])
# ax.set_ylim([0.35, 0.85])
# ax.tick_params(axis="x", labelsize=16)    
# ax.set_xticks([])
# plt.tight_layout()
# plt.savefig('mean ON state duration.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('mean ON state duration.pdf', dpi = 1000, format = 'pdf')

# fig, ax = plt.subplots(figsize = (1,1))
# ax.boxplot([np.mean(i) for i in OFF_durations_ALL], widths = 0.5)
# ax.set_yticks([0.6, 1])
# ax.set_ylim([0.5, 1.1])
# ax.tick_params(axis="x", labelsize=16)    
# ax.set_xticks([])
# plt.tight_layout()
# plt.savefig('mean OFF state duration.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('mean OFF state duration.pdf', dpi = 1000, format = 'pdf')

# # # autocorrelation of LFP, all mice
# auto_corr_ALL_conc = np.asarray([np.mean(i, axis = 0) for i in auto_corr_ALL])
# fig, ax = plt.subplots(figsize = (2,2))
# ax.plot(auto_corr_ALL_conc.T, linewidth = 0.75)
# ax.set_xlim([15000,25000])
# ax.set_xticks([16000,18000,20000,22000,24000])
# ax.set_yticks([0,1])
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.tick_params(width=2, length=4)
# # ax.set_xlabel('time (s)', size = 16)
# # ax.set_ylabel('LFP autocorrelation', size = 16)
# ax.tick_params(axis="x", labelsize=14)    
# ax.tick_params(axis="y", labelsize=14) 
# plt.tight_layout()
# plt.savefig('LFP autocorrelation ALL.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('LFP autocorrelation ALL.pdf', dpi = 1000, format = 'pdf')





#%% spike frequency adaptation across all mice
ON_duration_minimum_for_spikerates = 400 # how long does ON state have to be for it to be included in spike adaptation analysis? how long is a piece of string jp?
ON_start_spikes_duration = 200 # how many ms to extract spike rates from at the beginning of ON states?
ON_stop_spikes_duration = 200
spike_number_cutoff_adaptation = 1
start_stop_margin = 50 #how much before and after ON states to take out with the spikes

# 1) avg spike frequency 50-150 vs -150--50
os.chdir(os.path.join(overall_path, 'LAMINAR_UP'))
ON_start_spikes_ALL = pickle.load(open('ON_start_spikes_ALL', 'rb'))
ON_stop_spikes_ALL = pickle.load(open('ON_stop_spikes_ALL', 'rb'))
ON_start_spikes_rel_ALL = pickle.load(open('ON_start_spikes_rel_ALL', 'rb'))
ON_stop_spikes_rel_ALL = pickle.load(open('ON_stop_spikes_rel_ALL', 'rb'))

ON_start_number_for_adaptation_ALL = pickle.load(open('ON_start_number_for_adaptation_ALL', 'rb'))
# os.chdir(figures_path)

ON_start_avg_rate = [np.nanmean(np.nanmean(i, axis = 0)[100:200])*1000 for i in ON_start_spikes_ALL]
ON_stop_avg_rate = [np.nanmean(np.nanmean(i, axis = 0)[50:150])*1000 for i in ON_stop_spikes_ALL]
ON_start_avg_rate_rel = [np.nanmean(np.nanmean(i, axis = 0)[100:200])*100 for i in ON_start_spikes_rel_ALL]
ON_stop_avg_rate_rel = [np.nanmean(np.nanmean(i, axis = 0)[50:150])*100 for i in ON_stop_spikes_rel_ALL]

ON_start_rate_rel = np.asarray([np.nanmean(i, axis = 0) for i in ON_start_spikes_rel_ALL]) # average across all SOs
ON_stop_rate_rel = np.asarray([np.nanmean(i, axis = 0) for i in ON_stop_spikes_rel_ALL])

fig, ax = plt.subplots(figsize = (4,4))
to_plot_mean = np.mean(ON_start_rate_rel, axis = 0)*100
to_plot_err = scipy.ndimage.gaussian_filter(np.std(ON_start_rate_rel, axis = 0)/np.sqrt(6)*100, 2)
ax.plot(np.arange(- start_stop_margin, ON_start_spikes_duration-1, 1), scipy.ndimage.gaussian_filter1d(to_plot_mean, 1), 'k')
ax.fill_between(np.arange(- start_stop_margin, ON_start_spikes_duration-1, 1), to_plot_mean + to_plot_err, to_plot_mean - to_plot_err, color = 'k', alpha = 0.5)
ax.tick_params(axis="x", labelsize=16)    
ax.tick_params(axis="y", labelsize=16) 
ax.set_xlabel('time from ON start (ms)', size = 14)
ax.set_ylabel('MUA (% of mean spike rate)', size = 14)
ax.set_ylim([0,1000])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('ON start spike rate avg.jpg', dpi = 1000, format = 'jpg')
plt.savefig('ON start spike rate avg.pdf', dpi = 1000, format = 'pdf')


fig, ax = plt.subplots(figsize = (4,4))
to_plot_mean = np.mean(ON_stop_rate_rel, axis = 0)*100
to_plot_err = scipy.ndimage.gaussian_filter(np.std(ON_stop_rate_rel, axis = 0)/np.sqrt(6)*100, 2)
ax.plot(np.arange(- ON_stop_spikes_duration, start_stop_margin-1, 1), scipy.ndimage.gaussian_filter1d(to_plot_mean, 1), 'k')
ax.fill_between(np.arange(- ON_stop_spikes_duration, start_stop_margin-1, 1), to_plot_mean + to_plot_err, to_plot_mean - to_plot_err, color = 'k', alpha = 0.5)
ax.tick_params(axis="x", labelsize=16)    
ax.tick_params(axis="y", labelsize=16) 
ax.set_xlabel('time from ON start (ms)', size = 14)
ax.set_ylabel('MUA (% of mean spike rate)', size = 14)
ax.set_ylim([0,1000])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('ON end spike rate avg.jpg', dpi = 1000, format = 'jpg')
plt.savefig('ON end spike rate avg.pdf', dpi = 1000, format = 'pdf')



fig, ax = plt.subplots(figsize = (3,4.5))
ax.plot([np.repeat(1,len(ON_stop_avg_rate)), np.repeat(2,len(ON_stop_avg_rate))], [ON_start_avg_rate, ON_stop_avg_rate], color = 'k', linewidth = 1)
ax.scatter(np.concatenate([np.repeat(1,len(ON_stop_avg_rate)), np.repeat(2,len(ON_stop_avg_rate))]), np.concatenate([ON_start_avg_rate, ON_stop_avg_rate]), color = 'k')
ax.set_xticks([1,2])
ax.set_xticklabels(['ON start', 'ON end'], size = 16)
ax.set_xlim([0.75, 2.25])
ax.tick_params(axis="y", labelsize=14)    
ax.set_ylabel('average instantenous \n MUA firing rate', size = 16)
plt.tight_layout()
print(scipy.stats.ttest_rel(ON_start_avg_rate, ON_stop_avg_rate))
plt.savefig('MUA adaptation rates abs.jpg', format = 'jpg', dpi = 1000)
plt.savefig('MUA adaptation rates abs.pdf', format = 'pdf', dpi = 1000)

fig, ax = plt.subplots(figsize = (3,4.5))
ax.plot([np.repeat(1,len(ON_stop_avg_rate_rel)), np.repeat(2,len(ON_stop_avg_rate_rel))], [ON_start_avg_rate_rel, ON_stop_avg_rate_rel], color = 'k', linewidth = 1)
ax.scatter(np.concatenate([np.repeat(1,len(ON_stop_avg_rate_rel)), np.repeat(2,len(ON_stop_avg_rate_rel))]), np.concatenate([ON_start_avg_rate_rel, ON_stop_avg_rate_rel]), color = 'k')
ax.set_xticks([1,2])
ax.set_xticklabels(['ON start', 'ON end'], size = 16)
ax.set_xlim([0.75, 2.25])
ax.tick_params(axis="y", labelsize=14)    
ax.set_ylabel('average instantenous \n MUA firing rate', size = 16)
plt.tight_layout()
print(scipy.stats.ttest_rel(ON_start_avg_rate_rel, ON_stop_avg_rate_rel))
plt.savefig('MUA adaptation rates rel.jpg', format = 'jpg', dpi = 1000)
plt.savefig('MUA adaptation rates rel.pdf', format = 'pdf', dpi = 1000)




#%% correlation MUA power and LFP for channel detection

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# use days with a nowhisker stim
for day_ind, day in enumerate(days[0:6]):
# for day_ind, day in enumerate(['160614']):
    os.chdir(day)
    no_stim_folder = [s for s in os.listdir() if 'nowhisker' in s][0]
    os.chdir(no_stim_folder)
    if day == '160615': # massive artifacts after 4 minutes
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:218000]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:218000]
        log_base = 2
    else:
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]
        log_base = 1.3

    # take log and smooth MUA power across time
    MUA_power_binned_log_smoothed = scipy.ndimage.gaussian_filter(np.emath.logn(log_base,MUA_power_binned), (0,80))
    
    # normalize within each channel, take median value in each channel
    MUA_power_binned_rel = (MUA_power_binned.T/np.median(MUA_power_binned, axis = 1)).T
    MUA_power_binned_log_smoothed_rel = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_power_binned_rel), (0,80))

    LFP_MUA_coupling = []
    for chan in range(32):
        slope, intercept, r, p, std_err = stats.linregress(LFP[chan,:], MUA_power_binned_log_smoothed_rel[chan,:])
        LFP_MUA_coupling.append(r)
    
    fig, ax = plt.subplots(figsize = (2,4))
    ax.plot(LFP_MUA_coupling, np.arange(0, -32, -1), color = 'grey')
    ax.axvline(0, color = 'k', linestyle = '--')
    # ax.tick_params(axis="y", labelsize=14)   
    ax.tick_params(axis="x", labelsize=14) 
    ax.set_xlim(left = -0.7)
    # ax.set_ylim(lower = )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(np.linspace(-31, 0, 5))
    ax.set_ylim([-31,0])
    ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    ax.set_ylabel('depth (mm)', size = 16)
    ax.set_xlabel('$\mathregular{R_{logMUA, LFP}}$', size = 16)
    plt.tight_layout()
    plt.savefig('MUA LFP correlation depth.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('MUA LFP correlation depth.pdf', format = 'pdf', dpi = 1000)

    os.chdir('..')
    os.chdir('..')

    
    
    