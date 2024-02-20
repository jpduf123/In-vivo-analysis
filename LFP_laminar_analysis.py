# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 21:26:17 2021

@author: JPDUF
"""

import numpy as np
import matplotlib.pyplot as plt
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
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap

home_directory = r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS\LAMINAR_UP'
os.chdir(home_directory)

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


def interpolate_matrix(matrix, space_interp = 200):
    '''
    Parameters
    ----------
    matrix : timexchans
        matrix array.
        
    space_interp : number of channels in space you want

    Returns
    -------
    matrix interpolated in space.
    '''
    # interpolate in space, for better visualization
    #you have to flatten the matrix trace (so append each channel to the end of the previous one) and then define X and Y coords for every point
    flat_mean_matrix = np.transpose(matrix).flatten()
    grid_x = np.tile(np.linspace(1, matrix.shape[0], matrix.shape[0]), matrix.shape[1]) # repeat 1-768 16 times
    grid_y = np.repeat(np.linspace(1, matrix.shape[1], matrix.shape[1]),matrix.shape[0]) # do 1x768, 2x768 etc...
    grid_x_int, grid_y_int = np.meshgrid(np.linspace(1, matrix.shape[0], matrix.shape[0]), np.linspace(1, matrix.shape[1], space_interp)) # i.e. the grid you want to interpolate to
    mean_matrix_spatial_interpolated = scipy.interpolate.griddata((grid_x, grid_y), flat_mean_matrix, (grid_x_int, grid_y_int), method='cubic')
    return mean_matrix_spatial_interpolated


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=1000):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

new_jet = truncate_colormap(plt.get_cmap('jet'), 0.2, 1)


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


#%% -------------------------------------------------------------------------------------- channels for layers

# list of lists: 10 sweeps with 5 layers each (1, 2/3, 4, 5, 6)
# layer map for every mouse (approximate)
mouse_1 = list(map(np.asarray, [[1,2], [3,4,5,6], [7,8,9,10,11,12], [13,14,15,16,17,18], [19,20,21,22,23,]]))
mouse_2 = list(map(np.asarray, [[4,5,6,7,8], [9,10,11,12], [13,14,15,16,17], [18,19,20,21], [22,23,24,25]]))
mouse_3 = list(map(np.asarray, [[2,3,4], [5,6,7,8,9], [10,11,12,13], [14,15,16,17,18,19], [20,21,22,23,24]]))
mouse_4 = list(map(np.asarray, [[1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18], [19,20,21,22,23]]))
mouse_5 = list(map(np.asarray, [[1,2], [3,4,5,6,7,8], [9,10,11,12,13], [14,15,16,17], [18,19,20,21,22]]))
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


#%% -------------------------------------------------------------------------------------- extract spikes and resample
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

for highpass_cutoff in [3,4,5,6,7]:
    for day in days:
        os.chdir(day) 
        print(day)
        os.chdir('pre_AP5')
        # if os.path.exists(f'analysis_{day}') == False:
        #     os.mkdir(f'analysis_{day}')
        # os.chdir('..')
        sweeps = [s for s in os.listdir() if 'baseline' in s]     
        # REORDER THE SWEEPS BASELINE-BEFORE 
        before = [s for s in sweeps if 'before' in s]
        after = [s for s in sweeps if 'after' in s]
        sweeps_ordered = before + after
        
        if os.path.isfile('LFP_resampled') and os.path.isfile(f'spikes_allsweeps_{highpass_cutoff}'):
            os.chdir('..')
            continue
        #     LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
        #     if use_kilosort == False:
        #         spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
        #     else:
        #         spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
        #     stim_times = pickle.load(open('stim_times','rb'))
             
        #%
        # else:
        #get stim times, LFP and spikes
        LFP_all_sweeps = []
        MUA_all_sweeps = []
        spikes_allsweeps = []
        
        for ind_sweep, sweep in enumerate(sweeps_ordered):
            os.chdir(sweep)
            channels = [s for s in os.listdir() if 'amp' in s and '.dat' in s]
            curr_spikes = {}
            for ind_channel, channel in enumerate(channels):
                print(channel)
                with open(channel,'rb') as f:
                    curr_LFP = np.fromfile(f, np.int16)
                    
                    # take out spikes
                    highpass = elephant.signal_processing.butter(curr_LFP, highpass_frequency = 250, sampling_frequency=30000)
                    std = np.std(highpass)
    
                    # filter in MUA band
                    MUA = elephant.signal_processing.butter(curr_LFP, highpass_frequency = 200, lowpass_frequency = 1500, sampling_frequency=30000)
                    #binning
                    timestamps = np.linspace(0, MUA.shape[0], MUA.shape[0])
                    timeseries_df = pd.DataFrame({"Timestamps": timestamps, "Values": MUA})
                    timeseries_df["Bins"] = pd.cut(timeseries_df["Timestamps"], int(MUA.shape[0]/30)) 
                    curr_MUA_binned = timeseries_df.groupby("Bins").mean()["Values"].to_numpy()
    
                    #the highpass cutoff you choose is not that important, you'll just get more noise in the end but maybe not stupid for the slow wave spiking
                    crossings = np.argwhere(highpass<-highpass_cutoff*std)
                    # take out values within half a second of each other
                    crossings = crossings[np.roll(crossings,-1) - crossings > 20]
                    curr_spikes[channel[-6:-4]] = crossings/resample_factor
                    
                    # #resample for LFP. SHOULD TRY DECIMATE AS WELL AND SEE IF ITS DIFFERENT. scipy resample just deletes all frequencies above Nyquist resample frequency, it doesnt apply a filter like decimate does
                    curr_LFP = scipy.signal.resample(curr_LFP, int(np.ceil(len(curr_LFP)/resample_factor)))
                    
                if ind_channel == 0:
                    LFP = curr_LFP
                    MUA_binned = curr_MUA_binned
    
                elif ind_channel > 0:                
                    LFP = np.vstack((LFP,curr_LFP))
                    MUA_binned = np.vstack((MUA_binned, curr_MUA_binned))
    
            spikes_allsweeps.append(curr_spikes)    
            LFP_all_sweeps.append(LFP)
            MUA_all_sweeps.append(MUA_binned)
            os.chdir("..")
        
        pickle.dump(LFP_all_sweeps, open('LFP_resampled','wb'))
        pickle.dump(MUA_all_sweeps, open('MUA_all_sweeps','wb'))
        pickle.dump(spikes_allsweeps, open(f'spikes_allsweeps_{highpass_cutoff}','wb'))
    
        stim_times = []
        #get stim_times
        for ind_sweep, sweep in enumerate(sweeps_ordered):
            os.chdir(sweep)
            if os.path.isfile('board-DIN-00.dat'):
                stimfile = 'board-DIN-00.dat'   
            elif os.path.isfile('board-DIGITAL-IN-00.dat'):
                stimfile = 'board-DIGITAL-IN-00.dat'
            else:
                raise KeyError('no stim file')
            with open(stimfile,'rb') as f:
                    curr_stims = np.fromfile(f, np.int16)
                    curr_stims = np.where(np.diff(curr_stims) == 1)[0]/resample_factor
            stim_times.append(curr_stims)       
            os.chdir("..")
        
            
        #take out shit stims:
        for ind_sweep, stims in enumerate(stim_times):
            diff = np.diff(stims)
            for ind_stim, stim in enumerate(list(stims)):
                if ind_stim == len(stims) - 1:
                    break
                if stims[ind_stim + 1] - stims[ind_stim] < 4995:
                    for i in range(1,100):
                        if ind_stim + i > len(stims) - 1:
                            break
                        if stims[ind_stim + i] - stims[ind_stim] < 4995:
                            stims[ind_stim + i] = 0
                        elif stims[ind_stim + i] - stims[ind_stim] > 4996:
                            break 
            stim_times[ind_sweep] = np.delete(stims, np.where(stims == 0))
        pickle.dump(stim_times, open('stim_times', 'wb'))
    
    
        
        # # extract spike sorted spikes, every unit as one dictionary with its principal channel
        # if len([i for i in os.listdir() if 'kilosort' in i and os.path.isdir(i)]) > 0 and os.path.isfile('spikes_allsweeps_kilosort') == False:
        #     #get channel names
        #     os.chdir(sweeps_ordered[0])
        #     channels = [s for s in os.listdir() if 'amp' in s and '.dat' in s]
        #     os.chdir('..')
        #     os.chdir([i for i in os.listdir() if 'kilosort' in i and os.path.isdir(i)][0])
            
        #     spike_times_kilosort = np.load('spike_times.npy')
        #     spike_clusters_kilosort = np.load('spike_clusters.npy')
        #     cluster_info = pd.read_csv('cluster_info.tsv', sep = '\t', header = 0)
        #     7
        #     # get all clusters that are not noise
        #     good_clusters = np.asarray(cluster_info['cluster_id'][np.where(cluster_info['group'] != 'noise')[0]])
        
        #     sweep_cumsum = np.cumsum(np.asarray([LFP_all_sweeps[i].shape[1] for i in range(len(LFP_all_sweeps))]))
        #     sweep_cumsum = np.insert(sweep_cumsum, 0, 0)
        
        #     spikes_allsweeps = []
        #     for ind_sweep, sweep in enumerate(sweeps_ordered):
        #         print(sweep)
        #         curr_spikes = {}
        #         for channel in channels:
        #             curr_clusters = np.intersect1d(np.asarray(cluster_info['cluster_id'][np.where(cluster_info['ch'] == int(channel[-6:-4]))[0]]), good_clusters)
        #             curr_spikes[channel[-6:-4]] = spike_times_kilosort[np.intersect1d(np.argwhere(np.logical_and(sweep_cumsum[ind_sweep] < spike_times_kilosort/30, spike_times_kilosort/30 < sweep_cumsum[ind_sweep + 1])), np.where(np.isin(spike_clusters_kilosort, curr_clusters)))]/resample_factor - sweep_cumsum[ind_sweep]
        #         spikes_allsweeps.append(curr_spikes)
        #     os.chdir('..')
        #     pickle.dump(spikes_allsweeps, open('spikes_allsweeps_kilosort','wb'))
        
        
        # #plot some highpass
        # with open(f'amp-A-0{chanMap_32[5]+16}.dat','rb') as f:
        #     curr_LFP = np.fromfile(f, np.int16)
        # fig, ax = plt.subplots()
        # highpass = elephant.signal_processing.butter(curr_LFP, highpass_frequency = 250, sampling_frequency=30000)
        # ax.plot(highpass)
        # ax.axhline(-5*np.std(highpass))


        os.chdir('..')
        os.chdir('..')


#%% ------------------------------------------------------------------------------------------ plot examples

highpass_cutoff = 4
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]


# raster plot
for day_ind, day in enumerate(days[0:2]):
    os.chdir(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))

    nchans = LFP_all_sweeps[0].shape[0]
    if nchans == 16:
        chanMap = chanMap_16
    elif nchans == 32:
        chanMap = chanMap_32

    fig, ax = plt.subplots(figsize = (12,2.5))
    for chan_ind, spikes in enumerate(list(spikes_allsweeps[0].values())):
        # only if in a real layer
        if np.argwhere(chanMap == chan_ind)[0][0] in np.concatenate(layer_dict[day][0][1:]):
            ax.plot(spikes, np.argwhere(chanMap == chan_ind)[0][0]/10 * -np.ones_like(spikes), 'k.', markersize = 3)
        ax.set_xlim(200*new_fs,220*new_fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('example_MUA', dpi = 1000)
    
    
    fig, ax = plt.subplots()
    for i in range(len(LFP_all_sweeps[0])):
        ax.plot(LFP_all_sweeps[8][i,:] + np.argwhere(chanMap == i)[0][0] *1500 * -np.ones_like(LFP_all_sweeps[8][i,:]), linewidth = 0.5)
        ax.set_xlim(200*new_fs,220*new_fs)
        # ax.set_yticks(ticks = np.linspace(0, 31000, 32), labels = list(np.linspace(0, 31, 32, dtype = int)))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('example_LFP', dpi = 1000)
    
    
    # fig, ax = plt.subplots()
    # for i in range(len(LFP_all_sweeps[0])):
    #     ax.plot(LFP_all_sweeps[1][i,:] + np.argwhere(chanMap == i)[0][0] *1500 * -np.ones_like(LFP_all_sweeps[1][i,:]), linewidth = 0.5)
    #     ax.set_xlim(10400,11400)
    #     # ax.set_yticks(ticks = np.linspace(0, 31000, 32), labels = list(np.linspace(0, 31, 32, dtype = int)))
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.set_xticks([])
    #     ax.set_yticks([])


    os.chdir('..')
    
    
    
#%% ---------------------------------------------------------------------------------------- LFP whisker response -------------------------------------------------------------------------------------------

do_shift = False
highpass_cutoff = 4

to_plot_1 = [0,1,2,3]
to_plot_2 = [4,5,6,7,8,9]

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# for day in ['160810']:
for day in days:
    print(day)
    os.chdir(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    # if use_kilosort == False:
    #     spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    # else:
    #     spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    
    nchans = LFP_all_sweeps[0].shape[0]
    if nchans == 16:
        chanMap = chanMap_16
    elif nchans == 32:
        chanMap = chanMap_32
    
    def LFP_average(sweeps_to_plot, stims = stim_times, LFP_all_sweeps = LFP_all_sweeps):
        '''
    
        Parameters
        ----------
        sweeps_to_plot : list
            sweeps in python index to plot.
    
        Returns
        -------
        array nchansxLFP, averaged over all the stims you want.
    
        '''
        to_plot = np.zeros([len(LFP_all_sweeps[0]), int(0.6*new_fs), len(sweeps_to_plot)])    
        for ind_sweep, sweep in enumerate(sweeps_to_plot):
            curr_to_plot = np.zeros([len(LFP_all_sweeps[0]), int(0.6*new_fs), len(stims[sweep])])
            for ind_stim, stim in enumerate(list(stims[sweep])):
                if ind_stim == len(stims[sweep]) - 1:
                    break
                if stim < 0.3*new_fs:
                    continue
                if stim + 0.3*new_fs > LFP_all_sweeps[sweep].shape[1]:
                    continue
                else:
                    curr_to_plot[:,:,ind_stim] = LFP_all_sweeps[sweep][:,int(stim - 0.2*new_fs):int(stim + 0.4*new_fs)]
            to_plot[:,:,ind_sweep] = np.squeeze(np.mean(curr_to_plot,2)) # average across stims
        return np.squeeze(np.mean(to_plot,2)) #average across sweeps

    if os.path.exists('stims_for_LFP'):
        stims_for_LFP = pickle.load(open('stims_for_LFP', 'rb'))
    else:
        stims_for_LFP = copy.deepcopy(stim_times)
        # stims_for_LFP[5][25:] = 0
        pickle.dump(stims_for_LFP, open('stims_for_LFP', 'wb'))
    
    #select the sweeps to look at    
    to_plot_1_LFP = [0,1,2,3]
    to_plot_2_LFP = [4,5,6,7,8,9]
    # to_plot_2 = [6]
    
    LFP_min = np.empty([len(LFP_all_sweeps), nchans])
    LFP_min[:] = np.NaN
    LFP_min_rel = np.empty([len(LFP_all_sweeps), nchans])
    LFP_min_rel[:] = np.NaN
    LFP_std = np.zeros([len(LFP_all_sweeps), nchans])
    LFP_std[:] = np.NaN
    LFP_slope = np.empty([len(LFP_all_sweeps), nchans])
    LFP_slope[:] = np.NaN
    
    LFP_responses = np.zeros([len(LFP_all_sweeps), nchans, 600])
    LFP_responses[:] = np.NaN
    for sweep in range(len(LFP_all_sweeps)):
        LFP_responses[sweep, :, :] = LFP_average([sweep], stims = stims_for_LFP)
    
    for sweep in range(len(LFP_all_sweeps)):
        if LFP_all_sweeps[sweep].size == 0:
            continue
        else:
            LFP_min[sweep,:] = np.abs(np.min(LFP_average([sweep], stims = stims_for_LFP)[:,200:300], 1) - LFP_average([sweep], stims = stims_for_LFP)[:,210])
            LFP_std[sweep,:] = np.std(LFP_average([sweep]), 1)
            # slope: go from 20. There's no good way of getting it, would have to filter it down or something to get rid of high-frequency noise
            slope_start = 220
            LFP_slope[sweep,:] = (np.min(LFP_average([sweep], stims = stims_for_LFP)[:,200:300], 1) - LFP_average([sweep], stims = stims_for_LFP)[:,slope_start])/(np.argmin(LFP_average([sweep], stims = stims_for_LFP)[:,200:300], 1) - 20)
    
    # relative LFP min (relative to baseline LFP min of every channel)
    LFP_min_rel = LFP_min/np.nanmean(LFP_min[to_plot_1_LFP,:], axis = 0)
    LFP_min_rel_change = np.mean(LFP_min_rel[to_plot_2_LFP,:], axis = 0) - np.mean(LFP_min_rel[to_plot_1_LFP,:], axis = 0)
    LFP_slope_rel = LFP_slope/np.nanmean(LFP_slope[to_plot_1_LFP,:], axis = 0)
    LFP_slope_rel_change = np.mean(LFP_slope_rel[to_plot_2_LFP,:], axis = 0) - np.mean(LFP_slope_rel[to_plot_1_LFP,:], axis = 0)
    
    
    if do_shift:
        shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
        # you need to add the last XX (total shift) channels to be able to subtract every image with the next one if there's a shift
        # THIS IS ASSUMING YOU HAVE POSITIVE SHIFT (CHANNEL NUMBERS GET BIGGER), which is why I call max function if not would have to change it
        total_shift = max(shift)
    else:
        total_shift = 0
        
    LFP_all = np.asarray([LFP_average([i]) for i in range(10)])[:,chanMap,:]
    # LFP RESPONSES BEFORE AND AFTER before and after with shift
    if total_shift == 0:
        LFP_shift_all = np.asarray([LFP_all[i, :, :] for i in range(10)])
        LFP_before = np.mean(np.asarray([LFP_all[i, :, :] for i in to_plot_1]), axis = 0)
        LFP_after = np.mean(np.asarray([LFP_all[i, :, :] for i in to_plot_2]), axis = 0)
    
    else:
        # account for the shift in channels and redo it in chanMap order
        LFP_all = np.asarray([LFP_average([i]) for i in range(10)])[:,chanMap,:]
        # CSD_all = CSD_all[:,:,:-total_shift]
        
        LFP_shift_all = np.asarray([LFP_all[i, shift[i]:(nchans - (total_shift -shift[i])), :] for i in range(10)])
        LFP_before = np.mean(np.asarray([LFP_all[i, shift[i]:(nchans - (total_shift -shift[i])), :] for i in to_plot_1]), axis = 0)
        LFP_after = np.mean(np.asarray([LFP_all[i, shift[i]:(nchans - (total_shift -shift[i])), :] for i in to_plot_2]), axis = 0)
    tot_chans = LFP_before.shape[0]
    
    # fig, ax = plt.subplots(8,4, sharey = True) 
    # for ind, ax1 in enumerate(list(ax.flatten())):                        
    #     ax1.plot(LFP_average(to_plot_1_LFP)[chanMap[ind],:] - LFP_average(to_plot_1_LFP)[chanMap[ind],200], 'b')
    #     ax1.plot(LFP_average(to_plot_2_LFP)[chanMap[ind],:] - LFP_average(to_plot_2_LFP)[chanMap[ind],200], 'r')
    #     # if chan in LFP_resp_channels:
    #     #     ax[np.argwhere(chanMap == chan)[0][0]].set_facecolor("y")
    #     ax1.set_title(str(chanMap[ind]), size = 6)
    #     ax1.set_xlim([150,300])
    # plt.savefig(f'LFP_{to_plot_1_LFP}_vs_{to_plot_2_LFP}', dpi = 1000)
    
    interpolation_points = 8*nchans
    vmax_overall = np.max(-np.concatenate((interpolate_matrix(LFP_before[:,150:300], space_interp = interpolation_points), interpolate_matrix(LFP_after[:,150:300], space_interp = interpolation_points))))
    vmin_overall = np.min(-np.concatenate((interpolate_matrix(LFP_before[:,150:300], space_interp = interpolation_points), interpolate_matrix(LFP_after[:,150:300], space_interp = interpolation_points))))
    
    fig, ax = plt.subplots(figsize = (3,8))
    # fig.suptitle('before')
    im = ax.imshow(-interpolate_matrix(LFP_before.T, space_interp = interpolation_points), cmap = new_jet, vmin = vmin_overall, vmax = vmax_overall, aspect = 1.7)
    ax.set_xlim([150,300])
    ax.set_yticks(np.linspace(interpolation_points - 1, 0, 5))
    ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    ax.set_xticks([150,200,250,300])
    ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 14)
    ax.set_ylabel('depth (mm)', size = 16)
    ax.set_xlabel('time from stim (ms)', size = 16)
    plt.tight_layout()
    plt.savefig('LFP before.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('LFP before.pdf', dpi = 1000, format = 'pdf')
    
    fig, ax = plt.subplots(figsize = (3,8))
    # fig.suptitle('after')
    im = ax.imshow(-interpolate_matrix(LFP_after.T, space_interp = interpolation_points), cmap = new_jet, vmin = vmin_overall, vmax = vmax_overall, aspect = 1.7)
    ax.set_xlim([150,300])
    ax.set_yticks(np.linspace(interpolation_points - 1, 0, 5))
    ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    ax.set_xticks([150,200,250,300])
    ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 14)
    ax.set_ylabel('depth (mm)', size = 16)
    ax.set_xlabel('time from stim (ms)', size = 16)
    plt.tight_layout()    
    plt.savefig('LFP after.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('LFP after.pdf', dpi = 1000, format = 'pdf')

    
    
    spacer = np.max(LFP_min[:,3:])
    fig, ax = plt.subplots(figsize = (4,10))
    for ind in range(tot_chans):
        ax.plot(LFP_before[ind,:] + ind * -spacer *np.ones_like(LFP_before[ind,:]), 'k', linewidth = 1)                 
        ax.plot(LFP_after[ind,:] + ind * -spacer *np.ones_like(LFP_after[ind,:]), 'c', linewidth = 1)                     
        ax.set_xlim([150,300])
    ax.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, 5))
    # ax.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 6)
    # ax.set_yticks(np.linspace(31, 0, 5))
    ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    ax.set_xticks([150,200,250,300])
    ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 14)
    ax.set_ylabel('depth (mm)', size = 16)
    ax.set_xlabel('time from stim (ms)', size = 16)
    ax.set_ylim(bottom = -31*spacer - spacer/2)
    ax.set_ylim(top = spacer + spacer/2)
    plt.tight_layout()
    plt.savefig(f'LFP_laminar_{to_plot_1_LFP}_vs_{to_plot_2_LFP}', dpi = 1000)
    plt.savefig(f'LFP_laminar_{to_plot_1_LFP}_vs_{to_plot_2_LFP}.pdf', dpi = 1000, format = 'pdf')

    
    # #relative change as colorplot
    # fig, ax = plt.subplots()
    # fig.suptitle('relative change in LFP')
    # im = ax.imshow(np.reshape(LFP_min_rel_change[chanMap], (8,4)), cmap = 'jet', vmax = 0)
    # fig.colorbar(im)
    
    # #change over time in all channels (LFP_min)
    # fig, ax = plt.subplots(8,4) 
    # fig.suptitle('LFP timecourse')
    # for ind, ax1 in enumerate(list(ax.flatten())):                        
    #     ax1.plot(LFP_min_rel[:,chanMap[ind]])
    #     # if chan in LFP_resp_channels:
    #     #     ax[np.argwhere(chanMap == chan)[0][0]].set_facecolor("y")
    #     ax1.set_title(str(chanMap[ind]))
    
    
    # # change of slope over time
    # fig, ax = plt.subplots(8,4) 
    # fig.suptitle('slope timecourse')
    # for ind, ax1 in enumerate(list(ax.flatten())):                        
    #     ax1.plot(LFP_slope_rel[:,chanMap[ind]])
    #     # if chan in LFP_resp_channels:
    #     #     ax[np.argwhere(chanMap == chan)[0][0]].set_facecolor("y")
    #     ax1.set_title(str(chanMap[ind]))
    
    # # plot timecourse of average LFP peak with error bars
    # for ind_sweep, sweep in enumerate(sweeps_to_plot):
    #     curr_to_plot = np.zeros([64, int(0.6*new_fs), len(stim_times[sweep])])
    #     for ind_stim, stim in enumerate(list(stim_times[sweep])):
    #         if ind_stim == len(stim_times[sweep]) - 1:
    #             break
    #         if stim < 0.3*new_fs:
    #             continue
    #         else:
    #             curr_to_plot[:,:,ind_stim] = LFP_all_sweeps[sweep][:,int(stim - 0.2*new_fs):int(stim + 0.4*new_fs)]
    #     to_plot[:,:,ind_sweep] = np.squeeze(np.mean(curr_to_plot,2))
    
    

    # ---------- LFP response in every sweep
    
    spacer = np.max(LFP_min[:,3:]*1.2)
    fig, ax = plt.subplots(1,10, figsize = (15,15), sharey = True)
    for ind, ax1 in enumerate(list(ax.flatten())):
        for chan in range(nchans):                        
            ax1.plot(LFP_responses[ind,chanMap[chan],:] + chan * -spacer, 'b', linewidth = 1)                 
            ax1.set_xlim([150,400])
        ax1.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, tot_chans))
        ax1.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 8)
    plt.tight_layout()
    plt.savefig(f'LFP_all_sweeps', dpi = 1000)

    # LFP_responses_smoothed = scipy.ndimage.gaussian_filter1d(LFP_responses, 2, axis = 1)
    # spacer = np.max(LFP_min[:,3:]*1.2)
    # fig, ax = plt.subplots(1,10, figsize = (15,15), sharey = True)
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     for chan in range(nchans):                        
    #         ax1.plot(LFP_responses_smoothed[ind,chanMap[chan],:] + chan * -spacer, 'b', linewidth = 1)                 
    #         ax1.set_xlim([150,400])
    #     ax1.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, tot_chans))
    #     ax1.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 8)
    # plt.tight_layout()
    # plt.savefig(f'LFP_all_sweeps smoothed', dpi = 1000)






    # # ------------------------------------------------------------ LFP response heatmap ----------------------------------------------------------------------
    interpolation_points = 8*nchans
    fig, ax = plt.subplots()
    to_plot = -interpolate_matrix(np.mean(LFP_responses[[0,1,2,3],:,150:300], axis = 0)[chanMap,:].T, space_interp = interpolation_points)
    ax.imshow(to_plot, cmap = new_jet, aspect = 1.5)
    # ax.set_ylim([0,150])
    
    to_plot_traces = -np.mean(LFP_responses[[0,1,2,3],:,150:300], axis = 0)[chanMap,:]
    chunk = interpolation_points/nchans # how much of the yaxis will each channel occopy
    scaling = (np.max(to_plot_traces)-750)/(chunk/2) # scaling for the LFP signal to fit in one chunk on the colormap
    for chan in range(nchans):
        pos = chan*chunk + chunk/2 # position along the y axis (which is inverted in colormaps)
        ax.plot(to_plot_traces[chan,:]/scaling + pos, color = 'black', linewidth = 1)
    ax.set_yticks(np.linspace(interpolation_points - 1, 0, 5))
    ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 16)
    ax.set_xticks([0,50,100,150])
    ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 16)
    ax.set_ylabel('depth (mm)', size = 16)
    ax.set_xlabel('time from stim (ms)', size = 16)
    plt.tight_layout()
    plt.savefig('LFP colormap with traces.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('LFP colormap with traces.pdf', dpi = 1000, format = 'pdf')


    fig, ax = plt.subplots(figsize = (5,1.5))
    norm = colors.Normalize(vmin=np.min(to_plot_traces)/1000, vmax=np.max(to_plot_traces)/1000)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap= new_jet),
                  cax=ax, orientation = 'horizontal')
    ax.tick_params(axis="x", labelsize=14)    
    # ax.set_yticklabels(list(map(str, np.linspace(-1, 1.5, 6))), size = 16)
    ax.set_xlabel('LFP (mV)', size = 18)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.savefig('LFP colormap legend.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('LFP colormap legend.jpg', dpi = 1000, format = 'jpg')



    # os.chdir(home_directory)
    if os.path.exists(f'analysis_{day}') == False:
        os.mkdir(f'analysis_{day}')
    os.chdir(f'analysis_{day}')
    np.savetxt('LFP_min.csv', LFP_min, delimiter = ',')
    np.savetxt('LFP_slope.csv', LFP_slope, delimiter = ',')
    np.savetxt('LFP_before.csv', LFP_before, delimiter = ',')
    np.savetxt('LFP_after.csv', LFP_after, delimiter = ',')
    np.savetxt('LFP_min_rel.csv', LFP_min_rel, delimiter = ',')
    np.savetxt('LFP_min_rel_change.csv', LFP_min_rel_change, delimiter = ',')
    np.savetxt('LFP_slope_rel.csv', LFP_slope_rel, delimiter = ',')
    np.savetxt('LFP_slope_rel_change.csv', LFP_slope_rel_change, delimiter = ',')
    np.savetxt('to_plot_1_LFP.csv', to_plot_1_LFP, delimiter = ',')
    np.savetxt('to_plot_2_LFP.csv', to_plot_2_LFP, delimiter = ',')
    np.save('LFP_responses.npy', LFP_responses)
    if do_shift:
        np.save('LFP_shift_all.csv', LFP_shift_all)

    os.chdir('..')
    os.chdir('..')

    # cl()

#%% -------------------------------------------------------------------------------------- CSD of whisker response ---------------------------------------------------------

do_shift = False

to_plot_1 = [0,1,2,3]
to_plot_2 = [4,5,6,7,8,9]

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# for day in ['160810']:
for day in days:
    os.chdir(day)
    print(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    # if use_kilosort == False:
    #     spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    # else:
    #     spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    
    nchans = LFP_all_sweeps[0].shape[0]
    if nchans == 16:
        chanMap = chanMap_16
    elif nchans == 32:
        chanMap = chanMap_32
    coordinates = [[i] for i in list(np.linspace(0, 1.55, nchans))]*pq.mm

    
    #smooth across channels for the CSDs
    def CSD_average(sweeps_to_plot, stims = stim_times, smoothing = False, smooth_over = 1, time_before = 0.2, time_after = 0.4, smooth_over_time = False, smooth_over_time_window = 10):
        '''
    
        Parameters
        ----------
        sweeps_to_plot : list
            sweeps in python index to plot.
    
        Returns
        -------
        array nchansxLFP, averaged over all the stims you want.
    
        '''
        # chan x time x sweep
        to_plot = np.zeros([len(LFP_all_sweeps[0]), int((time_before + time_after)*new_fs), len(sweeps_to_plot)])    
        for ind_sweep, sweep in enumerate(sweeps_to_plot):
            #chan x time x stim
            if stims == stim_times:
                # then stims[sweep] gives you the right ones. but if your stims are a single list of stims it fucks it up (slow wave CSD)
                curr_to_plot = np.zeros([len(LFP_all_sweeps[0]), int((time_before + time_after)*new_fs), len(stims[sweep])])
                curr_stims = list(stims[sweep])
            elif len(stims) == 1:
                curr_to_plot = np.zeros([len(LFP_all_sweeps[0]), int((time_before + time_after)*new_fs), len(stims[ind_sweep])])
                curr_stims = list(stims[ind_sweep])
            for ind_stim, stim in enumerate(curr_stims):
                if ind_stim == len(curr_stims) - 1:
                    break
                if stim < 0.3*new_fs:
                    continue
                if stim + 0.3*new_fs > LFP_all_sweeps[sweep].shape[1]:
                    continue
                else:
                    if smoothing == True:     
                        #smooth over channels
                        curr_LFP = neo.core.AnalogSignal(np.transpose(scipy.ndimage.gaussian_filter1d(LFP_all_sweeps[sweep][chanMap,int(stim - time_before*new_fs):int(stim + time_after*new_fs)], smooth_over, axis = 0)), units = 'mV', sampling_rate = new_fs*pq.Hz)
                    else:
                        curr_LFP = neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep][chanMap,int(stim - time_before*new_fs):int(stim + time_after*new_fs)]), units = 'mV', sampling_rate = new_fs*pq.Hz)                    
                    # print(curr_LFP.shape)
                    if smooth_over_time:
                        curr_LFP = neo.core.AnalogSignal(smooth(curr_LFP, smooth_over_time_window, 0), units = 'mV', sampling_rate = new_fs*pq.Hz)
                        
                    curr_to_plot[:,:,ind_stim] = np.transpose(elephant.current_source_density.estimate_csd(curr_LFP, coordinates = coordinates, method = 'StandardCSD', process_estimate=False))
            to_plot[:,:,ind_sweep] = np.squeeze(np.mean(curr_to_plot,2)) # average CSD across stims
        return np.squeeze(np.mean(to_plot,2)) #average across sweeps
    
    
    if do_shift:
        shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
        # you need to add the last XX (total shift) channels to be able to subtract every image with the next one if there's a shift
        # THIS IS ASSUMING YOU HAVE POSITIVE SHIFT (CHANNEL NUMBERS GET BIGGER), which is why I call max function if not would have to change it
        total_shift = max(shift)
    else:
        total_shift = 0
        
        
    CSD_all = np.asarray([np.transpose(CSD_average([i], smoothing = False)) for i in range(10)])
    CSD_all_smoothed = np.asarray([np.transpose(CSD_average([i], smoothing = False, smooth_over_time = True, smooth_over_time_window = 20)) for i in range(10)])
    
    CSD_all_ch_smoothed = np.asarray([np.transpose(CSD_average([i], smoothing = True, smooth_over = 1)) for i in range(10)])
    CSD_all_ch_smoothed_smoothed = np.asarray([np.transpose(CSD_average([i], smoothing = True, smooth_over_time = True, smooth_over_time_window = 20)) for i in range(10)])


    if total_shift == 0:
        CSD_before = np.transpose(CSD_average(to_plot_1, smoothing = True))
        CSD_after = np.transpose(CSD_average(to_plot_2, smoothing = True))
    
    else:
        CSD_before = np.mean(np.asarray([CSD_all_ch_smoothed[i, :, shift[i]:(nchans - (total_shift -shift[i]))] for i in to_plot_1]), axis = 0)
        CSD_after = np.mean(np.asarray([CSD_all_ch_smoothed[i, :, shift[i]:(nchans - (total_shift -shift[i]))] for i in to_plot_2]), axis = 0)
    
    def interpolate_CSD(CSD, space_interp = 200):
        '''
        Parameters
        ----------
        CSD : timexchans
            CSD array.
            
        space_interp : number of channels in space you want
    
        Returns
        -------
        CSD interpolated in space.
        '''
        # interpolate in space, for better visualization
        #you have to flatten the CSD trace (so append each channel to the end of the previous one) and then define X and Y coords for every point
        flat_mean_CSD = np.transpose(CSD).flatten()
        grid_x = np.tile(np.linspace(1, CSD.shape[0], CSD.shape[0]), CSD.shape[1]) # repeat 1-768 16 times
        grid_y = np.repeat(np.linspace(1, CSD.shape[1], CSD.shape[1]),CSD.shape[0]) # do 1x768, 2x768 etc...
        grid_x_int, grid_y_int = np.meshgrid(np.linspace(1, CSD.shape[0], CSD.shape[0]), np.linspace(1, CSD.shape[1], space_interp)) # i.e. the grid you want to interpolate to
        mean_CSD_spatial_interpolated = scipy.interpolate.griddata((grid_x, grid_y), flat_mean_CSD, (grid_x_int, grid_y_int), method='cubic')
        return mean_CSD_spatial_interpolated
    
    vmax_overall = np.max(np.concatenate((interpolate_CSD(CSD_before[223:,1:-1]), interpolate_CSD(CSD_after[223:,1:-1]))))
    vmin_overall = np.min(np.concatenate((interpolate_CSD(CSD_before[223:,1:-1]), interpolate_CSD(CSD_after[223:,1:-1]))))
    
    # os.chdir(home_directory)
    
    smooth_over = 8
    
    # # how many timpoints to smooth over for the difference plots
    fig, ax = plt.subplots(figsize = (3,8))
    interpolation_points = 8*nchans
    fig.suptitle('before')
    im = ax.imshow(interpolate_CSD(CSD_before, space_interp = interpolation_points), cmap = 'jet', vmin = vmin_overall, vmax = vmax_overall, aspect = 1.7)
    ax.set_xlim([150,300])
    ax.set_yticks(np.linspace(interpolation_points - 1, 0, 5))
    ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    ax.set_xticks([150,200,250,300])
    ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 14)
    ax.set_ylabel('depth (mm)', size = 16)
    ax.set_xlabel('time from stim (ms)', size = 16)
    plt.tight_layout()
    plt.savefig('CSD before', dpi = 1000, format = 'jpg')
    plt.savefig('CSD before.pdf', dpi = 1000, format = 'pdf')
    
    fig, ax = plt.subplots(figsize = (3,8))
    fig.suptitle('after')
    im = ax.imshow(interpolate_CSD(CSD_after, space_interp = interpolation_points), cmap = 'jet', vmin = vmin_overall, vmax = vmax_overall, aspect = 1.7)
    ax.set_xlim([150,300])
    ax.set_yticks(np.linspace(interpolation_points - 1, 0, 5))
    ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    ax.set_xticks([150,200,250,300])
    ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 14)
    ax.set_ylabel('depth (mm)', size = 16)
    ax.set_xlabel('time from stim (ms)', size = 16)
    plt.tight_layout()    
    plt.savefig('CSD after', dpi = 1000, format = 'jpg')
    plt.savefig('CSD after.pdf', dpi = 1000, format = 'pdf')
    
    # fig, ax = plt.subplots()
    # fig.suptitle('diff')
    # #smooth over time too
    # im = ax.imshow(interpolate_CSD(smooth(CSD_after, smooth_over) - smooth(CSD_before, smooth_over)), cmap = 'jet', vmin = vmin_overall, vmax = vmax_overall)
    # ax.set_xlim([150,350])
    # plt.savefig('CSD diff', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD diff', dpi = 1000)


    # # #CSD traces    
    # fig, ax = plt.subplots(figsize = (8,10))
    # spacer = np.max(CSD_before)
    # tot_chans = np.transpose(CSD_before).shape[0]
    # for ind in range(tot_chans):
    #     ax.plot(np.transpose(CSD_before)[ind,:] + ind* -spacer, 'k', linewidth = 1)                                   
    # ax.set_xlim([200,350])
    # ax.set_yticks(np.linspace(-(spacer*(nchans - 1 - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace((nchans - 1 - total_shift),0,tot_chans).astype(int), size = 6)
    # plt.tight_layout()
    # plt.savefig('CSD before traces', dpi = 1000)

    # fig, ax = plt.subplots(figsize = (8,10))
    # spacer = np.max(CSD_after)
    # tot_chans = np.transpose(CSD_after).shape[0]
    # for ind in range(tot_chans):
    #     ax.plot(np.transpose(CSD_after)[ind,:] + ind* -spacer, 'k', linewidth = 1)                                   
    # ax.set_xlim([200,350])
    # ax.set_yticks(np.linspace(-(spacer*(nchans - 1 - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace((nchans - 1 - total_shift),0,tot_chans).astype(int), size = 6)
    # plt.tight_layout()
    # plt.savefig('CSD after traces', dpi = 1000)
    
    # fig, ax = plt.subplots(figsize = (8,10))
    # spacer = np.max(CSD_before)
    # tot_chans = np.transpose(CSD_after).shape[0]
    # for ind in range(tot_chans):
    #     ax.plot(np.transpose(smooth(CSD_after, smooth_over) - smooth(CSD_before, smooth_over))[ind,:] + ind* -spacer, 'k', linewidth = 1)                                   
    # ax.set_xlim([200,350])
    # ax.set_yticks(np.linspace(-(spacer*(nchans - 1 - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace((nchans - 1 - total_shift),0,tot_chans).astype(int), size = 6)
    # plt.tight_layout()
    # plt.savefig('CSD diff traces', dpi = 1000)
    


    spacer = np.max(CSD_all_ch_smoothed)
    fig, ax = plt.subplots(1,10, figsize = (15,15), sharey = True)
    for ind, ax1 in enumerate(list(ax.flatten())):
        for chan in range(nchans):                        
            ax1.plot(CSD_all_ch_smoothed[ind,:,chan] + chan * -spacer, 'b', linewidth = 1)                 
            ax1.set_xlim([150,400])
        ax1.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, tot_chans))
        ax1.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 8)
    plt.tight_layout()
    plt.savefig(f'CSD all sweeps traces smoothed over channels', dpi = 1000)

    # spacer = np.max(CSD_all_ch_smoothed_smoothed)
    # fig, ax = plt.subplots(1,10, figsize = (15,15), sharey = True)
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     for chan in range(nchans):                        
    #         ax1.plot(CSD_all_ch_smoothed_smoothed[ind,:,chan] + chan * -spacer, 'b', linewidth = 1)                 
    #         ax1.set_xlim([150,400])
    #     ax1.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, tot_chans))
    #     ax1.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 8)
    # plt.tight_layout()
    # plt.savefig(f'CSD all sweeps traces smoothed over channels and time', dpi = 1000)


    fig, ax = plt.subplots(figsize = (4,10))
    spacer = vmax_overall
    tot_chans = np.transpose(CSD_after).shape[0]
    for ind in range(tot_chans):
        ax.plot(np.transpose(CSD_before)[ind,:] + ind* -spacer, 'k', linewidth = 1)                                   
        ax.plot(np.transpose(CSD_after)[ind,:] + ind* -spacer, 'c', linewidth = 1)                                   
    ax.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, 5))
    ax.set_xlim([150,300])
    # ax.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 6)
    # ax.set_yticks(np.linspace(31, 0, 5))
    ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    ax.set_xticks([150,200,250,300])
    ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 14)
    ax.set_ylabel('depth (mm)', size = 16)
    ax.set_xlabel('time from stim (ms)', size = 16)
    ax.set_ylim(bottom = -31*spacer - spacer/2)
    ax.set_ylim(top = spacer + spacer/2)
    plt.tight_layout()
    plt.savefig(f'CSD laminar before vs after.jpg', dpi = 1000, format = 'jpg')
    plt.savefig(f'CSD laminar before vs after.pdf', dpi = 1000, format = 'pdf')

    # spacer = np.max(CSD_all)
    # fig, ax = plt.subplots(1,10, figsize = (15,15), sharey = True)
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     for chan in range(nchans):                        
    #         ax1.plot(CSD_all[ind,:,chan] + chan * -spacer, 'b', linewidth = 1)                 
    #         ax1.set_xlim([150,400])
    #     ax1.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, tot_chans))
    #     ax1.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 8)
    # plt.tight_layout()
    # plt.savefig(f'CSD all sweeps traces not smoothed over channels', dpi = 1000)

    # spacer = np.max(CSD_all_smoothed)
    # fig, ax = plt.subplots(1,10, figsize = (15,15), sharey = True)
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     for chan in range(nchans):                        
    #         ax1.plot(CSD_all_smoothed[ind,:,chan] + chan * -spacer, 'b', linewidth = 1)                 
    #         ax1.set_xlim([150,400])
    #     ax1.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, tot_chans))
    #     ax1.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 8)
    # plt.tight_layout()
    # plt.savefig(f'CSD all sweeps traces not smoothed over channels but time', dpi = 1000)




    # #for channel allocation: plot CSD plot for every sweep
    fig, ax = plt.subplots(2,5, figsize = (15,15))
    for ax1_ind, ax1 in enumerate(list(ax.flatten())):
        ax1.imshow(CSD_average([ax1_ind], smoothing = True), cmap = 'jet', vmin = vmin_overall, vmax = vmax_overall, aspect = 15)
        ax1.set_xlim([150,350])
        ax1.set_yticks(list(range(nchans)))
        ax1.set_yticklabels(list(map(str, ax1.get_yticks())), size = 10)
    plt.tight_layout()
    plt.savefig('CSD all sweeps heatmap', dpi = 1000, format = 'jpg')
    plt.savefig('CSD all sweeps heatmap', dpi = 1000)

    # # CSD traces for every sweep
    # fig, ax = plt.subplots(nchans,1, sharey = True) 
    fig, ax = plt.subplots(2,5, figsize = (8,10))
    spacer = 380
    for ax1_ind, ax1 in enumerate(list(ax.flatten())):
        to_plot = CSD_average([ax1_ind], smoothing = True)
        for ind in range(len(LFP_all_sweeps[0])):
            ax1.plot(to_plot[ind,:] + ind* -spacer *np.ones_like(to_plot[ind,:]), 'k', linewidth = 1)                                   
            # ax1.set_title(str(chanMap[ind]), size = 2)
        ax1.set_xlim([200,300])
        ax1.set_yticks(np.linspace(-(spacer*nchans - 1), 0, nchans))
        if ax1_ind == 0 or ax1_ind == 5:
            ax1.set_yticklabels(np.linspace(nchans - 1,0,nchans).astype(int), size = 6)
        else:
            ax1.set_yticklabels([])
    plt.tight_layout()
    plt.savefig('CSD_laminar all sweeps', dpi = 1000)






# ------------------------------------------------------------ CSD response heatmap with traces ----------------------------------------------------------------------
    CSD_before = CSD_average(to_plot_1, smoothing = True).T[150:300,:]
    interpolation_points = 8*nchans
    fig, ax = plt.subplots()
    to_plot = interpolate_matrix(CSD_before, space_interp = interpolation_points)
    ax.imshow(to_plot, cmap = 'jet', aspect = 1.5)
    # ax.set_ylim([0,150])
    
    to_plot_traces = CSD_before.T
    chunk = interpolation_points/nchans # how much of the yaxis will each channel occopy
    scaling = np.max(to_plot_traces)/(chunk/2) # scaling for the signal to fit in one chunk on the colormap
    for chan in range(nchans):
        pos = chan*chunk + chunk/2 # position along the y axis (which is inverted in colormaps)
        ax.plot(to_plot_traces[chan,:]/-scaling + pos, color = 'black', linewidth = 1)
    ax.set_yticks(np.linspace(interpolation_points - 1, 0, 5))
    ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 20)
    ax.set_ylim([interpolation_points, 0])
    ax.set_xticks([0,50,100,150])
    ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 20)
    ax.set_ylabel('depth (mm)', size = 20)
    ax.set_xlabel('time from stim (ms)', size = 20)
    plt.tight_layout()
    plt.savefig('CSD colormap with traces.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('CSD colormap with traces.pdf', dpi = 1000, format = 'pdf')

    fig, ax = plt.subplots(figsize = (5,1.5))
    norm = colors.Normalize(vmin=np.min(CSD_before)/1000, vmax=np.max(CSD_before)/1000)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'),
                  cax=ax, orientation = 'horizontal')
    ax.tick_params(axis="x", labelsize=14)    
    # ax.set_yticklabels(list(map(str, np.linspace(-1, 1.5, 6))), size = 16)
    ax.set_xlabel('CSD (mV/$\mathregular{mm^{2}}$)', size = 18)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.savefig('CSD colormap legend.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('CSD colormap legend.jpg', dpi = 1000, format = 'jpg')


    # #CSD profile 15ms after stim
    
    # fig, ax = plt.subplots(figsize = (2,5))
    # ax.plot(CSD_before[:,215], np.linspace(0,-31,32), color = 'k')
    # ax.axvline(0, color = 'k')
    # polygon = ax.fill_between(CSD_before[:,215], np.linspace(0,-31,32), lw=0, color='none')
    # xlim = plt.xlim()
    # ylim = plt.ylim()
    # verts = np.vstack([p.vertices for p in polygon.get_paths()])
    # gradient = plt.imshow(np.linspace(0, 1, 256).reshape(-1, 1).T, cmap='jet', aspect='auto', origin='lower',
    #                       extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    # gradient.set_clip_path(polygon.get_paths()[0], transform=plt.gca().transData)
    # plt.xlim([-6,6])
    # plt.ylim([-31,0])  
    # ax.set_yticks(np.linspace(-31, 0, 5))
    # ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 16)
    # ax.tick_params(axis = 'x', labelsize = 16)
    # plt.tight_layout()
    # plt.savefig('CSD at 15ms.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('CSD at 15ms.jpg', dpi = 1000, format = 'jpg')

    # cl()

    # os.chdir(home_directory)
    os.chdir(f'analysis_{day}')
    np.save('CSD_all.npy', CSD_all)
    np.savetxt('CSD_before.csv', CSD_before, delimiter = ',')
    np.savetxt('CSD_after.csv', CSD_after, delimiter = ',')

    os.chdir('..')
    os.chdir('..')
    
    
#%% -------------------------------------------------------------------------------------- MUA whisker response ------------------------------------------------------------------
# os.chdir(home_directory)
highpass_cutoff = 4

do_shift = False

use_kilosort = False

to_plot_1_PSTH = [0,1,2,3]
to_plot_2_PSTH = [4,5,6,7,8,9]

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# for day in ['160810']:
for day in days:
    os.chdir(day)
    print(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    
    nchans = LFP_all_sweeps[0].shape[0]
    if nchans == 16:
        chanMap = chanMap_16
    elif nchans == 32:
        chanMap = chanMap_32

    #clean up PSTH artifacts
    if day == '220810_2':
        artifacts = np.linspace(99,120,22, dtype = int)
    if day == '221004_2':
        artifacts = np.linspace(99,123,25, dtype = int)
    elif day == '221018_1':
        artifacts = np.linspace(99,120,22, dtype = int)
    elif day == '221021_1':
        artifacts = np.linspace(99,120,22, dtype = int)
    elif day == '221024_1':
        artifacts = np.linspace(99,121,23, dtype = int)
    elif day == '221025_1':
        artifacts = np.linspace(99,120,22, dtype = int)
    elif day == '221026_1':
        artifacts = np.linspace(99,123,25, dtype = int)
    elif day == '221206_2':
        artifacts = np.linspace(99,119,21, dtype = int)
    elif day == '221206_1':
        artifacts = np.linspace(99,119,21, dtype = int)

    else:
        artifacts = []
    
    PSTH_resp_magn = np.empty([len(LFP_all_sweeps), nchans])
    PSTH_resp_magn[:] = np.NaN
    PSTH_resp_magn_rel = np.empty([len(LFP_all_sweeps), nchans])
    PSTH_resp_magn_rel[:] = np.NaN
    PSTH_resp_peak = np.empty([len(LFP_all_sweeps), nchans])
    PSTH_resp_peak[:] = np.NaN
    PSTH_resp_peak_rel = np.empty([len(LFP_all_sweeps), nchans])
    PSTH_resp_peak_rel[:] = np.NaN
    
     
    # if you want to take out certain stims, you have to take them out of stim_times array.
    def PSTH_matrix(sweeps_to_plot, take_out_artifacts = True, artifact_locs = [], stims = stim_times):
        to_plot = np.zeros([299,nchans,len(sweeps_to_plot)])
        for ind_sweep, sweep in enumerate(sweeps_to_plot):
            #PSTH_matrix is mean across trials in one sweep
            PSTH_matrix = np.zeros([299,nchans])
            bins = np.linspace(1,300,300)
            for ind_chan, chan in enumerate(list(spikes_allsweeps[sweep].keys())):
                currstim = np.zeros([299,len(stims[sweep])])
                for ind_stim, j in enumerate(list(stims[sweep])):
                    currstim[:,ind_stim] = np.histogram((spikes_allsweeps[sweep][chan][(j - 0.1*new_fs < spikes_allsweeps[sweep][chan]) & (spikes_allsweeps[sweep][chan] < j+0.2*new_fs)] - (j-0.1*new_fs)), bins)[0]
                    if take_out_artifacts:
                        currstim[:,ind_stim][artifact_locs] = 0
                PSTH_matrix[:,ind_chan] = np.squeeze(np.mean(currstim, 1)) # mean across stims for every channel
            to_plot[:,:,ind_sweep] = PSTH_matrix
        return np.squeeze(np.mean(to_plot,2))

    # save PSTH for every sweep and channel
    PSTH_responses = np.zeros([len(LFP_all_sweeps), nchans, 299])
    PSTH_responses[:] = np.NaN
    for sweep in range(len(LFP_all_sweeps)):
        PSTH_responses[sweep, :, :] = np.transpose(PSTH_matrix([sweep]))
    PSTH_responses = PSTH_responses[:,chanMap,:]

    # calculate peak and magn
    for sweep in range(len(LFP_all_sweeps)):
        temp = PSTH_matrix([sweep], artifact_locs = artifacts)
        PSTH_resp_magn[sweep,:] = np.sum(temp[110:200,:], axis = 0)
        for chan in range(nchans):
            PSTH_resp_peak[sweep,chan] = np.max(smooth(temp[110:200,chan], 6), axis = 0)
    PSTH_resp_magn_rel = PSTH_resp_magn/np.nanmean(PSTH_resp_magn[to_plot_1_PSTH,:], axis = 0)
    PSTH_resp_peak_rel = PSTH_resp_peak/np.nanmean(PSTH_resp_peak[to_plot_1_PSTH,:], axis = 0)
    PSTH_resp_magn_rel_change = np.nanmean(PSTH_resp_magn_rel[to_plot_2_PSTH,:], axis = 0) - np.nanmean(PSTH_resp_magn_rel[to_plot_1_PSTH,:], axis = 0)
    PSTH_resp_peak_rel_change = np.nanmean(PSTH_resp_peak_rel[to_plot_2_PSTH,:], axis = 0) - np.nanmean(PSTH_resp_peak_rel[to_plot_1_PSTH,:], axis = 0)
    
        
        
    if do_shift:
        shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
        # you need to add the last XX (total shift) channels to be able to subtract every image with the next one if there's a shift
        # THIS IS ASSUMING YOU HAVE POSITIVE SHIFT (CHANNEL NUMBERS GET BIGGER), which is why I call max function if not would have to change it
        total_shift = max(shift)
    else:
        total_shift = 0
        
    PSTH_all = np.asarray([PSTH_matrix([i], artifact_locs = artifacts) for i in range(10)])[:,:,chanMap]
    # LFP RESPONSES BEFORE AND AFTER before and after with shift
    if total_shift == 0:
        PSTH_before = np.mean(np.asarray([PSTH_all[i, :, :] for i in to_plot_1_PSTH]), axis = 0)
        PSTH_after = np.mean(np.asarray([PSTH_all[i, :, :] for i in to_plot_2_PSTH]), axis = 0)
    
    else:
        PSTH_before = np.mean(np.asarray([PSTH_all[i, :, shift[i]:(nchans - (total_shift -shift[i]))] for i in to_plot_1]), axis = 0)
        PSTH_after = np.mean(np.asarray([PSTH_all[i, :, shift[i]:(nchans - (total_shift -shift[i]))] for i in to_plot_2]), axis = 0)
    tot_chans = PSTH_before.shape[1]
    
    if day == '160810': # artifacts
        PSTH_before[:,25:] = PSTH_before[:,25:]/5
        PSTH_after[:,25:] = PSTH_after[:,25:]/5


    # PSTH RESPONSES BEFORE AND AFTER
    # fig, ax = plt.subplots(8,4, sharey = True, figsize = (15,15))
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     ax1.plot(smooth(PSTH_matrix(to_plot_1_PSTH)[:,chanMap[ind]],6), 'b', linewidth = 1)
    #     ax1.plot(smooth(PSTH_matrix(to_plot_2_PSTH)[:,chanMap[ind]],6), 'r', linewidth = 1)
    #     # if chan in LFP_resp_channels:
    #     #     ax[np.argwhere(chanMap == chan)[0][0]].set_facecolor("y")
    #     ax1.set_title(str(chanMap[ind]), size = 4)
    #     ax1.set_xlim([50,200])   
    #plt.savefig(f'Spiking_{to_plot_1_PSTH}_vs_{to_plot_2_PSTH}', dpi = 1000)  


    spacer = np.max(smooth(PSTH_all, 8))/2
    fig, ax = plt.subplots(figsize = (4,10))
    for ind in range(tot_chans):
        ax.plot(smooth(PSTH_before[:,ind],8) + ind * -spacer *np.ones_like(PSTH_before[:,ind]), 'k', linewidth = 1)                 
        ax.plot(smooth(PSTH_after[:,ind],8) + ind * -spacer *np.ones_like(PSTH_after[:,ind]), 'c', linewidth = 1)                     
        ax.set_xlim([50,200])
    ax.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, 5))
    # ax.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 6)
    # ax.set_yticks(np.linspace(31, 0, 5))
    ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    ax.set_xticks([50,100,150,200])
    ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 14)
    ax.set_ylabel('depth (mm)', size = 16)
    ax.set_xlabel('time from stim (ms)', size = 16)
    ax.set_ylim(bottom = -31*spacer - spacer/2)
    ax.set_ylim(top = spacer + spacer/2)
    plt.tight_layout()
    plt.savefig(f'MUA laminar before vs after.jpg', dpi = 1000, format = 'jpg')
    plt.savefig(f'MUA laminar before vs after.pdf', dpi = 1000, format = 'pdf')

    interpolation_points = 8*nchans
    to_plot_before = interpolate_matrix(smooth(PSTH_before, 5, axis = 0), space_interp = interpolation_points)
    to_plot_after = interpolate_matrix(smooth(PSTH_after, 5, axis = 0), space_interp = interpolation_points)
    vmax_overall = np.max(np.concatenate((to_plot_before[:,50:200], to_plot_after[:,50:200])))
    vmin_overall = np.min(np.concatenate((to_plot_before[:,50:200], to_plot_after[:,50:200])))
    
    fig, ax = plt.subplots(figsize = (3,8))
    # fig.suptitle('before')
    im = ax.imshow(to_plot_before, cmap = new_jet, vmin = vmin_overall, vmax = vmax_overall, aspect = 1.7)
    ax.set_xlim([50,200])
    ax.set_yticks(np.linspace(interpolation_points - 1, 0, 5))
    ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    ax.set_xticks([50,100,150,200])
    ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 14)
    ax.set_ylabel('depth (mm)', size = 16)
    ax.set_xlabel('time from stim (ms)', size = 16)
    plt.tight_layout()
    plt.savefig('MUA before.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('MUA before.pdf', dpi = 1000, format = 'pdf')
    
    fig, ax = plt.subplots(figsize = (3,8))
    # fig.suptitle('after')
    im = ax.imshow(to_plot_after, cmap = new_jet, vmin = vmin_overall, vmax = vmax_overall, aspect = 1.7)
    ax.set_xlim([50,200])
    ax.set_yticks(np.linspace(interpolation_points - 1, 0, 5))
    ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    ax.set_xticks([50,100,150,200])
    ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 14)
    ax.set_ylabel('depth (mm)', size = 16)
    ax.set_xlabel('time from stim (ms)', size = 16)
    plt.tight_layout()    
    plt.savefig('MUA after.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('MUA after.pdf', dpi = 1000, format = 'pdf')

    
    






    # PSTH for every sweep to check depth
    fig, ax = plt.subplots(1,10, figsize = (15,15), sharey = True)
    spacer = max([np.max(PSTH_matrix([i], artifact_locs = artifacts)) for i in range(10)])/2
    # spacer = 0.05 # horizontal space between lines
    for ind, ax1 in enumerate(list(ax.flatten())):
        to_plot = PSTH_matrix([ind], artifact_locs = artifacts)
        for chan in range(nchans):                        
            ax1.plot(smooth(to_plot[:,chan],20) + np.argwhere(chanMap == chan)[0][0] * -spacer, 'b', linewidth = 1)
        ax1.set_xlim([50,200])   
        ax1.set_yticks(np.linspace(-(spacer*(nchans - 1)), 0, nchans))
        if ind == 0:
            ax1.set_yticklabels(np.linspace(nchans - 1, 0, nchans).astype(int), size = 6)
        # else:
        #     ax1.set_yticklabels([])
    plt.tight_layout()
    plt.savefig('Spiking_all_sweeps', dpi = 1000)  

    
    # #change over time in all channels (relative PSTH_magn)
    # fig, ax = plt.subplots(8,4) 
    # fig.suptitle('PSTH timecourse magn')
    # for ind, ax1 in enumerate(list(ax.flatten())):                        
    #     ax1.plot(PSTH_resp_magn_rel[:,chanMap[ind]])
    #     ax1.set_title(str(chanMap[ind]))
    #     ax1.axvline(3)
    
    # fig, ax = plt.subplots(8,4) 
    # fig.suptitle('PSTH timecourse peak')
    # for ind, ax1 in enumerate(list(ax.flatten())):                        
    #     ax1.plot(PSTH_resp_peak_rel[:,chanMap[ind]])
    #     ax1.set_title(str(chanMap[ind]))
    #     ax1.axvline(3)
    
    

    
    
    # ----------------------------------------------------------------- MUA heatmap with traces --------------------------------------------------
    # # save PSTH for every sweep and channel
    PSTH_responses = np.zeros([len(LFP_all_sweeps), nchans, 299])
    PSTH_responses[:] = np.NaN
    for sweep in range(len(LFP_all_sweeps)):
        PSTH_responses[sweep, :, :] = np.transpose(PSTH_matrix([sweep]))
    PSTH_responses = PSTH_responses[:,chanMap,:]
    
    
    interpolation_points = 8*nchans
    fig, ax = plt.subplots()
    if day == '160810': # artifacts
        PSTH_responses[:,25:,:] = PSTH_responses[:,25:,:]/5
    to_plot = interpolate_matrix(smooth(np.mean(PSTH_responses[[0,1,2,3],:,50:200], axis = 0).T, 5, axis = 0), space_interp = interpolation_points)
    ax.imshow(to_plot, cmap = new_jet, aspect = 1.5)
    # ax.set_ylim([0,150])
    
    to_plot_traces = smooth(np.mean(PSTH_responses[[0,1,2,3],:,50:200], axis = 0), 5, axis = 1)
    chunk = interpolation_points/nchans # how much of the yaxis will each channel occopy
    scaling = np.max(to_plot_traces)/(chunk/2)/2 # scaling for the LFP signal to fit in one chunk on the colormap
    for chan in range(nchans):
        pos = chan*chunk + chunk/2 # position along the y axis (which is inverted in colormaps)
        ax.plot(-to_plot_traces[chan,:]/scaling + pos, color = 'black', linewidth = 1)
    ax.set_yticks(np.linspace(interpolation_points - 1, 0, 5))
    ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 16)
    ax.set_xticks([0,50,100,150])
    ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 16)
    ax.set_ylabel('depth (mm)', size = 16)
    ax.set_xlabel('time from stim (ms)', size = 16)
    plt.tight_layout()
    plt.savefig('MUA colormap with traces.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('MUA colormap with traces.pdf', dpi = 1000, format = 'pdf')


    fig, ax = plt.subplots(figsize = (5,1.5))
    norm = colors.Normalize(vmin=0, vmax=np.max(to_plot_traces)*1000)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=new_jet),
                  cax=ax, orientation='horizontal')
    ax.tick_params(axis="x", labelsize=14)    
    # ax.set_yticklabels(list(map(str, np.linspace(-1, 1.5, 6))), size = 16)
    ax.set_xlabel('instantaneous MUA rate (Hz)', size = 18)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.savefig('MUA colormap legend.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('MUA colormap legend.jpg', dpi = 1000, format = 'jpg')

    
    
    
    #% ------------------------------------------------------------- MUA trace with CSD heatmap detail of start
    
    # def LFP_average(sweeps_to_plot, stims = stim_times, LFP_all_sweeps = LFP_all_sweeps):
    #     to_plot = np.zeros([len(LFP_all_sweeps[0]), int(0.6*new_fs), len(sweeps_to_plot)])    
    #     for ind_sweep, sweep in enumerate(sweeps_to_plot):
    #         curr_to_plot = np.zeros([len(LFP_all_sweeps[0]), int(0.6*new_fs), len(stims[sweep])])
    #         for ind_stim, stim in enumerate(list(stims[sweep])):
    #             if ind_stim == len(stims[sweep]) - 1:
    #                 break
    #             if stim < 0.3*new_fs:
    #                 continue
    #             if stim + 0.3*new_fs > LFP_all_sweeps[sweep].shape[1]:
    #                 continue
    #             else:
    #                 curr_to_plot[:,:,ind_stim] = LFP_all_sweeps[sweep][:,int(stim - 0.2*new_fs):int(stim + 0.4*new_fs)]
    #         to_plot[:,:,ind_sweep] = np.squeeze(np.mean(curr_to_plot,2)) # average across stims
    #     return np.squeeze(np.mean(to_plot,2)) #average across sweeps
    
    # LFP_before = LFP_average([0,1,2,3])[chanMap,:]
    # CSD_matrix = -np.eye(nchans) # 
    # for j in range(1, CSD_matrix.shape[0] - 1):
    #     CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
    # CSD_before = - np.dot(CSD_matrix, scipy.ndimage.gaussian_filter(scipy.signal.filtfilt(b_notch, a_notch, LFP_before), (2, 0)))
    # CSD_before[0,:] = 0
    # CSD_before[-1,:] = 0
    
    # fig, ax = plt.subplots(figsize = (2,6))
    # to_plot = CSD_before[:,210:220]
    # ax.imshow(to_plot, cmap = 'jet', vmin=np.min(CSD_before), vmax=np.max(CSD_before), aspect = 2)
    
    # # highpass = butter_highpass_filter(LFP)
    # to_plot_traces = smooth(np.mean(PSTH_responses[:,:,110:120], axis = 0), 7, axis = 1)
    # interpolation_points = 1*nchans
    # chunk = interpolation_points/nchans # how much of the yaxis will each channel occopy
    # scaling = np.max(to_plot_traces)/(chunk/2)/8 # scaling for the LFP signal to fit in one chunk on the colormap
    # for chan in range(nchans):
    #     pos = chan*chunk + chunk/2 # position along the y axis (which is inverted in colormaps)
    #     ax.plot(-to_plot_traces[chan,:]/scaling + pos, color = 'black', linewidth = 1)
    # ax.set_yticks(np.linspace(interpolation_points - 1, 0, 5))
    # # ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 16)
    # # ax.set_xticks([0,50,100,150])
    # # ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 16)
    # # ax.set_ylabel('depth (mm)', size = 16)
    # # ax.set_xlabel('time from stim (ms)', size = 16)
    # # ax.set_xlim([210,230])
    # plt.tight_layout()
    # plt.savefig('')

    
    
    
    # os.chdir(home_directory)
    os.chdir(f'analysis_{day}')
    np.savetxt('PSTH_resp_magn.csv', PSTH_resp_magn, delimiter = ',')
    np.savetxt('PSTH_resp_peak.csv', PSTH_resp_peak, delimiter = ',')
    np.savetxt('PSTH_resp_magn_rel.csv', PSTH_resp_magn_rel, delimiter = ',')
    np.savetxt('PSTH_resp_peak_rel.csv', PSTH_resp_peak_rel, delimiter = ',')
    np.savetxt('PSTH_resp_magn_rel_change.csv', PSTH_resp_magn_rel_change, delimiter = ',')
    np.savetxt('PSTH_resp_peak_rel_change.csv', PSTH_resp_peak_rel_change, delimiter = ',')
    np.savetxt('to_plot_1_PSTH.csv', to_plot_1_PSTH, delimiter = ',')
    np.savetxt('to_plot_2_PSTH.csv', to_plot_2_PSTH, delimiter = ',')
    np.savetxt('PSTH_before.csv', PSTH_before, delimiter = ',')
    np.savetxt('PSTH_after.csv', PSTH_after, delimiter = ',')

    np.save('PSTH_responses.npy', PSTH_responses)

    os.chdir('..')
    os.chdir('..')


#%% ----------------------------------------------------------------------------------- delta power of LFP and CSD

do_shift = False

exclude_before = 0.1
# maybe better to take 1 second after stim for slow waves as high change they get fucked up by the stim otherwise?
exclude_after = 0.9

to_plot_1_delta = [0,1,2,3]
to_plot_2_delta = [4,5,6,7,8,9]

delta_upper = 4
delta_lower = 0.5

b_notch, a_notch = scipy.signal.iirnotch(50, 100, 1000)

smooth_over_channels = True
smooth_over_channel_count = 1

# for day in ['160615']:
for day in [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]:
    os.chdir(day)
    print(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    stim_times = pickle.load(open('stim_times','rb'))

    nchans = LFP_all_sweeps[0].shape[0]
    if nchans == 16:
        chanMap = chanMap_16
    elif nchans == 32:
        chanMap = chanMap_32
    coordinates = [[i] for i in list(np.linspace(0, 1.55, nchans))]*pq.mm

    # take out noisy stims
    stims_for_delta = copy.deepcopy(stim_times)
    if '160615' in os.getcwd(): 
        stims_for_delta[8][9] = 0
        stims_for_delta[8][15] = 0   #
        stims_for_delta[4][32] = 0   #
    elif '160729' in os.getcwd():
        stims_for_delta[7][27] = 0    
    elif '160810' in os.getcwd(): 
        stims_for_delta[0][4] = 0  
    elif '220810_2' in os.getcwd(): # 612
        stims_for_delta[9][48] = 0  
    elif '221018_1' in os.getcwd():
        stims_for_delta[4][23:27] = 0    
        stims_for_delta[4][31] = 0    
        stims_for_delta[7][41] = 0    
    elif '221021_1' in os.getcwd():
        stims_for_delta[0][1] = 0   
        stims_for_delta[7][38] = 0   
        stims_for_delta[8][54] = 0  
        stims_for_delta[7][8] = 0    
        stims_for_delta[4][64] = 0
    elif '221025_1' in os.getcwd(): # 373, 397
        stims_for_delta[4][17] = 0    
    elif '221026_1' in os.getcwd(): # 373, 397
        stims_for_delta[5][61] = 0    
        stims_for_delta[6][15] = 0    

    pickle.dump(stims_for_delta, open('stims_for_delta','wb'))

    stim_cumsum = np.cumsum(np.asarray([len(stims_for_delta[i]) for i in range(len(stims_for_delta))]))
    stim_cumsum = np.insert(stim_cumsum, 0, 0)
    
    all_stims_delta = np.zeros([nchans, sum([len(stims_for_delta[i]) for i in range(len(stims_for_delta))])])
    all_stims_delta_auto_outliers = np.zeros([nchans, sum([len(stims_for_delta[i]) for i in range(len(stims_for_delta))])])
    all_stims_delta_CSD = np.zeros([nchans, sum([len(stims_for_delta[i]) for i in range(len(stims_for_delta))])])
    all_stims_delta_CSD_auto_outliers = np.zeros([nchans, sum([len(stims_for_delta[i]) for i in range(len(stims_for_delta))])])

    fftfreq = np.fft.fftfreq(int((5 - exclude_before - exclude_after)*new_fs), d = (1/new_fs))
    
    FFT = np.zeros([len(LFP_all_sweeps), nchans, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
    PSD = np.empty([len(LFP_all_sweeps), nchans, int((5 - exclude_before - exclude_after)*new_fs)])
    PSD[:] = np.NaN
    PSD_median = np.empty([len(LFP_all_sweeps), nchans, int((5 - exclude_before - exclude_after)*new_fs)])
    PSD_median[:] = np.NaN

    delta_power = np.empty([10, nchans])
    delta_power[:] = np.NaN
    delta_power_median = np.empty([10, nchans])
    delta_power_median[:] = np.NaN

    FFT_CSD = np.zeros([len(LFP_all_sweeps), nchans, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
    PSD_CSD = np.empty([len(LFP_all_sweeps), nchans, int((5 - exclude_before - exclude_after)*new_fs)])
    PSD_CSD[:] = np.NaN
    PSD_CSD_median = np.empty([len(LFP_all_sweeps), nchans, int((5 - exclude_before - exclude_after)*new_fs)])
    PSD_CSD_median[:] = np.NaN
    delta_power_CSD = np.empty([10, nchans])
    delta_power_CSD[:] = np.NaN
    delta_power_CSD_median = np.empty([10, nchans])
    delta_power_CSD_median[:] = np.NaN

    auto_outlier_stims = [[] for i in range(10)]
    auto_outlier_stims_indices = [[] for i in range(10)]

    # do fft for every interstim period, on LFP and CSD
    for ind_sweep, LFP in enumerate(LFP_all_sweeps):
        print(ind_sweep)
        #EXCLUDE first and last stim just in case there isnt enough time, makes it easier
        FFT_current_sweep = np.zeros([len(stims_for_delta[ind_sweep] - 2), nchans, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
        FFT_CSD_current_sweep = np.zeros([len(stims_for_delta[ind_sweep] - 2), nchans, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
        FFT_current_sweep[:] = np.NaN
        FFT_CSD_current_sweep[:] = np.NaN
        FFT_current_sweep_auto_outliers = np.zeros([len(stims_for_delta[ind_sweep] - 2), nchans, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
        FFT_CSD_current_sweep_auto_outliers = np.zeros([len(stims_for_delta[ind_sweep] - 2), nchans, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
        FFT_current_sweep_auto_outliers[:] = np.NaN
        FFT_CSD_current_sweep_auto_outliers[:] = np.NaN

        for ind_stim, stim in enumerate(list(stim_times[ind_sweep][1:-1])):
            if stim == 0:
                print(f'{ind_stim}: continue')
                continue
            curr_LFP = LFP[:, int(stim+exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs)]
            #take out 50Hz noise
            curr_LFP = scipy.signal.filtfilt(b_notch, a_notch, curr_LFP)
            FFT_current_sweep[ind_stim, :,:] = np.fft.fft(curr_LFP, axis = 1)
            FFT_current_sweep_auto_outliers[ind_stim, :,:] = np.fft.fft(curr_LFP, axis = 1)
            all_stims_delta[:,stim_cumsum[ind_sweep]+ind_stim] = np.nanmean(np.abs(FFT_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]])**2, axis = 0).T
            all_stims_delta_auto_outliers[:,stim_cumsum[ind_sweep]+ind_stim] = np.nanmean(np.abs(FFT_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]])**2, axis = 0).T


            # CSD and PSD of CSD
            if smooth_over_channels:
                curr_LFP_for_CSD = scipy.ndimage.gaussian_filter1d(curr_LFP[chanMap,:], smooth_over_channel_count, axis = 0)
            else:
                curr_LFP_for_CSD = curr_LFP[chanMap,:]
            if day == '160615': # probe not very far in many electrodes are too far out. This distorts the CSD so don't include them!
                curr_CSD = elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(curr_LFP_for_CSD.T, units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False)
                FFT_CSD_current_sweep[ind_stim, 9:, :] = np.fft.fft(curr_CSD.T, axis = 1)[9:,]
                FFT_CSD_current_sweep_auto_outliers[ind_stim, 9:, :] = np.fft.fft(curr_CSD.T, axis = 1)[9:,]
                all_stims_delta_CSD[9:,stim_cumsum[ind_sweep]+ind_stim] = np.nanmean(np.abs(FFT_CSD_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]])**2, axis = 0).T[9:]
                all_stims_delta_CSD_auto_outliers[9:,stim_cumsum[ind_sweep]+ind_stim] = np.nanmean(np.abs(FFT_CSD_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]])**2, axis = 0).T[9:]
            else:
                curr_CSD = elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(curr_LFP_for_CSD.T, units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False)
                FFT_CSD_current_sweep[ind_stim, :,:] = np.fft.fft(curr_CSD.T, axis = 1)
                FFT_CSD_current_sweep_auto_outliers[ind_stim, :,:] = np.fft.fft(curr_CSD.T, axis = 1)
                all_stims_delta_CSD[:,stim_cumsum[ind_sweep]+ind_stim] = np.nanmean(np.abs(FFT_CSD_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]])**2, axis = 0).T
                all_stims_delta_CSD_auto_outliers[:,stim_cumsum[ind_sweep]+ind_stim] = np.nanmean(np.abs(FFT_CSD_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]])**2, axis = 0).T

        # define auto outlier periods as exceeding statistical outlier threshold within each sweep in each channel
        for chan in range(nchans):
            curr_delta = all_stims_delta[chan, stim_cumsum[ind_sweep]:stim_cumsum[ind_sweep + 1]]
            curr_CSD_delta = all_stims_delta_CSD[chan, stim_cumsum[ind_sweep]:stim_cumsum[ind_sweep + 1]]
            outliers_delta = (curr_delta > (np.percentile(curr_delta, 75) + 1.5*(np.abs(np.percentile(curr_delta, 75) - np.percentile(curr_delta, 25)))))
            outliers_CSD_delta = (curr_CSD_delta > (np.percentile(curr_CSD_delta, 75) + 1.5*(np.abs(np.percentile(curr_CSD_delta, 75) - np.percentile(curr_CSD_delta, 25)))))
            # print(np.where(outliers_delta == True)[0])
            # print(np.where(outliers_CSD_delta == True)[0])
            if len(np.where(outliers_delta == True)[0]) > 0:
                all_stims_delta_auto_outliers[chan, np.where(outliers_delta == True)[0] + stim_cumsum[ind_sweep]] = 0
                FFT_current_sweep_auto_outliers[np.where(outliers_delta == True)[0],:,:] = np.NaN
            if len(np.where(outliers_CSD_delta == True)[0]) > 0:
                all_stims_delta_CSD_auto_outliers[chan, np.where(outliers_CSD_delta == True)[0] + stim_cumsum[ind_sweep]] = 0
                FFT_CSD_current_sweep_auto_outliers[np.where(outliers_CSD_delta == True)[0],:,:] = np.NaN
                
            # delta_power_auto_outliers[ind_sweep, chan] = np.nanmean(curr_delta[~outliers])
            # delta_power_median_auto_outliers[ind_sweep, chan] = np.nanmedian(curr_delta[~outliers])
            auto_outlier_stims[ind_sweep].append(outliers_delta)
            auto_outlier_stims_indices[ind_sweep].append(np.where(outliers_delta == True)[0])

        # average across stims within each sweep
        PSD[ind_sweep,:,:] = np.nanmean(np.abs(FFT_current_sweep_auto_outliers)**2, axis = 0) 
        PSD_median[ind_sweep,:,:] = np.nanmedian(np.abs(FFT_current_sweep_auto_outliers)**2, axis = 0) 
        FFT[ind_sweep,:,:] = np.nanmean(FFT_current_sweep_auto_outliers, axis = 0)
    
        PSD_CSD[ind_sweep,:,:] = np.nanmean(np.abs(FFT_CSD_current_sweep_auto_outliers)**2, axis = 0) 
        PSD_CSD_median[ind_sweep,:,:] = np.nanmedian(np.abs(FFT_CSD_current_sweep_auto_outliers)**2, axis = 0) 
        FFT_CSD[ind_sweep,:,:] = np.nanmean(FFT_CSD_current_sweep_auto_outliers, axis = 0)
    
        delta_power[ind_sweep,:] = np.nanmean(PSD[ind_sweep, :, np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 0)
        delta_power_CSD[ind_sweep,:] = np.nanmean(PSD_CSD[ind_sweep, :, np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 0)

        delta_power_median[ind_sweep,:] = np.nanmedian(PSD[ind_sweep, :, np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 0)
        delta_power_CSD_median[ind_sweep,:] = np.nanmedian(PSD_CSD[ind_sweep, :, np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 0)

    # delta_power_rel = delta_power/np.nanmean(delta_power[to_plot_1_delta,:], axis = 0)
    # delta_power_rel_change = np.mean(delta_power_rel[to_plot_2_delta,:], axis = 0) - np.mean(delta_power_rel[to_plot_1_delta,:], axis = 0)
    
    #delta power timecourse over whole recording
    fig, ax = plt.subplots(8,4,sharey = True) 
    fig.suptitle(f'{day} all stims LFP')
    for ind, ax1 in enumerate(list(ax.flatten())[:nchans]):                        
        ax1.plot(all_stims_delta[chanMap[ind],:])
        for sweep in range(10):
            ax1.axvline(stim_cumsum[sweep], linestyle = '--')
        # ax1.axhline(450000, linestyle = '--')
        ax1.set_title(str(chanMap[ind]))

    # fig, ax = plt.subplots(8,4,sharey = True) 
    # fig.suptitle(f'{day} all stims LFP no outliers')
    # for ind, ax1 in enumerate(list(ax.flatten())[:nchans]):                        
    #     ax1.plot(all_stims_delta_auto_outliers[chanMap[ind],:])
    #     for sweep in range(10):
    #         ax1.axvline(stim_cumsum[sweep], linestyle = '--')
    #     # ax1.axhline(450000, linestyle = '--')
    #     ax1.set_title(str(chanMap[ind]))
        
        
        
    # # CSD delta power timecourse over whole recording
    # fig, ax = plt.subplots(8,4,sharey = True) 
    # fig.suptitle(f'{day} all stims CSD')
    # for ind, ax1 in enumerate(list(ax.flatten())[:nchans]):                        
    #     ax1.plot(all_stims_delta_CSD[ind,:])
    #     for sweep in range(10):
    #         ax1.axvline(stim_cumsum[sweep], linestyle = '--')
    #     # ax1.axhline(450000, linestyle = '--')
    #     ax1.set_title(str(chanMap[ind]))

    # # CSD delta power timecourse over whole recording
    # fig, ax = plt.subplots(8,4,sharey = True) 
    # fig.suptitle(f'{day} all stims CSD no outliers')
    # for ind, ax1 in enumerate(list(ax.flatten())[:nchans]):                        
    #     ax1.plot(all_stims_delta_CSD_auto_outliers[ind,:])
    #     for sweep in range(10):
    #         ax1.axvline(stim_cumsum[sweep], linestyle = '--')
    #     # ax1.axhline(450000, linestyle = '--')
    #     ax1.set_title(str(chanMap[ind]))



    #adjust for shift in channels
    if do_shift:
        shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
        # you need to add the last XX (total shift) channels to be able to subtract every image with the next one if there's a shift
        # THIS IS ASSUMING YOU HAVE POSITIVE SHIFT (CHANNEL NUMBERS GET BIGGER), which is why I call max function if not would have to change it
        total_shift = max(shift)
    else:
        total_shift = 0

    # redo in chanMap order
    delta_power_ALL = delta_power[:, chanMap]
    delta_power_CSD_ALL = delta_power_CSD

    if total_shift > 0 and do_shift == True:
        delta_power_shifted = np.asarray([delta_power_ALL[i, shift[i]:(nchans - (total_shift -shift[i]))] for i in range(10)])
        delta_power_CSD_shifted = np.asarray([delta_power_CSD_ALL[i, shift[i]:(nchans - (total_shift -shift[i]))] for i in range(10)])
        
    else:
        delta_power_shifted = np.asarray([delta_power_ALL[i,:] for i in range(10)])
        delta_power_CSD_shifted = np.asarray([delta_power_CSD_ALL[i,:] for i in range(10)])


    delta_power_rel_shifted =  delta_power_shifted/np.nanmean(delta_power_shifted[to_plot_1_delta,:], axis = 0)
    delta_power_CSD_rel_shifted =  delta_power_CSD_shifted/np.nanmean(delta_power_CSD_shifted[to_plot_1_delta,:], axis = 0)

    tot_chans = delta_power_rel_shifted.shape[1]

    
    #Time course of LFP delta power (drift correction)
    spacer = 1
    fig, ax = plt.subplots(figsize = (5,10))
    fig.suptitle('LFP delta')
    for chan in range(tot_chans):
        ax.plot(delta_power_rel_shifted[:,chan] + (chan + 1) * -spacer *np.ones_like(delta_power_rel_shifted[:,chan]), 'b', linewidth = 1)                 
    ax.set_yticks(np.linspace(-(spacer*(nchans - 1 - total_shift)), 0, tot_chans))
    ax.set_yticklabels(np.linspace((nchans - 1 - total_shift),0,tot_chans).astype(int), size = 6)
    ax.set_xticks(np.arange(10))
    ax.axvline(x = 3.5, linestyle = '--', linewidth = 1)
    plt.grid()
    # ax.axhline(y = [0,1,2,3,4,5])
    plt.tight_layout()
    plt.savefig('delta power timecourse', dpi = 1000)          
    
    # #average before and after as electrode picture
    # fig, ax = plt.subplots(figsize = (3,10)) 
    # fig.suptitle('LFP delta')
    # plot = ax.imshow(np.mean(delta_power_rel_shifted[to_plot_2_delta,:], axis = 0)[:,np.newaxis], aspect = 0.25, cmap = 'jet')
    # # ax.set_yticks(np.linspace(-(spacer*(31 - total_shift)), 0, tot_chans))
    # # ax.set_yticklabels(np.linspace((31 - total_shift),0,tot_chans).astype(int), size = 6)
    # fig.colorbar(plot)
    # plt.tight_layout()
    # plt.savefig('delta power diff colorplot', dpi = 1000)          

    
    #Time course of CSD delta power (drift correction)
    spacer = 1
    fig, ax = plt.subplots(figsize = (5,10))
    fig.suptitle('CSD delta')
    for chan in range(tot_chans):
        ax.plot(delta_power_CSD_rel_shifted[:,chan] + (chan + 1) * -spacer *np.ones_like(delta_power_CSD_rel_shifted[:,chan]), 'b', linewidth = 1)                 
    ax.set_yticks(np.linspace(-(spacer*(nchans - 1 - total_shift)), 0, tot_chans))
    ax.set_yticklabels(np.linspace((nchans - 1 - total_shift),0,tot_chans).astype(int), size = 6)
    ax.set_xticks(np.arange(10))
    ax.axvline(x = 3.5, linestyle = '--', linewidth = 1)
    plt.grid()
    # ax.axhline(y = [0,1,2,3,4,5])
    plt.tight_layout()
    plt.savefig('delta power CSD timecourse', dpi = 1000)          
    
    # #average before and after as electrode picture
    # fig, ax = plt.subplots(figsize = (3,10)) 
    # fig.suptitle('CSD delta')
    # plot = ax.imshow(np.mean(delta_power_CSD_rel_shifted[to_plot_2_delta,:], axis = 0)[:,np.newaxis], aspect = 0.25, cmap = 'jet', vmax = 2, vmin = 0)
    # # ax.set_yticks(np.linspace(-(spacer*(31 - total_shift)), 0, tot_chans))
    # # ax.set_yticklabels(np.linspace((31 - total_shift),0,tot_chans).astype(int), size = 6)
    # fig.colorbar(plot)
    # plt.tight_layout()
    # plt.savefig('delta power CSD diff colorplot', dpi = 1000)          



    # plt.plot(np.transpose(curr_LFP))
    # plt.plot(curr_CSD)

    # fig, ax = plt.subplots(figsize = (5,10)) 
    # for ind, ax1 in enumerate(list(ax.flatten())):        
    #     ax1.bar(0,np.mean(delta_power[to_plot_1,chanMap[ind]]))
    #     ax1.bar(1,np.mean(delta_power[to_plot_2,chanMap[ind]]))
    #     ax1.set_title(str(chanMap[ind]))
    #     ax1.set_yticklabels([])
    
    
    # os.chdir(home_directory)
    PSD_before = np.mean(PSD[to_plot_1_delta,:, :], axis = 0)
    PSD_after = np.mean(PSD[to_plot_2_delta, :, :], axis = 0)
    fig, ax = plt.subplots(8,4) 
    for ind, ax1 in enumerate(list(ax.flatten())):        
        ax1.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 30 >= fftfreq))[0]], np.abs(PSD_before[chanMap[ind], np.where(np.logical_and(0.1 <= fftfreq , 30 >= fftfreq))[0]]), 'b')
        ax1.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 30 >= fftfreq))[0]], np.abs(PSD_after[chanMap[ind], np.where(np.logical_and(0.1 <= fftfreq , 30 >= fftfreq))[0]]), 'r')
    # plt.savefig('rel_delta_change.jpg', dpi = 1000, format = 'jpg')    
    plt.savefig(f'PSD_{to_plot_1_delta}_vs_{to_plot_2_delta}', dpi = 1000)
    
    # # relative change as color plot, all channels
    # rel_delta_change = (np.mean(delta_power[to_plot_2,:], axis = 0) - np.mean(delta_power[to_plot_1,:], axis = 0))/np.mean(delta_power[to_plot_1,:], axis = 0)
    # fig, ax = plt.subplots()
    # fig.suptitle('relative change in delta power')
    # im = ax.imshow(np.reshape(rel_delta_change[chanMap], (8, 4)), cmap = 'jet', vmax = 0)
    # fig.colorbar(im)
    # plt.savefig('rel_delta_change.jpg', dpi = 1000, format = 'jpg')
    
    # #correlation with change in LFP
    # fig, ax = plt.subplots()
    # fig.suptitle('delta power vs LFP')
    # ax.scatter(LFP_min_rel_change, rel_delta_change)
    # ax.set_xlabel('relative LFP change')
    # ax.set_ylabel('relative delta change')
    # ax.set_xlim(right = 0)
    # ax.set_ylim(top = 0)
    
    
    # save everything
    # os.chdir(home_directory)
    os.chdir(f'analysis_{day}')
    
    np.savetxt('fftfreq.csv', fftfreq, delimiter = ',')
    np.savetxt('delta_power.csv', delta_power, delimiter = ',')
    np.savetxt('delta_power_median.csv', delta_power_median, delimiter = ',')
    # np.savetxt('delta_power_rel.csv', delta_power_rel, delimiter = ',')
    # np.savetxt('delta_power_rel_change.csv', delta_power_rel_change, delimiter = ',')
    np.savetxt('delta_power_rel_shifted.csv', delta_power_rel_shifted, delimiter = ',')
    np.savetxt('delta_power_shifted.csv', delta_power_shifted, delimiter = ',')
    np.save('PSD.npy', PSD)
    np.save('PSD_median.npy', PSD_median)

    np.savetxt('delta_power_CSD.csv', delta_power_CSD, delimiter = ',')
    np.savetxt('delta_power_CSD_median.csv', delta_power_CSD_median, delimiter = ',')
    np.savetxt('delta_power_CSD_rel_shifted.csv', delta_power_CSD_rel_shifted, delimiter = ',')
    np.savetxt('delta_power_CSD_shifted.csv', delta_power_CSD_shifted, delimiter = ',')
    np.save('PSD_CSD.npy', PSD_CSD)
    np.save('PSD_CSD_median.npy', PSD_CSD_median)

    np.savetxt('to_plot_1_delta.csv', to_plot_1_delta, delimiter = ',')
    np.savetxt('to_plot_2_delta.csv', to_plot_2_delta, delimiter = ',')
    np.save('delta_lower.npy', delta_lower)
    np.save('delta_upper.npy', delta_upper)
    
    pickle.dump(auto_outlier_stims_indices, open('auto_outlier_stims_indices','wb'))
    os.chdir('..')
    os.chdir('..')


# delta power of the CSD
# 

#%% -------------------------------------------------------------------------------- SLOW WAVES and CSD of slow waves
#1) extract SO times in each channel separately and their parameters, mean and median within each sweep
#2) calculate peak to peak amplitude of SOs, mean and median within each sweep
#3) interpolate LFP of each SO to match UP peak and UP crossing and save average waveform

# 221021_1: REDO EXTRACTION 
highpass_cutoff = 4 # for spike extraction
use_kilosort = False

redo_SW_extraction = True
zero_LFP = False

UP_std_cutoff = 1.3
redo_CSD_SW = False
gaussian = 1 # gaussian to smooth over channels before doing CSD

high_pass_duration = 100
high_pass_cutoff = 4 # for MUA detection of SO


exclude_before = 0.05
exclude_after = 0.75
duration_criteria_UP = 50
duration_criteria_DOWN = 0

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for day_ind, day in enumerate(days):
    
    lowpass_filtering = 4

    if day == '160614':
        UP_std_cutoff = 1.15
    elif day == '160615':
        UP_std_cutoff = 1.15
    elif day == '160615':
        UP_std_cutoff = 1.15
    elif day == '160728':
        UP_std_cutoff = 1.15
    elif day == '160729':
        UP_std_cutoff = 1
    elif day == '160810':
        UP_std_cutoff = 1.15
    elif day == '220810_2':
        UP_std_cutoff = 1.15
        highpass_cutoff = 4
        high_pass_duration = 100
    elif day == '221018_1':
        UP_std_cutoff = 1.5
        high_pass_cutoff = 4
        high_pass_duration = 100
    elif day == '221021_1':
        UP_std_cutoff = 1.5
        high_pass_cutoff = 4
        high_pass_duration = 100
        duration_criteria_DOWN = 10
        lowpass_filtering = 4
    elif day == '221024_1':
        UP_std_cutoff = 1.15
        high_pass_cutoff = 4
        high_pass_duration = 100
        duration_criteria_DOWN = 10
    elif day == '221025_1':
        UP_std_cutoff = 1.15
        high_pass_cutoff = 3.8
        high_pass_duration = 100
        duration_criteria_DOWN = 10
    elif day == '221026_1':
        UP_std_cutoff = 1.15
        high_pass_cutoff = 3.8
        high_pass_duration = 100
        duration_criteria_DOWN = 10
    elif day == '221206_2':
        UP_std_cutoff = 1.15
        high_pass_cutoff = 3.8
        high_pass_duration = 100
        duration_criteria_DOWN = 10
    elif day == '221206_1':
        UP_std_cutoff = 1.15
        high_pass_cutoff = 3.5
        high_pass_duration = 100
        duration_criteria_DOWN = 10

    os.chdir(day)
    os.chdir('pre_AP5')
    print(day)
    # print(UP_std_cutoff)
    
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    # stims_for_delta = pickle.load(open('stims_for_delta','rb'))

    nchans = LFP_all_sweeps[0].shape[0]
    if nchans == 16:
        chanMap = chanMap_16
    elif nchans == 32:
        chanMap = chanMap_32
    coordinates = [[i] for i in list(np.linspace(0, 1.55, nchans))]*pq.mm
    
    # os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    auto_outlier_stims_indices = pickle.load(open('auto_outlier_stims_indices','rb'))
    # os.chdir('..')


    if redo_SW_extraction == False:
        os.chdir([i for i in os.listdir() if 'analysis' in i][0])
        
        UP_Cross_sweeps = pickle.load(open('UP_Cross_sweeps','rb'))
        SW_waveform_sweeps_avg = np.load('SW_waveform_sweeps_avg.npy')
        SW_frequency_sweeps_avg = np.load('SW_frequency_sweeps_avg.npy', )
        SW_spiking_sweeps_avg = np.load('SW_spiking_sweeps_avg.npy', )
    
        Peak_dur_sweeps_avg_overall = np.load('Peak_dur_sweeps_avg_overall.npy', )
        SW_fslope_sweeps_avg_overall = np.load('SW_fslope_sweeps_avg_overall.npy', )
        SW_sslope_sweeps_avg_overall = np.load('SW_sslope_sweeps_avg_overall.npy', )
        SW_famp_sweeps_avg_overall = np.load('SW_famp_sweeps_avg_overall.npy', )
        SW_samp_sweeps_avg_overall = np.load('SW_samp_sweeps_avg_overall.npy', )
        spont_spiking = np.load('spont_spiking.npy', )
        os.chdir('..')

        
    else:
        SW_waveform_sweeps = [[[] for i in range(nchans)] for j in range(len(LFP_all_sweeps))]
        SW_spiking_sweeps = [[[] for i in range(nchans)] for j in range(len(LFP_all_sweeps))]
        Peak_dur_sweeps = [[[] for i in range(nchans)] for j in range(len(LFP_all_sweeps))]
        SW_fslope_sweeps = [[[] for i in range(nchans)] for j in range(len(LFP_all_sweeps))]
        SW_sslope_sweeps = [[[] for i in range(nchans)] for j in range(len(LFP_all_sweeps))]
        SW_famp_sweeps = [[[] for i in range(nchans)] for j in range(len(LFP_all_sweeps))]
        SW_samp_sweeps = [[[] for i in range(nchans)] for j in range(len(LFP_all_sweeps))]
        UP_Cross_sweeps = [[[] for i in range(nchans)] for j in range(len(LFP_all_sweeps))]
        SW_peak_to_peak_sweeps = [[[] for i in range(nchans)] for j in range(len(LFP_all_sweeps))]
        
        spont_spiking = np.zeros([len(LFP_all_sweeps),nchans])
        
        # average value within sweeps
        SW_frequency_sweeps_avg = np.zeros([len(LFP_all_sweeps), nchans])
        SW_waveform_sweeps_avg = np.zeros([len(LFP_all_sweeps), nchans, 1000])
        SW_spiking_sweeps_avg = np.zeros([len(LFP_all_sweeps), nchans, 1000])
        Peak_dur_sweeps_avg = np.zeros([len(LFP_all_sweeps), nchans])
        SW_fslope_sweeps_avg = np.zeros([len(LFP_all_sweeps), nchans])
        SW_sslope_sweeps_avg = np.zeros([len(LFP_all_sweeps), nchans])
        SW_famp_sweeps_avg = np.zeros([len(LFP_all_sweeps), nchans])
        SW_samp_sweeps_avg = np.zeros([len(LFP_all_sweeps), nchans])
        SW_frequency_sweeps_avg[:] = np.NaN
        SW_waveform_sweeps_avg[:] = np.NaN
        SW_spiking_sweeps_avg[:] = np.NaN
        Peak_dur_sweeps_avg[:] = np.NaN
        SW_fslope_sweeps_avg[:] = np.NaN
        SW_sslope_sweeps_avg[:] = np.NaN
        SW_famp_sweeps_avg[:] = np.NaN
        SW_samp_sweeps_avg[:] = np.NaN


        # median value within sweep
        SW_waveform_sweeps_median = np.zeros([len(LFP_all_sweeps), 64, 1000])
        SW_spiking_sweeps_median = np.zeros([len(LFP_all_sweeps), 64, 1000])
        Peak_dur_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
        SW_fslope_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
        SW_sslope_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
        SW_famp_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
        SW_samp_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
        SW_waveform_sweeps_median[:] = np.NaN
        SW_spiking_sweeps_median[:] = np.NaN
        Peak_dur_sweeps_median[:] = np.NaN
        SW_fslope_sweeps_median[:] = np.NaN
        SW_sslope_sweeps_median[:] = np.NaN
        SW_famp_sweeps_median[:] = np.NaN
        SW_samp_sweeps_median[:] = np.NaN


        
        # filter in slow wave range, then find every time it goes under cutoffxSD i.e.=  upstate
        for ind_sweep, LFP in enumerate(LFP_all_sweeps):
            LFP_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = lowpass_filtering*pq.Hz).as_array()
            LFP_high_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP), units = 'mV', sampling_rate = new_fs*pq.Hz), highpass_frequency = 300*pq.Hz).as_array()
            
            if day == '221025_1' and ind_sweep == 2:
                high_pass_cutoff = 2.3
            elif day == '221025_1' and ind_sweep != 2:
                high_pass_cutoff = 3.8

            #EXCLUDE first and last stim just in case there isnt enough time, makes it easier
            # for ind_stim, stim in enumerate([stims_for_delta[ind_sweep][14]]):
            for ind_stim, stim in enumerate(list(stim_times[ind_sweep][1:-1])):
                print(ind_sweep, ind_stim)
                curr_LFP_filt_total = LFP_filt[int(stim):int(stim + 5*new_fs), :]
                curr_LFP_high_filt_total = LFP_high_filt[int(stim):int(stim + 5*new_fs), :]
                
                if zero_LFP:
                    curr_LFP_filt_total = curr_LFP_filt_total - np.mean(curr_LFP_filt_total, axis = 0) # zero for each channel
                    
                curr_LFP_filt = LFP_filt[int(stim + exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs), :]
                curr_LFP_high_filt = LFP_high_filt[int(stim + exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs), :]
                
                for chan in range(nchans):
                    if ind_stim in auto_outlier_stims_indices[ind_sweep][chan]:
                        continue
                    
                    # add number of spikes during each ON state iteratively
                    chan_spiking = list(spikes_allsweeps[ind_sweep].values())[chan]
                    spont_spiking[ind_sweep,chan] = spont_spiking[ind_sweep,chan] + np.diff(np.searchsorted(chan_spiking, [stim + exclude_after*new_fs, stim + (5 - exclude_before)*new_fs]))[0]
    
                        
                    # print(chan)
                    DOWN_Cross = np.where(np.diff((curr_LFP_filt[:,chan] < 0).astype(int)) == 1)[0]
                    UP_Cross = np.where(np.diff((curr_LFP_filt[:,chan] < 0).astype(int)) == -1)[0]
                    
                    if DOWN_Cross.size == 0:
                        continue
                    
                    #if no Down crossing before or after:
                    UP_Cross = np.delete(UP_Cross, UP_Cross < DOWN_Cross[0])
                    UP_Cross = np.delete(UP_Cross, UP_Cross > DOWN_Cross[-1])
                    
                    # If too close to end of trial (need to be able to take out 500ms after for waveform)
                    UP_Cross = np.delete(UP_Cross, UP_Cross > (4.5-exclude_before-exclude_after)*new_fs)
                    
                    if UP_Cross.size == 0:
                        continue
                        
                    # 1) SLOW WAVE DETECTION USING MUA
                    # if (day in ['221018_1', '221024_1', '221025_1', '221026_1', '221206_1'] or (day == '221021_1' and ind_sweep in [0,1,2])): # day 221021_1: use MUA for sweep 0,1,2 then LFP after (better detection)
                    if day in ['221018_1', '221024_1', '221025_1', '221026_1', '221206_1']: # day 221021_1: use MUA for sweep 0,1,2 then LFP after (better detection)
                        UP_high = np.where(curr_LFP_high_filt[:,chan] > high_pass_cutoff*np.std(LFP_high_filt[:,chan]))[0]
                        
                        # If no LFP UP crossing after
                        UP_high = np.delete(UP_high, UP_high > UP_Cross[-1])
                        # if no LFP DOWN crossing before
                        UP_high = np.delete(UP_high, UP_high < DOWN_Cross[0])
                        
                        if len(UP_high) == 0:
                            continue
                        
                        DOWN_Cross_before = []
                        UP_Cross_after = []
                        DOWN_Cross_after = []
                        
                        # get LFP UP_crosses which are within duration_criteria of a MUA threshold crossing
                        for cross in UP_Cross:
                            if cross <= UP_high[0]:
                                continue
                            
                            curr_DOWN_after = DOWN_Cross[np.argwhere((cross - DOWN_Cross) < 0)[0][0]]
                            if curr_DOWN_after - cross < duration_criteria_DOWN:
                                continue
                            
                            if np.min((cross - UP_high)[(cross - UP_high) > 0]) < high_pass_duration:
                                UP_Cross_after.append(cross)
                                DOWN_Cross_before.append(DOWN_Cross[np.argmin((cross - DOWN_Cross)[(cross - DOWN_Cross) > 0])])
                                DOWN_Cross_after.append(curr_DOWN_after)

                        
                    # 2) SLOW WAVE DETECTION USING LFP CROSSING A CERTAIN NEGATIVE THRESHOLD
                    else:
                        UP_LFP = np.where(curr_LFP_filt[:,chan] < -UP_std_cutoff*np.std(LFP_filt[:,chan]))[0]
                        
                        # If no UP crossing after
                        UP_LFP = np.delete(UP_LFP, UP_LFP > UP_Cross[-1])
                        
                        # only LFP points within 750ms of a UP Crossing afterwards
                        for i in range(len(UP_LFP)):
                           diff_to_crossing = UP_Cross - UP_LFP[i]
                           if min(diff_to_crossing[diff_to_crossing > 0]) > 500:
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
                            
                            if DOWN_Cross[idx_down] < UP_LFP[i]:
                                curr_DOWN_Cross_before = DOWN_Cross[idx_down]
                                curr_DOWN_Cross_after = DOWN_Cross[idx_down + 1]
            
                            elif DOWN_Cross[idx_down] > UP_LFP[i]:
                                curr_DOWN_Cross_before = DOWN_Cross[idx_down - 1]
                                curr_DOWN_Cross_after = DOWN_Cross[idx_down]
                                
                            if UP_Cross[idx_up] > UP_LFP[i]:
                                curr_UP_Cross_after = UP_Cross[idx_up]
                            elif UP_Cross[idx_up] < UP_LFP[i]:
                                curr_UP_Cross_after = UP_Cross[idx_up + 1]
                            
                            # duration criteria:
                            if curr_UP_Cross_after - curr_DOWN_Cross_before < duration_criteria_UP:
                                continue
                            if curr_DOWN_Cross_after - curr_UP_Cross_after < duration_criteria_DOWN:
                                continue
                            
                            # if already detected as slow wave (avoid duplicates which might fuck the average), happens if it crosses threshold twice within duration_criteria of a specific UP crossing
                            if curr_UP_Cross_after in UP_Cross_after:
                                print (f'found duplicate in sweep {ind_sweep}, channel {chan + 1}')
                                continue
                            
                            else:
                                DOWN_Cross_before.append(curr_DOWN_Cross_before)
                                DOWN_Cross_after.append(curr_DOWN_Cross_after)
                                UP_Cross_after.append(curr_UP_Cross_after)
                        
                        
                    for curr_UP_Cross_after, curr_DOWN_Cross_after, curr_DOWN_Cross_before in zip(UP_Cross_after, DOWN_Cross_after, DOWN_Cross_before):
                        # save UP_Cross_after in list of lists to get the spiking with the slow wave. remember UP_cross after is indexed with a 0.5s offset from stim start.
                        UP_Cross_sweeps[ind_sweep][chan].append(curr_UP_Cross_after + int(stim + exclude_after*new_fs))
                        
                        Peak_dur_sweeps[ind_sweep][chan].append(curr_DOWN_Cross_after - curr_DOWN_Cross_before)
                        
                        #save filtered LFP
                        SW_waveform_sweeps[ind_sweep][chan].append(curr_LFP_filt_total[int(curr_UP_Cross_after - 0.5*new_fs + exclude_after*new_fs) : int(curr_UP_Cross_after + 0.5*new_fs + exclude_after*new_fs), chan])
                        
                        #save spiking (as 1ms bins)
                        temp_spiking = np.zeros(1000)
                        # set all spikes there as 1. So take out spikes within 500ms of UP crossing, then subtract 500ms before UP crossing to start at 0
                        temp_spiking[np.round(chan_spiking[np.logical_and(int(curr_UP_Cross_after + exclude_after*new_fs + stim - 0.5*new_fs) < chan_spiking, int(curr_UP_Cross_after + exclude_after*new_fs + stim + 0.5*new_fs) > chan_spiking)] - int(curr_UP_Cross_after + exclude_after*new_fs + stim - 0.5*new_fs) - 1).astype(int)] = 1
                        SW_spiking_sweeps[ind_sweep][chan].append(temp_spiking)
                        
                        idx_peak = np.argmax(curr_LFP_filt[curr_UP_Cross_after:curr_DOWN_Cross_after,chan]) # DOWN state peak
                        idx_trough = np.argmin(curr_LFP_filt[curr_DOWN_Cross_before:curr_UP_Cross_after,chan]) # UP state peak
                        
                        SW_fslope_sweeps[ind_sweep][chan].append(np.mean(np.diff(curr_LFP_filt[curr_DOWN_Cross_before:curr_DOWN_Cross_before + idx_trough])))
                        SW_sslope_sweeps[ind_sweep][chan].append(np.mean(np.diff(curr_LFP_filt[curr_DOWN_Cross_before + idx_trough:curr_UP_Cross_after+idx_peak, chan])))
                        
                        SW_famp_sweeps[ind_sweep][chan].append(np.abs(min(curr_LFP_filt[curr_DOWN_Cross_before:curr_UP_Cross_after,chan])))
                        SW_samp_sweeps[ind_sweep][chan].append(np.abs(max(curr_LFP_filt[curr_UP_Cross_after:curr_DOWN_Cross_after,chan])))
                        
                        # SW_peak_to_peak_sweeps[ind_sweep][chan].append(SW_samp_sweeps[ind_sweep][chan] - SW_famp_sweeps[ind_sweep][chan])

                        
            
        # convert spontaneous spiking to Hz by dividing by seconds (number of non-outlier stims x inter stimulus interval)
        for chan in range(nchans):
            spont_spiking[ind_sweep,chan] = spont_spiking[ind_sweep,chan]/((5 - exclude_before - exclude_after)*(len(stim_times[ind_sweep]) - 2 - len(auto_outlier_stims_indices[ind_sweep][chan])))
        
        # #mean average spont spiking per sweep
        # spont_spiking[ind_sweep,:] = spont_spiking[ind_sweep,:]/((5 - exclude_before - exclude_after)*(len(stim_times[ind_sweep]) - 2))
        
        # os.chdir([i for i in os.listdir() if 'analysis' in i][0])

        np.save('spont_spiking.npy', spont_spiking)
        
        pickle.dump(UP_Cross_sweeps, open('UP_Cross_sweeps','wb'))
         
        pickle.dump(UP_Cross_sweeps, open('UP_Cross_sweeps', 'wb')) # UP cross times
        pickle.dump(SW_waveform_sweeps, open('SW_waveform_sweeps', 'wb'))
        # pickle.dump(SW_waveform_sweeps_4, open('SW_waveform_sweeps_4', 'wb'))
        pickle.dump(Peak_dur_sweeps, open('Peak_dur_sweeps', 'wb'))
        pickle.dump(SW_spiking_sweeps, open('SW_spiking_sweeps', 'wb'))
        pickle.dump(SW_fslope_sweeps, open('SW_fslope_sweeps', 'wb'))
        pickle.dump(SW_sslope_sweeps, open('SW_sslope_sweeps', 'wb'))
        pickle.dump(SW_famp_sweeps, open('SW_famp_sweeps', 'wb'))
        pickle.dump(SW_samp_sweeps, open('SW_samp_sweeps', 'wb'))      


        # average over SW, so 1 value per sweep. 
        for ind_sweep in range(len(LFP_all_sweeps)):
            for chan in range(nchans):
                SW_frequency_sweeps_avg[ind_sweep,chan] = len(Peak_dur_sweeps[ind_sweep][chan])/(len(stim_times[ind_sweep]) - 2) # -2 because exclude first and last stim

                SW_waveform_sweeps_avg[ind_sweep,chan,:] = np.mean(np.asarray(SW_waveform_sweeps[ind_sweep][chan]), axis = 0)
                SW_spiking_sweeps_avg[ind_sweep,chan,:] = np.mean(np.asarray(SW_spiking_sweeps[ind_sweep][chan]), axis = 0)
                Peak_dur_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(Peak_dur_sweeps[ind_sweep][chan]))
                SW_fslope_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_fslope_sweeps[ind_sweep][chan]))
                SW_sslope_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_sslope_sweeps[ind_sweep][chan]))
                SW_famp_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_famp_sweeps[ind_sweep][chan]))
                SW_samp_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_samp_sweeps[ind_sweep][chan]))
                # SW_peak_to_peak_sweeps_avg = np.mean(np.asarray(SW_peak_to_peak_sweeps[ind_sweep][chan]))
                
                SW_waveform_sweeps_median[ind_sweep,chan,:] = np.median(np.asarray(SW_waveform_sweeps[ind_sweep][chan]), axis = 0)
                SW_spiking_sweeps_median[ind_sweep,chan,:] = np.median(np.asarray(SW_spiking_sweeps[ind_sweep][chan]), axis = 0)
                Peak_dur_sweeps_median[ind_sweep,chan] = np.median(np.asarray(Peak_dur_sweeps[ind_sweep][chan]))
                SW_fslope_sweeps_median[ind_sweep,chan] = np.median(np.asarray(SW_fslope_sweeps[ind_sweep][chan]))
                SW_sslope_sweeps_median[ind_sweep,chan] = np.median(np.asarray(SW_sslope_sweeps[ind_sweep][chan]))
                SW_famp_sweeps_median[ind_sweep,chan] = np.median(np.asarray(SW_famp_sweeps[ind_sweep][chan]))
                SW_samp_sweeps_median[ind_sweep,chan] = np.median(np.asarray(SW_samp_sweeps[ind_sweep][chan]))

        np.save('SW_frequency_sweeps_avg.npy', SW_frequency_sweeps_avg)
        np.save('SW_waveform_sweeps_avg.npy', SW_waveform_sweeps_avg)
        np.save('SW_spiking_sweeps_avg.npy', SW_spiking_sweeps_avg)
        np.save('Peak_dur_sweeps_avg.npy', Peak_dur_sweeps_avg)
        np.save('SW_fslope_sweeps_avg.npy', SW_fslope_sweeps_avg)
        np.save('SW_sslope_sweeps_avg.npy', SW_sslope_sweeps_avg)
        np.save('SW_famp_sweeps_avg.npy', SW_famp_sweeps_avg)
        np.save('SW_samp_sweeps_avg.npy', SW_samp_sweeps_avg)
        # np.save('SW_peak_to_peak_sweeps_avg.npy', SW_peak_to_peak_sweeps_avg)

        np.save('SW_waveform_sweeps_median.npy', SW_waveform_sweeps_median)
        np.save('SW_spiking_sweeps_median.npy', SW_spiking_sweeps_median)
        np.save('Peak_dur_sweeps_median.npy', Peak_dur_sweeps_median)
        np.save('SW_fslope_sweeps_median.npy', SW_fslope_sweeps_median)
        np.save('SW_sslope_sweeps_median.npy', SW_sslope_sweeps_median)
        np.save('SW_famp_sweeps_median.npy', SW_famp_sweeps_median)
        np.save('SW_samp_sweeps_median.npy', SW_samp_sweeps_median)


        #redo values with the mean waveforms: (works better)
        Peak_dur_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), nchans])
        SW_fslope_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), nchans])
        SW_sslope_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), nchans])
        SW_famp_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), nchans])
        SW_samp_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), nchans])
        
        for ind_sweep in range(len(LFP_all_sweeps)):
            for chan in range(nchans):
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


    # #plot SW detection in layer 5
    time_bin = 30 # seconds for each time bin
    pnts_to_extract = 1000
    for sweep in range(len(LFP_all_sweeps)):
        if day == '221025_1' and sweep == 2:
            high_pass_cutoff = 2.3
        elif day == '221025_1' and sweep != 2:
            high_pass_cutoff = 3.8
            
        if day == '221206_2':
            chan_for_SW_times = chanMap[layer_dict_1[day][0][1][0]]
        else:
            chan_for_SW_times = chanMap[layer_dict_1[day][0][2][0]]

        # LFP_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep]), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = lowpass_filtering*pq.Hz)
        # CSD_filt = np.asarray(elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(scipy.ndimage.gaussian_filter1d(LFP_filt[:,chanMap], gaussian, axis = 1), units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False).T)
        # CSD_to_plot = CSD_filt[chanMap[layer_dict_1[day][0][0][0]],:]
        
        LFP_to_plot = LFP_all_sweeps[sweep][chan_for_SW_times,:]
        LFP_high_to_plot = elephant.signal_processing.butter(neo.core.AnalogSignal(LFP_to_plot, units = 'mV', sampling_rate = new_fs*pq.Hz), highpass_frequency = 300*pq.Hz).as_array()
        LFP_low_to_plot = elephant.signal_processing.butter(neo.core.AnalogSignal(LFP_to_plot, units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = lowpass_filtering*pq.Hz).as_array()
        SWs = UP_Cross_sweeps[sweep][chan_for_SW_times]
        stims = stim_times[sweep]
        bins_nr = np.ceil(LFP_to_plot.shape[0]/(time_bin*new_fs))
        fig, ax = plt.subplots(int(bins_nr), 1, figsize = (12,9), sharex = True, sharey = True)
        for bin_ind in range(int(bins_nr)):
            ax[bin_ind].plot(LFP_to_plot[int(bin_ind*time_bin*new_fs):int(np.clip((bin_ind+1)*time_bin*new_fs, 0, LFP_to_plot.shape[0]))], linewidth = 0.25)
            # ax[bin_ind].plot(CSD_to_plot[int(bin_ind*time_bin*new_fs):int(np.clip((bin_ind+1)*time_bin*new_fs, 0, CSD_to_plot.shape[0]))]*7, linewidth = 0.25, color = 'purple')

            ax[bin_ind].plot(LFP_high_to_plot[int(bin_ind*time_bin*new_fs):int(np.clip((bin_ind+1)*time_bin*new_fs, 0, LFP_to_plot.shape[0]))]*50 - 5000, color = 'green', linewidth = 0.25)
            ax[bin_ind].axhline(high_pass_cutoff*np.std(LFP_high_to_plot*50) - 5000, linestyle = '--', color = 'green', linewidth = 0.25)
            
            ax[bin_ind].plot(LFP_low_to_plot[int(bin_ind*time_bin*new_fs):int(np.clip((bin_ind+1)*time_bin*new_fs, 0, LFP_to_plot.shape[0]))], color = 'black', linewidth = 0.25)
            ax[bin_ind].axhline(-UP_std_cutoff*np.std(LFP_low_to_plot), linestyle = '--', color = 'black', linewidth = 0.25)
            ax[bin_ind].axhline(0, linestyle = '--', color = 'black', linewidth = 0.1)
            
            for SW in np.array(SWs[np.searchsorted(SWs, bin_ind*time_bin*new_fs):np.searchsorted(SWs, np.clip((bin_ind+1)*time_bin*new_fs, 0, LFP_to_plot.shape[0]))]) - bin_ind*time_bin*new_fs:
                ax[bin_ind].axvspan(np.clip(SW - pnts_to_extract/2, 0, time_bin*new_fs), np.clip(SW + pnts_to_extract/2, 0, time_bin*new_fs), color='red', alpha=0.5)
            for stim in np.array(stims[np.searchsorted(stims, bin_ind*time_bin*new_fs):np.searchsorted(stims, np.clip((bin_ind+1)*time_bin*new_fs, 0, LFP_to_plot.shape[0]))]) - bin_ind*time_bin*new_fs:
                ax[bin_ind].axvline(stim, color = 'black', linewidth = 0.75)
                ax[bin_ind].axvspan(np.clip(stim - (5 - exclude_after)*new_fs, 0, time_bin*new_fs), np.clip(stim - exclude_before*new_fs, 0, time_bin*new_fs), color='grey', alpha=0.1)
                
            ax[bin_ind].set_ylim([-8000, 2500])
        plt.tight_layout()
        # plt.savefig('1.jpg', dpi = 1000, format = 'jpg')
        plt.savefig(f'LFP SLOW WAVES SWEEP {sweep + 1}.jpg', dpi = 1000, format = 'jpg')
        cl()




    #------------------------------------------------------ peak to peak value of LFP and CSD from SO detected in L5:
    pnts_to_extract = 1000
    if day == '221206_2':
        chan_for_peak_to_peak = layer_dict_1[day][0][1][0]
    else:
        chan_for_peak_to_peak = layer_dict_1[day][0][2][0]
    
    CSD_SW_peak_to_peak_L5 = np.zeros([len(LFP_all_sweeps), nchans])
    LFP_SW_peak_to_peak_L5 = np.zeros([len(LFP_all_sweeps), nchans])
    CSD_SW_peak_to_peak_L5_median = np.zeros([len(LFP_all_sweeps), nchans])
    LFP_SW_peak_to_peak_L5_median = np.zeros([len(LFP_all_sweeps), nchans])

    # fig, ax = plt.subplots(2,5, sharex = True, sharey = True)
    # fig.suptitle(f'{day} LFP peak to peak')
    # fig1, ax1 = plt.subplots(2,5, sharex = True, sharey = True)
    # fig1.suptitle(f'{day} CSD peak to peak')
    # fig2, ax2 = plt.subplots(2,5, sharex = True, sharey = True)
    # fig2.suptitle(f'{day} CSD')
    # fig3, ax3 = plt.subplots(2,5, sharex = True, sharey = True)
    # fig3.suptitle(f'{day} LFP')
    # if day == '160614':
    #     fig4, ax4 = plt.subplots(figsize = (5,12))
    #     fig4.suptitle(f'{day} LFP')
    #     fig5, ax5 = plt.subplots(figsize = (5,12))
    #     fig5.suptitle(f'{day} CSD')
    # fig6, ax6 = plt.subplots(2,5, sharex = True, sharey = True)
    # fig6.suptitle(f'{day} spiking')
    
    fig7, ax7 = plt.subplots(1, len(LFP_all_sweeps), sharex = True, sharey = True)
    fig7.suptitle(f'{day} all LFP slow waves')
    fig8, ax8 = plt.subplots(1, len(LFP_all_sweeps), sharex = True, sharey = True)
    fig8.suptitle(f'{day} all CSD slow waves')

    
    for sweep in range(len(LFP_all_sweeps)): 
    # for sweep in range(10): 
        LFP_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep]), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = lowpass_filtering*pq.Hz)
        CSD_filt = np.asarray(elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(scipy.ndimage.gaussian_filter1d(LFP_filt[:,chanMap], gaussian, axis = 1), units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False).T)
        CSD_filt[0,:] = 0
        CSD_filt[-1,:] = 0
        curr_LFP_peak_to_peak = []
        curr_CSD_peak_to_peak = []
        for SW in UP_Cross_sweeps[sweep][chanMap[chan_for_peak_to_peak]]:
            curr_laminar = LFP_filt.as_array().T[:, SW - int(pnts_to_extract/2):SW+int(pnts_to_extract/2)]
            curr_LFP_peak_to_peak.append(np.max(curr_laminar[:, int(pnts_to_extract/2):], axis  = 1) - np.min(curr_laminar[:, :int(pnts_to_extract/2)], axis  = 1))
            curr_CSD = CSD_filt[:, SW - int(pnts_to_extract/2):SW+int(pnts_to_extract/2)]
            curr_CSD_peak_to_peak.append(np.max(curr_CSD[:, int(pnts_to_extract/2):], axis  = 1) - np.min(curr_CSD[:, :int(pnts_to_extract/2)], axis  = 1))
            # ax2.flatten()[sweep].plot(np.squeeze(curr_CSD[layer_dict_1[day][sweep][0],:]))
            # ax3.flatten()[sweep].plot(np.squeeze(curr_laminar[chanMap[layer_dict_1[day][sweep][0]],:]))
            
            # if day == '160614' and sweep == 0:
            #     ax4.plot(curr_laminar[chanMap,:].T - np.linspace(-nchans*3000, 0, nchans))
            #     ax5.plot(curr_CSD.T - np.linspace(-nchans*500, 0, nchans))
            
            chan_spiking = list(spikes_allsweeps[sweep].values())[chanMap[layer_dict_1[day][0][2][0]]] # layer 5 channel
            curr_spiking = chan_spiking[np.searchsorted(chan_spiking, SW - int(pnts_to_extract/2)):np.searchsorted(chan_spiking, SW + int(pnts_to_extract/2))] - (SW - int(pnts_to_extract/2))
            curr_spiking = np.histogram(np.digitize(curr_spiking, bins = np.linspace(0,pnts_to_extract, 20)), bins = np.arange(21))
            # ax6.flatten()[sweep].plot(scipy.ndimage.gaussian_filter1d(curr_spiking[0].astype('float64'), 3))
            
            ax7.flatten()[sweep].plot(curr_laminar[chanMap,:].T + np.linspace(nchans*3000, 0, nchans), linewidth = 0.05)
            ax8.flatten()[sweep].plot(curr_CSD.T - np.linspace(-nchans*500, 0, nchans), linewidth = 0.05)
        
        # ax7.flatten()[sweep].plot(curr_laminar[chanMap,:] - np.linspace(-nchans*3000, 0, nchans), linewidth = 0.3)
        # ax8.flatten()[sweep].plot(curr_laminar[chanMap,:] - np.linspace(-nchans*3000, 0, nchans), linewidth = 0.3)
        
        
        ax7.flatten()[sweep].set_yticks(np.linspace(3000*(nchans), 0, nchans))
        ax7.flatten()[sweep].set_yticklabels(np.linspace(0,(nchans - 1),nchans).astype(int), size = 6)
        ax7.flatten()[sweep].set_xticks([])
        ax8.flatten()[sweep].set_yticks(np.linspace(500*(nchans), 0, nchans))
        ax8.flatten()[sweep].set_yticklabels(np.linspace(0,(nchans - 1),nchans).astype(int), size = 6)
        ax8.flatten()[sweep].set_xticks([])

        CSD_SW_peak_to_peak_L5[sweep, :] = np.mean(np.asarray(curr_CSD_peak_to_peak), axis = 0)
        LFP_SW_peak_to_peak_L5[sweep, :] = np.mean(np.asarray(curr_LFP_peak_to_peak), axis = 0)
        
        CSD_SW_peak_to_peak_L5_median[sweep, :] = np.median(np.asarray(curr_CSD_peak_to_peak), axis = 0)
        LFP_SW_peak_to_peak_L5_median[sweep, :] = np.median(np.asarray(curr_LFP_peak_to_peak), axis = 0)

    # fig2.tight_layout()
    # fig3.tight_layout()
    # fig4.tight_layout()
    # fig5.tight_layout()
    # fig6.tight_layout()
    fig7.tight_layout()
    fig8.tight_layout()
    fig7.savefig('all LFP slow waves.jpg', dpi = 2000, format = 'jpg')
    fig8.savefig('all CSD slow waves.jpg', dpi = 2000, format = 'jpg')
    cl()
        # ax.flatten()[sweep].hist(np.asarray(curr_LFP_peak_to_peak)[:,layer_dict_1[day][sweep][0]])
        # ax1.flatten()[sweep].hist(np.asarray(curr_CSD_peak_to_peak)[:,layer_dict_1[day][sweep][0]])
    np.save('CSD_SW_peak_to_peak_L5', CSD_SW_peak_to_peak_L5)
    np.save('LFP_SW_peak_to_peak_L5', LFP_SW_peak_to_peak_L5)
    np.save('CSD_SW_peak_to_peak_L5_median', CSD_SW_peak_to_peak_L5_median)
    np.save('LFP_SW_peak_to_peak_L5_median', LFP_SW_peak_to_peak_L5_median)


    # fig, ax = plt.subplots(figsize = (5,10))
    # fig.suptitle(f'{day} CSD peak to peak timecourse median')
    # ax.plot(np.divide(CSD_SW_peak_to_peak_L5_median, np.mean(CSD_SW_peak_to_peak_L5_median[[0,1,2,3],:], axis = 0), out=np.ones_like(CSD_SW_peak_to_peak_L5_median), where = np.mean(CSD_SW_peak_to_peak_L5_median[[0,1,2,3],:], axis = 0)!=0) - np.linspace(0, nchans - 1, nchans))
    # ax.set_yticks(np.linspace(-(nchans - 2), 1, nchans))
    # ax.set_yticklabels(np.linspace((nchans - 1),0,nchans).astype(int), size = 6)
    # ax.set_xticks(np.arange(10))
    # ax.axvline(x = 3.5, linestyle = '--', linewidth = 1)
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('CSD peak to peak timecourse median', dpi = 1000)          
    
    
    # fig, ax = plt.subplots(figsize = (5,10))
    # fig.suptitle(f'{day} LFP peak to peak timecourse median')
    # ax.plot(np.divide(LFP_SW_peak_to_peak_L5_median, np.mean(LFP_SW_peak_to_peak_L5_median[[0,1,2,3],:], axis = 0), out=np.ones_like(LFP_SW_peak_to_peak_L5_median), where = np.mean(LFP_SW_peak_to_peak_L5_median[[0,1,2,3],:], axis = 0)!=0) - np.linspace(0, nchans - 1, nchans))
    # ax.set_yticks(np.linspace(-(nchans - 2), 1, nchans))
    # ax.set_yticklabels(np.linspace((nchans - 1),0,nchans).astype(int), size = 6)
    # ax.set_xticks(np.arange(10))
    # ax.axvline(x = 3.5, linestyle = '--', linewidth = 1)
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('LFP peak to peak timecourse median', dpi = 1000)          


    # fig, ax = plt.subplots(figsize = (5,10))
    # fig.suptitle(f'{day} CSD peak to peak timecourse mean')
    # ax.plot(np.divide(CSD_SW_peak_to_peak_L5, np.mean(CSD_SW_peak_to_peak_L5[[0,1,2,3],:], axis = 0), out=np.ones_like(CSD_SW_peak_to_peak_L5), where = np.mean(CSD_SW_peak_to_peak_L5[[0,1,2,3],:], axis = 0)!=0) - np.linspace(0, nchans - 1, nchans))
    # ax.set_yticks(np.linspace(-(nchans - 2), 1, nchans))
    # ax.set_yticklabels(np.linspace((nchans - 1),0,nchans).astype(int), size = 6)
    # ax.set_xticks(np.arange(10))
    # ax.axvline(x = 3.5, linestyle = '--', linewidth = 1)
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('CSD peak to peak timecourse mean', dpi = 1000)          
    
    
    # fig, ax = plt.subplots(figsize = (5,10))
    # fig.suptitle(f'{day} LFP peak to peak timecourse mean')
    # ax.plot(np.divide(LFP_SW_peak_to_peak_L5, np.mean(LFP_SW_peak_to_peak_L5[[0,1,2,3],:], axis = 0), out=np.ones_like(LFP_SW_peak_to_peak_L5), where = np.mean(LFP_SW_peak_to_peak_L5[[0,1,2,3],:], axis = 0)!=0) - np.linspace(0, nchans - 1, nchans))
    # ax.set_yticks(np.linspace(-(nchans - 2), 1, nchans))
    # ax.set_yticklabels(np.linspace((nchans - 1),0,nchans).astype(int), size = 6)
    # ax.set_xticks(np.arange(10))
    # ax.axvline(x = 3.5, linestyle = '--', linewidth = 1)
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('LFP peak to peak timecourse mean', dpi = 1000)          

    #%
    # # Slow wave CSD 
    # two possibilities: CSD of slow waves detected in every channel individually, or unique slow waves across channels.
    # def CSD_average(sweeps_to_plot, stims = stim_times, smoothing = False, smooth_over = 1, time_before = 0.2, time_after = 0.4):
    #     '''
    
    #     Parameters
    #     ----------
    #     sweeps_to_plot : list
    #         sweeps in python index to plot.
    
    #     Returns
    #     -------
    #     array nchansxLFP, averaged over all the stims you want.
    
    #     '''
    #     # chan x time x sweep
    #     to_plot = np.zeros([len(LFP_all_sweeps[0]), int((time_before + time_after)*new_fs), len(sweeps_to_plot)])    
    #     for ind_sweep, sweep in enumerate(sweeps_to_plot):
    #         #chan x time x stim
    #         if stims == stim_times:
    #             # then stims[sweep] gives you the right ones. but if your stims are a single list of stims it fucks it up (slow wave CSD)
    #             curr_to_plot = np.zeros([len(LFP_all_sweeps[0]), int((time_before + time_after)*new_fs), len(stims[sweep])])
    #             curr_stims = list(stims[sweep])
    #         elif len(stims) == 1:
    #             curr_to_plot = np.zeros([len(LFP_all_sweeps[0]), int((time_before + time_after)*new_fs), len(stims[ind_sweep])])
    #             curr_stims = list(stims[ind_sweep])
    #         for ind_stim, stim in enumerate(curr_stims):
    #             if ind_stim == len(curr_stims) - 1:
    #                 break
    #             if stim < 0.3*new_fs:
    #                 continue
    #             if stim + 0.3*new_fs > LFP_all_sweeps[sweep].shape[1]:
    #                 continue
    #             else:
    #                 if smoothing == True:                  
    #                     curr_LFP = neo.core.AnalogSignal(np.transpose(scipy.ndimage.gaussian_filter1d(LFP_all_sweeps[sweep][chanMap,int(stim - time_before*new_fs):int(stim + time_after*new_fs)], smooth_over, axis = 0)), units = 'mV', sampling_rate = new_fs*pq.Hz)
    #                 else:
    #                     curr_LFP = neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep][chanMap,int(stim - time_before*new_fs):int(stim + time_after*new_fs)]), units = 'mV', sampling_rate = new_fs*pq.Hz)                    
    #                 curr_to_plot[:,:,ind_stim] = np.transpose(elephant.current_source_density.estimate_csd(curr_LFP, coordinates = coordinates, method = 'StandardCSD', process_estimate=False))
    #         to_plot[:,:,ind_sweep] = np.squeeze(np.mean(curr_to_plot,2)) # average across stims
    #     return np.squeeze(np.mean(to_plot,2)) #average across sweeps
 
    
 
    # # 1. average CSD of slow waves detected in every channel in every sweep (average of CSDs, not CSD of average LFPS). not very good
    # if os.path.isfile(f'All_SW_CSD_smoothed_{gaussian}.npy') and redo_CSD_SW == False:
    #     np.load(f'All_SW_CSD_smoothed_{gaussian}.npy')
    # else:
    #     All_SW_CSD_smoothed = np.zeros([len(LFP_all_sweeps),LFP_all_sweeps[0].shape[0], LFP_all_sweeps[0].shape[0], 1000]) # sweep, chan it was detected in, chan, time
    #     for ind_sweep in range(len(LFP_all_sweeps)): 
    #         for chan in range(LFP_all_sweeps[0].shape[0]):
    #             print(chan)
    #             All_SW_CSD_smoothed[ind_sweep,chan,:,:] = CSD_average([ind_sweep], stims = [UP_Cross_sweeps[ind_sweep][chan]], smoothing = True, smooth_over = gaussian, time_before = 0.5, time_after = 0.5)
    #     np.save(f'All_SW_CSD_smoothed_{gaussian}.npy', All_SW_CSD_smoothed)      
    
    
    
    # 2 average laminar LFP profile of SW for every sweep for SWs detected in every channel, and CSD on the AVERAGE profile (low-pass filtered and not)
    # # ALSO INTERPOLATE THE LFP TO MATCH UP THE UP AND DOWN STATE PEAKS
    # pnts_to_extract = 2000
    # pnts_to_interpolate = 4000
    # SW_count = []
    # if os.path.isfile('All_SW_avg_laminar.npy') and redo_CSD_SW == False:
    #     All_SW_avg_laminar = np.load('All_SW_avg_laminar.npy')
    #     All_SW_avg_laminar_filt = np.load('All_SW_avg_laminar_filt.npy')
    #     All_SW_avg_laminar_interp = np.load('All_SW_avg_laminar_interp.npy')
    #     All_SW_avg_laminar_filt_interp = np.load('All_SW_avg_laminar_filt_interp.npy')
    #     All_CSD_avg_laminar = np.load('All_CSD_avg_laminar.npy')      
    #     All_CSD_avg_laminar_filt = np.load('All_CSD_avg_laminar_filt.npy')     
    #     All_CSD_avg_laminar_interp = np.load('All_CSD_avg_laminar_interp.npy')      
    #     All_CSD_avg_laminar_filt_interp = np.load('All_CSD_avg_laminar_filt_interp.npy')      

    # else:
    #     All_SW_avg_laminar = np.zeros([nchans, len(LFP_all_sweeps), nchans, pnts_to_extract])  # channel detected in, sweep, chans, time
    #     All_SW_avg_laminar_filt = np.zeros([nchans, len(LFP_all_sweeps), nchans, pnts_to_extract]) 
    #     All_SW_avg_laminar_interp = np.zeros([nchans, len(LFP_all_sweeps), nchans, pnts_to_interpolate])  # channel detected in, sweep, chans, time.
    #     All_SW_avg_laminar_filt_interp = np.zeros([nchans, len(LFP_all_sweeps), nchans, pnts_to_interpolate]) 
    #     All_SW_avg_laminar_median = np.zeros([nchans, len(LFP_all_sweeps), nchans, pnts_to_extract])  # channel detected in, sweep, chans, time
    #     All_SW_avg_laminar_filt_median = np.zeros([nchans, len(LFP_all_sweeps), nchans, pnts_to_extract]) 
    #     All_SW_avg_laminar_interp_median = np.zeros([nchans, len(LFP_all_sweeps), nchans, pnts_to_interpolate])  # channel detected in, sweep, chans, time.
    #     All_SW_avg_laminar_filt_interp_median = np.zeros([nchans, len(LFP_all_sweeps), nchans, pnts_to_interpolate]) 

    #     SW_count = [] # for every channel detected in, how many SW are then averaged after applying duration criteria etc...
        
    #     for chan in range(nchans): # chan detected in
    #         print(f'working on SW detected in {chan} out of {nchans}')
    #         curr_SW_count = []
    #         curr_SW_count_with_outliers = []
    #         for sweep in range(len(LFP_all_sweeps)):
    #             curr_laminar = np.zeros([len(UP_Cross_sweeps[sweep][chan]), nchans, pnts_to_extract]) 
    #             curr_laminar_filt = np.zeros([len(UP_Cross_sweeps[sweep][chan]), nchans, pnts_to_extract]) 
    #             curr_laminar_interp = np.zeros([len(UP_Cross_sweeps[sweep][chan]), nchans, pnts_to_interpolate]) 
    #             curr_laminar_filt_interp = np.zeros([len(UP_Cross_sweeps[sweep][chan]), nchans, pnts_to_interpolate]) 
    #             curr_laminar[:] = np.NaN
    #             curr_laminar_filt[:] = np.NaN
    #             curr_laminar_interp[:] = np.NaN
    #             curr_laminar_filt_interp[:] = np.NaN
    #             curr_SW_count_with_outliers.append(len(UP_Cross_sweeps[sweep][chan]))
    #             for stim_ind, stim in enumerate(UP_Cross_sweeps[sweep][chan]):
    #                 if LFP_all_sweeps[sweep].shape[1] - stim < int(pnts_to_extract/2):
    #                     print('skipped due to too close to end of recording')
    #                     continue
                    
    #                 curr_laminar[stim_ind, :, :] = LFP_all_sweeps[sweep][:, stim - int(pnts_to_extract/2):stim+int(pnts_to_extract/2)]
    #                 if zero_LFP:
    #                     curr_laminar[stim_ind, :, :] = (curr_laminar[stim_ind, :, :].T - np.mean(curr_laminar[stim_ind, :, :], axis = 1)).T # zero for each channel

    #                 curr_laminar_filt[stim_ind, :, :] = elephant.signal_processing.butter(neo.core.AnalogSignal(np.squeeze(curr_laminar[stim_ind, :, :]).T, units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = lowpass_filtering*pq.Hz).as_array().T

    #                 # apply duration criteria for DOWN state as well:
    #                 if len(scipy.signal.find_peaks(curr_laminar_filt[stim_ind, chan, int(pnts_to_extract/2):])[0]) == 0:
    #                     curr_laminar[stim_ind, :, :] = np.NaN
    #                     curr_laminar_filt[stim_ind, :, :] = np.NaN
    #                     print('skipped due to no DOWN peak')
    #                     continue
    #                 if scipy.signal.find_peaks(curr_laminar_filt[stim_ind, chan, int(pnts_to_extract/2):])[0][0] < 50:
    #                     curr_laminar[stim_ind, :, :] = np.NaN
    #                     curr_laminar_filt[stim_ind, :, :] = np.NaN
    #                     print('skipped due to DOWN peak too early')
    #                     continue
                    
                    
    #                 # interpolate LFP to lign up the UP peaks. Use the filtered UP peaks. Then take the distance times three to the UP peak in each direction, that's what you will interpolate. possibly more than what you extract above so re extract it
    #                 # end up with same distance from zero crossing to UP peak in each slow wave. Not perfect but should be better. Obvioulsy best would be same distance from zero crossing to start of UP state but impossible to really tell when the UP state is starting... could use spiking maybe?
    #                 UP_peak = scipy.signal.find_peaks(-curr_laminar_filt[stim_ind, chan, 0:int(pnts_to_extract/2)])[0][-1] 
    #                 # DOWN_peak = scipy.signal.find_peaks(curr_laminar_filt[stim_ind, chan, int(pnts_to_extract/2):])[0][0]
                    
    #                 LFP_pnts_extracted = (int(pnts_to_extract/2) - UP_peak)*3*2 # times 3 before and after crossing
    #                 if LFP_all_sweeps[sweep].shape[1] - stim < int(LFP_pnts_extracted/2): # if too close to end of recording
    #                     continue
    #                 LFP_to_interpolate = LFP_all_sweeps[sweep][:, stim - int(LFP_pnts_extracted/2):stim+int(LFP_pnts_extracted/2)]
    #                 LFP_to_interpolate_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(LFP_to_interpolate.T, units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = lowpass_filtering*pq.Hz).as_array().T
    #                 f1_unfilt = scipy.interpolate.interp1d(np.linspace(0, pnts_to_interpolate, LFP_pnts_extracted), LFP_to_interpolate, kind = 'cubic')
    #                 f1_filt = scipy.interpolate.interp1d(np.linspace(0, pnts_to_interpolate, LFP_pnts_extracted), LFP_to_interpolate_filt, kind = 'cubic')
    #                 curr_laminar_interp[stim_ind, :,:] = f1_unfilt(np.linspace(0, pnts_to_interpolate, pnts_to_interpolate)) # now solve the function for the points we want to interpolate
    #                 curr_laminar_filt_interp[stim_ind, :,:] = f1_filt(np.linspace(0, pnts_to_interpolate, pnts_to_interpolate))
                    
    #             curr_SW_count.append(np.sum(~np.isnan(curr_laminar_interp)[:,0,0]))
                
    #             # average across SWs, mean and median
    #             All_SW_avg_laminar[chan, sweep, :,:] = np.nanmean(curr_laminar, axis = 0)
    #             All_SW_avg_laminar_filt[chan, sweep, :,:] = np.nanmean(curr_laminar_filt, axis = 0)
    #             All_SW_avg_laminar_interp[chan, sweep, :,:] = np.nanmean(curr_laminar_interp, axis = 0)
    #             All_SW_avg_laminar_filt_interp[chan, sweep, :,:] = np.nanmean(curr_laminar_filt_interp, axis = 0)
    #             All_SW_avg_laminar_median[chan, sweep, :,:] = np.nanmedian(curr_laminar, axis = 0)
    #             All_SW_avg_laminar_filt_median[chan, sweep, :,:] = np.nanmedian(curr_laminar_filt, axis = 0)
    #             All_SW_avg_laminar_interp_median[chan, sweep, :,:] = np.nanmedian(curr_laminar_interp, axis = 0)
    #             All_SW_avg_laminar_filt_interp_median[chan, sweep, :,:] = np.nanmedian(curr_laminar_filt_interp, axis = 0)

    #         print(f'{curr_SW_count} versus {curr_SW_count_with_outliers}')
    #         SW_count.append(curr_SW_count)
            
    #     np.save('All_SW_avg_laminar.npy', All_SW_avg_laminar)
    #     np.save('All_SW_avg_laminar_filt.npy', All_SW_avg_laminar_filt)
    #     np.save('All_SW_avg_laminar_interp.npy', All_SW_avg_laminar_interp)
    #     np.save('All_SW_avg_laminar_filt_interp.npy', All_SW_avg_laminar_filt_interp)
    #     np.save('All_SW_avg_laminar_median.npy', All_SW_avg_laminar_median)
    #     np.save('All_SW_avg_laminar_filt_median.npy', All_SW_avg_laminar_filt_median)
    #     np.save('All_SW_avg_laminar_interp_median.npy', All_SW_avg_laminar_interp_median)
    #     np.save('All_SW_avg_laminar_filt_interp_median.npy', All_SW_avg_laminar_filt_interp_median)

    #     pickle.dump(SW_count, open('SW_count','wb'))


    #     #CSD from average LFP profile of slow waves detected in each channel (smooth over channels before doing CSD)
    #     All_CSD_avg_laminar = np.zeros([nchans,len(LFP_all_sweeps),nchans, pnts_to_extract]) 
    #     All_CSD_avg_laminar_filt = np.zeros([nchans,len(LFP_all_sweeps),nchans, pnts_to_extract]) 
    #     All_CSD_avg_laminar_interp = np.zeros([nchans,len(LFP_all_sweeps),nchans, pnts_to_interpolate]) 
    #     All_CSD_avg_laminar_filt_interp = np.zeros([nchans,len(LFP_all_sweeps),nchans, pnts_to_interpolate]) 

    #     for chan in range(nchans): # chan detected in
    #         for sweep in range(len(LFP_all_sweeps)):
    #             All_CSD_avg_laminar[chan,sweep,:,:] = elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(scipy.ndimage.gaussian_filter1d(All_SW_avg_laminar[chan,sweep,chanMap,:], gaussian, axis = 0).T, units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False).T
    #             All_CSD_avg_laminar_filt[chan,sweep,:,:] = elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(scipy.ndimage.gaussian_filter1d(All_SW_avg_laminar_filt[chan,sweep,chanMap,:], gaussian, axis = 0).T, units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False).T
    #             All_CSD_avg_laminar_interp[chan,sweep,:,:] = elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(scipy.ndimage.gaussian_filter1d(All_SW_avg_laminar_interp[chan,sweep,chanMap,:], gaussian, axis = 0).T, units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False).T
    #             All_CSD_avg_laminar_filt_interp[chan,sweep,:,:] = elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(scipy.ndimage.gaussian_filter1d(All_SW_avg_laminar_filt_interp[chan,sweep,chanMap,:], gaussian, axis = 0).T, units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False).T
                
    #     np.save('All_CSD_avg_laminar.npy', All_CSD_avg_laminar)      
    #     np.save('All_CSD_avg_laminar_filt.npy', All_CSD_avg_laminar_filt)      
    #     np.save('All_CSD_avg_laminar_interp.npy', All_CSD_avg_laminar_interp)      
    #     np.save('All_CSD_avg_laminar_filt_interp.npy', All_CSD_avg_laminar_filt_interp)      

        
        # # plot some example slow waves to check how well they match up:
        # sweep = 0
        # chan_start = 7
        # fig, ax = plt.subplots(3,4, sharey = True)
        # fig2, ax2 = plt.subplots(3,4)
        # for ax1_ind, ax1 in enumerate(list(ax.flatten())):
        #     chan = chanMap[ax1_ind + chan_start]
        #     DOWN_peaks = []
        #     for SW in UP_Cross_sweeps[sweep][chanMap[15]]:
        #         LFP_to_plot = np.squeeze(elephant.signal_processing.butter(neo.core.AnalogSignal(LFP_all_sweeps[sweep][chan, SW-2000:SW+2000].T, units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 2*pq.Hz).as_array())
        #         ax1.plot(LFP_to_plot)
        #         ax1.axhline(0, color = 'k')
        #         ax1.set_title(f'{ax1_ind + chan_start}')
        #         ax1.set_ylim([-2000,2000])
        #         DOWN_peaks.append(LFP_to_plot[scipy.signal.find_peaks(LFP_to_plot[2000:])[0][0]])
        #     ax2.flatten()[ax1_ind].hist(DOWN_peaks, bins = 20)
                
                # ax1.plot(neo.core.AnalogSignal(LFP_all_sweeps[2][chan, SW-1000:SW+1000].T, units = 'mV', sampling_rate = new_fs*pq.Hz).as_array())
                # ax1.axhline(0, color = 'k')



    



    os.chdir('..')
    os.chdir('..')




#%% -------------------------------------------------------------------------------------- SLOW WAVES plotting

# SW_resp_channels = []
# SW_resp_channels = list(range(LFP_all_sweeps[0].shape[0]))
do_shift = False

to_plot_1 = [0,1,2,3]
to_plot_2 = [4,5,6,7,8,9]
  
# #gaussian of smoothing over channels for CSD that are plotted
# gaussian = 1
# tolerance = 1200

# for day in ['160615']:
for day in [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]:
    os.chdir(day)
    print(day)
    
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
        
    nchans = LFP_all_sweeps[0].shape[0]
    if nchans == 16:
        chanMap = chanMap_16
    elif nchans == 32:
        chanMap = chanMap_32

    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    UP_Cross_sweeps = pickle.load(open('UP_Cross_sweeps','rb'))

    SW_waveform_sweeps_avg = np.load('SW_waveform_sweeps_avg.npy')
    SW_frequency_sweeps_avg = np.load('SW_frequency_sweeps_avg.npy', )
    SW_spiking_sweeps_avg = np.load('SW_spiking_sweeps_avg.npy', )

    Peak_dur_sweeps_avg_overall = np.load('Peak_dur_sweeps_avg_overall.npy', )
    SW_fslope_sweeps_avg_overall = np.load('SW_fslope_sweeps_avg_overall.npy', )
    SW_sslope_sweeps_avg_overall = np.load('SW_sslope_sweeps_avg_overall.npy', )
    SW_famp_sweeps_avg_overall = np.load('SW_famp_sweeps_avg_overall.npy', )
    SW_samp_sweeps_avg_overall = np.load('SW_samp_sweeps_avg_overall.npy', )
    
    spont_spiking = np.load('spont_spiking.npy', )
    
    # All_SW_CSD_smoothed = np.load(f'All_SW_CSD_smoothed_{gaussian}.npy')
    
    All_SW_avg_laminar = np.load('All_SW_avg_laminar.npy')
    All_SW_avg_laminar_filt = np.load('All_SW_avg_laminar_filt.npy')
    All_SW_avg_laminar_interp = np.load('All_SW_avg_laminar_interp.npy')
    All_SW_avg_laminar_filt_interp = np.load('All_SW_avg_laminar_filt_interp.npy')
    All_CSD_avg_laminar = np.load('All_CSD_avg_laminar.npy')      
    All_CSD_avg_laminar_filt = np.load('All_CSD_avg_laminar_filt.npy')     
    All_CSD_avg_laminar_interp = np.load('All_CSD_avg_laminar_interp.npy')      
    All_CSD_avg_laminar_filt_interp = np.load('All_CSD_avg_laminar_filt_interp.npy')      

    # CSD_unique_SW_avg = np.load(f'Unique_SW_CSD_{tolerance}.npy')

    def interpolate_CSD(CSD, space_interp = 200):
        '''
        Parameters
        ----------
        CSD : timexchans
            CSD array.
            
        space_interp : number of channels in space you want
    
        Returns
        -------
        CSD interpolated in space.
        '''
        # interpolate in space, for better visualization
        #you have to flatten the CSD trace (so append each channel to the end of the previous one) and then define X and Y coords for every point
        flat_mean_CSD = np.transpose(CSD).flatten()
        grid_x = np.tile(np.linspace(1, CSD.shape[0], CSD.shape[0]), CSD.shape[1]) # repeat 1-768 16 times
        grid_y = np.repeat(np.linspace(1, CSD.shape[1], CSD.shape[1]),CSD.shape[0]) # do 1x768, 2x768 etc...
        grid_x_int, grid_y_int = np.meshgrid(np.linspace(1, CSD.shape[0], CSD.shape[0]), np.linspace(1, CSD.shape[1], space_interp)) # i.e. the grid you want to interpolate to
        mean_CSD_spatial_interpolated = scipy.interpolate.griddata((grid_x, grid_y), flat_mean_CSD, (grid_x_int, grid_y_int), method='cubic')
        return mean_CSD_spatial_interpolated



    # # --------------------------------------------------------------- laminar LFP and CSD plots ---------------------------------------------------------------------------------------------------------    
    # # which channel to use for SW detection? --> use layer 5 channel
    # # channel_for_SW = layer_dict_1[day][0][2]
    # channel_for_SW = 9
    
    # LFP_to_plot = np.squeeze(All_SW_avg_laminar_filt_interp[:, :, chanMap, :][chanMap[channel_for_SW]])
    # CSD_to_plot = np.squeeze(All_CSD_avg_laminar_filt_interp[chanMap[channel_for_SW]]) # CSD already in correct channel order
    
    # #plot average laminar LFP profile over sweeps for SW detected in specific channel
    # spacer = 1000
    # fig, ax = plt.subplots(1,10, figsize = (17,10), sharey = True) 
    # for ind, ax1 in enumerate(list(ax.flatten())):  
    #     for chan in range(nchans):
    #         if chan == channel_for_SW:
    #             ax1.plot(LFP_to_plot[ind, chan, :] + chan * -spacer, linewidth = 1, color = 'r')
    #         else:
    #             ax1.plot(LFP_to_plot[ind, chan, :] + chan * -spacer, linewidth = 1, color = 'k')
    #     ax1.set_yticks(np.linspace(-spacer*(nchans - 1), 0, nchans))
    #     if ind == 0:
    #         ax1.set_yticklabels(np.linspace((nchans - 1),0,nchans).astype(int), size = 6)
    #     else:
    #         ax1.set_yticklabels([])
    # plt.tight_layout()
    # plt.savefig('SW LFP traces all sweeps.jpg', dpi = 1000, format = 'jpg')

    # #CSD traces over sweeps (smooth over time a bit)
    # spacer = 500
    # smooth_over_time = 1
    # fig, ax = plt.subplots(1,10, figsize = (17,10), sharey = True) 
    # for ind, ax1 in enumerate(list(ax.flatten())):  
    #     for chan in range(nchans):  
    #         if chan == channel_for_SW:
    #             ax1.plot(smooth(CSD_to_plot[ind,chan,:], smooth_over_time) + chan * -spacer, linewidth = 1, color = 'r')
    #         else:
    #             ax1.plot(smooth(CSD_to_plot[ind,chan,:], smooth_over_time) + chan * -spacer, linewidth = 1, color = 'k')
    #     ax1.set_yticks(np.linspace(-spacer*(nchans - 1), 0, nchans))
    #     if ind == 0:
    #         ax1.set_yticklabels(np.linspace((nchans - 1),0,nchans).astype(int), size = 6)
    #     else:
    #         ax1.set_yticklabels([])
    # plt.tight_layout()
    # plt.savefig('SW CSD traces all sweeps.jpg', dpi = 1000, format = 'jpg')

    # #CSD heatmaps over sweeps
    # vmax = np.max(CSD_to_plot)
    # vmin = np.min(CSD_to_plot)
    # fig, ax = plt.subplots(1,10, figsize = (17,10)) 
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     to_plot = smooth(np.squeeze(CSD_to_plot[ind,:,:]), smooth_over_time, axis = 1)
    #     to_plot = interpolate_CSD(np.transpose(to_plot))
    #     ax1.imshow(to_plot, cmap = 'jet', aspect = 'auto', vmax = vmax, vmin = vmin)
    # plt.tight_layout()
    # plt.savefig('SW CSD heatmaps all sweeps.jpg', dpi = 1000, format = 'jpg')

    
    # if do_shift:
    #     shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
    #     total_shift = max(shift)
    # else:
    #     total_shift = 0
    
    # if total_shift == 0:
    #     CSD_before = np.mean(CSD_to_plot[to_plot_1, :, :], axis = 0)
    #     CSD_after = np.mean(CSD_to_plot[to_plot_2, :, :], axis = 0)
    #     LFP_before = np.mean(LFP_to_plot[to_plot_1, :, :], axis = 0)
    #     LFP_after = np.mean(LFP_to_plot[to_plot_2, :, :], axis = 0)
    # else:
    #     CSD_before = np.mean(np.asarray([CSD_to_plot[i, shift[i]:(nchans - (total_shift -shift[i])), :] for i in to_plot_1]), axis = 0)
    #     CSD_after = np.mean(np.asarray([CSD_to_plot[i, shift[i]:(nchans - (total_shift -shift[i])), :] for i in to_plot_2]), axis = 0)
    #     LFP_before = np.mean(np.asarray([LFP_to_plot[i, shift[i]:(nchans - (total_shift -shift[i])), :] for i in to_plot_1]), axis = 0)
    #     LFP_after = np.mean(np.asarray([LFP_to_plot[i, shift[i]:(nchans - (total_shift -shift[i])), :] for i in to_plot_2]), axis = 0)

    # tot_chans = CSD_before.shape[0]

    # # LFP SW BEFORE AND AFTER with shift
    # spacer = np.max(All_SW_avg_laminar_filt[chanMap[channel_for_SW],:,:,:])/2
    # fig, ax = plt.subplots(figsize = (5,10))
    # for ind in range(tot_chans):
    #     ax.plot(LFP_before[ind,:] + ind * -spacer, 'b', linewidth = 1)                 
    #     ax.plot(LFP_after[ind,:] + ind * -spacer, 'r', linewidth = 1)                     
    # ax.set_yticks(np.linspace(-(spacer*(nchans - 1 - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace((nchans - 1 - total_shift),0,tot_chans).astype(int), size = 6)
    # plt.tight_layout()
    # plt.savefig('SW LFP traces before and after', dpi = 1000)  

    # #CSD SW before and after with shift    
    # spacer = np.max(All_CSD_avg_laminar_filt[chanMap[channel_for_SW],:,:,:])/2
    # fig, ax = plt.subplots(figsize = (5,10))
    # for ind in range(tot_chans):
    #     ax.plot(CSD_before[ind,:] + ind * -spacer, 'b', linewidth = 1)                 
    #     ax.plot(CSD_after[ind,:] + ind * -spacer, 'r', linewidth = 1)                     
    # ax.set_yticks(np.linspace(-(spacer*(nchans - 1 - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace((nchans - 1 - total_shift),0,tot_chans).astype(int), size = 6)
    # plt.tight_layout()
    # plt.savefig('SW CSD traces before and after', dpi = 1000)  

    # #CSD SW heatmap before
    # vmax = np.max([CSD_before, CSD_after])
    # vmin = np.min([CSD_before, CSD_after])
    # fig, ax = plt.subplots(figsize = (10,7)) 
    # ax.imshow(interpolate_CSD(np.transpose(CSD_before)), cmap = 'jet', aspect = 10, vmax = vmax, vmin = vmin)
    # plt.tight_layout()
    # plt.savefig('SW CSD heatmap before.jpg', dpi = 1000, format = 'jpg')

    # #CSD SW heatmap after
    # fig, ax = plt.subplots(figsize = (10,7)) 
    # ax.imshow(interpolate_CSD(np.transpose(CSD_after)), cmap = 'jet', aspect = 10, vmax = vmax, vmin = vmin)
    # plt.tight_layout()
    # plt.savefig('SW CSD heatmap after.jpg', dpi = 1000, format = 'jpg')

    # #CSD SW heatmap diff
    # fig, ax = plt.subplots(figsize = (10,7)) 
    # ax.imshow(interpolate_CSD(np.transpose(CSD_after - CSD_before)), cmap = 'jet', aspect = 10, vmax = vmax, vmin = vmin)
    # plt.tight_layout()
    # plt.savefig('SW CSD heatmap diff.jpg', dpi = 1000, format = 'jpg')



    # #plot average waveforms before vs after of slow waves (need drift correction)
    # fig, ax = plt.subplots(8,4, sharey = True) 
    # for ind, ax1 in enumerate(list(ax.flatten())):   
    #     ax1.plot(np.mean(All_SW_avg_laminar_filt[chanMap[channel_for_SW],to_plot_1,chanMap[ind],:], axis = 0), 'b')
    #     ax1.plot(np.mean(All_SW_avg_laminar_filt[chanMap[channel_for_SW],to_plot_2,chanMap[ind],:], axis = 0), 'r')
    #     ax1.set_title(str(ind))
    #     ax1.set_yticklabels([])
    
    # #plot average CSD before vs after of slow waves
    # fig, ax = plt.subplots(8,4, sharey = True) 
    # for ind, ax1 in enumerate(list(ax.flatten())):   
    #     ax1.plot(np.mean(All_CSD_avg_laminar_filt[chanMap[channel_for_SW],to_plot_1,ind,:], axis = 0), 'b')
    #     ax1.plot(np.mean(All_CSD_avg_laminar_filt[chanMap[channel_for_SW],to_plot_2,ind,:], axis = 0), 'r')
    #     ax1.set_title(str(ind))
    #     ax1.set_yticklabels([])
    
    # #LFP and CSD traces on same plot
    # fig, ax = plt.subplots() 
    # for i in range(len(LFP_all_sweeps[0])):    
    #     ax.plot(All_CSD_average_laminar[chanMap[channel_for_SW],0,i,:] + np.argwhere(chanMap == i)[0][0] *4000 * np.ones_like(All_CSD_average_laminar[chanMap[channel_for_SW],0,i,:]), linewidth = 1)
    #     ax.plot(All_SW_avg_laminar_filt[chanMap[channel_for_SW],0,i,:] + np.argwhere(chanMap == i)[0][0] *4000 * np.ones_like(All_SW_avg_laminar_filt[chanMap[channel_for_SW],0,i,:]), linewidth = 2)
    
        

    
    # # plotting SWs of all detected SW in all channels (needs drift correction)
    # vmax = np.max(np.concatenate([np.mean(All_CSD_avg_laminar[chanMap[ind],to_plot_1,:,:], axis = 0), np.mean(All_CSD_avg_laminar[chanMap[ind],to_plot_2,:,:], axis = 0)]))
    # vmin = np.min(np.concatenate([np.mean(All_CSD_avg_laminar[chanMap[ind],to_plot_1,:,:], axis = 0), np.mean(All_CSD_avg_laminar[chanMap[ind],to_plot_2,:,:], axis = 0)]))
    
    # fig, ax = plt.subplots(8,4, sharey = True)
    # fig.suptitle('before detected all')
    # for ind, ax1 in enumerate(list(ax.flatten())):  
    #     ax1.imshow(np.squeeze(np.mean(All_CSD_avg_laminar[chanMap[ind],to_plot_1,:,:], axis = 0)), aspect = 25, cmap = 'jet', vmin = vmin, vmax = vmax)
    #     ax1.set_title(str(chanMap[ind]))
    #     ax1.set_yticklabels([])
    # plt.tight_layout()
    # plt.savefig('SW CSD allchans before.jpg', dpi = 1000, format = 'jpg')

    # fig, ax = plt.subplots(8,4, sharey = True)
    # fig.suptitle('after detected all')
    # for ind, ax1 in enumerate(list(ax.flatten())):  
    #     ax1.imshow(np.squeeze(np.mean(All_CSD_avg_laminar[chanMap[ind],to_plot_2,:,:], axis = 0)), aspect = 25, cmap = 'jet', vmin = vmin, vmax = vmax)
    #     ax1.set_title(str(chanMap[ind]))
    #     ax1.set_yticklabels([])
    # plt.tight_layout()
    # plt.savefig('SW CSD allchans after.jpg', dpi = 1000, format = 'jpg')

    # # difference in CSD before and after
    # fig, ax = plt.subplots(8,4, sharey = True)
    # fig.suptitle('diff detected all')
    # for ind, ax1 in enumerate(list(ax.flatten())):  
    #     ax1.imshow(np.squeeze(np.mean(All_CSD_avg_laminar[chanMap[ind],to_plot_1,:,:], axis = 0) - np.mean(All_CSD_avg_laminar[chanMap[ind],to_plot_2,:,:], axis = 0)), aspect = 25, cmap = 'jet')
    #     ax1.set_title(str(chanMap[ind]))
    #     ax1.set_yticklabels([])
    # plt.tight_layout()
    # plt.savefig('SW CSD allchans difference.jpg', dpi = 1000, format = 'jpg')

    
    
    # #UNIQUE slow waves
    # fig, ax = plt.subplots()
    # fig.suptitle('all')
    # ax.imshow(np.squeeze(np.mean(CSD_unique_SW_avg, axis = 0)), aspect = 25, cmap = 'jet')
    # # ax.set_yticklabels([])
    # plt.tight_layout()
    
    # #unique SW difference
    # fig, ax = plt.subplots()
    # fig.suptitle('difference')
    # ax.imshow(np.squeeze(np.mean(CSD_unique_SW_avg[to_plot_1,:,:], axis = 0) - np.mean(CSD_unique_SW_avg[to_plot_2,:,:], axis = 0)), aspect = 25, cmap = 'jet')
    # # ax.set_yticklabels([])
    # plt.tight_layout()
    
    # #unique SW difference
    # vmax = np.max(CSD_unique_SW_avg)
    # vmin = np.min(CSD_unique_SW_avg)
    # fig, ax = plt.subplots(1,10)
    # for ind, ax1 in enumerate(list(ax.flatten())):    
    #     ax1.imshow(np.squeeze(CSD_unique_SW_avg[ind,:,:]), cmap = 'jet', aspect = 100, vmin = vmin, vmax = vmax)
    # plt.savefig('SW CSD UNIQUES all sweeps.jpg', dpi = 1000, format = 'jpg')

    

# --------------------------------------------------------------------------------- SW spiking ---------------------------------------------------------------------------
    #plot SW spiking
    channel_for_SW = layer_dict_1[day][0][2]
    to_plot = smooth(SW_spiking_sweeps_avg, 10, axis = 2)[:,chanMap,:]
    spacer = np.max(to_plot[~np.isnan(to_plot)])
    fig, ax = plt.subplots(1,10, figsize = (17,10), sharey = True) 
    for ind, ax1 in enumerate(list(ax.flatten())):  
        for chan in range(nchans):
            if chan == channel_for_SW:
                ax1.plot(to_plot[ind, chan, :] + chan * -spacer, linewidth = 1, color = 'r')
            else:
                ax1.plot(to_plot[ind, chan, :] + chan * -spacer, linewidth = 1, color = 'k')
        ax1.set_yticks(np.linspace(-spacer*(nchans - 1), 0, nchans))
        if ind == 0:
            ax1.set_yticklabels(np.linspace((nchans - 1),0,nchans).astype(int), size = 6)
        else:
            ax1.set_yticklabels([])
    plt.tight_layout()
    plt.savefig('SW spiking all sweeps.jpg', dpi = 1000, format = 'jpg')

    # #slow-wave evoked spiking before vs after on same axis
    # spacer = 1000
    # fig, ax = plt.subplots(1,10, figsize = (17,10), sharey = True) 
    # for ind, ax1 in enumerate(list(ax.flatten())):  
    #     ax1.plot(smooth(np.mean(SW_spiking_sweeps_avg[to_plot_1, chanMap[ind],:], axis = 0),8), 'b', linewidth = 1)
    #     ax1.plot(smooth(np.mean(SW_spiking_sweeps_avg[to_plot_2, chanMap[ind],:], axis = 0),8), 'r', linewidth = 1)
    #     ax1.set_title(str(chanMap[ind]))
    #     ax1.set_yticklabels([])
    # plt.savefig('SW spiking.jpg', dpi = 1000, format = 'jpg')





# ---------------------------------------------------------------------------------SW params and spiking change ---------------------------------------------------------------------------------------

    # #relative change in individual params before vs after in all channels (needs drift correction too)
    # Freq_change = (np.mean(SW_frequency_sweeps_avg[to_plot_2,:], axis = 0) - np.mean(SW_frequency_sweeps_avg[to_plot_1,:], axis = 0))/np.mean(SW_frequency_sweeps_avg[to_plot_1,:], axis = 0)
    # Peak_dur_change = (np.mean(Peak_dur_sweeps_avg[to_plot_2,:], axis = 0) - np.mean(Peak_dur_sweeps_avg[to_plot_1,:], axis = 0))/np.mean(Peak_dur_sweeps_avg[to_plot_1,:], axis = 0)
    # Fslope_change = (np.mean(SW_fslope_sweeps_avg[to_plot_2,:], axis = 0) - np.mean(SW_fslope_sweeps_avg[to_plot_1,:], axis = 0))/np.mean(SW_fslope_sweeps_avg[to_plot_1,:], axis = 0)
    # Sslope_change = (np.mean(SW_sslope_sweeps_avg[to_plot_2,:], axis = 0) - np.mean(SW_sslope_sweeps_avg[to_plot_1,:], axis = 0))/np.mean(SW_sslope_sweeps_avg[to_plot_1,:], axis = 0)
    # Famp_change = (np.mean(SW_famp_sweeps_avg[to_plot_2,:], axis = 0) - np.mean(SW_famp_sweeps_avg[to_plot_1,:], axis = 0))/np.mean(SW_famp_sweeps_avg[to_plot_1,:], axis = 0)
    # Samp_change = (np.mean(SW_samp_sweeps_avg[to_plot_2,:], axis = 0) - np.mean(SW_samp_sweeps_avg[to_plot_1,:], axis = 0))/np.mean(SW_samp_sweeps_avg[to_plot_1,:], axis = 0)
    
    # Peak_dur_overall_change = (np.nanmean(Peak_dur_sweeps_avg_overall[to_plot_2,:], axis = 0) - np.nanmean(Peak_dur_sweeps_avg_overall[to_plot_1,:], axis = 0))/np.nanmean(Peak_dur_sweeps_avg_overall[to_plot_1,:], axis = 0)
    # Fslope_overall_change = (np.nanmean(SW_fslope_sweeps_avg_overall[to_plot_2,:], axis = 0) - np.nanmean(SW_fslope_sweeps_avg_overall[to_plot_1,:], axis = 0))/np.nanmean(SW_fslope_sweeps_avg_overall[to_plot_1,:], axis = 0)
    # Sslope_overall_change = (np.nanmean(SW_sslope_sweeps_avg_overall[to_plot_2,:], axis = 0) - np.nanmean(SW_sslope_sweeps_avg_overall[to_plot_1,:], axis = 0))/np.nanmean(SW_sslope_sweeps_avg_overall[to_plot_1,:], axis = 0)
    # Famp_overall_change = (np.nanmean(SW_famp_sweeps_avg_overall[to_plot_2,:], axis = 0) - np.nanmean(SW_famp_sweeps_avg_overall[to_plot_1,:], axis = 0))/np.nanmean(SW_famp_sweeps_avg_overall[to_plot_1,:], axis = 0)
    # Samp_overall_change = (np.nanmean(SW_samp_sweeps_avg_overall[to_plot_2,:], axis = 0) - np.nanmean(SW_samp_sweeps_avg_overall[to_plot_1,:], axis = 0))/np.nanmean(SW_samp_sweeps_avg_overall[to_plot_1,:], axis = 0)
    
    
    # # individual params of slow waves over time                                    
    # SW_feature_to_plot = SW_famp_sweeps_avg
    # fig, ax = plt.subplots(8,4) 
    # fig.suptitle('SW_famp_sweeps_avg over time')
    # for ind, ax1 in enumerate(list(ax.flatten())):   
    #     ax1.plot(SW_feature_to_plot[:,chanMap[ind]])
    #     ax1.set_title(str(chanMap[ind]))
    #     ax1.axvline(x = 3)
    #     ax1.set_yticklabels([])
    
    
    # fig, ax = plt.subplots(8,4, sharey = True)
    # fig.suptitle('Freq, Dur, Fslope, Sslope, Famp, Samp')
    # for ind, ax1 in enumerate(list(ax.flatten())):  
    #     ax1.bar(range(6), [Freq_change[chanMap[ind]], Peak_dur_overall_change[chanMap[ind]], Fslope_overall_change[chanMap[ind]], Sslope_overall_change[chanMap[ind]], Famp_overall_change[chanMap[ind]], Samp_overall_change[chanMap[ind]]])
    #     ax1.set_title(str(chanMap[ind]))
    #     ax1.set_yticklabels([])
    # plt.savefig('SW params change.jpg', dpi = 1000, format = 'jpg')
    
        
    # #slow-wave evoked spiking before vs after on same axis
    # fig, ax = plt.subplots(8,4, sharey = True)
    # for ind, ax1 in enumerate(list(ax.flatten())):  
    #     ax1.plot(smooth(np.mean(SW_spiking_sweeps_avg[to_plot_1,chanMap[ind],:], axis = 0),8), 'b', linewidth = 1)
    #     ax1.plot(smooth(np.mean(SW_spiking_sweeps_avg[to_plot_2,chanMap[ind],:], axis = 0),8), 'r', linewidth = 1)
    #     ax1.set_title(str(chanMap[ind]))
    #     ax1.set_yticklabels([])
    # plt.savefig('SW spiking.jpg', dpi = 1000, format = 'jpg')
       
    # SW_spiking_peak_change = (np.nanmean(np.max(SW_spiking_sweeps_avg, axis = 2)[to_plot_2,:], axis = 0) - np.nanmean(np.max(SW_spiking_sweeps_avg, axis = 2)[to_plot_1,:], axis = 0))/np.nanmean(np.max(SW_spiking_sweeps_avg, axis = 2)[to_plot_1,:], axis = 0)
    # SW_spiking_area_change = (np.nanmean(np.sum(SW_spiking_sweeps_avg, axis = 2)[to_plot_2,:], axis = 0) - np.nanmean(np.sum(SW_spiking_sweeps_avg, axis = 2)[to_plot_1,:], axis = 0))/np.nanmean(np.sum(SW_spiking_sweeps_avg, axis = 2)[to_plot_1,:], axis = 0)


    
    #%
    # os.chdir(home_directory)
    # os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    # # np.savetxt('SW_resp_channels.csv', SW_resp_channels, delimiter = ',')
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
    
    os.chdir('..')
    os.chdir('..')









