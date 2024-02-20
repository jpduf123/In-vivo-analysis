# -*- coding: utf-8 -*-
"""
Created on Fri May 26 00:01:19 2023

@author: Mann Lab
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import neo
import quantities as pq
import elephant
import scipy
import scipy.signal
import os
import copy
import pickle
# import natsort
from statistics import mean
import xml.etree.ElementTree as ET
# from load_intan_rhd_format import *
from operator import itemgetter
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.interpolate import UnivariateSpline
import scipy.stats as stats
from mpl_toolkits.axes_grid1 import ImageGrid
import random


# home_directory = r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS\LAMINAR_UP'
# os.chdir(home_directory)

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

def interpolate_matrix(matrix, time_interp = None, space_interp = 200):
    '''
    CAVE changed 22 05 2023
    Parameters
    ----------
    matrix : timexchans
        matrix array. 
        
    space_interp : number of channels in space you want

    Returns
    -------
    matrix interpolated in space.
    '''
    if time_interp == None:
        time_interp = matrix.shape[0]
    # interpolate in space, for better visualization
    #you have to flatten the matrix trace (so append each channel to the end of the previous one) and then define X and Y coords for every point
    flat_mean_matrix = np.transpose(matrix).flatten()
    grid_x = np.tile(np.linspace(1, matrix.shape[0], matrix.shape[0]), matrix.shape[1]) # repeat 1-768 16 times
    grid_y = np.repeat(np.linspace(1, matrix.shape[1], matrix.shape[1]),matrix.shape[0]) # do 1x768, 2x768 etc...
    grid_x_int, grid_y_int = np.meshgrid(np.linspace(1, matrix.shape[0], time_interp), np.linspace(1, matrix.shape[1], space_interp)) # i.e. the grid you want to interpolate to
    mean_matrix_spatial_interpolated = scipy.interpolate.griddata((grid_x, grid_y), flat_mean_matrix, (grid_x_int, grid_y_int), method='cubic')
    return mean_matrix_spatial_interpolated


def PCA_normed(array_to_analyze, components = 2):
    '''
    Parameters
    ----------
    array_to_analyze : n_samples x n_features
        DESCRIPTION.

    '''
    scaled = StandardScaler().fit_transform(array_to_analyze) # z-normalize (input has to be n_samples x n_features)
    pca = PCA(n_components = components)
    princ_components = pca.fit_transform(scaled)
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_
    print(f'{eigenvalues}, {var_ratio}')
    return princ_components, eigenvectors, eigenvalues, var_ratio, scaled

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    

b_notch, a_notch = scipy.signal.iirnotch(50, 100, 1000)


#%% -------------------------------------------------------------------------------------- channels for layers
# list of lists: 10 sweeps with 5 layers each (1, 2/3, 4, 5, 6)
# layer map for every mouse (approximate)
mouse_1 = list(map(np.asarray, [[1,2], [3,4,5,6,7,8], [9,10,11], [12,13,14,15,16,17,18], [19,20,21,22,23,24,25,26,27,28,29]]))
mouse_2 = list(map(np.asarray, [[4,5,6,7,8], [9,10,11,12], [13,14,15], [16,17,18,19,20,21], [22,23,24,25]]))
mouse_3 = list(map(np.asarray, [[2,3,4], [5,6,7,8,9], [10,11,12], [13,14,15,16,17,18,19], [20,21,22,23,24,25,26,27,28,29]]))
mouse_4 = list(map(np.asarray, [[1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19,20,21], [22,23,24,25]]))
mouse_5 = list(map(np.asarray, [[1,2], [3,4,5,6,7], [8,9,10], [11,12,13,14,15,16,17], [18,19,20,21,22,23]]))
mouse_6 = list(map(np.asarray, [[1,2], [3,4,5,6,7], [8,9,10], [11,12,13,14,15,16,17,18], [19,20,21]]))
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
# mouse_1_1 = list(map(np.asarray, [[5], [9], [14], [22]]))
# mouse_2_1 = list(map(np.asarray, [[12], [14], [20], [24]])) # layer 2/3 could also be 10
# mouse_3_1 = list(map(np.asarray, [[7], [11], [17], [22]])) # layer 2/3 could also be 6
# mouse_4_1 = list(map(np.asarray, [[7], [11], [16], [21]])) # layer 2/3 could also be 5,6,7
# mouse_5_1 = list(map(np.asarray, [[5], [9], [14], [19]])) # layer 2/3 could also be 4
# mouse_6_1 = list(map(np.asarray, [[5], [8], [14], [17]])) # layer 2/3 could be 4,5,6 SW CSD a bit unclear
# mouse_7_1 = list(map(np.asarray, [[5], [7], [12], [14]])) # layer 5 could also be 11
# mouse_8_1 = list(map(np.asarray, [[4], [7], [11], [14]]))
# mouse_9_1 = list(map(np.asarray, [[3], [7], [11], [14]]))
# mouse_10_1 = list(map(np.asarray, [[4], [8], [12], [14]])) #layer 2/3 could also be 4
# mouse_11_1 = list(map(np.asarray, [[4], [7], [12], [14]]))
# mouse_12_1 = list(map(np.asarray, [[3], [7], [11], [14]]))
# mouse_13_1 = list(map(np.asarray, [[5], [8], [12], [15]]))

mouse_1_1 = list(map(np.asarray, [[5], [10], [14], [22]]))
mouse_2_1 = list(map(np.asarray, [[11], [14], [19], [24]])) # layer 2/3 could also be 10
mouse_3_1 = list(map(np.asarray, [[7], [11], [16], [22]])) # layer 2/3 could also be 6
mouse_4_1 = list(map(np.asarray, [[6], [11], [15], [21]])) # layer 2/3 could also be 5,6,7
mouse_5_1 = list(map(np.asarray, [[5], [9], [14], [19]])) # layer 2/3 could also be 4
mouse_6_1 = list(map(np.asarray, [[5], [9], [14], [17]])) # layer 2/3 could be 4,5,6 SW CSD a bit unclear
mouse_7_1 = list(map(np.asarray, [[4], [7], [12], [14]])) # layer 5 could also be 11
mouse_8_1 = list(map(np.asarray, [[3], [7], [11], [14]]))
mouse_9_1 = list(map(np.asarray, [[3], [7], [11], [14]]))
mouse_10_1 = list(map(np.asarray, [[3], [7], [12], [14]])) #layer 2/3 could also be 4
mouse_11_1 = list(map(np.asarray, [[4], [7], [11], [14]])) #layer 2/3 could also be 3, #layer 5 could also be 12
mouse_12_1 = list(map(np.asarray, [[3], [7], [11], [14]]))
mouse_13_1 = list(map(np.asarray, [[5], [8], [12], [15]]))

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


#%% ---------------------------------------------------------- extract ON/OFF times based on logMUA in all sweeps

smooth_over = 15 # smooth logMUA power over how many bins for state detection
log_base = 2

exclude_before = 0.1*new_fs
exclude_after = 1*new_fs

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# use days with a nowhisker stim
for day_ind, day in enumerate(days):
# for day_ind, day in enumerate(['160803']):
    os.chdir(day)
    print(day)
    os.chdir('pre_AP5')
    
    if day == '160614':
        highpass_cutoff = 3
    elif day == '160514':
        highpass_cutoff = 4
    else:
        highpass_cutoff = 4

    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    MUA_all_sweeps = pickle.load(open('MUA_all_sweeps','rb'))
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

    nsweeps = len(LFP_all_sweeps)
    
    # use a layer 5 channel log MUA to detect UP vs DOWN states. This is the number of channel from layer 1 onwards, not from the channel map!
    if day == '160614':
        chan_for_MUA = 17
    elif day == '160615':
        chan_for_MUA = 24
    elif day == '160622':
        chan_for_MUA = 21
    elif day == '160728':
        chan_for_MUA = 21
    elif day == '160729':
        chan_for_MUA = 19
    elif day == '160810':
        chan_for_MUA = 16
    else:
        chan_for_MUA = 20

    ON_states_starts_allchans_allsweeps = []
    ON_states_stops_allchans_allsweeps = []
    
    ON_states_starts_avg_allsweeps = []
    ON_states_stops_avg_allsweeps = []

    
    bins = 1000 # bins for logMUA histogram
    

    # -------------------------------------------------------------------- detect ON_OFF threshold in every channel -------------------------
    for sweep in range(nsweeps):
        print(sweep)
        curr_LFP = LFP_all_sweeps[sweep][chanMap,:]
        
        # # take log and smooth MUA power across time
        MUA_power_binned_log_smoothed = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_all_sweeps[sweep][chanMap,:]**2), (0,80))
        MUA_power_binned_smoothed = scipy.ndimage.gaussian_filter(MUA_all_sweeps[sweep][chanMap,:]**2, (0,80))
        # # normalize within each channel, take median value in each channel
        MUA_power_binned_rel = (MUA_all_sweeps[sweep][chanMap,:].T**2/np.median(MUA_all_sweeps[sweep][chanMap,:]**2, axis = 1)).T
        MUA_power_binned_log_smoothed_rel = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_power_binned_rel), (0,80))


        OFF_ON_thresholds = []
        bimodal_channels = []
        # take peaks of bimodal MUA power distribution
        for chan in range(nchans):
            if len(scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(MUA_power_binned_log_smoothed[chan,:], bins = bins)[0], smooth_over), distance = 90)[0]) == 1:
                #check if there is a hump in the derivative and use that
                X = np.histogram(MUA_power_binned_log_smoothed[chan,:], bins = bins)[1][:-1]
                fit = UnivariateSpline(X, scipy.ndimage.gaussian_filter1d(np.histogram(MUA_power_binned_log_smoothed[chan,:], bins = bins)[0], smooth_over),s=0,k=4)
                first_dev = scipy.ndimage.gaussian_filter1d(fit.derivative(n=1)(X), 3)
                if len(scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(first_dev[:500], 4), distance = 150, prominence = 25)[0]) > 1:
                    OFF_peak = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(MUA_power_binned_log_smoothed[chan,100:], bins = bins)[0], smooth_over), distance = 150)[0][0] + 100
                    ON_peak = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(first_dev[:500], 4), distance = 150, prominence = 25)[0][1]
                    OFF_value = np.histogram(MUA_power_binned_log_smoothed[chan,:], bins = bins)[1][OFF_peak]
                    ON_value = np.histogram(MUA_power_binned_log_smoothed[chan,:], bins = bins)[1][ON_peak]
                    OFF_ON_thresholds.append(OFF_value + 0.6*(ON_value - OFF_value))
                
                else:
                    OFF_ON_thresholds.append(np.NaN)
                
            else:
                OFF_peak = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(MUA_power_binned_log_smoothed[chan,:], bins = bins)[0], smooth_over), distance = 90)[0][0]
                ON_peak = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(MUA_power_binned_log_smoothed[chan,:], bins = bins)[0], smooth_over), distance = 90)[0][1]
                OFF_value = np.histogram(MUA_power_binned_log_smoothed[chan,:], bins = bins)[1][OFF_peak]
                ON_value = np.histogram(MUA_power_binned_log_smoothed[chan,:], bins = bins)[1][ON_peak]
                # OFF_ON_threshold is about halfway between both peaks (varies in Nghiem and Sanchez-vives)
                OFF_ON_thresholds.append(OFF_value + 0.6*(ON_value - OFF_value))
                bimodal_channels.append(chan)
        OFF_ON_thresholds = np.asarray(OFF_ON_thresholds)
        OFF_ON_threshold = OFF_ON_thresholds[chan_for_MUA]
        
        # channels with detectable ON OFF MUA peaks
        # chans_for_time_matrix = np.argwhere(np.isnan(OFF_ON_thresholds) != True)     
        
        # plot MUA power histogram for each channel
        fig, ax = plt.subplots(int(nchans/8),8, figsize = (20,10))
        to_plot = MUA_power_binned_log_smoothed
        # fig.suptitle(f'{day}')
        spacer = 0
        bins = 1000
        tot_chans = MUA_power_binned_log_smoothed.shape[0]
        for i in range(tot_chans):
            X = np.histogram(to_plot[i,:], bins = bins)[1][:-1]
            ax.flatten()[i].plot(X, np.histogram(to_plot[i,:], bins = bins)[0] - i*spacer*np.ones_like(np.histogram(to_plot[i,:], bins = bins)[0]), linewidth = 1)
            ax.flatten()[i].plot(X, scipy.ndimage.gaussian_filter1d(np.histogram(to_plot[i,:], bins = bins)[0], smooth_over) - i*spacer, color = 'k')    
    
            # ax.flatten()[i].plot(X, np.insert(np.diff(scipy.ndimage.gaussian_filter1d(np.histogram(to_plot[i,:], bins = 500)[0], smooth_over) - i*spacer), 0 , 0)*25, color = 'c')
            # ax.flatten()[i].plot(X[1:], np.insert(np.diff(np.diff(scipy.ndimage.gaussian_filter1d(np.histogram(to_plot[i,:], bins = 500)[0], smooth_over) - i*spacer)), 0 , 0)*200, color = 'green')
    
            ax.flatten()[i].set_yticks([])
            ax.flatten()[i].set_xticks([])
            ax.flatten()[i].axvline(OFF_ON_thresholds[i], color = 'red')    
            if i not in bimodal_channels:
                fit = UnivariateSpline(X, scipy.ndimage.gaussian_filter1d(np.histogram(to_plot[i,:], bins = bins)[0], smooth_over),s=0,k=4)
                first_dev = scipy.ndimage.gaussian_filter1d(fit.derivative(n=1)(X), 3)
                ax.flatten()[i].plot(X, scipy.ndimage.gaussian_filter1d(first_dev, 3), color = 'c')
                for peak in scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(first_dev[:600], 3), distance = 150, prominence = 15)[0]:
                    ax.flatten()[i].axvline(X[peak], color = 'green', linestyle = '--', linewidth = 1)
            plt.tight_layout()
        plt.savefig(f'logMUA threshold for all channels sweep {sweep + 1}.jpg', dpi = 1000, format = 'jpg')
        
        # clean up bimodal channels
        if day == '160614':
            bimodal_channels = [15,16,17,18,19,20,21,22,23,24,25,26]
        if day == '160615':
            bimodal_channels = [17,18,19,20,21,22,23,24,25,26]
        if day =='160622':
            bimodal_channels = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
        if day == '160728':
            bimodal_channels = [12,14,15,16,17,18,19,20,21,22,23,24,25]
        if day == '160729':
            bimodal_channels = [11,12,13,14,15,16,17,18,19,21,22,23]
        if day == '160810':
            bimodal_channels = [11,12,13,14,15,16,17,18]
        if day == '220810_2':
            bimodal_channels = [8,9,10,11,12,13,14,15]
        if day == '221018_1':
            bimodal_channels = [10,11,12,13,14,15]
        if day =='221021_1':
            bimodal_channels = [10,11,12,13,14,15]
        if day == '221024_1':
            bimodal_channels = [10,11,12,13,14,15]
        if day == '221025_1':
            bimodal_channels = [10,11,12,13,14,15]
        if day == '221026_1':
            bimodal_channels = [10,11,12,13,14,15]
        if day == '221206_1':
            bimodal_channels = [10,11,12,13,14,15]

        if day == '160801':
            bimodal_channels = [12,13,14,15,16,17,18]
        if day == '160803':
            bimodal_channels = [13,14,15,16,17,18]
        if day == '160804':
            bimodal_channels = [12,13,14,15,16,17,18]
        if day == '160811':
            bimodal_channels = [12,13,14,15,16,17,18]


        # ----------------------------------------------------------------------- ON OFF threshold in avg of logMUA during interstim interval
        smooth_over = 25
        inter_stim_indcs = []
        for stim in stim_times[sweep][:-1]:
            inter_stim_indcs.append(np.linspace(int(stim + exclude_after), int(stim + (5000-exclude_before)), int((5000 - exclude_after - exclude_before + 1))).astype(int))
        inter_stim_indcs = np.concatenate(inter_stim_indcs)
        
        bimodal_avg_rel_MUA = np.mean(MUA_power_binned_log_smoothed_rel[bimodal_channels,:], axis = 0)
        bimodal_avg_rel_MUA_nostims = np.mean(MUA_power_binned_log_smoothed_rel[bimodal_channels,:][:,inter_stim_indcs], axis = 0)
        OFF_peak_avg = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(bimodal_avg_rel_MUA_nostims, bins = bins)[0], smooth_over), distance = 90)[0][0]
        OFF_value_avg = np.histogram(bimodal_avg_rel_MUA_nostims, bins = bins)[1][OFF_peak_avg]
        # if day == '160615' or day == '160729' or day == '160810':
        if day == '220810_2':
        #     pass
            OFF_ON_threshold_avg = OFF_value_avg + np.abs(1*(OFF_value_avg-np.histogram(bimodal_avg_rel_MUA, bins = bins)[1][0]))
        else:
            ON_peak_avg = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(bimodal_avg_rel_MUA_nostims, bins = bins)[0], smooth_over), distance = 90)[0][1]
            ON_value_avg = np.histogram(bimodal_avg_rel_MUA_nostims, bins = bins)[1][ON_peak_avg]
            # OFF_ON_threshold is about halfway between both peaks (varies in Nghiem and Sanchez-vives)
            OFF_ON_threshold_avg = OFF_value_avg + 0.3*(ON_value_avg - OFF_value_avg)
            
        # fig, ax = plt.subplots()
        # fig.suptitle(f'{day}')
        # ax.hist(bimodal_avg_rel_MUA, bins = 1000)
        # ax.plot(np.histogram(bimodal_avg_rel_MUA, bins = 1000)[1][:-1], scipy.ndimage.gaussian_filter1d(np.histogram(bimodal_avg_rel_MUA, bins = 1000)[0], smooth_over), color = 'k')    
        # ax.axvline(OFF_ON_threshold_avg, color = 'red')    
        
        
        #---------------------------------------------------------------------------- detect ON state starts/ends in all channels individually
        OFF_duration_threshold = 100
        ON_duration_threshold = 100
    
        ON_states_starts_allchans = []
        ON_states_stops_allchans = []
    
        for chan in range(nchans):
            if np.isnan(OFF_ON_thresholds[chan]):
                print(f'{chan} doesnt have two spiking peaks')
                ON_states_starts_allchans.append([])
                ON_states_stops_allchans.append([])
            else:        
                ON_states_starts = np.where(np.diff((MUA_power_binned_log_smoothed[chan,1000:-1000] < OFF_ON_thresholds[chan]).astype(int)) == -1)[0]/new_fs + 1 # make sure there's enough time at start and end of recording (add 1 second because starts at 1 sec)
                ON_states_stops = np.where(np.diff((MUA_power_binned_log_smoothed[chan,1000:-1000] < OFF_ON_thresholds[chan]).astype(int)) == 1)[0]/new_fs + 1
                if ON_states_starts.size == 0:
                    ON_states_starts_allchans.append([])
                    ON_states_stops_allchans.append([])
                else:
                    #make sure the first ON_state start is before the first ON_state stop
                    ON_states_starts = ON_states_starts[ON_states_starts<ON_states_stops[-1]]
                    ON_states_stops = ON_states_stops[ON_states_stops>ON_states_starts[0]]
                    
                    # take out ON or OFF states that are too short
                    shorts = []
                    for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts, ON_states_stops)):
                        if ON_stop - ON_start < ON_duration_threshold/1000:
                            shorts.append(i)
                    ON_states_starts = np.delete(ON_states_starts, shorts)
                    ON_states_stops = np.delete(ON_states_stops, shorts)
                    
                    ON_states_starts_allchans.append(ON_states_starts)
                    ON_states_stops_allchans.append(ON_states_stops)
        ON_states_starts_allchans_allsweeps.append(ON_states_starts_allchans)
        ON_states_stops_allchans_allsweeps.append(ON_states_stops_allchans)

        #---------------------------------------------------------------------------- detect ON state starts/ends on avg MUA
    
        ON_states_starts_avg = []
        ON_states_stops_avg = []
    
        ON_states_starts_avg = np.where(np.diff((bimodal_avg_rel_MUA[1000:-1000] < OFF_ON_threshold_avg).astype(int)) == -1)[0]/new_fs + 1 # make sure there's enough time at start and end of recording (add 1 second because starts at 1 sec)
        ON_states_stops_avg = np.where(np.diff((bimodal_avg_rel_MUA[1000:-1000] < OFF_ON_threshold_avg).astype(int)) == 1)[0]/new_fs + 1
        
        #make sure the first ON_state start is before the first ON_state stop
        ON_states_starts_avg = ON_states_starts_avg[ON_states_starts_avg<ON_states_stops_avg[-1]]
        ON_states_stops_avg = ON_states_stops_avg[ON_states_stops_avg>ON_states_starts_avg[0]]
        
        # take out ON or OFF states that are too short
        shorts = []
        for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts_avg, ON_states_stops_avg)):
            if ON_stop - ON_start < ON_duration_threshold/1000:
                shorts.append(i)
        ON_states_starts_avg = np.delete(ON_states_starts_avg, shorts)
        ON_states_stops_avg = np.delete(ON_states_stops_avg, shorts)   
            
    
        ON_states_starts_avg_allsweeps.append(ON_states_starts_avg)
        ON_states_stops_avg_allsweeps.append(ON_states_stops_avg)

        # # plot all ON states in red
        # fig, ax = plt.subplots(2,1, sharex = True)
        # fig.suptitle(f'{day}')
        # ax[0].plot(np.linspace(0, MUA_power_binned_log_smoothed.shape[1], MUA_power_binned_log_smoothed.shape[1])/1000, bimodal_avg_rel_MUA)
        # ax[1].plot(np.linspace(0, curr_LFP.shape[1], curr_LFP.shape[1])/1000, curr_LFP[layer_dict_1[day][sweep][2][0]],:])
        # for ON_start, ON_stop in zip(ON_states_starts_avg, ON_states_stops_avg):
        #     ax[0].axvspan(ON_start, ON_stop, color = 'red', alpha = 0.1)
        #     ax[1].axvspan(ON_start, ON_stop, color = 'red', alpha = 0.1)

    pickle.dump(ON_states_starts_allchans_allsweeps, open('ON_states_starts_allchans_allsweeps', 'wb'))      
    pickle.dump(ON_states_stops_allchans_allsweeps, open('ON_states_stops_allchans_allsweeps', 'wb'))      

    pickle.dump(ON_states_starts_avg_allsweeps, open('ON_states_starts_avg_allsweeps', 'wb'))      
    pickle.dump(ON_states_stops_avg_allsweeps, open('ON_states_stops_avg_allsweeps', 'wb'))      

    os.chdir('..')
    os.chdir('..')
    cl()
    # os.chdir('..')




#%% ---------------------------------------------------------- extract CSD sink/source and spike start/end profiles in all sweeps


highpass_cutoff = 4
 
log_base = 2

exclude_before = 0.1*new_fs
exclude_after = 1*new_fs


days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# use days with a nowhisker stim
for day_ind, day in enumerate(days[0:6]):
# for day_ind, day in enumerate(['160614']):
    os.chdir(day)
    print(day)
    
    if day == '160614':
        highpass_cutoff = 4
    elif day == '160514':
        highpass_cutoff = 4
    else:
        highpass_cutoff = 4

    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    MUA_all_sweeps = pickle.load(open('MUA_all_sweeps','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    
    nchans = LFP_all_sweeps[0].shape[0]

    ON_states_starts_allchans_allsweeps = pickle.load(open('ON_states_starts_allchans_allsweeps', 'rb'))      
    ON_states_stops_allchans_allsweeps = pickle.load(open('ON_states_stops_allchans_allsweeps', 'rb'))    

    ON_states_starts_avg_allsweeps = pickle.load(open('ON_states_starts_avg_allsweeps', 'rb'))      
    ON_states_stops_avg_allsweeps = pickle.load(open('ON_states_stops_avg_allsweeps', 'rb'))    
    
        
    MUA_matrix_start_rel_avgs_allsweeps = []
    MUA_matrix_start_avgs_allsweeps = []
    CSD_matrix_start_avgs_allsweeps = []
    LFP_matrix_start_avgs_allsweeps = []
    MUA_matrix_end_rel_avgs_allsweeps = []
    MUA_matrix_end_avgs_allsweeps = []
    CSD_matrix_end_avgs_allsweeps = []
    LFP_matrix_end_avgs_allsweeps = []
    spikes_start_allsweeps = []
    spikes_end_allsweeps = []

    CSD_sink_timecourse_allsweeps = []
    CSD_source_timecourse_allsweeps = []

    ON_states_starts_avg_interstim = []
    ON_states_stops_avg_interstim = []
    
    for sweep in range(10):
        print(sweep)

        curr_LFP = LFP_all_sweeps[sweep][chanMap_32,:]
        # take out 50Hz
        # curr_LFP = scipy.signal.filtfilt(b_notch, a_notch, curr_LFP)
        nchans = curr_LFP.shape[0]
        
        spikes = spikes_allsweeps[sweep]
        
        # # take log and smooth MUA power across time
        MUA_power_binned_log_smoothed = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_all_sweeps[sweep][chanMap_32,:]**2), (0,80))
        MUA_power_binned_smoothed = scipy.ndimage.gaussian_filter(MUA_all_sweeps[sweep][chanMap_32,:]**2, (0,80))
        # # normalize within each channel, take median value in each channel
        MUA_power_binned_rel = (MUA_all_sweeps[sweep][chanMap_32,:].T**2/np.median(MUA_all_sweeps[sweep][chanMap_32,:]**2, axis = 1)).T
        MUA_power_binned_log_smoothed_rel = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_power_binned_rel), (0,80))
        
        stims = stim_times[sweep]
        
        CSD_matrix = -np.eye(nchans) # 
        for j in range(1, CSD_matrix.shape[0] - 1):
            CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
        CSD_all = - np.dot(CSD_matrix, curr_LFP)
        CSD_all_smoothed = - np.dot(CSD_matrix, scipy.ndimage.gaussian_filter(curr_LFP, (2, 0)))
        CSD_all[0,:] = 0
        CSD_all_smoothed[0,:] = 0
        CSD_all[-1,:] = 0
        CSD_all_smoothed[-1,:] = 0

        


        ON_states_starts_avg = ON_states_starts_avg_allsweeps[sweep]*1000
        ON_states_stops_avg = ON_states_stops_avg_allsweeps[sweep]*1000
         
        # plot a snippet of MUA power of each channel
        # fig, ax = plt.subplots()
        # for i in range(MUA_power_binned_log_smoothed.shape[0]):
        #     ax.plot(MUA_power_binned_log_smoothed[i,0:20000] - i*np.ones_like(MUA_power_binned_log_smoothed[i,0:20000]), linewidth = 0.5)
        
        # for start and stop analysis only use ON/OFF states longer than 200ms
        shorts = []
        for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts_avg, ON_states_stops_avg)):
            if ON_stop - ON_start < 200:
                shorts.append(i)
        ON_states_starts_avg = np.delete(ON_states_starts_avg, shorts)
        ON_states_stops_avg = np.delete(ON_states_stops_avg, shorts)  
        shorts = []
        for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts_avg[1:], ON_states_stops_avg[:-1])):
            if ON_start - ON_stop < 200:
                shorts.append(i)
        ON_states_starts_avg = np.delete(ON_states_starts_avg, shorts)
        ON_states_stops_avg = np.delete(ON_states_stops_avg, shorts)  
        print(len(ON_states_starts_avg))

    
        # only ON periods during interstim period
        stim_ons = []
        for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts_avg, ON_states_stops_avg)):
            for stim in stims:
                if stim - exclude_before < ON_start  < stim + exclude_after or stim - exclude_before < ON_stop  < stim + exclude_after:
                    stim_ons.append(i)
        ON_states_starts_avg = np.delete(ON_states_starts_avg, stim_ons)
        ON_states_stops_avg = np.delete(ON_states_stops_avg, stim_ons)  

        ON_states_starts_avg_interstim.append(ON_states_starts_avg)
        ON_states_stops_avg_interstim.append(ON_states_stops_avg)

        # # plot all ON states in red
        # fig, ax = plt.subplots()
        # fig.suptitle(f'{day}')
        # ax.plot(curr_LFP[layer_dict_1[day][sweep][2][0],:])
        # for ON_start, ON_stop in zip(ON_states_starts_avg, ON_states_stops_avg):
        #     ax.axvspan(ON_start, ON_stop, color = 'red', alpha = 0.1)




        # ON start and end on avg logMUA start times:
        # MUA_matrix_start_rel_avg = []
        # MUA_matrix_start_avg = []
        CSD_matrix_start_avg = []
        LFP_matrix_start_avg = []
        # MUA_matrix_end_rel_avg = []
        # MUA_matrix_end_avg = []
        CSD_matrix_end_avg = []
        # LFP_matrix_end_avg = []
        spikes_start = []
        spikes_end = []

        for ON_start, ON_end in zip((ON_states_starts_avg).astype(int), (ON_states_stops_avg).astype(int)):
            # MUA_matrix_start_rel_avg.append(MUA_power_binned_log_smoothed_rel[:,ON_start - 300:ON_start + 300])
            # MUA_matrix_start_avg.append(MUA_power_binned_log_smoothed[:,ON_start - 300:ON_start + 300])
            CSD_matrix_start_avg.append(CSD_all_smoothed[:,ON_start - 300:ON_start + 300])
            LFP_matrix_start_avg.append(curr_LFP[:,ON_start - 300:ON_start + 300])
            # MUA_matrix_end_rel_avg.append(MUA_power_binned_log_smoothed_rel[:,ON_end - 300:ON_end + 300])
            # MUA_matrix_end_avg.append(MUA_power_binned_log_smoothed[:,ON_end - 300:ON_end + 300])
            CSD_matrix_end_avg.append(CSD_all_smoothed[:,ON_end - 300:ON_end + 300])
            # LFP_matrix_end_avg.append(curr_LFP[:,ON_end - 300:ON_end + 300])
            
            curr_spikes_start = []
            curr_spikes_end = []
            for chan in range(32):
                # curr_spikes = list(spikes.values())[np.argwhere(chanMap_32 == chan)[0][0]]
                curr_spikes = list(spikes.values())[chan]
                on_spikes_start = curr_spikes[np.searchsorted(curr_spikes, ON_start - 300) : np.searchsorted(curr_spikes, ON_start + 300)] - ON_start #spikes during that ON state, with ON_start = time 0
                on_spikes_end = curr_spikes[np.searchsorted(curr_spikes, ON_end - 300) : np.searchsorted(curr_spikes, ON_end + 300)] - ON_end #spikes during that ON state, with ON_start = time 0

                curr_spikes_start.append(np.histogram(on_spikes_start, bins = np.linspace(-300, 299, 600))[0])    
                curr_spikes_end.append(np.histogram(on_spikes_end, bins = np.linspace(-300, 299, 600))[0])    

            spikes_start.append(curr_spikes_start)
            spikes_end.append(curr_spikes_end)

            
        # MUA_matrix_start_rel_avg = np.asarray(MUA_matrix_start_rel_avg)
        # MUA_matrix_start_avg = np.asarray(MUA_matrix_start_avg)
        CSD_matrix_start_avg = np.asarray(CSD_matrix_start_avg)
        # LFP_matrix_start_avg = np.asarray(LFP_matrix_start_avg)
        # MUA_matrix_end_rel_avg = np.asarray(MUA_matrix_end_rel_avg)
        # MUA_matrix_end_avg = np.asarray(MUA_matrix_end_avg)
        CSD_matrix_end_avg = np.asarray(CSD_matrix_end_avg)
        # LFP_matrix_end_avg = np.asarray(LFP_matrix_end_avg)
        spikes_start = np.asarray(spikes_start)[:,chanMap_32,:]
        spikes_end = np.asarray(spikes_end)[:,chanMap_32,:]

        # MUA_matrix_start_rel_avgs_allsweeps.append(MUA_matrix_start_rel_avg)
        # MUA_matrix_start_avgs_allsweeps.append(MUA_matrix_start_avg)
        CSD_matrix_start_avgs_allsweeps.append(CSD_matrix_start_avg)
        # LFP_matrix_start_avgs_allsweeps.append(LFP_matrix_start_avg)
        # MUA_matrix_end_rel_avgs_allsweeps.append(MUA_matrix_end_rel_avg)
        # MUA_matrix_end_avgs_allsweeps.append(MUA_matrix_end_avg)
        CSD_matrix_end_avgs_allsweeps.append(CSD_matrix_end_avg)
        # LFP_matrix_end_avgs_allsweeps.append(LFP_matrix_end_avg)
        spikes_start_allsweeps.append(spikes_start)
        spikes_end_allsweeps.append(spikes_end)
    
        # CSD sink starts across channels (ON start)
        CSD_sink_timecourse =[]
        for SO_ind, SO in enumerate(list(range(CSD_matrix_start_avg.shape[0]))):
            CSD_for_gradient = copy.deepcopy(scipy.ndimage.gaussian_filter(CSD_matrix_start_avg[SO_ind,:,200:], (0,10))) # start 100ms before average MUA was detected
            CSD_for_gradient[CSD_for_gradient > 0] = 0 #has to be a sink so negative
            curr_CSD_crossing = []
            for chan in range(nchans):   
                curr_CSD_crossing.append(np.argwhere(CSD_for_gradient[chan,:] <= np.min(CSD_for_gradient[chan,:])*0.25)[0][0]-100) 
            CSD_sink_timecourse.append(np.asarray(curr_CSD_crossing))
        CSD_sink_timecourse = np.asarray(CSD_sink_timecourse)
        CSD_sink_timecourse_allsweeps.append(CSD_sink_timecourse)

        # CSD source starts across channels (ON end)
        CSD_source_timecourse = []
        for SO_ind, SO in enumerate(list(range(CSD_matrix_end_avg.shape[0]))):
            CSD_for_gradient = copy.deepcopy(scipy.ndimage.gaussian_filter(CSD_matrix_end_avg[SO_ind,:,:500], (0,10))) # start 300ms before end of MUA to 150ms after
            CSD_for_gradient[CSD_for_gradient < 0] = 0 #has to be a source so positive
            curr_CSD_crossing = []
            for chan in range(nchans):   
                curr_CSD_crossing.append(np.argwhere(CSD_for_gradient[chan,:] >= np.max(CSD_for_gradient[chan,:])*0.75)[0][0] - 300) 
            CSD_source_timecourse.append(np.asarray(curr_CSD_crossing))
        CSD_source_timecourse = np.asarray(CSD_source_timecourse)
        CSD_source_timecourse_allsweeps.append(CSD_source_timecourse)
        
    pickle.dump(spikes_start_allsweeps, open('spikes_start_allsweeps.pkl', 'wb'))
    pickle.dump(spikes_end_allsweeps, open('spikes_end_allsweeps.pkl', 'wb'))

    pickle.dump(CSD_sink_timecourse_allsweeps, open('CSD_sink_timecourse_allsweeps.pkl', 'wb'))
    pickle.dump(CSD_source_timecourse_allsweeps, open('CSD_source_timecourse_allsweeps.pkl', 'wb'))

    pickle.dump(ON_states_starts_avg_interstim, open('ON_states_starts_avg_interstim.pkl', 'wb'))
    pickle.dump(ON_states_stops_avg_interstim, open('ON_states_stops_avg_interstim.pkl', 'wb'))

    os.chdir('..')


#%% -------------------------------------------------------------------- ON start and end profiles PCA

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# use days with a nowhisker stim
# for day_ind, day in enumerate(days[:6]):
for day_ind, day in enumerate(['160615']):
    os.chdir(day)
    print(day)

    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    MUA_all_sweeps = pickle.load(open('MUA_all_sweeps','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))

    nchans = LFP_all_sweeps[0].shape[0]

    # cortical channels 
    if day == '160614':
        cortical_chans = np.arange(0,25)
    elif day == '160615':
        cortical_chans = np.arange(6,28)
    elif day == '160622':
        cortical_chans = np.arange(3,29)
    elif day == '160728':
        cortical_chans = np.arange(0,25)
    elif day == '160729':
        cortical_chans = np.arange(0,27)
    elif day == '160810':
        cortical_chans = np.arange(0,21)
    
    #channels for PCA CSD sorting
    if day == '160614':
        PCA_CSD_chans = np.arange(7,25) #np.concatenate(layer_dict[day][0][2:])
    elif day == '160615':
        PCA_CSD_chans = np.arange(12,25) # np.concatenate(layer_dict[day][0][2:])
    elif day == '160622':
        PCA_CSD_chans = np.arange(10,28) #np.concatenate(layer_dict[day][0][2:])
    elif day == '160728':
        PCA_CSD_chans = np.arange(8,24) # np.concatenate(layer_dict[day][0][2:])
    elif day == '160729':
        PCA_CSD_chans = np.arange(8,24) # np.concatenate(layer_dict[day][0][2:])
    elif day == '160810':
        PCA_CSD_chans = np.arange(6,20) # np.concatenate(layer_dict[day][0][2:])

    # # channels of deep and superficial sink
    if day == '160614':
        superf_sink_chans = [10,11,12,13,14]
        deep_sink_chans = [19,20,21,22,23]
    elif day == '160615':
        superf_sink_chans = [11,12,13]
        deep_sink_chans = [19,20,21,22]
    elif day == '160622':
        superf_sink_chans = [11,12,13]
        deep_sink_chans = [19,20,21,22]
    elif day == '160728':
        superf_sink_chans = [10,11,12,13]
        deep_sink_chans = [17,18,19,20,21]
    elif day == '160729':
        superf_sink_chans = [8,9,10]
        deep_sink_chans = [14,15,16,17]
    elif day == '160810':
        superf_sink_chans = [8,9,10]
        deep_sink_chans = [14,15,16,17]


    ON_states_starts_avg_allsweeps = pickle.load(open('ON_states_starts_avg_interstim.pkl', 'rb'))      
    ON_states_stops_avg_allsweeps = pickle.load(open('ON_states_stops_avg_interstim.pkl', 'rb'))    
    
    spikes_start_allsweeps = pickle.load(open('spikes_start_allsweeps.pkl', 'rb'))
    spikes_start_before = np.concatenate(spikes_start_allsweeps[0:4])
    spikes_start_after = np.concatenate(spikes_start_allsweeps[4:])

    spikes_end_allsweeps = pickle.load(open('spikes_end_allsweeps.pkl', 'rb'))
    spikes_end_before = np.concatenate(spikes_end_allsweeps[0:4])
    spikes_end_after = np.concatenate(spikes_end_allsweeps[4:])

    CSD_sink_timecourse_allsweeps = pickle.load(open('CSD_sink_timecourse_allsweeps.pkl', 'rb'))
    CSD_sink_timecourse_before = np.concatenate(CSD_sink_timecourse_allsweeps[0:4])
    CSD_sink_timecourse_zeroed_before = CSD_sink_timecourse_before.T - np.median(CSD_sink_timecourse_before[:, layer_dict[day][0][3]], axis = 1)
    CSD_sink_timecourse_after = np.concatenate(CSD_sink_timecourse_allsweeps[4:])
    CSD_sink_timecourse_zeroed_after = CSD_sink_timecourse_after.T - np.median(CSD_sink_timecourse_after[:, layer_dict[day][0][3]], axis = 1)

    CSD_source_timecourse_allsweeps = pickle.load(open('CSD_source_timecourse_allsweeps.pkl', 'rb'))
    CSD_source_timecourse_before = np.concatenate(CSD_source_timecourse_allsweeps[0:4])
    CSD_source_timecourse_zeroed_before = CSD_source_timecourse_before.T - np.median(CSD_source_timecourse_before[:, layer_dict[day][0][3]], axis = 1)
    CSD_source_timecourse_after = np.concatenate(CSD_source_timecourse_allsweeps[4:])
    CSD_source_timecourse_zeroed_after = CSD_source_timecourse_after.T - np.median(CSD_source_timecourse_after[:, layer_dict[day][0][3]], axis = 1)



    # -------------------------------------------------- distribution of sinks deep vs superficial before vs after pairing
    
    CSD_sink_timecourse_L4_vs_L5_before = np.median(CSD_sink_timecourse_before[:, layer_dict[day][0][3]], axis = 1) - np.median(CSD_sink_timecourse_before[:, layer_dict[day][0][2]], axis = 1)
    CSD_sink_timecourse_L4_vs_L5_after = np.median(CSD_sink_timecourse_after[:, layer_dict[day][0][3]], axis = 1) - np.median(CSD_sink_timecourse_after[:, layer_dict[day][0][2]], axis = 1)
    # CSD_sink_timecourse_L4_vs_L5_before = np.median(CSD_sink_timecourse_before[:, deep_sink_chans], axis = 1) - np.median(CSD_sink_timecourse_before[:, superf_sink_chans], axis = 1)
    # CSD_sink_timecourse_L4_vs_L5_after = np.median(CSD_sink_timecourse_after[:, deep_sink_chans], axis = 1) - np.median(CSD_sink_timecourse_after[:, superf_sink_chans], axis = 1)
    np.save('CSD_sink_timecourse_L4_vs_L5_before.npy', CSD_sink_timecourse_L4_vs_L5_before)
    np.save('CSD_sink_timecourse_L4_vs_L5_after.npy', CSD_sink_timecourse_L4_vs_L5_after)
    np.save('CSD_sink_timecourse_L4_vs_L5.npy', np.hstack((CSD_sink_timecourse_L4_vs_L5_before, CSD_sink_timecourse_L4_vs_L5_after)))
    # bins = 50
    # fig, ax = plt.subplots()
    # fig.suptitle(f'{day}')
    # ax.hist(CSD_sink_timecourse_L4_vs_L5_before, bins = bins, density = True, color = 'black', alpha = 0.5)
    
    # ax.hist(CSD_sink_timecourse_L4_vs_L5_after, bins = bins, density = True, color = 'red', alpha = 0.5)
    # ax.set_xlim([-200,200])

    CSD_source_timecourse_L4_vs_L5_before = np.median(CSD_source_timecourse_before[:, layer_dict[day][0][3]], axis = 1) - np.median(CSD_source_timecourse_before[:, layer_dict[day][5][2]], axis = 1)
    CSD_source_timecourse_L4_vs_L5_after = np.median(CSD_source_timecourse_after[:, layer_dict[day][0][3]], axis = 1) - np.median(CSD_source_timecourse_after[:, layer_dict[day][5][2]], axis = 1)
    # CSD_source_timecourse_L4_vs_L5_before = np.median(CSD_source_timecourse_before[:, deep_sink_chans], axis = 1) - np.median(CSD_source_timecourse_before[:, superf_sink_chans], axis = 1)
    # CSD_source_timecourse_L4_vs_L5_after = np.median(CSD_source_timecourse_after[:, deep_sink_chans], axis = 1) - np.median(CSD_source_timecourse_after[:, superf_sink_chans], axis = 1)
    np.save('CSD_source_timecourse_L4_vs_L5_before.npy', CSD_source_timecourse_L4_vs_L5_before)
    np.save('CSD_source_timecourse_L4_vs_L5_after.npy', CSD_source_timecourse_L4_vs_L5_after)
    np.save('CSD_source_timecourse_L4_vs_L5.npy', np.hstack((CSD_source_timecourse_L4_vs_L5_before, CSD_source_timecourse_L4_vs_L5_after)))



    # ------------------------------------------------------------ average spike start before and after pairing
    # curr_layers_1_avg = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
    # fig, ax = plt.subplots()
    # for layer in [0,1,2,3]:
    #     ax.plot(scipy.ndimage.gaussian_filter(np.mean(np.mean(spikes_start_before, axis = 0)[curr_layers_1_avg[layer],:], axis = 0)*1000, (5)).T)
    # ax.set_ylim([0,80])
    # fig, ax = plt.subplots()
    # for layer in [0,1,2,3]:
    #     ax.plot(scipy.ndimage.gaussian_filter(np.mean(np.mean(spikes_start_after, axis = 0)[curr_layers_1_avg[layer],:], axis = 0)*1000, (5)).T)
    # ax.set_ylim([0,80])
    # ax.set_xlim([250,500])

    # fig, ax = plt.subplots(figsize = (1.5,12))
    # fig.suptitle(f'{day}')
    # for chan in range(32):
    #     ax.plot(scipy.ndimage.gaussian_filter(np.mean(spikes_start_before[:,chan,:], axis = 0)*1000 - 50*chan, 6), color = 'k')
    #     ax.plot(scipy.ndimage.gaussian_filter(np.mean(spikes_start_after[:,chan,:], axis = 0)*1000 - 50*chan, 6), color = 'r')


    # ------------------------------------------------------------ average spike end before and after pairing
    # curr_layers_1_avg = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
    # fig, ax = plt.subplots()
    # for layer in [0,1,2,3]:
    #     ax.plot(scipy.ndimage.gaussian_filter(np.mean(np.mean(spikes_end_before, axis = 0)[curr_layers_1_avg[layer],:], axis = 0)*1000, (5)).T)
    # ax.set_ylim([0,80])
    # fig, ax = plt.subplots()
    # for layer in [0,1,2,3]:
    #     ax.plot(scipy.ndimage.gaussian_filter(np.mean(np.mean(spikes_end_after, axis = 0)[curr_layers_1_avg[layer],:], axis = 0)*1000, (5)).T)
    # ax.set_ylim([0,80])
    # ax.set_xlim([250,500])



    
    # -------------------------------------------------------------------------- align CSD and spikes maps to L5 sink start
    
    CSD_matrix_start_avg_aligned_to_CSD_start_allsweeps = []
    CSD_matrix_end_avg_aligned_to_CSD_end_allsweeps = []
    spikes_start_avg_aligned_to_CSD_start_allsweeps = []
    spikes_end_avg_aligned_to_CSD_end_allsweeps = []

    for sweep in range(10):
        # print(sweep)
        curr_LFP = LFP_all_sweeps[sweep][chanMap_32,:]
        spikes = spikes_allsweeps[sweep]
        # take out 50Hz
        # curr_LFP = scipy.signal.filtfilt(b_notch, a_notch, curr_LFP)
        nchans = curr_LFP.shape[0]
                
        CSD_matrix = -np.eye(nchans) # 
        for j in range(1, CSD_matrix.shape[0] - 1):
            CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
        CSD_all_smoothed = - np.dot(CSD_matrix, scipy.ndimage.gaussian_filter(curr_LFP, (2, 0)))
        CSD_all_smoothed[0,:] = 0
        CSD_all_smoothed[-1,:] = 0
        
        curr_ON_states_starts = ON_states_starts_avg_allsweeps[sweep] 
        curr_ON_states_stops = ON_states_stops_avg_allsweeps[sweep] 

        CSD_matrix_start_avg_aligned_to_CSD_start = []
        CSD_matrix_end_avg_aligned_to_CSD_end = []
        spikes_start_aligned_to_CSD_start = []
        spikes_end_aligned_to_CSD_end = []
        for SO_ind, (ON_start, ON_end) in enumerate(list(zip(curr_ON_states_starts, curr_ON_states_stops))):
            curr_ON_start = int(ON_start) + int(np.median(CSD_sink_timecourse_allsweeps[sweep][SO_ind, layer_dict[day][0][3]]))
            curr_ON_end = int(ON_end) + int(np.median(CSD_source_timecourse_allsweeps[sweep][SO_ind, layer_dict[day][0][3]]))
            # curr_ON_start = int(ON_start)
            # curr_ON_end = int(ON_end)
            CSD_matrix_start_avg_aligned_to_CSD_start.append(CSD_all_smoothed[:,int(curr_ON_start - 300):int(curr_ON_start + 300)])
            CSD_matrix_end_avg_aligned_to_CSD_end.append(CSD_all_smoothed[:,int(curr_ON_end - 300):int(curr_ON_end + 300)])
            
            curr_spikes_start = []
            curr_spikes_end = []
            for chan in range(32):
                # curr_spikes = list(spikes.values())[np.argwhere(chanMap_32 == chan)[0][0]]
                curr_spikes = list(spikes.values())[chan]
                on_spikes_start = curr_spikes[np.searchsorted(curr_spikes, curr_ON_start - 300) : np.searchsorted(curr_spikes, curr_ON_start + 300)] - curr_ON_start #spikes during that ON state, with ON_start = time 0
                on_spikes_end = curr_spikes[np.searchsorted(curr_spikes, curr_ON_end - 300) : np.searchsorted(curr_spikes, curr_ON_end + 300)] - curr_ON_end #spikes during that ON state, with ON_start = time 0
                curr_spikes_start.append(np.histogram(on_spikes_start, bins = np.linspace(-300, 299, 600))[0])    
                curr_spikes_end.append(np.histogram(on_spikes_end, bins = np.linspace(-300, 299, 600))[0])    
            spikes_start_aligned_to_CSD_start.append(curr_spikes_start)
            spikes_end_aligned_to_CSD_end.append(curr_spikes_end)
            
        CSD_matrix_start_avg_aligned_to_CSD_start = np.asarray(CSD_matrix_start_avg_aligned_to_CSD_start)
        CSD_matrix_end_avg_aligned_to_CSD_end = np.asarray(CSD_matrix_end_avg_aligned_to_CSD_end)
        CSD_matrix_start_avg_aligned_to_CSD_start_allsweeps.append(CSD_matrix_start_avg_aligned_to_CSD_start)
        CSD_matrix_end_avg_aligned_to_CSD_end_allsweeps.append(CSD_matrix_end_avg_aligned_to_CSD_end)
        spikes_start_aligned_to_CSD_start = np.asarray(spikes_start_aligned_to_CSD_start)[:,chanMap_32,:]
        spikes_end_aligned_to_CSD_end = np.asarray(spikes_end_aligned_to_CSD_end)[:,chanMap_32,:]
        spikes_start_avg_aligned_to_CSD_start_allsweeps.append(spikes_start_aligned_to_CSD_start)
        spikes_end_avg_aligned_to_CSD_end_allsweeps.append(spikes_end_aligned_to_CSD_end)



    # -------------------------------------------------------------- CSD and spikes starts PCA whole experiment

    CSD_sink_timecourse_for_PCA = np.hstack((CSD_sink_timecourse_zeroed_before, CSD_sink_timecourse_zeroed_after)).T[:,PCA_CSD_chans]
    # CSD_sink_timecourse_for_PCA = CSD_sink_timecourse_zeroed_before.T[:,PCA_CSD_chans]
    CSD_sink_timecourse_for_PCA[np.abs(CSD_sink_timecourse_for_PCA) > 200] = 0
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD starts only chans for sorting')
    # ax.imshow(CSD_sink_timecourse_for_PCA.T, cmap = 'bwr', aspect = 'auto', vmin = -300, vmax = 300)
    princ_components_start, eigenvectors, eigenvalues, var_ratio, scaled = PCA_normed(CSD_sink_timecourse_for_PCA)
    
    # # plot ordered by first PC:
    # fig, ax = plt.subplots(figsize = (12,2.5))
    # # ax.imshow(np.hstack((CSD_sink_timecourse_zeroed_before, CSD_sink_timecourse_zeroed_after)).T[:,cortical_chans][np.argsort(princ_components_start[:,0]),:].T, cmap = 'bwr', aspect = 'auto', vmin = -300, vmax = 300)
    # image = ax.imshow(CSD_sink_timecourse_for_PCA[np.argsort(princ_components_start[:,0]),:].T, cmap = 'bwr', aspect = 'auto', vmin = -200, vmax = 200)
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')
    # ax.tick_params(axis = 'x', labelsize = 18)
    # ax.set_xlabel('ON-state start (#)', size = 18)
    # ax.set_yticks([12,7,2])
    # ax.set_yticklabels(['-1', '-0.75', '-0.5'])
    # ax.set_ylabel('depth (mm)', size = 18)
    # ax.tick_params(axis = 'y', labelsize = 18)
    # cbar = fig.colorbar(image, pad = 0.02)
    # cbar.ax.tick_params(axis = 'y', labelsize = 18)
    # cbar.ax.set_ylabel('CSD sink onset (ms)', size = 18)
    # plt.tight_layout()
    # plt.savefig('CSD sink PCA sorted.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD sink PCA sorted.pdf', dpi = 1000, format = 'pdf')
    
    
    # --------------------------------------------------------------------------------- divide into segments and plot starts
    tier_nr = 5
    tiers = np.array_split(np.argsort(princ_components_start[:,0]),tier_nr)
    
    fig = plt.figure(figsize=(12, 4))
    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                      nrows_ncols=(1,tier_nr),
                      axes_pad=0.15,
                      share_all=True,
                      cbar_location="right",
                      cbar_mode="single",
                      cbar_size="7%",
                      cbar_pad=0.15,
                      )
    # Add data to image grid
    vmax = np.max([np.max(np.mean(np.concatenate(CSD_matrix_start_avg_aligned_to_CSD_start_allsweeps)[tiers[tier],:,:], axis = 0)[cortical_chans,:]) for tier in range(tier_nr)])
    vmin = np.min([np.min(np.mean(np.concatenate(CSD_matrix_start_avg_aligned_to_CSD_start_allsweeps)[tiers[tier],:,:], axis = 0)[cortical_chans,:]) for tier in range(tier_nr)])
    for tier, ax1 in enumerate(list(grid)):
        image = ax1.imshow(np.mean(np.concatenate(CSD_matrix_start_avg_aligned_to_CSD_start_allsweeps)[tiers[tier],:,:], axis = 0)[cortical_chans,:], aspect = 25, cmap = 'jet', interpolation = 'bicubic')
        ax1.set_xticks([100,300,500])
        ax1.set_xticklabels(['-0.2','0', '0.2'], size = 14)
        # ax1.set_yticks(np.arange(cortical_chans[0], cortical_chans[-1] + 1, (cortical_chans[-1] - cortical_chans[0])/3))
        # if tier == 0:
        ax1.set_yticks(np.arange(0, (cortical_chans[-1] - cortical_chans[0]) + 1, 6)[[0,1,2,3]])
        ax1.set_yticklabels(['0', '-0.3', '-0.6', '-1.2'], size = 16)
        # else:
        #     ax1.set_yticks([])
        ax1.set_xlim([50,550])
    cbar = ax1.cax.colorbar(image)
    cbar.ax.tick_params(axis = 'y', labelsize = 18)
    cbar.ax.set_ylabel('CSD (mV/mm2)', size = 18)
    # plt.savefig('CSD start heatmaps PCA sorted ALL SWEEPS.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD start heatmaps PCA sorted ALL SWEEPS.pdf', dpi = 1000, format = 'pdf')

    

    fig = plt.figure(figsize=(12, 4))
    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                      nrows_ncols=(1,tier_nr),
                      axes_pad=0.15,
                      share_all=True,
                      cbar_location="right",
                      cbar_mode="single",
                      cbar_size="7%",
                      cbar_pad=0.15,
                      )
    # Add data to image grid
    # vmax = np.max([np.max(scipy.ndimage.gaussian_filter(np.mean(np.concatenate(spikes_start_avg_aligned_to_CSD_start_allsweeps)[tiers[tier],:,:], axis = 0)[cortical_chans,:], (0.5,8))) for tier in range(tier_nr)])*1000*0.8
    vmin = 0
    for tier, ax1 in enumerate(list(grid)):
        if tier == 0 and day == '160615':
            vmax = np.max([np.max(scipy.ndimage.gaussian_filter(np.mean(np.concatenate(spikes_start_avg_aligned_to_CSD_start_allsweeps)[tiers[tier],:,:], axis = 0)[cortical_chans,:], (0.5,8))) for tier in range(tier_nr)])*1000*0.6
        else:
            vmax = np.max([np.max(scipy.ndimage.gaussian_filter(np.mean(np.concatenate(spikes_start_avg_aligned_to_CSD_start_allsweeps)[tiers[tier],:,:], axis = 0)[cortical_chans,:], (0.5,8))) for tier in range(tier_nr)])*1000*0.55
        # image = ax1.imshow(scipy.ndimage.gaussian_filter(np.mean(np.concatenate(spikes_start_allsweeps)[tiers[tier],:,:], axis = 0)[cortical_chans,:], (0.5,8))*1000, aspect = 25, interpolation = 'bicubic', cmap = 'jet', vmin = vmin, vmax = vmax)
        image = ax1.imshow(scipy.ndimage.gaussian_filter(np.mean(np.concatenate(spikes_start_avg_aligned_to_CSD_start_allsweeps)[tiers[tier],:,:], axis = 0)[cortical_chans,:], (0.5,8))*1000, aspect = 25, interpolation = 'bicubic', cmap = 'jet', vmin = vmin, vmax = vmax)
        ax1.set_xticks([100,300,500])
        ax1.set_xticklabels(['-0.2','0', '0.2'], size = 14)
        # ax1.set_yticks(np.arange(cortical_chans[0], cortical_chans[-1] + 1, (cortical_chans[-1] - cortical_chans[0])/3))
        # if tier == 0:
        ax1.set_yticks(np.arange(0, (cortical_chans[-1] - cortical_chans[0]) + 1, 6)[[0,1,2,3]])
        ax1.set_yticklabels(['0', '-0.3', '-0.6', '-1.2'], size = 16)
        # else:
        #     ax1.set_yticks([])
        ax1.set_xlim([50,550])
    cbar = ax1.cax.colorbar(image)
    cbar.ax.tick_params(axis = 'y', labelsize = 18)
    cbar.ax.set_ylabel('MUA (Hz)', size = 18)
    # plt.savefig('MUA start heatmaps PCA sorted ALL SWEEPS.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('MUA start heatmaps PCA sorted ALL SWEEPS.pdf', dpi = 1000, format = 'pdf')

    # # color_layers = ['blue', 'green', 'orange', 'red']
    # # fig, ax = plt.subplots(1,tier_nr)
    # # fig.suptitle('spikes starts PCA ordered')
    # # curr_layers = layer_dict[day][0][1:]
    # # for tier, ax1 in enumerate(list(ax)):
    # #     for layer, color in zip([0,1,2,3], color_layers):
    # #         ax1.plot(scipy.ndimage.gaussian_filter(np.mean(np.mean(np.concatenate(spikes_start_allsweeps)[tiers[tier],:,:], axis = 0)[curr_layers[layer],:], axis = 0)*1000, (5)).T, color = color)


    # # -------------------------------------------------------------- CSD and spikes starts PCA before pairing
    # CSD_sink_timecourse_for_PCA = CSD_sink_timecourse_zeroed_before.T[:,PCA_CSD_chans]
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD starts only chans for sorting')
    # ax.imshow(CSD_sink_timecourse_for_PCA.T, cmap = 'bwr', aspect = 2, vmin = -300, vmax = 300)

    # princ_components, eigenvectors, eigenvalues, var_ratio, scaled = PCA_normed(CSD_sink_timecourse_for_PCA)
    # # plot ordered by first PC:
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD start ordered')
    # ax.imshow(CSD_sink_timecourse_for_PCA[np.argsort(princ_components[:,0]),:].T, cmap = 'bwr', aspect = 4, vmin = -300, vmax = 300)
    # # plt.savefig('spikes starts PCA sorted.jpg', dpi = 1000, format = 'jpg')
    # # plt.savefig('spikes starts PCA sorted.pdf', dpi = 1000, format = 'pdf')
    
    # #divide into segments and plot starts
    # tier_nr = 4
    # tiers = np.array_split(np.argsort(princ_components[:,0]),tier_nr)
        
    # fig, ax = plt.subplots(1,tier_nr)
    # fig.suptitle('spikes PCA ordered')
    # for tier, ax1 in enumerate(list(ax)):
    #     for layer in [0,1,2,3]:
    #         ax1.plot(scipy.ndimage.gaussian_filter(np.mean(np.mean(spikes_start_before[tiers[tier],:,:], axis = 0)[curr_layers_1_avg[layer],:], axis = 0)*1000, (5)).T)

    
    # # -------------------------------------------------------------- CSD and spikes starts PCA after pairing
    # CSD_sink_timecourse_for_PCA = CSD_sink_timecourse_zeroed_after.T[:,PCA_CSD_chans]
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD starts only chans for sorting')
    # ax.imshow(CSD_sink_timecourse_for_PCA.T, cmap = 'bwr', aspect = 2, vmin = -300, vmax = 300)

    # princ_components, eigenvectors, eigenvalues, var_ratio, scaled = PCA_normed(CSD_sink_timecourse_for_PCA)
    # # plot ordered by first PC:
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD start ordered')
    # ax.imshow(CSD_sink_timecourse_for_PCA[np.argsort(princ_components[:,0]),:].T, cmap = 'bwr', aspect = 4, vmin = -300, vmax = 300)
    # # plt.savefig('CSD starts PCA sorted.jpg', dpi = 1000, format = 'jpg')
    # # plt.savefig('CSD starts PCA sorted.pdf', dpi = 1000, format = 'pdf')
    
    # #divide into segments and plot start heatmaps
    # tier_nr = 4
    # tiers = np.array_split(np.argsort(princ_components[:,0]),tier_nr)
        
    # fig, ax = plt.subplots(1,tier_nr)
    # fig.suptitle('spikes PCA ordered')
    # for tier, ax1 in enumerate(list(ax)):
    #     for layer in [0,1,2,3]:
    #         ax1.plot(scipy.ndimage.gaussian_filter(np.mean(np.mean(spikes_start_after[tiers[tier],:,:], axis = 0)[curr_layers_1_avg[layer],:], axis = 0)*1000, (5)).T)








    # -------------------------------------------------------------- CSD and spikes ends PCA whole experiment
    CSD_source_timecourse_for_PCA = np.hstack((CSD_source_timecourse_zeroed_before, CSD_source_timecourse_zeroed_after)).T[:,PCA_CSD_chans]
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD ends only chans for sorting')
    # ax.imshow(CSD_source_timecourse_for_PCA.T, cmap = 'bwr', aspect = 'auto', vmin = -300, vmax = 300)
    princ_components_end, eigenvectors, eigenvalues, var_ratio, scaled = PCA_normed(CSD_source_timecourse_for_PCA)
    
    
    # fig, ax = plt.subplots(figsize = (12,2.5))
    # # ax.imshow(np.hstack((CSD_source_timecourse_zeroed_before, CSD_source_timecourse_zeroed_after)).T[:,cortical_chans][np.argsort(princ_components_end[:,0]),:].T, cmap = 'bwr', aspect = 'auto', vmin = -300, vmax = 300)
    # image = ax.imshow(CSD_source_timecourse_for_PCA[np.argsort(princ_components_end[:,0]),:].T, cmap = 'bwr', aspect = 'auto', vmin = -300, vmax = 300)
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')
    # ax.tick_params(axis = 'x', labelsize = 18)
    # ax.set_xlabel('ON-state end (#)', size = 18)
    # ax.set_yticks([12,7,2])
    # ax.set_yticklabels(['-1', '-0.75', '-0.5'])
    # ax.set_ylabel('depth (mm)', size = 18)
    # ax.tick_params(axis = 'y', labelsize = 18)
    # cbar = fig.colorbar(image, pad = 0.02)
    # cbar.ax.tick_params(axis = 'y', labelsize = 18)
    # cbar.ax.set_ylabel('CSD source onset (ms)', size = 18)
    # plt.tight_layout()
    # plt.savefig('CSD source PCA sorted.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD source PCA sorted.pdf', dpi = 1000, format = 'pdf')
    

    # # -------------------------------------------------------------------------- divide into segments and plot ends
    # tier_nr = 5
    # tiers = np.array_split(np.argsort(princ_components_end[:,0]),tier_nr)
    
    # fig = plt.figure(figsize=(12, 4))
    # grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
    #                   nrows_ncols=(1,tier_nr),
    #                   axes_pad=0.15,
    #                   share_all=True,
    #                   cbar_location="right",
    #                   cbar_mode="single",
    #                   cbar_size="7%",
    #                   cbar_pad=0.15,
    #                   )
    # # Add data to image grid
    # vmax = np.max([np.max(np.mean(np.concatenate(CSD_matrix_start_avg_aligned_to_CSD_start_allsweeps)[tiers[tier],:,:], axis = 0)[cortical_chans,:]) for tier in range(tier_nr)])
    # vmin = np.min([np.min(np.mean(np.concatenate(CSD_matrix_start_avg_aligned_to_CSD_start_allsweeps)[tiers[tier],:,:], axis = 0)[cortical_chans,:]) for tier in range(tier_nr)])
    # for tier, ax1 in enumerate(list(grid)):
    #     image = ax1.imshow(np.mean(np.concatenate(CSD_matrix_end_avg_aligned_to_CSD_end_allsweeps)[tiers[tier],:,:], axis = 0)[cortical_chans,:], aspect = 25, cmap = 'jet', interpolation = 'bicubic')
    #     ax1.set_xticks([100,300,500])
    #     ax1.set_xticklabels(['-0.2','0','0.2'], size = 14)
    #     # ax1.set_yticks(np.arange(cortical_chans[0], cortical_chans[-1] + 1, (cortical_chans[-1] - cortical_chans[0])/3))
    #     # if tier == 0:
    #     ax1.set_yticks(np.arange(0, (cortical_chans[-1] - cortical_chans[0]) + 1, 6)[[0,1,2,3]])
    #     ax1.set_yticklabels(['0', '-0.3', '-0.6', '-1.2'], size = 16)
    #     # else:
    #     #     ax1.set_yticks([])
    #     ax1.set_xlim([50,550])
    # cbar = ax1.cax.colorbar(image)
    # cbar.ax.tick_params(axis = 'y', labelsize = 18)
    # cbar.ax.set_ylabel('CSD (mV/mm2)', size = 18)
    # plt.savefig('CSD end heatmaps PCA sorted ALL SWEEPS.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD end heatmaps PCA sorted ALL SWEEPS.pdf', dpi = 1000, format = 'pdf')


    # fig = plt.figure(figsize=(12, 4))
    # grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
    #                   nrows_ncols=(1,tier_nr),
    #                   axes_pad=0.15,
    #                   share_all=True,
    #                   cbar_location="right",
    #                   cbar_mode="single",
    #                   cbar_size="7%",
    #                   cbar_pad=0.15,
    #                   )
    # # Add data to image grid
    # vmax = np.max([np.max(scipy.ndimage.gaussian_filter(np.mean(np.concatenate(spikes_end_avg_aligned_to_CSD_end_allsweeps)[tiers[tier],:,:], axis = 0)[cortical_chans,:], (0.5,8))) for tier in range(tier_nr)])*1000
    # vmin = 0
    # for tier, ax1 in enumerate(list(grid)):
    #     image = ax1.imshow(scipy.ndimage.gaussian_filter(np.mean(np.concatenate(spikes_end_avg_aligned_to_CSD_end_allsweeps)[tiers[tier],:,:], axis = 0)[cortical_chans,:], (1,8))*1000, aspect = 25, cmap = 'jet', vmin = 0, interpolation = 'bicubic')
    #     ax1.set_xticks([100,300,500])
    #     ax1.set_xticklabels(['-0.2','0','0.2'], size = 14)
    #     # ax1.set_yticks(np.arange(cortical_chans[0], cortical_chans[-1] + 1, (cortical_chans[-1] - cortical_chans[0])/3))
    #     # if tier == 0:
    #     ax1.set_yticks(np.arange(0, (cortical_chans[-1] - cortical_chans[0]) + 1, 6)[[0,1,2,3]])
    #     ax1.set_yticklabels(['0', '-0.3', '-0.6', '-1.2'], size = 16)
    #     # else:
    #     #     ax1.set_yticks([])
    #     ax1.set_xlim([50,550])
    # cbar = ax1.cax.colorbar(image)
    # cbar.ax.tick_params(axis = 'y', labelsize = 18)
    # cbar.ax.set_ylabel('MUA (Hz)', size = 18)
    # plt.savefig('MUA end heatmaps PCA sorted ALL SWEEPS.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('MUA end heatmaps PCA sorted ALL SWEEPS.pdf', dpi = 1000, format = 'pdf')

    # color_layers = ['blue', 'green', 'orange', 'red']
    # fig, ax = plt.subplots(1,tier_nr)
    # fig.suptitle('spikes ends PCA ordered')
    # curr_layers = layer_dict[day][0][1:]
    # for tier, ax1 in enumerate(list(ax)):
    #     for layer, color in zip([0,1,2,3], color_layers):
    #         ax1.plot(scipy.ndimage.gaussian_filter(np.mean(np.mean(np.concatenate(spikes_end_avg_aligned_to_CSD_end_allsweeps)[tiers[tier],:,:], axis = 0)[curr_layers[layer],:], axis = 0)*1000, (5)).T, color = color)

    
    # os.chdir('..')
    
    



    

    ON_length = np.concatenate(ON_states_stops_avg_allsweeps) - np.concatenate(ON_states_starts_avg_allsweeps)
    deep_vs_superficial_sink = np.median(np.concatenate(CSD_sink_timecourse_allsweeps)[:,deep_sink_chans], axis = 1) - np.median(np.concatenate(CSD_sink_timecourse_allsweeps)[:,superf_sink_chans], axis = 1)
    deep_vs_superficial_source = np.median(np.concatenate(CSD_source_timecourse_allsweeps)[:,deep_sink_chans], axis = 1) - np.median(np.concatenate(CSD_source_timecourse_allsweeps)[:,superf_sink_chans], axis = 1)
    
    # ---------------------------------------------------------------------------------------- correlation between onset PC1 and ON length

    # to_plot = np.vstack((ON_length/1000, princ_components_start[:,0]))
    # to_plot = np.delete(to_plot, np.where(to_plot[0,:] > (np.percentile(to_plot[0,:], 75) + 1.5*(np.abs(np.percentile(to_plot[0,:], 75) - np.percentile(to_plot[0,:], 25)))))[0], axis = 1)
    # to_plot = np.delete(to_plot, np.where(to_plot[1,:] > (np.percentile(to_plot[1,:], 75) + 1.5*(np.abs(np.percentile(to_plot[1,:], 75) - np.percentile(to_plot[1,:], 25)))))[0], axis = 1)
    # to_plot = np.delete(to_plot, np.where(to_plot[1,:] < (np.percentile(to_plot[1,:], 25) - 1.5*(np.abs(np.percentile(to_plot[1,:], 75) - np.percentile(to_plot[1,:], 25)))))[0], axis = 1)
    # slope, intercept, r, p, std_err = stats.linregress(to_plot[0,:], to_plot[1,:])
    # print(f'{r} and {p} for {len(to_plot[0,:])} ON states')
    # fig, ax = plt.subplots(figsize = (4,2))
    # ax.scatter(to_plot[0,:], to_plot[1,:], color = 'k', s = 4)
    # ax.plot([np.min(to_plot[0,:]), np.max(to_plot[0,:])], [(slope*np.min(to_plot[0,:]) + intercept), (slope*np.max(to_plot[0,:]) + intercept)], color = 'k')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # plt.tight_layout()
    # plt.savefig('ON start timecourse vs ON length.jpg', format = 'jpg', dpi = 1000)
    # ax.set_xlabel('ON state length (s)', size = 18)
    # ax.set_ylabel('PC1', size = 18)
    # # ax.tick_params(axis = 'x', labelsize = 16)
    # # ax.tick_params(axis = 'y', labelsize = 16)
    # plt.tight_layout()
    # plt.savefig('ON start timecourse vs ON length.pdf', format = 'pdf', dpi = 1000)

    # #shuffle
    # r_shuffle = []
    # for i in range(1000):
    #     X_shuffle = copy.deepcopy(to_plot[0,:])
    #     np.random.shuffle(X_shuffle)
    #     slope, intercept, r, p, std_err = scipy.stats.linregress(X_shuffle, to_plot[1,:])
    #     r_shuffle.append(r)
    # r_shuffle_std = np.std(r_shuffle)
    # r_shuffle_mean = np.mean(r_shuffle)
    # print(r_shuffle_mean + 2*r_shuffle_std)


    # to_plot = np.vstack((ON_length, deep_vs_superficial_sink))
    # to_plot = np.delete(to_plot, np.where(to_plot[0,:] > (np.percentile(to_plot[0,:], 75) + 1.5*(np.abs(np.percentile(to_plot[0,:], 75) - np.percentile(to_plot[0,:], 25)))))[0], axis = 1)
    # to_plot = np.delete(to_plot, np.where(to_plot[1,:] > (np.percentile(to_plot[1,:], 75) + 1.5*(np.abs(np.percentile(to_plot[1,:], 75) - np.percentile(to_plot[1,:], 25)))))[0], axis = 1)
    # to_plot = np.delete(to_plot, np.where(to_plot[1,:] < (np.percentile(to_plot[1,:], 25) - 1.5*(np.abs(np.percentile(to_plot[1,:], 75) - np.percentile(to_plot[1,:], 25)))))[0], axis = 1)
    # slope, intercept, r, p, std_err = stats.linregress(to_plot[0,:], to_plot[1,:])
    # print(f'{r} and {p} for {len(to_plot[0,:])} ON states')
    # fig, ax = plt.subplots()
    # ax.scatter(to_plot[0,:], to_plot[1,:])
    # ax.plot([np.min(to_plot[0,:]), np.max(to_plot[0,:])], [(slope*np.min(to_plot[0,:]) + intercept), (slope*np.max(to_plot[0,:]) + intercept)], color = 'k')
    # #shuffle
    # r_shuffle = []
    # for i in range(1000):
    #     X_shuffle = copy.deepcopy(to_plot[0,:])
    #     np.random.shuffle(X_shuffle)
    #     slope, intercept, r, p, std_err = scipy.stats.linregress(X_shuffle, to_plot[1,:])
    #     r_shuffle.append(r)
    # r_shuffle_std = np.std(r_shuffle)
    # r_shuffle_mean = np.mean(r_shuffle)
    # print(r_shuffle_mean + 2*r_shuffle_std)




    # ---------------------------------------------------------------------------------------- correlation between offset PC1 and ON length

    # to_plot = np.vstack((ON_length, princ_components_end[:,0]))
    # to_plot = np.delete(to_plot, np.where(to_plot[0,:] > (np.percentile(to_plot[0,:], 75) + 1.5*(np.abs(np.percentile(to_plot[0,:], 75) - np.percentile(to_plot[0,:], 25)))))[0], axis = 1)
    # to_plot = np.delete(to_plot, np.where(to_plot[1,:] > (np.percentile(to_plot[1,:], 75) + 1.5*(np.abs(np.percentile(to_plot[1,:], 75) - np.percentile(to_plot[1,:], 25)))))[0], axis = 1)
    # to_plot = np.delete(to_plot, np.where(to_plot[1,:] < (np.percentile(to_plot[1,:], 25) - 1.5*(np.abs(np.percentile(to_plot[1,:], 75) - np.percentile(to_plot[1,:], 25)))))[0], axis = 1)
    # slope, intercept, r, p, std_err = stats.linregress(to_plot[0,:], to_plot[1,:])
    # print(f'{r} and {p} for {len(to_plot[0,:])} ON states')
    # fig, ax = plt.subplots(figsize = (4,2))
    # ax.scatter(to_plot[0,:], to_plot[1,:], color = 'k', s = 4)
    # ax.plot([np.min(to_plot[0,:]), np.max(to_plot[0,:])], [(slope*np.min(to_plot[0,:]) + intercept), (slope*np.max(to_plot[0,:]) + intercept)], color = 'k')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # plt.tight_layout()
    # plt.savefig('ON end timecourse vs ON length.jpg', format = 'jpg', dpi = 1000)
    # ax.set_xlabel('ON state length (s)', size = 18)
    # ax.set_ylabel('PC1', size = 18)
    # # ax.tick_params(axis = 'x', labelsize = 16)
    # # ax.tick_params(axis = 'y', labelsize = 16)
    # plt.tight_layout()
    # plt.savefig('ON end timecourse vs ON length.pdf', format = 'pdf', dpi = 1000)

    # #shuffle
    # r_shuffle = []
    # for i in range(1000):
    #     X_shuffle = copy.deepcopy(to_plot[0,:])
    #     np.random.shuffle(X_shuffle)
    #     slope, intercept, r, p, std_err = scipy.stats.linregress(X_shuffle, to_plot[1,:])
    #     r_shuffle.append(r)
    # r_shuffle_std = np.std(r_shuffle)
    # r_shuffle_mean = np.mean(r_shuffle)
    # print(r_shuffle_mean + 2*r_shuffle_std)

    # # ---------------------------------------------------------------------------------------- correlation between offset PC1 and onset PC1

    # to_plot = np.vstack((princ_components_start[:,0], princ_components_end[:,0]))
    # # to_plot = np.vstack((princ_components_start[:,0], princ_components_end[:,0]))
    # to_plot = np.delete(to_plot, np.where(to_plot[0,:] > (np.percentile(to_plot[0,:], 75) + 1.5*(np.abs(np.percentile(to_plot[0,:], 75) - np.percentile(to_plot[0,:], 25)))))[0], axis = 1)
    # to_plot = np.delete(to_plot, np.where(to_plot[0,:] < (np.percentile(to_plot[0,:], 75) - 1.5*(np.abs(np.percentile(to_plot[0,:], 75) - np.percentile(to_plot[0,:], 25)))))[0], axis = 1)
    # to_plot = np.delete(to_plot, np.where(to_plot[1,:] > (np.percentile(to_plot[1,:], 75) + 1.5*(np.abs(np.percentile(to_plot[1,:], 75) - np.percentile(to_plot[1,:], 25)))))[0], axis = 1)
    # to_plot = np.delete(to_plot, np.where(to_plot[1,:] < (np.percentile(to_plot[1,:], 25) - 1.5*(np.abs(np.percentile(to_plot[1,:], 75) - np.percentile(to_plot[1,:], 25)))))[0], axis = 1)
    # slope, intercept, r, p, std_err = stats.linregress(to_plot[0,:], to_plot[1,:])
    # print(f'{r} and {p} for {len(to_plot[0,:])} ON states')
    # fig, ax = plt.subplots(figsize = (4,2))
    # ax.scatter(to_plot[0,:], to_plot[1,:], color = 'k', s = 4)
    # ax.plot([np.min(to_plot[0,:]), np.max(to_plot[0,:])], [(slope*np.min(to_plot[0,:]) + intercept), (slope*np.max(to_plot[0,:]) + intercept)], color = 'k')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # plt.tight_layout()
    # plt.savefig('ON start vs ON end timecourse.jpg', format = 'jpg', dpi = 1000)
    # ax.set_xlabel('ON state length (s)', size = 18)
    # ax.set_ylabel('PC1', size = 18)
    # # ax.tick_params(axis = 'x', labelsize = 16)
    # # ax.tick_params(axis = 'y', labelsize = 16)
    # plt.tight_layout()
    # plt.savefig('ON start vs ON end timecourse.pdf', format = 'pdf', dpi = 1000)

    # #shuffle
    # r_shuffle = []
    # for i in range(1000):
    #     X_shuffle = copy.deepcopy(to_plot[0,:])
    #     np.random.shuffle(X_shuffle)
    #     slope, intercept, r, p, std_err = scipy.stats.linregress(X_shuffle, to_plot[1,:])
    #     r_shuffle.append(r)
    # r_shuffle_std = np.std(r_shuffle)
    # r_shuffle_mean = np.mean(r_shuffle)
    # print(r_shuffle_mean + 2*r_shuffle_std)



    # ----------------------------------------------------------------------------------------- cross correlation: activation pattern vs previous ones
    
    # sweep_lengths = [len(i) for i in ON_states_starts_avg_allsweeps]
    # sweep_indices = np.insert(np.cumsum(sweep_lengths), 0 , 0) # how many ON states per sweep
    # sweep_indices = [np.arange(j, sweep_indices[j_ind + 1]) for j_ind, j in enumerate(list(sweep_indices[:-1]))] # how many ON states per sweep
    # deep_vs_superficial_sink_allsweeps = [deep_vs_superficial_sink[i] for i in sweep_indices]
    # PC1_start_allsweeps = [princ_components_start[i] for i in sweep_indices]
    # cross_corr = [[] for i in range(2)]
    # outliers_high = [np.where(i > (np.percentile(i, 75) + 1.5*(np.abs(np.percentile(i, 75) - np.percentile(i, 25)))))[0] for i in deep_vs_superficial_sink_allsweeps]
    # outliers_low = [np.where(i < (np.percentile(i, 75) - 1.5*(np.abs(np.percentile(i, 75) - np.percentile(i, 25)))))[0] for i in deep_vs_superficial_sink_allsweeps]
    # outliers = [np.unique(np.concatenate((outliers_high[i], outliers_low[i]))) for i in range(10)]
    # deep_vs_superficial_sink_allsweeps = [np.delete(i, outliers[ind]) for ind,i in enumerate(deep_vs_superficial_sink_allsweeps)]
    # X = np.concatenate(deep_vs_superficial_sink_allsweeps)
    # Y = np.concatenate(deep_vs_superficial_sink_allsweeps)
    # #shuffle
    # r_shuffle = []
    # for i in range(1000):
    #     X_shuffle = copy.deepcopy(X)
    #     np.random.shuffle(X_shuffle)
    #     slope, intercept, r, p, std_err = scipy.stats.linregress(X_shuffle, Y)
    #     r_shuffle.append(r)
    # r_shuffle_std = np.std(r_shuffle)
    # r_shuffle_mean = np.mean(r_shuffle)
               
    # #cross corr within each sweep
    # # lag correlation between OFF and next UP across all mice 
    # lag_correlation_all_sweeps = []
    # r_shuffle_std_allsweeps = []
    # r_shuffle_mean_allsweeps = []
    # for sweep, sink_timecourse in enumerate(PC1_start_allsweeps):
    #     # print(sweep)
    #     # correlation betwen activation pattern and activation pattern of n + k UP state
    #     cross_corr = [[] for i in range(2)]
    #     outliers_high = np.where(sink_timecourse > (np.percentile(sink_timecourse, 75) + 1.5*(np.abs(np.percentile(sink_timecourse, 75) - np.percentile(sink_timecourse, 25)))))[0]
    #     outliers_low = np.where(sink_timecourse < (np.percentile(sink_timecourse, 75) - 1.5*(np.abs(np.percentile(sink_timecourse, 75) - np.percentile(sink_timecourse, 25)))))[0]
    #     outliers = np.unique(np.concatenate((outliers_high, outliers_low)))
    #     sink_timecourse = np.delete(sink_timecourse, outliers)
    #     for offset in range(20):
    #         if offset == 0:
    #             X = sink_timecourse
    #             Y = sink_timecourse
    #             slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
    #             cross_corr[1].append(0)
                
    #             #shuffle
    #             r_shuffle = []
    #             for i in range(1000):
    #                 X_shuffle = copy.deepcopy(X)
    #                 np.random.shuffle(X_shuffle)
    #                 slope, intercept, r, p, std_err = scipy.stats.linregress(X_shuffle, Y)
    #                 r_shuffle.append(r)
    #             r_shuffle_std = np.std(r_shuffle)
    #             r_shuffle_mean = np.mean(r_shuffle)
    #             r_shuffle_std_allsweeps.append(r_shuffle_std)
    #             r_shuffle_mean_allsweeps.append(r_shuffle_mean)

    #         else:
    #             #backwards (previous OFFs)
    #             X = sink_timecourse[:-offset]
    #             Y = sink_timecourse[offset:]
    #             slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
    #             cross_corr[0].append(r)
    
    #             #forwards (OFFs after)
    #             X = sink_timecourse[offset:]
    #             Y = sink_timecourse[:-offset]
    #             slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
    #             cross_corr[1].append(r)
                
    #     lag_correlation_all_sweeps.append(np.concatenate(cross_corr))
    # lag_correlation_all_sweeps = np.asarray(lag_correlation_all_sweeps)

    # fig, ax = plt.subplots(figsize = (6,3))
    # ax.bar(np.linspace(-19, 19, 39), np.mean(lag_correlation_all_sweeps, axis = 0), color = 'k')
    # ax.axhline(np.mean(r_shuffle_mean_allsweeps) + 1*np.mean(r_shuffle_std_allsweeps), linestyle = '--', color = 'k')
    # ax.axhline(np.mean(r_shuffle_mean_allsweeps) - 1*np.mean(r_shuffle_std_allsweeps), linestyle = '--', color = 'k')
    # ax.set_ylim([-0.5, 0.5])
    # ax.tick_params(axis = 'x', labelsize = 16)
    # ax.tick_params(axis = 'y', labelsize = 16)
    # ax.set_xlabel('Offset (i)', size = 16)
    # ax.set_ylabel('R($\mathregular{ON_{n+i}}$, $\mathregular{ON_{n}}$)', size = 16)
    # plt.tight_layout()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.savefig('CSD start timecourse LAG.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD start timecourse LAG.pdf', dpi = 1000, format = 'pdf') 




    # ----------------------------------------------------------------------------------------- cross correlation: deactivation pattern vs previous ones
    
    # sweep_lengths = [len(i) for i in ON_states_starts_avg_allsweeps]
    # sweep_indices = np.insert(np.cumsum(sweep_lengths), 0 , 0) # how many ON states per sweep
    # sweep_indices = [np.arange(j, sweep_indices[j_ind + 1]) for j_ind, j in enumerate(list(sweep_indices[:-1]))] # how many ON states per sweep
    # deep_vs_superficial_source_allsweeps = [deep_vs_superficial_source[i] for i in sweep_indices]
    # PC1_end_allsweeps = [princ_components_end[i] for i in sweep_indices]
    # cross_corr = [[] for i in range(2)]
    # outliers_high = [np.where(i > (np.percentile(i, 75) + 1.5*(np.abs(np.percentile(i, 75) - np.percentile(i, 25)))))[0] for i in deep_vs_superficial_source_allsweeps]
    # outliers_low = [np.where(i < (np.percentile(i, 75) - 1.5*(np.abs(np.percentile(i, 75) - np.percentile(i, 25)))))[0] for i in deep_vs_superficial_source_allsweeps]
    # outliers = [np.unique(np.concatenate((outliers_high[i], outliers_low[i]))) for i in range(10)]
    # deep_vs_superficial_source_allsweeps = [np.delete(i, outliers[ind]) for ind,i in enumerate(deep_vs_superficial_source_allsweeps)]
    # X = np.concatenate(deep_vs_superficial_source_allsweeps)
    # Y = np.concatenate(deep_vs_superficial_source_allsweeps)
    # #shuffle
    # r_shuffle = []
    # for i in range(1000):
    #     X_shuffle = copy.deepcopy(X)
    #     np.random.shuffle(X_shuffle)
    #     slope, intercept, r, p, std_err = scipy.stats.linregress(X_shuffle, Y)
    #     r_shuffle.append(r)
    # r_shuffle_std = np.std(r_shuffle)
    # r_shuffle_mean = np.mean(r_shuffle)
               
    # #cross corr within each sweep
    # # lag correlation between OFF and next UP across all mice 
    # lag_correlation_all_sweeps = []
    # r_shuffle_std_allsweeps = []
    # r_shuffle_mean_allsweeps = []
    # for sweep, source_timecourse in enumerate(PC1_end_allsweeps):
    #     # print(sweep)
    #     # correlation betwen activation pattern and activation pattern of n + k UP state
    #     cross_corr = [[] for i in range(2)]
    #     outliers_high = np.where(source_timecourse > (np.percentile(source_timecourse, 75) + 1.5*(np.abs(np.percentile(source_timecourse, 75) - np.percentile(source_timecourse, 25)))))[0]
    #     outliers_low = np.where(source_timecourse < (np.percentile(source_timecourse, 75) - 1.5*(np.abs(np.percentile(source_timecourse, 75) - np.percentile(source_timecourse, 25)))))[0]
    #     outliers = np.unique(np.concatenate((outliers_high, outliers_low)))
    #     source_timecourse = np.delete(source_timecourse, outliers)
    #     for offset in range(20):
    #         if offset == 0:
    #             X = source_timecourse
    #             Y = source_timecourse
    #             slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
    #             cross_corr[1].append(0)
                
    #             #shuffle
    #             r_shuffle = []
    #             for i in range(1000):
    #                 X_shuffle = copy.deepcopy(X)
    #                 np.random.shuffle(X_shuffle)
    #                 slope, intercept, r, p, std_err = scipy.stats.linregress(X_shuffle, Y)
    #                 r_shuffle.append(r)
    #             r_shuffle_std = np.std(r_shuffle)
    #             r_shuffle_mean = np.mean(r_shuffle)
    #             r_shuffle_std_allsweeps.append(r_shuffle_std)
    #             r_shuffle_mean_allsweeps.append(r_shuffle_mean)

    #         else:
    #             #backwards (previous OFFs)
    #             X = source_timecourse[:-offset]
    #             Y = source_timecourse[offset:]
    #             slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
    #             cross_corr[0].append(r)
    
    #             #forwards (OFFs after)
    #             X = source_timecourse[offset:]
    #             Y = source_timecourse[:-offset]
    #             slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
    #             cross_corr[1].append(r)
                
    #     lag_correlation_all_sweeps.append(np.concatenate(cross_corr))
    # lag_correlation_all_sweeps = np.asarray(lag_correlation_all_sweeps)

    # fig, ax = plt.subplots(figsize = (6,3))
    # ax.bar(np.linspace(-19, 19, 39), np.mean(lag_correlation_all_sweeps, axis = 0), color = 'k')
    # ax.axhline(np.mean(r_shuffle_mean_allsweeps) + 1*np.mean(r_shuffle_std_allsweeps), linestyle = '--', color = 'k')
    # ax.axhline(np.mean(r_shuffle_mean_allsweeps) - 1*np.mean(r_shuffle_std_allsweeps), linestyle = '--', color = 'k')
    # ax.set_ylim([-0.5, 0.5])
    # ax.tick_params(axis = 'x', labelsize = 16)
    # ax.tick_params(axis = 'y', labelsize = 16)
    # ax.set_xlabel('Offset (i)', size = 16)
    # ax.set_ylabel('R($\mathregular{ON_{n+i}}$, $\mathregular{ON_{n}}$)', size = 16)
    # plt.tight_layout()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.savefig('CSD end timecourse LAG.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD end timecourse LAG.pdf', dpi = 1000, format = 'pdf') 





    os.chdir('..')
    
    
    


#%% -------------------------------------------------- ON start sink distribution deep vs superficial before vs after pairing across mice
bins = 100

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# use days with a nowhisker stim
CSD_sink_timecourse_L4_vs_L5_before_ALL = []
CSD_sink_timecourse_L4_vs_L5_after_ALL = []
for day_ind, day in enumerate(days[:6]):
# for day_ind, day in enumerate(['160615']):
    os.chdir(day)
    print(day)

    CSD_sink_timecourse_L4_vs_L5_before = np.load('CSD_sink_timecourse_L4_vs_L5_before.npy')
    CSD_sink_timecourse_L4_vs_L5_after = np.load('CSD_sink_timecourse_L4_vs_L5_after.npy')
    CSD_sink_timecourse_L4_vs_L5 = np.load('CSD_sink_timecourse_L4_vs_L5.npy')
    CSD_sink_timecourse_L4_vs_L5_before_ALL.append(CSD_sink_timecourse_L4_vs_L5_before)
    CSD_sink_timecourse_L4_vs_L5_after_ALL.append(CSD_sink_timecourse_L4_vs_L5_after)
    
    fig, ax = plt.subplots(figsize = (6,3))
    ax.hist(CSD_sink_timecourse_L4_vs_L5_before, density = True, bins = bins, color = 'black', alpha = 0.5)
    ax.hist(CSD_sink_timecourse_L4_vs_L5_after, density = True, bins = bins, color = 'cyan', alpha = 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 16)
    ax.tick_params(axis = 'y', labelsize = 16)
    ax.set_xlim([-200,200])
    plt.tight_layout()
    plt.savefig('CSD sink timecourse before vs after.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('CSD sink timecourse before vs after.pdf', dpi = 1000, format = 'pdf')
    os.chdir('..')
    
fig, ax = plt.subplots(figsize = (6,3))
color_mice = ['black', 'red', 'blue', 'green', 'purple', 'cyan']
cumsum_all_before = []
cumsum_all_after = []
for mouse, color in enumerate(color_mice):
    # ax.hist(CSD_sink_timecourse_L4_vs_L5_before_ALL[mouse], bins=bins, density = True, cumulative = True, color = color, histtype = 'step')
    y, binEdges = np.histogram(CSD_sink_timecourse_L4_vs_L5_before_ALL[mouse], bins=bins, density = True, range = [-300,300])
    y = 100*np.cumsum(y)/np.max(np.cumsum(y))
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    # ax.plot(bincenters, 100*np.cumsum(y)/np.max(np.cumsum(y)), color = color)
    # ax.plot(bincenters, y, color = 'k', alpha = 0.25)
    cumsum_all_before.append(y)
    
    y, binEdges = np.histogram(CSD_sink_timecourse_L4_vs_L5_after_ALL[mouse], bins=bins, density = True, range = [-300,300])
    y = 100*np.cumsum(y)/np.max(np.cumsum(y))
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    # ax.plot(bincenters, 100*np.cumsum(y)/np.max(np.cumsum(y)), color = color)
    # ax.plot(bincenters, y, color = 'cyan', alpha = 0.25)
    cumsum_all_after.append(y)

bincenters = bincenters/1000
ax.plot(bincenters, np.mean(np.asarray(cumsum_all_before), axis = 0), color = 'k', linewidth = 2)
before_err = np.std(np.asarray(cumsum_all_before), axis = 0)/np.sqrt(6)
ax.plot(bincenters, np.mean(np.asarray(cumsum_all_after), axis = 0), color = 'cyan', linewidth = 2)
after_err = np.std(np.asarray(cumsum_all_after), axis = 0)/np.sqrt(6)
ax.fill_between(bincenters, np.mean(np.asarray(cumsum_all_before), axis = 0) + before_err, np.mean(np.asarray(cumsum_all_before) - before_err, axis = 0), color = 'k', alpha = 0.3)
ax.fill_between(bincenters, np.mean(np.asarray(cumsum_all_after), axis = 0) + after_err, np.mean(np.asarray(cumsum_all_after) - after_err, axis = 0), color = 'cyan', alpha = 0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis = 'x', labelsize = 16)
ax.tick_params(axis = 'y', labelsize = 16)
ax.set_xlabel('deep vs superficial CSD onset (s)', size = 16)
ax.set_ylabel('% of ON-states', size = 16)
plt.tight_layout()
plt.savefig('CSD sink timecourse cumulative ALL.jpg', dpi = 1000, format = 'jpg')
plt.savefig('CSD sink timecourse cumulative ALL.pdf', dpi = 1000, format = 'pdf')

print(scipy.stats.kstest(np.mean(np.asarray(cumsum_all_before), axis = 0), np.mean(np.asarray(cumsum_all_after), axis = 0)))



#%% -------------------------------------------------- ON end source distribution deep vs superficial before vs after pairing across mice

bins = 100

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# use days with a nowhisker stim
CSD_source_timecourse_L4_vs_L5_before_ALL = []
CSD_source_timecourse_L4_vs_L5_after_ALL = []
for day_ind, day in enumerate(days[:6]):
# for day_ind, day in enumerate(['160615']):
    os.chdir(day)
    print(day)

    CSD_source_timecourse_L4_vs_L5_before = np.load('CSD_source_timecourse_L4_vs_L5_before.npy')
    CSD_source_timecourse_L4_vs_L5_after = np.load('CSD_source_timecourse_L4_vs_L5_after.npy')
    CSD_source_timecourse_L4_vs_L5 = np.load('CSD_source_timecourse_L4_vs_L5.npy')
    CSD_source_timecourse_L4_vs_L5_before_ALL.append(CSD_source_timecourse_L4_vs_L5_before)
    CSD_source_timecourse_L4_vs_L5_after_ALL.append(CSD_source_timecourse_L4_vs_L5_after)
    
    fig, ax = plt.subplots(figsize = (6,3))
    ax.hist(CSD_source_timecourse_L4_vs_L5_before, density = True, bins = bins, color = 'black', alpha = 0.5)
    ax.hist(CSD_source_timecourse_L4_vs_L5_after, density = True, bins = bins, color = 'cyan', alpha = 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 16)
    ax.tick_params(axis = 'y', labelsize = 16)
    # ax.set_xlim([-200,200])
    plt.tight_layout()
    plt.savefig('CSD source timecourse before vs after.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('CSD source timecourse before vs after.pdf', dpi = 1000, format = 'pdf')
    os.chdir('..')
    
fig, ax = plt.subplots(figsize = (6,3))
color_mice = ['black', 'red', 'blue', 'green', 'purple', 'cyan']
cumsum_all_before = []
cumsum_all_after = []
for mouse, color in enumerate(color_mice):
    # ax.hist(CSD_source_timecourse_L4_vs_L5_before_ALL[mouse], bins=bins, density = True, cumulative = True, color = color, histtype = 'step')
    y, binEdges = np.histogram(CSD_source_timecourse_L4_vs_L5_before_ALL[mouse], bins=bins, density = True, range = [-300,300])
    y = 100*np.cumsum(y)/np.max(np.cumsum(y))
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    # ax.plot(bincenters, 100*np.cumsum(y)/np.max(np.cumsum(y)), color = color)
    # ax.plot(bincenters, y, color = 'k', alpha = 0.25)
    cumsum_all_before.append(y)
    
    y, binEdges = np.histogram(CSD_source_timecourse_L4_vs_L5_after_ALL[mouse], bins=bins, density = True, range = [-300,300])
    y = 100*np.cumsum(y)/np.max(np.cumsum(y))
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    # ax.plot(bincenters, 100*np.cumsum(y)/np.max(np.cumsum(y)), color = color)
    # ax.plot(bincenters, y, color = 'cyan', alpha = 0.25)
    cumsum_all_after.append(y)

bincenters = bincenters/1000
ax.plot(bincenters, np.mean(np.asarray(cumsum_all_before), axis = 0), color = 'k', linewidth = 2)
before_err = np.std(np.asarray(cumsum_all_before), axis = 0)/np.sqrt(6)
ax.plot(bincenters, np.mean(np.asarray(cumsum_all_after), axis = 0), color = 'cyan', linewidth = 2)
after_err = np.std(np.asarray(cumsum_all_after), axis = 0)/np.sqrt(6)
ax.fill_between(bincenters, np.mean(np.asarray(cumsum_all_before), axis = 0) + before_err, np.mean(np.asarray(cumsum_all_before) - before_err, axis = 0), color = 'k', alpha = 0.3)
ax.fill_between(bincenters, np.mean(np.asarray(cumsum_all_after), axis = 0) + after_err, np.mean(np.asarray(cumsum_all_after) - after_err, axis = 0), color = 'cyan', alpha = 0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis = 'x', labelsize = 16)
ax.tick_params(axis = 'y', labelsize = 16)
ax.set_xlabel('deep vs superficial CSD onset (s)', size = 16)
ax.set_ylabel('% of ON-states', size = 16)
plt.tight_layout()
plt.savefig('CSD source timecourse cumulative ALL.jpg', dpi = 1000, format = 'jpg')
plt.savefig('CSD source timecourse cumulative ALL.pdf', dpi = 1000, format = 'pdf')

print(scipy.stats.kstest(np.mean(np.asarray(cumsum_all_before), axis = 0), np.mean(np.asarray(cumsum_all_after), axis = 0)))


#%%
    # --------------------------------------------------------- plot 100 ON starts
    # fig, ax = plt.subplots(10,10, figsize = (19,11))
    # for SO_ind, ax1 in enumerate(list(ax.flatten())):
    #     ax1.imshow(scipy.ndimage.gaussian_filter(MUA_matrix_start_avg[SO_ind,cortical_chans,:], (0,10)), cmap = 'jet', aspect = 10) # 
    #     ax1.tick_params(axis="x", labelsize=1)
    #     ax1.tick_params(axis="y", labelsize=1) 
    # plt.tight_layout(pad = 0.001)
    # plt.savefig('MUA ON starts 100.jpg', dpi = 1000, format = 'jpg')

                
    # if day == '160622':
    #     minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_start_avg[:,23:25,:], (0,0,10))))/6
    # else:
    #     minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_start_avg[:,cortical_chans,:], (0,0,10))))/3
    # CSD_sink_timecourse = []
    # fig, ax = plt.subplots(10,10, figsize = (19,11))
    # for SO_ind, ax1 in enumerate(list(ax.flatten())):
    #     ax1.imshow(scipy.ndimage.gaussian_filter(CSD_matrix_start_avg[SO_ind,:,:], (0,10)), cmap = 'jet', aspect = 10, vmin =  -minmax, vmax = minmax) #
    #     CSD_for_gradient = copy.deepcopy(scipy.ndimage.gaussian_filter(CSD_matrix_start_avg[SO_ind,:,200:], (0,10)))
    #     CSD_for_gradient[CSD_for_gradient > 0] = 0 #has to be a sink so negative
        
    #     #try with first time it crosses 0.25% of min CSD
    #     curr_CSD_crossing = []
    #     for chan in range(32):   
    #         curr_CSD_crossing.append(np.argwhere(CSD_for_gradient[chan,:] <= np.min(CSD_for_gradient[chan,:])*0.25)[0][0] + 200) 
    #     CSD_sink_timecourse.append(np.asarray(curr_CSD_crossing))
        
    #     #try with steepest gradient
    #     CSD_for_gradient[CSD_for_gradient > 0] = 0 #has to be a sink so negative
    #     curr_CSD_sink_gradient = np.argmin(np.diff(CSD_for_gradient, axis = 1), axis = 1) + 200
        
    #     ax1.plot(np.asarray(curr_CSD_crossing)[PCA_CSD_chans], np.arange(PCA_CSD_chans[0], PCA_CSD_chans[-1]+1, 1), color = 'white')
    #     ax1.tick_params(axis="x", labelsize=1)
    #     ax1.tick_params(axis="y", labelsize=1) 
    # plt.tight_layout(pad = 0.001)
    # plt.savefig('CSD ON starts 100.jpg', dpi = 1000, format = 'jpg')


    # CSD sink starts across channels
    CSD_sink_timecourse = []
    for SO_ind, SO in enumerate(list(range(CSD_matrix_start_avg.shape[0]))):
        CSD_for_gradient = copy.deepcopy(scipy.ndimage.gaussian_filter(CSD_matrix_start_avg[SO_ind,:,200:], (0,10))) # start 100ms before average MUA was detected
        CSD_for_gradient[CSD_for_gradient > 0] = 0 #has to be a sink so negative
        curr_CSD_crossing = []
        for chan in range(nchans):   
            curr_CSD_crossing.append(np.argwhere(CSD_for_gradient[chan,:] <= np.min(CSD_for_gradient[chan,:])*0.25)[0][0]-100) 
        CSD_sink_timecourse.append(np.asarray(curr_CSD_crossing))
    CSD_sink_timecourse = np.asarray(CSD_sink_timecourse)
    
    
    # extract maps aligned to median L5 CSD sink start time as used for PCA to average
    CSD_matrix_start_avg_aligned_to_CSD_start = []
    MUA_matrix_start_avg_aligned_to_CSD_start = []
    for SO_ind, ON_start in enumerate(list(ON_states_starts_avg*1000)):
        curr_ON_start = int(ON_start) + int(np.median(CSD_sink_timecourse[SO_ind, layer_dict[day][0][3]]))
        CSD_matrix_start_avg_aligned_to_CSD_start.append(CSD_all_smoothed[:,int(curr_ON_start - 300):int(curr_ON_start + 300)])
        MUA_matrix_start_avg_aligned_to_CSD_start.append(MUA_power_binned_log_smoothed[:,int(curr_ON_start - 300):int(curr_ON_start + 300)])
    CSD_matrix_start_avg_aligned_to_CSD_start = np.asarray(CSD_matrix_start_avg_aligned_to_CSD_start)
    MUA_matrix_start_avg_aligned_to_CSD_start = np.asarray(MUA_matrix_start_avg_aligned_to_CSD_start)


    # fig, ax = plt.subplots()
    # fig.suptitle('CSD starts')
    # ax.imshow(CSD_sink_timecourse.T, cmap = 'bwr', aspect = 2, vmin = -300, vmax = 300)
    
    # take only chans layer 4 to layer 6 and normalize to median layer 5 time
    CSD_sink_timecourse_zeroed = CSD_sink_timecourse.T - np.median(CSD_sink_timecourse[:, layer_dict[day][0][3]], axis = 1)
    CSD_sink_timecourse_for_PCA = CSD_sink_timecourse_zeroed.T[:,PCA_CSD_chans]
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD starts only chans for sorting')
    # ax.imshow(CSD_sink_timecourse_for_PCA.T, cmap = 'bwr', aspect = 2, vmin = -300, vmax = 300)

    princ_components, eigenvectors, eigenvalues, var_ratio, scaled = PCA_normed(CSD_sink_timecourse_for_PCA)
    # # plot ordered by first PC:
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD start ordered')
    # ax.imshow(CSD_sink_timecourse_for_PCA[np.argsort(princ_components[:,0]),:].T, cmap = 'bwr', aspect = 4, vmin = -300, vmax = 300)
    # plt.savefig('CSD starts PCA sorted.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD starts PCA sorted.pdf', dpi = 1000, format = 'pdf')
    
    # #plot on PC1 vs PC2 scatterplot and take out outliers
    # fig, ax = plt.subplots()
    # ax.scatter(princ_components[:,0], princ_components[:,1], s = 8, color = 'k')  
    # X_outliers = np.where(X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))[0]
    # Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
    # outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
    # if exclude_outliers:
    #     X = np.delete(X, outliers)
    #     Y = np.delete(Y, outliers)



    #divide into segments and plot start heatmaps
    tier_nr = 4
    tiers = np.array_split(np.argsort(princ_components[:,0]),tier_nr)
    
    if day == '160622':
        minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_start_avg_aligned_to_CSD_start[:,23:25,:], (0,0,10))))/6
    else:
        minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_start_avg_aligned_to_CSD_start[:,cortical_chans,:], (0,0,10))))/3
        
    fig, ax = plt.subplots(1,tier_nr)
    fig.suptitle('CSD PCA ordered')
    for tier, ax1 in enumerate(list(ax)):
        ax1.imshow(np.mean(CSD_matrix_start_avg_aligned_to_CSD_start[tiers[tier],:,:], axis = 0), aspect = 40, cmap = 'jet', vmin =  -minmax, vmax = minmax)
    # plt.savefig('CSD starts PCA sorted heatmaps.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD starts PCA sorted heatmaps.pdf', dpi = 1000, format = 'pdf')

    # minmax = max([np.max(np.mean(MUA_matrix_start_avg_aligned_to_CSD_start[tiers[ind],:,:][:,cortical_chans,:], axis = 0)) for ind in range(tier_nr)])
    # fig, ax = plt.subplots(1,tier_nr)
    # fig.suptitle('MUA PCA ordered')
    # for ind, ax1 in enumerate(list(ax)):
    #     ax1.imshow(np.mean(MUA_matrix_start_avg_aligned_to_CSD_start[tiers[ind],:,:], axis = 0), aspect = 40, cmap = 'jet', vmax = minmax)
    # plt.savefig('MUA starts PCA sorted.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('MUA starts PCA sorted.pdf', dpi = 1000, format = 'pdf')

    # PC1 vs length of UP states
    fig, ax = plt.subplots()
    fig.suptitle('CSD PCA ordered next UP')
    ax.scatter(princ_components[:,0], ON_states_stops_avg - ON_states_starts_avg, s = 8, color = 'k')  
    # slope, intercept, r, p, std_err = stats.linregress(princ_components[:,0], ON_states_stops_avg - ON_states_starts_avg)
    # print(f'{r**2} and {p} for {len(princ_components[:,0])} channels')

    # PC1 vs length of previous DOWN states
    fig, ax = plt.subplots()
    fig.suptitle('CSD PCA ordered previous DOWN')
    ax.scatter(princ_components[1:,0], ON_states_starts_avg[1:] - ON_states_starts_avg[:-1], s = 8, color = 'k')  

    # slope, intercept, r, p, std_err = stats.linregress(X, Y)
    # print(f'{r**2} and {p} for {len(X)} channels')
    # ax.scatter(X,Y, color = 'k')
    # ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')

    CSD_sink_starts_allsweeps.append(CSD_sink_timecourse_zeroed)
        
        
        
    os.chdir('..')
    os.chdir('..')







#%% ON start and end based on detected LFP slow waves


days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# use days with a nowhisker stim
# for day_ind, day in enumerate(days[0:6]):
for day_ind, day in enumerate(['160615']):
    os.chdir(day)
    print(day)
    
    if day == '160614':
        highpass_cutoff = 3
    elif day == '160514':
        highpass_cutoff = 4
    else:
        highpass_cutoff = 4

    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    MUA_all_sweeps = pickle.load(open('MUA_all_sweeps','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    
    nchans = LFP_all_sweeps[0].shape[0]

    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    UP_Cross_sweeps = pickle.load(open('UP_Cross_sweeps','rb'))
    
    for sweep in range(10):
        curr_LFP = LFP_all_sweeps[sweep]
        curr_MUA = MUA_all_sweeps[sweep]
        stims = stim_times[sweep]
        
        CSD_matrix = -np.eye(nchans) # 
        for j in range(1, CSD_matrix.shape[0] - 1):
            CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
        curr_CSD_all = - np.dot(CSD_matrix, curr_LFP)
        curr_CSD_all_smoothed = - np.dot(CSD_matrix, scipy.ndimage.gaussian_filter(curr_LFP, (2, 0)))
        curr_CSD_all[0,:] = 0
        curr_CSD_all_smoothed[0,:] = 0
        curr_CSD_all[-1,:] = 0
        curr_CSD_all_smoothed[-1,:] = 0

        UP_peaks = []
        for UP_state_ind, UP_state in enumerate(UP_Cross_sweeps[sweep][layer_dict_1[day][sweep][2][0]]):
            UP_peaks.append(np.argmin(curr_LFP[UP_state - 1000:UP_state]) + UP_state - 1000)
            

        






