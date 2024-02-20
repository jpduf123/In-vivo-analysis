# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:57:21 2023

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
import pycircstat

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


#%% -------------------------------------------------------------------------- detect ON states from bimodal distribution in each channel or from average logMUA -----------------------------------------------

smooth_over = 15 # smooth logMUA power over how many bins for state detection

# for cond in ['UP_pairing', 'DOWN_pairing']:
    # os.chdir(os.path.join(overall_path, fr'{cond}'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

# use days with a nowhisker stim
for day_ind, day in enumerate(days[0:6]):
# for day_ind, day in enumerate(['160810']):
    os.chdir(day)
    print(day)
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
    elif day == '160614':
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]**2
        spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 1.3
    else:
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]**2
        spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 1.3

    nchans = LFP.shape[0]
    # # take log and smooth MUA power across time
    MUA_power_binned_log_smoothed = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_power_binned), (0,80))
    MUA_power_binned_smoothed = scipy.ndimage.gaussian_filter(MUA_power_binned, (0,80))
    # # normalize within each channel, take median value in each channel
    MUA_power_binned_rel = (MUA_power_binned.T/np.median(MUA_power_binned, axis = 1)).T
    MUA_power_binned_log_smoothed_rel = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_power_binned_rel), (0,80))
    
    # calculate CSD across the whole recording, smoothed across channels
    CSD_matrix = -np.eye(nchans) # 
    for j in range(1, CSD_matrix.shape[0] - 1):
        CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
    CSD_all = - np.dot(CSD_matrix, LFP)
    CSD_all_smoothed = - np.dot(CSD_matrix, scipy.ndimage.gaussian_filter(LFP, (2, 0)))
    CSD_all[0,:] = 0
    CSD_all_smoothed[0,:] = 0
    CSD_all[-1,:] = 0
    CSD_all_smoothed[-1,:] = 0
    
    # plot a snippet of MUA power of each channel
    # fig, ax = plt.subplots()
    # for i in range(MUA_power_binned_log_smoothed.shape[0]):
    #     ax.plot(MUA_power_binned_log_smoothed[i,0:20000] - i*np.ones_like(MUA_power_binned_log_smoothed[i,0:20000]), linewidth = 0.5)
    
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




    # -------------------------------------------------------------------- detect ON_OFF threshold in every channel -------------------------
    OFF_ON_thresholds = []
    bimodal_channels = []
    # take peaks of bimodal MUA power distribution
    bins = 1000
    for chan in range(32):
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
    chans_for_time_matrix = np.argwhere(np.isnan(OFF_ON_thresholds) != True)     
            
    # clean up bimodadl channels in some mice
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
        
    bimodal_avg_rel_MUA = np.mean(MUA_power_binned_log_smoothed_rel[bimodal_channels,:], axis = 0)
    OFF_peak_avg = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(bimodal_avg_rel_MUA, bins = bins)[0], 25), distance = 90)[0][0]
    ON_peak_avg = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(np.histogram(bimodal_avg_rel_MUA, bins = bins)[0], 25), distance = 90)[0][1]
    OFF_value_avg = np.histogram(bimodal_avg_rel_MUA, bins = bins)[1][OFF_peak_avg]
    ON_value_avg = np.histogram(bimodal_avg_rel_MUA, bins = bins)[1][ON_peak_avg]
    # OFF_ON_threshold is about halfway between both peaks (varies in Nghiem and Sanchez-vives)
    OFF_ON_threshold_avg = OFF_value_avg + 0.3*(ON_value_avg - OFF_value_avg)

    fig, ax = plt.subplots()
    fig.suptitle(f'{day}')
    ax.hist(bimodal_avg_rel_MUA, bins = 1000)
    ax.plot(np.histogram(bimodal_avg_rel_MUA, bins = 1000)[1][:-1], scipy.ndimage.gaussian_filter1d(np.histogram(bimodal_avg_rel_MUA, bins = 1000)[0], 20), color = 'k')    
    ax.axvline(OFF_ON_threshold_avg, color = 'red')    
    
    # os.chdir('..')
    # os.chdir('..')

    # plot MUA power histogram for each channel to decide which is best bimodal
    # fig, ax = plt.subplots(4,8, figsize = (20,10))
    # to_plot = MUA_power_binned_log_smoothed
    # # fig.suptitle(f'{day}')
    # spacer = 0
    # bins = 1000
    # tot_chans = MUA_power_binned_log_smoothed.shape[0]
    # for i in range(tot_chans):
    #     X = np.histogram(to_plot[i,:], bins = bins)[1][:-1]
    #     ax.flatten()[i].plot(X, np.histogram(to_plot[i,:], bins = bins)[0] - i*spacer*np.ones_like(np.histogram(to_plot[i,:], bins = bins)[0]), linewidth = 1)
    #     ax.flatten()[i].plot(X, scipy.ndimage.gaussian_filter1d(np.histogram(to_plot[i,:], bins = bins)[0], smooth_over) - i*spacer, color = 'k')    

    #     # ax.flatten()[i].plot(X, np.insert(np.diff(scipy.ndimage.gaussian_filter1d(np.histogram(to_plot[i,:], bins = 500)[0], smooth_over) - i*spacer), 0 , 0)*25, color = 'c')
    #     # ax.flatten()[i].plot(X[1:], np.insert(np.diff(np.diff(scipy.ndimage.gaussian_filter1d(np.histogram(to_plot[i,:], bins = 500)[0], smooth_over) - i*spacer)), 0 , 0)*200, color = 'green')

    #     ax.flatten()[i].set_yticks([])
    #     ax.flatten()[i].set_xticks([])
    #     ax.flatten()[i].axvline(OFF_ON_thresholds[i], color = 'red')    
    #     if i not in bimodal_channels:
    #         fit = UnivariateSpline(X, scipy.ndimage.gaussian_filter1d(np.histogram(to_plot[i,:], bins = bins)[0], smooth_over),s=0,k=4)
    #         first_dev = scipy.ndimage.gaussian_filter1d(fit.derivative(n=1)(X), 3)
    #         ax.flatten()[i].plot(X, scipy.ndimage.gaussian_filter1d(first_dev, 3), color = 'c')
    #         for peak in scipy.signal.find_peaks(scipy.ndimage.gaussian_filter1d(first_dev[:600], 3), distance = 150, prominence = 15)[0]:
    #             ax.flatten()[i].axvline(X[peak], color = 'green', linestyle = '--', linewidth = 1)
    #     plt.tight_layout()
    # plt.savefig('logMUA threshold for all channels.jpg', dpi = 1000, format = 'jpg')

    # plot MUA power histogram of designated channel smoothed with threshold
    # fig, ax = plt.subplots(4,1)
    # fig.suptitle(f'{day}')
    # ax[0].plot(np.linspace(0, MUA_power_binned_log_smoothed.shape[1], MUA_power_binned_log_smoothed.shape[1])/1000, MUA_power_binned_log_smoothed[chan_for_MUA,:])
    # ax[1].plot(np.linspace(0, LFP.shape[1], LFP.shape[1])/1000, LFP[chan_for_MUA,:])
    # ax[2].hist(MUA_power_binned_log_smoothed[chan_for_MUA,:], bins = 1000)
    # ax[3].plot(np.histogram(MUA_power_binned_log_smoothed[chan_for_MUA,:], bins = 1000)[1][:-1], scipy.ndimage.gaussian_filter1d(np.histogram(MUA_power_binned_log_smoothed[chan_for_MUA,:], bins = 1000)[0], 20), color = 'k')    
    # ax[3].axvline(OFF_ON_threshold, color = 'red')    





    #---------------------------------------------------------------------------- detect ON state starts/ends in all channels individually
    OFF_duration_threshold = 100
    ON_duration_threshold = 100

    ON_states_starts_allchans = []
    ON_states_stops_allchans = []

    for chan in range(32):
        if np.isnan(OFF_ON_thresholds[chan]):
            print(f'{chan} doesnt have two spiking peaks')
            ON_states_starts_allchans.append([])
            ON_states_stops_allchans.append([])
        else:        
            ON_states_starts = np.where(np.diff((MUA_power_binned_log_smoothed[chan,1000:-1000] < OFF_ON_threshold).astype(int)) == -1)[0]/new_fs + 1 # make sure there's enough time at start and end of recording (add 1 second because starts at 1 sec)
            ON_states_stops = np.where(np.diff((MUA_power_binned_log_smoothed[chan,1000:-1000] < OFF_ON_threshold).astype(int)) == 1)[0]/new_fs + 1
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
        
    # # plot all ON states in red
    # fig, ax = plt.subplots(2,1, sharex = True)
    # fig.suptitle(f'{day}')
    # ax[0].plot(np.linspace(0, MUA_power_binned_log_smoothed.shape[1], MUA_power_binned_log_smoothed.shape[1])/1000, MUA_power_binned_log_smoothed[chan_for_MUA,:])
    # ax[1].plot(np.linspace(0, LFP.shape[1], LFP.shape[1])/1000, LFP[chan_for_MUA,:])
    # for ON_start, ON_stop in zip(ON_states_starts_allchans[chan_for_MUA], ON_states_stops_allchans[chan_for_MUA]):
    #     ax[0].axvspan(ON_start, ON_stop, color = 'red', alpha = 0.1)
    #     ax[1].axvspan(ON_start, ON_stop, color = 'red', alpha = 0.1)
    
    pickle.dump(ON_states_starts_allchans, open('ON_states_starts_allchans', 'wb'))      
    pickle.dump(ON_states_stops_allchans, open('ON_states_stops_allchans', 'wb'))      

    pickle.dump(ON_states_starts_avg, open('ON_states_starts_avg', 'wb'))      
    pickle.dump(ON_states_stops_avg, open('ON_states_stops_avg', 'wb'))      

    os.chdir('..')
    os.chdir('..')

    
#%% ---------------------------------------------------------------------- LFP and CSD frequency depth profile and coherence between channels

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

# use days with a nowhisker stim
# for day_ind, day in enumerate(days[0:6]):
for day_ind, day in enumerate(['160614']):
    os.chdir(day)
    print(day)
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
    elif day == '160614':
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]**2
        spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 1.3
    else:
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]**2
        spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 1.3

    nchans = LFP.shape[0]
    
    #plot cortical channels only
    if day == '160614':
        cortical_chans = np.arange(1,28)
    elif day == '160615':
        cortical_chans = np.arange(4,27)
    elif day == '160622':
        cortical_chans = np.arange(3,29)
    elif day == '160728':
        cortical_chans = np.arange(1,24)
    elif day == '160729':
        cortical_chans = np.arange(1,24)
    elif day == '160810':
        cortical_chans = np.arange(1,22)

    # # take log and smooth MUA power across time
    # MUA_power_binned_log_smoothed = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_power_binned), (0,80))
    # MUA_power_binned_smoothed = scipy.ndimage.gaussian_filter(MUA_power_binned, (0,80))

    
    # # normalize within each channel, take median value in each channel
    # MUA_power_binned_rel = (MUA_power_binned.T/np.median(MUA_power_binned, axis = 1)).T
    # MUA_power_binned_log_smoothed_rel = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_power_binned_rel), (0,80))
    
    # calculate CSD across the whole recording, smoothed across channels
    CSD_matrix = -np.eye(nchans) # 
    for j in range(1, CSD_matrix.shape[0] - 1):
        CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
    CSD_all = - np.dot(CSD_matrix, LFP)
    CSD_all_smoothed = - np.dot(CSD_matrix, scipy.ndimage.gaussian_filter(LFP, (2, 0)))
    CSD_all[0,:] = 0
    CSD_all_smoothed[0,:] = 0
    CSD_all[-1,:] = 0
    CSD_all_smoothed[-1,:] = 0
    
    # plot a snippet of MUA power of each channel
    # fig, ax = plt.subplots()
    # for i in range(MUA_power_binned_log_smoothed.shape[0]):
    #     ax.plot(MUA_power_binned_log_smoothed[i,0:20000] - i*np.ones_like(MUA_power_binned_log_smoothed[i,0:20000]), linewidth = 0.5)


    upper_freq_to_plot = 4
    
    # power spectral depth profile for LFP
    LFP_FFT = np.fft.fft(LFP, axis = 1)
    fftfreq = np.fft.fftfreq(LFP.shape[1], d = 1/new_fs)
    freq_indices_to_plot = np.where((0 <= fftfreq) & (fftfreq <= upper_freq_to_plot))[0]
    to_plot_freq = (np.abs(LFP_FFT))[:, freq_indices_to_plot]/1000/len(freq_indices_to_plot)
    # smooth over frequencies, select only cortical channels
    to_plot_freq = scipy.ndimage.gaussian_filter(to_plot_freq[cortical_chans,:], (0,10))
    cmap = plt.get_cmap('jet')
    cNorm = matplotlib.colors.Normalize(vmin=np.min(to_plot_freq), vmax=np.max(to_plot_freq))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

    # interpolation_points = 8*nchans
    # fig, ax = plt.subplots()
    # ax.imshow(interpolate_matrix(to_plot_freq.T, space_interp = interpolation_points), cmap = 'jet', aspect =5, vmax = np.max(to_plot_freq))
    # # ax.set_yticks(np.linspace(interpolation_points - 1, 0, 5))
    # # ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 16)
    # # ax.set_ylabel('depth (mm)', size = 16)
    # ax.set_yticks([])
    # ax.set_xticks(np.linspace(0, freq_indices_to_plot[-1], 5))
    # ax.set_xticklabels(list(map(str, [0,1,2,3,4])), size = 16)
    # ax.set_xlabel('frequency (Hz)', size = 16)
    # bar = fig.colorbar(scalarMap)
    # bar.set_label('LFP power ($\mathregular{mV^2}$/Hz)', size = 20)
    # bar.ax.tick_params(labelsize=20)
    # plt.tight_layout()
    # plt.savefig('LFP freq depth profile.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('LFP freq depth profile.pdf', dpi = 1000, format = 'pdf')


    # power spectral depth profile for CSD
    # if day == '160810':
    #     # remove CSD noise in the white matter
    #     CSD_all_smoothed[-2,:] = 0
    CSD_FFT = np.fft.fft(CSD_all_smoothed, axis = 1)
    fftfreq = np.fft.fftfreq(CSD_all_smoothed.shape[1], d = 1/new_fs)
    freq_indices_to_plot = np.where((0 <= fftfreq) & (fftfreq <= upper_freq_to_plot))[0]
    to_plot_freq = (np.abs(CSD_FFT))[:, freq_indices_to_plot]/1000/len(freq_indices_to_plot)
    # smooth over frequencies
    to_plot_freq = scipy.ndimage.gaussian_filter(to_plot_freq[cortical_chans,:], (0,10))
    cmap = plt.get_cmap('jet')
    cNorm = matplotlib.colors.Normalize(vmin=np.min(to_plot_freq), vmax=np.max(to_plot_freq))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

    interpolation_points = 8*nchans
    fig, ax = plt.subplots()
    ax.imshow(interpolate_matrix(to_plot_freq.T, space_interp = interpolation_points), cmap = 'jet', aspect =5, vmax = np.max(to_plot_freq)*0.9)
    # ax.set_yticks(np.linspace(interpolation_points - 1, 0, 5))
    # ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 16)
    # ax.set_ylabel('depth (mm)', size = 16)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, freq_indices_to_plot[-1], 5))
    ax.set_xticklabels(list(map(str, [0,1,2,3,4])), size = 16)
    ax.set_xlabel('frequency (Hz)', size = 16)
    bar = fig.colorbar(scalarMap)
    bar.set_label('CSD power', size = 20)
    bar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig('CSD freq depth profile.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('CSD freq depth profile.pdf', dpi = 1000, format = 'pdf')

    




    coherence_LFP_matrix = np.load('coherence_LFP_matrix.npy')
    coherence_CSD_matrix = np.load('coherence_CSD_matrix.npy')
    
    # fill in other half of matrix for better visualisation
    coherence_LFP_matrix_1 = copy.deepcopy(coherence_LFP_matrix)
    coherence_LFP_matrix_1[np.isnan(coherence_LFP_matrix_1)] = 0
    coherence_LFP_matrix_1 = coherence_LFP_matrix_1 + np.rot90(np.fliplr(coherence_LFP_matrix_1))
    np.fill_diagonal(coherence_LFP_matrix_1, 1)
    
    coherence_CSD_matrix_1 = copy.deepcopy(coherence_CSD_matrix)
    coherence_CSD_matrix_1[np.isnan(coherence_CSD_matrix_1)] = 0
    coherence_CSD_matrix_1 = coherence_CSD_matrix_1 + np.rot90(np.fliplr(coherence_CSD_matrix_1))
    np.fill_diagonal(coherence_CSD_matrix_1, 1)



    # # Magnitude-squared coherence matrix between channels, average within delta range
    # coherence_freq_lower = 0
    # coherence_freq_upper = 4
    # nperseg = 5000
    # fftfreq = np.fft.fftfreq(nperseg, d = 1/new_fs)
    # freq_indices_for_coherence = np.where((coherence_freq_lower <= fftfreq) & (fftfreq <= coherence_freq_upper))[0]
    
    # coherence_LFP_matrix = np.zeros([nchans, nchans])
    # coherence_CSD_matrix = np.zeros([nchans, nchans])
    # for chan1 in range(nchans):
    #     for chan2 in range(nchans):
    #         if chan2 > chan1:
    #             coherence_LFP_matrix[chan1, chan2] = np.NaN
    #             coherence_CSD_matrix[chan1, chan2] = np.NaN
    #             continue
    #         print(chan1, chan2)
    #         coherence_LFP_matrix[chan1, chan2] = np.mean(scipy.signal.coherence(LFP[chan1,:], LFP[chan2,:], fs = new_fs, nperseg = nperseg)[1][freq_indices_for_coherence])
    #         coherence_CSD_matrix[chan1, chan2] = np.mean(scipy.signal.coherence(CSD_all_smoothed[chan1,:], CSD_all_smoothed[chan2,:], fs = new_fs, nperseg = nperseg)[1][freq_indices_for_coherence])
    # np.save('coherence_LFP_matrix.npy', coherence_LFP_matrix)
    # np.save('coherence_CSD_matrix.npy', coherence_CSD_matrix)



    # cmap = plt.get_cmap('jet')
    # cNorm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    # scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    
    # fig, ax = plt.subplots()
    # # ax.imshow(coherence_LFP_matrix_1[cortical_chans,:][:,cortical_chans], cmap = 'jet')
    # ax.imshow(interpolate_matrix(coherence_LFP_matrix_1, 200, 200), cmap = 'jet')
    # ax.set_yticks([])
    # ax.set_xticks([])
    # bar = fig.colorbar(scalarMap)
    # bar.set_label('LFP inter-site coherence', size = 20)
    # bar.ax.tick_params(labelsize=20)  
    # plt.tight_layout()
    # plt.savefig('LFP coherence.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('LFP coherence.pdf', dpi = 1000, format = 'pdf')

    
    # fig, ax = plt.subplots()
    # ax.imshow(interpolate_matrix(coherence_CSD_matrix_1[cortical_chans,:][:,cortical_chans], 200, 200), cmap = 'jet')
    # # ax.imshow(coherence_CSD_matrix_1, cmap = 'jet')
    # ax.set_yticks([])
    # ax.set_xticks([])
    # bar = fig.colorbar(scalarMap)
    # bar.set_label('CSD inter-site coherence', size = 20)
    # bar.ax.tick_params(labelsize=20)  
    # plt.tight_layout()
    # plt.savefig('CSD coherence.jpg', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD coherence.pdf', dpi = 1000, format = 'pdf')




    os.chdir('..')
    os.chdir('..')





#%% -------------------------------------------------------------------------------- example SOs LFP, MUA and CSD

nchans = 32
b_notch, a_notch = scipy.signal.iirnotch(50, 100, 1000)
b_butter, a_butter = scipy.signal.butter(4, 200, 'highpass', fs = new_fs)

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

# use days with a nowhisker stim
# for day_ind, day in enumerate(days[0:6]):
for day_ind, day in enumerate(['160614']):
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
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:218000]
        spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 2
    elif day == '160614':
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]
        spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 1.3
    else:
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]
        spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 1.3

    # last channel
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
        
    nchans = LFP.shape[0]
    # # take log and smooth MUA power across time
    MUA_power_binned_log_smoothed = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_power_binned**2), (0,80))
    MUA_power_binned_smoothed = scipy.ndimage.gaussian_filter(MUA_power_binned**2, (0,80))
    # # normalize within each channel, take median value in each channel
    MUA_power_binned_rel = ((MUA_power_binned**2).T/np.median(MUA_power_binned**2, axis = 1)).T
    MUA_power_binned_log_smoothed_rel = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_power_binned_rel), (0,80))
    
    #plot cortical channels only
    if day == '160614':
        chans_for_plot = np.arange(1,28)
    elif day == '160615':
        chans_for_plot = np.arange(4,25)
    elif day == '160622':
        chans_for_plot = np.arange(3,29)
    elif day == '160728':
        chans_for_plot = np.arange(0,25)
    elif day == '160729':
        chans_for_plot = np.arange(0,24)
    elif day == '160810':
        chans_for_plot = np.arange(0,21)

    else:
        # chans_to_plot = np.concatenate(layer_list_LFP[0][0])
        chans_for_plot = np.arange(0,32)
        chans_for_plot = np.arange(3,24)
    
    
    # # show one ON state
    # time_start = 101
    # time_stop = 105.7
    # fig, ax = plt.subplots(figsize = (5,10))
    # spacer = 1500
    # for i in chans_for_plot:
    #     ax.plot(scipy.signal.filtfilt(b_notch, a_notch, LFP[i,:]) -  i *spacer, linewidth = 0.5, color = 'k')
    #     ax.set_xlim(time_start*new_fs,time_stop*new_fs)
    #     # ax.set_yticks(ticks = np.linspace(0, 31000, 32), labels = list(np.linspace(0, 31, 32, dtype = int)))
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.savefig('example SOs LFP.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('example SOs LFP.pdf', format = 'pdf', dpi = 1000)

    # fig, ax = plt.subplots(figsize = (5,10))
    # MUA_to_plot = MUA_power_binned*np.abs(MUA_power_binned)
    # spacer = np.max(np.abs(MUA_to_plot[chans_for_plot,int(time_start*new_fs):int(time_stop*new_fs)]))*1.5
    # for i in chans_for_plot:
    #     ax.plot(MUA_to_plot[i,:] -  i *spacer, linewidth = 0.5, color = 'k')
    #     ax.set_xlim(time_start*new_fs,time_stop*new_fs)
    # ax.set_ylim([-(chans_for_plot[-1] + 1)*spacer, -(chans_for_plot[0] - 1)*spacer])
    # # ax.set_yticks(ticks = np.linspace(0, 31000, 32), labels = list(np.linspace(0, 31, 32, dtype = int)))
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.tight_layout()
    # plt.savefig('example SOs MUA.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('example SOs MUA.pdf', format = 'pdf', dpi = 1000)

    
    #show a longer time period as heatmaps
    time_start = 95
    time_stop = 105.7
    fig, ax = plt.subplots(figsize = (5,5))
    to_plot_LFP = scipy.signal.filtfilt(b_notch, a_notch, LFP[chans_for_plot,int(time_start*new_fs):int(time_stop*new_fs)])
    ax.imshow(-to_plot_LFP, cmap = 'jet', aspect = 200)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('example SOs LFP heatmap.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('example SOs LFP heatmap.pdf', format = 'pdf', dpi = 1000)

    fig, ax = plt.subplots(figsize = (2,5))
    norm = colors.Normalize(vmin=np.min(to_plot_LFP)/1000, vmax=np.max(to_plot_LFP)/1000)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap= 'jet'),
                  cax=ax)
    ax.tick_params(axis="y", labelsize=32)    
    # ax.set_yticklabels(list(map(str, np.linspace(-1, 1.5, 6))), size = 16)
    ax.set_ylabel('LFP (mV)', size = 32)
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.savefig('example SOs LFP heatmap legend.pdf', dpi = 1000, format = 'pdf')



    #smooth CSD over time a bit
    CSD_matrix = -np.eye(nchans) # 
    for j in range(1, CSD_matrix.shape[0] - 1):
        CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
    CSD_smoothed = - np.dot(CSD_matrix, scipy.ndimage.gaussian_filter(LFP[:,int(time_start*new_fs):int(time_stop*new_fs)], (2, 10)))
    CSD_smoothed[0,:] = 0
    CSD_smoothed[-1,:] = 0
    # CSD_toplot = interpolate_matrix(CSD_smoothed[chans_for_plot,:].T)
    
    CSD_toplot = CSD_smoothed[chans_for_plot,:]
    minmax = np.max(np.abs(CSD_toplot))
    fig, ax = plt.subplots(figsize = (5,5))
    ax.imshow(CSD_toplot, cmap = 'jet', aspect = 200, vmin = -minmax, vmax = minmax)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('example SOs CSD heatmap.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('example SOs CSD heatmap.pdf', format = 'pdf', dpi = 1000)
    
    fig, ax = plt.subplots(figsize = (2,5))
    norm = colors.Normalize(vmin=-minmax, vmax=minmax)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap= 'jet'),
                  cax=ax)
    ax.tick_params(axis="y", labelsize=32)    
    # ax.set_yticklabels(list(map(str, np.linspace(-1, 1.5, 6))), size = 16)
    ax.set_ylabel('CSD (mV/mm2)', size = 32)
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.savefig('example SOs CSD heatmap legend.pdf', dpi = 1000, format = 'pdf')



    fig, ax = plt.subplots(figsize = (5,5))
    to_plot_MUA = MUA_power_binned_log_smoothed[chans_for_plot,int(time_start*new_fs):int(time_stop*new_fs)]
    ax.imshow(to_plot_MUA, cmap = 'jet', aspect = 200)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('example SOs logMUA heatmap.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('example SOs logMUA heatmap.pdf', format = 'pdf', dpi = 1000)

    fig, ax = plt.subplots(figsize = (2,5))
    norm = colors.Normalize(vmin=0, vmax=np.max(to_plot_MUA))
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap= 'jet'),
                  cax=ax)
    ax.tick_params(axis="y", labelsize=32)    
    # ax.set_yticklabels(list(map(str, np.linspace(-1, 1.5, 6))), size = 16)
    ax.set_ylabel('log(MUA)', size = 32)
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.savefig('example SOs logMUA heatmap legend.pdf', dpi = 1000, format = 'pdf')



    # ax.imshow(interpolate_matrix(CSD_smoothed[:,:].T), cmap = 'jet', aspect = 10)
    # ax.imshow(scipy.signal.filtfilt(b_notch, a_notch, CSD_smoothed[i,:]) +  i *spacer * -np.ones_like(LFP[i,:]), linewidth = 0.5, color = 'k')
    # ax.set_xlim(time_start*new_fs,time_stop*new_fs)
    # ax.set_yticks(ticks = np.linspace(0, 31000, 32), labels = list(np.linspace(0, 31, 32, dtype = int)))
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.set_xticks([])
    # ax.set_yticks([])


    # fig, ax = plt.subplots(figsize = (7,10))
    # spacer = 5
    # for i in range(nchans):
    #     ax.plot(scipy.signal.filtfilt(b_notch, a_notch, MUA_power_binned_log_smoothed_rel[i,:]) +  i *spacer * -np.ones_like(LFP[i,:]), linewidth = 0.5, color = 'k')
    #     ax.set_xlim(time_start*new_fs,time_stop*new_fs)
    #     # ax.set_yticks(ticks = np.linspace(0, 31000, 32), labels = list(np.linspace(0, 31, 32, dtype = int)))
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.set_xticks([])
    #     ax.set_yticks([])

        



    cl()
    os.chdir('..')
    os.chdir('..')

    # plt.savefig('example_LFP', dpi = 1000)



#%% ----------------------------------------------------------------------------------- LFP and CSD amplitudes (RMS) during ON states, PCA and 3D scatter

exclude_outliers = True

# for cond in ['UP_pairing', 'DOWN_pairing']:
    # os.chdir(os.path.join(overall_path, fr'{cond}'))
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

# use days with a nowhisker stim
for day_ind, day in enumerate(days[0:6]):
# for day_ind, day in enumerate(['160810']):
    os.chdir(day)
    print(day)
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
        # spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 2
    elif day == '160614':
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]**2
        # spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 1.3
    else:
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]**2
        # spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 1.3
    
    # # take log and smooth MUA power across time
    MUA_power_binned_log_smoothed = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_power_binned), (0,80))
    # MUA_power_binned_smoothed = scipy.ndimage.gaussian_filter(MUA_power_binned, (0,80))
    
    # # normalize within each channel, take median value in each channel
    MUA_power_binned_rel = (MUA_power_binned.T/np.median(MUA_power_binned, axis = 1)).T
    MUA_power_binned_log_smoothed_rel = scipy.ndimage.gaussian_filter(np.emath.logn(log_base, MUA_power_binned_rel), (0,80))

    nchans = LFP.shape[0]
    # calculate CSD across the whole recording, smoothed across channels
    CSD_matrix = -np.eye(nchans) # 
    for j in range(1, CSD_matrix.shape[0] - 1):
        CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
    CSD_all = - np.dot(CSD_matrix, LFP)
    CSD_all_smoothed = - np.dot(CSD_matrix, scipy.ndimage.gaussian_filter(LFP, (2, 0)))
    CSD_all[0,:] = 0
    CSD_all_smoothed[0,:] = 0
    CSD_all[-1,:] = 0
    CSD_all_smoothed[-1,:] = 0

    ON_states_starts_allchans = pickle.load(open('ON_states_starts_allchans', 'rb'))      
    ON_states_stops_allchans = pickle.load(open('ON_states_stops_allchans', 'rb'))      

    ON_states_starts_avg = pickle.load(open('ON_states_starts_avg', 'rb'))      
    ON_states_stops_avg = pickle.load(open('ON_states_stops_avg', 'rb'))    
    print(len(ON_states_starts_avg))
    # only use ON/OFF states longer than 200ms
    shorts = []
    for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts_avg, ON_states_stops_avg)):
        if ON_stop - ON_start < 200/1000:
            shorts.append(i)
    ON_states_starts_avg = np.delete(ON_states_starts_avg, shorts)
    ON_states_stops_avg = np.delete(ON_states_stops_avg, shorts)  
    # shorts = []
    # for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts_avg[1:], ON_states_stops_avg[:-1])):
    #     if ON_start - ON_stop < 200/1000:
    #         shorts.append(i)
    # ON_states_starts_avg = np.delete(ON_states_starts_avg, shorts)
    # ON_states_stops_avg = np.delete(ON_states_stops_avg, shorts)  


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
        cortical_chans = np.arange(1,28)
    elif day == '160615':
        cortical_chans = np.arange(4,27)
    elif day == '160622':
        cortical_chans = np.arange(3,29)
    elif day == '160728':
        cortical_chans = np.arange(1,24)
    elif day == '160729':
        cortical_chans = np.arange(1,24)
    elif day == '160810':
        cortical_chans = np.arange(1,22)


    MUA_matrix_rel = []
    MUA_matrix = []
    CSD_matrix = []
    LFP_matrix = []
    # for ON_start, ON_end in zip((ON_states_starts_allchans[chan_for_MUA]*1000).astype(int), (ON_states_stops_allchans[chan_for_MUA]*1000).astype(int)):
    for ON_start, ON_end in zip((ON_states_starts_avg*1000).astype(int), (ON_states_stops_avg*1000).astype(int)):
        if ON_end - ON_start < 200:
            continue
        MUA_matrix_rel.append(MUA_power_binned_log_smoothed_rel[:,ON_start - 100:ON_end + 100])
        MUA_matrix.append(MUA_power_binned_log_smoothed[:,ON_start - 100:ON_end + 100])
        CSD_matrix.append(CSD_all_smoothed[:,ON_start - 100:ON_end + 100])
        LFP_matrix.append(LFP[:,ON_start - 100:ON_end + 100])
        
    # take RMS within each channel
    MUA_matrix_rel_rms = [np.sqrt(np.mean(i**2, axis = 1)) for i in MUA_matrix_rel]
    MUA_matrix_rms = [np.sqrt(np.mean(i**2, axis = 1)) for i in MUA_matrix]
    CSD_matrix_rms = [np.sqrt(np.mean(i**2, axis = 1)) for i in CSD_matrix]
    LFP_matrix_rms = [np.sqrt(np.mean(i**2, axis = 1)) for i in LFP_matrix]

    
    # average across channels in a layer (1-5)
    curr_layer_dict = layer_dict[day][0]
    CSD_matrix_rms_layered = []
    LFP_matrix_rms_layered = []
    for layer in range(4):
        CSD_matrix_rms_layered.append([np.mean(i[curr_layer_dict[layer]]) for i in CSD_matrix_rms])
        LFP_matrix_rms_layered.append([np.mean(i[curr_layer_dict[layer]]) for i in LFP_matrix_rms])
    LFP_matrix_rms_layered = np.asarray(LFP_matrix_rms_layered)
    CSD_matrix_rms_layered = np.asarray(CSD_matrix_rms_layered)
    

    to_plot = CSD_matrix_rms_layered
    # take out outliers
    outliers = [[] for i in range(CSD_matrix_rms_layered.shape[0])]
    for i in range(CSD_matrix_rms_layered.shape[0]):
        outliers[i] = np.where(to_plot[i,:] > (np.percentile(to_plot[i,:], 75) + 1.5*(np.abs(np.percentile(to_plot[i,:], 75) - np.percentile(to_plot[i,:], 25)))))[0]
    outliers = np.unique(np.concatenate(outliers))
    if exclude_outliers:
        to_plot = np.delete(to_plot, outliers, axis = 1)

    cmap = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=min(to_plot[0,:]), vmax=max(to_plot[0,:]))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, projection = '3d')
    s = ax.scatter(to_plot[1,:], to_plot[2,:], to_plot[3,:], c=scalarMap.to_rgba(to_plot[0,:]), s = 25)
    scalarMap.set_array(to_plot[:,3])
    ax.tick_params(axis="x", labelsize=20)    
    ax.tick_params(axis="y", labelsize=20) 
    ax.tick_params(axis="z", labelsize=20)  
    ax.set_xlabel('Layer 2/3 RMS of CSD (mV/mm2)', size = 20)
    ax.set_ylabel('Layer 4 RMS of CSD (mV/mm2)', size = 20)
    ax.set_zlabel('Layer 5 RMS of CSD (mV/mm2)', size = 20)
    ax.view_init(elev = 12, azim= -35)
    bar = fig.colorbar(scalarMap, fraction=0.02)
    bar.set_label('Layer 1 RMS of CSD (mV/mm2)', size = 20)
    bar.ax.tick_params(labelsize=20)  
    plt.tight_layout(pad = 2)
    # plt.savefig('CSD ON states all layers.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('CSD ON states all layers.jpg', dpi = 1000, format = 'jpg')
    # fig, ax = plt.subplots(figsize = (5,1.5))
    # norm = colors.Normalize(vmin=np.min(to_plot_traces)/1000, vmax=np.max(to_plot_traces)/1000)
    # fig.colorbar(cm.ScalarMappable(norm=norm, cmap= new_jet),
    #               cax=ax, orientation = 'horizontal')
    # ax.tick_params(axis="x", labelsize=14)    
    # # ax.set_yticklabels(list(map(str, np.linspace(-1, 1.5, 6))), size = 16)
    # ax.set_xlabel('LFP (mV)', size = 18)
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')
    # plt.tight_layout()
    # plt.savefig('LFP colormap legend.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('LFP colormap legend.jpg', dpi = 1000, format = 'jpg')

    
    #plot along PC1 and PC2 axis of RMS in all channels
    # princ_components, eigenvectors, eigenvalues, var_ratio, scaled = PCA_normed(CSD_matrix_rms_layered.T)
    princ_components, eigenvectors, eigenvalues, var_ratio, scaled = PCA_normed(np.asarray(CSD_matrix_rms)[:,cortical_chans])
    fig, ax = plt.subplots()
    ax.scatter(princ_components[:,0], princ_components[:,1], s = 8, color = 'k')    
    ax.set_xlim([-3.3,3.3])
    ax.set_ylim([-2.5,2.5])
    ax.tick_params(axis="x", labelsize=20)    
    ax.tick_params(axis="y", labelsize=20) 
    ax.set_xlabel('PC1', size = 20)
    ax.set_ylabel('PC2', size = 20)
    plt.tight_layout()
    # plt.savefig('CSD ON states all layers PCA.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('CSD ON states all layers PCA.jpg', dpi = 1000, format = 'jpg')
    
    os.chdir('..')
    os.chdir('..')






    
#%%   ------------------------------------------------------------------------------- ON starts with PCA sorting

highpass_cutoff = 4

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for day_ind, day in enumerate(days[0:6]): #nowhisker days
# for day_ind, day in enumerate(['160614']):
    os.chdir(day)
    print(day)
    no_stim_folder = [s for s in os.listdir() if 'nowhisker' in s][0]
    os.chdir(no_stim_folder)
    
    if day == '160615':
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
    elif day == '160614':
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]**2
        spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 1.3
    else:
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]**2
        spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 1.3

    # take out 50Hz
    LFP = scipy.signal.filtfilt(b_notch, a_notch, LFP)
    MUA_power_binned = scipy.signal.filtfilt(b_notch, a_notch, MUA_power_binned)
    
    nchans = LFP.shape[0]
    # take log and smooth MUA power across time
    MUA_power_binned_log_smoothed = scipy.ndimage.gaussian_filter(np.abs(np.emath.logn(log_base, MUA_power_binned)), (0,80))
    MUA_power_binned_smoothed = scipy.ndimage.gaussian_filter(MUA_power_binned, (0,80))
    # normalize within each channel, take median value in each channel
    MUA_power_binned_rel = (MUA_power_binned.T/np.median(MUA_power_binned, axis = 1)).T
    MUA_power_binned_log_smoothed_rel = scipy.ndimage.gaussian_filter(np.abs(np.emath.logn(log_base, MUA_power_binned_rel)), (0,80))
    
    # CSD across the whole recording, smoothed across channels
    CSD_matrix = -np.eye(nchans) # 
    for j in range(1, CSD_matrix.shape[0] - 1):
        CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
    CSD_all = - np.dot(CSD_matrix, LFP)
    CSD_all_smoothed = - np.dot(CSD_matrix, scipy.ndimage.gaussian_filter(LFP, (2, 0)))
    CSD_all[0,:] = 0
    CSD_all_smoothed[0,:] = 0
    CSD_all[-1,:] = 0
    CSD_all_smoothed[-1,:] = 0
    
    # plot a snippet of MUA power of each channel
    # fig, ax = plt.subplots()
    # for i in range(MUA_power_binned_log_smoothed.shape[0]):
    #     ax.plot(MUA_power_binned_log_smoothed[i,0:20000] - i*np.ones_like(MUA_power_binned_log_smoothed[i,0:20000]), linewidth = 0.5)
    
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

    # clean up bimodal channels in some mice
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

    # cortical channels 
    if day == '160614':
        cortical_chans = np.arange(0,20)
    elif day == '160615':
        cortical_chans = np.arange(4,25)
    elif day == '160622':
        cortical_chans = np.arange(3,29)
    elif day == '160728':
        cortical_chans = np.arange(0,25)
    elif day == '160729':
        cortical_chans = np.arange(0,24)
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


    ON_states_starts_allchans = pickle.load(open('ON_states_starts_allchans', 'rb'))      
    ON_states_stops_allchans = pickle.load(open('ON_states_stops_allchans', 'rb'))    

    ON_states_starts_avg = pickle.load(open('ON_states_starts_avg', 'rb'))      
    ON_states_stops_avg = pickle.load(open('ON_states_stops_avg', 'rb'))    

    # for start and stop analysis only use ON/OFF states longer than 200ms
    shorts = []
    for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts_avg, ON_states_stops_avg)):
        if ON_stop - ON_start < 200/1000:
            shorts.append(i)
    ON_states_starts_avg = np.delete(ON_states_starts_avg, shorts)
    ON_states_stops_avg = np.delete(ON_states_stops_avg, shorts)  
    shorts = []
    for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts_avg[1:], ON_states_stops_avg[:-1])):
        if ON_start - ON_stop < 200/1000:
            shorts.append(i)
    ON_states_starts_avg = np.delete(ON_states_starts_avg, shorts)
    ON_states_stops_avg = np.delete(ON_states_stops_avg, shorts)  

    # 1) ----------------------------------------------------------------- LFP, CSD and MUA values in each cortical channel around every ON start and ON end
    MUA_matrix_start_rel_allchans = []
    MUA_matrix_start_allchans = []
    CSD_matrix_start_allchans = []
    LFP_matrix_start_allchans = []
    MUA_matrix_end_rel_allchans = []
    MUA_matrix_end_allchans = []
    CSD_matrix_end_allchans = []
    LFP_matrix_end_allchans = []

    for chan in range(32):
        MUA_matrix_start_rel = []
        MUA_matrix_start = []
        CSD_matrix_start = []
        LFP_matrix_start = []
        MUA_matrix_end_rel = []
        MUA_matrix_end = []
        CSD_matrix_end = []
        LFP_matrix_end = []
        if len(ON_states_starts_allchans[chan]) > 0:
            for ON_start, ON_end in zip((ON_states_starts_allchans[chan]*1000).astype(int), (ON_states_stops_allchans[chan]*1000).astype(int)):
                MUA_matrix_start_rel.append(MUA_power_binned_log_smoothed_rel[:,ON_start - 200:ON_start + 200])
                MUA_matrix_start.append(MUA_power_binned_log_smoothed[:,ON_start - 200:ON_start + 200])
                CSD_matrix_start.append(CSD_all_smoothed[:,ON_start - 200:ON_start + 200])
                LFP_matrix_start.append(LFP[:,ON_start - 200:ON_start + 200])
                
                MUA_matrix_end_rel.append(MUA_power_binned_log_smoothed_rel[:,ON_end - 200:ON_end + 200])
                MUA_matrix_end.append(MUA_power_binned_log_smoothed[:,ON_end - 200:ON_end + 200])
                CSD_matrix_end.append(CSD_all_smoothed[:,ON_end - 200:ON_end + 200])
                LFP_matrix_end.append(LFP[:,ON_end - 200:ON_end + 200])
                
            MUA_matrix_start_rel = np.asarray(MUA_matrix_start_rel)
            MUA_matrix_start = np.asarray(MUA_matrix_start)
            CSD_matrix_start = np.asarray(CSD_matrix_start)
            LFP_matrix_start = np.asarray(LFP_matrix_start)
            MUA_matrix_end_rel = np.asarray(MUA_matrix_end_rel)
            MUA_matrix_end = np.asarray(MUA_matrix_end)
            CSD_matrix_end = np.asarray(CSD_matrix_end)
            LFP_matrix_end = np.asarray(LFP_matrix_end)
        
        MUA_matrix_start_rel_allchans.append(MUA_matrix_start_rel)
        MUA_matrix_start_allchans.append(MUA_matrix_start)
        CSD_matrix_start_allchans.append(CSD_matrix_start)
        LFP_matrix_start_allchans.append(LFP_matrix_start)
        MUA_matrix_end_rel_allchans.append(MUA_matrix_end_rel)
        MUA_matrix_end_allchans.append(MUA_matrix_end)
        CSD_matrix_end_allchans.append(CSD_matrix_end)
        LFP_matrix_end_allchans.append(LFP_matrix_end)

    # plot several MUA starts to get an idea
    # fig, ax = plt.subplots(10,10, figsize = (19,11))
    # for SO_ind, ax1 in enumerate(list(ax.flatten())):
    #     ax1.imshow(MUA_matrix_start_allchans[chan_for_MUA][SO_ind,cortical_chans,:], cmap = 'jet', aspect = 10)
    #     ax1.tick_params(axis="x", labelsize=1)
    #     ax1.tick_params(axis="y", labelsize=1) 
    # plt.tight_layout(pad = 0.001)

    # plot several CSD starts to get an idea
    # if day == '160622':
    #     minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_start_allchans[chan_for_MUA][:,23:25,:], (0,0,10))))/3
    # else:
    #     minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_start_allchans[chan_for_MUA][:,cortical_chans,:], (0,0,10))))/1.5
    # fig, ax = plt.subplots(10,10, figsize = (19,11))
    # for SO_ind, ax1 in enumerate(list(ax.flatten())):
    #     ax1.imshow(scipy.ndimage.gaussian_filter(CSD_matrix_start_allchans[chan_for_MUA][SO_ind,cortical_chans,:], (0,10)), cmap = 'jet', aspect = 10, vmin =  -minmax, vmax = minmax)
    #     ax1.tick_params(axis="x", labelsize=1)
    #     ax1.tick_params(axis="y", labelsize=1) 
    # plt.tight_layout(pad = 0.001)


    # ON start and end on avg logMUA start times:
    MUA_matrix_start_rel_avg = []
    MUA_matrix_start_avg = []
    CSD_matrix_start_avg = []
    LFP_matrix_start_avg = []
    MUA_matrix_end_rel_avg = []
    MUA_matrix_end_avg = []
    CSD_matrix_end_avg = []
    LFP_matrix_end_avg = []
    spikes_start = []
    spikes_end = []
    for ON_start, ON_end in zip((ON_states_starts_avg*1000).astype(int), (ON_states_stops_avg*1000).astype(int)):
        MUA_matrix_start_rel_avg.append(MUA_power_binned_log_smoothed_rel[:,ON_start - 300:ON_start + 300])
        MUA_matrix_start_avg.append(MUA_power_binned_log_smoothed[:,ON_start - 300:ON_start + 300])
        CSD_matrix_start_avg.append(CSD_all_smoothed[:,ON_start - 300:ON_start + 300])
        LFP_matrix_start_avg.append(LFP[:,ON_start - 300:ON_start + 300])
        
        curr_spikes_start = []
        for chan in range(32):
            # curr_spikes = list(spikes.values())[np.argwhere(chanMap_32 == chan)[0][0]]
            curr_spikes = list(spikes.values())[chan]
            on_spikes = curr_spikes[np.searchsorted(curr_spikes, ON_start - 300) : np.searchsorted(curr_spikes, ON_start + 300)] - ON_start #spikes during that ON state, with ON_start = time 0
            # print(on_spikes)
            curr_spikes_start.append(np.histogram(on_spikes, bins = np.linspace(-300, 299, 600))[0])    
        spikes_start.append(curr_spikes_start)

        MUA_matrix_end_rel_avg.append(MUA_power_binned_log_smoothed_rel[:,ON_end - 300:ON_end + 300])
        MUA_matrix_end_avg.append(MUA_power_binned_log_smoothed[:,ON_end - 300:ON_end + 300])
        CSD_matrix_end_avg.append(CSD_all_smoothed[:,ON_end - 300:ON_end + 300])
        LFP_matrix_end_avg.append(LFP[:,ON_end - 300:ON_end + 300])
    MUA_matrix_start_rel_avg = np.asarray(MUA_matrix_start_rel_avg)
    MUA_matrix_start_avg = np.asarray(MUA_matrix_start_avg)
    CSD_matrix_start_avg = np.asarray(CSD_matrix_start_avg)
    LFP_matrix_start_avg = np.asarray(LFP_matrix_start_avg)
    MUA_matrix_end_rel_avg = np.asarray(MUA_matrix_end_rel_avg)
    MUA_matrix_end_avg = np.asarray(MUA_matrix_end_avg)
    CSD_matrix_end_avg = np.asarray(CSD_matrix_end_avg)
    LFP_matrix_end_avg = np.asarray(LFP_matrix_end_avg)

    spikes_start = np.asarray(spikes_start)
    spikes_start = spikes_start[:,chanMap_32,:]
    
    curr_layers_1_avg = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
    fig, ax = plt.subplots()
    for layer in [0,1,2,3]:
        ax.plot(scipy.ndimage.gaussian_filter(np.mean(np.mean(spikes_start, axis = 0)[curr_layers_1_avg[layer],:], axis = 0)*1000, (10)).T)
    ax.set_xlim([250,500])
    
    fig, ax = plt.subplots(figsize = (1.5,12))
    for chan in range(32):
        ax.plot(scipy.ndimage.gaussian_filter(np.mean(spikes_start[:,chan,:], axis = 0)*1000 - 10*chan, 6))
        # ax.plot(np.mean(spikes_start[:,chan,:], axis = 0)*1000 -  np.argwhere(chanMap_32 == chan)[0][0]*50)
        # ax.set_xlim(200*new_fs,220*new_fs)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.set_xticks([])
        # ax.set_yticks([])
    plt.tight_layout()

    # fig, ax = plt.subplots(figsize = (12,2.5))
    # for chan in range(32):
    #     ax.plot(np.mean(MUA_matrix_start_rel_avg[:,chan,:], axis = 0)*1000 - 100*chan)

    # os.chdir('..')
    # os.chdir('..')


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
    # fig.suptitle('CSD starts all chans')
    # ax.imshow(CSD_sink_timecourse.T, cmap = 'bwr', aspect = 2, vmin = -300, vmax = 300)
    
    # take only chans layer 4 to layer 6 and normalize to median layer 5 time
    CSD_sink_timecourse_zeroed = CSD_sink_timecourse.T - np.median(CSD_sink_timecourse[:, layer_dict[day][0][3]], axis = 1)
    CSD_sink_timecourse_for_PCA = CSD_sink_timecourse_zeroed.T[:,PCA_CSD_chans]
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD starts only chans for sorting norm to layer 5 time')
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
    # # ax.set_xlim([-3.3,3.3])
    # # ax.set_ylim([-2.5,2.5])
    # ax.tick_params(axis="x", labelsize=20)    
    # ax.tick_params(axis="y", labelsize=20) 
    # ax.set_xlabel('PC1', size = 20)
    # ax.set_ylabel('PC2', size = 20)
    # plt.tight_layout()

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
        ax1.set_yticks(layer_dict_1[day][3][0:3])
    plt.savefig('CSD starts PCA sorted heatmaps.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('CSD starts PCA sorted heatmaps.pdf', dpi = 1000, format = 'pdf')

    minmax = max([np.max(np.mean(MUA_matrix_start_avg_aligned_to_CSD_start[tiers[ind],:,:][:,cortical_chans,:], axis = 0)) for ind in range(tier_nr)])
    fig, ax = plt.subplots(1,tier_nr)
    fig.suptitle('MUA PCA ordered')
    for ind, ax1 in enumerate(list(ax)):
        ax1.imshow(np.mean(MUA_matrix_start_avg_aligned_to_CSD_start[tiers[ind],:,:], axis = 0), aspect = 40, cmap = 'jet', vmax = minmax)
        ax1.set_yticks(layer_dict_1[day][3][0:3])
    plt.savefig('MUA starts PCA sorted.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('MUA starts PCA sorted.pdf', dpi = 1000, format = 'pdf')

    # # PC1 vs length of UP states
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD PCA ordered next UP')
    # ax.scatter(princ_components[:,0], ON_states_stops_avg - ON_states_starts_avg, s = 8, color = 'k')  
    # # slope, intercept, r, p, std_err = stats.linregress(princ_components[:,0], ON_states_stops_avg - ON_states_starts_avg)
    # # print(f'{r**2} and {p} for {len(princ_components[:,0])} channels')

    # # PC1 vs length of previous DOWN states
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD PCA ordered previous DOWN')
    # ax.scatter(princ_components[1:,0], ON_states_starts_avg[1:] - ON_states_starts_avg[:-1], s = 8, color = 'k')  

    # slope, intercept, r, p, std_err = stats.linregress(X, Y)
    # print(f'{r**2} and {p} for {len(X)} channels')
    # ax.scatter(X,Y, color = 'k')
    # ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')

    


    # #-----------------------------  PCA on the CSD ON flattened start image in cortical channels (gives the exact same thing)
    
    # CSD_matrix_start_avg_aligned_to_CSD_start_flattened = CSD_matrix_start_avg_aligned_to_CSD_start[:,cortical_chans,100:500].reshape(CSD_matrix_start_avg_aligned_to_CSD_start.shape[0], -1)
    # princ_components, eigenvectors, eigenvalues, var_ratio, scaled = PCA_normed(CSD_matrix_start_avg_aligned_to_CSD_start_flattened, components = 4)
    # tier_nr = 4
    # tiers = np.array_split(np.argsort(princ_components[:,0]),tier_nr)
    # #plot on PC scatterplot and take out outliers
    # # ax_PCA = plt.subplot(111, projection='3d')
    # # ax_PCA.scatter(princ_components[:,0], princ_components[:,1], princ_components[:,2], s = 8, color = 'k')  

    # if day == '160622':
    #     minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_start_avg_aligned_to_CSD_start[:,23:25,:], (0,0,10))))/6
    # else:
    #     minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_start_avg_aligned_to_CSD_start[:,cortical_chans,:], (0,0,10))))/3
    # fig, ax = plt.subplots(1,tier_nr)
    # fig.suptitle('CSD PCA ordered flattened')
    # for tier, ax1 in enumerate(list(ax)):
    #     ax1.imshow(np.mean(CSD_matrix_start_avg_aligned_to_CSD_start[tiers[tier],:,:], axis = 0), aspect = 40, cmap = 'jet', vmin =  -minmax, vmax = minmax)

    # # PC1 vs length of UP states
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD PCA ordered flattened next UP')
    # ax.scatter(princ_components[:,0], ON_states_stops_avg - ON_states_starts_avg, s = 8, color = 'k')  

    # # PC1 vs length of preivous DOWN states
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD PCA ordered flattened previous DOWN')
    # ax.scatter(princ_components[1:,0], ON_states_starts_avg[1:] - ON_states_starts_avg[:-1], s = 8, color = 'k')  


    os.chdir('..')
    os.chdir('..')

    # # plot start and stop averages
    # interpolation_length = 200
    # fig, ax = plt.subplots()
    # fig.suptitle(f'rel LFP {day} start')
    # ax.imshow(interpolate_matrix(np.mean(-LFP_matrix_start_allchans[chan_for_MUA][:,cortical_chans,:], axis = 0).T, space_interp = interpolation_length), aspect = 2, cmap = 'jet')
    # # ax.axhline(bimodal_channels[-1]*interpolation_length/32, color = 'k')
    # ax.set_xticks([0,100,200,300])
    # ax.set_xticklabels([-100,0,100,200], size = 20)
    # ax.set_yticks(np.linspace(interpolation_length-1, 0, 5))
    # ax.set_yticklabels(list(map(str, np.linspace(-cortical_chans[-1], cortical_chans[0], 5))), size = 20)
    # plt.tight_layout()
    # plt.savefig('LFP start.jpg', dpi = 1000, format = 'jpg')

    # fig, ax = plt.subplots()
    # fig.suptitle(f'rel MUA {day} start')
    # ax.imshow(interpolate_matrix(np.mean(MUA_matrix_start_rel_allchans[chan_for_MUA][:,cortical_chans,:], axis = 0).T, space_interp = interpolation_length), aspect = 2, cmap = 'jet')
    # # ax.axhline(bimodal_channels[-1]*interpolation_length/32, color = 'k')
    # ax.set_xticks([0,100,200,300])
    # ax.set_xticklabels([-100,0,100,200], size = 20)
    # ax.set_yticks(np.linspace(interpolation_length-1, 0, 5))
    # ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5))), size = 20)
    # plt.tight_layout()
    # plt.savefig('MUA start.jpg', dpi = 1000, format = 'jpg')

    # fig, ax = plt.subplots()
    # fig.suptitle(f'CSD {day} start')
    # ax.imshow(interpolate_matrix(np.mean(CSD_matrix_start_allchans[chan_for_MUA][:,cortical_chans,:], axis = 0).T, space_interp = interpolation_length), aspect = 2, cmap = 'jet', vmin = np.min(np.mean(CSD_matrix_start_allchans[chan_for_MUA], axis = 0)), vmax = np.max(np.mean(CSD_matrix_start_allchans[chan_for_MUA], axis = 0)))
    # # ax.axhline(bimodal_channels[-1]*interpolation_length/32, color = 'k')
    # ax.set_xticks([0,100,200,300])
    # ax.set_xticklabels([-100,0,100,200], size = 20)
    # ax.set_yticks(np.linspace(interpolation_length-1, 0, 5))
    # ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5))), size = 20)
    # plt.tight_layout()
    # plt.savefig('CSD start.jpg', dpi = 1000, format = 'jpg')


    # fig, ax = plt.subplots()
    # fig.suptitle(f'rel LFP {day} end')
    # ax.imshow(interpolate_matrix(np.mean(-LFP_matrix_end_allchans[chan_for_MUA][:,cortical_chans,:], axis = 0).T, space_interp = interpolation_length), aspect = 2, cmap = 'jet')
    # # ax.axhline(bimodal_channels[-1]*interpolation_length/32, color = 'k')
    # ax.set_xticks([0,100,200,300])
    # ax.set_xticklabels([-200,-100,100,200], size = 20)
    # ax.set_yticks(np.linspace(interpolation_length-1, 0, 5))
    # ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5))), size = 20)
    # plt.tight_layout()
    # plt.savefig('LFP end.jpg', dpi = 1000, format = 'jpg')

    # fig, ax = plt.subplots()
    # fig.suptitle(f'rel MUA {day} end')
    # ax.imshow(interpolate_matrix(np.mean(MUA_matrix_end_rel_allchans[chan_for_MUA][:,cortical_chans,:], axis = 0).T, space_interp = interpolation_length), aspect = 2, cmap = 'jet')
    # # ax.axhline(bimodal_channels[-1]*interpolation_length/32, color = 'k')
    # ax.set_xticks([0,100,200,300])
    # ax.set_xticklabels([-200,-100,100,200], size = 20)
    # ax.set_yticks(np.linspace(interpolation_length-1, 0, 5))
    # ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5))), size = 20)
    # plt.tight_layout()
    # plt.savefig('MUA end.jpg', dpi = 1000, format = 'jpg')

    # fig, ax = plt.subplots()
    # fig.suptitle(f'CSD {day} end')
    # ax.imshow(interpolate_matrix(np.mean(CSD_matrix_end_allchans[chan_for_MUA][:,cortical_chans,:], axis = 0).T, space_interp = interpolation_length), aspect = 2, cmap = 'jet', vmin = np.min(np.mean(CSD_matrix_start_allchans[chan_for_MUA], axis = 0)), vmax = np.max(np.mean(CSD_matrix_start_allchans[chan_for_MUA], axis = 0)))
    # # ax.axhline(bimodal_channels[-1]*interpolation_length/32, color = 'k')
    # ax.set_xticks([0,100,200,300])
    # ax.set_xticklabels([-200,-100,100,200], size = 20)
    # ax.set_yticks(np.linspace(interpolation_length-1, 0, 5))
    # ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5))), size = 20)
    # plt.tight_layout()
    # plt.savefig('CSD end.jpg', dpi = 1000, format = 'jpg')



#%%--------------------------------------------------------------------------------- ON end with PCA sorting

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for day_ind, day in enumerate(days[0:6]): #nowhisker days
# for day_ind, day in enumerate(['160810']):
    os.chdir(day)
    print(day)
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
    elif day == '160614':
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]**2
        spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 1.3
    else:
        LFP = pickle.load(open('LFP_resampled_nostim','rb'))[chanMap_32,:]
        MUA_power_binned = pickle.load(open('MUA_power_binned','rb'))[chanMap_32,:]**2
        spikes = pickle.load(open(f'spikes_nostim_{highpass_cutoff}','rb'))
        log_base = 1.3

    # take out 50Hz
    LFP = scipy.signal.filtfilt(b_notch, a_notch, LFP)
    MUA_power_binned = scipy.signal.filtfilt(b_notch, a_notch, MUA_power_binned)
    
    nchans = LFP.shape[0]
    # take log and smooth MUA power across time
    MUA_power_binned_log_smoothed = scipy.ndimage.gaussian_filter(np.abs(np.emath.logn(log_base, MUA_power_binned)), (0,80))
    MUA_power_binned_smoothed = scipy.ndimage.gaussian_filter(MUA_power_binned, (0,80))
    # normalize within each channel, take median value in each channel
    MUA_power_binned_rel = (MUA_power_binned.T/np.median(MUA_power_binned, axis = 1)).T
    MUA_power_binned_log_smoothed_rel = scipy.ndimage.gaussian_filter(np.abs(np.emath.logn(log_base, MUA_power_binned_rel)), (0,80))
    
    # CSD across the whole recording, smoothed across channels
    CSD_matrix = -np.eye(nchans) # 
    for j in range(1, CSD_matrix.shape[0] - 1):
        CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
    CSD_all = - np.dot(CSD_matrix, LFP)
    CSD_all_smoothed = - np.dot(CSD_matrix, scipy.ndimage.gaussian_filter(LFP, (2, 0)))
    CSD_all[0,:] = 0
    CSD_all_smoothed[0,:] = 0
    CSD_all[-1,:] = 0
    CSD_all_smoothed[-1,:] = 0
    
    # plot a snippet of MUA power of each channel
    # fig, ax = plt.subplots()
    # for i in range(MUA_power_binned_log_smoothed.shape[0]):
    #     ax.plot(MUA_power_binned_log_smoothed[i,0:20000] - i*np.ones_like(MUA_power_binned_log_smoothed[i,0:20000]), linewidth = 0.5)
    
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

    # clean up bimodal channels in some mice
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

    # cortical channels 
    if day == '160614':
        cortical_chans = np.arange(0,20)
    elif day == '160615':
        cortical_chans = np.arange(4,25)
    elif day == '160622':
        cortical_chans = np.arange(3,29)
    elif day == '160728':
        cortical_chans = np.arange(0,25)
    elif day == '160729':
        cortical_chans = np.arange(0,24)
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


    ON_states_starts_allchans = pickle.load(open('ON_states_starts_allchans', 'rb'))      
    ON_states_stops_allchans = pickle.load(open('ON_states_stops_allchans', 'rb'))    

    ON_states_starts_avg = pickle.load(open('ON_states_starts_avg', 'rb'))      
    ON_states_stops_avg = pickle.load(open('ON_states_stops_avg', 'rb'))    

    # for start and stop analysis only use ON/OFF states longer than 200ms
    shorts = []
    for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts_avg, ON_states_stops_avg)):
        if ON_stop - ON_start < 200/1000:
            shorts.append(i)
    ON_states_starts_avg = np.delete(ON_states_starts_avg, shorts)
    ON_states_stops_avg = np.delete(ON_states_stops_avg, shorts)  
    shorts = []
    for i, (ON_start, ON_stop) in enumerate(zip(ON_states_starts_avg[1:], ON_states_stops_avg[:-1])):
        if ON_start - ON_stop < 200/1000:
            shorts.append(i)
    ON_states_starts_avg = np.delete(ON_states_starts_avg, shorts)
    ON_states_stops_avg = np.delete(ON_states_stops_avg, shorts)  


    # 1) ----------------------------------------------------------------- LFP, CSD and MUA values in each cortical channel around every ON start and ON end
    MUA_matrix_end_rel_allchans = []
    MUA_matrix_end_allchans = []
    CSD_matrix_end_allchans = []
    LFP_matrix_end_allchans = []

    for chan in range(32):
        MUA_matrix_end_rel = []
        MUA_matrix_end = []
        CSD_matrix_end = []
        LFP_matrix_end = []
        if len(ON_states_starts_allchans[chan]) > 0:
            for ON_start, ON_end in zip((ON_states_starts_allchans[chan]*1000).astype(int), (ON_states_stops_allchans[chan]*1000).astype(int)):
                MUA_matrix_end_rel.append(MUA_power_binned_log_smoothed_rel[:,ON_end - 200:ON_end + 200])
                MUA_matrix_end.append(MUA_power_binned_log_smoothed[:,ON_end - 200:ON_end + 200])
                CSD_matrix_end.append(CSD_all_smoothed[:,ON_end - 200:ON_end + 200])
                LFP_matrix_end.append(LFP[:,ON_end - 200:ON_end + 200])
            MUA_matrix_end_rel = np.asarray(MUA_matrix_end_rel)
            MUA_matrix_end = np.asarray(MUA_matrix_end)
            CSD_matrix_end = np.asarray(CSD_matrix_end)
            LFP_matrix_end = np.asarray(LFP_matrix_end)
        
        MUA_matrix_end_rel_allchans.append(MUA_matrix_end_rel)
        MUA_matrix_end_allchans.append(MUA_matrix_end)
        CSD_matrix_end_allchans.append(CSD_matrix_end)
        LFP_matrix_end_allchans.append(LFP_matrix_end)

    # plot several MUA ends to get an idea
    # fig, ax = plt.subplots(10,10, figsize = (19,11))
    # for SO_ind, ax1 in enumerate(list(ax.flatten())):
    #     ax1.imshow(MUA_matrix_start_allchans[chan_for_MUA][SO_ind,cortical_chans,:], cmap = 'jet', aspect = 10)
    #     ax1.tick_params(axis="x", labelsize=1)
    #     ax1.tick_params(axis="y", labelsize=1) 
    # plt.tight_layout(pad = 0.001)

    # plot several CSD ends to get an idea
    # if day == '160622':
    #     minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_start_allchans[chan_for_MUA][:,23:25,:], (0,0,10))))/3
    # else:
    #     minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_start_allchans[chan_for_MUA][:,cortical_chans,:], (0,0,10))))/1.5
    # fig, ax = plt.subplots(10,10, figsize = (19,11))
    # for SO_ind, ax1 in enumerate(list(ax.flatten())):
    #     ax1.imshow(scipy.ndimage.gaussian_filter(CSD_matrix_start_allchans[chan_for_MUA][SO_ind,cortical_chans,:], (0,10)), cmap = 'jet', aspect = 10, vmin =  -minmax, vmax = minmax)
    #     ax1.tick_params(axis="x", labelsize=1)
    #     ax1.tick_params(axis="y", labelsize=1) 
    # plt.tight_layout(pad = 0.001)


    # ON start and end on avg logMUA start times:
    MUA_matrix_end_rel_avg = []
    MUA_matrix_end_avg = []
    CSD_matrix_end_avg = []
    LFP_matrix_end_avg = []
    for ON_start, ON_end in zip((ON_states_starts_avg*1000).astype(int), (ON_states_stops_avg*1000).astype(int)):
        MUA_matrix_end_rel_avg.append(MUA_power_binned_log_smoothed_rel[:,ON_end - 300:ON_end + 300])
        MUA_matrix_end_avg.append(MUA_power_binned_log_smoothed[:,ON_end - 300:ON_end + 300])
        CSD_matrix_end_avg.append(CSD_all_smoothed[:,ON_end - 300:ON_end + 300])
        LFP_matrix_end_avg.append(LFP[:,ON_end - 300:ON_end + 300])
    MUA_matrix_end_rel_avg = np.asarray(MUA_matrix_end_rel_avg)
    MUA_matrix_end_avg = np.asarray(MUA_matrix_end_avg)
    CSD_matrix_end_avg = np.asarray(CSD_matrix_end_avg)
    LFP_matrix_end_avg = np.asarray(LFP_matrix_end_avg)



    # --------------------------------------------------------- plot 100 ON ends
    
    MUAmax = np.max(scipy.ndimage.gaussian_filter(MUA_matrix_end_avg[:,cortical_chans,:], (0,0,10)))
    fig, ax = plt.subplots(10,10, figsize = (19,11))
    for SO_ind, ax1 in enumerate(list(ax.flatten())):
        ax1.imshow(scipy.ndimage.gaussian_filter(MUA_matrix_end_avg[SO_ind,:,:], (0,10)), cmap = 'jet', aspect = 10, vmax = MUAmax) # 
        ax1.tick_params(axis="x", labelsize=1)
        ax1.tick_params(axis="y", labelsize=1) 
    plt.tight_layout(pad = 0.001)
    plt.savefig('MUA ON ends 100.jpg', dpi = 1000, format = 'jpg')

                
    if day == '160622':
        minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_end_avg[:,23:25,:], (0,0,10))))/6
    else:
        minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_end_avg[:,cortical_chans,:], (0,0,10))))/3
    CSD_source_timecourse = []
    fig, ax = plt.subplots(10,10, figsize = (19,11))
    for SO_ind, ax1 in enumerate(list(ax.flatten())):
        ax1.imshow(scipy.ndimage.gaussian_filter(CSD_matrix_end_avg[SO_ind,:,:], (0,10)), cmap = 'jet', aspect = 10, vmin =  -minmax, vmax = minmax) #
        
        CSD_for_gradient = copy.deepcopy(scipy.ndimage.gaussian_filter(CSD_matrix_end_avg[SO_ind,:,:500], (0,10))) # start 300ms before end of MUA to 150ms after
        CSD_for_gradient[CSD_for_gradient < 0] = 0 #has to be a source so positive
        curr_CSD_crossing = []
        for chan in range(nchans):   
            curr_CSD_crossing.append(np.argwhere(CSD_for_gradient[chan,:] >= np.max(CSD_for_gradient[chan,:])*0.75)[0][0]) 
        CSD_source_timecourse.append(np.asarray(curr_CSD_crossing))
        
        ax1.plot(np.asarray(curr_CSD_crossing)[PCA_CSD_chans], np.arange(PCA_CSD_chans[0], PCA_CSD_chans[-1]+1, 1), color = 'white')
        ax1.tick_params(axis="x", labelsize=1)
        ax1.tick_params(axis="y", labelsize=1) 
    plt.tight_layout(pad = 0.001)
    plt.savefig('CSD ON ends 100.jpg', dpi = 1000, format = 'jpg')
    
    
    # CSD source starts across channels
    CSD_source_timecourse = []
    for SO_ind, SO in enumerate(list(range(CSD_matrix_end_avg.shape[0]))):
        CSD_for_gradient = copy.deepcopy(scipy.ndimage.gaussian_filter(CSD_matrix_end_avg[SO_ind,:,:500], (0,10))) # start 300ms before end of MUA to 150ms after
        CSD_for_gradient[CSD_for_gradient < 0] = 0 #has to be a source so positive
        curr_CSD_crossing = []
        for chan in range(nchans):   
            curr_CSD_crossing.append(np.argwhere(CSD_for_gradient[chan,:] >= np.max(CSD_for_gradient[chan,:])*0.75)[0][0] - 300) 
        CSD_source_timecourse.append(np.asarray(curr_CSD_crossing))
    CSD_source_timecourse = np.asarray(CSD_source_timecourse)

    # extract maps aligned to median L5 CSD sink start time as used for PCA to average
    CSD_matrix_end_avg_aligned_to_CSD_end = []
    MUA_matrix_end_avg_aligned_to_CSD_end = []
    for SO_ind, ON_end in enumerate(list(ON_states_stops_avg*1000)):
        curr_ON_end = int(ON_end) + int(np.median(CSD_source_timecourse[SO_ind, layer_dict[day][0][3]]))
        CSD_matrix_end_avg_aligned_to_CSD_end.append(CSD_all_smoothed[:,int(curr_ON_end - 300):int(curr_ON_end + 300)])
        MUA_matrix_end_avg_aligned_to_CSD_end.append(MUA_power_binned_log_smoothed[:,int(curr_ON_end - 300):int(curr_ON_end + 300)])
    CSD_matrix_end_avg_aligned_to_CSD_end = np.asarray(CSD_matrix_end_avg_aligned_to_CSD_end)
    MUA_matrix_end_avg_aligned_to_CSD_end = np.asarray(MUA_matrix_end_avg_aligned_to_CSD_end)




    fig, ax = plt.subplots()
    fig.suptitle('CSD starts')
    ax.imshow(CSD_source_timecourse.T, cmap = 'bwr', aspect = 2, vmin = -300, vmax = 300)
    
    # take only chans layer 4 to layer 6 and normalize to median layer 5 time
    CSD_source_timecourse_zeroed = CSD_source_timecourse.T - np.median(CSD_source_timecourse[:, layer_dict[day][0][3]], axis = 1)
    CSD_source_timecourse_for_PCA = CSD_source_timecourse_zeroed.T[:,PCA_CSD_chans]
    # fig, ax = plt.subplots()
    # fig.suptitle('CSD starts only chans for sorting')
    # ax.imshow(CSD_source_timecourse_for_PCA.T, cmap = 'bwr', aspect = 2, vmin = -300, vmax = 300)

    princ_components, eigenvectors, eigenvalues, var_ratio, scaled = PCA_normed(CSD_source_timecourse_for_PCA)
    # # plot ordered by first PC:
    fig, ax = plt.subplots()
    fig.suptitle('CSD end ordered')
    ax.imshow(CSD_source_timecourse_for_PCA[np.argsort(princ_components[:,0]),:].T, cmap = 'bwr', aspect = 4, vmin = -300, vmax = 300)
    plt.savefig('CSD ends PCA sorted.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('CSD ends PCA sorted.pdf', dpi = 1000, format = 'pdf')
    
    # #plot on PC1 vs PC2 scatterplot and take out outliers
    fig, ax = plt.subplots()
    ax.scatter(princ_components[:,0], princ_components[:,1], s = 8, color = 'k')  
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
        minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_end_avg_aligned_to_CSD_end[:,23:25,:], (0,0,10))))/6
    else:
        minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_end_avg_aligned_to_CSD_end[:,cortical_chans,:], (0,0,10))))/3
    fig, ax = plt.subplots(1,tier_nr)
    fig.suptitle('CSD PCA ordered')
    for tier, ax1 in enumerate(list(ax)):
        ax1.imshow(np.mean(CSD_matrix_end_avg_aligned_to_CSD_end[tiers[tier],:,:], axis = 0), aspect = 40, cmap = 'jet', vmin =  -minmax, vmax = minmax)
    plt.savefig('CSD ends PCA sorted.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('CSD ends PCA sorted.pdf', dpi = 1000, format = 'pdf')

    
    minmax = max([np.max(np.mean(MUA_matrix_end_avg_aligned_to_CSD_end[tiers[ind],:,:][:,cortical_chans,:], axis = 0)) for ind in range(tier_nr)])
    fig, ax = plt.subplots(1,tier_nr)
    fig.suptitle('MUA PCA ordered')
    for ind, ax1 in enumerate(list(ax)):
        ax1.imshow(np.mean(MUA_matrix_end_avg_aligned_to_CSD_end[tiers[ind],:,:], axis = 0), aspect = 40, cmap = 'jet', vmax = minmax)
    plt.savefig('MUA ends PCA sorted.jpg', dpi = 1000, format = 'jpg')
    plt.savefig('MUA ends PCA sorted.pdf', dpi = 1000, format = 'pdf')

    
    
    
    # PCA on the CSD ON flattened end image in cortical channels
    CSD_matrix_end_avg_aligned_to_CSD_end_flattened = CSD_matrix_end_avg_aligned_to_CSD_end[:,cortical_chans,:500].reshape(CSD_matrix_end_avg_aligned_to_CSD_end.shape[0], -1)
    princ_components, eigenvectors, eigenvalues, var_ratio, scaled = PCA_normed(CSD_matrix_end_avg_aligned_to_CSD_end_flattened, components = 4)
    tier_nr = 4
    tiers = np.array_split(np.argsort(princ_components[:,0]),tier_nr)
    #plot on PC scatterplot and take out outliers
    # ax_PCA = plt.subplot(111, projection='3d')
    # ax_PCA.scatter(princ_components[:,0], princ_components[:,1], princ_components[:,2], s = 8, color = 'k')  

    if day == '160622':
        minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_end_avg_aligned_to_CSD_end[:,23:25,:], (0,0,10))))/6
    else:
        minmax = np.max(np.abs(scipy.ndimage.gaussian_filter(CSD_matrix_end_avg_aligned_to_CSD_end[:,cortical_chans,:], (0,0,10))))/3
    fig, ax = plt.subplots(1,tier_nr)
    fig.suptitle('CSD PCA ordered flattened')
    for tier, ax1 in enumerate(list(ax)):
        ax1.imshow(np.mean(CSD_matrix_end_avg_aligned_to_CSD_end[tiers[tier],:,:], axis = 0), aspect = 40, cmap = 'jet', vmin =  -minmax, vmax = minmax)

    minmax = max([np.max(np.mean(MUA_matrix_end_avg_aligned_to_CSD_end[tiers[ind],:,:][:,cortical_chans,:], axis = 0)) for ind in range(tier_nr)])
    fig, ax = plt.subplots(1,tier_nr)
    fig.suptitle('MUA PCA ordered flattened')
    for ind, ax1 in enumerate(list(ax)):
        ax1.imshow(np.mean(MUA_matrix_end_avg_aligned_to_CSD_end[tiers[ind],:,:], axis = 0), aspect = 40, cmap = 'jet', vmax = minmax)

    
    
    
    
    os.chdir('..')
    os.chdir('..')




#%% ----------------------------------------------------------------------------- extract spike-LFP  phase coupling with hilbert transform

lowpass_filtering = 2

exclude_before = 0.1*new_fs
exclude_after = 1*new_fs

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# use days with a nowhisker stim
# for day_ind, day in enumerate(days[0:6]):
for day_ind, day in enumerate(['160810']):
    os.chdir(day)
    print(day)
    
    if day == '160810':
        highpass_cutoff = 5
    elif day == '160514':
        highpass_cutoff = 5
    else:
        highpass_cutoff = 5

    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    MUA_all_sweeps = pickle.load(open('MUA_all_sweeps','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    
    nchans = LFP_all_sweeps[0].shape[0]
    
    spike_phase_allsweeps = []
    for sweep in range(10):
        print(sweep)

        curr_LFP = LFP_all_sweeps[sweep][chanMap_32,:]
        nchans = curr_LFP.shape[0]
        spikes = [list(spikes_allsweeps[sweep].values())[chan] for chan in chanMap_32]
        stims = stim_times[sweep]
        #lowpass filter
        LFP_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(curr_LFP), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = lowpass_filtering*pq.Hz).as_array().T
        
        # #hilbert transform and get angle
        LFP_angle = np.angle(scipy.signal.hilbert(LFP_filt))
        #shift by pi
        neg_ind = LFP_angle<0
        pos_ind = LFP_angle>0
        LFP_angle[neg_ind] = LFP_angle[neg_ind] + np.pi
        LFP_angle[pos_ind] = LFP_angle[pos_ind] - np.pi

        fig, ax = plt.subplots()
        ax.plot(LFP_angle[10,:10000]*10)
        ax.plot(LFP_filt[10,:10000])
        
        # fig, ax = plt.subplots(figsize = (10,3))
        # ax.plot(np.sin(np.arange(np.pi/2.1, 2.52*np.pi, 0.001)), color = 'k')
        # ax.set_yticks([])
        # ax.set_xticks([])
        # plt.savefig('SO example phase.pdf', format = 'pdf', dpi = 1000)
        
        # take out spikes during stim period
        spikes_nostim = []
        inter_stim_indcs = []
        for stim in stims:
            inter_stim_indcs.append(np.linspace(int(stim + exclude_after), int(stim + (5000-exclude_before)), int((5000 - exclude_after - exclude_before + 1))).astype(int))
            for chan in range(32):
                spikes[chan] = np.delete(spikes[chan], np.argwhere(np.logical_and(spikes[chan] < stim + exclude_after, spikes[chan] > stim - exclude_before))) 
        inter_stim_indcs = np.concatenate(inter_stim_indcs)

        # extract hilbert phase of each spike to layer 5 LFP
        spike_phase = []
        for chan  in range(32):
            spike_phase.append(LFP_angle[chan, spikes[chan].astype(int)])
        spike_phase_allsweeps.append(spike_phase)
        
    pickle.dump(spike_phase_allsweeps, open('spike_phase_allsweeps.pkl', 'wb'))
    
    curr_layers = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
    curr_layers = layer_dict[day][0][1:]
    bins = 100
    fig, ax = plt.subplots()
    ax.hist(np.concatenate([np.concatenate([i[j] for j in curr_layers[0]]) for i in spike_phase_allsweeps]), bins = bins, density = True, alpha = 0.25, color = 'blue')
    ax.hist(np.concatenate([np.concatenate([i[j] for j in curr_layers[1]]) for i in spike_phase_allsweeps]), bins = bins, density = True, alpha = 0.25, color = 'green')
    ax.hist(np.concatenate([np.concatenate([i[j] for j in curr_layers[2]]) for i in spike_phase_allsweeps]), bins = bins, density = True, alpha = 0.25, color = 'red')

    os.chdir('..')

#%% ----------------------------------------------------------------------------- spike phase coupling across layers and across mice

nchans = 32

mean_angle_L2_all = []
mean_angle_L4_all = []
mean_angle_L5_all = []
peak_angle_L2_all = []
peak_angle_L4_all = []
peak_angle_L5_all = []
resultant_length_L2_all = []
resultant_length_L4_all = []
resultant_length_L5_all = []
resultant_length_L2_peak_all = []
resultant_length_L4_peak_all = []
resultant_length_L5_peak_all = []

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# use days with a nowhisker stim
# for day_ind, day in enumerate(days[0:6]):
for day_ind, day in enumerate(['160622']):
    os.chdir(day)
    print(day)
    spike_phase_allsweeps = pickle.load(open('spike_phase_allsweeps.pkl', 'rb'))
    curr_layers = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
    curr_layers = layer_dict[day][0][1:]
    
    L2_concat = np.concatenate([np.concatenate([i[j] for j in curr_layers[0]]) for i in spike_phase_allsweeps])
    L4_concat = np.concatenate([np.concatenate([i[j] for j in curr_layers[1]]) for i in spike_phase_allsweeps])
    L5_concat = np.concatenate([np.concatenate([i[j] for j in curr_layers[2]]) for i in spike_phase_allsweeps])
    
    # probability density function
    bins = 100
    fig, ax = plt.subplots(figsize = (7,3.5))
    ax.hist(L2_concat, bins = bins, density = True, alpha = 0.25, color = 'blue')
    ax.hist(L4_concat, bins = bins, density = True, alpha = 0.25, color = 'green')
    ax.hist(L5_concat, bins = bins, density = True, alpha = 0.25, color = 'red')
    ax.set_xticks([-np.pi,-np.pi/2, 0, np.pi/2, np.pi])
    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    ax.set_xlabel('LFP phase', size = 20)
    ax.set_ylabel('spiking PDF', size = 20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('spike phase PDF.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('spike phase PDF.pdf', format = 'pdf', dpi = 1000)
    
    
    # probability density function
    bins = 100
    fig, ax = plt.subplots(figsize = (7,3.5))
    y, binEdges = np.histogram(L2_concat, bins=bins)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    ax.plot(bincenters, y/1200, color = 'blue') # an average of about 1200 ON state across all sweeps during interstim periods
    y, binEdges = np.histogram(L4_concat, bins=bins)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    ax.plot(bincenters, y/1200, color = 'green')
    y, binEdges = np.histogram(L5_concat, bins=bins)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    ax.plot(bincenters, y/1200, color = 'red')
    ax.set_xticks([-np.pi,-np.pi/2, 0, np.pi/2, np.pi])
    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    ax.set_xlabel('LFP phase', size = 20)
    ax.set_ylabel('spiking probability', size = 20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('spike phase histogram.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('spike phase histogram.pdf', format = 'pdf', dpi = 1000)


    mean_angle_L2 = -pycircstat.mean(L2_concat)
    mean_angle_L4 = -pycircstat.mean(L4_concat)
    mean_angle_L5 = -pycircstat.mean(L5_concat)
    peak_angle_L2 = -np.histogram(L2_concat, bins = bins)[1][np.argmax(np.histogram(L2_concat, bins = bins)[0])]
    peak_angle_L4 = -np.histogram(L4_concat, bins = bins)[1][np.argmax(np.histogram(L4_concat, bins = bins)[0])]
    peak_angle_L5 = -np.histogram(L5_concat, bins = bins)[1][np.argmax(np.histogram(L5_concat, bins = bins)[0])]
    resultant_length_L2 = pycircstat.resultant_vector_length(L2_concat)
    resultant_length_L4 = pycircstat.resultant_vector_length(L4_concat)
    resultant_length_L5 = pycircstat.resultant_vector_length(L5_concat)
    resultant_length_L2_peak = np.max(np.histogram(L2_concat, bins = bins, density = True)[0])
    resultant_length_L4_peak = np.max(np.histogram(L4_concat, bins = bins, density = True)[0])
    resultant_length_L5_peak = np.max(np.histogram(L5_concat, bins = bins, density = True)[0])

    # #compass plot with resultant vector
    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize = (3,3))
    # # fig.suptitle(f'{day} angle')
    # ax.plot([0, peak_angle_L2], [0,resultant_length_L2], linewidth = 3, color = 'blue')#
    # ax.plot([0, peak_angle_L4], [0,resultant_length_L4], linewidth = 3, color = 'green')#
    # ax.plot([0, peak_angle_L5], [0,resultant_length_L5], linewidth = 3, color = 'red')#
    # ax.set_ylim([0,1])
    # ax.set_yticklabels([])
    
    mean_angle_L2_all.append(mean_angle_L2)
    mean_angle_L4_all.append(mean_angle_L4)
    mean_angle_L5_all.append(mean_angle_L5)
    peak_angle_L2_all.append(peak_angle_L2)
    peak_angle_L4_all.append(peak_angle_L4)
    peak_angle_L5_all.append(peak_angle_L5)
    resultant_length_L2_all.append(resultant_length_L2)
    resultant_length_L4_all.append(resultant_length_L4)
    resultant_length_L5_all.append(resultant_length_L5)
    resultant_length_L2_peak_all.append(resultant_length_L2_peak)
    resultant_length_L4_peak_all.append(resultant_length_L4_peak)
    resultant_length_L5_peak_all.append(resultant_length_L5_peak)

    os.chdir('..')

    
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(projection='polar')
# theta = np.mean(peak_angle_L2_all)
# r = np.mean(resultant_length_L2_peak_all)
# ax.errorbar(theta, r, xerr=np.std(peak_angle_L2_all)/np.sqrt(6), yerr=np.std(resultant_length_L2_all)/np.sqrt(6), capsize=7, fmt="o", c="blue", markersize = 8, linewidth = 3, capthick = 3)
# theta = np.mean(peak_angle_L4_all)
# r = np.mean(resultant_length_L4_peak_all)
# ax.errorbar(theta, r, xerr=np.std(peak_angle_L4_all)/np.sqrt(6), yerr=np.std(resultant_length_L4_all)/np.sqrt(6), capsize=7, fmt="o", c="green", markersize = 8, linewidth = 3, capthick = 3)
# theta = np.mean(peak_angle_L5_all)
# r = np.mean(resultant_length_L5_peak_all)
# ax.errorbar(theta, r, xerr=np.std(peak_angle_L5_all)/np.sqrt(6), yerr=np.std(resultant_length_L5_all)/np.sqrt(6), capsize=7, fmt="o", c="red", markersize = 8, linewidth = 3, capthick = 3)
# ax.set_ylim([0,1])
# ax.set_yticklabels([])
# plt.savefig('spike-phase coupling polar per layer.jpg', format = 'jpg', dpi = 1000)
# plt.savefig('spike-phase coupling polar per layer.pdf', format = 'pdf', dpi = 1000)

# ax.set_xticks([0, np.pi/2, np.pi, -np.pi/2])
# ax.set_xticklabels([])

# UP_before_resultant = np.asarray([np.nanmean(i[0:4]) for i in resultant_all_mice_all_sweeps_UP_DOWN[0]])/4
# UP_before_angle = np.asarray([pycircstat.mean(i[0:4]) for i in angle_all_mice_all_sweeps_UP_DOWN[0]])
# UP_after_resultant = np.asarray([np.nanmean(i[4:]) for i in resultant_all_mice_all_sweeps_UP_DOWN[0]])/4
# UP_after_angle = np.asarray([pycircstat.mean(i[4:]) for i in angle_all_mice_all_sweeps_UP_DOWN[0]])
# U_b = ax.errorbar(pycircstat.mean(UP_before_angle), np.mean(UP_before_resultant)*100, xerr = pycircstat.std(UP_before_angle)/np.sqrt(len(mice_UP)), yerr=np.mean(UP_before_resultant)*100/np.sqrt(len(mice_UP)), c="black", linewidth = 4, capsize = 10, marker = 'o', markersize = 15)
# U_d = ax.errorbar(pycircstat.mean(UP_after_angle), np.mean(UP_after_resultant)*100, xerr = pycircstat.std(UP_after_angle)/np.sqrt(len(mice_UP)), yerr=np.mean(UP_after_resultant)*100/np.sqrt(len(mice_UP)), c="cyan", linewidth = 4, capsize = 10, marker = 'o', markersize = 15)
# # fig = plt.figure(figsize=(10, 10))
# # ax = fig.add_subplot(projection='polar')
# theta = np.linspace(np.pi, -np.pi, bins_to_plot) # angles for the plots in radians
# width = (2*np.pi)/bins_to_plot
# UP_radius_mean_all_before = np.mean(np.asarray([np.mean(np.asarray(i[0:4]), axis = 0) for i in angles_histogram_all_sweeps_UP_DOWN[0]]), axis = 0)
# UP_radius_sem_all_before = np.std(np.asarray([np.mean(np.asarray(i[0:4]), axis = 0) for i in angles_histogram_all_sweeps_UP_DOWN[0]]), axis = 0)/np.sqrt(len(mice_UP))
# ax.plot(theta, UP_radius_mean_all_before, 'ko')
# ax.errorbar(theta, UP_radius_mean_all_before, yerr = UP_radius_sem_all_before, color = 'k')
# plt.fill_between(theta, np.repeat(0, len(theta)), UP_radius_mean_all_before, color = 'k', alpha = 0.1)
# UP_radius_mean_all_after = np.mean(np.asarray([np.mean(np.asarray(i[4:]), axis = 0) for i in angles_histogram_all_sweeps_UP_DOWN[0]]), axis = 0)
# UP_radius_sem_all_after = np.std(np.asarray([np.mean(np.asarray(i[4:]), axis = 0) for i in angles_histogram_all_sweeps_UP_DOWN[0]]), axis = 0)/np.sqrt(len(mice_UP))
# ax.plot(theta, UP_radius_mean_all_after, 'co')
# ax.errorbar(theta, UP_radius_mean_all_after, yerr = UP_radius_sem_all_after, color = 'c')
# plt.fill_between(theta, np.repeat(0, len(theta)), UP_radius_mean_all_after, color = 'c', alpha = 0.1)
# plt.tight_layout()

    

    

        
        