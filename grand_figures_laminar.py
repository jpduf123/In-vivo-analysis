# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 23:01:05 2022

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

home_path = r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS\LAMINAR_UP'
os.chdir(home_path)

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

numb = len(days)


def smooth(y, box_pts, axis = 0):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.apply_along_axis(lambda m: np.convolve(m, box, mode='same'), axis = axis, arr = y)
    return y_smooth

chanMap_32 = np.array([31,32,30,33,27,36,26,37,16,47,18,38,17,46,29,39,19,45,28,40,20,44,22,41,21,34,24,42,23,43,25,35]) - 16
chanMap_16 = np.linspace(0, 15, 16).astype(int)


def cl():
    plt.close('all')

#%% -------------------------------------------------------------------------------- channel layer assignment
# layer map for every mouse (approximate)
# do it for the first sweep here so you can then adjust the drift in the dictionary of all sweeps
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
            
            '221206_1' : [mouse_13]*10
            }

layer_list_LFP = list(layer_dict.values())
layer_list_CSD = copy.deepcopy(layer_list_LFP)



# ONE CHANNEL PER LAYER. 
#LAYER 2: BIGGEST CSD PEAK FROM WHISKER STIM. LAYER 4: EARLIEST CSD DEFLECTION FROM WHISKER STIM. LAYER 5: EARLIEST DEFLECTION FROM SW CSD. LAYER 6: MIDDLE OF CURRENT SINK OF SW, NO IDEA IF THAT'S GOOD OR NOT
# do it for the first sweep here so you can then adjust the drift in the dictionary of all sweeps
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

layer_list_LFP_1 = list(layer_dict_1.values())
layer_list_CSD_1 = copy.deepcopy(layer_list_LFP_1)


# PSTH channels for analysis (i.e. one channel up and down from the one selected for layers)
# do it for the first sweep here so you can then adjust the drift in the dictionary of all sweeps
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
mouse_13_PSTH = list(map(np.asarray, [[4,5,6], [7,8,9], [11,12,13], [15]]))

layer_dict_PSTH = {'160614' : [mouse_1_PSTH]*10,
                  
            '160615' : [mouse_2_PSTH]*4 + [[i + 1 for i in mouse_2_PSTH]]*6,
            
            '160622' : [mouse_3_PSTH]*10,
                        
            '160728' : [mouse_4_PSTH]*10,
            
            '160729' : [mouse_5_PSTH]*5 + [[i + 1 for i in mouse_5_PSTH]]*1 + [[i + 2 for i in mouse_5_PSTH]]*1 + [[i + 3 for i in mouse_5_PSTH]]*3,
            
            # '160810' : [mouse_6_PSTH]*4 + [[i + 1 for i in mouse_6_PSTH]]*6,
            '160810' : [mouse_6_PSTH]*10,

            '220810_2' : [mouse_7_PSTH]*10,
            
            '221018_1' : [mouse_8_PSTH]*10,
            
            '221021_1' : [mouse_9_PSTH]*10,

            '221024_1' : [mouse_10_PSTH]*10,

            '221025_1' : [mouse_11_PSTH]*10,

            '221026_1' : [mouse_12_PSTH]*10,
            
            '221206_1' : [mouse_13_PSTH]*10
            }

layer_list_PSTH = list(layer_dict_PSTH.values())



#%% -------------------------------------------------------------------------------- load up LFP, PSTH, CSD and delta group array

smooth_CSD_over_channels = True
smooth_over_channel_count = 1

os.chdir(home_path)

#1 LFP, PSTH and CSD change per layer for whisker stim. Load all channels in and do the relative change and time plots using the layer dict
# load all channels in and do the min/max etc... and the shift again for every mouse individually so can change it here on the go
LFP_all = []
LFP_shift_all = []

PSTH_all = []
PSTH_shift_all = []

CSD_all = []
CSD_shift_all = []

#this is not shifted so not really useful
LFP_min_ALL = np.zeros([numb,32,5])
LFP_slope_ALL, LFP_slope_rel_ALL = (np.zeros([numb,32,5]) for i in range(2))




# #3 slow wave laminar change (LFP and CSD) after pairing
# Freq_change_ALLCHANS, Peak_dur_overall_change_ALLCHANS, Fslope_overall_change_ALLCHANS, Sslope_overall_change_ALLCHANS, Famp_overall_change_ALLCHANS, Samp_overall_change_ALLCHANS = ([] for i in range(6))


#%
os.chdir(home_path)
for day_ind, day in enumerate(days):
# for day_ind, day in enumerate(days[11:]):
    os.chdir(day)
    print(day)
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    nchans = np.load('LFP_responses.npy').shape[1]
    if nchans == 16:
        chanMap = chanMap_16
    elif nchans == 32:
        chanMap = chanMap_32
    coordinates = [[i] for i in list(np.linspace(0, 1.55, nchans))]*pq.mm

    LFP_all.append(np.load('LFP_responses.npy')[:,chanMap,:])
    PSTH_all.append(np.load('PSTH_responses.npy')) # CAVE MAKE SURE YOU SAVE IT AS CHANMAPPED IN THE LFP ALL SWEEP LAMINAR FILE
    # CSD_all.append(np.load('CSD_all.npy'))

    
    #redo the CSD here 
    # for specific days where the second electrode bugged out, take it out before doing CSD and adjust coordinates
    # smooth_over = 1 # how many channels to smooth over before doing CSD
    # for sweep in range(10):
    #     for stim in LFP_all
    #     curr_LFP = neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep][chanMap,int(stim - time_before*new_fs):int(stim + time_after*new_fs)]), units = 'mV', sampling_rate = new_fs*pq.Hz)                    
    #     curr_to_plot[:,:,ind_stim] = np.transpose(elephant.current_source_density.estimate_csd(curr_LFP, coordinates = coordinates, method = 'StandardCSD', process_estimate=False))

    # curr_CSD_all = 
    
    # curr_LFP = np.load('LFP_responses.npy')[:,chanMap,:]
    # curr_LFP_1 = neo.core.AnalogSignal(scipy.ndimage.gaussian_filter1d(curr_LFP[3,:,:].T, 1, axis = 1), units = 'mV', sampling_rate = new_fs*pq.Hz)
    
    # coordinates = [[i] for i in list(np.linspace(0, 1.55, nchans))]*pq.mm
    # coordinates = [i for i_ind, i in enumerate(coordinates) if i_ind != 1]
    # curr_CSD = elephant.current_source_density.estimate_csd(curr_LFP_1[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]], coordinates = coordinates, method = 'KCSD1D', process_estimate=False)
    
    # channel_borders = np.linspace(0,100,17, dtype = int)
    # channel_indices = channel_borders[0:-1] + (np.diff(channel_borders)/2).astype(int)
    # plt.imshow(curr_CSD.T, aspect = 50, cmap = 'jet', filterrad = 4)
    # fig, ax = plt.subplots(figsize = (8,10))
    # spacer = 5000
    # to_plot = np.asarray(curr_CSD.T)[channel_indices,:]
    # tot_chans = to_plot.shape[0]
    # for ind in range(tot_chans):
    #     ax.plot(np.asarray(to_plot[ind,:]) + ind*-spacer, 'k', linewidth = 1)                                   
    # ax.set_xlim([200,350])
    # plt.tight_layout()
    
    CSD_matrix = np.eye(nchans) # 
    for j in range(1, CSD_matrix.shape[0] - 1):
        CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
    if smooth_CSD_over_channels:
        curr_CSD = - np.asarray([np.dot(CSD_matrix, scipy.ndimage.gaussian_filter1d(np.squeeze(LFP_all[day_ind][i,:,:]), smooth_over_channel_count, axis = 0)) for i in range(10)])
    else:
        curr_CSD = - np.asarray([np.dot(CSD_matrix, LFP_all[day_ind][i,:,:]) for i in range(10)])
        
    # fig, ax = plt.subplots()
    # ax.imshow(curr_CSD[0,:,:], aspect = 'auto', cmap = 'jet', vmin = np.min(curr_CSD[0,1:-1,223:]), vmax = np.max(curr_CSD[0,1:-1,223:]))
    # ax.set_xlim([150,350])
    
    # fig, ax = plt.subplots(figsize = (8,10))
    # spacer = 1000
    # for ind in range(nchans):
    #     ax.plot(LFP_all[day_ind][0,ind,:] + ind* -spacer, 'k', linewidth = 1)                                   
    # ax.set_xlim([150,350])

    # fig, ax = plt.subplots(figsize = (8,10))
    # spacer = np.max(np.max(np.diff(LFP_all[day_ind][0,:,:], axis = 1)))
    # for ind in range(nchans-1):
    #     ax.plot(np.diff(LFP_all[day_ind], axis = 1)[0,ind,:] + ind* -spacer, 'k', linewidth = 1)                                   
    # ax.set_xlim([150,350])

    # fig, ax = plt.subplots(figsize = (8,10))
    # spacer = np.max(np.max(curr_CSD[0,:,:], axis = 1))
    # for ind in range(nchans):
    #     ax.plot(curr_CSD[0,ind,:] + ind* -spacer, 'k', linewidth = 1)                                   
    # ax.set_xlim([150,350])

    # fig, ax = plt.subplots(3,1,figsize = (8,10))
    # ax[0].plot(scipy.ndimage.gaussian_filter1d(LFP_all[day_ind][0,:,232], smooth_over_channel_count))
    # ax[1].plot(np.diff(LFP_all[day_ind][0,:,232]))
    # ax[2].plot(curr_CSD[0,:,232])



    os.chdir('..')
    os.chdir('..')

    CSD_all.append(curr_CSD)

    shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
    # you need to add the last XX (total shift) channels to be able to subtract every image with the next one if there's a shift
    # THIS IS ASSUMING YOU HAVE POSITIVE SHIFT (CHANNEL NUMBERS GET BIGGER), which is why I call max function (if not positive shift would have to change it, to either the min or max, whichever-s absolute value is bigger?)
    total_shift = max(shift)
    
    LFP_shift_all.append(np.asarray([LFP_all[day_ind][i, shift[i]:(32 - (total_shift -shift[i])), :] for i in range(10)]))
    PSTH_shift_all.append(np.asarray([PSTH_all[day_ind][i, shift[i]:(32 - (total_shift -shift[i])), :] for i in range(10)]))
    CSD_shift_all.append(np.asarray([CSD_all[day_ind][i, shift[i]:(32 - (total_shift -shift[i])), :] for i in range(10)]))
        
    tot_chans = LFP_shift_all[day_ind].shape[1]
    
    #clean up PSTH artifacts
    if day == '220810_2':
        PSTH_shift_all[day_ind][:,:,:120] = 0
    if day == '221018_1':
        PSTH_shift_all[day_ind][:,:,:120] = 0
    if day == '221021_1':
        PSTH_shift_all[day_ind][:,:,:120] = 0
    if day == '221024_1':
        PSTH_shift_all[day_ind][:,:,:121] = 0
    if day == '221025_1':
        PSTH_shift_all[day_ind][:,:,:120] = 0
    if day == '221026_1':
        PSTH_shift_all[day_ind][:,:,:123] = 0

    
    # spacer = np.abs(np.min(LFP_shift_all[day_ind][:,:,200:]*1.5))
    # fig, ax = plt.subplots(figsize = (3,10))
    # LFP_before = np.mean(LFP_shift_all[day_ind][[0,1,2,3],:,:], axis = 0)
    # LFP_after = np.mean(LFP_shift_all[day_ind][[4,5,6,7,8,9],:,:], axis = 0)
    # for ind in range(tot_chans):
    #     ax.plot(LFP_before[ind,:] + ind * -spacer, 'b', linewidth = 1)                
    #     ax.plot(LFP_after[ind,:] + ind * -spacer, 'r', linewidth = 1)                     
    #     ax.set_xlim([150,400])
    # ax.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 6)
    # plt.tight_layout()

    # spacer = np.abs(np.max(smooth(PSTH_shift_all[day_ind][:,:,125:],10)))
    # fig, ax = plt.subplots(figsize = (3,10))
    # PSTH_before = np.mean(PSTH_shift_all[day_ind][[0,1,2,3],:,:], axis = 0)
    # PSTH_after = np.mean(PSTH_shift_all[day_ind][[4,5,6,7,8,9],:,:], axis = 0)
    # for ind in range(tot_chans):
    #     ax.plot(smooth(PSTH_before[ind,:], 10) + ind * -spacer, 'b', linewidth = 1)                 
    #     ax.plot(smooth(PSTH_after[ind,:], 10) + ind * -spacer, 'r', linewidth = 1)                     
    #     # ax.set_xlim([150,400])
    # ax.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 6)
    # fig.suptitle(f'{day}')
    # plt.tight_layout()

    # spacer = 500
    # fig, ax = plt.subplots(figsize = (3,10))
    # CSD_before = np.mean(CSD_shift_all[day_ind][[0,1,2,3],:,:], axis = 0)
    # CSD_after = np.mean(CSD_shift_all[day_ind][[4,5,6,7,8,9],:,:], axis = 0)
    # for ind in range(tot_chans):
    #     ax.plot(CSD_before[ind,:] + ind * -spacer), 'b', linewidth = 1)                 
    #     ax.plot(CSD_after[ind,:] + ind * -spacer), 'r', linewidth = 1)                     
    #     ax.set_xlim([150,400])
    #     if ind in 
    # ax.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 6)
    # plt.tight_layout()

    os.chdir(home_path)
    
    
    
#%% ------------------------------------------------------------------- LFP, PSTH AND CSD PER LAYER ACROSS MICE

#create overall figures with change in each layer:
LFP_min_shift_all = [np.min(i[:,:,220:260], axis = 2) - i[:,:,200] for i in LFP_shift_all]
CSD_max_shift_all = [np.min(i[:,:,220:260], axis = 2) for i in CSD_shift_all]
PSTH_max_shift_all = [np.max(smooth(i[:,:,100:160], 20, axis = 2), axis = 2) for i in PSTH_shift_all]
PSTH_magn_shift_all = [np.sum(i[:,:,110:160], axis = 2) for i in PSTH_shift_all]


LFP_min_shift_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0)*100 for i in LFP_min_shift_all]
CSD_max_shift_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0)*100 for i in CSD_max_shift_all]
PSTH_max_shift_all_rel = [i/np.nanmean(i[[0,1,2,3],:], axis = 0)*100 for i in PSTH_max_shift_all]
PSTH_magn_shift_all_rel = [i/np.nanmean(i[[0,1,2,3],:], axis = 0)*100 for i in PSTH_magn_shift_all]


# average across channels in each layer with the channel map for each sweep
# mouse, layer (1, 2/3, 4, 5, 6), sweep
LFP_min_layer = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
PSTH_max_layer = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
PSTH_magn_layer = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
CSD_max_layer = np.zeros([len(LFP_min_shift_all_rel), 4, 10])

LFP_min_layer_rel = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
PSTH_max_layer_rel = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
PSTH_magn_layer_rel = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
CSD_max_layer_rel = np.zeros([len(LFP_min_shift_all_rel), 4, 10])

LFP_min_layer_1 = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
PSTH_max_layer_1 = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
PSTH_magn_layer_1 = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
CSD_max_layer_1 = np.zeros([len(LFP_min_shift_all_rel), 4, 10])

LFP_min_layer_rel_1 = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
PSTH_max_layer_rel_1 = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
PSTH_magn_layer_rel_1 = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
CSD_max_layer_rel_1 = np.zeros([len(LFP_min_shift_all_rel), 4, 10])

for m_ind, day in enumerate(days):
    if m_ind > 6:
        nchans = 16
    else:
        nchans = 32
    if day == '221206_1':
        nchans =  12 # artifacty in 13, messed up CSD
    curr_layers_1_avg = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
    curr_layers_1 = layer_dict_1[day][0]
    for sweep in range(10): 
        
        # curr_layers_CSD = layer_dict[day][0]
        # curr_layers_LFP_1 = layer_dict_1[day][0]
        # curr_layers_CSD_1 = layer_dict_1[day][0]
        # curr_layers_PSTH = layer_dict_PSTH[day][0]
        
        #average over selected channel plus minus 1
        # curr_layers_LFP = [np.array([i[0]-1,i[0],i[0]+1]) for i in layer_dict_1[day][0]]
        # curr_layers_CSD = layer_dict[day][0]
        # curr_layers_PSTH = layer_dict_PSTH[day][0]

        # average across channels in each layer (plus minus one from selected channel)
        LFP_min_layer[m_ind, :, sweep] = np.asarray([np.mean(LFP_min_shift_all[m_ind][sweep, curr_layers_1_avg[i]]) for i in range(4)])
        PSTH_max_layer[m_ind, :, sweep] = np.asarray([np.nanmean(PSTH_max_shift_all[m_ind][sweep,curr_layers_1_avg[i]]) for i in range(4)])
        PSTH_magn_layer[m_ind, :, sweep] = np.asarray([np.nanmean(PSTH_magn_shift_all[m_ind][sweep,curr_layers_1_avg[i]]) for i in range(4)])
        CSD_max_layer[m_ind, :, sweep] = np.asarray([np.mean(CSD_max_shift_all[m_ind][sweep,curr_layers_1_avg[i]]) for i in range(4)]) 

        LFP_min_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(LFP_min_shift_all_rel[m_ind][sweep, curr_layers_1_avg[i]]) for i in range(4)])
        PSTH_max_layer_rel[m_ind, :, sweep] = np.asarray([np.nanmean(PSTH_max_shift_all_rel[m_ind][sweep,curr_layers_1_avg[i]]) for i in range(4)])
        PSTH_magn_layer_rel[m_ind, :, sweep] = np.asarray([np.nanmean(PSTH_magn_shift_all_rel[m_ind][sweep,curr_layers_1_avg[i]]) for i in range(4)])
        CSD_max_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(CSD_max_shift_all_rel[m_ind][sweep,curr_layers_1_avg[i]]) for i in range(4)]) 
        
        #only selected channel
        LFP_min_layer_1[m_ind, :, sweep] = np.concatenate([LFP_min_shift_all[m_ind][sweep, curr_layers_1[i]] for i in range(4)])
        PSTH_max_layer_1[m_ind, :, sweep] = np.concatenate([PSTH_max_shift_all[m_ind][sweep,curr_layers_1[i]] for i in range(4)])
        PSTH_magn_layer_1[m_ind, :, sweep] = np.concatenate([PSTH_magn_shift_all[m_ind][sweep,curr_layers_1[i]] for i in range(4)])
        CSD_max_layer_1[m_ind, :, sweep] = np.concatenate([CSD_max_shift_all[m_ind][sweep,curr_layers_1[i]] for i in range(4)]) 

        LFP_min_layer_rel_1[m_ind, :, sweep] = np.concatenate([LFP_min_shift_all_rel[m_ind][sweep, curr_layers_1[i]] for i in range(4)])
        PSTH_max_layer_rel_1[m_ind, :, sweep] = np.concatenate([PSTH_max_shift_all_rel[m_ind][sweep,curr_layers_1[i]] for i in range(4)])
        PSTH_magn_layer_rel_1[m_ind, :, sweep] = np.concatenate([PSTH_magn_shift_all_rel[m_ind][sweep,curr_layers_1[i]] for i in range(4)])
        CSD_max_layer_rel_1[m_ind, :, sweep] = np.concatenate([CSD_max_shift_all_rel[m_ind][sweep,curr_layers_1[i]] for i in range(4)]) 



# ----------------------------------------------------------------------------------------- LFP

# # LFP min timecourse in averaged in layers
# fig, ax = plt.subplots()
# for i, layer in zip([1,2,3,4], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
#     ax.plot(np.nanmean(LFP_min_layer_rel[:,i,:], axis = 0).T, label = str(layer))
#     ax.fill_between(list(range(10)), np.nanmean(LFP_min_layer_rel[:,i,:], axis = 0).T + 1*np.nanstd(LFP_min_layer_rel[:,i,:], axis = 0).T/np.sqrt(LFP_min_layer_rel.shape[0]), np.nanmean(LFP_min_layer_rel[:,i,:], axis = 0).T - 1*np.nanstd(LFP_min_layer_rel[:,i,:], axis = 0).T/np.sqrt(LFP_min_layer_rel.shape[0]), alpha = 0.1)
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# fig.suptitle('LFP rel averaged within layer')
# plt.savefig('LFP rel averaged within layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('LFP rel averaged within layer.svg', dpi = 1000, format = 'svg')
# plt.tight_layout()

# #normalize LFP to layer2/3 baseline peak
# LFP_min_layer_rel_1_norm = np.transpose(LFP_min_layer_1, (1,2,0))/np.mean(LFP_min_layer_1[:,0,[0,1,2,3]], axis = 1)
# # LFP_min_layer_rel_1_norm = np.transpose(LFP_min_layer_1, (1,2,0))/np.mean(LFP_min_layer_1[:,0,[0,1,2,3]], axis = 1)
# LFP_min_layer_rel_1_norm = np.transpose(LFP_min_layer_rel_1_norm, (2,0,1))

# LFP min timecourse one channel per layers
arr_to_plot = LFP_min_layer_rel
fig, ax = plt.subplots(figsize = (10,4))
for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
    to_plot_mean = np.mean(arr_to_plot[:,i,:], axis = 0) # average across mice
    to_plot_err = 1*np.nanstd(arr_to_plot[:,i,:], axis = 0).T/np.sqrt(arr_to_plot.shape[0])
    ax.plot(to_plot_mean.T, label = str(layer))
    ax.fill_between(list(range(10)), to_plot_mean.T + 1*to_plot_err, to_plot_mean.T - 1*to_plot_err, alpha = 0.1)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
ax.set_ylim([50, 120])
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 20)
ax.set_ylabel('LFP response \n (% of baseline)', size = 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 20)
ax.set_yticks([50,75,100])
ax.set_yticklabels(list(map(str, ax.get_yticks())), size = 20)
plt.tight_layout()
# fig.suptitle('LFP rel one channel within layer')
# plt.savefig('LFP rel per layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('LFP rel per layer.pdf', dpi = 1000, format = 'pdf')

# layers_for_anova = [0,1,2] # layers 2/3, 4, 5
# LFP_min_layer_rel_1_for_ANOVA = np.zeros([len(days), 30])
# curr_for_ANOVA = LFP_min_layer_rel_1[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
# curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
# for layer in range(3):
#     LFP_min_layer_rel_1_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
# # LFP_min_layer_rel_1_for_ANOVA = LFP_min_layer_rel_1_for_ANOVA[LFP_min_layer_rel_1_for_ANOVA[:,-1].argsort()]
# np.savetxt('LFP min 1 channel per layer.csv', LFP_min_layer_rel_1_for_ANOVA, delimiter = ',')

# LFP_min_layer_rel_for_ANOVA = np.zeros([len(days), 30])
# curr_for_ANOVA = LFP_min_layer_rel[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
# curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
# for layer in range(3):
#     LFP_min_layer_rel_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
# # LFP_min_layer_rel_1_for_ANOVA = LFP_min_layer_rel_1_for_ANOVA[LFP_min_layer_rel_1_for_ANOVA[:,-1].argsort()]
# np.savetxt('LFP min 3 channels per layer.csv', LFP_min_layer_rel_for_ANOVA, delimiter = ',')


# #LFP min timecourse in every mouse
# fig, ax = plt.subplots(3,5, sharey = True)
# for ax1_ind, ax1 in enumerate(list(ax.flatten())):
#     if ax1_ind >= len(days):
#         continue
#     for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5']):
#         ax1.plot(LFP_min_layer_rel[ax1_ind,i,:], label = str(layer))
#     handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# fig.suptitle('LFP min in one chanel')
# plt.tight_layout()

print(np.mean(np.mean(LFP_min_layer_rel[:,:,[4,5,6,7,8,9]], axis = 2), axis = 0))
print(np.std(np.mean(LFP_min_layer_rel[:,:,[4,5,6,7,8,9]], axis = 2), axis = 0))


# -----------------------------------------------------------------------------------------PSTH

# #PSTH max timecourse in different layers. Choose mice with good spiking in the first place
mice_with_spikes = [0,1,2,3,4,5,6,7,9,10,11,12] # mouse 8 is not really useable because of weird layer 4.... could still put it in doesn't change the significance of the result

# fig, ax = plt.subplots()
# for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
#     ax.plot(np.transpose(np.nanmean(PSTH_max_layer_rel[mice_with_spikes,i,:], axis = 0)), label = str(layer))
#     ax.fill_between(list(range(10)), np.nanmean(PSTH_max_layer_rel[mice_with_spikes,i,:], axis = 0).T + 1*np.nanstd(PSTH_max_layer_rel[mice_with_spikes,i,:], axis = 0).T/np.sqrt(len(mice_with_spikes)), np.nanmean(PSTH_max_layer_rel[mice_with_spikes,i,:], axis = 0).T - 1*np.nanstd(PSTH_max_layer_rel[mice_with_spikes,i,:], axis = 0).T/np.sqrt(len(mice_with_spikes)), alpha = 0.1)
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# fig.suptitle('PSTH average within layer')
# plt.savefig('PSTH average within layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('PSTH average within layer.svg', dpi = 1000, format = 'svg')
# plt.tight_layout()

#normalize MUA to layer2/3 baseline peak
PSTH_magn_layer_rel_1_norm = np.transpose(PSTH_magn_layer_1, (1,2,0))/np.mean(PSTH_magn_layer_1[:,0,[0,1,2,3]], axis = 1)
# CSD_max_layer_rel_1_norm = np.transpose(CSD_max_layer_1, (1,2,0))/np.mean(CSD_max_layer_1[:,0,[0,1,2,3]], axis = 1)
PSTH_magn_layer_rel_1_norm = np.transpose(PSTH_magn_layer_rel_1_norm, (2,0,1))

arr_to_plot = PSTH_magn_layer_rel
# some 
# arr_to_plot[]
fig, ax = plt.subplots(figsize = (10,4))
for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
    to_plot_mean = np.mean(arr_to_plot[:,i,:], axis = 0) # average across mice
    to_plot_err = 1*np.nanstd(arr_to_plot[:,i,:], axis = 0).T/np.sqrt(arr_to_plot.shape[0])
    ax.plot(to_plot_mean.T, label = str(layer))
    ax.fill_between(list(range(10)), to_plot_mean.T + 1*to_plot_err, to_plot_mean.T - 1*to_plot_err, alpha = 0.1)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
ax.set_ylim([40, 150])
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 20)
ax.set_ylabel('MUA response \n (% of baseline)', size = 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 20)
ax.set_yticks([50,75,100,125,150])
ax.set_yticklabels(list(map(str, ax.get_yticks())), size = 20)
plt.tight_layout()
plt.savefig('PSTH rel per layer.jpg', dpi = 1000, format = 'jpg')
plt.savefig('PSTH rel per layer.pdf', dpi = 1000, format = 'pdf')

# layers_for_anova = [0,1,2]
# PSTH_max_layer_rel_1_for_ANOVA = np.zeros([len(days), 30])
# curr_for_ANOVA = PSTH_max_layer_rel_1[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
# curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
# for layer in range(3):
#     PSTH_max_layer_rel_1_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
# np.savetxt('PSTH peak 1 channel per layer.csv', PSTH_max_layer_rel_1_for_ANOVA, delimiter = ',')
# # PSTH_max_layer_rel_1_for_ANOVA = PSTH_max_layer_rel_1_for_ANOVA[PSTH_max_layer_rel_1_for_ANOVA[:,-1].argsort()]

layers_for_anova = [0,1,2]
PSTH_max_layer_rel_for_ANOVA = np.zeros([len(days), 30])
curr_for_ANOVA = PSTH_max_layer_rel[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
for layer in range(3):
    PSTH_max_layer_rel_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
np.savetxt('PSTH peak 3 channels per layer.csv', PSTH_max_layer_rel_for_ANOVA, delimiter = ',')


# #PSTH max timecourse in every mouse
# fig, ax = plt.subplots(3,5, sharey = True)
# for ax1_ind, ax1 in enumerate(list(ax.flatten())):
#     if ax1_ind not in mice_with_spikes:
#         continue
#     for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5']):
#         ax1.plot(PSTH_max_layer_rel[ax1_ind,i,:], label = str(layer))
#     handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# fig.suptitle('PSTH max in one chanel')
# plt.tight_layout()

print(np.mean(np.mean(PSTH_magn_layer_rel[:,:,[4,5,6,7,8,9]], axis = 2), axis = 0))
print(np.std(np.mean(PSTH_magn_layer_rel[:,:,[4,5,6,7,8,9]], axis = 2), axis = 0))



# -----------------------------------------------------------------------------------------CSD

# #CSD whisker response max timecourse in different layers, averaged (bit buggy in layer 5)
# fig, ax = plt.subplots()
# for i, layer in zip([1,2,3,4], ['layer 2/3', 'layer 4', 'layer 5']):
#     ax.plot(np.transpose(np.mean(CSD_max_layer_rel[:,i,:], axis = 0)), label = str(layer))
#     ax.fill_between(list(range(10)), np.nanmean(CSD_max_layer_rel[:,i,:], axis = 0).T + 1*np.nanstd(CSD_max_layer_rel[:,i,:], axis = 0).T/np.sqrt(CSD_max_layer_rel.shape[0]), np.nanmean(CSD_max_layer_rel[:,i,:], axis = 0).T - 1*np.nanstd(CSD_max_layer_rel[:,i,:], axis = 0).T/np.sqrt(CSD_max_layer_rel.shape[0]), alpha = 0.1)
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# fig.suptitle('CSD rel averaged within layer')
# plt.savefig('CSD rel averaged within layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('CSD rel averaged within layer.svg', dpi = 1000, format = 'svg')
# plt.tight_layout()

#normalize CSD to layer2/3 baseline peak
CSD_max_layer_rel_1_norm = np.transpose(CSD_max_layer_1, (1,2,0))/np.mean(CSD_max_layer_1[:,0,[0,1,2,3]], axis = 1)
# CSD_max_layer_rel_1_norm = np.transpose(CSD_max_layer_1, (1,2,0))/np.mean(CSD_max_layer_1[:,0,[0,1,2,3]], axis = 1)
CSD_max_layer_rel_1_norm = np.transpose(CSD_max_layer_rel_1_norm, (2,0,1))
#CSD max timecourse in different layers, one channel per layer
arr_to_plot = CSD_max_layer_rel
fig, ax = plt.subplots(figsize = (10,4))
for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
    to_plot_mean = np.mean(arr_to_plot[:,i,:], axis = 0) # average across mice
    to_plot_err = 1*np.nanstd(arr_to_plot[:,i,:], axis = 0).T/np.sqrt(arr_to_plot.shape[0])
    ax.plot(to_plot_mean.T, label = str(layer))
    ax.fill_between(list(range(10)), to_plot_mean.T + 1*to_plot_err, to_plot_mean.T - 1*to_plot_err, alpha = 0.1)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 20)
ax.set_ylabel('CSD response \n (% of baseline)', size = 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 20)
ax.set_yticks([50,75,100,125])
ax.set_yticklabels(list(map(str, ax.get_yticks())), size = 20)
ax.set_ylim([45, 130])
# fig.suptitle('CSD rel 1 channel within layer')
plt.tight_layout()
plt.savefig('CSD rel per layer.jpg', dpi = 1000, format = 'jpg')
plt.savefig('CSD rel per layer.pdf', dpi = 1000, format = 'pdf')

layers_for_anova = [0,1,2]
CSD_max_layer_rel_1_for_ANOVA = np.zeros([len(days), 30])
curr_for_ANOVA = CSD_max_layer_rel_1[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
for layer in range(3):
    CSD_max_layer_rel_1_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
# CSD_max_layer_rel_1_for_ANOVA = CSD_max_layer_rel_1_for_ANOVA[CSD_max_layer_rel_1_for_ANOVA[:,-1].argsort()]
np.savetxt('CSD peak 1 channel per layer.csv', CSD_max_layer_rel_1_for_ANOVA, delimiter = ',')

layers_for_anova = [0,1,2]
CSD_max_layer_rel_for_ANOVA = np.zeros([len(days), 30])
curr_for_ANOVA = CSD_max_layer_rel[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
for layer in range(3):
    CSD_max_layer_rel_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
# CSD_max_layer_rel_for_ANOVA = CSD_max_layer_rel_for_ANOVA[CSD_max_layer_rel_for_ANOVA[:,-1].argsort()]
np.savetxt('CSD peak 3 channels per layer.csv', CSD_max_layer_rel_for_ANOVA, delimiter = ',')


#CSD max timecourse in different layers in every mouse
fig, ax = plt.subplots(3,5, sharey = True)
for ax1_ind, ax1 in enumerate(list(ax.flatten())):
    if ax1_ind >= len(days):
        continue
    for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5']):
        ax1.plot(CSD_max_layer_rel[ax1_ind,i,:], label = str(layer))
    handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.suptitle('CSD max in one chanel')
plt.tight_layout()


print(np.mean(np.mean(CSD_max_layer_rel[:,:,[4,5,6,7,8,9]], axis = 2), axis = 0))
print(np.std(np.mean(CSD_max_layer_rel[:,:,[4,5,6,7,8,9]], axis = 2), axis = 0))




#%% -------------------------------------------------------------------------------- PSD, DELTA POWER OF LFP AND CSD PER LAYER ACROSS MICE
new_fs = 1000
exclude_before = 0.1
# maybe better to take 1 second after stim for slow waves as high change they get fucked up by the stim otherwise?
exclude_after = 0.9

fftfreq = np.fft.fftfreq(int((5 - exclude_before - exclude_after)*new_fs), d = (1/new_fs))

#2 delta power of LFP and delta power of CSD
delta_LFP_all = []
delta_LFP_shift_all = []
delta_LFP_median_all = []
delta_LFP_median_shift_all = []

delta_CSD_all = []
delta_CSD_shift_all = []
delta_CSD_median_all = []
delta_CSD_median_shift_all = []

PSD_LFP_all = []
PSD_LFP_shift_all = []
PSD_LFP_median_all = []
PSD_LFP_median_shift_all = []

PSD_CSD_all = []
PSD_CSD_shift_all = []
PSD_CSD_median_all = []
PSD_CSD_median_shift_all = []


# os.chdir(home_path)
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for day_ind, day in enumerate(days):
    os.chdir(day)
    print(day)
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    nchans = np.load('LFP_responses.npy').shape[1]
    if nchans == 16:
        chanMap = chanMap_16
    elif nchans == 32:
        chanMap = chanMap_32
    coordinates = [[i] for i in list(np.linspace(0, 1.55, nchans))]*pq.mm

    delta_LFP_all.append(np.loadtxt('delta_power.csv', delimiter = ',')[:,chanMap])
    delta_CSD_all.append(np.loadtxt('delta_power_CSD.csv', delimiter = ',')) # CSD already in channel map
    delta_LFP_median_all.append(np.loadtxt('delta_power_median.csv', delimiter = ',')[:,chanMap])
    delta_CSD_median_all.append(np.loadtxt('delta_power_CSD_median.csv', delimiter = ',')) # CSD already in channel map

    PSD_LFP_all.append(np.load('PSD.npy')[:,chanMap])
    PSD_CSD_all.append(np.load('PSD_CSD.npy')) # CSD already in channel map
    PSD_LFP_median_all.append(np.load('PSD_median.npy')[:,chanMap])
    PSD_CSD_median_all.append(np.load('PSD_CSD_median.npy')) # CSD already in channel map
    
    
    # adjust for electrode shift
    shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
    # you need to add the last XX (total shift) channels to be able to subtract every image with the next one if there's a shift
    # THIS IS ASSUMING YOU HAVE POSITIVE SHIFT (CHANNEL NUMBERS GET BIGGER), which is why I call max function (if not positive shift would have to change it, to either the min or max, whichever-s absolute value is bigger?)
    total_shift = max(shift)

    delta_LFP_shift_all.append(np.asarray([delta_LFP_all[day_ind][i, shift[i]:(32 - (total_shift - shift[i]))] for i in range(10)]))
    delta_CSD_shift_all.append(np.asarray([delta_CSD_all[day_ind][i, shift[i]:(32 - (total_shift - shift[i]))] for i in range(10)]))
    delta_LFP_median_shift_all.append(np.asarray([delta_LFP_median_all[day_ind][i, shift[i]:(32 - (total_shift - shift[i]))] for i in range(10)]))
    delta_CSD_median_shift_all.append(np.asarray([delta_CSD_median_all[day_ind][i, shift[i]:(32 - (total_shift - shift[i]))] for i in range(10)]))

    PSD_LFP_shift_all.append(np.asarray([PSD_LFP_all[day_ind][i, shift[i]:(32 - (total_shift - shift[i]))] for i in range(10)]))
    PSD_CSD_shift_all.append(np.asarray([PSD_CSD_all[day_ind][i, shift[i]:(32 - (total_shift - shift[i]))] for i in range(10)]))
    PSD_LFP_median_shift_all.append(np.asarray([PSD_LFP_median_all[day_ind][i, shift[i]:(32 - (total_shift - shift[i]))] for i in range(10)]))
    PSD_CSD_median_shift_all.append(np.asarray([PSD_CSD_median_all[day_ind][i, shift[i]:(32 - (total_shift - shift[i]))] for i in range(10)]))

    os.chdir('..')
    os.chdir('..')


delta_LFP_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in delta_LFP_all]
delta_CSD_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in delta_CSD_all]

delta_LFP_shift_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in delta_LFP_shift_all]
delta_CSD_shift_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in delta_CSD_shift_all]

delta_LFP_median_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in delta_LFP_median_all]
delta_CSD_median_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in delta_CSD_median_all]

delta_LFP_median_shift_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in delta_LFP_median_shift_all]
delta_CSD_median_shift_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in delta_CSD_median_shift_all]


# LFP_delta_layer = np.zeros([len(days), 5, 10])
# CSD_delta_layer = np.zeros([len(days), 5, 10])

LFP_delta_layer_rel = np.zeros([len(days), 4, 10])
CSD_delta_layer_rel = np.zeros([len(days), 4, 10])

# LFP_delta_layer_1 = np.zeros([len(days), 4, 10])
# CSD_delta_layer_1 = np.zeros([len(days), 4, 10])

LFP_delta_layer_rel_1 = np.zeros([len(days), 4, 10])
CSD_delta_layer_rel_1 = np.zeros([len(days), 4, 10])

PSD_LFP_layer = np.zeros([len(days), 4, 4000, 10])
PSD_LFP_median_layer = np.zeros([len(days), 4, 4000, 10])
PSD_CSD_layer = np.zeros([len(days), 4, 4000, 10])
PSD_CSD_median_layer = np.zeros([len(days), 4, 4000, 10])


delta_to_use = delta_LFP_median_shift_all_rel
CSD_delta_to_use = delta_CSD_median_shift_all_rel

for m_ind, day in enumerate(days):
    for sweep in range(10): 
        curr_layers_1_avg = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
        curr_layers_LFP = layer_dict[day][0]
        curr_layers_CSD = layer_dict[day][0]
        curr_layers_LFP_1 = layer_dict_1[day][0]
        curr_layers_CSD_1 = layer_dict_1[day][0]

        # LFP_delta_layer[m_ind, :, sweep] = np.asarray([np.mean(delta_LFP_shift_all[m_ind][sweep, curr_layers_LFP[i]]) for i in range(5)])
        # CSD_delta_layer[m_ind, :, sweep] = np.asarray([np.mean(delta_CSD_shift_all[m_ind][sweep,curr_layers_CSD[i]]) for i in range(5)]) 

        LFP_delta_layer_rel[m_ind, :, sweep] = np.asarray([np.median(delta_to_use[m_ind][sweep, curr_layers_1_avg[i]]) for i in range(4)])
        CSD_delta_layer_rel[m_ind, :, sweep] = np.asarray([np.median(CSD_delta_to_use[m_ind][sweep,curr_layers_1_avg[i]]) for i in range(4)]) 
        
        # LFP_delta_layer_1[m_ind, :, sweep] = np.concatenate([delta_LFP_shift_all[m_ind][sweep, curr_layers_LFP_1[i]] for i in range(4)])
        # CSD_delta_layer_1[m_ind, :, sweep] = np.concatenate([delta_CSD_shift_all[m_ind][sweep,curr_layers_CSD_1[i]] for i in range(4)]) 

        LFP_delta_layer_rel_1[m_ind, :, sweep] = np.concatenate([delta_to_use[m_ind][sweep, curr_layers_LFP_1[i]] for i in range(4)])
        CSD_delta_layer_rel_1[m_ind, :, sweep] = np.concatenate([CSD_delta_to_use[m_ind][sweep,curr_layers_CSD_1[i]] for i in range(4)]) 

        
        PSD_LFP_layer[m_ind, :, :, sweep] = np.asarray([np.median(PSD_LFP_all[m_ind][sweep, curr_layers_1_avg[i], :], axis = 0) for i in range(4)])
        PSD_LFP_median_layer[m_ind, :, :, sweep] = np.asarray([np.median(PSD_LFP_median_all[m_ind][sweep, curr_layers_1_avg[i], :], axis = 0) for i in range(4)])
        
        PSD_CSD_layer[m_ind, :, :, sweep] = np.asarray([np.median(PSD_CSD_all[m_ind][sweep, curr_layers_1_avg[i], :], axis = 0) for i in range(4)])
        PSD_CSD_median_layer[m_ind, :, :, sweep] = np.asarray([np.median(PSD_CSD_median_all[m_ind][sweep, curr_layers_1_avg[i], :], axis = 0) for i in range(4)])
        
       
        
# ---------------------------------------------------- PSD across mice in different layers
fft_freq_ind = np.where(np.logical_and(0 <= fftfreq , 10 >= fftfreq))[0]
fig, ax = plt.subplots(3,1, sharey = True, figsize = (7, 18))
mice_to_plot = [0,1,2,3,4,5]
# mice_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11,12]
to_plot_before = np.mean(scipy.ndimage.gaussian_filter(PSD_LFP_median_layer[:,:,:,[0,1,2,3]][mice_to_plot,:,:,:], (0,0,1,0)), axis = 3)/3500/1000
to_plot_after = np.mean(scipy.ndimage.gaussian_filter(PSD_LFP_median_layer[:,:,:,[4,5,6,7,8,9]][mice_to_plot,:,:,:], (0,0,1,0)), axis = 3)/3500/1000
for ind, ax1 in enumerate(list(ax.flatten())):
    # ax1.semilogy(delta_fft_freq_ind, scipy.ndimage.gaussian_filter(to_plot_before[:,ind,delta_fft_freq_ind].T, (1,0)))
    to_plot_mean = np.mean(to_plot_before[:,ind,fft_freq_ind], axis = 0)
    to_plot_err = np.std(to_plot_before[:,ind,fft_freq_ind], axis = 0)/np.sqrt(len(mice_to_plot))
    ax1.semilogy(fftfreq[fft_freq_ind], to_plot_mean, color = 'k')
    ax1.fill_between(fftfreq[fft_freq_ind], to_plot_mean - to_plot_err, to_plot_mean + to_plot_err, alpha = 0.5, color = 'k')
    to_plot_mean = np.mean(to_plot_after[:,ind,fft_freq_ind], axis = 0)
    to_plot_err = np.std(to_plot_after[:,ind,fft_freq_ind], axis = 0)/np.sqrt(len(mice_to_plot))
    ax1.plot(fftfreq[fft_freq_ind], to_plot_mean, color = 'c')
    ax1.fill_between(fftfreq[fft_freq_ind], to_plot_mean - to_plot_err, to_plot_mean + to_plot_err, alpha = 0.5, color = 'c')
    ax1.tick_params(axis="x", labelsize=16)    
    ax1.tick_params(axis="y", labelsize=16, size = 12)   
    ax1.tick_params(which = 'minor', axis="y", size = 9)    
    ax1.set_xlabel('frequency (Hz)', size=16)
    ax1.set_ylabel('LFP power density ($\mathregular{mV^2}$/Hz)', size=16)
    ax1.fill_between([0.5,4], [100000,100000], alpha = 0.1, color = 'k')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.tight_layout()
plt.savefig('PSD LFP averaged within layer.jpg', dpi = 1000, format = 'jpg')
plt.savefig('PSD LFP rel averaged within layer.pdf', dpi = 1000, format = 'pdf')



# ------------------------------------------------------------------------------------ delta power of LFP
# rel delta timecourse in different layer averaged within layers
# mice_to_plot = [0,1,2,3,4,5]
mice_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11,12]
fig, ax = plt.subplots(figsize = (10,4))
for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
    to_plot_mean = np.nanmean(LFP_delta_layer_rel[mice_to_plot,i,:], axis = 0).T*100
    to_plot_err = np.nanstd(LFP_delta_layer_rel[mice_to_plot,i,:], axis = 0).T/np.sqrt(len(mice_to_plot))*100
    ax.plot(to_plot_mean, label = str(layer))
    ax.fill_between(list(range(10)), to_plot_mean + to_plot_err, to_plot_mean - to_plot_err, alpha = 0.1)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
ax.set_ylim([55, 110])
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 20)
ax.set_ylabel('LFP delta power \n (% of baseline)', size = 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 20)
ax.set_yticks([60,80,100])
ax.set_yticklabels(list(map(str, ax.get_yticks())), size = 20)
plt.tight_layout()
plt.savefig('LFP delta rel averaged within layer.jpg', dpi = 1000, format = 'jpg')
plt.savefig('LFP delta rel averaged within layer.pdf', dpi = 1000, format = 'pdf')

layers_for_anova = [0,1,2] 
LFP_delta_layer_rel_for_ANOVA = np.zeros([len(days), 30])
curr_for_ANOVA = LFP_delta_layer_rel[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
for layer in range(3):
    LFP_delta_layer_rel_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
# LFP_delta_layer_rel_1_for_ANOVA = LFP_delta_layer_rel_1_for_ANOVA[LFP_delta_layer_rel_1_for_ANOVA[:,-1].argsort()]
np.savetxt('LFP delta avg per layer.csv', LFP_delta_layer_rel_for_ANOVA, delimiter = ',')


# plot delta power timecourse in every mouse
fig, ax = plt.subplots(3,5, sharey = True)
for ax1_ind, ax1 in enumerate(list(ax.flatten())):
    if ax1_ind >= len(days):
        continue
    for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5']):
        ax1.plot(LFP_delta_layer_rel[ax1_ind,i,:], label = str(layer))
    handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()



# # rel delta timecourse in one channel per layer
# fig, ax = plt.subplots(figsize = (10,4))
# for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
#     to_plot_mean = np.nanmean(LFP_delta_layer_rel_1[mice_to_plot,i,:], axis = 0).T*100
#     to_plot_err = np.nanstd(LFP_delta_layer_rel_1[mice_to_plot,i,:], axis = 0).T/np.sqrt(len(mice_to_plot))*100
#     ax.plot(to_plot_mean, label = str(layer))
#     ax.fill_between(list(range(10)), to_plot_mean + to_plot_err, to_plot_mean - to_plot_err, alpha = 0.1)
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# ax.set_ylim([50, 120])
# ax.axvline(3.5, linestyle = '--', color = 'k')
# ax.set_xlabel('time from pairing (min)', size = 20)
# ax.set_ylabel('LFP delta power \n (% of baseline)', size = 20)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xticks([0,2,5,7,9])
# ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 20)
# ax.set_yticks([50,75,100])
# ax.set_yticklabels(list(map(str, ax.get_yticks())), size = 20)
# plt.tight_layout()
# plt.savefig('LFP delta rel one channel per layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('LFP delta rel one channel per layer.pdf', dpi = 1000, format = 'pdf')


# layers_for_anova = [0,1,2]
# LFP_delta_layer_rel_1_for_ANOVA = np.zeros([len(days), 30])
# curr_for_ANOVA = LFP_delta_layer_rel_1[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
# curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
# for layer in range(3):
#     LFP_delta_layer_rel_1_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
# # LFP_delta_layer_rel_1_for_ANOVA = LFP_delta_layer_rel_1_for_ANOVA[LFP_delta_layer_rel_1_for_ANOVA[:,-1].argsort()]
# np.savetxt('LFP delta 1 channel per layer.csv', LFP_delta_layer_rel_1_for_ANOVA, delimiter = ',')


# # plot delta power timecourse in every mouse
# fig, ax = plt.subplots(3,5, sharey = True)
# for ax1_ind, ax1 in enumerate(list(ax.flatten())):
#     if ax1_ind >= len(days):
#         continue
#     for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5']):
#         ax1.plot(LFP_delta_layer_rel_1[ax1_ind,i,:], label = str(layer))
#     handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# plt.tight_layout()





# ------------------------------------------------------------------------------------ delta power of CSD of spontaneous activity 
# fig, ax = plt.subplots()
# to_plot = CSD_delta_layer_rel
# for i, layer in zip([1,2,3], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
#     ax.plot(np.nanmean(to_plot[:,i,:], axis = 0).T, label = str(layer))
#     ax.fill_between(list(range(10)), np.nanmean(to_plot[:,i,:], axis = 0).T + 1*np.nanstd(to_plot[:,i,:], axis = 0).T/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot[:,i,:], axis = 0).T - 1*np.nanstd(to_plot[:,i,:], axis = 0).T/np.sqrt(to_plot.shape[0]), alpha = 0.1)
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# fig.suptitle('CSD delta rel averaged within layer')
# plt.tight_layout()
# plt.savefig('CSD delta rel averaged within layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('CSD delta rel averaged within layer.svg', dpi = 1000, format = 'svg')



mice_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11,12]
# mice_to_plot = [0,1,2,3,4,5]
fig, ax = plt.subplots(figsize = (10,4))
for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
    to_plot_mean = np.nanmean(CSD_delta_layer_rel[mice_to_plot,i,:], axis = 0).T*100
    to_plot_err = np.nanstd(CSD_delta_layer_rel[mice_to_plot,i,:], axis = 0).T/np.sqrt(len(mice_to_plot))*100
    ax.plot(to_plot_mean, label = str(layer))
    ax.fill_between(list(range(10)), to_plot_mean + to_plot_err, to_plot_mean - to_plot_err, alpha = 0.1)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
ax.set_ylim([50, 120])
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 20)
ax.set_ylabel('CSD delta power \n (% of baseline)', size = 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 20)
ax.set_yticks([50,75,100])
ax.set_yticklabels(list(map(str, ax.get_yticks())), size = 20)
plt.tight_layout()
plt.savefig('CSD delta rel avg per layer.jpg', dpi = 1000, format = 'jpg')
plt.savefig('CSD delta rel avg per layer.pdf', dpi = 1000, format = 'pdf')

layers_for_anova = [0,1,2] #don't use layer 6 as buggy 
CSD_delta_layer_rel_for_ANOVA = np.zeros([len(days), 30])
curr_for_ANOVA = CSD_delta_layer_rel[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
for layer in range(3):
    CSD_delta_layer_rel_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
# CSD_delta_layer_rel_1_for_ANOVA = CSD_delta_layer_rel_1_for_ANOVA[CSD_delta_layer_rel_1_for_ANOVA[:,-1].argsort()]
np.savetxt('CSD delta avg per layer.csv', CSD_delta_layer_rel_for_ANOVA, delimiter = ',')


# plot for every mouse individually
fig, ax = plt.subplots(3,5, sharey = True)
for ax1_ind, ax1 in enumerate(list(ax.flatten())):
    if ax1_ind >= len(days):
        continue
    for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5']):
        ax1.plot(CSD_delta_layer_rel[ax1_ind,i,:], label = str(layer))
    handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()








# fig, ax = plt.subplots()
# to_plot = CSD_delta_layer_rel_1
# for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
#     ax.plot(np.nanmean(to_plot[:,i,:], axis = 0).T, label = str(layer))
#     ax.fill_between(list(range(10)), np.nanmean(to_plot[:,i,:], axis = 0).T + 1*np.nanstd(to_plot[:,i,:], axis = 0).T/np.sqrt(to_plot.shape[0]), np.nanmean(to_plot[:,i,:], axis = 0).T - 1*np.nanstd(to_plot[:,i,:], axis = 0).T/np.sqrt(to_plot.shape[0]), alpha = 0.1)
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# fig.suptitle('CSD delta rel one channel per layer')
# plt.tight_layout()
# plt.savefig('CSD delta rel one channel per layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('CSD delta rel one channel per layer.svg', dpi = 1000, format = 'svg')

# layers_for_anova = [0,1,2] #don't use layer 6 as buggy 
# CSD_delta_layer_rel_1_for_ANOVA = np.zeros([len(days), 30])
# curr_for_ANOVA = CSD_delta_layer_rel_1[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
# curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
# for layer in range(3):
#     CSD_delta_layer_rel_1_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
# # CSD_delta_layer_rel_1_for_ANOVA = CSD_delta_layer_rel_1_for_ANOVA[CSD_delta_layer_rel_1_for_ANOVA[:,-1].argsort()]
# np.savetxt('CSD delta 1 channel per layer.csv', CSD_delta_layer_rel_1_for_ANOVA, delimiter = ',')

# # plot for every mouse individually
# fig, ax = plt.subplots(3,5, sharey = True)
# for ax1_ind, ax1 in enumerate(list(ax.flatten())):
#     if ax1_ind >= len(days):
#         continue
#     for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5']):
#         ax1.plot(CSD_delta_layer_rel_1[ax1_ind,i,:], label = str(layer))
#     handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# plt.tight_layout()

# fig, ax = plt.subplots(3,4)
# for ax1_ind, ax1 in enumerate(list(ax.flatten())):
#     for i, layer in zip([1,2,3], ['layer 2/3', 'layer 4', 'layer 5']):
#         ax1.plot(CSD_delta_layer_rel[ax1_ind,i,:], label = str(layer))
#     handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# plt.tight_layout()


#%% --------------------------------------------------------------------------- SO spiking and spontaneous firing rate change after pairing


SW_spiking_all = []
SW_spiking_shift_all = []

SW_spiking_median_all = []
SW_spiking_median_shift_all = []


os.chdir(home_path)
for day_ind, day in enumerate(days):
# for day_ind, day in enumerate(days[11:]):
    os.chdir(day)
    print(day)
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    nchans = np.load('LFP_responses.npy').shape[1]
    if nchans == 16:
        chanMap = chanMap_16
    elif nchans == 32:
        chanMap = chanMap_32

    SW_spiking_all.append(np.load('SW_spiking_sweeps_avg.npy')[:,chanMap,:])
    SW_spiking_median_all.append(np.load('SW_spiking_sweeps_median.npy')[:,chanMap,:])

    shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
    total_shift = max(shift) 
    SW_spiking_shift_all.append(np.asarray([SW_spiking_all[day_ind][i, shift[i]:(32 - (total_shift -shift[i])), :] for i in range(10)]))
    SW_spiking_median_shift_all.append(np.asarray([SW_spiking_median_all[day_ind][i, shift[i]:(32 - (total_shift -shift[i])), :] for i in range(10)]))
    
    os.chdir('..')
    os.chdir('..')



# average across channels in each layer
SW_spiking_layer = np.zeros([len(SW_spiking_shift_all), 4, 10, 1000])
SW_spiking_median_layer = np.zeros([len(SW_spiking_shift_all), 4, 10, 1000])

for m_ind, day in enumerate(days):
    if m_ind > 6:
        nchans = 16
    else:
        nchans = 32
    curr_layers_1_avg = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]] # centre and two surrounding channels
    curr_layers_1 = layer_dict_1[day][0]
    for sweep in range(10): 
        
        # curr_layers_CSD = layer_dict[day][0]
        # curr_layers_LFP_1 = layer_dict_1[day][0]
        # curr_layers_CSD_1 = layer_dict_1[day][0]
        # curr_layers_PSTH = layer_dict_PSTH[day][0]
        
        # average across channels in each layer
        SW_spiking_layer[m_ind, :, sweep, :] = np.asarray([np.nanmean(SW_spiking_shift_all[m_ind][sweep, curr_layers_1_avg[i], :], axis = 0) for i in range(4)])
        SW_spiking_median_layer[m_ind, :, sweep, :] = np.asarray([np.nanmean(SW_spiking_median_shift_all[m_ind][sweep,curr_layers_1_avg[i], :], axis = 0) for i in range(4)])


fig, ax = plt.subplots(3,1, sharey = True, figsize = (7, 18))
mice_to_plot = [0,1,2,3,4,5]
# mice_to_plot = [0,1,2,3,4,5,6,7,9,10,12]
to_plot_before = np.mean(SW_spiking_layer[:,:,[0,1,2,3],:][mice_to_plot,:,:,:], axis = 2)
to_plot_after = np.mean(SW_spiking_layer[:,:,[4,5,6,7,8,9],:][mice_to_plot,:,:,:], axis = 2)
for ind, ax1 in enumerate(list(ax.flatten())):
    # ax1.semilogy(delta_fft_freq_ind, scipy.ndimage.gaussian_filter(to_plot_before[:,ind,delta_fft_freq_ind].T, (1,0)))
    to_plot_mean = np.mean(to_plot_before[:,ind,:], axis = 0)
    to_plot_err = np.std(to_plot_before[:,ind,:], axis = 0)/np.sqrt(len(mice_to_plot))
    ax1.plot(np.arange(-500,500,1),to_plot_mean, color = 'k')
    ax1.fill_between(np.arange(-500,500,1), to_plot_mean - to_plot_err, to_plot_mean + to_plot_err, alpha = 0.5, color = 'k')
    
    to_plot_mean = np.mean(to_plot_after[:,ind,:], axis = 0)
    to_plot_err = np.std(to_plot_after[:,ind,:], axis = 0)/np.sqrt(len(mice_to_plot))
    ax1.plot(np.arange(-500,500,1),to_plot_mean, color = 'c')
    ax1.fill_between(np.arange(-500,500,1), to_plot_mean - to_plot_err, to_plot_mean + to_plot_err, alpha = 0.5, color = 'c')

    ax1.tick_params(axis="x", labelsize=16)    
    ax1.tick_params(axis="y", labelsize=16, size = 12)   
    ax1.tick_params(which = 'minor', axis="y", size = 9)    
    ax1.set_xlabel('time from UP-crossing (ms)', size=16)
    ax1.set_ylabel('spiking probability', size=16)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.tight_layout()
plt.savefig('SW spiking before vs after layered.jpg', format = 'jpg', dpi = 1000)
plt.savefig('SW spiking before vs after layered.pdf', format = 'pdf', dpi = 1000)



#%% -------------------------------------------------------------------------------- SO EXAMPLES, AMPLITUDE, PEAK TO PEAK LAMINAR ANALYSIS ---------------------------------------------
smooth_CSD_over_channels = True
smooth_over = 1


# --------------------- LOAD UP THE SO WAVEFORMS DETECTED IN SPECIFIC CHANEL (LAYER 5 CHANNEL) ---------------------------------------------

LFP_SW_averages_all = []
CSD_SW_averages_all = []
LFP_SW_averages_interp_all = []
CSD_SW_averages_interp_all = []

LFP_SW_shift_averages_all = []
CSD_SW_shift_averages_all = []
LFP_SW_shift_averages_interp_all = []
CSD_SW_shift_averages_interp_all = []

SW_count_all = []
SW_count_layer = []

SW_peak_to_peak_all = []
SW_peak_to_peak_shift_all = []

LFP_peak_to_peak_L5 = []
CSD_peak_to_peak_L5 = []
LFP_peak_to_peak_L5_median = []
CSD_peak_to_peak_L5_median = []
LFP_peak_to_peak_L5_shift = []
CSD_peak_to_peak_L5_shift = []
LFP_peak_to_peak_L5_median_shift = []
CSD_peak_to_peak_L5_median_shift = []

os.chdir(home_path)
for day_ind, day in enumerate(days):
    os.chdir(day)
    print(day)
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    nchans = np.load('LFP_responses.npy').shape[1]
    if nchans == 16:
        chanMap = chanMap_16
    elif nchans == 32:
        chanMap = chanMap_32
    
    layer_idx_for_SW_extract = 2 # layer 2/3, 4, 5 = [0,1,2]
    chan_for_SW = layer_list_LFP_1[day_ind][0][layer_idx_for_SW_extract][0]
    
    shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
    total_shift = max(shift)
    
    SW_count_all.append(pickle.load(open('SW_count', 'rb')))
    SW_count_layer.append(pickle.load(open('SW_count', 'rb'))[chan_for_SW])
    

    #1)--------------------------------------------- average peak to peak calculated within each SO detected in each channel separately
    SW_peak_to_peak_all.append((np.load('SW_samp_sweeps_avg.npy')[:,chanMap] + np.load('SW_famp_sweeps_avg.npy'))[:,chanMap])
    SW_peak_to_peak_shift_all.append(np.asarray([SW_peak_to_peak_all[day_ind][i, shift[i]:(32 - (total_shift - shift[i]))] for i in range(10)]))



    #2)------------------------------------------------ Average and median within sweep of LFP and CSD peak to peak from SOs detected in layer 5
    LFP_peak_to_peak_L5.append(np.load('LFP_SW_peak_to_peak_L5.npy')[:,chanMap])
    CSD_peak_to_peak_L5.append(np.load('CSD_SW_peak_to_peak_L5.npy'))
    LFP_peak_to_peak_L5_median.append(np.load('LFP_SW_peak_to_peak_L5_median.npy')[:,chanMap])
    CSD_peak_to_peak_L5_median.append(np.load('CSD_SW_peak_to_peak_L5_median.npy'))
    LFP_peak_to_peak_L5_shift.append(np.asarray([LFP_peak_to_peak_L5[day_ind][i, shift[i]:(32 - (total_shift - shift[i]))] for i in range(10)]))
    CSD_peak_to_peak_L5_shift.append(np.asarray([CSD_peak_to_peak_L5[day_ind][i, shift[i]:(32 - (total_shift - shift[i]))] for i in range(10)]))
    LFP_peak_to_peak_L5_median_shift.append(np.asarray([LFP_peak_to_peak_L5_median[day_ind][i, shift[i]:(32 - (total_shift - shift[i]))] for i in range(10)]))
    CSD_peak_to_peak_L5_median_shift.append(np.asarray([CSD_peak_to_peak_L5_median[day_ind][i, shift[i]:(32 - (total_shift - shift[i]))] for i in range(10)]))



    #3)----------------------------------------------- peak to peak on average waveform
    # load the average slow wave LFP and CSD waveforms and reorder them in chanMap order
    # All_SW_avg_laminar = np.load('All_SW_avg_laminar.npy')[chanMap, :, :, :][:, :, chanMap,:]
    All_SW_avg_laminar_filt = np.load('All_SW_avg_laminar_filt.npy')[chanMap, :, :, :][:, :, chanMap,:]
    # All_SW_avg_laminar_interp = np.load('All_SW_avg_laminar_interp.npy')[chanMap, :, :, :][:, :, chanMap,:]
    All_SW_avg_laminar_filt_interp = np.load('All_SW_avg_laminar_filt_interp.npy')[chanMap, :, :, :][:, :, chanMap,:]
    # # All_CSD_avg_laminar = np.load('All_CSD_avg_laminar.npy')[chanMap, :, :, :][:, :, chanMap,:]
    # All_CSD_avg_laminar_filt = np.load('All_CSD_avg_laminar_filt.npy')[chanMap, :, :, :][:, :, chanMap,:]
    # # All_CSD_avg_laminar_interp = np.load('All_CSD_avg_laminar_interp.npy')[chanMap, :, :, :][:, :, chanMap,:]   
    # All_CSD_avg_laminar_filt_interp = np.load('All_CSD_avg_laminar_filt_interp.npy')[chanMap, :, :, :][:, :, chanMap,:]

    # CSD of average LFP profile of slow waves detected in each channel
    All_CSD_avg_laminar_filt = np.zeros([nchans, 10, nchans, 2000]) 
    All_CSD_avg_laminar_filt_interp = np.zeros([nchans, 10, nchans, 4000])
    CSD_matrix = np.eye(nchans) # 
    for j in range(1, CSD_matrix.shape[0] - 1):
        CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
    if smooth_CSD_over_channels:
        All_CSD_avg_laminar_filt = - np.asarray([np.asarray([np.dot(CSD_matrix, scipy.ndimage.gaussian_filter1d(np.squeeze(All_SW_avg_laminar_filt[chan,i,:,:]), smooth_over, axis = 0)) for i in range(10)]) for chan in range(nchans)])
        All_CSD_avg_laminar_filt_interp = - np.asarray([np.asarray([np.dot(CSD_matrix, scipy.ndimage.gaussian_filter1d(np.squeeze(All_SW_avg_laminar_filt_interp[chan,i,:,:]), smooth_over, axis = 0)) for i in range(10)]) for chan in range(nchans)])
    else:
        All_CSD_avg_laminar_filt = - np.asarray([np.asarray([np.dot(CSD_matrix, np.squeeze(All_SW_avg_laminar_filt[chan,i,:,:])) for i in range(10)]) for chan in range(nchans)])
        All_CSD_avg_laminar_filt_interp = - np.asarray([np.asarray([np.dot(CSD_matrix, np.squeeze(All_SW_avg_laminar_filt_interp[chan,i,:,:])) for i in range(10)]) for chan in range(nchans)])
    #set first and last channel to 0
    All_CSD_avg_laminar_filt[:,:,0,:] = 0
    All_CSD_avg_laminar_filt_interp[:,:,0,:] = 0
    All_CSD_avg_laminar_filt[:,:,-1,:] = 0
    All_CSD_avg_laminar_filt_interp[:,:,-1,:] = 0

    # choose average SW waveform you want from layer detection
    LFP_SW_averages_all.append(np.asarray([np.squeeze(All_SW_avg_laminar_filt[layer_list_LFP_1[day_ind][i][layer_idx_for_SW_extract],i,:,:]) for i in range(10)]))
    CSD_SW_averages_all.append(np.asarray([np.squeeze(All_CSD_avg_laminar_filt[layer_list_LFP_1[day_ind][i][layer_idx_for_SW_extract],i,:,:]) for i in range(10)]))
    LFP_SW_averages_interp_all.append(np.asarray([np.squeeze(All_SW_avg_laminar_filt_interp[layer_list_LFP_1[day_ind][i][layer_idx_for_SW_extract],i,:,:]) for i in range(10)]))
    CSD_SW_averages_interp_all.append(np.asarray([np.squeeze(All_CSD_avg_laminar_filt_interp[layer_list_LFP_1[day_ind][i][layer_idx_for_SW_extract],i,:,:]) for i in range(10)]))
    LFP_SW_shift_averages_all.append(np.asarray([LFP_SW_averages_all[day_ind][i, shift[i]:(32 - (total_shift - shift[i])),:] for i in range(10)]))
    CSD_SW_shift_averages_all.append(np.asarray([CSD_SW_averages_all[day_ind][i, shift[i]:(32 - (total_shift - shift[i])),:] for i in range(10)]))
    LFP_SW_shift_averages_interp_all.append(np.asarray([LFP_SW_averages_interp_all[day_ind][i, shift[i]:(32 - (total_shift - shift[i])),:] for i in range(10)]))
    CSD_SW_shift_averages_interp_all.append(np.asarray([CSD_SW_averages_interp_all[day_ind][i, shift[i]:(32 - (total_shift - shift[i])),:] for i in range(10)]))



    # fig, ax = plt.subplots()
    # ax.imshow(curr_CSD[0, ], aspect = 'auto', cmap = 'jet', vmin = np.min(curr_CSD[0,1:-1,223:]), vmax = np.max(curr_CSD[0,1:-1,223:]))
    # ax.set_xlim([150,350])
    #CSD traces over sweeps (smooth over time a bit)
    # spacer = np.max(np.max(All_CSD_avg_laminar_filt_interp[chan_for_SW, :, 1:-1, :]))
    # smooth_over_time = 1
    # fig, ax = plt.subplots(1,10, figsize = (17,10), sharey = True) 
    # for ind, ax1 in enumerate(list(ax.flatten())):  
    #     for chan in range(nchans):  
    #         if chan == chan_for_SW:
    #             ax1.plot(np.squeeze(All_CSD_avg_laminar_filt_interp[chan_for_SW, ind, chan,:]) + chan * -spacer, 'r')                                   
    #         else:
    #             ax1.plot(np.squeeze(All_CSD_avg_laminar_filt_interp[chan_for_SW, ind, chan,:]) + chan * -spacer, 'k')                                   
    #     ax1.set_yticks(np.linspace(-spacer*(nchans - 1), 0, nchans))
    #     if ind == 0:
    #         ax1.set_yticklabels(np.linspace((nchans - 1),0,nchans).astype(int), size = 6)
    #     else:
    #         ax1.set_yticklabels([])
    # plt.tight_layout()
    # plt.savefig('SW CSD traces all sweeps.jpg', dpi = 1000, format = 'jpg')

    # fig, ax = plt.subplots(figsize = (8,10))
    # spacer = np.max(np.max(All_CSD_avg_laminar_filt_interp[chan_for_SW, 0, 1:-1, :]))
    # for ind in range(nchans):
    #     ax.plot(np.squeeze(All_CSD_avg_laminar_filt_interp[chan_for_SW, 0, ind,:]) + ind * -spacer, 'k')                                   
    # ax.set_xlim([200,400])

    # for chan in range(nchans): # chan detected in
    #     for sweep in range(len(LFP_all_sweeps)):
    #         All_CSD_avg_laminar_filt[chan,sweep,:,:] = elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(scipy.ndimage.gaussian_filter1d(All_SW_avg_laminar_filt[chan,sweep,chanMap,:], gaussian, axis = 0).T, units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False).T
    #         All_CSD_avg_laminar_filt_interp[chan,sweep,:,:] = elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(scipy.ndimage.gaussian_filter1d(All_SW_avg_laminar_filt_interp[chan,sweep,chanMap,:], gaussian, axis = 0).T, units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False).T
    
    os.chdir(home_path)


LFP_SW_averages_layer = np.zeros([len(days), 10, 4, 4000])
CSD_SW_averages_layer = np.zeros([len(days), 10, 4, 4000])

LFP_SW_averages_layer_1 = np.zeros([len(days), 10, 4, 4000])
CSD_SW_averages_layer_1 = np.zeros([len(days), 10, 4, 4000])

# shift or no shift, interpolated or no?
LFP_SW_to_use = LFP_SW_shift_averages_interp_all
CSD_SW_to_use = CSD_SW_shift_averages_interp_all

for m_ind, day in enumerate(days):
    for sweep in range(10): 
        curr_layers_1_avg = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
        curr_layers_LFP = layer_dict[day][0]
        curr_layers_CSD = layer_dict[day][0]
        curr_layers_LFP_1 = layer_dict_1[day][0]
        curr_layers_CSD_1 = layer_dict_1[day][0]

        LFP_SW_averages_layer[m_ind, sweep, :, :] = np.asarray([np.median(LFP_SW_to_use[m_ind][sweep, curr_layers_1_avg[i],:], axis = 0) for i in range(4)])
        CSD_SW_averages_layer[m_ind, sweep, :, :] = np.asarray([np.median(CSD_SW_to_use[m_ind][sweep,curr_layers_1_avg[i],:], axis = 0) for i in range(4)]) 

        LFP_SW_averages_layer_1[m_ind, sweep, :, :] = np.concatenate([LFP_SW_to_use[m_ind][sweep, curr_layers_LFP_1[i],:] for i in range(4)])
        CSD_SW_averages_layer_1[m_ind, sweep, :, :] = np.concatenate([CSD_SW_to_use[m_ind][sweep,curr_layers_CSD_1[i],:] for i in range(4)]) 


# Average and SEM of SOs in different layers before and after pairing
mice_to_plot = [0,1,2,3,4,5]
to_plot_time = np.linspace(-2000,1999,4000)
# mice_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11,12]
to_plot_before = np.mean(LFP_SW_averages_layer[:,[0,1,2,3],:,:][mice_to_plot,:,:,:], axis = 1)
to_plot_after = np.mean(LFP_SW_averages_layer[:,[4,5,6,7,8,9],:,:][mice_to_plot,:,:,:], axis = 1)
fig, ax = plt.subplots(3,1, sharey = True, figsize = (7, 18))
for ind, ax1 in enumerate(list(ax.flatten())):
    # ax1.plot(to_plot_time, to_plot_before[:,ind,:].T)
    to_plot_mean = np.mean(to_plot_before[:,ind,:], axis = 0) # average across mice
    to_plot_err = np.std(to_plot_before[:,ind,:], axis = 0)/np.sqrt(len(mice_to_plot))
    ax1.plot(to_plot_time, to_plot_mean, color = 'k')
    ax1.fill_between(to_plot_time, to_plot_mean - to_plot_err, to_plot_mean + to_plot_err, alpha = 0.5, color = 'k')
    ax1.tick_params(axis="x", labelsize=16)    
    ax1.tick_params(axis="y", labelsize=16, size = 12)   
    ax1.set_xlabel('time (ms)', size=16)
    ax1.set_ylabel('LFP', size=16)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.tight_layout()
# fig, ax = plt.subplots(3,1, sharey = True, figsize = (7, 18))
# for ind, ax1 in enumerate(list(ax.flatten())):
    # ax1.plot(to_plot_time, to_plot_after[:,ind,:].T)
    to_plot_mean = np.mean(to_plot_after[:,ind,:], axis = 0)
    to_plot_err = np.std(to_plot_after[:,ind,:], axis = 0)/np.sqrt(len(mice_to_plot))
    ax1.plot(to_plot_time, to_plot_mean, color = 'c')
    ax1.fill_between(to_plot_time, to_plot_mean - to_plot_err, to_plot_mean + to_plot_err, alpha = 0.5, color = 'c')
    ax1.tick_params(axis="x", labelsize=16)    
    ax1.tick_params(axis="y", labelsize=16, size = 12)   
    ax1.set_xlabel('time (ms)', size=16)
    ax1.set_ylabel('LFP', size=16)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.tight_layout()

for day in range(13):
    fig, ax = plt.subplots(3,1,sharey = True)
    for ind, ax1 in enumerate(list(ax.flatten())):
        ax1.plot(to_plot_time, to_plot_before[day,ind,:], color = 'k')
        ax1.plot(to_plot_time, to_plot_after[day,ind,:], color = 'r')



#%% ----------------------------------------------------------------------- CHOOSE THE PEAK TO PEAK METRIC (PEAK TO PEAK OF AVERAGE WAVEFORM OR AVERAGE PEAK TO PEAK)

# ------------------------------------ CHOOSE WHICH AVERAGE SW WAVEFORM TO USE AND CALCULATE THE AMPLITUDES AND PEAKS ETC.. ------------------------------------------

LFP_waveform_to_use = LFP_SW_shift_averages_interp_all
CSD_waveform_to_use = CSD_SW_shift_averages_interp_all

LFP_SW_peak_to_peak_amp_all = [np.max(i[:,:,int(i.shape[2]/2):], axis = 2) - np.min(i[:,:,:int(i.shape[2]/2)], axis = 2) for i in LFP_waveform_to_use]
LFP_SW_famp_all = [np.min(i[:,:,:int(i.shape[2]/2)], axis = 2) for i in LFP_waveform_to_use]
LFP_SW_samp_all = [np.max(i[:,:,int(i.shape[2]/2):], axis = 2) for i in LFP_waveform_to_use]

CSD_SW_peak_to_peak_amp_all = [np.max(i, axis = 2) - np.min(i, axis = 2) for i in CSD_waveform_to_use]
CSD_SW_max_all = [np.min(i[:,:,:int(i.shape[2]/2)], axis = 2) for i in CSD_waveform_to_use]

LFP_SW_peak_to_peak_amp_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in LFP_SW_peak_to_peak_amp_all]
LFP_SW_famp_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in LFP_SW_famp_all]
LFP_SW_samp_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in LFP_SW_samp_all]

CSD_SW_peak_to_peak_amp_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in CSD_SW_peak_to_peak_amp_all]
CSD_SW_max_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in CSD_SW_max_all]

SW_peak_to_peak_shift_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in SW_peak_to_peak_shift_all]


# LFP_peak_to_peak_L5_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in LFP_peak_to_peak_L5_shift]
# CSD_peak_to_peak_L5_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in CSD_peak_to_peak_L5_shift]
# LFP_peak_to_peak_L5_median_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in LFP_peak_to_peak_L5_median_shift]
# CSD_peak_to_peak_L5_median_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in CSD_peak_to_peak_L5_median_shift]

LFP_peak_to_peak_L5_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in LFP_peak_to_peak_L5]
CSD_peak_to_peak_L5_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in CSD_peak_to_peak_L5]
LFP_peak_to_peak_L5_median_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in LFP_peak_to_peak_L5_median]
CSD_peak_to_peak_L5_median_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0) for i in CSD_peak_to_peak_L5_median]



# ------------------------------------- LAYER THE RESULTS ---------------------------------------------


LFP_SW_peak_to_peak_amp_layer = np.zeros([len(days), 5, 10])
LFP_SW_famp_layer = np.zeros([len(days), 5, 10])
LFP_SW_samp_layer = np.zeros([len(days), 5, 10])
CSD_SW_max_layer = np.zeros([len(days), 5, 10])
CSD_SW_peak_to_peak_layer = np.zeros([len(days), 5, 10])

LFP_SW_peak_to_peak_amp_layer_rel = np.zeros([len(days), 5, 10])
LFP_SW_famp_layer_rel = np.zeros([len(days), 5, 10])
LFP_SW_samp_layer_rel = np.zeros([len(days), 5, 10])
CSD_SW_max_layer_rel = np.zeros([len(days), 5, 10])
CSD_SW_peak_to_peak_layer_rel = np.zeros([len(days), 5, 10])

LFP_SW_peak_to_peak_amp_layer_1 = np.zeros([len(days), 4, 10])
LFP_SW_famp_layer_1 = np.zeros([len(days), 4, 10])
LFP_SW_samp_layer_1 = np.zeros([len(days), 4, 10])
CSD_SW_max_layer_1 = np.zeros([len(days), 4, 10])
CSD_SW_peak_to_peak_layer_1 = np.zeros([len(days), 4, 10])

LFP_SW_peak_to_peak_amp_layer_rel_1 = np.zeros([len(days), 4, 10])
LFP_SW_famp_layer_rel_1 = np.zeros([len(days), 4, 10])
LFP_SW_samp_layer_rel_1 = np.zeros([len(days), 4, 10])
CSD_SW_max_layer_rel_1 = np.zeros([len(days), 4, 10])
CSD_SW_peak_to_peak_layer_rel_1 = np.zeros([len(days), 4, 10])

SW_peak_to_peak_shift_layer_rel = np.zeros([len(days), 5, 10])
SW_peak_to_peak_shift_layer_rel_1 = np.zeros([len(days), 4, 10])



LFP_peak_to_peak_L5_layer_rel = np.zeros([len(days), 5, 10])
CSD_peak_to_peak_L5_layer_rel = np.zeros([len(days), 5, 10])
LFP_peak_to_peak_L5_median_layer_rel = np.zeros([len(days), 5, 10])
CSD_peak_to_peak_L5_median_layer_rel = np.zeros([len(days), 5, 10])

LFP_peak_to_peak_L5_layer_rel_1 = np.zeros([len(days), 4, 10])
CSD_peak_to_peak_L5_layer_rel_1 = np.zeros([len(days), 4, 10])
LFP_peak_to_peak_L5_median_layer_rel_1 = np.zeros([len(days), 4, 10])
CSD_peak_to_peak_L5_median_layer_rel_1 = np.zeros([len(days), 4, 10])



for m_ind in range(len(days)):
    for sweep in range(10): 
        curr_layers_LFP = layer_list_LFP[m_ind][0]
        curr_layers_CSD = layer_list_CSD[m_ind][0]
        curr_layers_LFP_1 = layer_list_LFP_1[m_ind][0]
        curr_layers_CSD_1 = layer_list_CSD_1[m_ind][0]

        # average peak to peak value
        LFP_SW_peak_to_peak_amp_layer[m_ind, :, sweep] = np.asarray([np.mean(LFP_SW_peak_to_peak_amp_all[m_ind][sweep, curr_layers_LFP[i]]) for i in range(5)])
        LFP_SW_famp_layer[m_ind, :, sweep] = np.asarray([np.mean(LFP_SW_famp_all[m_ind][sweep, curr_layers_LFP[i]]) for i in range(5)])
        LFP_SW_samp_layer[m_ind, :, sweep] = np.asarray([np.mean(LFP_SW_samp_all[m_ind][sweep, curr_layers_LFP[i]]) for i in range(5)]) 
        CSD_SW_max_layer[m_ind, :, sweep] = np.asarray([np.mean(CSD_SW_max_all[m_ind][sweep, curr_layers_CSD[i]]) for i in range(5)]) 
        CSD_SW_peak_to_peak_layer[m_ind, :, sweep] = np.asarray([np.mean(CSD_SW_peak_to_peak_amp_all[m_ind][sweep, curr_layers_CSD[i]]) for i in range(5)]) 
        
        LFP_SW_peak_to_peak_amp_layer_1[m_ind, :, sweep] = np.asarray([np.mean(LFP_SW_peak_to_peak_amp_all[m_ind][sweep, curr_layers_LFP_1[i]]) for i in range(4)])
        LFP_SW_famp_layer_1[m_ind, :, sweep] = np.asarray([np.mean(LFP_SW_famp_all[m_ind][sweep, curr_layers_LFP_1[i]]) for i in range(4)])
        LFP_SW_samp_layer_1[m_ind, :, sweep] = np.asarray([np.mean(LFP_SW_samp_all[m_ind][sweep, curr_layers_LFP_1[i]]) for i in range(4)]) 
        CSD_SW_max_layer_1[m_ind, :, sweep] = np.asarray([np.mean(CSD_SW_max_all[m_ind][sweep, curr_layers_CSD_1[i]]) for i in range(4)]) 
        CSD_SW_peak_to_peak_layer_1[m_ind, :, sweep] = np.asarray([np.mean(CSD_SW_peak_to_peak_amp_all[m_ind][sweep, curr_layers_CSD_1[i]]) for i in range(4)]) 

        LFP_SW_peak_to_peak_amp_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(LFP_SW_peak_to_peak_amp_all_rel[m_ind][sweep, curr_layers_LFP[i]]) for i in range(5)])
        LFP_SW_famp_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(LFP_SW_famp_all_rel[m_ind][sweep, curr_layers_LFP[i]]) for i in range(5)])
        LFP_SW_samp_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(LFP_SW_samp_all_rel[m_ind][sweep, curr_layers_LFP[i]]) for i in range(5)]) 
        CSD_SW_max_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(CSD_SW_max_all_rel[m_ind][sweep, curr_layers_CSD[i]]) for i in range(5)]) 
        CSD_SW_peak_to_peak_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(CSD_SW_peak_to_peak_amp_all_rel[m_ind][sweep, curr_layers_CSD[i]]) for i in range(5)]) 
        
        LFP_SW_peak_to_peak_amp_layer_rel_1[m_ind, :, sweep] = np.asarray([np.mean(LFP_SW_peak_to_peak_amp_all_rel[m_ind][sweep, curr_layers_LFP_1[i]]) for i in range(4)])
        LFP_SW_famp_layer_rel_1[m_ind, :, sweep] = np.asarray([np.mean(LFP_SW_famp_all_rel[m_ind][sweep, curr_layers_LFP_1[i]]) for i in range(4)])
        LFP_SW_samp_layer_rel_1[m_ind, :, sweep] = np.asarray([np.mean(LFP_SW_samp_all_rel[m_ind][sweep, curr_layers_LFP_1[i]]) for i in range(4)]) 
        CSD_SW_max_layer_rel_1[m_ind, :, sweep] = np.asarray([np.mean(CSD_SW_max_all_rel[m_ind][sweep, curr_layers_CSD_1[i]]) for i in range(4)]) 
        CSD_SW_peak_to_peak_layer_rel_1[m_ind, :, sweep] = np.asarray([np.mean(CSD_SW_peak_to_peak_amp_all_rel[m_ind][sweep, curr_layers_CSD_1[i]]) for i in range(4)]) 

        
        SW_peak_to_peak_shift_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(SW_peak_to_peak_shift_all_rel[m_ind][sweep, curr_layers_CSD[i]]) for i in range(5)]) 
        SW_peak_to_peak_shift_layer_rel_1[m_ind, :, sweep] = np.asarray([np.mean(SW_peak_to_peak_shift_all_rel[m_ind][sweep, curr_layers_CSD_1[i]]) for i in range(4)]) 
        
        
        # peak to peak on average waveform
        LFP_peak_to_peak_L5_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(LFP_peak_to_peak_L5_rel[m_ind][sweep, curr_layers_LFP[i]]) for i in range(5)]) 
        CSD_peak_to_peak_L5_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(CSD_peak_to_peak_L5_rel[m_ind][sweep, curr_layers_CSD[i]]) for i in range(5)]) 
        LFP_peak_to_peak_L5_median_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(LFP_peak_to_peak_L5_median_rel[m_ind][sweep, curr_layers_LFP[i]]) for i in range(5)]) 
        CSD_peak_to_peak_L5_median_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(CSD_peak_to_peak_L5_median_rel[m_ind][sweep, curr_layers_CSD[i]]) for i in range(5)]) 

        LFP_peak_to_peak_L5_layer_rel_1[m_ind, :, sweep] = np.asarray([np.mean(LFP_peak_to_peak_L5_rel[m_ind][sweep, curr_layers_LFP_1[i]]) for i in range(4)]) 
        CSD_peak_to_peak_L5_layer_rel_1[m_ind, :, sweep] = np.asarray([np.mean(CSD_peak_to_peak_L5_rel[m_ind][sweep, curr_layers_CSD_1[i]]) for i in range(4)]) 
        LFP_peak_to_peak_L5_median_layer_rel_1[m_ind, :, sweep] = np.asarray([np.mean(LFP_peak_to_peak_L5_median_rel[m_ind][sweep, curr_layers_LFP_1[i]]) for i in range(4)]) 
        CSD_peak_to_peak_L5_median_layer_rel_1[m_ind, :, sweep] = np.asarray([np.mean(CSD_peak_to_peak_L5_median_rel[m_ind][sweep, curr_layers_CSD_1[i]]) for i in range(4)]) 







# ------------------------------ PLOT -----------------------------------------------

# # SW peak to peak timecourse averaged per layer
# to_plot = LFP_peak_to_peak_L5_median_layer_rel
# fig, ax = plt.subplots()
# for i, layer in zip([1,2,3], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
#     ax.plot(np.nanmean(to_plot[:,i,:], axis = 0).T, label = str(layer))
#     to_plot_mean = np.nanmean(to_plot[:,i,:], axis = 0).T
#     to_plot_std = np.nanstd(to_plot[:,i,:], axis = 0).T
#     ax.fill_between(list(range(10)), to_plot_mean + 1*to_plot_std/np.sqrt(12), to_plot_mean - 1*to_plot_std/np.sqrt(12), alpha = 0.1)
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# fig.suptitle('SW peak to peak rel average within layer')
# plt.tight_layout()
# # # plt.savefig('SW peak to peak rel one channel per layer.jpg', dpi = 1000, format = 'jpg')
# # # plt.savefig('SW peak to peak rel one channel per layer.svg', dpi = 1000, format = 'svg')


# --------------------------------------------------------------------------- LFP

# SW peak to peak timecourse average per layer
to_plot = LFP_peak_to_peak_L5_layer_rel
fig, ax = plt.subplots()
for i, layer in zip([1,2,3], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
    ax.plot(np.nanmean(to_plot[:,i,:], axis = 0).T, label = str(layer))
    to_plot_mean = np.nanmean(to_plot[:,i,:], axis = 0).T
    to_plot_std = np.nanstd(to_plot[:,i,:], axis = 0).T
    ax.fill_between(list(range(10)), to_plot_mean + 1*to_plot_std/np.sqrt(12), to_plot_mean - 1*to_plot_std/np.sqrt(12), alpha = 0.1)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()
fig.suptitle('SW peak to peak rel avg channel per layer')
plt.savefig('LFP SW peak to peak avg channel per layer.jpg', dpi = 1000, format = 'jpg')
plt.savefig('LFP SW peak to peak avg channel per layer.svg', dpi = 1000, format = 'svg')

layers_for_anova = [1,2,3] #don't use layer 6 as buggy 
LFP_SW_peak_to_peak_rel_for_ANOVA = np.zeros([len(days), 30])
curr_for_ANOVA = LFP_peak_to_peak_L5_layer_rel[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
# LFP_SW_peak_to_peak_rel_1_for_ANOVA = LFP_SW_peak_to_peak_rel_1_for_ANOVA[LFP_SW_peak_to_peak_rel_1_for_ANOVA[:,-1].argsort()]
for layer in range(3):
    LFP_SW_peak_to_peak_rel_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
np.savetxt('LFP SW peak to peak 1 channel per layer.csv', LFP_SW_peak_to_peak_rel_for_ANOVA, delimiter = ',')

fig, ax = plt.subplots(3,5, sharey = True)
for ax1_ind, ax1 in enumerate(list(ax.flatten())):
    if ax1_ind >= len(days):
        continue
    for i, layer in zip([1,2,3], ['layer 2/3', 'layer 4', 'layer 5']):
        ax1.plot(LFP_peak_to_peak_L5_median_layer_rel[ax1_ind,i,:], label = str(layer))
    handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.suptitle('SW peak to peak one channel per layer')
plt.tight_layout()





# SW peak to peak timecourse in one channel per layer
to_plot = LFP_peak_to_peak_L5_layer_rel_1
fig, ax = plt.subplots()
for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
    ax.plot(np.nanmean(to_plot[:,i,:], axis = 0).T, label = str(layer))
    to_plot_mean = np.nanmean(to_plot[:,i,:], axis = 0).T
    to_plot_std = np.nanstd(to_plot[:,i,:], axis = 0).T
    ax.fill_between(list(range(10)), to_plot_mean + 1*to_plot_std/np.sqrt(12), to_plot_mean - 1*to_plot_std/np.sqrt(12), alpha = 0.1)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()
fig.suptitle('SW peak to peak rel one channel per layer')
plt.savefig('LFP SW peak to peak 1 channel per layer.jpg', dpi = 1000, format = 'jpg')
plt.savefig('LFP SW peak to peak 1 channel per layer.svg', dpi = 1000, format = 'svg')

layers_for_anova = [0,1,2] #don't use layer 6 as buggy 
LFP_SW_peak_to_peak_rel_1_for_ANOVA = np.zeros([len(days), 30])
curr_for_ANOVA = LFP_peak_to_peak_L5_layer_rel_1[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
# LFP_SW_peak_to_peak_rel_1_for_ANOVA = LFP_SW_peak_to_peak_rel_1_for_ANOVA[LFP_SW_peak_to_peak_rel_1_for_ANOVA[:,-1].argsort()]
for layer in range(3):
    LFP_SW_peak_to_peak_rel_1_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
np.savetxt('LFP SW peak to peak 1 channel per layer.csv', LFP_SW_peak_to_peak_rel_1_for_ANOVA, delimiter = ',')

fig, ax = plt.subplots(3,5, sharey = True)
for ax1_ind, ax1 in enumerate(list(ax.flatten())):
    if ax1_ind >= len(days):
        continue
    for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5']):
        ax1.plot(to_plot[ax1_ind,i,:], label = str(layer))
    handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.suptitle('SW peak to peak one channel per layer')
plt.tight_layout()




# --------------------------------------------------------------------------------------------  CSD

# CSD peak to peak timecourse avg channel per layer --> this doesn't work for layer 2/3 because some channels above the one as layer 2 are buggy with CSD
to_plot = CSD_peak_to_peak_L5_layer_rel
fig, ax = plt.subplots()
for i, layer in zip([1,2,3], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
    ax.plot(np.nanmean(to_plot[:,i,:], axis = 0).T, label = str(layer))
    to_plot_mean = np.nanmean(to_plot[:,i,:], axis = 0).T
    to_plot_std = np.nanstd(to_plot[:,i,:], axis = 0).T
    ax.fill_between(list(range(10)), to_plot_mean + 1*to_plot_std/np.sqrt(12), to_plot_mean - 1*to_plot_std/np.sqrt(12), alpha = 0.1)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.suptitle('CSD peak to peak avg channel per layer')
plt.tight_layout()
ax.set_ylim([0.6, 1.15])
plt.savefig('CSD SW peak to peak avg channel per layer.jpg', dpi = 1000, format = 'jpg')
plt.savefig('CSD SW peak to peak avg channel per layer.svg', dpi = 1000, format = 'svg')

layers_for_anova = [1,2,3] #don't use layer 6 as buggy 
CSD_SW_peak_to_peak_rel_for_ANOVA = np.zeros([len(days), 30])
curr_for_ANOVA = CSD_peak_to_peak_L5_layer_rel[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
# LFP_SW_peak_to_peak_rel_1_for_ANOVA = LFP_SW_peak_to_peak_rel_1_for_ANOVA[LFP_SW_peak_to_peak_rel_1_for_ANOVA[:,-1].argsort()]
for layer in range(3):
    CSD_SW_peak_to_peak_rel_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
np.savetxt('CSD SW peak to peak 1 channel per layer.csv', CSD_SW_peak_to_peak_rel_for_ANOVA, delimiter = ',')

fig, ax = plt.subplots(3,5, sharey = True)
for ax1_ind, ax1 in enumerate(list(ax.flatten())):
    if ax1_ind >= len(days):
        continue
    for i, layer in zip([1,2,3], ['layer 2/3', 'layer 4', 'layer 5']):
        ax1.plot(CSD_peak_to_peak_L5_layer_rel[ax1_ind,i,:], label = str(layer))
    handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()




# CSD peak to peak timecourse in one channel per layer
to_plot = CSD_peak_to_peak_L5_layer_rel_1
fig, ax = plt.subplots()
for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
    ax.plot(np.nanmean(to_plot[:,i,:], axis = 0).T, label = str(layer))
    to_plot_mean = np.nanmean(to_plot[:,i,:], axis = 0).T
    to_plot_std = np.nanstd(to_plot[:,i,:], axis = 0).T
    ax.fill_between(list(range(10)), to_plot_mean + 1*to_plot_std/np.sqrt(12), to_plot_mean - 1*to_plot_std/np.sqrt(12), alpha = 0.1)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.suptitle('CSD peak to peak one channel per layer')
plt.tight_layout()
ax.set_ylim([0.6, 1.15])
plt.savefig('CSD SW peak to peak 1 channel per layer.jpg', dpi = 1000, format = 'jpg')
plt.savefig('CSD SW peak to peak 1 channel per layer.svg', dpi = 1000, format = 'svg')

layers_for_anova = [0,1,2] #don't use layer 6 as buggy 
CSD_SW_peak_to_peak_rel_1_for_ANOVA = np.zeros([len(days), 30])
curr_for_ANOVA = CSD_peak_to_peak_L5_layer_rel_1[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
# LFP_SW_peak_to_peak_rel_1_for_ANOVA = LFP_SW_peak_to_peak_rel_1_for_ANOVA[LFP_SW_peak_to_peak_rel_1_for_ANOVA[:,-1].argsort()]
for layer in range(3):
    CSD_SW_peak_to_peak_rel_1_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
# CSD_SW_peak_to_peak_rel_1_for_ANOVA = CSD_SW_peak_to_peak_rel_1_for_ANOVA[CSD_SW_peak_to_peak_rel_1_for_ANOVA[:,-1].argsort()]
np.savetxt('CSD SW peak to peak 1 channel per layer.csv', CSD_SW_peak_to_peak_rel_1_for_ANOVA, delimiter = ',')

fig, ax = plt.subplots(3,5, sharey = True)
for ax1_ind, ax1 in enumerate(list(ax.flatten())):
    if ax1_ind >= len(days):
        continue
    for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5']):
        ax1.plot(CSD_peak_to_peak_L5_layer_rel_1[ax1_ind,i,:], label = str(layer))
    handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()

# fig, ax = plt.subplots(3,5, sharey = True)
# for ax1_ind, ax1 in enumerate(list(ax.flatten())):
#     if ax1_ind >= len(days):
#         continue
#     for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5']):
#         ax1.plot(CSD_peak_to_peak_L5_median_layer_rel_1[ax1_ind,i,:], label = str(layer))
#     handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# plt.tight_layout()
