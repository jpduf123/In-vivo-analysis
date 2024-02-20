# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:27:28 2023

@author: Mann Lab
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
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

b_notch, a_notch = scipy.signal.iirnotch(50, 100, 1000)

#%% ----------------------------------------------------- channel-layer selection

mouse_1 = list(map(np.asarray, [[2,3,4,], [5,6,7,8,9], [10,11], [12,13,14,15,16,17], [19,20,21,22,23,24,25,26]]))
mouse_2 = list(map(np.asarray, [[0,1,2,], [3,4,5,6,7,8,9], [10,11], [12,13,14,15,16,17,18,19], [20,21,22,23,24,25]]))
mouse_3 = list(map(np.asarray, [[0,1,2,3], [4,5,6,7,8,9], [10,11], [12,13,14,15,16,17,18,19], [20,21,22,23,24,25,26]]))
mouse_4 = list(map(np.asarray, [[2,3,4,5], [6,7,8,9,10,11], [12,13], [14,15,16,17,18], [19,20,21,22,23,24,25]]))


layer_dict = {'160801' : [mouse_1]*10,
                 
            '160803' : [mouse_2]*10,

            '160804' : [mouse_3]*4 + [[i + 1 for i in mouse_3]]*6,
            
            '160811' : [mouse_4]*4 + [[i + 1 for i in mouse_4]]*6,
            }

layer_list_LFP = list(layer_dict.values())
layer_list_CSD = copy.deepcopy(layer_list_LFP)

mouse_1_1 = list(map(np.asarray, [[8], [11], [15], [22]]))
mouse_2_1 = list(map(np.asarray, [[11], [14], [19], [24]]))
mouse_3_1 = list(map(np.asarray, [[7], [11], [16], [22]]))
mouse_4_1 = list(map(np.asarray, [[9], [13], [17], [21]]))

layer_dict_1 = {'160801' : [mouse_1_1]*10,
                 
            '160803' : [mouse_2_1]*10,

            '160804' : [mouse_3_1]*4 + [[i + 1 for i in mouse_3_1]]*6,
            
            '160811' : [mouse_4_1]*10,   
            }

layer_list_LFP_1 = list(layer_dict_1.values())
layer_list_CSD_1 = copy.deepcopy(layer_list_LFP_1)


#%% ----------------------------------------------------- example of activity before and after AP/5 application

chans_for_plot = np.arange(0,27)

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# for day in ['160801']:
for day in days:
    print(day)
    os.chdir(day)
    
    os.chdir('pre_AP5')
    LFP = pickle.load(open('LFP_resampled','rb'))[3][chanMap_32,:]
    MUA_power_binned = pickle.load(open('MUA_all_sweeps','rb'))[3][chanMap_32,:]
    # show one ON state
    time_start = 104
    time_stop = 108
    fig, ax = plt.subplots(figsize = (4,10))
    spacer = 2000
    for i in chans_for_plot:
        ax.plot(scipy.signal.filtfilt(b_notch, a_notch, LFP[i,:]) -  i *spacer, linewidth = 1, color = 'k')
        ax.set_xlim(time_start*new_fs,time_stop*new_fs)
        # ax.set_yticks(ticks = np.linspace(0, 31000, 32), labels = list(np.linspace(0, 31, 32, dtype = int)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('example SOs pre AP5 LFP.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('example SOs pre AP5 LFP.pdf', format = 'pdf', dpi = 1000)
    
    # fig, ax = plt.subplots(figsize = (4,10))
    # MUA_to_plot = MUA_power_binned
    # # MUA_to_plot = MUA_power_binned*np.abs(MUA_power_binned) 
    # spacer = np.max(np.abs(MUA_to_plot[chans_for_plot[:25],int(time_start*new_fs):int(time_stop*new_fs)]))*1.3
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
    # plt.savefig('example SOs pre AP5 MUA.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('example SOs pre AP5 MUA.pdf', format = 'pdf', dpi = 1000)

    os.chdir('..')
    
    
    LFP = pickle.load(open('LFP_resampled','rb'))[1][chanMap_32,:]
    MUA_power_binned = pickle.load(open('MUA_all_sweeps','rb'))[1][chanMap_32,:]
    # show one ON state
    time_start = 104
    time_stop = 108
    fig, ax = plt.subplots(figsize = (4,10))
    spacer = 2000
    for i in chans_for_plot:
        ax.plot(scipy.signal.filtfilt(b_notch, a_notch, LFP[i,:]) -  i *spacer, linewidth = 1, color = 'k')
        ax.set_xlim(time_start*new_fs,time_stop*new_fs)
        # ax.set_yticks(ticks = np.linspace(0, 31000, 32), labels = list(np.linspace(0, 31, 32, dtype = int)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('example SOs post AP5 LFP.jpg', format = 'jpg', dpi = 1000)
    plt.savefig('example SOs post AP5 LFP.pdf', format = 'pdf', dpi = 1000)
     
    # fig, ax = plt.subplots(figsize = (4,10))
    # MUA_to_plot = MUA_power_binned
    # # MUA_to_plot = MUA_power_binned*np.abs(MUA_power_binned)
    # spacer = np.max(np.abs(MUA_to_plot[chans_for_plot[:25],int(time_start*new_fs):int(time_stop*new_fs)]))*1.3
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
    # plt.savefig('example SOs post AP5 MUA.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('example SOs post AP5 MUA.pdf', format = 'pdf', dpi = 1000)
 
    os.chdir('..')


#%% 1) -------------------------------------------------- effect of AP5 on whisker responses

do_shift = True
highpass_cutoff = 4
chanMap = chanMap_32
nchans = 32
CSD_smooth_over = 1

LFP_min_layer_pre_AP5 = np.zeros([4, 3, 4]) # mouse, layer, sweep
PSTH_magn_layer_pre_AP5 = np.zeros([4, 3, 4])
CSD_max_layer_pre_AP5 = np.zeros([4, 3, 4])
CSD_absmax_layer_pre_AP5 = np.zeros([4, 3, 4])

LFP_min_layer_post_AP5 = np.zeros([4, 3, 4])
PSTH_magn_layer_post_AP5 = np.zeros([4, 3, 4])
CSD_max_layer_post_AP5 = np.zeros([4, 3, 4])
CSD_absmax_layer_post_AP5 = np.zeros([4, 3, 4])


days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# for day in ['160810']:
for m_ind, day in enumerate(days):
    print(day)
    os.chdir(day)
    
    # ------------------------------------------------------------- load up pre AP5
    os.chdir('pre_AP5')
    # LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))#
    # spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    # stim_times = pickle.load(open('stim_times','rb'))
    # # nchans = LFP_all_sweeps[0].shape[0]
    # # def LFP_average(sweeps_to_plot, stims = stim_times, LFP_all_sweeps = LFP_all_sweeps):
    # #     to_plot = np.zeros([len(LFP_all_sweeps[0]), int(0.6*new_fs), len(sweeps_to_plot)])    
    # #     for ind_sweep, sweep in enumerate(sweeps_to_plot):
    # #         curr_to_plot = np.zeros([len(LFP_all_sweeps[0]), int(0.6*new_fs), len(stims[sweep])])
    # #         for ind_stim, stim in enumerate(list(stims[sweep])):
    # #             if ind_stim == len(stims[sweep]) - 1:
    # #                 break
    # #             if stim < 0.3*new_fs:
    # #                 continue
    # #             if stim + 0.3*new_fs > LFP_all_sweeps[sweep].shape[1]:
    # #                 continue
    # #             else:
    # #                 curr_to_plot[:,:,ind_stim] = LFP_all_sweeps[sweep][:,int(stim - 0.2*new_fs):int(stim + 0.4*new_fs)]
    # #         to_plot[:,:,ind_sweep] = np.squeeze(np.mean(curr_to_plot,2)) # average across stims
    # #     return np.squeeze(np.mean(to_plot,2)) #average across sweeps
    # # LFP_responses = np.zeros([len(LFP_all_sweeps), nchans, 600])
    # # LFP_responses[:] = np.NaN
    # # for sweep in range(len(LFP_all_sweeps)):
    # #     LFP_responses[sweep, :, :] = LFP_average([sweep], stims = stim_times)
    # # np.save('LFP_responses.npy', LFP_responses)

    # def PSTH_matrix(sweeps_to_plot, take_out_artifacts = True, artifact_locs = [], stims = stim_times):
    #     to_plot = np.zeros([299,nchans,len(sweeps_to_plot)])
    #     for ind_sweep, sweep in enumerate(sweeps_to_plot):
    #         #PSTH_matrix is mean across trials in one sweep
    #         PSTH_matrix = np.zeros([299,nchans])
    #         bins = np.linspace(1,300,300)
    #         for ind_chan, chan in enumerate(list(spikes_allsweeps[sweep].keys())):
    #             currstim = np.zeros([299,len(stims[sweep])])
    #             for ind_stim, j in enumerate(list(stims[sweep])):
    #                 currstim[:,ind_stim] = np.histogram((spikes_allsweeps[sweep][chan][(j - 0.1*new_fs < spikes_allsweeps[sweep][chan]) & (spikes_allsweeps[sweep][chan] < j+0.2*new_fs)] - (j-0.1*new_fs)), bins)[0]
    #                 if take_out_artifacts:
    #                     currstim[:,ind_stim][artifact_locs] = 0
    #             PSTH_matrix[:,ind_chan] = np.squeeze(np.mean(currstim, 1)) # mean across stims for every channel
    #         to_plot[:,:,ind_sweep] = PSTH_matrix
    #     return np.squeeze(np.mean(to_plot,2))

    # # save PSTH for every sweep and channel
    # PSTH_responses = np.zeros([len(LFP_all_sweeps), nchans, 299])
    # PSTH_responses[:] = np.NaN
    # for sweep in range(len(LFP_all_sweeps)):
    #     PSTH_responses[sweep, :, :] = np.transpose(PSTH_matrix([sweep]))
    # np.save('PSTH_responses.npy', PSTH_responses[:,chanMap,:])
    # os.chdir('..')
    # os.chdir('..')

    LFP_responses_pre_AP5 = np.load('LFP_responses.npy')[:,chanMap,:]
    LFP_min_pre_AP5 = np.empty([4, nchans])
    LFP_min_pre_AP5[:] = np.NaN
    for sweep in range(4):
        LFP_min_pre_AP5[sweep,:] = np.abs(np.min(LFP_responses_pre_AP5[sweep,:,200:300], 1) - LFP_responses_pre_AP5[sweep,:,210])
    LFP_responses_pre_AP5 = scipy.signal.filtfilt(b_notch, a_notch, LFP_responses_pre_AP5) # take out 50Hz noise
    
    PSTH_responses_pre_AP5 = np.load('PSTH_responses.npy')
    PSTH_magn_pre_AP5 = np.empty([4, nchans])
    PSTH_magn_pre_AP5[:] = np.NaN
    for sweep in range(4):
        PSTH_magn_pre_AP5[sweep,:] = np.sum(PSTH_responses_pre_AP5[sweep,:,110:200], axis = 1) - np.sum(PSTH_responses_pre_AP5[sweep,:,:100], axis = 1)
        # PSTH_magn_pre_AP5[sweep,:] = np.sum(PSTH_responses_pre_AP5[sweep,:,110:200], axis = 1)

    LFP_for_CSD = scipy.ndimage.gaussian_filter(LFP_responses_pre_AP5, (0, CSD_smooth_over, 0))
    CSD_matrix = -np.eye(nchans) # 
    for j in range(1, CSD_matrix.shape[0] - 1):
        CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
    CSD_pre_AP5 = - np.asarray([np.dot(CSD_matrix, LFP_for_CSD[i]) for i in range(4)])
    CSD_pre_AP5[:,0,:] = 0
    CSD_pre_AP5[:,-1,:] = 0
    
    CSD_max_pre_AP5 = np.empty([4, nchans])
    CSD_max_pre_AP5[:] = np.NaN
    CSD_absmax_pre_AP5 = np.empty([4, nchans])
    CSD_absmax_pre_AP5[:] = np.NaN
    for sweep in range(4):
        CSD_max_pre_AP5[sweep,:] = np.abs(np.min(CSD_pre_AP5[sweep,:,200:300], 1))
        CSD_absmax_pre_AP5[sweep,:] = np.max(np.abs(CSD_pre_AP5[sweep,:,200:300]),1)
    os.chdir('..')
    

    
    # -------------------------------------------------------------------------- load up post AP5
    os.chdir(f'analysis_{day}')
    LFP_responses_post_AP5 = np.load('LFP_responses.npy')[:,chanMap,:]
    LFP_min_post_AP5 = np.loadtxt('LFP_min.csv', delimiter = ',')[:,chanMap]
    LFP_responses_post_AP5 = scipy.signal.filtfilt(b_notch, a_notch, LFP_responses_post_AP5) # take out 50Hz noise
    PSTH_responses_post_AP5 = np.load('PSTH_responses.npy')
    PSTH_magn_post_AP5 = np.empty([4, nchans])
    PSTH_magn_post_AP5[:] = np.NaN
    for sweep in range(4):
        PSTH_magn_post_AP5[sweep,:] = np.sum(PSTH_responses_post_AP5[sweep,:,110:200], axis = 1) - np.sum(PSTH_responses_post_AP5[sweep,:,:100], axis = 1)
        # PSTH_magn_post_AP5[sweep,:] = np.sum(PSTH_responses_post_AP5[sweep,:,110:200], axis = 1)
    # PSTH_magn_post_AP5 = np.loadtxt('PSTH_resp_magn.csv', delimiter = ',')[:,chanMap]

    LFP_for_CSD = scipy.ndimage.gaussian_filter(LFP_responses_post_AP5, (0, CSD_smooth_over, 0))
    CSD_matrix = -np.eye(nchans) # 
    for j in range(1, CSD_matrix.shape[0] - 1):
        CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
    CSD_post_AP5 = - np.asarray([np.dot(CSD_matrix, LFP_for_CSD[i]) for i in range(4)])
    CSD_post_AP5[:,0,:] = 0
    CSD_post_AP5[:,-1,:] = 0
    
    CSD_max_post_AP5 = np.empty([4, nchans])
    CSD_max_post_AP5[:] = np.NaN
    CSD_absmax_post_AP5 = np.empty([4, nchans])
    CSD_absmax_post_AP5[:] = np.NaN
    for sweep in range(4):
        CSD_max_post_AP5[sweep,:] = np.abs(np.min(CSD_post_AP5[sweep,:,200:300], 1))
        CSD_absmax_post_AP5[sweep,:] = np.max(np.abs(CSD_post_AP5[sweep,:,200:300]),1)
    os.chdir('..')

    
    
    
    # -------------------------------------------------------------------------- layer
    if day == '160811': # shifted one down between pre and post AP5
        curr_layers_1_avg_pre = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1])-1, 0, nchans-1)) for i in layer_dict_1[day][0]]
    else:
        curr_layers_1_avg_pre = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
    curr_layers_1_avg_post = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
    
    for sweep in range(4):
        # average across channels in each layer (plus minus one from selected channel)
        LFP_min_layer_pre_AP5[m_ind, :, sweep] = np.asarray([np.mean(LFP_min_pre_AP5[sweep, curr_layers_1_avg_pre[i]]) for i in range(3)])
        PSTH_magn_layer_pre_AP5[m_ind, :, sweep] = np.asarray([np.nanmedian(PSTH_magn_pre_AP5[sweep,curr_layers_1_avg_pre[i]]) for i in range(3)])
        CSD_max_layer_pre_AP5[m_ind, :, sweep] = np.asarray([np.mean(CSD_max_pre_AP5[sweep,curr_layers_1_avg_pre[i]]) for i in range(3)]) 
        CSD_absmax_layer_pre_AP5[m_ind, :, sweep] = np.asarray([np.mean(CSD_absmax_pre_AP5[sweep,curr_layers_1_avg_pre[i]]) for i in range(3)]) 

        LFP_min_layer_post_AP5[m_ind, :, sweep] = np.asarray([np.mean(LFP_min_post_AP5[sweep, curr_layers_1_avg_post[i]]) for i in range(3)])
        PSTH_magn_layer_post_AP5[m_ind, :, sweep] = np.asarray([np.nanmedian(PSTH_magn_post_AP5[sweep,curr_layers_1_avg_post[i]]) for i in range(3)])
        CSD_max_layer_post_AP5[m_ind, :, sweep] = np.asarray([np.mean(CSD_max_post_AP5[sweep,curr_layers_1_avg_post[i]]) for i in range(3)]) 
        CSD_absmax_layer_post_AP5[m_ind, :, sweep] = np.asarray([np.mean(CSD_absmax_post_AP5[sweep,curr_layers_1_avg_post[i]]) for i in range(3)]) 


    # # --------------------------------------------------------------------------- LFP plots
    # #4 sweeps pre AP5, 4 sweeps post AP5 LFP
    # spacer = np.max(np.abs(LFP_responses_pre_AP5[:,:,200:300]))
    # fig, ax = plt.subplots(1,8, figsize = (8,15), sharey = True)
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     for chan in range(nchans):
    #         if ind < 4: #pre AP5
    #             ax1.plot(LFP_responses_pre_AP5[ind,chan,:] + chan * -spacer, 'b', linewidth = 1)                 
    #             ax1.set_xlim([150,400])
    #         else:
    #             ax1.plot(LFP_responses_post_AP5[ind-4,chan,:] + chan * -spacer, 'b', linewidth = 1)                 
    #             ax1.set_xlim([150,400])

    #     # ax1.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, tot_chans))
    #     # ax1.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 8)
    # plt.tight_layout()
    # plt.savefig('LFP response before after AP5 all sweeps.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('LFP response before after AP5 all sweeps.pdf', format = 'pdf', dpi = 1000)

    # # average LFP response before and after
    # spacer = np.max(np.abs(LFP_responses_pre_AP5[:,:,200:300]))/2
    # fig, ax = plt.subplots(figsize = (3,5))
    # for chan in range(nchans):
    #     ax.plot(np.mean(LFP_responses_pre_AP5[:,chan,:], axis = 0) + chan * -spacer, 'k', linewidth = 1.5)                 
    #     ax.plot(np.mean(LFP_responses_post_AP5[:,chan,:], axis = 0) + chan * -spacer, 'r', linewidth = 1.5)                     
    #     ax.set_xlim([150,300])
    # # ax.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, 5))
    # # ax.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 6)
    # ax.set_yticks(np.linspace(-31*spacer, 0, 5))
    # ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    # ax.set_xticks([150,200,250,300])
    # ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 14)
    # ax.set_ylabel('depth (mm)', size = 14)
    # ax.set_xlabel('time from stim (ms)', size = 14)
    # ax.set_ylim(bottom = -31*spacer - spacer/2)
    # ax.set_ylim(top = spacer + spacer/2)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tight_layout()
    # plt.savefig('LFP response before after AP5.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('LFP response before after AP5.pdf', format = 'pdf', dpi = 1000)



    # # --------------------------------------------------------------------------- CSD plots
    # ##4 sweeps pre AP5, 4 sweeps post AP5 CSD
    # spacer = np.max(np.abs(CSD_pre_AP5[:,:,200:300]))/2
    # fig, ax = plt.subplots(1,8, figsize = (15,15), sharey = True)
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     for chan in range(nchans):
    #         if ind < 4: #pre AP5
    #             ax1.plot(CSD_pre_AP5[ind,chan,:] + chan * -spacer, 'b', linewidth = 1)                 
    #             ax1.set_xlim([150,400])
    #         else:
    #             ax1.plot(CSD_post_AP5[ind-4,chan,:] + chan * -spacer, 'b', linewidth = 1)                 
    #             ax1.set_xlim([150,400])

    #     ax1.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, tot_chans))
    #     ax1.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 8)
    # plt.tight_layout()
    # plt.savefig('CSD response before after AP5 all sweeps.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('CSD response before after AP5 all sweeps.pdf', format = 'pdf', dpi = 1000)

    # # average CSD response before and after
    # spacer = np.max(np.abs(CSD_pre_AP5[:,:,200:300]))/2
    # fig, ax = plt.subplots(figsize = (3,5))
    # for chan in range(nchans):
    #     ax.plot(np.mean(CSD_pre_AP5[:,chan,:], axis = 0) + chan * -spacer, 'k', linewidth = 1.5)                 
    #     ax.plot(np.mean(CSD_post_AP5[:,chan,:], axis = 0) + chan * -spacer, 'r', linewidth = 1.5)                     
    #     ax.set_xlim([150,300])
    # ax.set_yticks(np.linspace(-31*spacer, 0, 5))
    # # ax.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 6)
    # # ax.set_yticks(np.linspace(31, 0, 5))
    # ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    # ax.set_xticks([150,200,250,300])
    # ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 14)
    # ax.set_ylabel('depth (mm)', size = 14)
    # ax.set_xlabel('time from stim (ms)', size = 14)
    # ax.set_ylim(bottom = -31*spacer - spacer/2)
    # ax.set_ylim(top = spacer + spacer/2)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tight_layout()
    # plt.savefig('CSD response before after AP5.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('CSD response before after AP5.pdf', format = 'pdf', dpi = 1000)




    # # --------------------------------------------------------------------------- MUA plots
    # # 4 sweeps pre AP5, 4 sweeps post AP5 CSD
    # spacer = np.max(np.abs(PSTH_responses_pre_AP5[:,:,100:200]))/2
    # fig, ax = plt.subplots(1,8, figsize = (15,15), sharey = True)
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     for chan in range(nchans):
    #         if ind < 4: #pre AP5
    #             ax1.plot(PSTH_responses_pre_AP5[ind, chan,:] + chan * -spacer, 'b', linewidth = 1)                 
    #         else:
    #             ax1.plot(PSTH_responses_post_AP5[ind-4, chan,:] + chan * -spacer, 'b', linewidth = 1)                 

    #     # ax1.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, tot_chans))
    #     # ax1.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 8)
    # plt.tight_layout()
    # plt.savefig('MUA response before after AP5 all sweeps.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('MUA response before after AP5 all sweeps.pdf', format = 'pdf', dpi = 1000)

    # PSTH_responses_pre_AP5[:,26:,:] = PSTH_responses_pre_AP5[:,26:,:]/5
    # PSTH_responses_post_AP5[:,26:,:] = PSTH_responses_post_AP5[:,26:,:]/5
    # # average CSD response before and after
    # spacer = np.max(np.abs(PSTH_responses_pre_AP5[:,:,100:200]))/2
    # fig, ax = plt.subplots(figsize = (3,5))
    # for chan in range(nchans):
    #     ax.plot(scipy.ndimage.gaussian_filter(np.mean(PSTH_responses_pre_AP5[:,chan,:], axis = 0), 2) + chan * -spacer, 'k', linewidth = 1.5)                 
    #     ax.plot(scipy.ndimage.gaussian_filter(np.mean(PSTH_responses_post_AP5[:,chan,:], axis = 0), 2) + chan * -spacer, 'r', linewidth = 1.5)                     
    # # ax.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, 5))
    # # ax.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 6)
    # ax.set_xlim([50,200])
    # ax.set_yticks(np.linspace(-31*spacer, 0, 5))
    # ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    # ax.set_xticks([50,100,150,200])
    # ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 14)
    # ax.set_ylabel('depth (mm)', size = 14)
    # ax.set_xlabel('time from stim (ms)', size = 14)
    # ax.set_ylim(bottom = -31*spacer - spacer/2)
    # ax.set_ylim(top = spacer + spacer/2)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tight_layout()
    # plt.savefig('MUA response before after AP5.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('MUA response before after AP5.pdf', format = 'pdf', dpi = 1000)

    os.chdir('..')



# Bar plot values pre vs post AP5
# 
# LFP
fig, ax = plt.subplots(figsize = (5,5))
rel_avg = np.mean(LFP_min_layer_post_AP5, axis = 2)/np.mean(LFP_min_layer_pre_AP5, axis = 2) # average across sweeps
print(np.mean(rel_avg, axis = 0))
print(np.std(rel_avg, axis = 0))
ax.bar([0,1,2], np.mean(rel_avg, axis = 0)[:3]*100, color = 'k', yerr = np.std(rel_avg, axis = 0)*100/2, capsize = 10)
ax.set_ylim([0,100])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['Layer 2/3', 'Layer 4', 'Layer 5'], rotation = 45)
ax.tick_params(axis="x", labelsize=18)    
ax.tick_params(axis="y", labelsize=18) 
ax.set_ylabel('LFP magnitude (% pre-AP5)', size = 18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) 
plt.tight_layout()
# plt.savefig('LFP pre vs post AP5 layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('LFP pre vs post AP5 layer.pdf', dpi = 1000, format = 'pdf')

# # MUA
fig, ax = plt.subplots(figsize = (5,5))
rel_avg = np.mean(PSTH_magn_layer_post_AP5, axis = 2)/np.mean(PSTH_magn_layer_pre_AP5, axis = 2) # average across sweeps
print(np.mean(rel_avg, axis = 0))
print(np.std(rel_avg, axis = 0))
ax.bar([0,1,2], np.mean(rel_avg, axis = 0)[:3]*100, color = 'k', yerr = np.std(rel_avg, axis = 0)*100/2, capsize = 10)
ax.set_ylim([0,100])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['Layer 2/3', 'Layer 4', 'Layer 5'], rotation = 45)
ax.tick_params(axis="x", labelsize=18)    
ax.tick_params(axis="y", labelsize=18) 
ax.set_ylabel('MUA magnitude (% pre-AP5)', size = 18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)  
plt.tight_layout()
# plt.savefig('MUA pre vs post AP5 layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('MUA pre vs post AP5 layer.pdf', dpi = 1000, format = 'pdf')

# CSD
fig, ax = plt.subplots(figsize = (5,5))
rel_avg = np.mean(CSD_absmax_layer_post_AP5, axis = 2)/np.mean(CSD_absmax_layer_pre_AP5, axis = 2) # average across sweeps
print(np.mean(rel_avg, axis = 0))
print(np.std(rel_avg, axis = 0))
ax.bar([0,1,2], np.mean(rel_avg, axis = 0)[:3]*100, color = 'k', yerr = np.std(rel_avg, axis = 0)*100/2, capsize = 10)
ax.set_ylim([0,100])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['Layer 2/3', 'Layer 4', 'Layer 5'], rotation = 45)
ax.tick_params(axis="x", labelsize=18)    
ax.tick_params(axis="y", labelsize=18)   
ax.set_ylabel('CSD magnitude (% pre-AP5)', size = 18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# plt.savefig('CSD pre vs post AP5 layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('CSD pre vs post AP5 layer.pdf', dpi = 1000, format = 'pdf')





    

#%% 2) -------------------------------------------------- effect of AP5 on depression after pairing, compared to control with no AP5

smooth_CSD_over_channels = True
smooth_over_channel_count = 1
chanMap = chanMap_32
nchans = 32

LFP_all = []
LFP_shift_all = []
PSTH_all = []
PSTH_shift_all = []
CSD_all = []
CSD_shift_all = []

# os.chdir(home_path)
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for day_ind, day in enumerate(days):
# for day_ind, day in enumerate(days[11:]):
    os.chdir(day)
    print(day)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_responses = np.load('LFP_responses.npy')[:,chanMap,:]
    LFP_responses = scipy.signal.filtfilt(b_notch, a_notch, LFP_responses) # take out 50Hz noise
    LFP_all.append(LFP_responses)
    PSTH_all.append(np.load('PSTH_responses.npy'))
    CSD_matrix = np.eye(nchans) # 
    for j in range(1, CSD_matrix.shape[0] - 1):
        CSD_matrix[j, j - 1: j + 2] = np.array([1., -2., 1.])
    if smooth_CSD_over_channels:
        curr_CSD = - np.asarray([np.dot(CSD_matrix, scipy.ndimage.gaussian_filter1d(np.squeeze(LFP_all[day_ind][i,:,:]), smooth_over_channel_count, axis = 0)) for i in range(10)])
    else:
        curr_CSD = - np.asarray([np.dot(CSD_matrix, LFP_all[day_ind][i,:,:]) for i in range(10)])
    curr_CSD[:,0,:] = 0
    curr_CSD[:,-1,:] = 0
    CSD_all.append(curr_CSD)
    os.chdir('..')
    
    
    
    # spacer = np.max(curr_CSD[:,1:-1,:])
    # fig, ax = plt.subplots(1,10, figsize = (15,15), sharey = True)
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     for chan in range(nchans):                        
    #         ax1.plot(curr_CSD[ind,chan,:] + chan * -spacer, 'b', linewidth = 1)              
    #         ax1.set_xlim([150,400])
    #     ax1.set_yticks(np.linspace(-spacer*(nchans - 1), 0, nchans))
    #     ax1.set_yticklabels(np.linspace((nchans - 1), 0, nchans).astype(int), size = 8)
    # plt.tight_layout()
    # plt.savefig('CSD all sweeps traces smoothed over channels', dpi = 1000)
    
    shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
    total_shift = max(shift)
    LFP_shift_all.append(np.asarray([LFP_all[day_ind][i, shift[i]:(32 - (total_shift -shift[i])), :] for i in range(10)]))
    PSTH_shift_all.append(np.asarray([PSTH_all[day_ind][i, shift[i]:(32 - (total_shift -shift[i])), :] for i in range(10)]))
    CSD_shift_all.append(np.asarray([CSD_all[day_ind][i, shift[i]:(32 - (total_shift -shift[i])), :] for i in range(10)]))
    
    os.chdir('..')


LFP_min_shift_all = [np.min(i[:,:,220:260], axis = 2) - i[:,:,200] for i in LFP_shift_all]
CSD_max_shift_all = [np.min(i[:,:,220:260], axis = 2) for i in CSD_shift_all]
PSTH_magn_shift_all = [np.sum(i[:,:,110:160], axis = 2) for i in PSTH_shift_all]
LFP_min_shift_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0)*100 for i in LFP_min_shift_all]
CSD_max_shift_all_rel = [i/np.mean(i[[0,1,2,3],:], axis = 0)*100 for i in CSD_max_shift_all]
PSTH_magn_shift_all_rel = [i/np.nanmean(i[[0,1,2,3],:], axis = 0)*100 for i in PSTH_magn_shift_all]

# average across channels in each layer with the channel map for each sweep
LFP_min_layer = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
PSTH_magn_layer = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
CSD_max_layer = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
LFP_min_layer_rel = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
PSTH_magn_layer_rel = np.zeros([len(LFP_min_shift_all_rel), 4, 10])
CSD_max_layer_rel = np.zeros([len(LFP_min_shift_all_rel), 4, 10])

for m_ind, day in enumerate(days):
    nchans = 32
    curr_layers_1_avg = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
    curr_layers_1 = layer_dict_1[day][0]
    for sweep in range(10): 

        # average across channels in each layer (plus minus one from selected channel)
        LFP_min_layer[m_ind, :, sweep] = np.asarray([np.mean(LFP_min_shift_all[m_ind][sweep, curr_layers_1_avg[i]]) for i in range(4)])
        PSTH_magn_layer[m_ind, :, sweep] = np.asarray([np.nanmean(PSTH_magn_shift_all[m_ind][sweep,curr_layers_1_avg[i]]) for i in range(4)])
        CSD_max_layer[m_ind, :, sweep] = np.asarray([np.mean(CSD_max_shift_all[m_ind][sweep,curr_layers_1_avg[i]]) for i in range(4)]) 

        LFP_min_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(LFP_min_shift_all_rel[m_ind][sweep, curr_layers_1_avg[i]]) for i in range(4)])
        PSTH_magn_layer_rel[m_ind, :, sweep] = np.asarray([np.nanmean(PSTH_magn_shift_all_rel[m_ind][sweep,curr_layers_1_avg[i]]) for i in range(4)])
        CSD_max_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(CSD_max_shift_all_rel[m_ind][sweep,curr_layers_1_avg[i]]) for i in range(4)]) 


# ----------------------------------------------------------------------------------------- LFP

# LFP min timecourse one channel per layers
arr_to_plot = LFP_min_layer_rel[[0,2,3],:,:]
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
plt.savefig('LFP rel per layer.jpg', dpi = 1000, format = 'jpg')
plt.savefig('LFP rel per layer.pdf', dpi = 1000, format = 'pdf')

layers_for_anova = [0,1,2]
LFP_min_layer_rel_for_ANOVA = np.zeros([len(days), 30])
curr_for_ANOVA = LFP_min_layer_rel[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
for layer in range(3):
    LFP_min_layer_rel_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
# LFP_min_layer_rel_1_for_ANOVA = LFP_min_layer_rel_1_for_ANOVA[LFP_min_layer_rel_1_for_ANOVA[:,-1].argsort()]
np.savetxt('LFP min 3 channels per layer.csv', LFP_min_layer_rel_for_ANOVA, delimiter = ',')

#LFP min timecourse in every mouse
fig, ax = plt.subplots(3,5, sharey = True)
for ax1_ind, ax1 in enumerate(list(ax.flatten())):
    if ax1_ind >= len(days):
        continue
    for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5']):
        ax1.plot(LFP_min_layer_rel[ax1_ind,i,:], label = str(layer))
    handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.suptitle('LFP min in one chanel')
plt.tight_layout()

print(np.mean(np.mean(LFP_min_layer_rel[:,:,[4,5,6,7,8,9]], axis = 2), axis = 0))
print(np.std(np.mean(LFP_min_layer_rel[:,:,[4,5,6,7,8,9]], axis = 2), axis = 0))


# -----------------------------------------------------------------------------------------PSTH

# mice_with_spikes = [0,1,2,3,4,5,6,7,9,10,11,12] # mouse 8 is not really useable because of weird layer 4.... could still put it in doesn't change the significance of the result

# arr_to_plot = PSTH_magn_layer_rel
# fig, ax = plt.subplots(figsize = (10,4))
# for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
#     to_plot_mean = np.mean(arr_to_plot[:,i,:], axis = 0) # average across mice
#     to_plot_err = 1*np.nanstd(arr_to_plot[:,i,:], axis = 0).T/np.sqrt(arr_to_plot.shape[0])
#     ax.plot(to_plot_mean.T, label = str(layer))
#     ax.fill_between(list(range(10)), to_plot_mean.T + 1*to_plot_err, to_plot_mean.T - 1*to_plot_err, alpha = 0.1)
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')
# ax.set_ylim([40, 150])
# ax.axvline(3.5, linestyle = '--', color = 'k')
# ax.set_xlabel('time from pairing (min)', size = 20)
# ax.set_ylabel('MUA response \n (% of baseline)', size = 20)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xticks([0,2,5,7,9])
# ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 20)
# ax.set_yticks([50,75,100,125,150])
# ax.set_yticklabels(list(map(str, ax.get_yticks())), size = 20)
# plt.tight_layout()
# plt.savefig('PSTH rel per layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('PSTH rel per layer.pdf', dpi = 1000, format = 'pdf')

# layers_for_anova = [0,1,2]
# PSTH_max_layer_rel_for_ANOVA = np.zeros([len(days), 30])
# curr_for_ANOVA = PSTH_max_layer_rel[:,layers_for_anova,:].reshape((len(days)*len(layers_for_anova),10))
# curr_for_ANOVA = np.append(curr_for_ANOVA, np.tile(np.linspace(1, len(layers_for_anova), len(layers_for_anova)),len(days))[:,np.newaxis], axis = 1)
# for layer in range(3):
#     PSTH_max_layer_rel_for_ANOVA[:, layer*10:layer*10 + 10] = np.squeeze(curr_for_ANOVA[np.argwhere(curr_for_ANOVA[:,-1].astype(int) == layer + 1), :])[:,:-1]
# np.savetxt('PSTH peak 3 channels per layer.csv', PSTH_max_layer_rel_for_ANOVA, delimiter = ',')

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

# print(np.mean(np.mean(PSTH_magn_layer_rel[:,:,[4,5,6,7,8,9]], axis = 2), axis = 0))
# print(np.std(np.mean(PSTH_magn_layer_rel[:,:,[4,5,6,7,8,9]], axis = 2), axis = 0))


# -----------------------------------------------------------------------------------------CSD

arr_to_plot = CSD_max_layer_rel[[0,2,3],:,:] # mouse 160803 has extremely strange CSD progression, not seen in any other mouse... would completely mess up the average 
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
fig.suptitle('CSD max avg per layer chanel')
plt.tight_layout()

print(np.mean(np.mean(CSD_max_layer_rel[:,:,[4,5,6,7,8,9]], axis = 2), axis = 0))
print(np.std(np.mean(CSD_max_layer_rel[:,:,[4,5,6,7,8,9]], axis = 2), axis = 0))





#%% 3) -------------------------------------------------- effect of AP5 on power spectrum

do_shift = False

exclude_before = 0.1
exclude_after = 0.9
fftfreq = np.fft.fftfreq(int((5 - exclude_before - exclude_after)*new_fs), d = (1/new_fs))
fftfreq_to_plot = np.where(np.logical_and(0.0 <= fftfreq , 10 >= fftfreq))[0]

delta_lower = 0.5
delta_upper = 4
fftfreq_to_plot_delta = np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]

b_notch, a_notch = scipy.signal.iirnotch(50, 100, 1000)

smooth_over_channels = True
smooth_over_channel_count = 1

nchans = 32
chanMap = chanMap_32
coordinates = [[i] for i in list(np.linspace(0, 1.55, nchans))]*pq.mm    


delta_LFP_layer_pre_AP5 = np.zeros([4, 3, 4]) # mouse, layer, sweep
delta_LFP_median_layer_pre_AP5 = np.zeros([4, 3, 4])
PSD_LFP_layer_pre_AP5 = np.zeros([4, 3, 4, 4000]) # mouse, layer, sweep
PSD_LFP_median_layer_pre_AP5 = np.zeros([4, 3, 4, 4000])

delta_CSD_layer_pre_AP5 = np.zeros([4, 3, 4]) # mouse, layer, sweep
delta_CSD_median_layer_pre_AP5 = np.zeros([4, 3, 4])
PSD_CSD_layer_pre_AP5 = np.zeros([4, 3, 4, 4000])
PSD_CSD_median_layer_pre_AP5 = np.zeros([4, 3, 4, 4000])

delta_LFP_layer_post_AP5 = np.zeros([4, 3, 4]) # mouse, layer, sweep
delta_LFP_median_layer_post_AP5 = np.zeros([4, 3, 4])
PSD_LFP_layer_post_AP5 = np.zeros([4, 3, 4, 4000]) # mouse, layer, sweep
PSD_LFP_median_layer_post_AP5 = np.zeros([4, 3, 4, 4000])

delta_CSD_layer_post_AP5 = np.zeros([4, 3, 4]) # mouse, layer, sweep
delta_CSD_median_layer_post_AP5 = np.zeros([4, 3, 4])
PSD_CSD_layer_post_AP5 = np.zeros([4, 3, 4, 4000])
PSD_CSD_median_layer_post_AP5 = np.zeros([4, 3, 4, 4000])

delta_LFP_layer_pre_AP5[:] = np.NaN
delta_LFP_median_layer_pre_AP5[:] = np.NaN
PSD_LFP_layer_pre_AP5[:] = np.NaN
PSD_LFP_median_layer_pre_AP5[:] = np.NaN
delta_CSD_layer_pre_AP5[:] = np.NaN
delta_CSD_median_layer_pre_AP5[:] = np.NaN
PSD_CSD_layer_pre_AP5[:] = np.NaN
PSD_CSD_median_layer_pre_AP5[:] = np.NaN
delta_LFP_layer_post_AP5[:] = np.NaN
delta_LFP_median_layer_post_AP5[:] = np.NaN
PSD_LFP_layer_post_AP5[:] = np.NaN
PSD_LFP_median_layer_post_AP5[:] = np.NaN
delta_CSD_layer_post_AP5[:] = np.NaN
delta_CSD_median_layer_post_AP5[:] = np.NaN
PSD_CSD_layer_post_AP5[:] = np.NaN
PSD_CSD_median_layer_post_AP5[:] = np.NaN


days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for m_ind, day in enumerate(days):
    os.chdir(day)
    print(day)
    os.chdir('pre_AP5')
    
    # LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    # stim_times = pickle.load(open('stim_times','rb'))
    # coordinates = [[i] for i in list(np.linspace(0, 1.55, nchans))]*pq.mm
    # nsweeps = len(LFP_all_sweeps)
    # stim_cumsum = np.cumsum(np.asarray([len(i) for i in stim_times]))
    # stim_cumsum = np.insert(stim_cumsum, 0, 0)
    
    # all_stims_delta = np.zeros([nchans, sum([len(stim_times[i]) for i in range(len(stim_times))])])
    # all_stims_delta_auto_outliers = np.zeros([nchans, sum([len(stim_times[i]) for i in range(len(stim_times))])])
    # all_stims_delta_CSD = np.zeros([nchans, sum([len(stim_times[i]) for i in range(len(stim_times))])])
    # all_stims_delta_CSD_auto_outliers = np.zeros([nchans, sum([len(stim_times[i]) for i in range(len(stim_times))])])

    # FFT = np.zeros([len(LFP_all_sweeps), nchans, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
    # PSD = np.empty([len(LFP_all_sweeps), nchans, int((5 - exclude_before - exclude_after)*new_fs)])
    # PSD[:] = np.NaN
    # PSD_median = np.empty([len(LFP_all_sweeps), nchans, int((5 - exclude_before - exclude_after)*new_fs)])
    # PSD_median[:] = np.NaN

    # delta_power = np.empty([nsweeps, nchans])
    # delta_power[:] = np.NaN
    # delta_power_median = np.empty([nsweeps, nchans])
    # delta_power_median[:] = np.NaN

    # FFT_CSD = np.zeros([len(LFP_all_sweeps), nchans, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
    # PSD_CSD = np.empty([len(LFP_all_sweeps), nchans, int((5 - exclude_before - exclude_after)*new_fs)])
    # PSD_CSD[:] = np.NaN
    # PSD_CSD_median = np.empty([len(LFP_all_sweeps), nchans, int((5 - exclude_before - exclude_after)*new_fs)])
    # PSD_CSD_median[:] = np.NaN
    # delta_power_CSD = np.empty([nsweeps, nchans])
    # delta_power_CSD[:] = np.NaN
    # delta_power_CSD_median = np.empty([nsweeps, nchans])
    # delta_power_CSD_median[:] = np.NaN

    # auto_outlier_stims = [[] for i in range(nsweeps)]
    # auto_outlier_stims_indices = [[] for i in range(nsweeps)]

    # # do fft for every interstim period, on LFP and CSD
    # for ind_sweep, LFP in enumerate(LFP_all_sweeps):
    #     print(ind_sweep)
    #     #EXCLUDE first and last stim just in case there isnt enough time, makes it easier
    #     FFT_current_sweep = np.zeros([len(stim_times[ind_sweep] - 2), nchans, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
    #     FFT_CSD_current_sweep = np.zeros([len(stim_times[ind_sweep] - 2), nchans, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
    #     FFT_current_sweep[:] = np.NaN
    #     FFT_CSD_current_sweep[:] = np.NaN
    #     FFT_current_sweep_auto_outliers = np.zeros([len(stim_times[ind_sweep] - 2), nchans, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
    #     FFT_CSD_current_sweep_auto_outliers = np.zeros([len(stim_times[ind_sweep] - 2), nchans, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
    #     FFT_current_sweep_auto_outliers[:] = np.NaN
    #     FFT_CSD_current_sweep_auto_outliers[:] = np.NaN

    #     for ind_stim, stim in enumerate(list(stim_times[ind_sweep][1:-1])):
    #         if stim == 0:
    #             continue
    #         curr_LFP = LFP[:, int(stim+exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs)]
    #         curr_LFP = scipy.signal.filtfilt(b_notch, a_notch, curr_LFP)
    #         FFT_current_sweep[ind_stim, :,:] = np.fft.fft(curr_LFP, axis = 1)
    #         FFT_current_sweep_auto_outliers[ind_stim, :,:] = np.fft.fft(curr_LFP, axis = 1)
    #         all_stims_delta[:,stim_cumsum[ind_sweep]+ind_stim] = np.nanmean(np.abs(FFT_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]])**2, axis = 0).T
    #         all_stims_delta_auto_outliers[:,stim_cumsum[ind_sweep]+ind_stim] = np.nanmean(np.abs(FFT_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]])**2, axis = 0).T

    #         # CSD and PSD of CSD
    #         if smooth_over_channels:
    #             curr_LFP_for_CSD = scipy.ndimage.gaussian_filter1d(curr_LFP[chanMap,:], smooth_over_channel_count, axis = 0)
    #         else:
    #             curr_LFP_for_CSD = curr_LFP[chanMap,:]
    #         curr_CSD = elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(curr_LFP_for_CSD.T, units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False)
    #         FFT_CSD_current_sweep[ind_stim, :,:] = np.fft.fft(curr_CSD.T, axis = 1)
    #         FFT_CSD_current_sweep_auto_outliers[ind_stim, :,:] = np.fft.fft(curr_CSD.T, axis = 1)
    #         all_stims_delta_CSD[:,stim_cumsum[ind_sweep]+ind_stim] = np.nanmean(np.abs(FFT_CSD_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]])**2, axis = 0).T
    #         all_stims_delta_CSD_auto_outliers[:,stim_cumsum[ind_sweep]+ind_stim] = np.nanmean(np.abs(FFT_CSD_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]])**2, axis = 0).T

    #     # define auto outlier periods as exceeding statistical outlier threshold within each sweep in each channel
    #     for chan in range(nchans):
    #         curr_delta = all_stims_delta[chan, stim_cumsum[ind_sweep]:stim_cumsum[ind_sweep + 1]]
    #         curr_CSD_delta = all_stims_delta_CSD[chan, stim_cumsum[ind_sweep]:stim_cumsum[ind_sweep + 1]]
    #         outliers_delta = (curr_delta > (np.percentile(curr_delta, 75) + 1.5*(np.abs(np.percentile(curr_delta, 75) - np.percentile(curr_delta, 25)))))
    #         outliers_CSD_delta = (curr_CSD_delta > (np.percentile(curr_CSD_delta, 75) + 1.5*(np.abs(np.percentile(curr_CSD_delta, 75) - np.percentile(curr_CSD_delta, 25)))))
    #         if len(np.where(outliers_delta == True)[0]) > 0:
    #             all_stims_delta_auto_outliers[chan, np.where(outliers_delta == True)[0] + stim_cumsum[ind_sweep]] = 0
    #             FFT_current_sweep_auto_outliers[np.where(outliers_delta == True)[0],:,:] = np.NaN
    #         if len(np.where(outliers_CSD_delta == True)[0]) > 0:
    #             all_stims_delta_CSD_auto_outliers[chan, np.where(outliers_CSD_delta == True)[0] + stim_cumsum[ind_sweep]] = 0
    #             FFT_CSD_current_sweep_auto_outliers[np.where(outliers_CSD_delta == True)[0],:,:] = np.NaN 
    #         auto_outlier_stims[ind_sweep].append(outliers_delta)
    #         auto_outlier_stims_indices[ind_sweep].append(np.where(outliers_delta == True)[0])

    #     # average across stims within each sweep
    #     PSD[ind_sweep,:,:] = np.nanmean(np.abs(FFT_current_sweep_auto_outliers)**2, axis = 0) 
    #     PSD_median[ind_sweep,:,:] = np.nanmedian(np.abs(FFT_current_sweep_auto_outliers)**2, axis = 0) 
    #     FFT[ind_sweep,:,:] = np.nanmean(FFT_current_sweep_auto_outliers, axis = 0)
    
    #     PSD_CSD[ind_sweep,:,:] = np.nanmean(np.abs(FFT_CSD_current_sweep_auto_outliers)**2, axis = 0) 
    #     PSD_CSD_median[ind_sweep,:,:] = np.nanmedian(np.abs(FFT_CSD_current_sweep_auto_outliers)**2, axis = 0) 
    #     FFT_CSD[ind_sweep,:,:] = np.nanmean(FFT_CSD_current_sweep_auto_outliers, axis = 0)
    
    #     delta_power[ind_sweep,:] = np.nanmean(PSD[ind_sweep, :, np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 0)
    #     delta_power_CSD[ind_sweep,:] = np.nanmean(PSD_CSD[ind_sweep, :, np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 0)

    #     delta_power_median[ind_sweep,:] = np.nanmedian(PSD[ind_sweep, :, np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 0)
    #     delta_power_CSD_median[ind_sweep,:] = np.nanmedian(PSD_CSD[ind_sweep, :, np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 0)

    
    # #delta power timecourse over whole recording
    # fig, ax = plt.subplots(8,4,sharey = True) 
    # fig.suptitle(f'{day} all stims LFP')
    # for ind, ax1 in enumerate(list(ax.flatten())[:nchans]):                        
    #     ax1.plot(all_stims_delta[chanMap[ind],:])
    #     for sweep in range(nsweeps):
    #         ax1.axvline(stim_cumsum[sweep], linestyle = '--')
    #     # ax1.axhline(450000, linestyle = '--')
    #     ax1.set_title(str(chanMap[ind]))

    # np.savetxt('delta_power.csv', delta_power, delimiter = ',')
    # np.savetxt('delta_power_median.csv', delta_power_median, delimiter = ',')
    # np.save('PSD.npy', PSD)
    # np.save('PSD_median.npy', PSD_median)

    # np.savetxt('delta_power_CSD.csv', delta_power_CSD, delimiter = ',')
    # np.savetxt('delta_power_CSD_median.csv', delta_power_CSD_median, delimiter = ',')
    # np.save('PSD_CSD.npy', PSD_CSD)
    # np.save('PSD_CSD_median.npy', PSD_CSD_median)

    # pickle.dump(auto_outlier_stims_indices, open('auto_outlier_stims_indices','wb'))
    
    delta_LFP_pre_AP5 = np.loadtxt('delta_power.csv', delimiter = ',')[:,chanMap]
    delta_LFP_median_pre_AP5 = np.loadtxt('delta_power_median.csv', delimiter = ',')[:,chanMap]
    PSD_LFP_pre_AP5 = np.load('PSD.npy')[:,chanMap,:]
    PSD_median_LFP_pre_AP5 = np.load('PSD_median.npy')[:,chanMap,:]

    delta_CSD_pre_AP5 = np.loadtxt('delta_power_CSD.csv', delimiter = ',')
    delta_CSD_median_pre_AP5 = np.loadtxt('delta_power_CSD_median.csv', delimiter = ',')
    PSD_CSD_pre_AP5 = np.load('PSD_CSD.npy')
    PSD_CSD_median_pre_AP5 = np.load('PSD_CSD_median.npy')

    os.chdir('..')
    
    
    # ----------------------------------------------- post AP5 baseline
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    delta_LFP_post_AP5 = np.loadtxt('delta_power.csv', delimiter = ',')[:,chanMap]
    delta_LFP_median_post_AP5 = np.loadtxt('delta_power_median.csv', delimiter = ',')[:,chanMap]
    PSD_LFP_post_AP5 = np.load('PSD.npy')[:,chanMap,:]
    PSD_median_LFP_post_AP5 = np.load('PSD_median.npy')[:,chanMap,:]

    delta_CSD_post_AP5 = np.loadtxt('delta_power_CSD.csv', delimiter = ',')
    delta_CSD_median_post_AP5 = np.loadtxt('delta_power_CSD_median.csv', delimiter = ',')
    PSD_CSD_post_AP5 = np.load('PSD_CSD.npy')
    PSD_CSD_median_post_AP5 = np.load('PSD_CSD_median.npy')

    os.chdir('..')

    # -------------------------------------------------------------------------- layer
    if day == '160801':
        sweeps_pre = [2,3]
        sweeps_post = [0,1,2,3]
    if day == '160803':
        sweeps_pre = [2,3]
        sweeps_post = [0,1,2,3]
    if day == '160804':
        sweeps_pre = [2,3]
        sweeps_post = [0,1,2,3]
    if day == '160811':
        sweeps_pre = [2,3]
        sweeps_post = [0,1,2,3]
    
    if day == '160811': # shifted one down between pre and post AP5
        curr_layers_1_avg_pre = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1])-1, 0, nchans-1)) for i in layer_dict_1[day][0]]
    else:
        curr_layers_1_avg_pre = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
    curr_layers_1_avg_post = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
    print(curr_layers_1_avg_pre, curr_layers_1_avg_post)
    
    for sweep_ind, (pre, post) in enumerate(zip(sweeps_pre, sweeps_post)):
        # average across channels in each layer (plus minus one from selected channel)
        delta_LFP_layer_pre_AP5[m_ind, :, sweep_ind] = np.asarray([np.mean(delta_LFP_pre_AP5[pre, curr_layers_1_avg_pre[i]]) for i in range(3)])
        delta_LFP_median_layer_pre_AP5[m_ind, :, sweep_ind] = np.asarray([np.mean(delta_LFP_median_pre_AP5[pre,curr_layers_1_avg_pre[i]]) for i in range(3)])
        PSD_LFP_layer_pre_AP5[m_ind, :, sweep_ind, :] = np.asarray([np.mean(PSD_LFP_pre_AP5[pre,curr_layers_1_avg_pre[i],:], axis = 0) for i in range(3)]) 
        PSD_LFP_median_layer_pre_AP5[m_ind, :, sweep_ind, :] = np.asarray([np.mean(PSD_median_LFP_pre_AP5[pre,curr_layers_1_avg_pre[i],:], axis = 0) for i in range(3)])
        
        delta_CSD_layer_pre_AP5[m_ind, :, sweep_ind] = np.asarray([np.mean(delta_CSD_pre_AP5[pre, curr_layers_1_avg_pre[i]]) for i in range(3)])
        delta_CSD_median_layer_pre_AP5[m_ind, :, sweep_ind] = np.asarray([np.mean(delta_CSD_median_pre_AP5[pre,curr_layers_1_avg_pre[i]]) for i in range(3)])
        PSD_CSD_layer_pre_AP5[m_ind, :, sweep_ind, :] = np.asarray([np.mean(PSD_CSD_pre_AP5[pre,curr_layers_1_avg_pre[i],:], axis = 0) for i in range(3)]) 
        PSD_CSD_median_layer_pre_AP5[m_ind, :, sweep_ind, :] = np.asarray([np.mean(PSD_CSD_median_pre_AP5[pre,curr_layers_1_avg_pre[i],:], axis = 0) for i in range(3)]) 

        delta_LFP_layer_post_AP5[m_ind, :, sweep_ind] = np.asarray([np.mean(delta_LFP_post_AP5[post, curr_layers_1_avg_post[i]]) for i in range(3)])
        delta_LFP_median_layer_post_AP5[m_ind, :, sweep_ind] = np.asarray([np.mean(delta_LFP_median_post_AP5[post,curr_layers_1_avg_post[i]]) for i in range(3)])
        PSD_LFP_layer_post_AP5[m_ind, :, sweep_ind, :] = np.asarray([np.mean(PSD_LFP_post_AP5[post,curr_layers_1_avg_post[i],:], axis = 0) for i in range(3)]) 
        PSD_LFP_median_layer_post_AP5[m_ind, :, sweep_ind, :] = np.asarray([np.mean(PSD_median_LFP_post_AP5[post,curr_layers_1_avg_post[i],:], axis = 0) for i in range(3)])
        
        delta_CSD_layer_post_AP5[m_ind, :, sweep_ind] = np.asarray([np.mean(delta_CSD_post_AP5[post, curr_layers_1_avg_post[i]]) for i in range(3)])
        delta_CSD_median_layer_post_AP5[m_ind, :, sweep_ind] = np.asarray([np.mean(delta_CSD_median_post_AP5[post,curr_layers_1_avg_post[i]]) for i in range(3)])
        PSD_CSD_layer_post_AP5[m_ind, :, sweep_ind, :] = np.asarray([np.mean(PSD_CSD_post_AP5[post,curr_layers_1_avg_post[i],:], axis = 0) for i in range(3)]) 
        PSD_CSD_median_layer_post_AP5[m_ind, :, sweep_ind, :] = np.asarray([np.mean(PSD_CSD_median_post_AP5[post,curr_layers_1_avg_post[i],:], axis = 0) for i in range(3)]) 

    
# np.mean(PSD_LFP_layer_pre_AP5[3,0,:,fftfreq_to_plot_delta], axis = 0)
# delta_LFP_layer_pre_AP5[3,0,:]

# np.mean(np.mean(PSD_LFP_layer_post_AP5[:,:,:,fftfreq_to_plot_delta], axis = -1), axis = -1)
# delta_LFP_layer_post_AP5[3,0,:]

# np.mean(PSD_LFP_layer_post_AP5[3,0,:,fftfreq_to_plot_delta], axis = 1)

    # # --------------------------------------------------------------------------- PSD plots
    # ##4 sweeps pre AP5, 4 sweeps post AP5 LFP
    # spacer = (np.max(np.log(PSD_LFP_pre_AP5)) - np.min(np.log(PSD_LFP_pre_AP5)))/4
    # fig, ax = plt.subplots(1,8, figsize = (15,15), sharey = True)
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     for chan in range(nchans):
    #         if ind < 4: #pre AP5
    #             ax1.plot(np.log(PSD_LFP_pre_AP5[ind,chan,fftfreq_to_plot_delta]) + chan * -spacer, 'b', linewidth = 1)    
    #             # ax1.semilogy(PSD_LFP_pre_AP5[ind,chan,fftfreq_to_plot] + chan * -1000, 'b', linewidth = 1)                 

    #         else:
    #             ax1.plot(np.log(PSD_LFP_post_AP5[ind-4,chan,fftfreq_to_plot_delta]) + chan * -spacer, 'b', linewidth = 1)                 
    #     # ax1.set_yticks(np.linspace(-(spacer*(nchans - 1)), 0, nchans))
    #     # ax1.set_yticklabels(np.linspace((nchans - 1), 0, nchans).astype(int), size = 8)
    # plt.tight_layout()
    
    # fig, ax = plt.subplots(3,1, sharey = True)
    # ax[0].semilogy(np.mean(PSD_LFP_layer_pre_AP5[m_ind, 0, :, fftfreq_to_plot], axis = 1), color = 'k')
    # ax[0].semilogy(np.mean(PSD_LFP_layer_post_AP5[m_ind, 0, :, fftfreq_to_plot], axis = 1))
    # ax[1].semilogy(np.mean(PSD_LFP_layer_pre_AP5[m_ind, 1, :, fftfreq_to_plot], axis = 1), color = 'k')
    # ax[1].semilogy(np.mean(PSD_LFP_layer_post_AP5[m_ind, 1, :, fftfreq_to_plot], axis = 1))
    # ax[2].semilogy(np.mean(PSD_LFP_layer_pre_AP5[m_ind, 2, :, fftfreq_to_plot], axis = 1), color = 'k')
    # ax[2].semilogy(np.mean(PSD_LFP_layer_post_AP5[m_ind, 2, :, fftfreq_to_plot], axis = 1))
    # ax[0].axvline(fftfreq_to_plot_delta[-1])
    # ax[0].axvline(fftfreq_to_plot_delta[0])

    # spacer = (np.max(np.log(PSD_LFP_pre_AP5)) - np.min(np.log(PSD_LFP_pre_AP5)))/4
    # fig, ax = plt.subplots(figsize = (2,10))
    # for chan in range(nchans):
    #     ax.plot(np.mean(np.log(PSD_LFP_pre_AP5[:,chan,fftfreq_to_plot_delta]), axis = 0) + chan * -spacer, 'k', linewidth = 1)                 
    #     ax.plot(np.mean(np.log(PSD_LFP_post_AP5[:,chan,fftfreq_to_plot_delta]), axis = 0) + chan * -spacer, 'k', linewidth = 1, linestyle = '--')                  
    # ax.set_yticks(np.linspace(-(spacer*((nchans - 1) - total_shift)), 0, 5))
    # ax.set_yticklabels(np.linspace(((nchans - 1) - total_shift), 0, tot_chans).astype(int), size = 6)
    # ax.set_yticks(np.linspace(31, 0, 5))
    # ax.set_yticklabels(list(map(str, np.linspace(-16, 0, 5).astype(int)/10)), size = 14)
    # ax.set_xticks([150,200,250,300])
    # ax.set_xticklabels(list(map(str, [-50, 0, 50, 100])), size = 14)
    # ax.set_ylabel('depth (mm)', size = 16)
    # ax.set_xlabel('time from stim (ms)', size = 16)
    # ax.set_ylim(bottom = -31*spacer - spacer/2)
    # ax.set_ylim(top = spacer + spacer/2)
    # plt.tight_layout()
    # plt.savefig('LFP response before after AP5.jpg', format = 'jpg', dpi = 1000)
    # plt.savefig('LFP response before after AP5.pdf', format = 'pdf', dpi = 1000)



    os.chdir('..')



#average spectrum before and after AP5 across sweeps
fig, ax = plt.subplots(3,1, sharey = True, figsize = (7, 18))
for layer in [0,1,2]:
    avg_PSD_pre = scipy.ndimage.gaussian_filter(np.nanmean(PSD_LFP_layer_pre_AP5[:, layer, :, :], axis = 1)/4000/1000, (0,1))
    avg_PSD_post = scipy.ndimage.gaussian_filter(np.nanmean(PSD_LFP_layer_post_AP5[:, layer, :, :], axis = 1)/4000/1000, (0,1))
    to_plot_mean = np.nanmean(avg_PSD_pre, axis = 0)[fftfreq_to_plot] # average across mice
    to_plot_err = np.nanstd(avg_PSD_pre, axis = 0)[fftfreq_to_plot]/2
    ax[layer].semilogy(fftfreq[fftfreq_to_plot], to_plot_mean, color = 'k')
    ax[layer].fill_between(fftfreq[fftfreq_to_plot], to_plot_mean - to_plot_err, to_plot_mean + to_plot_err, alpha = 0.5, color = 'k')
    to_plot_mean = np.nanmean(avg_PSD_post, axis = 0)[fftfreq_to_plot]
    to_plot_err = np.nanstd(avg_PSD_post, axis = 0)[fftfreq_to_plot]/2
    ax[layer].plot(fftfreq[fftfreq_to_plot], to_plot_mean, color = 'r')
    ax[layer].fill_between(fftfreq[fftfreq_to_plot], to_plot_mean - to_plot_err, to_plot_mean + to_plot_err, alpha = 0.5, color = 'r')
    ax[layer].tick_params(axis="x", labelsize=16)    
    ax[layer].tick_params(axis="y", labelsize=16, size = 12)   
    ax[layer].tick_params(which = 'minor', axis="y", size = 9)    
    ax[layer].set_xlabel('frequency (Hz)', size=16)
    ax[layer].set_ylabel('LFP power density ($\mathregular{mV^2}$/Hz)', size=16)
    ax[layer].fill_between([0.5,4], [100000,100000], alpha = 0.1, color = 'k')
    ax[layer].spines['top'].set_visible(False)
    ax[layer].spines['right'].set_visible(False)
    plt.tight_layout()
plt.savefig('PSD LFP averaged within layer.jpg', dpi = 1000, format = 'jpg')
plt.savefig('PSD LFP rel averaged within layer.pdf', dpi = 1000, format = 'pdf')



# Bar plot values pre vs post AP5
# 
# LFP delta power
rel_avg = np.mean(delta_LFP_layer_post_AP5, axis = 2)/np.mean(delta_LFP_layer_pre_AP5, axis = 2) # average across sweeps
print(rel_avg)
# fig, ax = plt.subplots(figsize = (5,5))
# ax.bar([0,1,2], np.mean(rel_avg, axis = 0)[:3], color = 'k', yerr = np.std(rel_avg, axis = 0)/2, capsize = 10)
# ax.set_ylim([0,1.2])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xticks([0,1,2])
# ax.set_xticklabels(['Layer 2/3', 'Layer 4', 'Layer 5'], rotation = 45)
# ax.tick_params(axis="x", labelsize=18)    
# ax.tick_params(axis="y", labelsize=18)   
# plt.tight_layout()
# plt.savefig('LFP delta power pre vs post AP5 layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('LFP delta power  pre vs post AP5 layer.pdf', dpi = 1000, format = 'pdf')

# # CSD delta power
rel_avg = np.mean(delta_CSD_layer_post_AP5, axis = 2)/np.mean(delta_CSD_layer_pre_AP5, axis = 2) # average across sweeps
print(rel_avg)
# fig, ax = plt.subplots(figsize = (5,5))
# ax.bar([0,1,2], np.mean(rel_avg, axis = 0)[:3], color = 'k', yerr = np.std(rel_avg, axis = 0)/2, capsize = 10)
# ax.set_ylim([0,1.2])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xticks([0,1,2])
# ax.set_xticklabels(['Layer 2/3', 'Layer 4', 'Layer 5'], rotation = 45)
# ax.tick_params(axis="x", labelsize=18)    
# ax.tick_params(axis="y", labelsize=18)   
# plt.tight_layout()
# plt.savefig('CSD delta power pre vs post AP5 layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('CSD delta power  pre vs post AP5 layer.pdf', dpi = 1000, format = 'pdf')




#%% 4) -------------------------------------------------- effect of AP5 on SO spiking


nchans = 32
chanMap = chanMap_32

spont_spiking_layer_pre_AP5 = np.zeros([4, 3, 4]) # mouse, layer, sweep
SW_spiking_sweeps_avg_layer_pre_AP5 = np.zeros([4, 3, 4, 1000])

spont_spiking_layer_post_AP5 = np.zeros([4, 3, 4]) # mouse, layer, sweep
SW_spiking_sweeps_avg_layer_post_AP5 = np.zeros([4, 3, 4, 1000])

spont_spiking_layer_pre_AP5[:] = np.NaN
SW_spiking_sweeps_avg_layer_pre_AP5[:] = np.NaN
spont_spiking_layer_post_AP5[:] = np.NaN
SW_spiking_sweeps_avg_layer_post_AP5[:] = np.NaN

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for m_ind, day in enumerate(days):
    os.chdir(day)
    print(day)
    
    os.chdir('pre_AP5')
    spont_spiking_pre_AP5 = np.load('spont_spiking.npy', )[:,chanMap]
    SW_spiking_sweeps_avg_pre_AP5 = np.load('SW_spiking_sweeps_avg.npy')[:,chanMap,:]
    os.chdir('..')

    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    spont_spiking_post_AP5 = np.load('spont_spiking.npy', )[:,chanMap]
    SW_spiking_sweeps_avg_post_AP5 = np.load('SW_spiking_sweeps_avg.npy')[:,chanMap,:]
    os.chdir('..')


# mouse_1_1 = list(map(np.asarray, [[7], [11], [15], [22]]))
# mouse_2_1 = list(map(np.asarray, [[11], [14], [19], [24]]))
# mouse_3_1 = list(map(np.asarray, [[7], [11], [16], [22]])) 
# mouse_4_1 = list(map(np.asarray, [[9], [13], [17], [21]]))


    # -------------------------------------------------------------------------- layer
    
    if day == '160801':
        sweeps_pre = [3]
        sweeps_post = [0]
    if day == '160803':
        sweeps_pre = [3]
        sweeps_post = [0]
    if day == '160804':
        sweeps_pre = [1,2,3]
        sweeps_post = [0]
    if day == '160811':
        sweeps_pre = [3]
        sweeps_post = [0]
      
        
    if day == '160811' or day == '160803': # shifted one down between pre and post AP5
        curr_layers_1_avg_pre = [(i-1) for i in layer_dict[day][0][1:]]
    else:
        curr_layers_1_avg_pre = layer_dict[day][0][1:]
    curr_layers_1_avg_post = layer_dict[day][0][1:]
    # if day == '160811' or day == '160803': # shifted one down between pre and post AP5
    #     curr_layers_1_avg_pre = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1])-1, 0, nchans-1)) for i in layer_dict_1[day][0]]
    # else:
    #     curr_layers_1_avg_pre = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
    # curr_layers_1_avg_post = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
    print(curr_layers_1_avg_pre, curr_layers_1_avg_post, sweeps_pre, sweeps_post)
    
    for sweep_ind, (pre, post) in enumerate(zip(sweeps_pre, sweeps_post)):
        # average across channels in each layer (plus minus one from selected channel)
        spont_spiking_layer_pre_AP5[m_ind, :, sweep_ind] = np.asarray([np.mean(spont_spiking_pre_AP5[pre, curr_layers_1_avg_pre[i]]) for i in range(3)])
        SW_spiking_sweeps_avg_layer_pre_AP5[m_ind, :, sweep_ind,:] = np.asarray([np.mean(SW_spiking_sweeps_avg_pre_AP5[pre, curr_layers_1_avg_pre[i],:], axis = 0) for i in range(3)])
        spont_spiking_layer_post_AP5[m_ind, :, sweep_ind,] = np.asarray([np.mean(spont_spiking_post_AP5[post, curr_layers_1_avg_pre[i]]) for i in range(3)]) 
        SW_spiking_sweeps_avg_layer_post_AP5[m_ind, :, sweep_ind, :] = np.asarray([np.mean(SW_spiking_sweeps_avg_post_AP5[post, curr_layers_1_avg_pre[i],:], axis = 0) for i in range(3)])
    os.chdir('..')

    # # ##4 sweeps pre AP5, 4 sweeps post AP5 LFP
    # spacer = (np.max(SW_spiking_sweeps_avg_pre_AP5))
    # fig, ax = plt.subplots(1,8, figsize = (15,15), sharey = True)
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     for chan in range(nchans):
    #         if ind < 4: #pre AP5
    #             ax1.plot(SW_spiking_sweeps_avg_pre_AP5[ind,chan,:] + chan * -spacer, 'b', linewidth = 1)    
    #         else:
    #             ax1.plot(SW_spiking_sweeps_avg_post_AP5[ind-4,chan,:] + chan * -spacer, 'b', linewidth = 1)                 
    #     # ax1.set_yticks(np.linspace(-(spacer*(nchans - 1)), 0, nchans))
    #     # ax1.set_yticklabels(np.linspace((nchans - 1), 0, nchans).astype(int), size = 8)
    # plt.tight_layout()


mice_to_plot = [0,2,3]

fig, ax = plt.subplots(3,1, sharey = True, figsize = (7, 18))
x_to_plot = np.linspace(-500,499,1000)
for layer in [0,1,2]:
    avg_pre = np.nanmean(SW_spiking_sweeps_avg_layer_pre_AP5[:, layer, :, :], axis = 1) 
    avg_post = np.nanmean(SW_spiking_sweeps_avg_layer_post_AP5[:, layer, :, :], axis = 1) 
    to_plot_mean = np.nanmean(avg_pre[mice_to_plot,:], axis = 0) 
    to_plot_err = np.nanstd(avg_pre[mice_to_plot,:], axis = 0)/2
    ax[layer].plot(x_to_plot, to_plot_mean, color = 'k')
    ax[layer].fill_between(x_to_plot, to_plot_mean - to_plot_err, to_plot_mean + to_plot_err, alpha = 0.5, color = 'k')
    to_plot_mean = np.nanmean(avg_post[mice_to_plot,:], axis = 0) 
    to_plot_err = np.nanstd(avg_post[mice_to_plot,:], axis = 0)/2
    ax[layer].plot(x_to_plot, to_plot_mean, color = 'r')
    ax[layer].fill_between(x_to_plot, to_plot_mean - to_plot_err, to_plot_mean + to_plot_err, alpha = 0.5, color = 'r')
    ax[layer].tick_params(axis="x", labelsize=16)    
    ax[layer].tick_params(axis="y", labelsize=16, size = 12)   
    ax[layer].tick_params(which = 'minor', axis="y", size = 9)    
    ax[layer].set_xlabel('time from UP-crossing (ms)', size=16)
    ax[layer].set_ylabel('MUA spiking probability', size=16)
    # ax[layer].fill_between([0.5,4], [100000,100000], alpha = 0.1, color = 'k')
    ax[layer].spines['top'].set_visible(False)
    ax[layer].spines['right'].set_visible(False)
    plt.tight_layout()
plt.savefig('SO spiking pre post AP5.jpg', dpi = 1000, format = 'jpg')
plt.savefig('SO spiking pre post AP5.pdf', dpi = 1000, format = 'pdf')


# fig, ax = plt.subplots(3,1, sharey = True, figsize = (7, 18))
# x_to_plot = np.linspace(-500,499,1000)
# for layer in [0,1,2]:
#     avg_pre = np.nanmean(SW_spiking_sweeps_avg_layer_pre_AP5[:, layer, :, :], axis = 1) 
#     avg_post = np.nanmean(SW_spiking_sweeps_avg_layer_post_AP5[:, layer, :, :], axis = 1) 
#     to_plot_mean = avg_pre[mice_to_plot,:].T
#     ax[layer].plot(x_to_plot, to_plot_mean)

    # ax[0].plot(np.mean(SW_spiking_sweeps_avg_layer_post_AP5[m_ind, 0, :, :], axis = 0))
    # ax[1].plot(np.mean(SW_spiking_sweeps_avg_layer_pre_AP5[m_ind, 1, :, :], axis = 0), color = 'k')
    # ax[1].plot(np.mean(SW_spiking_sweeps_avg_layer_post_AP5[m_ind, 1, :, :], axis = 0))
    # ax[2].plot(np.mean(SW_spiking_sweeps_avg_layer_pre_AP5[m_ind, 2, :, :], axis = 0), color = 'k')
    # ax[2].plot(np.mean(SW_spiking_sweeps_avg_layer_post_AP5[m_ind, 2, :, :], axis = 0))
    # ax[0].axvline(fftfreq_to_plot_delta[-1])
    # ax[0].axvline(fftfreq_to_plot_delta[0])


# # spont spiking
# rel_avg = np.nanmean(spont_spiking_layer_post_AP5, axis = 2)/np.nanmean(spont_spiking_layer_pre_AP5, axis = 2) # average across sweeps
# print(rel_avg)
# fig, ax = plt.subplots(figsize = (5,5))
# ax.bar([0,1,2], np.mean(rel_avg, axis = 0)[:3], color = 'k', yerr = np.std(rel_avg, axis = 0)/2, capsize = 10)
# # ax.set_ylim([0,1.2])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xticks([0,1,2])
# ax.set_xticklabels(['Layer 2/3', 'Layer 4', 'Layer 5'], rotation = 45)
# ax.tick_params(axis="x", labelsize=18)    
# ax.tick_params(axis="y", labelsize=18)   
# plt.tight_layout()



#%% 5) -------------------------------------------------- effect of AP5 on ON OFF length

ON_durations_pre_AP5 = []
OFF_durations_pre_AP5 = []
ON_durations_post_AP5 = []
OFF_durations_post_AP5 = []

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# for day in ['160810']:
for m_ind, day in enumerate(days):
    print(day)
    os.chdir(day)
    
    # ------------------------------------------------------------- load up pre AP5
    os.chdir('pre_AP5')

    ON_states_starts_avg_allsweeps = pickle.load(open('ON_states_starts_avg_allsweeps', 'rb'))      
    ON_states_stops_avg_allsweeps = pickle.load(open('ON_states_stops_avg_allsweeps', 'rb')) 
    
    for sweep in range(4):
        ON_durations = [(j - i)[1:] for i,j in zip(ON_states_starts_avg_allsweeps, ON_states_stops_avg_allsweeps)]
        OFF_durations =[( i[1:] - j[:-1]) for i,j in zip(ON_states_starts_avg_allsweeps, ON_states_stops_avg_allsweeps)]
    ON_durations_pre_AP5.append(ON_durations)
    OFF_durations_pre_AP5.append(OFF_durations)

    os.chdir('..')
    
    ON_states_starts_avg_allsweeps = pickle.load(open('ON_states_starts_avg_allsweeps', 'rb'))      
    ON_states_stops_avg_allsweeps = pickle.load(open('ON_states_stops_avg_allsweeps', 'rb')) 
    
    for sweep in range(4):
        ON_durations = [(j - i)[1:] for i,j in zip(ON_states_starts_avg_allsweeps, ON_states_stops_avg_allsweeps)]
        OFF_durations =[( i[1:] - j[:-1]) for i,j in zip(ON_states_starts_avg_allsweeps, ON_states_stops_avg_allsweeps)]
    ON_durations_post_AP5.append(ON_durations)
    OFF_durations_post_AP5.append(OFF_durations)

    os.chdir('..')


    # ------------------------------------------------ ON length
    
    # fig, ax = plt.subplots(2,1, sharex = True)
    # ax[0].hist(np.concatenate(ON_durations_pre_AP5[m_ind]), bins = 25, color = 'k', alpha = 0.5, density = True, stacked = True)
    # ax[1].hist(np.concatenate(OFF_durations_pre_AP5[m_ind]), bins = 25, color = 'k', alpha = 0.5, density = True, stacked = True)
    # ax[0].hist(np.concatenate(ON_durations_post_AP5[m_ind][0:4]), bins = 25, color = 'r', alpha = 0.5, density = True, stacked = True)
    # ax[1].hist(np.concatenate(OFF_durations_post_AP5[m_ind][0:4]), bins = 25, color = 'r', alpha = 0.5, density = True, stacked = True)




bins = 100
# ------------------------------- ON length across mice

fig, ax = plt.subplots(figsize = (6,3))
color_mice = ['black', 'red', 'blue', 'green']
cumsum_all_before = []
cumsum_all_after = []
for m_ind, color in enumerate(color_mice):
    # ax.hist(CSD_sink_timecourse_L4_vs_L5_before_ALL[m_ind], bins=bins, density = True, cumulative = True, color = color, histtype = 'step')
    y, binEdges = np.histogram(np.concatenate(ON_durations_pre_AP5[m_ind]), bins=bins, density = True, range = [0,2])
    y = np.cumsum(y)/np.max(np.cumsum(y))
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    # ax.plot(bincenters, 100*np.cumsum(y)/np.max(np.cumsum(y)), color = color)
    # ax.plot(bincenters, y, color = 'k', alpha = 0.25)
    cumsum_all_before.append(y*100)
    
    y, binEdges = np.histogram(np.concatenate(ON_durations_post_AP5[m_ind][0:4]), bins=bins, density = True, range = [0,2])
    y = np.cumsum(y)/np.max(np.cumsum(y))
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    # ax.plot(bincenters, 100*np.cumsum(y)/np.max(np.cumsum(y)), color = color)
    # ax.plot(bincenters, y, color = 'cyan', alpha = 0.25)
    cumsum_all_after.append(y*100)
    
bincenters = bincenters
ax.plot(bincenters, np.mean(np.asarray(cumsum_all_before), axis = 0), color = 'k', linewidth = 2)
before_err = np.std(np.asarray(cumsum_all_before), axis = 0)/np.sqrt(4)
ax.plot(bincenters, np.mean(np.asarray(cumsum_all_after), axis = 0), color = 'red', linewidth = 2)
after_err = np.std(np.asarray(cumsum_all_after), axis = 0)/np.sqrt(4)
ax.fill_between(bincenters, np.mean(np.asarray(cumsum_all_before), axis = 0) + before_err, np.mean(np.asarray(cumsum_all_before) - before_err, axis = 0), color = 'k', alpha = 0.3)
ax.fill_between(bincenters, np.mean(np.asarray(cumsum_all_after), axis = 0) + after_err, np.mean(np.asarray(cumsum_all_after) - after_err, axis = 0), color = 'red', alpha = 0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis = 'x', labelsize = 16)
ax.tick_params(axis = 'y', labelsize = 16)
ax.set_xlabel('ON-state length (s)', size = 16)
ax.set_ylabel('% of ON-states', size = 16)
plt.tight_layout()
# plt.savefig('ON state length pre post AP5''.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('ON state length pre post AP5.pdf', dpi = 1000, format = 'pdf')

print(scipy.stats.kstest(np.mean(np.asarray(cumsum_all_before), axis = 0), np.mean(np.asarray(cumsum_all_after), axis = 0)))



# ------------------------------- OFF length across mice

fig, ax = plt.subplots(figsize = (6,3))
color_mice = ['black', 'red', 'blue', 'green']
cumsum_all_before = []
cumsum_all_after = []
for m_ind, color in enumerate(color_mice):
    # ax.hist(CSD_sink_timecourse_L4_vs_L5_before_ALL[m_ind], bins=bins, density = True, cumulative = True, color = color, histtype = 'step')
    y, binEdges = np.histogram(np.concatenate(OFF_durations_pre_AP5[m_ind]), bins=bins, density = True, range = [0,2])
    y = np.cumsum(y)/np.max(np.cumsum(y))
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    # ax.plot(bincenters, 100*np.cumsum(y)/np.max(np.cumsum(y)), color = color)
    # ax.plot(bincenters, y, color = 'k', alpha = 0.25)
    cumsum_all_before.append(y*100)
    y, binEdges = np.histogram(np.concatenate(OFF_durations_post_AP5[m_ind][0:4]), bins=bins, density = True, range = [0,2])
    y = np.cumsum(y)/np.max(np.cumsum(y))
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    # ax.plot(bincenters, 100*np.cumsum(y)/np.max(np.cumsum(y)), color = color)
    # ax.plot(bincenters, y, color = 'cyan', alpha = 0.25)
    cumsum_all_after.append(y*100)
    
bincenters = bincenters
ax.plot(bincenters, np.mean(np.asarray(cumsum_all_before), axis = 0), color = 'k', linewidth = 2)
before_err = np.std(np.asarray(cumsum_all_before), axis = 0)/np.sqrt(4)
ax.plot(bincenters, np.mean(np.asarray(cumsum_all_after), axis = 0), color = 'red', linewidth = 2)
after_err = np.std(np.asarray(cumsum_all_after), axis = 0)/np.sqrt(4)
ax.fill_between(bincenters, np.mean(np.asarray(cumsum_all_before), axis = 0) + before_err, np.mean(np.asarray(cumsum_all_before) - before_err, axis = 0), color = 'k', alpha = 0.3)
ax.fill_between(bincenters, np.mean(np.asarray(cumsum_all_after), axis = 0) + after_err, np.mean(np.asarray(cumsum_all_after) - after_err, axis = 0), color = 'red', alpha = 0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis = 'x', labelsize = 16)
ax.tick_params(axis = 'y', labelsize = 16)
ax.set_xlabel('OFF-state length (s)', size = 16)
ax.set_ylabel('% of OFF-states', size = 16)
plt.tight_layout()
# plt.savefig('OFF state length pre post AP5.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('OFF state length pre post AP5.pdf', dpi = 1000, format = 'pdf')

print(scipy.stats.kstest(np.mean(np.asarray(cumsum_all_before), axis = 0), np.mean(np.asarray(cumsum_all_after), axis = 0)))


#%% 6) -------------------------------------------------- effect of AP5 on delta power change and SO change after pairing


mouse_1 = list(map(np.asarray, [[2,3,4,], [5,6,7,8,9], [10,11], [12,13,14,15,16,17,18,19], [19,20,21,22,23,24,25,26]]))
mouse_2 = list(map(np.asarray, [[0,1,2,], [3,4,5,6,7,8,9], [10,11], [12,13,14,15,16,17,18,19], [20,21,22,23,24,25]]))
mouse_3 = list(map(np.asarray, [[0,1,2,3], [4,5,6,7,8,9], [10,11], [12,13,14,15,16,17,18,19,20,21,22], [20,21,22,23,24,25,26]]))
mouse_4 = list(map(np.asarray, [[2,3,4,5], [6,7,8,9,10,11], [12,13], [14,15,16,17,18,19,20,21], [19,20,21,22,23,24,25]]))


layer_dict = {'160801' : [mouse_1]*10,
                 
            '160803' : [mouse_2]*10,

            '160804' : [mouse_3]*4 + [[i + 1 for i in mouse_3]]*6,
            
            '160811' : [mouse_4]*4 + [[i + 1 for i in mouse_4]]*6,
            }

layer_list_LFP = list(layer_dict.values())
layer_list_CSD = copy.deepcopy(layer_list_LFP)

mouse_1_1 = list(map(np.asarray, [[8], [11], [15], [22]]))
mouse_2_1 = list(map(np.asarray, [[11], [14], [19], [24]]))
mouse_3_1 = list(map(np.asarray, [[7], [11], [16], [22]]))
mouse_4_1 = list(map(np.asarray, [[9], [13], [17], [21]]))

layer_dict_1 = {'160801' : [mouse_1_1]*10,
                 
            '160803' : [mouse_2_1]*10,

            '160804' : [mouse_3_1]*4 + [[i + 1 for i in mouse_3_1]]*6,
            
            '160811' : [mouse_4_1]*10,   
            }

layer_list_LFP_1 = list(layer_dict_1.values())
layer_list_CSD_1 = copy.deepcopy(layer_list_LFP_1)


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


delta_LFP_all_rel[2] = delta_LFP_all[2]/np.mean(delta_LFP_all[2][[1,2,3],:], axis = 0)
delta_CSD_all_rel[2] = delta_CSD_all[2]/np.mean(delta_CSD_all[2][[1,2,3],:], axis = 0)

delta_LFP_shift_all_rel[2] = delta_LFP_shift_all[2]/np.mean(delta_LFP_shift_all[2][[1,2,3],:], axis = 0)
delta_CSD_shift_all_rel[2] = delta_CSD_shift_all[2]/np.mean(delta_CSD_shift_all[2][[1,2,3],:], axis = 0)

delta_LFP_median_all_rel[2] = delta_LFP_median_all[2]/np.mean(delta_LFP_median_all[2][[1,2,3],:], axis = 0)
delta_CSD_median_all_rel[2] = delta_CSD_median_all[2]/np.mean(delta_CSD_median_all[2][[1,2,3],:], axis = 0)

delta_LFP_median_shift_all_rel[2] = delta_LFP_median_shift_all[2]/np.mean(delta_LFP_median_shift_all[2][[1,2,3],:], axis = 0)
delta_CSD_median_shift_all_rel[2] = delta_CSD_median_shift_all[2]/np.mean(delta_CSD_median_shift_all[2][[1,2,3],:], axis = 0)








# -------------------------------------------------------- layer the results
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


# delta_to_use = delta_LFP_median_all_rel
# CSD_delta_to_use = delta_CSD_median_all_rel

delta_to_use = delta_LFP_median_shift_all_rel
CSD_delta_to_use = delta_CSD_median_shift_all_rel

# PSD_LFP_to_use = PSD_LFP_shift_all
# PSD_CSD_to_use = PSD_CSD_shift_all

for m_ind, day in enumerate(days):
    for sweep in range(10): 
        # curr_layers_1_avg = [np.unique(np.clip(np.array([i[0]-1,i[0],i[0]+1]), 0, nchans-1)) for i in layer_dict_1[day][0]]
        curr_layers_1_avg = layer_dict[day][0][1:]
        curr_layers_LFP = layer_dict[day][0]
        curr_layers_CSD = layer_dict[day][0]
        curr_layers_LFP_1 = layer_dict_1[day][0]
        curr_layers_CSD_1 = layer_dict_1[day][0]

        # LFP_delta_layer[m_ind, :, sweep] = np.asarray([np.mean(delta_LFP_shift_all[m_ind][sweep, curr_layers_LFP[i]]) for i in range(5)])
        # CSD_delta_layer[m_ind, :, sweep] = np.asarray([np.mean(delta_CSD_shift_all[m_ind][sweep,curr_layers_CSD[i]]) for i in range(5)]) 
        
        #median across channels
        LFP_delta_layer_rel[m_ind, :, sweep] = np.asarray([np.median(delta_to_use[m_ind][sweep, curr_layers_1_avg[i]]) for i in range(4)])
        CSD_delta_layer_rel[m_ind, :, sweep] = np.asarray([np.median(CSD_delta_to_use[m_ind][sweep,curr_layers_1_avg[i]]) for i in range(4)]) 
        
        PSD_LFP_layer[m_ind, :, :, sweep] = np.asarray([np.median(PSD_LFP_all[m_ind][sweep, curr_layers_1_avg[i], :], axis = 0) for i in range(4)])
        PSD_LFP_median_layer[m_ind, :, :, sweep] = np.asarray([np.median(PSD_LFP_median_all[m_ind][sweep, curr_layers_1_avg[i], :], axis = 0) for i in range(4)])
        
        PSD_CSD_layer[m_ind, :, :, sweep] = np.asarray([np.median(PSD_CSD_all[m_ind][sweep, curr_layers_1_avg[i], :], axis = 0) for i in range(4)])
        PSD_CSD_median_layer[m_ind, :, :, sweep] = np.asarray([np.median(PSD_CSD_median_all[m_ind][sweep, curr_layers_1_avg[i], :], axis = 0) for i in range(4)])
        
        #mean across channels
        # LFP_delta_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(delta_to_use[m_ind][sweep, curr_layers_1_avg[i]]) for i in range(4)])
        # CSD_delta_layer_rel[m_ind, :, sweep] = np.asarray([np.mean(CSD_delta_to_use[m_ind][sweep,curr_layers_1_avg[i]]) for i in range(4)]) 
        
        # PSD_LFP_layer[m_ind, :, :, sweep] = np.asarray([np.mean(PSD_LFP_all[m_ind][sweep, curr_layers_1_avg[i], :], axis = 0) for i in range(4)])
        # PSD_LFP_median_layer[m_ind, :, :, sweep] = np.asarray([np.mean(PSD_LFP_median_all[m_ind][sweep, curr_layers_1_avg[i], :], axis = 0) for i in range(4)])
        
        # PSD_CSD_layer[m_ind, :, :, sweep] = np.asarray([np.mean(PSD_CSD_all[m_ind][sweep, curr_layers_1_avg[i], :], axis = 0) for i in range(4)])
        # PSD_CSD_median_layer[m_ind, :, :, sweep] = np.asarray([np.mean(PSD_CSD_median_all[m_ind][sweep, curr_layers_1_avg[i], :], axis = 0) for i in range(4)])

        
# # ---------------------------------------------------- PSD across mice in different layers
# fft_freq_ind = np.where(np.logical_and(0 <= fftfreq , 20 >= fftfreq))[0]
# fig, ax = plt.subplots(3,1, sharey = True, figsize = (7, 18))
# mice_to_plot = [0,1,2,3]
# # mice_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11,12]
# to_plot_before = np.mean(scipy.ndimage.gaussian_filter(PSD_LFP_median_layer[:,:,:,[0,1,2,3]][mice_to_plot,:,:,:], (0,0,1,0)), axis = 3)/3500/1000
# to_plot_after = np.mean(scipy.ndimage.gaussian_filter(PSD_LFP_median_layer[:,:,:,[4,5,6,7,8,9]][mice_to_plot,:,:,:], (0,0,1,0)), axis = 3)/3500/1000
# for ind, ax1 in enumerate(list(ax.flatten())):
#     # ax1.semilogy(delta_fft_freq_ind, scipy.ndimage.gaussian_filter(to_plot_before[:,ind,delta_fft_freq_ind].T, (1,0)))
#     to_plot_mean = np.mean(to_plot_before[:,ind,fft_freq_ind], axis = 0)
#     to_plot_err = np.std(to_plot_before[:,ind,fft_freq_ind], axis = 0)/np.sqrt(len(mice_to_plot))
#     ax1.semilogy(fftfreq[fft_freq_ind], to_plot_mean, color = 'k')
#     ax1.fill_between(fftfreq[fft_freq_ind], to_plot_mean - to_plot_err, to_plot_mean + to_plot_err, alpha = 0.5, color = 'k')
#     to_plot_mean = np.mean(to_plot_after[:,ind,fft_freq_ind], axis = 0)
#     to_plot_err = np.std(to_plot_after[:,ind,fft_freq_ind], axis = 0)/np.sqrt(len(mice_to_plot))
#     ax1.plot(fftfreq[fft_freq_ind], to_plot_mean, color = 'c')
#     ax1.fill_between(fftfreq[fft_freq_ind], to_plot_mean - to_plot_err, to_plot_mean + to_plot_err, alpha = 0.5, color = 'c')
#     ax1.tick_params(axis="x", labelsize=16)    
#     ax1.tick_params(axis="y", labelsize=16, size = 12)   
#     ax1.tick_params(which = 'minor', axis="y", size = 9)    
#     ax1.set_xlabel('frequency (Hz)', size=16)
#     ax1.set_ylabel('LFP power density ($\mathregular{mV^2}$/Hz)', size=16)
#     ax1.fill_between([0.5,4], [100000,100000], alpha = 0.1, color = 'k')
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     plt.tight_layout()
# plt.savefig('PSD LFP averaged within layer.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('PSD LFP rel averaged within layer.pdf', dpi = 1000, format = 'pdf')



# ------------------------------------------------------------------------------------ delta power of LFP
# rel delta timecourse in different layer averaged within layers
LFP_delta_to_plot = copy.deepcopy(LFP_delta_layer_rel)
LFP_delta_to_plot[2,0,0] = np.NaN
mice_to_plot = [0,1,2,3]
fig, ax = plt.subplots(figsize = (10,4))
for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
    to_plot_mean = np.nanmean(LFP_delta_to_plot[mice_to_plot,i,:], axis = 0).T*100
    to_plot_err = np.nanstd(LFP_delta_to_plot[mice_to_plot,i,:], axis = 0).T/np.sqrt(len(mice_to_plot))*100
    ax.plot(to_plot_mean, label = str(layer))
    ax.fill_between(list(range(10)), to_plot_mean + to_plot_err, to_plot_mean - to_plot_err, alpha = 0.1)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
# ax.set_ylim([55, 110])
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 20)
ax.set_ylabel('LFP delta power \n (% of baseline)', size = 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 20)
ax.set_yticks([50,100,150])
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


# ------------------------------------------------------------------------------------ delta power of CSD of spontaneous activity 
CSD_delta_to_plot = copy.deepcopy(CSD_delta_layer_rel)
CSD_delta_to_plot[2,0,0] = np.NaN
mice_to_plot = [0,1,2,3]
fig, ax = plt.subplots(figsize = (10,4))
for i, layer in zip([0,1,2], ['layer 2/3', 'layer 4', 'layer 5', 'layer 6']):
    to_plot_mean = np.nanmean(CSD_delta_to_plot[mice_to_plot,i,:], axis = 0).T*100
    to_plot_err = np.nanstd(CSD_delta_to_plot[mice_to_plot,i,:], axis = 0).T/np.sqrt(len(mice_to_plot))*100
    ax.plot(to_plot_mean, label = str(layer))
    ax.fill_between(list(range(10)), to_plot_mean + to_plot_err, to_plot_mean - to_plot_err, alpha = 0.1)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
# ax.set_ylim([50, 120])
ax.axvline(3.5, linestyle = '--', color = 'k')
ax.set_xlabel('time from pairing (min)', size = 20)
ax.set_ylabel('CSD delta power \n (% of baseline)', size = 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0,2,5,7,9])
ax.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 20)
ax.set_yticks([50,100,150])
ax.set_yticklabels(list(map(str, ax.get_yticks())), size = 20)
plt.tight_layout()
plt.savefig('CSD delta rel avg within layer.jpg', dpi = 1000, format = 'jpg')
plt.savefig('CSD delta rel avg within layer.pdf', dpi = 1000, format = 'pdf')

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









