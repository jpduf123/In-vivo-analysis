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

home_directory = r'I:\LAMINAR_UP_JP'
os.chdir(home_directory)

# day = os.getcwd()[-6:]
# if os.path.exists(f'analysis_{day}') == False:
#     os.mkdir(f'analysis_{day}')

fs = 30000
resample_factor = 30
new_fs = fs/resample_factor

# chanMap = np.array([31,32,30,33,27,36,26,37,16,47,18,38,17,46,29,39,19,45,28,40,20,44,22,41,21,34,24,42,23,43,25,35]) - 16
chanMap = np.array([[39,45,46,40,38,44,47,41,37,34,36,42,33,43,32,35],[19,29,28,17,20,18,22,16,21,26,24,27,23,30,25,31]]) - 16

nChans = len(chanMap.flatten())
#coordinates of electrodes, to be given as a list of lists (in case of laminar 1 coord per point):
coordinates = [[i] for i in list(np.linspace(0, 1.55, 16))]*pq.mm


def smooth(y, box_pts, axis = 0):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.apply_along_axis(lambda m: np.convolve(m, box, mode='same'), axis = axis, arr = y)
    return y_smooth

def cl():
    plt.close('all')

# fig, ax = plt.subplots()
# ax.plot(highpass[300000:600000])
# ax.axhline(-5*std)

highpass_cutoff = 4

use_kilosort = False


#%% extract LFP spikes and stim times
highpass_cutoff = 4

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# for day in days:
for day in ['25_10_22']:
    os.chdir(day) 
    print(day)
    # if os.path.exists(f'analysis_{day}') == False:
    #     os.mkdir(f'analysis_{day}')
    # os.chdir('..')
    sweeps = [s for s in os.listdir() if 'baseline' in s]     
    # REORDER THE SWEEPS BASELINE-BEFORE 
    before = [s for s in sweeps if 'before' in s]
    after = [s for s in sweeps if 'after' in s]
    sweeps_ordered = before + after
    
    # if os.path.isfile('LFP_resampled') and os.path.isfile(f'spikes_allsweeps_{highpass_cutoff}'):
    #     pass
    #     # LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    #     # if use_kilosort == False:
    #     #     spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    #     # else:
    #     #     spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    #     # stim_times = pickle.load(open('stim_times','rb'))
         
    # #%
    # else:
    # #extract stim times, LFP and spikes
    #     print(day)
    #     LFP_all_sweeps = []
    #     spikes_allsweeps = []
        
    #     for ind_sweep, sweep in enumerate(sweeps_ordered):
    #         os.chdir(sweep)
    #         channels = [s for s in os.listdir() if 'amp' in s and '.dat' in s]
    #         curr_spikes = {}
    #         for ind_channel, channel in enumerate(channels):
    #             print(channel)
    #             with open(channel,'rb') as f:
    #                 curr_LFP = np.fromfile(f, np.int16)
                    
    #                 # take out spikes
    #                 highpass = elephant.signal_processing.butter(curr_LFP, highpass_frequency = 250, sampling_frequency=30000)
    #                 if day == '25_10_22': # noise in the third baseline, so use std of first baseline as cutoff
    #                     if ind_sweep == 0:
    #                         std = np.std(highpass)
    #                 elif day != '25_10_22':
    #                     std = np.std(highpass)
    #                 #the highpass cutoff you choose is not that important, you'll just get more noise in the end but maybe not stupid for the slow wave spiking
    #                 crossings = np.argwhere(highpass<-highpass_cutoff*std)
    #                 # take out values within half a second of each other
    #                 crossings = crossings[np.roll(crossings,-1) - crossings > 20]
    #                 curr_spikes[channel[-6:-4]] = crossings/resample_factor
                    
    #                 #resample for LFP. SHOULD TRY DECIMATE AS WELL AND SEE IF ITS DIFFERENT. scipy resample just deletes all frequencies above Nyquist resample frequency, it doesnt apply a filter like decimate does
    #                 curr_LFP = scipy.signal.resample(curr_LFP, int(np.ceil(len(curr_LFP)/resample_factor)))
                    
    #             if ind_channel == 0:
    #                 LFP = curr_LFP
    #             elif ind_channel > 0:                
    #                 LFP = np.vstack((LFP,curr_LFP))
                    
    #         spikes_allsweeps.append(curr_spikes)    
    #         LFP_all_sweeps.append(LFP)
    #         os.chdir("..")
        
    #     pickle.dump(LFP_all_sweeps, open('LFP_resampled','wb'))
    #     pickle.dump(spikes_allsweeps, open(f'spikes_allsweeps_{highpass_cutoff}','wb'))
    
        # stim_times = []
        # #get stim_times
        # for ind_sweep, sweep in enumerate(sweeps_ordered):
        #     os.chdir(sweep)
        #     if os.path.isfile('board-DIN-00.dat'):
        #         stimfile = 'board-DIN-00.dat'   
        #     elif os.path.isfile('board-DIGITAL-IN-00.dat'):
        #         stimfile = 'board-DIGITAL-IN-00.dat'
        #     else:
        #         raise KeyError('no stim file')
        #     with open(stimfile,'rb') as f:
        #             curr_stims = np.fromfile(f, np.int16)
        #             curr_stims = np.where(np.diff(curr_stims) == 1)[0]/resample_factor
        #     stim_times.append(curr_stims)       
        #     os.chdir("..")
        
            
        # #take out shit stims:
        # for ind_sweep, stims in enumerate(stim_times):
        #     diff = np.diff(stims)
        #     for ind_stim, stim in enumerate(list(stims)):
        #         if ind_stim == len(stims) - 1:
        #             break
        #         if stims[ind_stim + 1] - stims[ind_stim] < 4995:
        #             for i in range(1,100):
        #                 if ind_stim + i > len(stims) - 1:
        #                     break
        #                 if stims[ind_stim + i] - stims[ind_stim] < 4995:
        #                     stims[ind_stim + i] = 0
        #                 elif stims[ind_stim + i] - stims[ind_stim] > 4996:
        #                     break 
        #     stim_times[ind_sweep] = np.delete(stims, np.where(stims == 0))
        # pickle.dump(stim_times, open('stim_times', 'wb'))
    
    
        
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
        
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for day in days:
# for day in ['25_10_22']:
    os.chdir(day) 
    print(day)
       
    # save shanks separately
    # spikes_allsweeps_4 = pickle.load(open(f'spikes_allsweeps_4','rb'))
    MUA_all_sweeps = pickle.load(open(f'MUA_all_sweeps','rb'))
    for shank_to_save in range(2):
        # os.chdir(f'{day}_{shank_to_save + 1}')
        pickle.dump([i[chanMap[shank_to_save,:],:] for i in MUA_all_sweeps], open(f'MUA_all_sweeps_{shank_to_save + 1}','wb'))
        # pickle.dump([i[chanMap[shank_to_save,:],:] for i in LFP_all_sweeps], open(f'LFP_resampled_shank_{shank_to_save + 1}','wb'))
        # pickle.dump([{chan : spikes_allsweeps_4[i][chan] for chan in list(map(str, chanMap[shank_to_save,:] + 16))} for i in range(10)], open(f'spikes_allsweeps_4_{shank_to_save + 1}','wb'))

    os.chdir('..')



#%%
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for day in days:
    os.chdir(day)
    print(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    if use_kilosort == False:
        spikes_allsweeps_5 = pickle.load(open(f'spikes_allsweeps_5','rb'))
        spikes_allsweeps_4 = pickle.load(open(f'spikes_allsweeps_4','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    
    # os.chdir('..')
    for shank_to_save in range(2):
        # os.chdir(f'{day}_{shank_to_save + 1}')
        pickle.dump([i[chanMap[shank_to_save,:],:] for i in LFP_all_sweeps], open(f'LFP_resampled_shank_{shank_to_save + 1}','wb'))
        pickle.dump([{chan : spikes_allsweeps_4[i][chan] for chan in list(map(str, chanMap[shank_to_save,:] + 16))} for i in range(10)], open(f'spikes_allsweeps_4_{shank_to_save + 1}','wb'))
        pickle.dump([{chan : spikes_allsweeps_5[i][chan] for chan in list(map(str, chanMap[shank_to_save,:] + 16))} for i in range(10)], open(f'spikes_allsweeps_5_{shank_to_save + 1}','wb'))
    os.chdir('..')



#%%
# plot examples
# raster plot
for day in os.listdir():
    os.chdir(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))


    fig, ax = plt.subplots()
    i = 0
    for key, value in spikes_allsweeps[0].items():
        ax.plot(value, np.argwhere(chanMap == i)[0][0] * -np.ones_like(value), 'k.', markersize = 1)
        ax.set_xlim(200*new_fs,220*new_fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        i +=1
    plt.savefig('example_MUA', dpi = 1000)
    
    
    fig, ax = plt.subplots()
    for i in range(len(LFP_all_sweeps[0])):
        ax.plot(LFP_all_sweeps[8][i,:] + np.argwhere(chanMap == i)[0][0] *15000 * -np.ones_like(LFP_all_sweeps[8][i,:]), linewidth = 0.5)
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
    
#%% plot individual sweeps of days with too many sweeps

# for day in ['20_10_22','21_10_22','22_10_22','24_10_22','25_10_22']:
#     os.chdir(day)
#     LFP_all_sweeps = pickle.load(open('LFP_resampled_total','rb'))
#     if use_kilosort == False:
#         spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}_total','rb'))
#     else:
#         spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort_total','rb'))
#     stim_times = pickle.load(open('stim_times_total','rb'))



  
    
#%%
# ---------------------------------------------------------------------------------------- LFP -------------------------------------------------------------------------------------------
do_shift = False

to_plot_1 = [0,1,2,3]
to_plot_2 = [4,5,6,7,8,9]

for day in [i for i in os.listdir() if os.path.isdir(i)]:
    os.chdir(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    
    def LFP_average(sweeps_to_plot, stims = stim_times):
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
    
    LFP_min = np.empty([len(LFP_all_sweeps), 32])
    LFP_min[:] = np.NaN
    LFP_min_rel = np.empty([len(LFP_all_sweeps), 32])
    LFP_min_rel[:] = np.NaN
    LFP_std = np.zeros([len(LFP_all_sweeps), 32])
    LFP_std[:] = np.NaN
    LFP_slope = np.empty([len(LFP_all_sweeps), 32])
    LFP_slope[:] = np.NaN
    
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
    
        
    # fig, ax = plt.subplots(8,4, sharey = True) 
    # for ind, ax1 in enumerate(list(ax.flatten())):                        
    #     ax1.plot(LFP_average(to_plot_1_LFP)[chanMap[ind],:] - LFP_average(to_plot_1_LFP)[chanMap[ind],200], 'b')
    #     ax1.plot(LFP_average(to_plot_2_LFP)[chanMap[ind],:] - LFP_average(to_plot_2_LFP)[chanMap[ind],200], 'r')
    #     # if chan in LFP_resp_channels:
    #     #     ax[np.argwhere(chanMap == chan)[0][0]].set_facecolor("y")
    #     ax1.set_title(str(chanMap[ind]), size = 6)
    #     ax1.set_xlim([150,300])
    # plt.savefig(f'LFP_{to_plot_1_LFP}_vs_{to_plot_2_LFP}', dpi = 1000)
    
    spacer = 2000
    fig, ax = plt.subplots(1, 2, figsize = (5,10))
    fig.suptitle(f'{day}')
    LFP_before = LFP_average(to_plot_1_LFP)
    LFP_after = LFP_average(to_plot_2_LFP)
    for ind in range(16):
        ax[0].plot(LFP_before[chanMap[0,ind],:] + ind * -spacer *np.ones_like(LFP_before[chanMap[0,ind],:]), 'b', linewidth = 1)                 
        ax[0].plot(LFP_after[chanMap[0,ind],:] + ind * -spacer *np.ones_like(LFP_after[chanMap[0,ind],:]), 'r', linewidth = 1) 
        ax[1].plot(LFP_before[chanMap[1,ind],:] + ind * -spacer *np.ones_like(LFP_before[chanMap[1,ind],:]), 'b', linewidth = 1)                 
        ax[1].plot(LFP_after[chanMap[1,ind],:] + ind * -spacer *np.ones_like(LFP_after[chanMap[1,ind],:]), 'r', linewidth = 1)                            
        # ax.set_xlim([150,300])
    # ax.set_yticks(np.linspace(-(spacer*(31 - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace((31 - total_shift),0,tot_chans).astype(int), size = 6)
    plt.tight_layout()
    plt.savefig(f'LFP_laminar_{to_plot_1_LFP}_vs_{to_plot_2_LFP}', dpi = 1000)
    
     
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
    
    
    LFP_responses = np.zeros([len(LFP_all_sweeps), 32, 600])
    LFP_responses[:] = np.NaN
    for sweep in range(len(LFP_all_sweeps)):
        LFP_responses[sweep, :, :] = LFP_average([sweep], stims = stims_for_LFP)
    
    
    # os.chdir(home_directory)
    # os.chdir(f'analysis_{day}')
    # np.savetxt('LFP_min.csv', LFP_min, delimiter = ',')
    # np.savetxt('LFP_slope.csv', LFP_slope, delimiter = ',')
    # np.savetxt('LFP_before.csv', LFP_before, delimiter = ',')
    # np.savetxt('LFP_after.csv', LFP_after, delimiter = ',')
    # np.savetxt('LFP_min_rel.csv', LFP_min_rel, delimiter = ',')
    # np.savetxt('LFP_min_rel_change.csv', LFP_min_rel_change, delimiter = ',')
    # np.savetxt('LFP_slope_rel.csv', LFP_slope_rel, delimiter = ',')
    # np.savetxt('LFP_slope_rel_change.csv', LFP_slope_rel_change, delimiter = ',')
    # np.savetxt('to_plot_1_LFP.csv', to_plot_1_LFP, delimiter = ',')
    # np.savetxt('to_plot_2_LFP.csv', to_plot_2_LFP, delimiter = ',')
    # np.save('LFP_responses.npy', LFP_responses)
    # if do_shift:
    #     np.save('LFP_shift_all.csv', LFP_shift_all)

    os.chdir('..')
    # os.chdir('..')



#%% CSD of averaged LFP stims before vs after

do_shift = False

to_plot_1 = [0,1,2,3]
to_plot_2 = [4,5,6,7,8,9]

# for day in ['10_08_22']:
for day in [i for i in os.listdir() if os.path.isdir(i)]:
    os.chdir(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    
    
    #smooth across channels for the CSDs
    def CSD_average(sweeps_to_plot, stims = stim_times, smoothing = False, smooth_over = 1, time_before = 0.2, time_after = 0.4, coordinates = coordinates, chanMap = chanMap):
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
        to_plot = np.zeros([len(chanMap), int((time_before + time_after)*new_fs), len(sweeps_to_plot)])    
        for ind_sweep, sweep in enumerate(sweeps_to_plot):
            #chan x time x stim
            if stims == stim_times:
                # then stims[sweep] gives you the right ones. but if your stims are a single list of stims it fucks it up (slow wave CSD)
                curr_to_plot = np.zeros([len(chanMap), int((time_before + time_after)*new_fs), len(stims[sweep])])
                curr_stims = list(stims[sweep])
            elif len(stims) == 1:
                curr_to_plot = np.zeros([len(chanMap), int((time_before + time_after)*new_fs), len(stims[ind_sweep])])
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
                    curr_to_plot[:,:,ind_stim] = np.transpose(elephant.current_source_density.estimate_csd(curr_LFP, coordinates = coordinates, method = 'StandardCSD', process_estimate=False))
            to_plot[:,:,ind_sweep] = np.squeeze(np.mean(curr_to_plot,2)) # average CSD across stims
        return np.squeeze(np.mean(to_plot,2)) #average across sweeps
    
    
    # if do_shift:
    #     shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
    #     # you need to add the last XX (total shift) channels to be able to subtract every image with the next one if there's a shift
    #     # THIS IS ASSUMING YOU HAVE POSITIVE SHIFT (CHANNEL NUMBERS GET BIGGER), which is why I call max function if not would have to change it
    #     total_shift = max(shift)
    # else:
    #     total_shift = 0
        
    # # account for the shift in channels
    # CSD_all = np.asarray([np.transpose(CSD_average([i], smoothing = True)) for i in range(10)])

    # if total_shift == 0:
    #     CSD_before = np.transpose(CSD_average(to_plot_1, smoothing = True))
    #     CSD_after = np.transpose(CSD_average(to_plot_2, smoothing = True))
    
    # else:
    #     CSD_before = np.mean(np.asarray([CSD_all[i, :, shift[i]:(32 - (total_shift -shift[i]))] for i in to_plot_1]), axis = 0)
    #     CSD_after = np.mean(np.asarray([CSD_all[i, :, shift[i]:(32 - (total_shift -shift[i]))] for i in to_plot_2]), axis = 0)
    
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
    
    # vmax_overall = np.max(np.concatenate((interpolate_CSD(CSD_before[:,1:-1]), interpolate_CSD(CSD_after[:,1:-1]))))
    # vmin_overall = np.min(np.concatenate((interpolate_CSD(CSD_before[:,1:-1]), interpolate_CSD(CSD_after[:,1:-1]))))
    
    # os.chdir(home_directory)
    
    
    
    fig, ax = plt.subplots(1,4, figsize = (10,10), sharex = True)
    fig.suptitle(f'{day}')
    spacer = 1
    for shank in range(2):
        to_plot_1_CSD = CSD_average(to_plot_1, chanMap = chanMap[shank, :], smoothing = True, smooth_over = 2)
        to_plot_2_CSD = CSD_average(to_plot_2, chanMap = chanMap[shank, :], smoothing = True, smooth_over = 2)
        vmin = min(np.min(to_plot_1_CSD[:,223:]), np.min(to_plot_2_CSD[:,223:]))
        vmax = min(np.max(to_plot_1_CSD[:,223:]), np.max(to_plot_2_CSD[:,223:]))
        im = ax[0+shank*2].imshow(to_plot_1_CSD, cmap = 'jet', aspect = 40, vmin = vmin, vmax = vmax)
        im = ax[1+shank*2].imshow(to_plot_2_CSD, cmap = 'jet', aspect = 40, vmin = vmin, vmax = vmax)
        ax[0+shank*2].set_title(f'shank {shank+1}, before')
        ax[1+shank*2].set_title(f'shank {shank+1}, after')
        # to_plot = CSD_average(LFP = LFP[chanMap[shank,:],:], chanMap = np.arange(0,16), smoothing = True, smooth_over = 2)
        # scaling = (np.max(np.abs(to_plot)) + 50)/(spacer*4)
        # for ind in range(chans_per_shank):
        #     ax[shank].plot(to_plot[ind,:]/scaling + (ind*spacer*np.ones_like(to_plot[ind,:])), 'b', linewidth = 1, color = 'k')                     
        ax[shank].set_xlim([150,300])
    plt.tight_layout()
    plt.savefig(f'CSD_{to_plot_1_LFP}_vs_{to_plot_2_LFP}', dpi = 1000)




    fig, ax = plt.subplots(1,2)
    fig.suptitle(f'{day}')
    spacer = 1
    for shank in range(2):
        to_plot_1_CSD = CSD_average(to_plot_1, chanMap = chanMap[shank, :], smoothing = True, smooth_over = 2)
        to_plot_2_CSD = CSD_average(to_plot_2, chanMap = chanMap[shank, :], smoothing = True, smooth_over = 2)
        to_plot_difference = to_plot_2_CSD - to_plot_1_CSD
        vmin = np.min(to_plot_difference[:,223:])
        vmax = np.max(to_plot_difference[:,223:])
        im = ax[shank].imshow(to_plot_difference, cmap = 'jet', aspect = 40, vmin = vmin, vmax = vmax)
        ax[shank].set_title(f'shank {shank+1}')
        # to_plot = CSD_average(LFP = LFP[chanMap[shank,:],:], chanMap = np.arange(0,16), smoothing = True, smooth_over = 2)
        # scaling = (np.max(np.abs(to_plot)) + 50)/(spacer*4)
        # for ind in range(chans_per_shank):
        #     ax[shank].plot(to_plot[ind,:]/scaling + (ind*spacer*np.ones_like(to_plot[ind,:])), 'b', linewidth = 1, color = 'k')                     
        ax[shank].set_xlim([150,300])
    plt.tight_layout()
    plt.savefig(f'CSD_difference', dpi = 1000)


    
    # smooth_over = 8
    # # how many timpoints to smooth over for the difference plots
    # fig, ax = plt.subplots()
    # fig.suptitle('before')
    # im = ax.imshow(interpolate_CSD(CSD_before), cmap = 'jet', vmin = vmin_overall, vmax = vmax_overall)
    # ax.set_xlim([150,300])
    # plt.savefig('CSD before', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD before', dpi = 1000)
    
    # fig, ax = plt.subplots()
    # fig.suptitle('after')
    # im = ax.imshow(interpolate_CSD(CSD_after), cmap = 'jet', vmin = vmin_overall, vmax = vmax_overall)
    # ax.set_xlim([150,300])
    # plt.savefig('CSD after', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD after', dpi = 1000)
    
    # fig, ax = plt.subplots()
    # fig.suptitle('diff')
    # #smooth over time too
    # im = ax.imshow(interpolate_CSD(smooth(CSD_after, smooth_over) - smooth(CSD_before, smooth_over)), cmap = 'jet', vmin = vmin_overall, vmax = vmax_overall)
    # ax.set_xlim([150,300])
    # plt.savefig('CSD diff', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD diff', dpi = 1000)


    # # #CSD traces    
    # fig, ax = plt.subplots(figsize = (8,10))
    # spacer = 380
    # tot_chans = np.transpose(CSD_before).shape[0]
    # for ind in range(tot_chans):
    #     ax.plot(np.transpose(CSD_before)[ind,:] + ind* -spacer *np.ones_like(np.transpose(CSD_before)[ind,:]), 'k', linewidth = 1)                                   
    # ax.set_xlim([200,300])
    # ax.set_yticks(np.linspace(-(spacer*(31 - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace((31 - total_shift),0,tot_chans).astype(int), size = 6)
    # plt.tight_layout()
    # plt.savefig('CSD before traces', dpi = 1000)

    # fig, ax = plt.subplots(figsize = (8,10))
    # spacer = 380
    # tot_chans = np.transpose(CSD_after).shape[0]
    # for ind in range(tot_chans):
    #     ax.plot(np.transpose(CSD_after)[ind,:] + ind* -spacer *np.ones_like(np.transpose(CSD_after)[ind,:]), 'k', linewidth = 1)                                   
    # ax.set_xlim([200,300])
    # ax.set_yticks(np.linspace(-(spacer*(31 - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace((31 - total_shift),0,tot_chans).astype(int), size = 6)
    # plt.tight_layout()
    # plt.savefig('CSD after traces', dpi = 1000)
    
    # fig, ax = plt.subplots(figsize = (8,10))
    # spacer = 380
    # tot_chans = np.transpose(CSD_after).shape[0]
    # for ind in range(tot_chans):
    #     ax.plot(np.transpose(smooth(CSD_after, smooth_over) - smooth(CSD_before, smooth_over))[ind,:] + ind* -spacer *np.ones_like(np.transpose(CSD_after - CSD_before)[ind,:]), 'k', linewidth = 1)                                   
    # ax.set_xlim([200,300])
    # ax.set_yticks(np.linspace(-(spacer*(31 - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace((31 - total_shift),0,tot_chans).astype(int), size = 6)
    # plt.tight_layout()
    # plt.savefig('CSD diff traces', dpi = 1000)

    # #for channel allocation: plot CSD plot for every sweep
    # fig, ax = plt.subplots(2,5, figsize = (15,15))
    # for ax1_ind, ax1 in enumerate(list(ax.flatten())):
    #     ax1.imshow(CSD_average([ax1_ind], smoothing = True), cmap = 'jet', vmin = vmin_overall, vmax = vmax_overall, aspect = 15)
    #     ax1.set_xlim([150,300])
    #     ax1.set_yticks(list(range(32)))
    #     ax1.set_yticklabels(list(map(str, ax1.get_yticks())), size = 10)
    # plt.tight_layout()
    # plt.savefig('CSD all sweeps', dpi = 1000, format = 'jpg')
    # plt.savefig('CSD all sweeps', dpi = 1000)

    # # CSD traces for every sweep
    # # fig, ax = plt.subplots(32,1, sharey = True) 
    # fig, ax = plt.subplots(2,5, figsize = (8,10))
    # spacer = 380
    # for ax1_ind, ax1 in enumerate(list(ax.flatten())):
    #     to_plot = CSD_average([ax1_ind], smoothing = True)
    #     for ind in range(len(LFP_all_sweeps[0])):
    #         ax1.plot(to_plot[ind,:] + ind* -spacer *np.ones_like(to_plot[ind,:]), 'k', linewidth = 1)                                   
    #         # ax1.set_title(str(chanMap[ind]), size = 2)
    #     ax1.set_xlim([200,300])
    #     ax1.set_yticks(np.linspace(-(spacer*31), 0, 32))
    #     if ax1_ind == 0 or ax1_ind == 5:
    #         ax1.set_yticklabels(np.linspace(31,0,32).astype(int), size = 6)
    #     else:
    #         ax1.set_yticklabels([])
    # # ax.set_yticklabels([32,24,16,8,1])
    # plt.tight_layout()
    # plt.savefig('CSD_laminar all sweeps', dpi = 1000)


    # # os.chdir(home_directory)
    # os.chdir(f'analysis_{day}')
    # np.save('CSD_all.npy', CSD_all)
    # np.savetxt('CSD_before.csv', CSD_before, delimiter = ',')
    # np.savetxt('CSD_after.csv', CSD_after, delimiter = ',')

    os.chdir('..')
    # os.chdir('..')
    
    
#%% plot PSTH responses
# os.chdir(home_directory)
do_shift = False

to_plot_1_PSTH = [0,1,2,3]
to_plot_2_PSTH = [4,5,6,7,8,9]


for day in [i for i in os.listdir() if os.path.isdir(i)]:
    os.chdir(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    
    PSTH_resp_magn = np.empty([len(LFP_all_sweeps), 32])
    PSTH_resp_magn[:] = np.NaN
    PSTH_resp_magn_rel = np.empty([len(LFP_all_sweeps), 32])
    PSTH_resp_magn_rel[:] = np.NaN
    PSTH_resp_peak = np.empty([len(LFP_all_sweeps), 32])
    PSTH_resp_peak[:] = np.NaN
    PSTH_resp_peak_rel = np.empty([len(LFP_all_sweeps), 32])
    PSTH_resp_peak_rel[:] = np.NaN
    
    if day == '10_08_22':
        artifact_locs = [99,119]
    else:
        artifact_locs = []

    # if you want to take out certain stims, you have to take them out of stim_times array.
    def PSTH_matrix(sweeps_to_plot, take_out_artifacts = True, artifact_locs = [], stims = stim_times):
        to_plot = np.zeros([299,len(LFP_all_sweeps[0]),len(sweeps_to_plot)])
        for ind_sweep, sweep in enumerate(sweeps_to_plot):
            #PSTH_matrix is mean across trials in one sweep
            PSTH_matrix = np.zeros([299,32])
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

    # calculate peak and magn
    for sweep in range(len(LFP_all_sweeps)):
        temp = PSTH_matrix([sweep], artifact_locs = artifact_locs)
        PSTH_resp_magn[sweep,:] = np.sum(temp[110:200,:], axis = 0)
        for chan in range(32):
            PSTH_resp_peak[sweep,chan] = np.max(smooth(temp[110:200,chan], 6), axis = 0)
    #relative peak and magn
    PSTH_resp_magn_rel = PSTH_resp_magn/np.nanmean(PSTH_resp_magn[to_plot_1_PSTH,:], axis = 0)
    PSTH_resp_peak_rel = PSTH_resp_peak/np.nanmean(PSTH_resp_peak[to_plot_1_PSTH,:], axis = 0)
    PSTH_resp_magn_rel_change = np.nanmean(PSTH_resp_magn_rel[to_plot_2_PSTH,:], axis = 0) - np.nanmean(PSTH_resp_magn_rel[to_plot_1_PSTH,:], axis = 0)
    PSTH_resp_peak_rel_change = np.nanmean(PSTH_resp_peak_rel[to_plot_2_PSTH,:], axis = 0) - np.nanmean(PSTH_resp_peak_rel[to_plot_1_PSTH,:], axis = 0)
    
    # if do_shift:
    #     shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
    #     # you need to add the last XX (total shift) channels to be able to subtract every image with the next one if there's a shift
    #     # THIS IS ASSUMING YOU HAVE POSITIVE SHIFT (CHANNEL NUMBERS GET BIGGER), which is why I call max function if not would have to change it
    #     total_shift = max(shift)
    # else:
    #     total_shift = 0
        
    # PSTH_all = np.asarray([PSTH_matrix([i]) for i in range(10)])[:,:,chanMap]
    # # LFP RESPONSES BEFORE AND AFTER before and after with shift
    # if total_shift == 0:
    #     PSTH_before = np.mean(np.asarray([PSTH_all[i, :, :] for i in to_plot_1]), axis = 0)
    #     PSTH_after = np.mean(np.asarray([PSTH_all[i, :, :] for i in to_plot_2]), axis = 0)
    
    # else:
    #     PSTH_before = np.mean(np.asarray([PSTH_all[i, :, shift[i]:(32 - (total_shift -shift[i]))] for i in to_plot_1]), axis = 0)
    #     PSTH_after = np.mean(np.asarray([PSTH_all[i, :, shift[i]:(32 - (total_shift -shift[i]))] for i in to_plot_2]), axis = 0)
    # tot_chans = PSTH_before.shape[1]
    
    
    
    PSTH_all = PSTH_matrix(to_plot_1_PSTH + to_plot_2_PSTH, artifact_locs = artifact_locs)
    spacer = np.max(PSTH_all)/2
    fig, ax = plt.subplots(1, 2, figsize = (5,10), sharey = True, sharex = True)
    fig.suptitle(f'{day}')
    PSTH_before = smooth(PSTH_matrix(to_plot_1_PSTH, artifact_locs = artifact_locs), 6)
    PSTH_after = smooth(PSTH_matrix(to_plot_2_PSTH, artifact_locs = artifact_locs), 6)
    for ind in range(16):
        ax[0].plot(PSTH_before[:,chanMap[0,ind]] + ind * -spacer *np.ones_like(PSTH_before[:,chanMap[0,ind]]), 'b', linewidth = 1)                 
        ax[0].plot(PSTH_after[:,chanMap[0,ind]] + ind * -spacer *np.ones_like(PSTH_after[:,chanMap[0,ind]]), 'r', linewidth = 1) 
        ax[1].plot(PSTH_before[:,chanMap[1,ind]] + ind * -spacer *np.ones_like(PSTH_before[:,chanMap[1,ind]]), 'b', linewidth = 1)                 
        ax[1].plot(PSTH_after[:,chanMap[1,ind]] + ind * -spacer *np.ones_like(PSTH_after[:,chanMap[1,ind]]), 'r', linewidth = 1)        
    ax[0].set_xlim([50,200])
    ax[0].set_title('shank 1')
    ax[1].set_title('shank 2')                    
    plt.tight_layout()
    plt.savefig(f'Spiking_{to_plot_1_PSTH}_vs_{to_plot_2_PSTH}', dpi = 1000)  

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

    
    # # PSTH for every sweep to check depth
    # fig, ax = plt.subplots(1,10, figsize = (15,15), sharey = True)
    # spacer = max([np.max(PSTH_matrix([i])) for i in range(10)])/2
    # # spacer = 0.05 # horizontal space between lines
    # for ind, ax1 in enumerate(list(ax.flatten())):
    #     to_plot = PSTH_matrix([ind])
    #     for chan in range(32):                        
    #         ax1.plot(smooth(to_plot[:,chan],6) + np.argwhere(chanMap == chan)[0][0] * -spacer * np.ones_like(to_plot[:,chanMap[chan]]), 'b', linewidth = 1)
    #     ax1.set_xlim([50,200])   
    #     ax1.set_yticks(np.linspace(-(spacer*31), 0, 32))
    #     if ind == 0:
    #         ax1.set_yticklabels(np.linspace(31,0,32).astype(int), size = 6)
    #     # else:
    #     #     ax1.set_yticklabels([])
    # plt.tight_layout()
    # plt.savefig('Spiking_all_sweeps', dpi = 1000)  

    
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
    
    
    # save PSTH for every sweep and channel
    PSTH_responses = np.zeros([len(LFP_all_sweeps), 32, 299])
    PSTH_responses[:] = np.NaN
    for sweep in range(len(LFP_all_sweeps)):
        PSTH_responses[sweep, :, :] = np.transpose(PSTH_matrix([sweep]))
    
    
    # # os.chdir(home_directory)
    # os.chdir(f'analysis_{day}')
    # np.savetxt('PSTH_resp_magn.csv', PSTH_resp_magn, delimiter = ',')
    # np.savetxt('PSTH_resp_peak.csv', PSTH_resp_peak, delimiter = ',')
    # np.savetxt('PSTH_resp_magn_rel.csv', PSTH_resp_magn_rel, delimiter = ',')
    # np.savetxt('PSTH_resp_peak_rel.csv', PSTH_resp_peak_rel, delimiter = ',')
    # np.savetxt('PSTH_resp_magn_rel_change.csv', PSTH_resp_magn_rel_change, delimiter = ',')
    # np.savetxt('PSTH_resp_peak_rel_change.csv', PSTH_resp_peak_rel_change, delimiter = ',')
    # np.savetxt('to_plot_1_PSTH.csv', to_plot_1_PSTH, delimiter = ',')
    # np.savetxt('to_plot_2_PSTH.csv', to_plot_2_PSTH, delimiter = ',')
    # np.savetxt('PSTH_before.csv', PSTH_before, delimiter = ',')
    # np.savetxt('PSTH_after.csv', PSTH_after, delimiter = ',')

    # np.save('PSTH_responses.npy', PSTH_responses)

    # os.chdir('..')
    os.chdir('..')




#%% delta power analysis of inter-stimulus LFP and CSD

do_shift = False

exclude_before = 0.1
# maybe better to take 1 second after stim for slow waves as high change they get fucked up by the stim otherwise?
exclude_after = .9

to_plot_1_delta = [0,1,2,3]
to_plot_2_delta = [4,5,6,7,8,9]

delta_upper = 4
delta_lower = 0.5

b_notch, a_notch = scipy.signal.iirnotch(50, 100, 1000)

do_shift_correction = True

for day in [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]:
    os.chdir(day)
    print(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))

    fftfreq = np.fft.fftfreq(int((5 - exclude_before - exclude_after)*new_fs), d = (1/new_fs))
    
    FFT = np.zeros([len(LFP_all_sweeps), LFP_all_sweeps[0].shape[0], int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
    PSD = np.empty([len(LFP_all_sweeps), LFP_all_sweeps[0].shape[0], int((5 - exclude_before - exclude_after)*new_fs)])
    PSD[:] = np.NaN
    delta_power = np.empty([10, LFP_all_sweeps[0].shape[0]])
    delta_power[:] = np.NaN

    FFT_CSD = np.zeros([len(LFP_all_sweeps), LFP_all_sweeps[0].shape[0], int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
    PSD_CSD = np.empty([len(LFP_all_sweeps), LFP_all_sweeps[0].shape[0], int((5 - exclude_before - exclude_after)*new_fs)])
    PSD_CSD[:] = np.NaN
    delta_power_CSD = np.empty([10, LFP_all_sweeps[0].shape[0]])
    delta_power_CSD[:] = np.NaN

    # do fft for every interstim period, on LFP and CSD
    for ind_sweep, LFP in enumerate(LFP_all_sweeps):
        print(ind_sweep)
        #EXCLUDE first and last stim just in case there isnt enough time, makes it easier
        FFT_current_sweep = np.zeros([len(stim_times[ind_sweep] - 2), LFP_all_sweeps[0].shape[0], int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
        FFT_CSD_current_sweep = np.zeros([len(stim_times[ind_sweep] - 2), LFP_all_sweeps[0].shape[0], int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)

        for ind_stim, stim in enumerate(list(stim_times[ind_sweep][1:-1])):
            curr_LFP = LFP[:, int(stim+exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs)]
            #take out 50Hz noise
            curr_LFP = scipy.signal.filtfilt(b_notch, a_notch, curr_LFP)
            
            FFT_current_sweep[ind_stim, :,:] = np.fft.fft(curr_LFP, axis = 1)
            
            # for shank in range(2):
            #     curr_CSD = elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(np.transpose(curr_LFP[chanMap[shank,:]]), units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False)
            #     FFT_CSD_current_sweep[ind_stim, chanMap[shank,:], :] = np.fft.fft(np.transpose(curr_CSD), axis = 1)
            
        # average across stims
        PSD[ind_sweep,:,:] = np.nanmean(np.abs(FFT_current_sweep)**2, axis = 0) 
        FFT[ind_sweep,:,:] = np.mean(FFT_current_sweep, axis = 0)
        delta_power[ind_sweep,:] = np.mean(PSD[ind_sweep, :, np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 0)

        # PSD_CSD[ind_sweep,:,:] = np.nanmean(np.abs(FFT_CSD_current_sweep)**2, axis = 0) 
        # FFT_CSD[ind_sweep,:,:] = np.mean(FFT_CSD_current_sweep, axis = 0)
        # delta_power_CSD[ind_sweep,:] = np.mean(PSD_CSD[ind_sweep, :, np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 0)

    delta_power_rel = delta_power/np.nanmean(delta_power[to_plot_1_delta,:], axis = 0)
    delta_power_rel_change = np.mean(delta_power_rel[to_plot_2_delta,:], axis = 0) - np.mean(delta_power_rel[to_plot_1_delta,:], axis = 0)
    
    # if do_shift:
    #     shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
    #     # you need to add the last XX (total shift) channels to be able to subtract every image with the next one if there's a shift
    #     # THIS IS ASSUMING YOU HAVE POSITIVE SHIFT (CHANNEL NUMBERS GET BIGGER), which is why I call max function if not would have to change it
    #     total_shift = max(shift)
    # else:
    #     total_shift = 0

    # # redo in chanMap order
    # delta_power_ALL = delta_power[:, chanMap]
    # delta_power_CSD_ALL = delta_power_CSD[:, chanMap]

    # if total_shift > 0 and do_shift_correction == True:
    #     delta_power_shifted = np.asarray([delta_power_ALL[i, shift[i]:(32 - (total_shift -shift[i]))] for i in range(10)])
    #     delta_power_CSD_shifted = np.asarray([delta_power_CSD_ALL[i, shift[i]:(32 - (total_shift -shift[i]))] for i in range(10)])
        
    # else:
    #     delta_power_shifted = np.asarray([delta_power_ALL[i,:] for i in range(10)])
    #     delta_power_CSD_shifted = np.asarray([delta_power_CSD_ALL[i,:] for i in range(10)])


    # delta_power_rel_shifted =  delta_power_shifted/np.nanmean(delta_power_shifted[to_plot_1_delta,:], axis = 0)
    # delta_power_CSD_rel_shifted =  delta_power_CSD_shifted/np.nanmean(delta_power_CSD_shifted[to_plot_1_delta,:], axis = 0)

    # tot_chans = delta_power_rel_shifted.shape[1]

    
    spacer = 1
    fig, ax = plt.subplots(1, 2, figsize = (5,10), sharey = True)
    fig.suptitle(f'{day}')
    for chan in range(16):
        ax[0].plot(delta_power_rel[:, chanMap[0, chan]] + (chan + 1) * -spacer, 'b', linewidth = 1)  
        ax[1].plot(delta_power_rel[:, chanMap[1, chan]] + (chan + 1) * -spacer, 'b', linewidth = 1)                                
        # ax[0].plot(delta_power_rel_shifted[:, chanMap[0, chan]] + (chan + 1) * -spacer *np.ones_like(delta_power_rel_shifted[:,chan]), 'b', linewidth = 1)  
        # ax[1].plot(delta_power_rel_shifted[:, chanMap[1, chan]] + (chan + 1) * -spacer *np.ones_like(delta_power_rel_shifted[:,chan]), 'b', linewidth = 1)                                
    for shank in range(2):
        ax[shank].set_yticks(np.linspace(-15, 0, 16))
        ax[shank].grid()
    ax[0].set_ylim([-17,1])
    # ax.axhline(y = [0,1,2,3,4,5])
    plt.tight_layout()
    plt.savefig('delta power timecourse', dpi = 1000)          


    # #Time course of LFP delta power (drift correction)
    # spacer = 1
    # fig, ax = plt.subplots(figsize = (5,10))
    # fig.suptitle('LFP delta')
    # for chan in range(tot_chans):
    #     ax.plot(delta_power_rel_shifted[:,chan] + (chan + 1) * -spacer *np.ones_like(delta_power_rel_shifted[:,chan]), 'b', linewidth = 1)                 
    # ax.set_yticks(np.linspace(-(spacer*(31 - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace((31 - total_shift),0,tot_chans).astype(int), size = 6)
    # ax.set_xticks(np.arange(10))
    # ax.axvline(x = 3.5, linestyle = '--', linewidth = 1)
    # plt.grid()
    # # ax.axhline(y = [0,1,2,3,4,5])
    # plt.tight_layout()
    # plt.savefig('delta power timecourse', dpi = 1000)          
    
    # #average before and after as electrode picture
    # fig, ax = plt.subplots(figsize = (3,10)) 
    # fig.suptitle('LFP delta')
    # plot = ax.imshow(np.mean(delta_power_rel_shifted[to_plot_2_delta,:], axis = 0)[:,np.newaxis], aspect = 0.25, cmap = 'jet')
    # # ax.set_yticks(np.linspace(-(spacer*(31 - total_shift)), 0, tot_chans))
    # # ax.set_yticklabels(np.linspace((31 - total_shift),0,tot_chans).astype(int), size = 6)
    # fig.colorbar(plot)
    # plt.tight_layout()
    # plt.savefig('delta power diff colorplot', dpi = 1000)          

    
    # #Time course of CSD delta power (drift correction)
    # spacer = 1
    # fig, ax = plt.subplots(figsize = (5,10))
    # fig.suptitle('CSD delta')
    # for chan in range(tot_chans):
    #     ax.plot(delta_power_CSD_rel_shifted[:,chan] + (chan + 1) * -spacer *np.ones_like(delta_power_CSD_rel_shifted[:,chan]), 'b', linewidth = 1)                 
    # ax.set_yticks(np.linspace(-(spacer*(31 - total_shift)), 0, tot_chans))
    # ax.set_yticklabels(np.linspace((31 - total_shift),0,tot_chans).astype(int), size = 6)
    # ax.set_xticks(np.arange(10))
    # ax.axvline(x = 3.5, linestyle = '--', linewidth = 1)
    # plt.grid()
    # # ax.axhline(y = [0,1,2,3,4,5])
    # plt.tight_layout()
    # plt.savefig('delta power CSD timecourse', dpi = 1000)          
    
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
    # PSD_before = np.mean(PSD[to_plot_1_delta,:, :], axis = 0)
    # PSD_after = np.mean(PSD[to_plot_2_delta, :, :], axis = 0)
    # fig, ax = plt.subplots(8,4) 
    # for ind, ax1 in enumerate(list(ax.flatten())):        
    #     ax1.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 30 >= fftfreq))[0]], np.abs(PSD_before[chanMap[ind], np.where(np.logical_and(0.1 <= fftfreq , 30 >= fftfreq))[0]]), 'b')
    #     ax1.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 30 >= fftfreq))[0]], np.abs(PSD_after[chanMap[ind], np.where(np.logical_and(0.1 <= fftfreq , 30 >= fftfreq))[0]]), 'r')
    # plt.savefig('rel_delta_change.jpg', dpi = 1000, format = 'jpg')
    
    # fig, ax = plt.subplots()
    # fig.suptitle('delta_resp')
    # ax.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]], np.abs(PSD_before_delta_resp[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]]), 'b')
    # ax.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]], np.abs(PSD_after_delta_resp[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]]), 'r')
    
    # plt.savefig(f'PSD_{to_plot_1_delta}_vs_{to_plot_2_delta}', dpi = 1000)
    
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
    
    
    # # save everything
    # # os.chdir(home_directory)
    # os.chdir(f'analysis_{day}')
    
    # np.savetxt('fftfreq.csv', fftfreq, delimiter = ',')
    # np.savetxt('delta_power.csv', delta_power, delimiter = ',')
    # # np.savetxt('delta_power_rel.csv', delta_power_rel, delimiter = ',')
    # # np.savetxt('delta_power_rel_change.csv', delta_power_rel_change, delimiter = ',')
    # np.savetxt('delta_power_rel_shifted.csv', delta_power_rel_shifted, delimiter = ',')
    # np.savetxt('delta_power_shifted.csv', delta_power_shifted, delimiter = ',')
    # np.save('PSD.npy', PSD)
    
    # np.savetxt('delta_power_CSD.csv', delta_power_CSD, delimiter = ',')
    # np.savetxt('delta_power_CSD_rel_shifted.csv', delta_power_CSD_rel_shifted, delimiter = ',')
    # np.savetxt('delta_power_CSD_shifted.csv', delta_power_CSD_shifted, delimiter = ',')
    # np.save('PSD_CSD.npy', PSD_CSD)

    # np.savetxt('to_plot_1_delta.csv', to_plot_1_delta, delimiter = ',')
    # np.savetxt('to_plot_2_delta.csv', to_plot_2_delta, delimiter = ',')
    # np.save('delta_lower.npy', delta_lower)
    # np.save('delta_upper.npy', delta_upper)

    # os.chdir('..')
    os.chdir('..')


# delta power of the CSD
# 

#%% SLOW WAVES extracting and CSD
# extract slow waves. You want waveform, firstamp, secondamp, firstslope, secondslope and duration for every sweep. as a list because number chagnes for every sweep and channel

redo_SW_extraction = True

UP_std_cutoff = 1.15

redo_CSD_SW = True
gaussian = 1


for day in [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]:
    os.chdir(day)
    print(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    
    
    if redo_SW_extraction == False:
        os.chdir([i for i in os.listdir() if 'analysis' in i][0])
        
        pickle.load(open('UP_Cross_sweeps','rb'))
    
        SW_waveform_sweeps_avg = np.load('SW_waveform_sweeps_avg.npy')
        SW_frequency_sweeps_avg = np.load('SW_frequency_sweeps_avg.npy', )
        SW_spiking_sweeps_avg = np.load('SW_spiking_sweeps_avg.npy', )
    
        Peak_dur_sweeps_avg_overall = np.load('Peak_dur_sweeps_avg_overall.npy', )
        SW_fslope_sweeps_avg_overall = np.load('SW_fslope_sweeps_avg_overall.npy', )
        SW_sslope_sweeps_avg_overall = np.load('SW_sslope_sweeps_avg_overall.npy', )
        SW_famp_sweeps_avg_overall = np.load('SW_famp_sweeps_avg_overall.npy', )
        SW_samp_sweeps_avg_overall = np.load('SW_samp_sweeps_avg_overall.npy', )
        
        spont_spiking = np.load('spont_spiking.npy', )
                
    else:        
        SW_waveform_sweeps = [[[] for i in range(32)] for j in range(len(LFP_all_sweeps))]
        SW_spiking_sweeps = [[[] for i in range(32)] for j in range(len(LFP_all_sweeps))]
        Peak_dur_sweeps = [[[] for i in range(32)] for j in range(len(LFP_all_sweeps))]
        SW_fslope_sweeps = [[[] for i in range(32)] for j in range(len(LFP_all_sweeps))]
        SW_sslope_sweeps = [[[] for i in range(32)] for j in range(len(LFP_all_sweeps))]
        SW_famp_sweeps = [[[] for i in range(32)] for j in range(len(LFP_all_sweeps))]
        SW_samp_sweeps = [[[] for i in range(32)] for j in range(len(LFP_all_sweeps))]
        UP_Cross_sweeps = [[[] for i in range(32)] for j in range(len(LFP_all_sweeps))]
        
        spont_spiking = np.zeros([10,32])
        
        #the average you want as a numpy array to manipulate later on
        SW_frequency_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64])
        SW_waveform_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64, 1000])
        SW_spiking_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64, 1000])
        Peak_dur_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64])
        SW_fslope_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64])
        SW_sslope_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64])
        SW_famp_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64])
        SW_samp_sweeps_avg = np.zeros([len(LFP_all_sweeps), 64])
        
        exclude_before = 0.1
        # maybe better to take 1 second after stim for slow waves as high change they get fucked up by the stim otherwise?
        exclude_after = 1.9
        duration_criteria = 100
        
        # filter in slow wave range, then find every time it goes under 2xSD i.e.=  upstate
        for ind_sweep, LFP in enumerate(LFP_all_sweeps):
            LFP_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 2*pq.Hz).as_array()
            #EXCLUDE first and last stim just in case there isnt enough time, makes it easier
            for ind_stim, stim in enumerate(list(stim_times[ind_sweep][1:-1])):
                if stim == 0:
                    continue
                print(ind_sweep, ind_stim)
                curr_LFP_filt_total = LFP_filt[int(stim):int(stim + 5*new_fs), :]
                curr_LFP_filt = LFP_filt[int(stim + exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs), :]
                for chan in range(32):
                    
                    # because spiking is saved as dict of channels need to convert it to list to be able to access channels
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
                    
                    UP_LFP = np.where(curr_LFP_filt[:,chan] < -UP_std_cutoff*np.std(curr_LFP_filt[:,chan]))[0]
                    
                    # If no UP crossing after
                    UP_LFP = np.delete(UP_LFP, UP_LFP > UP_Cross[-1])
                    
                    # only LFP points within 500ms of a UP Crossing afterwards
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
                        
                        Peak_dur_sweeps[ind_sweep][chan].append(DOWN_Cross_after[i] - DOWN_Cross_before[i])
                        
                        #save filtered LFP
                        SW_waveform_sweeps[ind_sweep][chan].append(curr_LFP_filt_total[int(UP_Cross_after[i] - 0.5*new_fs + exclude_after*new_fs) : int(UP_Cross_after[i] + 0.5*new_fs + exclude_after*new_fs), chan])
                        
                        #save spiking (as 1ms bins)
                        temp_spiking = np.zeros(1000)
                        # set all spikes there as 1. So take out spikes within 500ms of UP crossing, then subtract 500ms before UP crossing to start at 0
                        temp_spiking[np.round(chan_spiking[np.logical_and(int(UP_Cross_after[i] + exclude_after*new_fs + stim - 0.5*new_fs) < chan_spiking, int(UP_Cross_after[i] + exclude_after*new_fs + stim + 0.5*new_fs) > chan_spiking)] - int(UP_Cross_after[i] + exclude_after*new_fs + stim - 0.5*new_fs) - 1).astype(int)] = 1
                        SW_spiking_sweeps[ind_sweep][chan].append(temp_spiking)
                        
                        idx_peak = np.argmax(curr_LFP_filt[UP_Cross_after[i]:DOWN_Cross_after[i],chan])
                        idx_trough = np.argmin(curr_LFP_filt[DOWN_Cross_before[i]:UP_Cross_after[i],chan])
                        
                        SW_fslope_sweeps[ind_sweep][chan].append(np.mean(np.diff(curr_LFP_filt[DOWN_Cross_before[i]:DOWN_Cross_before[i] + idx_trough])))
                        SW_sslope_sweeps[ind_sweep][chan].append(np.mean(np.diff(curr_LFP_filt[DOWN_Cross_before[i] + idx_trough:UP_Cross_after[i]+idx_peak, chan])))
                        
                        SW_famp_sweeps[ind_sweep][chan].append(np.abs(min(curr_LFP_filt[DOWN_Cross_before[i]:UP_Cross_after[i],chan])))
                        SW_samp_sweeps[ind_sweep][chan].append(np.abs(max(curr_LFP_filt[UP_Cross_after[i]:DOWN_Cross_after[i],chan])))
            
        #convert to Hz: divide by number of sweep intervals and seconds per sweep interval                    
        spont_spiking[ind_sweep,:] = spont_spiking[ind_sweep,:]/((5 - exclude_before - exclude_after)*(len(stim_times[ind_sweep]) - 2))
        
        os.chdir([i for i in os.listdir() if 'analysis' in i][0])

        np.save('spont_spiking.npy', spont_spiking)
        
        pickle.dump(UP_Cross_sweeps, open('UP_Cross_sweeps','wb'))
        
        # average over SW, so 1 value per sweep. MAYBE ALSO TAKE OUT OUTLIERS?? so  maybe first concatenate before and after, take out outliers and average then 
        for ind_sweep in range(len(LFP_all_sweeps)):
            for chan in range(32):
                SW_waveform_sweeps_avg[ind_sweep,chan,:] = np.mean(np.asarray(SW_waveform_sweeps[ind_sweep][chan]), axis = 0)
                SW_frequency_sweeps_avg[ind_sweep,chan] = len(Peak_dur_sweeps[ind_sweep][chan])/(len(stim_times[ind_sweep]) - 2) # -2 because exclude first and last stim
                SW_spiking_sweeps_avg[ind_sweep,chan,:] = np.mean(np.asarray(SW_spiking_sweeps[ind_sweep][chan]), axis = 0)
                Peak_dur_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(Peak_dur_sweeps[ind_sweep][chan]))
                SW_fslope_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_fslope_sweeps[ind_sweep][chan]))
                SW_sslope_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_sslope_sweeps[ind_sweep][chan]))
                SW_famp_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_famp_sweeps[ind_sweep][chan]))
                SW_samp_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_samp_sweeps[ind_sweep][chan]))
        
        
        np.save('SW_waveform_sweeps_avg.npy', SW_waveform_sweeps_avg)
        np.save('SW_frequency_sweeps_avg.npy', SW_frequency_sweeps_avg)
        np.save('SW_spiking_sweeps_avg.npy', SW_spiking_sweeps_avg)
        np.save('Peak_dur_sweeps_avg.npy', Peak_dur_sweeps_avg)
        np.save('SW_fslope_sweeps_avg.npy', SW_fslope_sweeps_avg)
        np.save('SW_sslope_sweeps_avg.npy', SW_sslope_sweeps_avg)
        np.save('SW_famp_sweeps_avg.npy', SW_famp_sweeps_avg)
        np.save('SW_samp_sweeps_avg.npy', SW_samp_sweeps_avg)
        
        
        #redo values with the mean waveforms:
        Peak_dur_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), 32])
        SW_fslope_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), 32])
        SW_sslope_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), 32])
        SW_famp_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), 32])
        SW_samp_sweeps_avg_overall = np.zeros([len(LFP_all_sweeps), 32])
        
        for ind_sweep in range(len(LFP_all_sweeps)):
            for chan in range(32):
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



    #%
    # # Slow wave CSD 
    # 2 ways: CSD of slow waves for ones detected in every channel, or unique slow waves across channels. For that obviously need to define slow waves as unique events. So maybe if the UP crossing is within 500ms of each other?
    def CSD_average(sweeps_to_plot, stims = stim_times, smoothing = False, smooth_over = 1, time_before = 0.2, time_after = 0.4):
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
                        curr_LFP = neo.core.AnalogSignal(np.transpose(scipy.ndimage.gaussian_filter1d(LFP_all_sweeps[sweep][chanMap,int(stim - time_before*new_fs):int(stim + time_after*new_fs)], smooth_over, axis = 0)), units = 'mV', sampling_rate = new_fs*pq.Hz)
                    else:
                        curr_LFP = neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep][chanMap,int(stim - time_before*new_fs):int(stim + time_after*new_fs)]), units = 'mV', sampling_rate = new_fs*pq.Hz)                    
                    curr_to_plot[:,:,ind_stim] = np.transpose(elephant.current_source_density.estimate_csd(curr_LFP, coordinates = coordinates, method = 'StandardCSD', process_estimate=False))
            to_plot[:,:,ind_sweep] = np.squeeze(np.mean(curr_to_plot,2)) # average across stims
        return np.squeeze(np.mean(to_plot,2)) #average across sweeps
 
    
 
    # # # 1. average CSD of slow waves detected in every channel in every sweep (average of CSDs, not CSD of average LFPS)
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
    if os.path.isfile('All_SW_avg_laminar') and redo_CSD_SW == False:
        All_SW_avg_laminar = np.load('All_SW_avg_laminar.npy')
        All_SW_avg_laminar_filt = np.load('All_SW_avg_laminar_filt.npy')
        All_CSD_avg_laminar = np.load('All_CSD_avg_laminar.npy')      
        All_CSD_avg_laminar_filt = np.load('All_CSD_avg_laminar_filt.npy', )      
    else:
        All_SW_avg_laminar = np.zeros([32, len(LFP_all_sweeps), 32, 1000])  # channel detected in, sweep, chans, time
        All_SW_avg_laminar_filt = np.zeros([32, len(LFP_all_sweeps), 32, 1000]) 
        for chan in range(32):
            for sweep in range(len(LFP_all_sweeps)):
                curr_laminar = np.zeros([len(UP_Cross_sweeps[sweep][chan]), 32, 1000]) 
                # filtered below 4Hz
                curr_laminar_filt = np.zeros([len(UP_Cross_sweeps[sweep][chan]), 32, 1000]) 
                for stim_ind, stim in enumerate(UP_Cross_sweeps[sweep][chan]):
                    curr_laminar[stim_ind, :,:] = LFP_all_sweeps[sweep][:, stim - 500:stim+500]
                    curr_laminar_filt[stim_ind, :,:] = np.transpose(elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep][:, stim - 500:stim+500]), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 4*pq.Hz).as_array())
                # average across SWs
                All_SW_avg_laminar[chan, sweep, :,:] = np.mean(curr_laminar, axis = 0)
                All_SW_avg_laminar_filt[chan, sweep, :,:] = np.mean(curr_laminar_filt, axis = 0)
              
        np.save('All_SW_avg_laminar.npy', All_SW_avg_laminar)
        np.save('All_SW_avg_laminar_filt.npy', All_SW_avg_laminar_filt)
    
        #CSD from average LFP profile of slow waves detected in each channel (smooth over channels before doing CSD)
        All_CSD_avg_laminar = np.zeros([32,len(LFP_all_sweeps),32,1000]) 
        All_CSD_avg_laminar_filt = np.zeros([32,len(LFP_all_sweeps),32,1000]) 
        for chan in range(32):
            for sweep in range(len(LFP_all_sweeps)):
                All_CSD_avg_laminar[chan,sweep,:,:] = np.transpose(elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(np.transpose(scipy.ndimage.gaussian_filter1d(All_SW_avg_laminar[chan,sweep,chanMap,:], gaussian, axis = 0)), units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False))
                All_CSD_avg_laminar_filt[chan,sweep,:,:] = np.transpose(elephant.current_source_density.estimate_csd(neo.core.AnalogSignal(np.transpose(scipy.ndimage.gaussian_filter1d(All_SW_avg_laminar_filt[chan,sweep,chanMap,:], gaussian, axis = 0)), units = 'mV', sampling_rate = new_fs*pq.Hz), coordinates = coordinates, method = 'StandardCSD', process_estimate=False))
        np.save('All_CSD_avg_laminar.npy', All_CSD_avg_laminar)      
        np.save('All_CSD_avg_laminar_filt.npy', All_CSD_avg_laminar_filt)      



    # # 3 unique slow waves, from channels in layer 4 and 5:
    # # I think tolerance has to be set quite high, or else you cancel out the CSD profiles if they are a little offset in time
    # CSD_unique_SW_channels = list(np.concatenate([layer_dict[day][0][i] for i in [2,3]])) # NOT in chanmap order
    # CSD_unique_SW_avg = np.zeros([len(LFP_all_sweeps), LFP_all_sweeps[0].shape[0], 1000]) # sweep, chan, time
    # tolerance = 1200
    # if os.path.isfile(f'Unique_SW_CSD_{tolerance}.npy') and redo_CSD_SW == False:
    #     np.load(f'Unique_SW_CSD_{tolerance}.npy')
    # else:
    #     for ind_sweep in range(len(LFP_all_sweeps)): 
    #         # get all unique stim times with tolerance of xsec
    #         all_times = np.sort(np.concatenate([np.asarray(UP_Cross_sweeps[ind_sweep][i]) for i in CSD_unique_SW_channels]))
    #         unique_times = list(all_times[~(np.triu(np.abs(all_times[:,None] - all_times) <= tolerance,1)).any(0)])  # now you have only the unique SW times with a tolerance of x sec
    #         CSD_unique_SW_avg[ind_sweep,:,:] = CSD_average([ind_sweep], stims = [unique_times], time_before = 0.5, time_after = 0.5) 
    #     np.save(f'Unique_SW_CSD_{tolerance}', CSD_unique_SW_avg)

        
    os.chdir('..')
    os.chdir('..')




#%% SLOW WAVES plotting
# SW_resp_channels = []
# SW_resp_channels = list(range(LFP_all_sweeps[0].shape[0]))
do_shift = False

to_plot_1 = [0,1,2,3]
to_plot_2 = [4,5,6,7,8,9]
  
#gaussian of smoothing over channels for CSD that are plotted
gaussian = 1
tolerance = 1200

for day in [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]:
    os.chdir(day)

    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    if use_kilosort == False:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    else:
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_kilosort','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    pickle.load(open('UP_Cross_sweeps','rb'))

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
    All_CSD_avg_laminar = np.load('All_CSD_avg_laminar.npy')      
    All_CSD_avg_laminar_filt = np.load('All_CSD_avg_laminar_filt.npy', )      

    # CSD_unique_SW_avg = np.load(f'Unique_SW_CSD_{tolerance}.npy')

    



    # ------------------------------------------------------------------------------------- laminar LFP and CSD analysis ---------------------------------------------------------------------------------------------------------    
    # which channel to use for SW detection? (not in chanmap) --> use first layer 5 channel
    # channel_for_SW = layer_dict[day][0][3][0]
    channel_for_SW = 15
    #plot average laminar LFP profile over sweeps for SW detected in specific channel
    spacer = 1000
    fig, ax = plt.subplots(1,10, figsize = (17,10), sharey = True) 
    for ind, ax1 in enumerate(list(ax.flatten())):  
        for chan in range(len(LFP_all_sweeps[0])):
            if chan == chanMap[channel_for_SW]:
                ax1.plot(All_SW_avg_laminar_filt[chanMap[channel_for_SW],ind,chan,:] + np.argwhere(chanMap == chan)[0][0] * -spacer * np.ones_like(All_SW_avg_laminar_filt[chanMap[channel_for_SW],ind,chan,:]), linewidth = 1, color = 'r')
            else:
                ax1.plot(All_SW_avg_laminar_filt[chanMap[channel_for_SW],ind,chan,:] + np.argwhere(chanMap == chan)[0][0] * -spacer * np.ones_like(All_SW_avg_laminar_filt[chanMap[channel_for_SW],ind,chan,:]), linewidth = 1, color = 'k')

        ax1.set_yticks(np.linspace(-spacer*31, 0, 32))
        if ind == 0:
            ax1.set_yticklabels(np.linspace(31,0,32).astype(int), size = 6)
        else:
            ax1.set_yticklabels([])
    plt.tight_layout()
    plt.savefig('SW LFP traces all sweeps.jpg', dpi = 1000, format = 'jpg')

    #CSD traces over sweeps (smooth over time a bit)
    spacer = 500
    smooth_over_time = 1
    fig, ax = plt.subplots(1,10, figsize = (17,10), sharey = True) 
    for ind, ax1 in enumerate(list(ax.flatten())):  
        for chan in range(len(LFP_all_sweeps[0])):    
            ax1.plot(smooth(All_CSD_avg_laminar_filt[chanMap[channel_for_SW],ind,chan,:], smooth_over_time) + chan * -spacer * np.ones_like(All_CSD_avg_laminar_filt[chanMap[channel_for_SW],ind,chan,:]), linewidth = 1, color = 'k')
        ax1.set_yticks(np.linspace(-spacer*31, 0, 32))
        if ind == 0:
            ax1.set_yticklabels(np.linspace(31,0,32).astype(int), size = 6)
        else:
            ax1.set_yticklabels([])
    plt.tight_layout()
    plt.savefig('SW CSD traces all sweeps.jpg', dpi = 1000, format = 'jpg')

    #CSD heatmaps over sweeps
    vmax = np.max(np.concatenate([All_CSD_avg_laminar_filt[chanMap[channel_for_SW],ind,:,:] for ind in range(10)]))
    vmin = np.min(np.concatenate([All_CSD_avg_laminar_filt[chanMap[channel_for_SW],ind,:,:] for ind in range(10)]))
    fig, ax = plt.subplots(1,10, figsize = (17,10)) 
    for ind, ax1 in enumerate(list(ax.flatten())):
        to_plot = smooth(np.squeeze(All_CSD_avg_laminar_filt[chanMap[channel_for_SW],ind,:,:]), smooth_over_time, axis = 1)
        to_plot = interpolate_CSD(np.transpose(to_plot))
        ax1.imshow(to_plot, cmap = 'jet', aspect = 10, vmax = vmax, vmin = vmin)
    plt.tight_layout()
    plt.savefig('SW CSD heatmaps all sweeps.jpg', dpi = 1000, format = 'jpg')

    
    if do_shift:
        shift = np.asarray([int(np.median(layer_dict[day][i][0] - layer_dict[day][0][0])) for i in range(10)])
        # you need to add the last XX (total shift) channels to be able to subtract every image with the next one if there's a shift
        # THIS IS ASSUMING YOU HAVE POSITIVE SHIFT (CHANNEL NUMBERS GET BIGGER), which is why I call max function if not would have to change it
        total_shift = max(shift)
    else:
        total_shift = 0
    
    # redo in chanMap order
    LFP_SW_all = All_SW_avg_laminar_filt[chanMap[channel_for_SW], :, chanMap, :]
    if total_shift == 0:
        CSD_before = np.mean(np.asarray([All_CSD_avg_laminar_filt[chanMap[channel_for_SW], i, :, :] for i in to_plot_1]), axis = 0)
        CSD_after = np.mean(np.asarray([All_CSD_avg_laminar_filt[chanMap[channel_for_SW], i, :, :] for i in to_plot_2]), axis = 0)
        LFP_before = np.mean(np.asarray([LFP_SW_all[:, i, :] for i in to_plot_1]), axis = 0)
        LFP_after = np.mean(np.asarray([LFP_SW_all[:, i, :] for i in to_plot_2]), axis = 0)
    else:
        CSD_before = np.mean(np.asarray([All_CSD_avg_laminar_filt[chanMap[channel_for_SW], i, shift[i]:(32 - (total_shift -shift[i])), :] for i in to_plot_1]), axis = 0)
        CSD_after = np.mean(np.asarray([All_CSD_avg_laminar_filt[chanMap[channel_for_SW], i, shift[i]:(32 - (total_shift -shift[i])), :] for i in to_plot_2]), axis = 0)
        LFP_before = np.mean(np.asarray([LFP_SW_all[shift[i]:(32 - (total_shift -shift[i])), i, :] for i in to_plot_1]), axis = 0)
        LFP_after = np.mean(np.asarray([LFP_SW_all[shift[i]:(32 - (total_shift -shift[i])), i, :] for i in to_plot_2]), axis = 0)

    tot_chans = CSD_before.shape[0]

    # LFP SW BEFORE AND AFTER with shift
    spacer = np.max(All_SW_avg_laminar_filt[chanMap[channel_for_SW],:,:,:])/2
    fig, ax = plt.subplots(figsize = (2.5,10))
    for ind in range(tot_chans):
        ax.plot(LFP_before[ind,:] + ind * -spacer *np.ones_like(LFP_before[ind,:]), 'b', linewidth = 1)                 
        ax.plot(LFP_after[ind,:] + ind * -spacer *np.ones_like(LFP_after[ind,:]), 'r', linewidth = 1)                     
    ax.set_yticks(np.linspace(-(spacer*(31 - total_shift)), 0, tot_chans))
    ax.set_yticklabels(np.linspace((31 - total_shift),0,tot_chans).astype(int), size = 6)
    plt.tight_layout()
    plt.savefig('SW LFP traces before and after', dpi = 1000)  

    #CSD SW before and after with shift    
    spacer = np.max(All_CSD_avg_laminar_filt[chanMap[channel_for_SW],:,:,:])/2
    fig, ax = plt.subplots(figsize = (2.5,10))
    for ind in range(tot_chans):
        ax.plot(CSD_before[ind,:] + ind * -spacer *np.ones_like(CSD_before[ind,:]), 'b', linewidth = 1)                 
        ax.plot(CSD_after[ind,:] + ind * -spacer *np.ones_like(CSD_after[ind,:]), 'r', linewidth = 1)                     
    ax.set_yticks(np.linspace(-(spacer*(31 - total_shift)), 0, tot_chans))
    ax.set_yticklabels(np.linspace((31 - total_shift),0,tot_chans).astype(int), size = 6)
    plt.tight_layout()
    plt.savefig('SW CSD traces before and after', dpi = 1000)  

    #CSD SW heatmap before
    vmax = np.max([CSD_before, CSD_after])
    vmin = np.min([CSD_before, CSD_after])
    fig, ax = plt.subplots(figsize = (10,7)) 
    ax.imshow(interpolate_CSD(np.transpose(CSD_before)), cmap = 'jet', aspect = 10, vmax = vmax, vmin = vmin)
    plt.tight_layout()
    plt.savefig('SW CSD heatmap before.jpg', dpi = 1000, format = 'jpg')

    #CSD SW heatmap after
    fig, ax = plt.subplots(figsize = (10,7)) 
    ax.imshow(interpolate_CSD(np.transpose(CSD_after)), cmap = 'jet', aspect = 10, vmax = vmax, vmin = vmin)
    plt.tight_layout()
    plt.savefig('SW CSD heatmap after.jpg', dpi = 1000, format = 'jpg')

    #CSD SW heatmap diff
    fig, ax = plt.subplots(figsize = (10,7)) 
    ax.imshow(interpolate_CSD(np.transpose(CSD_after - CSD_before)), cmap = 'jet', aspect = 10, vmax = vmax, vmin = vmin)
    plt.tight_layout()
    plt.savefig('SW CSD heatmap diff.jpg', dpi = 1000, format = 'jpg')

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




# ---------------------------------------------------------------------------------SW params and spiking change ---------------------------------------------------------------------------------------

    #relative change in individual params before vs after in all channels (needs drift correction too)
    Freq_change = (np.mean(SW_frequency_sweeps_avg[to_plot_2,:], axis = 0) - np.mean(SW_frequency_sweeps_avg[to_plot_1,:], axis = 0))/np.mean(SW_frequency_sweeps_avg[to_plot_1,:], axis = 0)
    Peak_dur_change = (np.mean(Peak_dur_sweeps_avg[to_plot_2,:], axis = 0) - np.mean(Peak_dur_sweeps_avg[to_plot_1,:], axis = 0))/np.mean(Peak_dur_sweeps_avg[to_plot_1,:], axis = 0)
    Fslope_change = (np.mean(SW_fslope_sweeps_avg[to_plot_2,:], axis = 0) - np.mean(SW_fslope_sweeps_avg[to_plot_1,:], axis = 0))/np.mean(SW_fslope_sweeps_avg[to_plot_1,:], axis = 0)
    Sslope_change = (np.mean(SW_sslope_sweeps_avg[to_plot_2,:], axis = 0) - np.mean(SW_sslope_sweeps_avg[to_plot_1,:], axis = 0))/np.mean(SW_sslope_sweeps_avg[to_plot_1,:], axis = 0)
    Famp_change = (np.mean(SW_famp_sweeps_avg[to_plot_2,:], axis = 0) - np.mean(SW_famp_sweeps_avg[to_plot_1,:], axis = 0))/np.mean(SW_famp_sweeps_avg[to_plot_1,:], axis = 0)
    Samp_change = (np.mean(SW_samp_sweeps_avg[to_plot_2,:], axis = 0) - np.mean(SW_samp_sweeps_avg[to_plot_1,:], axis = 0))/np.mean(SW_samp_sweeps_avg[to_plot_1,:], axis = 0)
    
    Peak_dur_overall_change = (np.nanmean(Peak_dur_sweeps_avg_overall[to_plot_2,:], axis = 0) - np.nanmean(Peak_dur_sweeps_avg_overall[to_plot_1,:], axis = 0))/np.nanmean(Peak_dur_sweeps_avg_overall[to_plot_1,:], axis = 0)
    Fslope_overall_change = (np.nanmean(SW_fslope_sweeps_avg_overall[to_plot_2,:], axis = 0) - np.nanmean(SW_fslope_sweeps_avg_overall[to_plot_1,:], axis = 0))/np.nanmean(SW_fslope_sweeps_avg_overall[to_plot_1,:], axis = 0)
    Sslope_overall_change = (np.nanmean(SW_sslope_sweeps_avg_overall[to_plot_2,:], axis = 0) - np.nanmean(SW_sslope_sweeps_avg_overall[to_plot_1,:], axis = 0))/np.nanmean(SW_sslope_sweeps_avg_overall[to_plot_1,:], axis = 0)
    Famp_overall_change = (np.nanmean(SW_famp_sweeps_avg_overall[to_plot_2,:], axis = 0) - np.nanmean(SW_famp_sweeps_avg_overall[to_plot_1,:], axis = 0))/np.nanmean(SW_famp_sweeps_avg_overall[to_plot_1,:], axis = 0)
    Samp_overall_change = (np.nanmean(SW_samp_sweeps_avg_overall[to_plot_2,:], axis = 0) - np.nanmean(SW_samp_sweeps_avg_overall[to_plot_1,:], axis = 0))/np.nanmean(SW_samp_sweeps_avg_overall[to_plot_1,:], axis = 0)
    
    
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
    
    # np.savetxt('SW_resp_channels.csv', SW_resp_channels, delimiter = ',')
    np.savetxt('Freq_change.csv', Freq_change, delimiter = ',')
    np.savetxt('Peak_dur_change.csv', Peak_dur_change, delimiter = ',')
    np.savetxt('Fslope_change.csv', Fslope_change, delimiter = ',')
    np.savetxt('Sslope_change.csv', Sslope_change, delimiter = ',')
    np.savetxt('Famp_change.csv', Famp_change, delimiter = ',')
    np.savetxt('Samp_change.csv', Samp_change, delimiter = ',')
    np.savetxt('Peak_dur_overall_change.csv', Peak_dur_overall_change, delimiter = ',')
    np.savetxt('Fslope_overall_change.csv', Fslope_overall_change, delimiter = ',')
    np.savetxt('Sslope_overall_change.csv', Sslope_overall_change, delimiter = ',')
    np.savetxt('Famp_overall_change.csv', Famp_overall_change, delimiter = ',')
    np.savetxt('Samp_overall_change.csv', Samp_overall_change, delimiter = ',')
    # np.savetxt('SW_spiking_peak_change.csv', SW_spiking_peak_change, delimiter = ',')
    # np.savetxt('SW_spiking_area_change.csv', SW_spiking_area_change, delimiter = ',')
    
    os.chdir('..')
    os.chdir('..')










#%% analyze UP vs DOWN stim delivery
#UP stim when spikes in 100ms before stim, if not DOWN stim. This is not exactly perfect as often UP/DOWN states not synchronous across channels... choose channel wisely

# UP_DOWN_channels = [38,40]
UP_DOWN_channels = list(range(32))
#UP_DOWN_channels = np.argmax(np.mean(LFP_min[0:4,:], axis = 0)) #channel with biggest deflection during baseline
time_for_detection = 100

UP_stims = [[] for sweep in range(len(stim_times))]
DOWN_stims = [[] for sweep in range(len(stim_times))]
#JUST use LFP responsive channels (or maybe most responsive channel) to separate stims, probably better for this! (can try also all channels separately if doesnt work properly)
for sweep_ind, stims in enumerate(stim_times):
    if (isinstance(UP_DOWN_channels, list) and len(UP_DOWN_channels) > 1):
        spikes = np.sort(np.concatenate(np.asarray(list(spikes_allsweeps[sweep_ind].values()))[UP_DOWN_channels]))
    else:
        spikes = np.sort(np.asarray(list(spikes_allsweeps[sweep_ind].values()))[UP_DOWN_channels])
    for stim in stims:
        # concatenate all spikes of the LFP responsive channels in that sweep
        if spikes[np.logical_and((stim - time_for_detection) < spikes, stim > spikes)].size == 0:
            DOWN_stims[sweep_ind].append(stim)
        else:
            UP_stims[sweep_ind].append(stim)

#plot UP vs DOWN stim delivery LFP responses during baseline:
fig, ax = plt.subplots(8,4, sharey = True) 
for ind, ax1 in enumerate(list(ax.flatten())):                         
    ax1.plot(LFP_average(to_plot_1, stims=DOWN_stims)[chanMap[ind],:] - LFP_average(to_plot_1, stims=DOWN_stims)[chanMap[ind],200], 'b')
    ax1.plot(LFP_average(to_plot_1, stims=UP_stims)[chanMap[ind],:] - LFP_average(to_plot_1, stims=UP_stims)[chanMap[ind],200], 'r')
    ax1.set_title(str(chanMap[ind]))
    ax1.set_xlim([150,300])
            
#plot UP vs DOWN stim delivery PSTH responses during baseline:
fig, ax = plt.subplots(8,4,sharey = True)
for ind, ax1 in enumerate(list(ax.flatten())):  
    ax1.plot(smooth(PSTH_matrix(to_plot_1, stims = DOWN_stims)[:,chanMap[ind]], 6), 'b', linewidth = 1)
    ax1.plot(smooth(PSTH_matrix(to_plot_1, stims = UP_stims)[:,chanMap[ind]], 6), 'r', linewidth = 1) 
    ax1.set_title(str(chanMap[ind]))
    ax1.set_xlim([50,200])
    # ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_xlim([50,200])
    
#plot distribution of frequencies of UP/DOWN deliveries over time
fig, ax = plt.subplots()
ax.bar(range(len(UP_stims)), [len(UP_stims[i])/len(DOWN_stims[i]) for i in range(len(UP_stims))])
# maybe do a t-test with pooled averages for before and after across sweeps showing it's not a significant change


# plot strength of response during UP/DOWN deliveries over time (as % of baseline response). Error bars across stimulations are probably messy as would have to get the individual stims and if there is noise or other it's impossible. Would have to try maybe though.
# better to do error bars across channels for the average plot.
sweeps_to_plot_UP_DOWN = list(range(len(LFP_all_sweeps))) # how many sweeps to look at
DOWN_LFP_overall_magnitude = [np.min(LFP_average(sweeps_to_plot_UP_DOWN, stims = DOWN_stims)[chan,:]) for chan in range(len(chanMap))]
UP_LFP_overall_magnitude = [np.min(LFP_average(sweeps_to_plot_UP_DOWN, stims = UP_stims)[chan,:]) for chan in range(len(chanMap))]
fig, ax = plt.subplots(8,4)
for ind, ax1 in enumerate(list(ax.flatten())):                        
    ax1.plot([np.min(LFP_average([i], stims=DOWN_stims)[chanMap[ind],:])/DOWN_LFP_overall_magnitude[chanMap[ind]] for i in sweeps_to_plot_UP_DOWN], 'b')
    ax1.plot([np.min(LFP_average([i], stims=UP_stims)[chanMap[ind],:])/UP_LFP_overall_magnitude[chanMap[ind]] for i in sweeps_to_plot_UP_DOWN], 'r')
    ax1.set_title(str(chanMap[ind]))

# do average across channels with errorbars across channels

#%% analyze pairings:
# you have to redo the mock pairings with the last baseline before. Extract the paired channel, the threshold and if filter was on from info of intan folder.
# then take LFP signal of paired channel in last baseline before. If UP paired: everytime it crosses the treshold you know when the 5 stims are at 10Hz (so exclude all threshold crossings 500ms after threshold crossing).
# If DOWN paired: Everytime it doesn't cross the threshold for 200ms, 5 stims, then no stims until it crosses the threshold again.

#change to pairing directory
# os.chdir([os.listdir()[i] for i in range(len(os.listdir())) if 'pairing' in os.listdir()[i]][0])
# settings = ET.parse('settings.xml')
# threshold = int(settings.getroot()[0].attrib['AnalogOut1ThresholdMicroVolts'])*10
# paired_channel = settings.getroot()[0].attrib['AnalogOut1Channel']
# if settings.getroot()[0].attrib['AnalogOutHighpassFilterEnabled'] == 'False':
#     high_filtered = False
# else:
#     high_filtered = True

# #if didn't use spikes to pair, just use resampled LFP. if used spikes need to load the real thing and filter
# if high_filtered == False:
#     pair_LFP = LFP_all_sweeps[3][int(paired_channel[3:5]),:]

# crossings = np.argwhere(pair_LFP < threshold)
# #take out first second as LFP is not balanced yet
# crossings = np.delete(crossings, np.where(crossings < 1000))

#now delete all crossings within 500ms of the one before

    