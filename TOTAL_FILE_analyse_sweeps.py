# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 21:26:17 2021

@author: JPDUF
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
import matplotlib.colors as colors

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

#%% FUNCTIONS
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
    to_plot = np.zeros([64, int(0.6*new_fs), len(sweeps_to_plot)])    
    for ind_sweep, sweep in enumerate(sweeps_to_plot):
        curr_to_plot = np.zeros([64, int(0.6*new_fs), len(np.where(stims[sweep] > 0)[0])])
        for ind_stim, stim in enumerate(list(stims[sweep])):
            if stim == 0:
                continue
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



# # if you want to take out certain stims, you have to take them out of stim_times array.
def PSTH_matrix(sweeps_to_plot, take_out_artifacts = True, artifact_locs = [], stims = stim_times):
    to_plot = np.zeros([299,64,len(sweeps_to_plot)])
    for ind_sweep, sweep in enumerate(sweeps_to_plot):
        #PSTH_matrix is mean across trials in one sweep
        PSTH_matrix = np.zeros([299,64])
        bins = np.linspace(1,300,300)
        for ind_chan, chan in enumerate(natsort.natsorted(list(spikes_allsweeps[sweep].keys()))):
            # print(chan)
            currchan = np.zeros([299,len(stims[sweep])])
            for ind_stim, j in enumerate(list(stims[sweep])):
                currchan[:,ind_stim] = np.histogram((spikes_allsweeps[sweep][chan][(j - 0.1*new_fs < spikes_allsweeps[sweep][chan]) & (spikes_allsweeps[sweep][chan] < j+0.2*new_fs)] - (j-0.1*new_fs)), bins)[0]
                if take_out_artifacts:
                    currchan[:,ind_stim][artifact_locs] = 0
            PSTH_matrix[:,ind_chan] = np.squeeze(np.mean(currchan, 1)) # mean across stims for every channel
        to_plot[:,:,ind_sweep] = PSTH_matrix
    return np.squeeze(np.mean(to_plot,2))

# # if you want to take out certain stims, you have to take them out of stim_times array.
def SD_spikes_before(sweeps_to_plot, stims = stim_times, time_before = 100):
    SD_matrix = np.zeros([64,len(sweeps_to_plot)])
    for ind_sweep, sweep in enumerate(sweeps_to_plot):
        bins = np.linspace(1,time_before,time_before)
        for ind_chan, chan in enumerate(natsort.natsorted(list(spikes_allsweeps[sweep].keys()))):
            # print(chan)
            currchan = np.zeros([time_before-1,len(stims[sweep])])
            for ind_stim, j in enumerate(list(stims[sweep])):
                currchan[:,ind_stim] = np.histogram((spikes_allsweeps[sweep][chan][(j - 0.1*new_fs < spikes_allsweeps[sweep][chan]) & (spikes_allsweeps[sweep][chan] < j+0.2*new_fs)] - (j-0.1*new_fs)), bins)[0]
            SD_matrix[ind_chan,ind_sweep] = np.std(np.sum(currchan, 0)/(time_before-1)) # std across stims for every channel
    return SD_matrix




#%% EXAMPLE SLOW WAVES
# raster plot
highpass_cutoff = 4

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i) and 'old' not in i]
# for day in ['160414_D1']:
for day in days:
    if day == '160624_B2':
        continue
    os.chdir(day) 
    print(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    try:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    except FileNotFoundError:
        print(f'spikes with highpass {highpass_cutoff} not found')
        spikes_allsweeps = pickle.load(open([i for i in os.listdir() if 'spikes_allsweeps' in i][0],'rb'))
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    chans_to_plot = list(np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int))
    SW_spiking_channels = list(np.loadtxt('SW_spiking_channels.csv', delimiter = ',', dtype = int))
    os.chdir('..')
    
    sweep = 1
    xlim1 = 137
    xlim2 = 152
    
    # chans_to_plot = list(range(64))
    # chans_to_plot.remove(54)
    # chans_to_plot.remove(56)
    # chans_to_plot.remove(17)
    # chans_to_plot.remove(31)
    b_notch, a_notch = scipy.signal.iirnotch(50, 100, 1000)
    
    i = 1
    fig, ax = plt.subplots(figsize = (8,3))
    for chan, value in enumerate(list(spikes_allsweeps[sweep].values())):
        if chan in SW_spiking_channels:
            ax.plot(value, i * np.ones_like(value), 'k.', markersize = 1)
            ax.set_xlim(xlim1*new_fs,xlim2*new_fs)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            i += 1 
    plt.savefig(f'example_MUA_{sweep}_{xlim1}.pdf', dpi = 1000, format = 'pdf')
    plt.savefig(f'example_MUA_{sweep}_{xlim1}.jpg', dpi = 1000, format = 'jpg')
    
    i = 1
    fig, ax = plt.subplots(figsize = (8,3))
    for i_ind, i in enumerate(SW_spiking_channels):
        ax.plot(scipy.signal.filtfilt(b_notch, a_notch, LFP_all_sweeps[sweep][i,:])[int(xlim1*new_fs):int(xlim2*new_fs)] + i_ind *1000, linewidth = 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        i += 1 
    plt.savefig(f'example_LFP_{sweep}_{xlim1}.pdf', dpi = 1000, format = 'pdf')
    plt.savefig(f'example_LFP_{sweep}_{xlim1}.jpg', dpi = 1000, format = 'jpg')
    
    os.chdir('..')

#%% example LFP filtered and cross correlation
sweep = 2
xlim1 = 105
xlim2 = 110
LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
os.chdir([i for i in os.listdir() if 'analysis' in i][0])
chans_to_plot = list(np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int))
os.chdir('..')

# chans_to_plot.remove(54)
# chans_to_plot.remove(56)

LFP_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep]), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 4*pq.Hz).as_array()
i = 1
fig, ax = plt.subplots(figsize = (5,5))
for i_ind, i in enumerate(chans_to_plot):
    ax.plot(LFP_filt[int(xlim1*new_fs):int(xlim2*new_fs), i] + i_ind *1000 * np.ones_like(int(xlim2*new_fs) - int(xlim1*new_fs)), linewidth = 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    i += 1 
plt.savefig(f'example_LFP_{sweep}_{xlim1}_filtered.pdf', dpi = 1000, format = 'pdf')
plt.savefig(f'example_LFP_{sweep}_{xlim1}_filtered.jpg', dpi = 1000, format = 'jpg')

chan1 = chans_to_plot[4]
chan2 = chans_to_plot[8]

curr_sweep = scipy.signal.correlate(LFP_filt[int(xlim1*new_fs):int(xlim2*new_fs),chan1], LFP_filt[int(xlim1*new_fs):int(xlim2*new_fs),chan2])
curr_sweep_norm = curr_sweep/np.sqrt(np.sum(LFP_filt[int(xlim1*new_fs):int(xlim2*new_fs),chan1]**2)*np.sum(LFP_filt[int(xlim1*new_fs):int(xlim2*new_fs),chan2]**2))

fig, ax = plt.subplots(figsize = (3,3))
ax.plot(np.linspace(-(xlim2 - xlim1)*1000,(xlim2 - xlim1)*1000,curr_sweep_norm.size), curr_sweep_norm, color = 'k')
ax.axvline(0, color = 'r', linestyle = '--', linewidth = 0.5)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
ax.set_yticks([0,1])
ax.set_yticklabels(list(map(str,list(ax.get_yticks()))), size = 12)
ax.set_xticks([-500,0,500])
ax.set_xlim([-750,750])
ax.set_xticklabels(list(map(str,list(ax.get_xticks()))), size = 12)
plt.savefig('example_corr.pdf', dpi = 1000, format = 'pdf')
plt.savefig('example_corr.jpg', dpi = 1000, format = 'jpg')










#%% LFP analysis
# --------------------------------------------------------------------------------- LFP ------------------------------------------------------------------------------------------------------------

# os.chdir(home_directory)
reanalyze = True
plot = False

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i) and 'old' not in i]
# for day in ['121121']:
for day in days:
    os.chdir(day) 
    print(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    # try:
    #     spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    # except FileNotFoundError:
    #     print(f'spikes with highpass {highpass_cutoff} not found')
    #     spikes_allsweeps = pickle.load(open([i for i in os.listdir() if 'spikes_allsweeps' in i][0],'rb'))
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
        to_plot = np.zeros([64, int(0.6*new_fs), len(sweeps_to_plot)])    
        for ind_sweep, sweep in enumerate(sweeps_to_plot):
            curr_to_plot = np.zeros([64, int(0.6*new_fs), len(np.where(stims[sweep] > 0)[0])])
            for ind_stim, stim in enumerate(list(stims[sweep])):
                if stim == 0:
                    continue
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
        # stims_for_LFP[3][10:] = 0    
        # stims_for_LFP[5][25:] = 0
        pickle.dump(stims_for_LFP, open('stims_for_LFP', 'wb'))
    
    
    if os.path.exists(fr'analysis_{day}\LFP_resp_channels.csv') and reanalyze == False:
        os.chdir(f'analysis_{day}')
        LFP_resp_channels = np.loadtxt('LFP_resp_channels.csv', delimiter = ',', dtype = int)
        LFP_resp_channels_automatic = np.loadtxt('LFP_resp_channels_automatic.csv', delimiter = ',', dtype = int)
        LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
        LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
        LFP_slope = np.loadtxt('LFP_slope.csv', delimiter = ',')
        LFP_min_rel = np.loadtxt('LFP_min_rel.csv', delimiter = ',')
        LFP_min_rel_change = np.loadtxt('LFP_min_rel_change.csv', delimiter = ',')
        LFP_slope_rel = np.loadtxt('LFP_slope_rel.csv', delimiter = ',')
        LFP_slope_rel_change = np.loadtxt('LFP_slope_rel_change.csv', delimiter = ',')
        to_plot_1_LFP = np.loadtxt('to_plot_1_LFP.csv', delimiter = ',', dtype = int)
        to_plot_2_LFP = np.loadtxt('to_plot_2_LFP.csv', delimiter = ',', dtype = int)
        # princ_channel = np.load('princ_channel.npy')
        
    else: 
        os.chdir([i for i in os.listdir() if 'analysis' in i][0])
        #select the sweeps to look at   
        to_plot_1_LFP = [0,1,2,3]
        to_plot_2_LFP = list(np.linspace(4,len(LFP_all_sweeps) - 1, len(LFP_all_sweeps) - 4, dtype = int))
        
        # LFP min peaks and std, slope and time to peak per sweep
        LFP_min = np.empty([len(LFP_all_sweeps), 64])
        LFP_min[:] = np.NaN
        LFP_min_rel = np.empty([len(LFP_all_sweeps), 64])
        LFP_min_rel[:] = np.NaN
        LFP_std = np.zeros([len(LFP_all_sweeps), 64])
        LFP_std[:] = np.NaN
        LFP_slope = np.empty([len(LFP_all_sweeps), 64])
        LFP_slope[:] = np.NaN
        time_to_peak = np.empty([len(LFP_all_sweeps), 64])
        time_to_peak[:] = np.NaN
        slope_start = np.empty([len(LFP_all_sweeps), 64])
        slope_start[:] = np.NaN
        slope_end = np.empty([len(LFP_all_sweeps), 64])
        slope_end[:] = np.NaN
        
        for sweep in range(len(LFP_all_sweeps)):
            if LFP_all_sweeps[sweep].size == 0:
                continue
            else:
                curr_LFP_avg = LFP_average([sweep], stims = stims_for_LFP)
                if '160624_B2' in os.getcwd():
                    LFP_min[sweep,:] = np.abs(np.min(curr_LFP_avg[:,200:260], 1) - curr_LFP_avg[:,210])
                elif '221212' in os.getcwd():
                    LFP_min[sweep,:] = np.abs(np.min(curr_LFP_avg[:,200:260], 1) - curr_LFP_avg[:,210])
                elif '221219_1' in os.getcwd():
                    LFP_min[sweep,:] = np.abs(np.min(curr_LFP_avg[:,200:260], 1) - curr_LFP_avg[:,210])
                elif '160128' in os.getcwd():
                    LFP_min[sweep,:] = np.abs(np.min(curr_LFP_avg[:,200:260], 1) - curr_LFP_avg[:,210])
                elif '121121' in os.getcwd(): # some DC drift
                    LFP_min[sweep,:] = np.abs(np.min(curr_LFP_avg[:,200:260], 1) - np.mean(curr_LFP_avg[:,:210], axis = 1))
                else:                     
                    LFP_min[sweep,:] = np.abs(np.min(curr_LFP_avg[:,200:300], 1) - curr_LFP_avg[:,210])
                LFP_std[sweep,:] = np.std(curr_LFP_avg, 1)
                
                
                # slope start: take when it's 1/4 from the max, and end when it's at 3/4 
                for chan in range(64):
                    if scipy.signal.find_peaks(scipy.ndimage.gaussian_filter(-curr_LFP_avg[chan,220:300], 2))[0].__len__() == 0: # no LFP peak
                        continue
                    else:
                        time_to_peak[sweep, chan] = scipy.signal.find_peaks(scipy.ndimage.gaussian_filter(-curr_LFP_avg[chan,220:300], 2))[0][0] + 20
                    if np.argwhere(curr_LFP_avg[chan,200:200+int(time_to_peak[sweep,chan])] > 0.25*np.min(curr_LFP_avg[chan,200+int(time_to_peak[sweep,chan])])).size == 0:
                        continue
                    if np.argwhere(curr_LFP_avg[chan,200:200+int(time_to_peak[sweep,chan])] < 0.75*np.min(curr_LFP_avg[chan,200+int(time_to_peak[sweep,chan])])).size == 0:
                        continue 
                    else:
                        slope_start[sweep,chan] = np.argwhere(curr_LFP_avg[chan,200:200+int(time_to_peak[sweep,chan])] > 0.25*curr_LFP_avg[chan,200+int(time_to_peak[sweep,chan])])[-1]
                        # slope_end[sweep,chan] = np.argmin(curr_LFP_avg[:,200:300], 1)
                        slope_end[sweep,chan] = np.argwhere(curr_LFP_avg[chan,200:200+int(time_to_peak[sweep,chan])] < 0.75*curr_LFP_avg[chan,200+int(time_to_peak[sweep,chan])])[0]
                        LFP_slope[sweep,chan] = (curr_LFP_avg[chan, 200 + int(slope_end[sweep,chan])] - curr_LFP_avg[chan, 200 + int(slope_start[sweep,chan])])/(slope_end[sweep,chan] - slope_start[sweep,chan])
           
                
                # temp_slope = [np.diff(curr_LFP_avg[chan,200:int(time_to_peak[sweep, chan]) + 201]) for chan in range(64)]
                # for chan in range(64):
                #     if temp_slope[chan].size == 0:
                #         continue
                #     else:                  
                #         #last positive slope index before it goes negative to the peak:
                #         if np.argwhere(temp_slope[chan] > 0).size == 0:
                #             continue
                #         else:
                #             slope_start[sweep,chan] = np.argwhere(temp_slope[chan] > 0)[-1] + 1
                #             LFP_slope[sweep,chan] = (np.min(LFP_average([sweep], stims = stims_for_LFP)[chan,200:300]) - LFP_average([sweep], stims = stims_for_LFP)[chan, int(slope_start[sweep,chan])])/(time_to_peak[sweep, chan] - slope_start[sweep,chan])
                
                # # there's gonna be some fucked up chans for slope so take them out
                # LFP_slope_not_outliers = copy.deepcopy(LFP_slope)
                # time_to_peak_not_outliers = copy.deepcopy(time_to_peak)
                # for chan in range(64):
                #     LFP_slope_not_outliers[[np.where(np.logical_or((LFP_slope[:,chan] > (np.percentile(LFP_slope[:,chan], 75) + 1.5*(np.abs(np.percentile(LFP_slope[:,chan], 75) - np.percentile(LFP_slope[:,chan], 25))))), (LFP_slope[:,chan] < (np.percentile(LFP_slope[:,chan], 25) - 1.5*(np.abs(np.percentile(LFP_slope[:,chan], 75) - np.percentile(LFP_slope[:,chan], 25)))))))[0]], chan] = np.NaN
                #     print(f'slope outliers {chan}')
                #     time_to_peak_not_outliers[[np.where(np.logical_or((time_to_peak[:,chan] > (np.percentile(time_to_peak[:,chan], 75) + 1.5*(np.abs(np.percentile(time_to_peak[:,chan], 75) - np.percentile(time_to_peak[:,chan], 25))))), (time_to_peak[:,chan] < (np.percentile(time_to_peak[:,chan], 25) - 1.5*(np.abs(np.percentile(time_to_peak[:,chan], 75) - np.percentile(time_to_peak[:,chan], 25)))))))[0]], chan] = np.NaN
                
                
                
        # relative LFP min (relative to baseline LFP min of every channel)
        LFP_min_rel = LFP_min/np.nanmean(LFP_min[to_plot_1_LFP,:], axis = 0)
        LFP_min_rel_change = np.mean(LFP_min_rel[to_plot_2_LFP,:], axis = 0) - np.mean(LFP_min_rel[to_plot_1_LFP,:], axis = 0)
        LFP_slope_rel = LFP_slope/np.nanmean(LFP_slope[to_plot_1_LFP,:], axis = 0)
        # LFP_slope_rel_not_outliers = LFP_slope_not_outliers/np.nanmean(LFP_slope_not_outliers[to_plot_1_LFP,:], axis = 0)
        LFP_slope_rel_change = np.nanmean(LFP_slope_rel[to_plot_2_LFP,:], axis = 0) - np.nanmean(LFP_slope_rel[to_plot_1_LFP,:], axis = 0)
        # LFP_slope_rel_change_not_outliers = np.nanmean(LFP_slope_rel_not_outliers[to_plot_2_LFP,:], axis = 0) - np.nanmean(LFP_slope_rel_not_outliers[to_plot_1_LFP,:], axis = 0)
        time_to_peak_change = np.nanmedian(time_to_peak[to_plot_2_LFP,:], axis = 0) - np.nanmedian(time_to_peak[to_plot_1_LFP,:], axis = 0)
        
        #select responsive channels within sweeps to plot
        std_cutoff_resp_channels = 2.5
        lfp_cutoff_resp_channels = 200 #-50yv
        LFP_resp_channels_automatic = [chan for chan in range(64) if (LFP_min[to_plot_1_LFP,chan] > LFP_std[to_plot_1_LFP,chan]*std_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > LFP_std[to_plot_2_LFP,chan]*std_cutoff_resp_channels).all()]
        LFP_resp_channels_cutoff =  np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP,chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()])
        LFP_resp_channels = copy.deepcopy(LFP_resp_channels_cutoff)
        
    os.chdir('..')
 
    #save waveform of every channel and sweep
    LFP_responses = np.zeros([len(LFP_all_sweeps), 64, 600])
    LFP_responses[:] = np.NaN
    for sweep in range(len(LFP_all_sweeps)):
        LFP_responses[sweep, :, :] = LFP_average([sweep], stims = stims_for_LFP)




    # fig, ax = plt.subplots(8,8, sharey = True) 
    # fig.suptitle(f'{day} LFP')
    # plot_before = LFP_average(to_plot_1_LFP, stims = stims_for_LFP)
    # plot_after = LFP_average(to_plot_2_LFP, stims = stims_for_LFP)
    # for chan in range(64):                        
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(plot_before[chan,:] - plot_before[chan,200], 'b', linewidth = .5)
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(plot_after[chan,:] - plot_after[chan,200], 'r', linewidth = .5)
    #     if chan in LFP_resp_channels_cutoff:
    #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 6)
    #     # ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].axvline([260], linestyle = '--', linewidth = 0.2)
    #     # ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].axhline([-200], linestyle = '--', linewidth = 0.2)
    # plt.savefig(f'LFP_{to_plot_1_LFP}_vs_{to_plot_2_LFP}', dpi = 1000)
    
    # #change over time in all channels (relative LFP_min)
    # fig, ax = plt.subplots(8,8) 
    # fig.suptitle('LFP timecourse')
    # for ind, ax1 in enumerate(list(ax.flatten())):                        
    #     ax1.plot(LFP_min_rel[:,chanMap[ind]])
    #     if chanMap[ind] in LFP_resp_channels_cutoff:
    #         ax1.set_facecolor("y")
    #     ax1.set_title(str(chanMap[ind]))
    #     ax1.axvline(3.5)
    # plt.savefig(f'LFP min in all chans', dpi = 1000)

    # fig, ax = plt.subplots(8,8) 
    # fig.suptitle('LFP slope timecourse')
    # for ind, ax1 in enumerate(list(ax.flatten())):                        
    #     ax1.plot(LFP_slope_rel[:,chanMap[ind]])
    #     if chanMap[ind] in LFP_resp_channels_cutoff:
    #         ax1.set_facecolor("y")
    #     ax1.set_title(str(chanMap[ind]))
    #     ax1.axvline(3.5)
    # plt.savefig(f'LFP slope in all chans', dpi = 1000)

    if plot:
        # # average responses after for every sweep before pairing, to check
        fig, ax = plt.subplots(8,8, sharey = True) 
        fig.suptitle('before')
        color = cm.rainbow(np.linspace(0, 1, 4))
        for chan in range(64): 
            for i, c in zip(range(4), color):                      
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(LFP_average([i], stims = stims_for_LFP)[chan,:] - LFP_average([i], stims = stims_for_LFP)[chan,200], c = c, label = str(i), linewidth = 0.5)
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 6)
                handles, labels = ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].get_legend_handles_labels()
                # ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].axvline([210], linestyle = '--', linewidth = 0.2)
                # ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].axvline([260], linestyle = '--', linewidth = 0.2)
                # ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].axhline([-200], linestyle = '--', linewidth = 0.2)
                if chan in LFP_resp_channels_cutoff:
                    ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
        fig.legend(handles, labels, loc='upper center')
        plt.savefig(f'LFP min before', dpi = 1000)
    
        # # average responses after for every sweep after pairing, to check
        fig, ax = plt.subplots(8,8, sharey = True) 
        fig.suptitle('after')
        color = cm.rainbow(np.linspace(0, 1, len(LFP_all_sweeps) - 4))
        for chan in range(64): 
            for i, c in zip(range(len(LFP_all_sweeps) - 4), color):                      
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(LFP_average([i + 4], stims = stims_for_LFP)[chan,:] - LFP_average([i + 4], stims = stims_for_LFP)[chan,200], c = c, label = str(i + 4), linewidth = 0.5)
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 6)
                handles, labels = ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')
        plt.savefig(f'LFP min after', dpi = 1000)

        
        # # plot LFP min as a continuous plot for every single trial:
        # stim_cumsum = np.cumsum(np.asarray([len(stim_times[i]) for i in range(len(stim_times))]))
        # stim_cumsum = np.insert(stim_cumsum, 0, 0)
        # all_stims_min = np.zeros([64,sum([len(stim_times[i]) for i in range(len(stim_times))])])
        # for sweep_ind, LFP in enumerate(LFP_all_sweeps):
        #     for stim_ind, stim in enumerate(list(stim_times[sweep_ind])):
        #         all_stims_min[:,stim_cumsum[sweep_ind]+stim_ind] = np.min(LFP[:,int(stim):int(stim+100)], axis = 1) - LFP[:,int(stim+10)]
            
        # fig, ax = plt.subplots(8,8,sharey = True) 
        # fig.suptitle('LFP min in all stims')
        # for ind, ax1 in enumerate(list(ax.flatten())):                        
        #     ax1.plot(all_stims_min[chanMap[ind],:])
        #     if chanMap[ind] in LFP_resp_channels:
        #         ax1.set_facecolor("y")
        #     ax1.set_title(str(chanMap[ind]))
            # ax1.axvline(3.5)
            
        
    
    
    print(f'relative change mean channels = {np.mean(LFP_min_rel_change[LFP_resp_channels_cutoff])}')
    print(f'relative change average signal of all channels = {(np.mean(LFP_min[np.asarray(to_plot_2_LFP)[:,None], LFP_resp_channels_cutoff]) - np.mean(LFP_min[np.asarray(to_plot_1_LFP)[:,None], LFP_resp_channels_cutoff]))/np.mean(LFP_min[[np.asarray(to_plot_1_LFP)[:,None], LFP_resp_channels_cutoff]])}')
    print(f'relative change average slope of all channels = {np.nanmean(LFP_slope_rel_change[LFP_resp_channels_cutoff])}')

    
    
    # ---------------------------------------------------------------------- plots for thesis
    
    
    #slope vs LFP min change
    # fig, ax = plt.subplots()
    # X = LFP_min_rel_change[LFP_resp_channels_cutoff]
    # X_outliers = np.where(np.isnan(X))[0]
    # # X_outliers = np.where(np.logical_or((X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25))))), (X < (np.percentile(X, 25) - 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))))[0]
    # Y = LFP_slope_rel_change[LFP_resp_channels_cutoff]
    # Y_outliers = np.where(np.isnan(Y))[0]
    # if X_outliers.size >0:
    #     outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
    # else:
    #     outliers = Y_outliers
    # X = np.delete(X, outliers)
    # Y = np.delete(Y, outliers)
    # slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
    # print(f'{r} and {p} for {len(X)} channels')
    # ax.scatter(X,Y, color = 'k')
    # ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_xlabel('LFP peak change', size = 18)
    # # ax.set_xticks([0,250,500,750,1000])
    # # ax.set_xticklabels(list(map(str, [0,250,500,750,1000])), size = 18)
    # # ax.set_yticks([24,29,34])
    # # ax.set_yticklabels(list(map(str, [24,29,34])), size = 18)
    # ax.set_ylabel('LFP slope change', size = 18)
    # plt.tight_layout()
    # plt.savefig('distance vs time to peak.svg', dpi = 1000, format = 'svg')
    # plt.savefig('distance vs time to peak.jpg', dpi = 1000, format = 'jpg')


    
    # os.chdir(home_directory)
    #  colorplot of LFP min magnitude
    # fig, ax = plt.subplots()
    # ax.imshow(np.reshape(LFP_min[1,chanMap], (8, 8)), cmap = 'Blues')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.savefig('LFP min colormap.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('LFP min colormap.jpg', dpi = 1000, format = 'jpg')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    
    # before vs after average LFP response on a chanMap 
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',').astype(int)
    os.chdir('..')
    # #overall before vs after average LFP response 
    fig, ax = plt.subplots()
    # fig.suptitle('before vs after')
    if LFP_resp_channels_cutoff.size == 1:
        ax.plot(LFP_average(to_plot_1_LFP, stims = stims_for_LFP)[LFP_resp_channels_cutoff,:] - LFP_average(to_plot_1_LFP, stims = stims_for_LFP)[LFP_resp_channels_cutoff,:][210], 'b')
        ax.plot(LFP_average(to_plot_2_LFP, stims = stims_for_LFP)[LFP_resp_channels_cutoff,:] - LFP_average(to_plot_2_LFP, stims = stims_for_LFP)[LFP_resp_channels_cutoff,:][210], 'r')
    else:
        ax.plot(np.mean(LFP_average(to_plot_1_LFP, stims = stims_for_LFP)[LFP_resp_channels_cutoff,:], axis = 0) - np.mean(LFP_average(to_plot_1_LFP, stims = stims_for_LFP)[LFP_resp_channels_cutoff,:], axis = 0)[210], 'b')
        ax.plot(np.mean(LFP_average(to_plot_2_LFP, stims = stims_for_LFP)[LFP_resp_channels_cutoff,:], axis = 0) - np.mean(LFP_average(to_plot_2_LFP, stims = stims_for_LFP)[LFP_resp_channels_cutoff,:], axis = 0)[210], 'r')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('time from stim (ms)', size = 16)
    ax.set_ylabel('LFP (yV)', size = 16)
    ax.set_xticks([0,200,400])
    ax.set_xticklabels(list(map(str, [-200,0,200])), size = 16)
    ax.set_xlim([0,400])
    ax.set_yticks([-1000,-500,0])
    ax.set_yticklabels(list(map(str, [-1000,-500,0])), size = 16)
    plt.tight_layout()
    plt.savefig('overall LFP before and after.svg', dpi = 1000, format = 'svg')
    plt.savefig('overall LFP before and after.jpg', dpi = 1000, format = 'jpg')
    
    
    # np.savetxt('LFP_resp_channels_cutoff.csv', LFP_resp_channels_cutoff, delimiter = ',')
    plot_before = LFP_average(to_plot_1_LFP, stims = stims_for_LFP)
    plot_after = LFP_average(to_plot_2_LFP, stims = stims_for_LFP)
    fig, ax = plt.subplots(8,8, sharey = True, constrained_layout = True)
    for ind, ax1 in enumerate(list(ax.flatten())):
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        if chanMap[ind] in LFP_resp_channels_cutoff:
            ax1.plot(plot_before[chanMap[ind],100:400] - plot_before[chanMap[ind],200], 'b', linewidth = 1)
            ax1.plot(plot_after[chanMap[ind],100:400] - plot_after[chanMap[ind],200], 'r', linewidth = 1)
    plt.savefig('overall LFP before and after chanMap.svg', dpi = 1000, format = 'svg')
    plt.savefig('overall LFP before and after chanMap.jpg', dpi = 1000, format = 'jpg')

    # # response waveforms, with average in bold
    # chans_to_plot = list(copy.deepcopy(LFP_resp_channels_cutoff))
    # if '160322' in os.getcwd():
    #     chans_to_plot.remove(54)
    #     chans_to_plot.remove(52)
    # fig, ax = plt.subplots()
    # ax.plot(np.transpose(LFP_responses[3,chans_to_plot,:])/10, alpha = 0.3)
    # ax.set_xlim([150,300])
    # ax.plot(np.mean(np.transpose(LFP_responses[3,chans_to_plot,:])/10, axis = 1), linewidth = 2.5, color = 'k')
    # ax.set_xticks([150,200,250,300])
    # ax.set_xticklabels(['-50','0','50','100'], size = 16)
    # ax.set_xlabel('time from stim (msec)', size = 16)
    # ax.set_ylabel('LFP (uV)', size = 16)
    # ax.set_yticks([-300,-200,-100,0])
    # ax.set_yticklabels(list(map(str, [-300,-200,-100,0])), size = 16)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tight_layout()
    # plt.savefig('LFP min all chans.svg', dpi = 1000, format = 'svg')
    # plt.savefig('LFP min all chans.jpg', dpi = 1000, format = 'jpg')
    
    
    # # time to peak vs distance from principal channel
    # # os.chdir(home_directory)
    # exclude_outliers = True
    # os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    # princ_channel = np.load('princ_channel.npy')
    # os.chdir('..')
    # electrode_distances = []
    # for chan in range(64):
    #     electrode_distance_in_indices = np.squeeze(np.argwhere(channelMapArray == princ_channel) - np.argwhere(channelMapArray == chan))
    #     electrode_distances.append((np.sqrt(electrode_distance_in_indices[0]**2 + electrode_distance_in_indices[1]**2))*200)
    
    # # ax.scatter(np.asarray(electrode_distances)[LFP_resp_channels_cutoff], LFP_min[3,LFP_resp_channels_cutoff])
    # X = np.asarray(electrode_distances)[LFP_resp_channels_cutoff]
    # X_outliers = np.where(X > 2000)[0]
    # # X_outliers = np.where(np.logical_or((X > (np.percentile(X, 75) + 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25))))), (X < (np.percentile(X, 25) - 1.5*(np.abs(np.percentile(X, 75) - np.percentile(X, 25)))))))[0]
    # Y = np.argmin(LFP_responses[3,LFP_resp_channels_cutoff,:], axis=1) - 200
    # Y_outliers = np.where(np.logical_or((Y > (np.percentile(Y, 75) + 5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25))))), (Y < (np.percentile(Y, 25) - 10*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))))[0]
    # # Y_outliers = np.where(Y > (np.percentile(Y, 75) + 1.5*(np.abs(np.percentile(Y, 75) - np.percentile(Y, 25)))))[0]
    # if X_outliers.size > 0:
    #     outliers = np.unique(np.concatenate((X_outliers, Y_outliers)))
    # else:
    #     outliers = Y_outliers
    # if exclude_outliers:
    #     X = np.delete(X, outliers)
    #     Y = np.delete(Y, outliers)
        
    # fig, ax = plt.subplots()
    # slope, intercept, r, p, std_err = scipy.stats.linregress(X, Y)
    # print(f'{r} and {p} for {len(X)} channels')
    # ax.scatter(X,Y, color = 'k')
    # ax.plot([np.min(X), np.max(X)], [(slope*np.min(X) + intercept), (slope*np.max(X) + intercept)], color = 'k')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_xlabel('distance from principal channel (ym)', size = 18)
    # ax.set_xticks([0,250,500,750,1000])
    # ax.set_xticklabels(list(map(str, [0,250,500,750,1000])), size = 18)
    # ax.set_yticks([24,29,34])
    # ax.set_yticklabels(list(map(str, [24,29,34])), size = 18)
    # ax.set_ylabel('evoked LFP time to peak (ms)', size = 18)
    # plt.tight_layout()
    # plt.savefig('distance vs time to peak.svg', dpi = 1000, format = 'svg')
    # plt.savefig('distance vs time to peak.jpg', dpi = 1000, format = 'jpg')


    #%
    # # os.chdir(home_directory)
    # os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    # np.savetxt('LFP_resp_channels.csv', LFP_resp_channels, delimiter = ',')
    # np.savetxt('LFP_resp_channels_automatic.csv', LFP_resp_channels_automatic, delimiter = ',')
    # # np.savetxt('LFP_resp_channels_cutoff.csv', LFP_resp_channels_cutoff, delimiter = ',')
    # np.savetxt('LFP_min.csv', LFP_min, delimiter = ',')
    # np.savetxt('LFP_slope.csv', LFP_slope, delimiter = ',')
    # np.savetxt('LFP_slope.csv', LFP_slope, delimiter = ',')
    # np.savetxt('LFP_min_rel.csv', LFP_min_rel, delimiter = ',')
    # np.savetxt('LFP_min_rel_change.csv', LFP_min_rel_change, delimiter = ',')
    # np.savetxt('LFP_slope_rel.csv', LFP_slope_rel, delimiter = ',')
    # np.savetxt('LFP_slope_rel_change.csv', LFP_slope_rel_change, delimiter = ',')
    # np.savetxt('LFP_slope_rel_not_outliers.csv', LFP_slope_rel_not_outliers, delimiter = ',')   
    # np.savetxt('LFP_slope_rel_change_not_outliers.csv', LFP_slope_rel_change_not_outliers, delimiter = ',')    
    # np.savetxt('LFP_time_to_peak.csv', time_to_peak, delimiter = ',')
    # np.savetxt('LFP_time_to_peak_change.csv', time_to_peak_change, delimiter = ',')
    # np.savetxt('to_plot_1_LFP.csv', to_plot_1_LFP, delimiter = ',')
    # np.savetxt('to_plot_2_LFP.csv', to_plot_2_LFP, delimiter = ',')
    # np.save('LFP_responses.npy', LFP_responses)
    # np.save('princ_channel.npy', princ_channel)
    # os.chdir('..')
    
    os.chdir('..')
    
    # cl()
    
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

# princ_channel = 63
# np.save('princ_channel.npy', princ_channel)




#%% LFP response examples

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i) and 'old' not in i]
for day in ['160331']:
# for day in days:
    os.chdir(day) 
    print(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    # try:
    #     spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    # except FileNotFoundError:
    #     print(f'spikes with highpass {highpass_cutoff} not found')
    #     spikes_allsweeps = pickle.load(open([i for i in os.listdir() if 'spikes_allsweeps' in i][0],'rb'))
    stim_times = pickle.load(open('stim_times','rb'))

    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_resp_channels = np.loadtxt('LFP_resp_channels.csv', delimiter = ',', dtype = int)
    LFP_resp_channels_automatic = np.loadtxt('LFP_resp_channels_automatic.csv', delimiter = ',', dtype = int)
    LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_responses = np.load('LFP_responses.npy')
    os.chdir('..')
    
    
    # os.chdir(home_directory)
    #  colorplot of LFP min magnitude
    fig, ax = plt.subplots()
    plot = ax.imshow(np.reshape(LFP_min[1,chanMap], (8, 8)), cmap = 'Blues')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('LFP min colormap.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('LFP min colormap.jpg', dpi = 1000, format = 'jpg')
    # plt.colorbar(plot)
    
    fig, ax = plt.subplots(figsize = (0.1,5))
    cmap = cm.Blues
    norm = colors.Normalize(vmin=0, vmax=1)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=ax, ticks = [0,0.33,0.66,1])
    ax.set_yticklabels(list(map(str, np.linspace(-0, -3, 4).astype(int))), size = 18)
    ax.set_ylabel('LFP response peak (mVolt)', size = 16)
    plt.tight_layout()
    plt.savefig('LFP peak colormap legend.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('LFP peak colormap legend.jpg', dpi = 1000, format = 'jpg')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    
    
    # # response waveforms, with average in bold
    chans_to_plot = list(copy.deepcopy(LFP_resp_channels_cutoff))
    # if '160322' in os.getcwd():
    #     chans_to_plot.remove(54)
    #     chans_to_plot.remove(52)
    fig, ax = plt.subplots()
    ax.plot(np.transpose(LFP_responses[3,chans_to_plot,:]), alpha = 0.3)
    ax.set_xlim([150,300])
    ax.plot(np.mean(np.transpose(LFP_responses[3,chans_to_plot,:]), axis = 1), linewidth = 2.5, color = 'k')
    ax.set_xticks([150,200,250,300])
    ax.set_xticklabels(['-50','0','50','100'], size = 18)
    ax.set_xlabel('time from stim (msec)', size = 18)
    ax.set_ylabel('LFP (mV)', size = 18)
    ax.set_yticks([-3000,-2000,-1000,0])
    ax.set_yticklabels(list(map(str, [-3,-2,-1,0])), size = 18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('LFP min all chans.svg', dpi = 1000, format = 'svg')
    plt.savefig('LFP min all chans.jpg', dpi = 1000, format = 'jpg')




#%% PSTH
highpass_cutoff = 4
smooth_over = 10

reanalyze = True
plot = True

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i) and 'old' not in i]
# for day in ['221220_2']:
for day in days:
    os.chdir(day) 
    print(day)
    if day in ['160310', '160414_D1', '160426_D1', '160519_B2', '160624_B2', '160628_D1', '160128', '160202', '160218', '160308', '160322', '160331', '160420', '160427']:        
        artifacts = []
    else:
        artifacts = []
        # artifacts = list(np.linspace(97,123,27,dtype = int))                 
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    
    try:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    except FileNotFoundError:
        print(f'spikes with highpass {highpass_cutoff} not found')
        spikes_allsweeps = pickle.load(open([i for i in os.listdir() if 'spikes_allsweeps' in i][0],'rb'))
    
    stim_times = pickle.load(open('stim_times','rb'))
    
    
    # # if you want to take out certain stims, you have to take them out of stim_times array.
    def PSTH_matrix(sweeps_to_plot, take_out_artifacts = True, artifact_locs = [], stims = stim_times):
        to_plot = np.zeros([299,64,len(sweeps_to_plot)])
        for ind_sweep, sweep in enumerate(sweeps_to_plot):
            #PSTH_matrix is mean across trials in one sweep
            PSTH_matrix = np.zeros([299,64])
            bins = np.linspace(1,300,300)
            for ind_chan, chan in enumerate(natsort.natsorted(list(spikes_allsweeps[sweep].keys()))):
                # print(chan)
                currchan = np.zeros([299,len(stims[sweep])])
                for ind_stim, j in enumerate(list(stims[sweep])):
                    currchan[:,ind_stim] = np.histogram((spikes_allsweeps[sweep][chan][(j - 0.1*new_fs < spikes_allsweeps[sweep][chan]) & (spikes_allsweeps[sweep][chan] < j+0.2*new_fs)] - (j-0.1*new_fs)), bins)[0]
                    if take_out_artifacts:
                        currchan[:,ind_stim][artifact_locs] = 0
                PSTH_matrix[:,ind_chan] = np.squeeze(np.mean(currchan, 1)) # mean across stims for every channel
            to_plot[:,:,ind_sweep] = PSTH_matrix
        return np.squeeze(np.mean(to_plot,2))

    def SD_spikes_before(sweeps_to_plot, stims = stim_times, time_before = 100, time_after = 0):
        SD_matrix = np.zeros([64,len(sweeps_to_plot)])
        for ind_sweep, sweep in enumerate(sweeps_to_plot):
            bins = np.linspace(1,time_before,time_before)
            for ind_chan, chan in enumerate(natsort.natsorted(list(spikes_allsweeps[sweep].keys()))):
                # print(chan)
                currchan = np.zeros([time_before-1,len(stims[sweep])])
                for ind_stim, j in enumerate(list(stims[sweep])):
                    currchan[:,ind_stim] = np.histogram((spikes_allsweeps[sweep][chan][(j - time_before < spikes_allsweeps[sweep][chan]) & (spikes_allsweeps[sweep][chan] < j+time_after)] - (j-time_before)), bins)[0]
                SD_matrix[ind_chan,ind_sweep] = np.std(np.sum(currchan, 0)/(time_before-1)) # std of the spike sum across stims for every channel
        return SD_matrix

    if os.path.exists(fr'analysis_{day}\PSTH_resp_channels.csv') and reanalyze == False:
        os.chdir([i for i in os.listdir() if 'analysis' in i][0])
        PSTH_resp_channels = np.loadtxt('PSTH_resp_channels.csv', delimiter = ',', dtype = int)
        PSTH_resp_magn = np.loadtxt('PSTH_resp_magn.csv', delimiter = ',')
        PSTH_resp_peak = np.loadtxt('PSTH_resp_peak.csv', delimiter = ',')
        PSTH_resp_magn_rel = np.loadtxt('PSTH_resp_magn_rel.csv', delimiter = ',')
        PSTH_resp_peak_rel = np.loadtxt('PSTH_resp_peak_rel.csv', delimiter = ',')
        PSTH_resp_magn_rel_change = np.loadtxt('PSTH_resp_magn_rel_change.csv', delimiter = ',')
        PSTH_resp_peak_rel_change = np.loadtxt('PSTH_resp_peak_rel_change.csv', delimiter = ',')
        to_plot_1_PSTH = np.loadtxt('to_plot_1_PSTH.csv', delimiter = ',', dtype = int)
        to_plot_2_PSTH = np.loadtxt('to_plot_2_PSTH.csv', delimiter = ',', dtype  = int)
    
    else: 
        to_plot_1_PSTH = [0,1,2,3]
        to_plot_2_PSTH = list(np.linspace(4,len(LFP_all_sweeps) - 1, len(LFP_all_sweeps) - 4, dtype = int))
        
        PSTH_resp_magn = np.empty([len(LFP_all_sweeps), 64])
        PSTH_resp_magn[:] = np.NaN
        PSTH_resp_magn_rel = np.empty([len(LFP_all_sweeps), 64])
        PSTH_resp_magn_rel[:] = np.NaN
        PSTH_resp_peak = np.empty([len(LFP_all_sweeps), 64])
        PSTH_resp_peak[:] = np.NaN
        PSTH_resp_peak_rel = np.empty([len(LFP_all_sweeps), 64])
        PSTH_resp_peak_rel[:] = np.NaN
        PSTH_responses = np.zeros([len(LFP_all_sweeps), 299, 64])
        PSTH_responses[:] = np.NaN
    
        for sweep in range(len(LFP_all_sweeps)):
            PSTH_responses[sweep, :, :] = PSTH_matrix([sweep], artifact_locs = artifacts)
            
            if day == '121121':
                PSTH_resp_magn[sweep,:] = np.sum(PSTH_responses[sweep,110:180,:], axis = 0)
            if day == '160424_B2':
                PSTH_resp_magn[sweep,:] = np.sum(PSTH_responses[sweep,110:180,:], axis = 0)
            else:
                PSTH_resp_magn[sweep,:] = np.sum(PSTH_responses[sweep,110:180,:], axis = 0)
            
            for chan in range(64):
                PSTH_resp_peak[sweep,chan] = np.max(smooth(PSTH_responses[sweep,110:180,chan], smooth_over), axis = 0) # *1000 for instantaneous spike rate
        PSTH_resp_magn_rel = PSTH_resp_magn/np.nanmean(PSTH_resp_magn[to_plot_1_PSTH,:], axis = 0)
        PSTH_resp_peak_rel = PSTH_resp_peak/np.nanmean(PSTH_resp_peak[to_plot_1_PSTH,:], axis = 0)
        PSTH_resp_magn_rel_change = np.nanmean(PSTH_resp_magn_rel[to_plot_2_PSTH,:], axis = 0) - np.nanmean(PSTH_resp_magn_rel[to_plot_1_PSTH,:], axis = 0)
        PSTH_resp_peak_rel_change = np.nanmean(PSTH_resp_peak_rel[to_plot_2_PSTH,:], axis = 0) - np.nanmean(PSTH_resp_peak_rel[to_plot_1_PSTH,:], axis = 0)
        
        # PSTH resp channels: above 99.9% of the confidence interval of spike rate 100ms before stim within each sweep (mean + approx. 3.5*SEM) AND instantaneous spike rate peak of at least 10Hz
        SD_before = SD_spikes_before([0,1,2,3,4,5,6,7,8,9])
        PSTH_resp_channels = [chan for chan in range(64) if all([PSTH_resp_peak[j,chan] > (np.sum(PSTH_responses[j,0:100,chan])/99 + 3.5*SD_before[chan,j]/np.sqrt(60)) and PSTH_resp_peak[j,chan]*1000 > 10 for j in range(10)])]
        
    if plot:
        fig, ax = plt.subplots(8,8, sharey = True)
        fig.suptitle(f'before vs after {day}')
        plot_before = PSTH_matrix(to_plot_1_PSTH, artifact_locs = artifacts)*1000 # *1000 for instantaneous spike rate
        plot_after = PSTH_matrix(to_plot_2_PSTH, artifact_locs = artifacts)*1000
        for chan in range(64):
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(plot_before[:,chan], smooth_over), 'b', linewidth = .5)
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(plot_after[:,chan], smooth_over),'r', linewidth = .5)
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 4)
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_xlim([0,200])
            if chan in PSTH_resp_channels:
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
        # plt.tight_layout()
        plt.savefig(f'Spiking_{to_plot_1_PSTH}_vs_{to_plot_2_PSTH}', dpi = 1000)
        
        fig, ax = plt.subplots(8,8) 
        fig.suptitle('PSTH timecourse peak')
        for ind, ax1 in enumerate(list(ax.flatten())):                        
            ax1.plot(PSTH_resp_peak_rel[:,chanMap[ind]])
            if chanMap[ind] in PSTH_resp_channels:
                ax1.set_facecolor("y")
            ax1.set_title(str(chanMap[ind]))
            ax1.axvline(3.5)
        plt.savefig(f'Spiking peak all chans', dpi = 1000)

        fig, ax = plt.subplots(8,8) 
        fig.suptitle('PSTH timecourse magn')
        for ind, ax1 in enumerate(list(ax.flatten())):                        
            ax1.plot(PSTH_resp_magn_rel[:,chanMap[ind]])
            if chanMap[ind] in PSTH_resp_channels:
                ax1.set_facecolor("y")
            ax1.set_title(str(chanMap[ind]))
            ax1.axvline(3.5)
        plt.savefig(f'Spiking all chans magn', dpi = 1000)

        # # #check individual sweeps before
        fig, ax = plt.subplots(8,8, sharey = True)
        fig.suptitle('before')
        plot_1 = PSTH_matrix([0], artifact_locs = artifacts)*1000
        plot_2 = PSTH_matrix([1], artifact_locs = artifacts)*1000
        plot_3 = PSTH_matrix([2], artifact_locs = artifacts)*1000
        plot_4 = PSTH_matrix([3], artifact_locs = artifacts)*1000
        for chan in range(64):
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(plot_1[:,chan], smooth_over), 'b', label = '1', linewidth = .5)
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(plot_2[:,chan], smooth_over), 'r', label = '2', linewidth = .5)
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(plot_3[:,chan], smooth_over), 'k', label = '3', linewidth = .5)
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(plot_4[:,chan], smooth_over), 'c', label = '4', linewidth = .5)    
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 4)
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_xlim([0,200])
            if chan in PSTH_resp_channels:
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("yellow")
        plt.legend()   
        plt.savefig(f'Spiking before', dpi = 1000)

        
        # # #check individual sweeps after
        fig, ax = plt.subplots(8,8, sharey = True)
        fig.suptitle('after')
        # color = cm.rainbow(np.linspace(0, 1, len(LFP_all_sweeps) - 4))
        plot_1 = PSTH_matrix([4], artifact_locs = artifacts)*1000
        plot_2 = PSTH_matrix([5], artifact_locs = artifacts)*1000
        plot_3 = PSTH_matrix([6], artifact_locs = artifacts)*1000
        plot_4 = PSTH_matrix([7], artifact_locs = artifacts)*1000
        plot_5 = PSTH_matrix([8], artifact_locs = artifacts)*1000
        plot_6 = PSTH_matrix([9], artifact_locs = artifacts)*1000
        for chan in range(64):
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(plot_1[:,chan], smooth_over), 'b', label = '5', linewidth = .5)
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(plot_2[:,chan], smooth_over), 'r', label = '6', linewidth = .5)
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(plot_3[:,chan], smooth_over), 'c', label = '7', linewidth = .5)
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(plot_4[:,chan], smooth_over), 'k', label = '8', linewidth = .5)   
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(plot_5[:,chan], smooth_over), 'g', label = '9', linewidth = .5)
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(plot_6[:,chan], smooth_over), 'm', label = '10', linewidth = .5)  
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_xlim([0,200])
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 4)
            if chan in PSTH_resp_channels:
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("yellow")
        # fig.patch.set_alpha(0.1)
        plt.legend()  
        plt.savefig(f'Spiking after', dpi = 1000)

        
        # # relative change over time in responsive channels average
        # fig, ax = plt.subplots()
        # fig.suptitle('response magnitude (number of spikes)')
        # ax.plot(np.nanmean(PSTH_resp_magn_rel[:,PSTH_resp_channels], axis = 1))
        
        # fig, ax = plt.subplots()
        # fig.suptitle('response peak')
        # ax.plot(np.nanmean(PSTH_resp_peak_rel[:,PSTH_resp_channels], axis = 1))
        
        # #average before and after of all responsive channels
        # fig, ax = plt.subplots()
        # fig.suptitle(f'before vs after {day}')
        # # fig.suptitle('before vs after')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.plot(smooth(np.mean(PSTH_matrix(to_plot_1_PSTH, artifact_locs = artifacts)[:,PSTH_resp_channels], axis = 1)*1000, smooth_over), 'b')
        # ax.plot(smooth(np.mean(PSTH_matrix(to_plot_2_PSTH, artifact_locs = artifacts)[:,PSTH_resp_channels], axis = 1)*1000, smooth_over), 'r')
        # ax.set_xlim([50,200])
        # ax.set_xticks([50,100,150,200])
        # ax.set_xticklabels(['-50','0','50','100'], size = 16)
        # ax.set_xlabel('time from stim (msec)', size = 16)
        # ax.set_ylabel('spike rate (Hz)', size = 16)
        # ax.set_yticks([0,100,200,300])
        # ax.set_yticklabels(list(map(str, [0,100,200,300])), size = 16)
        # plt.tight_layout()
        # plt.savefig('overall PSTH before and after.pdf', dpi = 1000, format = 'pdf')
        # plt.savefig('overall PSTH before and after.jpg', dpi = 1000, format = 'jpg')

        
    
    print(f'PSTH magn change rel average: {np.nanmean(np.nanmean(PSTH_resp_magn_rel[np.asarray(to_plot_2_PSTH)[:,None],PSTH_resp_channels] - 1, axis = 0), axis = 0)}')
    print(f'PSTH peak change rel average: {np.nanmean(np.nanmean(PSTH_resp_peak_rel[np.asarray(to_plot_2_PSTH)[:,None],PSTH_resp_channels] - 1, axis = 0), axis = 0)}')
    
    
    print(f'relative change average signal of all channels MAGN = {(np.mean(PSTH_resp_magn[np.asarray(to_plot_2_PSTH)[:,None], PSTH_resp_channels]) - np.mean(PSTH_resp_magn[np.asarray(to_plot_1_PSTH)[:,None], PSTH_resp_channels]))/np.mean(PSTH_resp_magn[[np.asarray(to_plot_1_PSTH)[:,None], PSTH_resp_channels]])}')
    print(f'relative change average signal of all channels PEAK = {(np.mean(PSTH_resp_peak[np.asarray(to_plot_2_PSTH)[:,None], PSTH_resp_channels]) - np.mean(PSTH_resp_peak[np.asarray(to_plot_1_PSTH)[:,None], PSTH_resp_channels]))/np.mean(PSTH_resp_peak[[np.asarray(to_plot_1_PSTH)[:,None], PSTH_resp_channels]])}')
    
    
    # example colorplot of PSTH max magnitude
    fig, ax = plt.subplots()
    ax.imshow(np.reshape(PSTH_resp_magn[4,chanMap], (8, 8)), cmap = 'Blues')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('PSTH max colormap.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('PSTH max colormap.jpg', dpi = 1000, format = 'jpg')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # chans_to_plot = [47,1,45,3,51,27,43,5,53,25,41,7,55,39,9,57,21,11]
    # chans_to_plot = LFP_resp_channels
    # response waveforms, with average in bold
    # fig, ax = plt.subplots()
    # for chan in chans_to_plot:
    #     ax.plot(smooth(np.transpose(np.mean(PSTH_responses[0:9,:,:], axis = 0)[chan,:]), 6)*1000, alpha = 0.3)
    #     ax.plot(smooth(np.mean(np.transpose(np.mean(PSTH_responses[0:9,:,:], axis = 0)[chans_to_plot,:])*1000, axis = 1), 6), linewidth = 2.5, color = 'k')
    # ax.set_xlim([50,200])
    # ax.set_xticks([50,100,150,200])
    # ax.set_xticklabels(['-50','0','50','100'], size = 16)
    # ax.set_xlabel('time from stim (msec)', size = 16)
    # ax.set_ylabel('spike rate (Hz)', size = 16)
    # ax.set_yticks([0,100,200,300,400])
    # ax.set_yticklabels(list(map(str, [0,100,200,300,400])), size = 16)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tight_layout()
    # plt.savefig('PSTH max all chans.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('PSTH max all chans.jpg', dpi = 1000, format = 'jpg')



    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    np.savetxt('PSTH_resp_channels.csv', PSTH_resp_channels, delimiter = ',')
    np.savetxt('PSTH_resp_magn.csv', PSTH_resp_magn, delimiter = ',')
    np.savetxt('PSTH_resp_peak.csv', PSTH_resp_peak, delimiter = ',')
    np.savetxt('PSTH_resp_magn_rel.csv', PSTH_resp_magn_rel, delimiter = ',')
    np.savetxt('PSTH_resp_peak_rel.csv', PSTH_resp_peak_rel, delimiter = ',')
    np.savetxt('PSTH_resp_magn_rel_change.csv', PSTH_resp_magn_rel_change, delimiter = ',')
    np.savetxt('PSTH_resp_peak_rel_change.csv', PSTH_resp_peak_rel_change, delimiter = ',')
    np.savetxt('to_plot_1_PSTH.csv', to_plot_1_PSTH, delimiter = ',')
    np.savetxt('to_plot_2_PSTH.csv', to_plot_2_PSTH, delimiter = ',')
    np.savetxt('PSTH_artifacts.csv', np.asarray(artifacts))
    np.save('PSTH_responses.npy', PSTH_responses)
    
    os.chdir('..')
    os.chdir('..')
    

#%% spiking example figures
highpass_cutoff = 4
smooth_over = 10

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i) and 'old' not in i]
# for day in ['221220_2']:
for day in days:
    os.chdir(day) 
    print(day)
    if day in ['160310', '160414_D1', '160426_D1', '160519_B2', '160624_B2', '160628_D1', '160128', '160202', '160218', '160308', '160322', '160331', '160420', '160427']:        
        artifacts = []
    else:
        artifacts = []
        # artifacts = list(np.linspace(97,123,27,dtype = int))                 

    try:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    except FileNotFoundError:
        print(f'spikes with highpass {highpass_cutoff} not found')
        spikes_allsweeps = pickle.load(open([i for i in os.listdir() if 'spikes_allsweeps' in i][0],'rb'))

    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    PSTH_resp_channels = np.loadtxt('PSTH_resp_channels.csv', delimiter = ',', dtype = int)
    PSTH_resp_channels_cutoff = np.loadtxt('PSTH_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
    PSTH_resp_magn = np.loadtxt('PSTH_resp_magn.csv', delimiter = ',')
    PSTH_resp_peak = np.loadtxt('PSTH_resp_peak.csv', delimiter = ',')
    PSTH_responses = np.load('PSTH_responses.npy')
    os.chdir('..')
    
    # example colorplot of PSTH max magnitude
    fig, ax = plt.subplots()
    ax.imshow(np.reshape(PSTH_resp_peak[4,chanMap], (8, 8))*1000, cmap = 'Blues')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('PSTH max colormap.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('PSTH max colormap.jpg', dpi = 1000, format = 'jpg')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    fig, ax = plt.subplots(figsize = (1,5))
    cmap = cm.Blues
    norm = colors.Normalize(vmin=0, vmax=1)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=ax, ticks = [0,0.33,0.66,1])
    ax.set_yticklabels(list(map(str, np.linspace(0, 600, 4).astype(int))), size = 18)
    ax.set_ylabel('MUA response peak (Hz)', size = 16)
    plt.tight_layout()
    plt.savefig('PSTH peak colormap legend.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('PSTH peak colormap legend.jpg', dpi = 1000, format = 'jpg')


    # chans_to_plot = [47,1,45,3,51,27,43,5,53,25,41,7,55,39,9,57,21,11]
    # chans_to_plot = LFP_resp_channels
    # response waveforms, with average in bold
    fig, ax = plt.subplots()
    for chan in PSTH_resp_channels_cutoff:
        ax.plot((np.mean(PSTH_responses, axis = 0)[:,chan]*1000), alpha = 0.3)
        ax.plot(np.mean(np.mean(PSTH_responses, axis = 0)[:, PSTH_resp_channels_cutoff]*1000, axis = 1), linewidth = 2.5, color = 'k')
    ax.set_xlim([50,200])
    ax.set_xticks([50,100,150,200])
    ax.set_xticklabels(['-50','0','50','100'], size = 16)
    ax.set_xlabel('time from stim (msec)', size = 16)
    ax.set_ylabel('spike rate (Hz)', size = 16)
    ax.set_yticks([0,200,400,600])
    ax.set_yticklabels(list(map(str, ax.get_yticks())), size = 16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('PSTH max all chans.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('PSTH max all chans.jpg', dpi = 1000, format = 'jpg')




#%% go through stims individually in responsive channels to exclude them (never used)

sweep = 0
stim_ind = 0

LFP_max_chan = np.argmax(LFP_min[sweep,:])
LFP_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep]), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 4*pq.Hz).as_array()
fig, ax = plt.subplots(int(np.ceil(np.sqrt(len(LFP_resp_channels)))), int(np.ceil(np.sqrt(len(LFP_resp_channels)))), sharey = True)
for ind, ax1 in enumerate(list(ax.flatten())):    
    try:
        ax1.plot(LFP_filt[int(stim_times[sweep][stim_ind]):int(stim_times[sweep][stim_ind] + 4.9*new_fs), LFP_resp_channels[ind]])
        ax1.set_title(f'{stim_ind}: {LFP_resp_channels[ind]}')
        ax1.set_ylim([-2*LFP_min[sweep, LFP_max_chan], 2*LFP_min[sweep, LFP_max_chan]])
    except IndexError:
        continue

              
class event_handling:
    stim_ind = 0
    def Next(self, event):
        self.stim_ind += 1 #increases by the number of cells in the plot
        for ind, ax1 in enumerate(list(ax.flatten())):    
            try:
                ax1.clear()
                ax1.plot(LFP_filt[int(stim_times[sweep][self.stim_ind]):int(stim_times[sweep][self.stim_ind] + 4.9*new_fs), LFP_resp_channels[ind]])
                ax1.set_title(f'{self.stim_ind}: {LFP_resp_channels[ind]}')
                ax1.set_ylim([-2*LFP_min[sweep, LFP_max_chan], 2*LFP_min[sweep, LFP_max_chan]])
            except IndexError:
                continue
        plt.draw()

    def prev(self, event):
        self.stim_ind -= 1 #increases by the number of cells in the plot
        for ind, ax1 in enumerate(ax.flatten()): 
            try:
                ax1.clear()
                ax1.plot(LFP_filt[int(stim_times[sweep][self.stim_ind]):int(stim_times[sweep][self.stim_ind] + 4.9*new_fs), LFP_resp_channels[ind]])
                ax1.set_title(f'{self.stim_ind}: {LFP_resp_channels[ind]}')
                ax1.set_ylim([-2*LFP_min[sweep, LFP_max_chan], 2*LFP_min[sweep, LFP_max_chan]])
            except IndexError:
                continue
        plt.draw()


callback = event_handling()
axprev = plt.axes([0.7, 0.01, 0.075, 0.075])
axnext = plt.axes([0.81, 0.01, 0.075, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.Next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)
axprev._button = bnext #create dummy reference (don't quite get why but I think it needs an explicit reference as an attribute of the plt.axes because the variable bnext is gone after function is called). Putting globals()['bnext'] = Button(axnext, 'Next') works too
axnext._button = bprev



#%% DELTA POWER
# --------------------------------------------------------------------------------- delta ----------------------------------------------------------------------------------------------------
# Play with time window and Hanning window!
# os.chdir(home_directory)

reanalyze = True

plot = True 
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i) and 'old' not in i]
# for day in ['160310']:
for day in days:
    os.chdir(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    # try:
    #     spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    # except FileNotFoundError:
    #     print(f'spikes with highpass {highpass_cutoff} not found')
    #     spikes_allsweeps = pickle.load(open('spikes_allsweeps','rb'))
    stim_times = pickle.load(open('stim_times','rb'))
    
    to_plot_1_LFP = [0,1,2,3]
    to_plot_2_LFP = list(np.linspace(4,len(LFP_all_sweeps) - 1, len(LFP_all_sweeps) - 4, dtype = int))

    # if os.path.exists('stims_for_delta'):
    #     stims_for_delta = pickle.load(open('stims_for_delta','rb'))
    # else:
    stims_for_delta = copy.deepcopy(stim_times)
    if '131121' in os.getcwd():
        stims_for_delta[4][9] = 0    
        stims_for_delta[8][41] = 0    
        stims_for_delta[8][55] = 0
    if '160218' in os.getcwd():
        stims_for_delta[2][39] = 0    
    if '160322' in os.getcwd():
        stims_for_delta[0][10] = 0    
        stims_for_delta[2][12] = 0    
    if '201121' in os.getcwd():
        stims_for_delta[9][2] = 0
        stims_for_delta[9][3] = 0
        stims_for_delta[5][15] = 0
    if '221213' in os.getcwd():
        stims_for_delta[4][6] = 0    
    if '221218' in os.getcwd():
        stims_for_delta[0][49] = 0
        stims_for_delta[0][51] = 0
        stims_for_delta[2][37] = 0
        stims_for_delta[3][33] = 0                
        stims_for_delta[4][51] = 0    
    if '221219_1' in os.getcwd():
        stims_for_delta[1][32] = 0
        stims_for_delta[3][30] = 0
        stims_for_delta[3][29] = 0   
        stims_for_delta[3][1] = 0   
        stims_for_delta[5][53] = 0   
    # if '160128' in os.getcwd():
    #     stims_for_delta[3][12] = 0    
    #     stims_for_delta[3][15] = 0    
    #     stims_for_delta[3][19] = 0    
    #     stims_for_delta[3][25] = 0    
    #     stims_for_delta[3][39] = 0
    #     stims_for_delta[6][1] = 0
    #     stims_for_delta[6][2] = 0
    #     stims_for_delta[7][1] = 0
    #     stims_for_delta[8][1] = 0
    # if '160420' in os.getcwd():
        
        
    if '121121' in os.getcwd():
        stims_for_delta[4][2] = 0    
        stims_for_delta[4][32] = 0    
        stims_for_delta[4][56] = 0    
        stims_for_delta[5][49] = 0    
        stims_for_delta[5][65] = 0    
        stims_for_delta[6][48] = 0    
        stims_for_delta[7][3] = 0    
        stims_for_delta[9][34] = 0
    if '160310' in os.getcwd():
        stims_for_delta[4][6] = 0      
        stims_for_delta[4][12] = 0      
        stims_for_delta[4][51] = 0   
        stims_for_delta[6][5] = 0      
        stims_for_delta[6][23] = 0      
        stims_for_delta[6][22] = 0      
        stims_for_delta[6][25] = 0      
        stims_for_delta[6][37] = 0     
        stims_for_delta[6][41] = 0      
        stims_for_delta[6][45] = 0      
        stims_for_delta[6][50:52] = 0      
        stims_for_delta[7][3] = 0      
        stims_for_delta[8][34] = 0      
    if '160414_D1' in os.getcwd():
        stims_for_delta[2][2] = 0      
        stims_for_delta[5][52] = 0      
        stims_for_delta[7][1:6] = 0      
    if '160426_D1' in os.getcwd():
        stims_for_delta[4][2] = 0    
        stims_for_delta[4][3] = 0    
        stims_for_delta[6][53] = 0  
        stims_for_delta[7][5] = 0  
        stims_for_delta[7][30] = 0  
        stims_for_delta[8][3] = 0    
        stims_for_delta[8][50] = 0    
        stims_for_delta[9][12] = 0    
        stims_for_delta[9][53] = 0    
    if '160519_B2' in os.getcwd():
        stims_for_delta[8][34] = 0    
        stims_for_delta[3][31] = 0    
        stims_for_delta[2][44] = 0    
        stims_for_delta[4][54] = 0    
        stims_for_delta[5][3] = 0
        stims_for_delta[6][57] = 0    
        stims_for_delta[7][2] = 0    
    if '160624_B2' in os.getcwd():
        stims_for_delta[0][40] = 0    
        stims_for_delta[1][40] = 0 
        stims_for_delta[4][51] = 0    
        stims_for_delta[5][4] = 0    
        stims_for_delta[5][14] = 0    
        stims_for_delta[6][51] = 0    
        stims_for_delta[7][4] = 0    
        stims_for_delta[7][42] = 0    
        stims_for_delta[7][56] = 0    
        stims_for_delta[8][2] = 0    
        stims_for_delta[8][15] = 0    
        stims_for_delta[8][40] = 0    
        stims_for_delta[8][45] = 0    
        stims_for_delta[8][47] = 0    
        stims_for_delta[9][5] = 0    
        stims_for_delta[9][14] = 0    
        stims_for_delta[9][26] = 0    
        stims_for_delta[9][50] = 0    
    if '160628_D1' in os.getcwd():
        stims_for_delta[0][10] = 0    
        stims_for_delta[1][14] = 0    
        stims_for_delta[4][47] = 0    
        stims_for_delta[4][51:58] = 0  
        stims_for_delta[5][4] = 0
        stims_for_delta[5][18] = 0
        stims_for_delta[5][41] = 0
        stims_for_delta[6][41] = 0    
        stims_for_delta[6][44:46] = 0    
        stims_for_delta[6][29] = 0    
        stims_for_delta[6][25:27] = 0    
        stims_for_delta[6][33] = 0    
        stims_for_delta[6][39] = 0    
        stims_for_delta[7][27] = 0    
        stims_for_delta[7][47] = 0   
        stims_for_delta[8][22] = 0 
        stims_for_delta[8][57] = 0 
        stims_for_delta[8][15:17] = 0    
        stims_for_delta[8][19:22] = 0    
        stims_for_delta[9][22:24] = 0    
        stims_for_delta[9][34] = 0    
        stims_for_delta[9][38] = 0    
    if '191121' in os.getcwd():
        stims_for_delta[4][16:18] = 0    
        stims_for_delta[3][6] = 0
        stims_for_delta[5][38] = 0    
        stims_for_delta[7][6] = 0 
        stims_for_delta[7][15] = 0
        stims_for_delta[7][41] = 0    
        stims_for_delta[9][31] = 0    
    if '201121' in os.getcwd():
        stims_for_delta[1][28] = 0    
        stims_for_delta[2][22] = 0    
        stims_for_delta[3][1] = 0    
        stims_for_delta[3][19] = 0    
        stims_for_delta[3][59] = 0    
        stims_for_delta[7][12] = 0    
        stims_for_delta[7][30] = 0    
        stims_for_delta[8][22] = 0    
    if '221220_3' in os.getcwd():
        stims_for_delta[0][59] = 0    
        stims_for_delta[3][16] = 0    
        stims_for_delta[3][21] = 0    
        stims_for_delta[4][24] = 0    
        stims_for_delta[5][9:11] = 0    
        stims_for_delta[5][48] = 0    
        stims_for_delta[6][18:20] = 0    
        stims_for_delta[6][31] = 0    
        stims_for_delta[6][34] = 0    
        stims_for_delta[6][35] = 0    
        stims_for_delta[6][68] = 0    
        stims_for_delta[7][14] = 0    
        stims_for_delta[7][22] = 0    
        stims_for_delta[7][37:39] = 0    
        stims_for_delta[7][56] = 0    
        stims_for_delta[7][59] = 0    
        stims_for_delta[8][7] = 0    
        stims_for_delta[8][39] = 0    
        stims_for_delta[9][2:4] = 0    
        stims_for_delta[9][7] = 0    
        stims_for_delta[9][10] = 0    
        stims_for_delta[9][43:48] = 0    
        stims_for_delta[9][55] = 0    
        stims_for_delta[9][58:61] = 0    
    if '281021' in os.getcwd():
        stims_for_delta[0][41] = 0 
        stims_for_delta[1][20] = 0 
        stims_for_delta[2][49] = 0 
        stims_for_delta[3][47] = 0 
        stims_for_delta[3][51] = 0 
        stims_for_delta[3][52] = 0 
        stims_for_delta[3][56] = 0 
        stims_for_delta[5][38] = 0
        stims_for_delta[5][15] = 0 
        stims_for_delta[5][28] = 0 
        stims_for_delta[5][33] = 0 
        stims_for_delta[6][14] = 0 
        stims_for_delta[6][18] = 0 
        stims_for_delta[6][35] = 0 
        stims_for_delta[7][8] = 0 
        stims_for_delta[7][36] = 0 
        stims_for_delta[8][1] = 0 
        stims_for_delta[9][49] = 0 
        stims_for_delta[9][54] = 0 
    if '291021' in os.getcwd():
        stims_for_delta[0][44] = 0 
        stims_for_delta[4][10] = 0 
        stims_for_delta[4][20] = 0 
        stims_for_delta[5][5] = 0 
        stims_for_delta[7][8] = 0    
        stims_for_delta[7][9] = 0    
        stims_for_delta[8][24] = 0    
        stims_for_delta[8][26] = 0    
        stims_for_delta[8][28] = 0  
        stims_for_delta[8][44] = 0   
        stims_for_delta[9][12] = 0    
        stims_for_delta[9][46] = 0    
        stims_for_delta[9][52] = 0    

    pickle.dump(stims_for_delta, open('stims_for_delta','wb'))

    stim_cumsum = np.cumsum(np.asarray([len(stims_for_delta[i]) for i in range(len(stims_for_delta))]))
    stim_cumsum = np.insert(stim_cumsum, 0, 0)
    all_stims_delta = np.zeros([64, sum([len(stims_for_delta[i]) for i in range(len(stims_for_delta))])])
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_min_rel_change = np.loadtxt('LFP_min_rel_change.csv', delimiter = ',')
    lfp_cutoff_resp_channels = 200
    LFP_resp_channels_cutoff = np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
    os.chdir('..')
    
    def find_stim(stim, stim_cumsum= stim_cumsum):
        where_sweep = np.argwhere(stim_cumsum <= stim)[-1][0]
        print(f'sweep {where_sweep}, stim {stim - stim_cumsum[where_sweep]+ 1}')
    
    # #unpaired whisker
    # if day == '160414_B2':
    #     chans_to_append = [35,37,11] # doesnt change significance of results but clearly good channels
    #     chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels_cutoff]
    #     for chan in chans_to_append:
    #         LFP_resp_channels_cutoff = np.append(PSTH_resp_channels, chan)
    #     # chans_to_append = [8,56,13,61,62] # doesnt change significance of results but clearly good channels
    #     # chans_to_append = [i for i in chans_to_append if i not in PSTH_resp_channels]
    #     # for chan in chans_to_append:
    #     #     PSTH_resp_channels = np.append(PSTH_resp_channels, chan)
    # if day == '160426_B2':
    #     chans_to_append = [9,32,11,20,18,38,23,25,36] # doesnt change significance of results but clearly good channels
    #     chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels_cutoff]
    #     for chan in chans_to_append:
    #         LFP_resp_channels_cutoff = np.append(PSTH_resp_channels, chan)
    #     # chans_to_append = [35,33] # doesnt change significance of results but clearly good channels, not sure why not included with cutoff??
    #     # chans_to_append = [i for i in chans_to_append if i not in PSTH_resp_channels]
    #     # for chan in chans_to_append:
    #     #     PSTH_resp_channels = np.append(PSTH_resp_channels, chan)
    # if day == '160519_D1': # get noisy
    #     chans_to_append = [19,26] # doesnt change significance of results but clearly good channels
    #     chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels_cutoff]
    #     for chan in chans_to_append:
    #         LFP_resp_channels_cutoff = np.append(PSTH_resp_channels, chan)
    #     # PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == 42)[0]) 
    #     # PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == 40)[0])
    #     # PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == 38)[0])
    #     # PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == 23)[0]) 
    # if day == '160624_D1':
    #     chans_to_append = [9,11,13,25,61] # doesnt change significance of results but clearly good channels
    #     chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels_cutoff]
    #     for chan in chans_to_append:
    #         LFP_resp_channels_cutoff = np.append(PSTH_resp_channels, chan)  
    #     # PSTH_resp_channels = np.delete(PSTH_resp_channels, np.where(PSTH_resp_channels == 41)[0]) # gets noisy
    # # if day == '160628_B2':
    # #     chans_to_append = [39,27,17] # doesnt change significance of results but clearly good channels, not sure why not included with cutoff??
    # #     chans_to_append = [i for i in chans_to_append if i not in PSTH_resp_channels]
    # #     for chan in chans_to_append:
    # #         PSTH_resp_channels = np.append(PSTH_resp_channels, chan)
    
    
    if os.path.exists(fr'analysis_{day}\delta_channels.csv') and reanalyze == False:
        os.chdir()
        # delta_channels = np.loadtxt('delta_channels.csv', delimiter = ',', dtype = int)
        fftfreq = np.loadtxt('fftfreq.csv', delimiter = ',')
        delta_power = np.loadtxt('delta_power.csv', delimiter = ',')
        delta_power_rel = np.loadtxt('delta_power_rel.csv', delimiter = ',')
        delta_power_rel_change = np.loadtxt('delta_power_rel_change.csv', delimiter = ',')
        PSD = np.load('PSD.npy')
        delta_lower = np.load('delta_lower.npy')
        delta_upper = np.load('delta_upper.npy')
        to_plot_1_delta = np.loadtxt('to_plot_1_delta.csv', delimiter = ',', dtype = int)
        to_plot_2_delta = np.loadtxt('to_plot_2_delta.csv', delimiter = ',', dtype = int)
    
    else: 
        to_plot_1_delta = [0,1,2,3]    
        to_plot_2_delta = list(np.linspace(4,len(LFP_all_sweeps) - 1, len(LFP_all_sweeps) - 4, dtype = int))
        
        delta_sweeps_for_average = to_plot_1_delta + to_plot_2_delta
        
        exclude_before = 0.1
        # maybe better to take 1 second after stim for slow waves as high change they get fucked up by the stim otherwise?
        exclude_after = 1.9
        
        delta_lower = .5
        delta_upper = 4
        
        fftfreq = np.fft.fftfreq(int((5 - exclude_before - exclude_after)*new_fs), d = (1/new_fs))
        hanning_window = np.tile(np.hanning((5 - exclude_before - exclude_after)*new_fs), (64, 1))
        hamming_window = np.tile(np.hamming((5 - exclude_before - exclude_after)*new_fs), (64, 1))
        
        PSD = np.empty([len(LFP_all_sweeps), LFP_all_sweeps[1].shape[0], int((5 - exclude_before - exclude_after)*new_fs)])
        PSD[:] = np.NaN
        
        delta_power = np.empty([10, 64])
        delta_power[:] = np.NaN
        delta_power_rel = np.empty([10, 64])
        delta_power_rel[:] = np.NaN
        
        for ind_sweep, LFP in enumerate(LFP_all_sweeps):
            if LFP_all_sweeps[ind_sweep].size == 0:
                continue
            print(ind_sweep)
            #EXCLUDE first and last stim just in case there isnt enough time, makes it easier
            FFT_current_sweep = np.zeros([len(stims_for_delta[ind_sweep] - 2), LFP_all_sweeps[ind_sweep].shape[0], int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
            FFT_current_sweep[:] = np.NaN
            for ind_stim, stim in enumerate(list(stims_for_delta[ind_sweep][1:-1])):
                if stim == 0:
                    print(f'{ind_stim}: continue')
                    continue
                # apply hamming (or hanning?) window first
                FFT_current_sweep[ind_stim,:,:] = np.fft.fft(hanning_window*LFP[:, int(stim+exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs)], axis = 1)
                all_stims_delta[:,stim_cumsum[ind_sweep]+ind_stim] = np.transpose(np.nanmean(np.abs(FFT_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]]), axis = 0))
                    
                # if '160202' in os.getcwd(): #  noisy recording, need to take out some stims
                #     if np.any(all_stims_delta[LFP_resp_channels_cutoff,stim_cumsum[ind_sweep]+ind_stim] > 280000):
                #         FFT_current_sweep[ind_stim,:,:] = np.NaN
                #         all_stims_delta[:,stim_cumsum[ind_sweep]+ind_stim] = 0
                #         stims_for_delta[ind_sweep][ind_stim + 1] = 0
                #         print(f'{ind_stim}: continue')
                #         continue
                    
                # if '160128' in os.getcwd():
                #     if np.any(all_stims_delta[LFP_resp_channels_cutoff,stim_cumsum[ind_sweep]+ind_stim] > 550000):
                #         FFT_current_sweep[ind_stim,:,:] = np.NaN
                #         all_stims_delta[:,stim_cumsum[ind_sweep]+ind_stim] = 0
                #         stims_for_delta[ind_sweep][ind_stim + 1] = 0
                #         print(f'{ind_stim}: continue')
                #         continue

            PSD[ind_sweep,:,:] = np.nanmean(np.abs(FFT_current_sweep)**2, axis = 0)
        
            delta_power[ind_sweep,:] = np.nanmean(PSD[ind_sweep,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 0)
        delta_power_rel = delta_power/np.nanmean(delta_power[to_plot_1_delta,:], axis = 0)
        delta_power_rel_change = np.nanmean(delta_power_rel[to_plot_2_delta,:], axis = 0) - np.mean(delta_power_rel[to_plot_1_delta,:], axis = 0)
        
        delta_channels = LFP_resp_channels_cutoff
        
    # to_plot_2_delta = [4,5,6,7,9]
    pickle.dump(stims_for_delta, open('stims_for_delta','wb'))
    
    #delta power timecourse over whole recording
    fig, ax = plt.subplots(8,8, figsize = (12,10)) 
    fig.suptitle(f'delta in all stims {day}')
    for ind, ax1 in enumerate(list(ax.flatten())):                        
        ax1.plot(all_stims_delta[chanMap[ind],:], linewidth = 1)
        # ax1.axhline(450000, linestyle = '--')
        if chanMap[ind] in LFP_resp_channels_cutoff:
            ax1.set_facecolor("y")
        ax1.set_yticks([])
        ax1.set_xticks([])
        ax1.set_title(str(chanMap[ind]), size = 4)
    # plt.tight_layout()
    plt.savefig('delta power whole recording no y-share', dpi = 1000)

    if plot:   
    #delta power timecourse in each channel
        fig, ax = plt.subplots(8,8, figsize = (12,10))
        fig.suptitle(f'delta in all chans {day}')
        for chan in range(64):
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(delta_power_rel[:,chan])
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 4)
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].axvline(x = 4)
            # ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_yticklabels([])
            if chan in LFP_resp_channels_cutoff:
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
        plt.savefig('delta power in all chans', dpi = 1000)

        # delta_channels = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        #        17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        #        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50,
        #        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
        
        # #delta power timecourse in all channels
        # fig, ax = plt.subplots()
        # ax.plot(np.nanmean(delta_power_rel[:,:], axis = 1),'b', label = 'all')
        # ax.plot(np.nanmean(delta_power_rel[:,np.asarray(LFP_resp_channels)[:,np.newaxis]], axis = 1),'r', label = 'LFP_resp')
        # ax.plot(np.nanmean(delta_power_rel[:,np.asarray(delta_channels)[:,np.newaxis]], axis = 1),'c', label = 'delta_resp')
        # plt.legend()
        
        #delta power timecourse over whole recording
        fig, ax = plt.subplots(8,8,sharey = True, figsize = (12,10)) 
        fig.suptitle(f'delta in all stims {day}')
        for ind, ax1 in enumerate(list(ax.flatten())):                        
            ax1.plot(all_stims_delta[chanMap[ind],:], linewidth = 1)
            # ax1.axhline(450000, linestyle = '--')
            if chanMap[ind] in LFP_resp_channels_cutoff:
                ax1.set_facecolor("y")
            ax1.set_yticks([])
            ax1.set_xticks([])
            ax1.set_title(str(chanMap[ind]), size = 4)
        # plt.tight_layout()
        plt.savefig('delta power whole recording', dpi = 1000)



        # frequency as PSD before and after: 
        # with delta sweeps
        # PSD_before_LFP_resp = np.mean(np.mean(PSD[to_plot_1_delta[:,None], LFP_resp_channels, :], axis = 0),axis = 0)
        # PSD_before_delta_resp = np.mean(np.mean(PSD[to_plot_1_delta[:,None], delta_channels, :], axis = 0),axis = 0)
        # PSD_after_LFP_resp = np.mean(np.mean(PSD[to_plot_2_delta[:,None], LFP_resp_channels, :], axis = 0),axis = 0)
        # PSD_after_delta_resp = np.mean(np.mean(PSD[to_plot_2_delta[:,None], delta_channels, :], axis = 0),axis = 0)
        # PSD_after_relative_LFP_resp = PSD_after_LFP_resp/PSD_before_LFP_resp
        # PSD_after_relative_delta_resp = PSD_after_delta_resp/PSD_after_relative_LFP_resp
        
    #PSD with only delta sweeps
    PSD_before_LFP_resp = np.mean(np.mean(PSD[np.asarray(to_plot_1_delta)[:,None], LFP_resp_channels_cutoff, :], axis = 0),axis = 0)
    # PSD_before_delta_resp = np.mean(np.mean(PSD[np.asarray(to_plot_1_delta)[:,None], delta_channels, :], axis = 0),axis = 0)
    PSD_after_LFP_resp = np.mean(np.mean(PSD[np.asarray(to_plot_2_delta)[:,None], LFP_resp_channels_cutoff, :], axis = 0),axis = 0)
    # PSD_after_delta_resp = np.mean(np.mean(PSD[np.asarray(to_plot_2_delta)[:,None], delta_channels, :], axis = 0),axis = 0)
    PSD_after_relative_LFP_resp = PSD_after_LFP_resp/PSD_before_LFP_resp
    # PSD_after_relative_delta_resp = PSD_after_delta_resp/PSD_after_relative_LFP_resp
    
    if plot:    
        fig, ax = plt.subplots()
        fig.suptitle(f'LFP_resp {day}')
        ax.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]], np.abs(PSD_before_LFP_resp[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]]), 'b')
        ax.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]], np.abs(PSD_after_LFP_resp[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]]), 'r')
        plt.savefig(f'PSD_{to_plot_1_delta}_vs_{to_plot_2_delta}', dpi = 1000)
    
        # fig, ax = plt.subplots()
        # fig.suptitle('delta_resp')
        # ax.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]], np.abs(PSD_before_delta_resp[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]]), 'b')
        # ax.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]], np.abs(PSD_after_delta_resp[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]]), 'r')
        
        
        # #average before and after as bar plots
        # fig, ax = plt.subplots(8,8)
        # for chan in range(64):
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].bar(0,np.mean(delta_power[to_plot_1_delta,chan]))
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].bar(1,np.mean(delta_power[to_plot_2_delta,chan]))
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_yticklabels([])
        
        
        # # relative change as color plot, all channels
        # fig, ax = plt.subplots()
        # fig.suptitle('relative change in delta power')
        # im = ax.imshow(np.reshape(delta_power_rel_change[chanMap], (8, 8)), cmap = 'jet', vmax = 0)
        # fig.colorbar(im)
        
        # #correlation with change in LFP
        fig, ax = plt.subplots(2,1)
        fig.suptitle('delta power vs LFP, ALL vs lfp resp')
        #take out outliers??
        ax[0].scatter(LFP_min_rel_change, delta_power_rel_change)
        ax[0].set_xlabel('relative LFP change')
        ax[0].set_ylabel('relative delta change')
        ax[1].scatter(LFP_min_rel_change[LFP_resp_channels_cutoff], delta_power_rel_change[LFP_resp_channels_cutoff])
        ax[1].set_xlabel('relative LFP change')
        ax[1].set_ylabel('relative delta change')
        # # ax.set_xlim(right = 0)
        # # ax.set_ylim(bottom = -60000)
        plt.savefig('LFP change vs delta change', dpi = 1000)
    
    
    # #overall change in LFP resp channels:
    # overall_delta_change_LFP_resp = np.mean(delta_power_rel_change[LFP_resp_channels_cutoff])
    # overall_delta_change_all = np.mean(delta_power_rel_change)
    # print(f'overall relative delta change in LFP resp channels: {overall_delta_change_LFP_resp}')
    # print(f'overall relative delta change in all channels: {overall_delta_change_all}')
    
    # channel_avg_delta_change = (np.mean(PSD_after_LFP_resp[np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]]) - np.mean(PSD_before_LFP_resp[np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]]))/np.mean(PSD_before_LFP_resp[np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]])
    # print(f'overall relative delta change averaging PSD in all LFP resp channels: {channel_avg_delta_change}')
    
    
    # os.chdir(home_directory)
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    np.savetxt('delta_channels.csv', delta_channels, delimiter = ',')
    np.savetxt('fftfreq.csv', fftfreq, delimiter = ',')
    np.savetxt('delta_power.csv', delta_power, delimiter = ',')
    np.savetxt('delta_power_rel.csv', delta_power_rel, delimiter = ',')
    np.savetxt('delta_power_rel_change.csv', delta_power_rel_change, delimiter = ',')
    np.save('PSD.npy', PSD)
    np.savetxt('to_plot_1_delta.csv', to_plot_1_delta, delimiter = ',')
    np.savetxt('to_plot_2_delta.csv', to_plot_2_delta, delimiter = ',')
    np.save('delta_lower.npy', delta_lower)
    np.save('delta_upper.npy', delta_upper)
    
    os.chdir('..')
    os.chdir('..')

    # cl()



#%% SLOW WAVES extracting
highpass_cutoff = 4

zero_mean = True

 # -------------------------------------------------------------------------------------- SW --------------------------------------------------------------------------------------------
# You want waveform, firstamp, secondamp, firstslope, secondslope and duration for every sweep
# os.chdir(home_directory)
lfp_cutoff_resp_channels = 200
to_plot_1_LFP = [0,1,2,3]
to_plot_2_LFP = [4,5,6,7,8,9]  
to_plot_1_SW = [0,1,2,3]
to_plot_2_SW = [4,5,6,7,8,9]  

UP_std_cutoff = 1.75

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for day in days:
# for day in ['221216', '221212']:
# for day in ['221208']:
            
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
        # stims_for_delta[5][50:] = 0    
        pickle.dump(stims_for_delta, open('stims_for_delta','wb'))

    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_resp_channels_cutoff =  np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
    # SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',', dtype = int)
    
    spont_spiking = np.zeros([10,64])
    
    SW_waveform_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    SW_spiking_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    Peak_dur_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    SW_fslope_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    SW_sslope_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    SW_famp_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
    SW_samp_sweeps = [[[] for i in range(64)] for j in range(len(LFP_all_sweeps))]
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
    SW_frequency_sweeps_avg[:] = np.NaN
    SW_waveform_sweeps_avg[:] = np.NaN
    SW_spiking_sweeps_avg[:] = np.NaN
    Peak_dur_sweeps_avg[:] = np.NaN
    SW_fslope_sweeps_avg[:] = np.NaN
    SW_sslope_sweeps_avg[:] = np.NaN
    SW_famp_sweeps_avg[:] = np.NaN
    SW_samp_sweeps_avg[:] = np.NaN
    # median value within sweep
    SW_frequency_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
    SW_waveform_sweeps_median = np.zeros([len(LFP_all_sweeps), 64, 1000])
    SW_spiking_sweeps_median = np.zeros([len(LFP_all_sweeps), 64, 1000])
    Peak_dur_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
    SW_fslope_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
    SW_sslope_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
    SW_famp_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
    SW_samp_sweeps_median = np.zeros([len(LFP_all_sweeps), 64])
    SW_frequency_sweeps_median[:] = np.NaN
    SW_waveform_sweeps_median[:] = np.NaN
    SW_spiking_sweeps_median[:] = np.NaN
    Peak_dur_sweeps_median[:] = np.NaN
    SW_fslope_sweeps_median[:] = np.NaN
    SW_sslope_sweeps_median[:] = np.NaN
    SW_famp_sweeps_median[:] = np.NaN
    SW_samp_sweeps_median[:] = np.NaN

    
    exclude_before = 0.1
    # maybe better to take 1 second after stim for slow waves as high change they get fucked up by the stim otherwise?
    exclude_after = 1.4
    duration_criteria = 100
    
    # filter in slow wave range, then find every time it goes under varxSD i.e.=  upstate
    for ind_sweep, LFP in enumerate(LFP_all_sweeps):
        if LFP_all_sweeps[ind_sweep].size == 0:
            continue
        LFP_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 2*pq.Hz).as_array()
        #EXCLUDE first and last stim just in case there isnt enough time, makes it easier for basic dumb bitchezz like jp <3
        for ind_stim, stim in enumerate(list(stims_for_delta[ind_sweep][1:-1])):
            if stim == 0:
                continue
            print(ind_sweep, ind_stim)
            curr_LFP_filt_total = LFP_filt[int(stim):int(stim + 5*new_fs), :]
            curr_LFP_filt = LFP_filt[int(stim + exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs), :]
            if zero_mean:
                curr_LFP_filt = curr_LFP_filt - np.mean(curr_LFP_filt, axis = 0) # zero mean

            for chan in range(64):
                
                # because spiking is saved as dict of channels need to convert it to list to be able to access channels
                chan_spiking = list(spikes_allsweeps[ind_sweep].values())[chan]
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
        
        #convert spontaneous spiking in Hz by dividing by seconds (stims times inter stimulus interval)
        spont_spiking[ind_sweep,:] = spont_spiking[ind_sweep,:]/((5 - exclude_before - exclude_after)*(len(stims_for_delta[ind_sweep]) - 2))
        
        
    np.save('spont_spiking.npy', spont_spiking)
    
    # average over whole sweep. MAYBE ALSO TAKE OUT OUTLIERS?? so  maybe first concatenate before and after, take out outliers and average then 
    for ind_sweep in range(len(LFP_all_sweeps)):
        if LFP_all_sweeps[ind_sweep].size == 0:
            continue
        for chan in range(64):
            SW_frequency_sweeps_avg[ind_sweep,chan] = len(Peak_dur_sweeps[ind_sweep][chan])/(len(stims_for_delta[ind_sweep][stims_for_delta[ind_sweep] != 0]) - 2) # -2 because exclude first and last stim

            SW_waveform_sweeps_avg[ind_sweep,chan,:] = np.mean(np.asarray(SW_waveform_sweeps[ind_sweep][chan]), axis = 0)
            SW_spiking_sweeps_avg[ind_sweep,chan,:] = np.mean(np.asarray(SW_spiking_sweeps[ind_sweep][chan]), axis = 0)
            Peak_dur_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(Peak_dur_sweeps[ind_sweep][chan]))
            SW_fslope_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_fslope_sweeps[ind_sweep][chan]))
            SW_sslope_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_sslope_sweeps[ind_sweep][chan]))
            SW_famp_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_famp_sweeps[ind_sweep][chan]))
            SW_samp_sweeps_avg[ind_sweep,chan] = np.mean(np.asarray(SW_samp_sweeps[ind_sweep][chan]))
            
            SW_waveform_sweeps_median[ind_sweep,chan,:] = np.median(np.asarray(SW_waveform_sweeps[ind_sweep][chan]), axis = 0)
            SW_spiking_sweeps_median[ind_sweep,chan,:] = np.median(np.asarray(SW_spiking_sweeps[ind_sweep][chan]), axis = 0)
            Peak_dur_sweeps_median[ind_sweep,chan] = np.median(np.asarray(Peak_dur_sweeps[ind_sweep][chan]))
            SW_fslope_sweeps_median[ind_sweep,chan] = np.median(np.asarray(SW_fslope_sweeps[ind_sweep][chan]))
            SW_sslope_sweeps_median[ind_sweep,chan] = np.median(np.asarray(SW_sslope_sweeps[ind_sweep][chan]))
            SW_famp_sweeps_median[ind_sweep,chan] = np.median(np.asarray(SW_famp_sweeps[ind_sweep][chan]))
            SW_samp_sweeps_median[ind_sweep,chan] = np.median(np.asarray(SW_samp_sweeps[ind_sweep][chan]))

    
    pickle.dump(UP_Cross_sweeps, open('UP_Cross_sweeps', 'wb')) # UP cross times
    pickle.dump(SW_waveform_sweeps, open('SW_waveform_sweeps', 'wb'))
    pickle.dump(Peak_dur_sweeps, open('Peak_dur_sweeps', 'wb'))
    pickle.dump(SW_spiking_sweeps, open('SW_spiking_sweeps', 'wb'))
    pickle.dump(SW_fslope_sweeps, open('SW_fslope_sweeps', 'wb'))
    pickle.dump(SW_sslope_sweeps, open('SW_sslope_sweeps', 'wb'))
    pickle.dump(SW_famp_sweeps, open('SW_famp_sweeps', 'wb'))
    pickle.dump(SW_samp_sweeps, open('SW_samp_sweeps', 'wb'))      

    # save mean and median across all SW in a sweep in each channel and sweep
    np.save('SW_waveform_sweeps_avg.npy', SW_waveform_sweeps_avg)
    np.save('SW_frequency_sweeps_avg.npy', SW_frequency_sweeps_avg)
    np.save('SW_spiking_sweeps_avg.npy', SW_spiking_sweeps_avg)
    np.save('Peak_dur_sweeps_avg.npy', Peak_dur_sweeps_avg)
    np.save('SW_fslope_sweeps_avg.npy', SW_fslope_sweeps_avg)
    np.save('SW_sslope_sweeps_avg.npy', SW_sslope_sweeps_avg)
    np.save('SW_famp_sweeps_avg.npy', SW_famp_sweeps_avg)
    np.save('SW_samp_sweeps_avg.npy', SW_samp_sweeps_avg)
    
    np.save('SW_waveform_sweeps_median.npy', SW_waveform_sweeps_median)
    np.save('SW_spiking_sweeps_median.npy', SW_spiking_sweeps_median)
    np.save('Peak_dur_sweeps_median.npy', Peak_dur_sweeps_median)
    np.save('SW_fslope_sweeps_median.npy', SW_fslope_sweeps_median)
    np.save('SW_sslope_sweeps_median.npy', SW_sslope_sweeps_median)
    np.save('SW_famp_sweeps_median.npy', SW_famp_sweeps_median)
    np.save('SW_samp_sweeps_median.npy', SW_samp_sweeps_median)



    #relative change in individual params before vs after in all channels, of mean and median across all SW in a sweep
    Freq_change = (np.mean(SW_frequency_sweeps_avg[to_plot_2_SW,:], axis = 0) - np.mean(SW_frequency_sweeps_avg[to_plot_1_SW,:], axis = 0))/np.mean(SW_frequency_sweeps_avg[to_plot_1_SW,:], axis = 0)
    
    Peak_dur_change_mean = (np.mean(Peak_dur_sweeps_avg[to_plot_2_SW,:], axis = 0) - np.mean(Peak_dur_sweeps_avg[to_plot_1_SW,:], axis = 0))/np.mean(Peak_dur_sweeps_avg[to_plot_1_SW,:], axis = 0)
    Fslope_change_mean = (np.mean(SW_fslope_sweeps_avg[to_plot_2_SW,:], axis = 0) - np.mean(SW_fslope_sweeps_avg[to_plot_1_SW,:], axis = 0))/np.mean(SW_fslope_sweeps_avg[to_plot_1_SW,:], axis = 0)
    Sslope_change_mean = (np.mean(SW_sslope_sweeps_avg[to_plot_2_SW,:], axis = 0) - np.mean(SW_sslope_sweeps_avg[to_plot_1_SW,:], axis = 0))/np.mean(SW_sslope_sweeps_avg[to_plot_1_SW,:], axis = 0)
    Famp_change_mean = (np.mean(SW_famp_sweeps_avg[to_plot_2_SW,:], axis = 0) - np.mean(SW_famp_sweeps_avg[to_plot_1_SW,:], axis = 0))/np.mean(SW_famp_sweeps_avg[to_plot_1_SW,:], axis = 0)
    Samp_change_mean = (np.mean(SW_samp_sweeps_avg[to_plot_2_SW,:], axis = 0) - np.mean(SW_samp_sweeps_avg[to_plot_1_SW,:], axis = 0))/np.mean(SW_samp_sweeps_avg[to_plot_1_SW,:], axis = 0)
    
    Peak_dur_change_median = (np.mean(Peak_dur_sweeps_median[to_plot_2_SW,:], axis = 0) - np.mean(Peak_dur_sweeps_median[to_plot_1_SW,:], axis = 0))/np.mean(Peak_dur_sweeps_median[to_plot_1_SW,:], axis = 0)
    Fslope_change_median = (np.mean(SW_fslope_sweeps_median[to_plot_2_SW,:], axis = 0) - np.mean(SW_fslope_sweeps_median[to_plot_1_SW,:], axis = 0))/np.mean(SW_fslope_sweeps_median[to_plot_1_SW,:], axis = 0)
    Sslope_change_median = (np.mean(SW_sslope_sweeps_median[to_plot_2_SW,:], axis = 0) - np.mean(SW_sslope_sweeps_median[to_plot_1_SW,:], axis = 0))/np.mean(SW_sslope_sweeps_median[to_plot_1_SW,:], axis = 0)
    Famp_change_median = (np.mean(SW_famp_sweeps_median[to_plot_2_SW,:], axis = 0) - np.mean(SW_famp_sweeps_median[to_plot_1_SW,:], axis = 0))/np.mean(SW_famp_sweeps_median[to_plot_1_SW,:], axis = 0)
    Samp_change_median = (np.mean(SW_samp_sweeps_median[to_plot_2_SW,:], axis = 0) - np.mean(SW_samp_sweeps_median[to_plot_1_SW,:], axis = 0))/np.mean(SW_samp_sweeps_median[to_plot_1_SW,:], axis = 0)

    np.savetxt('Freq_change.csv', Freq_change, delimiter = ',')
    
    np.savetxt('Peak_dur_change_mean.csv', Peak_dur_change_mean, delimiter = ',')
    np.savetxt('Fslope_change_mean.csv', Fslope_change_mean, delimiter = ',')
    np.savetxt('Sslope_change_mean.csv', Sslope_change_mean, delimiter = ',')
    np.savetxt('Famp_change_mean.csv', Famp_change_mean, delimiter = ',')
    np.savetxt('Samp_change_mean.csv', Samp_change_mean, delimiter = ',')
    
    np.savetxt('Peak_dur_change_median.csv', Peak_dur_change_median, delimiter = ',')
    np.savetxt('Fslope_change_median.csv', Fslope_change_median, delimiter = ',')
    np.savetxt('Sslope_change_median.csv', Sslope_change_median, delimiter = ',')
    np.savetxt('Famp_change_median.csv', Famp_change_median, delimiter = ',')
    np.savetxt('Samp_change_median.csv', Samp_change_median, delimiter = ',')



    
    
    #redo SW param values with the mean waveforms (better):
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
    # PLOT INDIVIDUAL SLOW WAVES
    for sweep in range(10):
        fig, ax = plt.subplots(8,8, figsize = (15,12))
        fig.suptitle(f'Slow Waves sweep {sweep + 1}')
        for ind, ax1 in enumerate(list(ax.flatten())):  
            ax1.tick_params(axis='both', which='minor', labelsize=4)
            ax1.tick_params(axis='both', which='major', labelsize=4)
            ax1.set_xticks([])
            if chanMap[ind] in LFP_resp_channels_cutoff:                     
                ax1.plot(np.asarray(SW_waveform_sweeps[sweep][chanMap[ind]]).T, linewidth = 0.45)
                ax1.set_title(str(chanMap[ind]), size = 5)
        plt.tight_layout()
        plt.savefig(f'Slow Waves sweep {sweep + 1}', dpi = 1000)
        cl()

    #  changes when done on the average slow wave
    fig, ax = plt.subplots(8,8)
    fig.suptitle('MEDIAN: Freq, Dur, Fslope, Sslope, Famp, Samp')
    for ind, ax1 in enumerate(list(ax.flatten())):
        chan = chanMap[ind]
        ax1.bar(range(6), [Freq_change[chan], Peak_dur_change_median[chan], Fslope_change_median[chan], Sslope_change_median[chan], Famp_change_median[chan], Samp_change_median[chan]])
        ax1.set_title(str(chan), size = 5)
        ax1.set_yticklabels([])
        ax1.set_ylim([-1,1])
        if chan in LFP_resp_channels_cutoff:
            ax1.set_facecolor("y")
    plt.savefig(f'Slow waves params median', dpi = 1000)

    fig, ax = plt.subplots()
    fig.suptitle('MEDIAN of medians: Freq, Dur, Fslope, Sslope, Famp, Samp')
    ax.bar(range(6), list(map(np.median, [Freq_change[LFP_resp_channels_cutoff], Peak_dur_change_median[LFP_resp_channels_cutoff], Fslope_change_median[LFP_resp_channels_cutoff], Sslope_change_median[LFP_resp_channels_cutoff], Famp_change_median[LFP_resp_channels_cutoff], Samp_change_median[LFP_resp_channels_cutoff]])))
    ax1.set_ylim([-1,1])
    plt.savefig('Slow waves params median MEDIAN', dpi = 1000)

    os.chdir('..')


#%% SLOW WAVES plotting.
# os.chdir(home_directory)

lfp_cutoff_resp_channels = 200
# for day in [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]:

# for day in ['221216', '221212']:
for day in ['221208']:
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
    
    # if os.path.exists(fr'analysis_{day}\SW_resp_channels.csv'):
    # os.chdir(f'analysis_{day}')
    # SW_resp_channels = np.loadtxt('SW_resp_channels.csv', delimiter = ',')
    # Freq_change = np.loadtxt('Freq_change.csv', delimiter = ',')
    # Peak_dur_change = np.loadtxt('Peak_dur_change.csv', delimiter = ',')
    # Fslope_change = np.loadtxt('Fslope_change.csv', delimiter = ',')
    # Sslope_change = np.loadtxt('Sslope_change.csv', delimiter = ',')
    # Famp_change = np.loadtxt('Famp_change.csv', delimiter = ',')
    # Samp_change = np.loadtxt('Samp_change.csv', delimiter = ',')
    # Peak_dur_overall_change = np.loadtxt('Peak_dur_overall_change.csv', delimiter = ',')
    # Fslope_overall_change = np.loadtxt('Fslope_overall_change.csv', delimiter = ',')
    # Sslope_overall_change = np.loadtxt('Sslope_overall_change.csv', delimiter = ',')
    # Famp_overall_change = np.loadtxt('Famp_overall_change.csv', delimiter = ',')
    # Samp_overall_change = np.loadtxt('Samp_overall_change.csv', delimiter = ',')
    # SW_params_channels = np.loadtxt('SW_params_channels.csv', delimiter = ',')
    # SW_spiking_peak_change = np.loadtxt('SW_spiking_peak_change.csv', delimiter = ',')
    # SW_spiking_area_change = np.loadtxt('SW_spiking_area_change.csv', delimiter = ',')
    # to_plot_2_delta = np.loadtxt('to_plot_1_SW.csv', delimiter = ',')
    # to_plot_2_delta = np.loadtxt('to_plot_2_SW.csv', delimiter = ',')
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
    


    #check waveforms for every sweep individually to check for outliers.
    # first baseline
    fig, ax = plt.subplots(8,8)
    fig.suptitle('before')
    for chan in range(64):
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[0,chan,:], 'b', label = '1')
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[1,chan,:], 'r', label = '2')
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[2,chan,:], 'y', label = '3')
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[3,chan,:], 'c', label = '4')
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_yticklabels([])
        # if chan in LFP_resp_channels_cutoff:
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
    plt.legend()
    
    # check after pairing every sweep
    fig, ax = plt.subplots(8,8)
    fig.suptitle('after')
    for chan in range(64):
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[4,chan,:], 'b', label = '5')
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[5,chan,:], 'r', label = '6')
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[6,chan,:], 'y', label = '7')
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[7,chan,:], 'c', label = '8')
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[8,chan,:], 'k', label = '9')
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(SW_waveform_sweeps_avg[9,chan,:], 'm', label = '10')
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
    #     if chan in LFP_resp_channels_cutoff:
    #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
    plt.legend()
    
    
    #     if chan in LFP_resp_channels:
    #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
    
    #  changes when done on the average slow wave
    fig, ax = plt.subplots(8,8)
    fig.suptitle('AVERAGED: Freq, Dur, Fslope, Sslope, Famp, Samp')
    for chan in range(64):
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].bar(range(6), [Freq_change[chan], Peak_dur_overall_change[chan], Fslope_overall_change[chan], Sslope_overall_change[chan], Famp_overall_change[chan], Samp_overall_change[chan]])
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_yticklabels([])
        if chan in LFP_resp_channels_cutoff:
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
        
    # # SW_params_channels = list(np.linspace(0,63,64, dtype = int))
    
    # #average change in slow wave characteristics in channels LFP channels
    # fig, ax = plt.subplots()
    # fig.suptitle('LFP resp channels')
    # ax.bar(range(6), list(map(np.nanmean, [Freq_change[LFP_resp_channels], Peak_dur_overall_change[LFP_resp_channels], Fslope_overall_change[LFP_resp_channels], Sslope_overall_change[LFP_resp_channels], Famp_overall_change[LFP_resp_channels], Samp_overall_change[LFP_resp_channels]])))
    # ax.set_xticks([0,1,2,3,4,5])
    # ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'])
    
    # #average change in slow wave characteristics in channels delta power channels
    # fig, ax = plt.subplots()
    # fig.suptitle('delta resp channels')
    # ax.bar(range(6), list(map(np.nanmean, [Freq_change[delta_channels], Peak_dur_overall_change[delta_channels], Fslope_overall_change[delta_channels], Sslope_overall_change[delta_channels], Famp_overall_change[delta_channels], Samp_overall_change[delta_channels]])))
    # ax.set_xticks([0,1,2,3,4,5])
    # ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'])
    
    # #average change in slow wave characteristics in SW channels
    # fig, ax = plt.subplots()
    # fig.suptitle('SW resp channels MEAN')
    # ax.bar(np.linspace(0,5,6), list(map(np.nanmean, [Freq_change[SW_params_channels], Peak_dur_overall_change[SW_params_channels], Fslope_overall_change[SW_params_channels], Sslope_overall_change[SW_params_channels], Famp_overall_change[SW_params_channels], Samp_overall_change[SW_params_channels]])))
    # ax.set_xticks([0,1,2,3,4,5])
    # ax.set_xticklabels(['freq', 'dur', 'fslope', 'sslope', 'famp', 'samp'])
    # plt.savefig('SW resp channels mean', dpi = 1000)
    
    
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
    SW_spiking_channels = [30,46,31,47,1,49,0,44,29,45,3,51,2,27,43,4,52,25,41,54,36,21,34,19,35,16,32,17,33,15]
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
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 6)
        # ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].axvline(x = 3)
        ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_yticklabels([])
        if chan in SW_spiking_channels:
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")        
    plt.savefig(f'SW Spiking {to_plot_1_SW} vs {to_plot_2_SW}', dpi = 1000)
    
    # #change in slow-wave evoked spiking? do peak, area under the curve
    # #overall change in all channels:
    fig, ax = plt.subplots()
    ax.plot(np.mean(np.nanmean(SW_spiking_sweeps_avg[to_plot_1_SW,:,:], axis = 0), axis = 0), 'b')
    ax.plot(np.mean(np.nanmean(SW_spiking_sweeps_avg[to_plot_2_SW,:,:], axis = 0), axis = 0), 'r')
    
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


# correlation between different slow wave characteristics/delta power, averaged across channels



# explained delta power variance from different slow wave characteristics



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
    os.chdir('..')
    
    
    



#%% UP vs DOWN stim delivery. FOR OLDER MICE THIS WAS DONE WITH HIGHPASS 5

#UP stim when spikes in 100ms before stim, if not DOWN stim. This is not exactly perfect as often UP/DOWN states not synchronous across channels... choose channel wisely
# os.chdir(home_directory)
to_plot_1_LFP = [0,1,2,3]
to_plot_2_LFP = [4,5,6,7,8,9]
lfp_cutoff_resp_channels = 200
new_fs = 1000
highpass_cutoff = 4

# artifacts = list(np.linspace(97,123,26,dtype = int))

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
# for day in ['281021']:    
for day in days:
    os.chdir(day) 
    print(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    try:
        spikes_allsweeps = pickle.load(open(f'spikes_allsweeps_{highpass_cutoff}','rb'))
    except FileNotFoundError:
        print(f'spikes with highpass {highpass_cutoff} not found')
        spikes_allsweeps = pickle.load(open('spikes_allsweeps_5','rb'))
    stim_times = pickle.load(open('stim_times','rb'))

    if '221220_3' in os.getcwd() or '221213' in os.getcwd() or '221216' in os.getcwd() or '221219_1' in os.getcwd():  
        plot = False
    else:
        plot = False
    
    if day in ['160310', '160414_D1', '160426_D1', '160519_B2', '160624_B2', '160628_D1', '160128', '160202', '160218', '160308', '160322', '160331', '160420', '160427']:        
        artifacts = []
    else:
        # artifacts = []
        artifacts = list(np.linspace(97,123,27,dtype = int))                 
 
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
        
        to_plot = np.zeros([64, int(0.6*new_fs), len(sweeps_to_plot)])    
        for ind_sweep, sweep in enumerate(sweeps_to_plot):
            curr_to_plot = np.zeros([64, int(0.6*new_fs), len(np.where(stims[sweep] > 0)[0])])
            for ind_stim, stim in enumerate(list(stims[sweep])):
                if stim == 0:
                    continue
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

    def PSTH_matrix(sweeps_to_plot, take_out_artifacts = True, artifact_locs = [], stims = stim_times):
        to_plot = np.zeros([299,64,len(sweeps_to_plot)])
        for ind_sweep, sweep in enumerate(sweeps_to_plot):
            #PSTH_matrix is mean across trials in one sweep
            PSTH_matrix = np.zeros([299,64])
            bins = np.linspace(1,300,300)
            for ind_chan, chan in enumerate(natsort.natsorted(list(spikes_allsweeps[sweep].keys()))):
                # print(chan)
                currchan = np.zeros([299,len(stims[sweep])])
                for ind_stim, j in enumerate(list(stims[sweep])):
                    currchan[:,ind_stim] = np.histogram((spikes_allsweeps[sweep][chan][(j - 0.1*new_fs < spikes_allsweeps[sweep][chan]) & (spikes_allsweeps[sweep][chan] < j+0.2*new_fs)] - (j-0.1*new_fs)), bins)[0]
                    if take_out_artifacts:
                        currchan[:,ind_stim][artifact_locs] = 0
                PSTH_matrix[:,ind_chan] = np.squeeze(np.mean(currchan, 1)) # mean across stims for every channel
            to_plot[:,:,ind_sweep] = PSTH_matrix
        return np.squeeze(np.mean(to_plot,2))

    if os.path.exists('stims_for_LFP'):
        stims_for_LFP = pickle.load(open('stims_for_LFP', 'rb'))
    else:
        stims_for_LFP = copy.deepcopy(stim_times)
        # stims_for_LFP[3][10:] = 0    
        # stims_for_LFP[5][25:] = 0
        pickle.dump(stims_for_LFP, open('stims_for_LFP', 'wb'))
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',', dtype = int)
    LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
    PSTH_resp_channels = np.loadtxt('PSTH_resp_channels.csv', delimiter = ',', dtype = int)
    os.chdir('..')
    

    for sweep in range(len(LFP_all_sweeps)):
        if LFP_all_sweeps[sweep].size == 0:
            continue
        else:
            curr_LFP_avg = LFP_average([sweep], stims = stims_for_LFP) 
            LFP_min[sweep,:] = np.abs(np.min(curr_LFP_avg[:,210:300], 1) - curr_LFP_avg[:,210])

    # UP_DOWN_channels = [38,40]
    # if '160414_D1' in os.getcwd():
    #     UP_DOWN_channels = LFP_resp_channels_cutoff
    # else:
    #     UP_DOWN_channels = np.argmax(np.mean(LFP_min[0:4,:], axis = 0)) #channel with biggest LFP deflection during baseline
    # UP_DOWN_channels = np.argmax(np.mean(PSTH_resp_magn[0:4,:], axis = 0)) #channel with biggest PSTH magn during baseline
    
    if '160310' in os.getcwd():
        UP_DOWN_channels = np.array([31,29,27,47,45,43])
    elif '160414' in os.getcwd():
        UP_DOWN_channels = np.array([10,13,12,60,63,14,62,7,55,39])        
    elif '160624_B2' in os.getcwd():
        UP_DOWN_channels = np.array([23,39,9,19,35,17,33])
    else:
        UP_DOWN_channels = PSTH_resp_channels
        
    UP_stims = [[] for sweep in range(len(stim_times))]
    DOWN_stims = [[] for sweep in range(len(stim_times))]
    
    tolerance_before_1 = 25 # how long before stim to look at
    tolerance_before_2 = 5 # how long before stim to not look at (CAVE stim artifacts)
    
    #JUST use LFP responsive channels (or maybe most responsive channel) to separate stims, probably better for this! (can try also all channels separately if doesnt work properly)
    for sweep_ind, stims in enumerate(stims_for_LFP):
        if UP_DOWN_channels.size > 1:
            spikes = np.sort(np.concatenate(np.asarray(list(spikes_allsweeps[sweep_ind].values()))[UP_DOWN_channels]))
        else:
            spikes = np.sort(np.asarray(list(spikes_allsweeps[sweep_ind].values()))[UP_DOWN_channels])
        for stim in stims:
            # concatenate all spikes of the LFP responsive channels in that sweep
            if spikes[np.logical_and((stim - tolerance_before_1) < spikes, (stim - tolerance_before_2) > spikes)].size == 0:
                DOWN_stims[sweep_ind].append(stim)
            else:
                UP_stims[sweep_ind].append(stim)
        DOWN_stims[sweep_ind] = np.asarray(DOWN_stims[sweep_ind])
        UP_stims[sweep_ind] = np.asarray(UP_stims[sweep_ind])
    UP_stims_freq = np.asarray([len(UP_stims[i])/(len(DOWN_stims[i]) + len(UP_stims[i])) for i in range(len(UP_stims))])


# --------------------------------------------------------------------- plot in all channels as chanmap ----------------------------------------------------
    #plot UP vs DOWN stim delivery LFP responses during baseline:
    if plot:
        fig, ax = plt.subplots(8,8, sharey = True) 
        fig.suptitle('UP vs DOWN')
        to_plot_1 = LFP_average([0,1,2,3], stims=DOWN_stims)
        to_plot_2 = LFP_average([0,1,2,3], stims=UP_stims)    
        for chan in range(64):                  
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(to_plot_1[chan,:] - to_plot_1[chan,200], 'b')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(to_plot_2[chan,:] - to_plot_2[chan,200], 'r')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
            if chan in LFP_resp_channels_cutoff:
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
            # ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_xlim([50,200])
                    
        #plot UP vs DOWN stim delivery PSTH responses during baseline:
        fig, ax = plt.subplots(8,8,sharey = True)
        fig.suptitle('UP vs DOWN')
        to_plot_1 = PSTH_matrix([0,1,2,3], stims=DOWN_stims, artifact_locs=artifacts)
        to_plot_2 = PSTH_matrix([0,1,2,3], stims=UP_stims, artifact_locs=artifacts)
        for chan in range(64):
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(to_plot_1[:,chan], 6), 'b', linewidth = 1)
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(to_plot_2[:,chan], 6), 'r', linewidth = 1) 
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_xlim([50,200])
            if chan in PSTH_resp_channels:
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
    
        # #plot UP vs DOWN stim delivery PSTH responses during baseline:
        fig, ax = plt.subplots(8,8,sharey = True)
        fig.suptitle('UP')
        to_plot_1 = PSTH_matrix([0], stims=UP_stims, artifact_locs=artifacts)
        to_plot_2 = PSTH_matrix([1], stims=UP_stims, artifact_locs=artifacts)
        to_plot_3 = PSTH_matrix([2], stims=UP_stims, artifact_locs=artifacts)
        to_plot_4 = PSTH_matrix([3], stims=UP_stims, artifact_locs=artifacts)
        for chan in range(64):
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(to_plot_1[:,chan], 6), 'b', linewidth = .5, label = '1')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(to_plot_2[:,chan], 6), 'r', linewidth = .5, label = '2')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(to_plot_3[:,chan], 6), 'k', linewidth = .5, label = '3')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(to_plot_4[:,chan], 6), 'c', linewidth = .5, label = '4') 
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_xlim([50,200])
            if chan in PSTH_resp_channels:
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
        plt.legend()  


        fig, ax = plt.subplots(8,8,sharey = True)
        fig.suptitle('DOWN')
        to_plot_1 = PSTH_matrix([0], stims=DOWN_stims, artifact_locs=artifacts)
        to_plot_2 = PSTH_matrix([1], stims=DOWN_stims, artifact_locs=artifacts)
        to_plot_3 = PSTH_matrix([2], stims=DOWN_stims, artifact_locs=artifacts)
        to_plot_4 = PSTH_matrix([3], stims=DOWN_stims, artifact_locs=artifacts)
        for chan in range(64):
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(to_plot_1[:,chan], 6), 'b', linewidth = .5, label = '1')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(to_plot_2[:,chan], 6), 'r', linewidth = .5, label = '2')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(to_plot_3[:,chan], 6), 'k', linewidth = .5, label = '3')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(smooth(to_plot_4[:,chan], 6), 'c', linewidth = .5, label = '4') 
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_xlim([50,200])
            if chan in PSTH_resp_channels:
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
        plt.legend()  

        #plot distribution of frequencies of UP/DOWN deliveries over time
        fig, ax = plt.subplots()
        ax.bar(range(len(UP_stims)), UP_stims_freq)
    
    
    
    # --------------------------------------------------------------- plot overall average response, UP vs DOWN stim during baseline for example figure
    fig, ax = plt.subplots(figsize = (5,3))
    to_plot_DOWN_mean = np.nanmean(LFP_average(to_plot_1_LFP, stims=DOWN_stims)[LFP_resp_channels_cutoff,:], axis = 0)/1000
    to_plot_DOWN_err = np.nanstd(LFP_average(to_plot_1_LFP, stims=DOWN_stims)[LFP_resp_channels_cutoff,:], axis = 0)/1000
    to_plot_UP_mean = np.nanmean(LFP_average(to_plot_1_LFP, stims=UP_stims)[LFP_resp_channels_cutoff,:], axis = 0)/1000
    to_plot_UP_err = np.nanstd(LFP_average(to_plot_1_LFP, stims=UP_stims)[LFP_resp_channels_cutoff,:], axis = 0)/1000
    ax.plot(to_plot_DOWN_mean, color = 'k')
    ax.fill_between(np.linspace(0, 599, 600), to_plot_DOWN_mean + patch*to_plot_DOWN_err/np.sqrt(LFP_resp_channels_cutoff.size), to_plot_DOWN_mean - patch*to_plot_DOWN_err/np.sqrt(LFP_resp_channels_cutoff.size), alpha = 0.1, color = 'k')
    ax.plot(to_plot_UP_mean, color = 'r')
    ax.fill_between(np.linspace(0, 599, 600), to_plot_UP_mean + patch*to_plot_UP_err/np.sqrt(LFP_resp_channels_cutoff.size), to_plot_UP_mean - patch*to_plot_UP_err/np.sqrt(LFP_resp_channels_cutoff.size), alpha = 0.1, color = 'r')
    ax.set_xlim([100,400])
    ax.set_xlabel('time from stim (ms)', size = 16)
    ax.set_ylabel('LFP (mV)', size = 16)
    ax.set_xticks([100,200,300,400])
    ax.set_xticklabels(['-100', '0', '100', '200'], size = 16)
    ax.set_yticks([-1.5, -1, -0.5, 0])
    ax.set_yticklabels(list(map(str,ax.get_yticks())), size = 16)
    plt.tight_layout()
    plt.savefig('LFP during UP vs DOWN.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('LFP during UP vs DOWN.jpg', dpi = 1000, format = 'jpg')

    fig, ax = plt.subplots(figsize = (5,3))
    to_plot_DOWN_mean = np.nanmean(PSTH_matrix([0,1,2,3], stims=DOWN_stims, artifact_locs=artifacts).T[UP_DOWN_channels,:], axis = 0)*1000
    to_plot_DOWN_err = np.nanstd(PSTH_matrix([0,1,2,3], stims=DOWN_stims, artifact_locs=artifacts).T[UP_DOWN_channels,:], axis = 0)*1000
    to_plot_UP_mean = np.nanmean(PSTH_matrix([0,1,2,3], stims=UP_stims, artifact_locs=artifacts).T[UP_DOWN_channels,:], axis = 0)*1000
    to_plot_UP_err = np.nanstd(PSTH_matrix([0,1,2,3], stims=UP_stims, artifact_locs=artifacts).T[UP_DOWN_channels,:], axis = 0)*1000
    ax.plot(to_plot_DOWN_mean, color = 'k')
    ax.fill_between(np.linspace(0, 299, 299), to_plot_DOWN_mean + patch*to_plot_DOWN_err/np.sqrt(UP_DOWN_channels.size), to_plot_DOWN_mean - patch*to_plot_DOWN_err/np.sqrt(PSTH_resp_channels.size), alpha = 0.1, color = 'k')
    ax.plot(to_plot_UP_mean, color = 'r')
    ax.fill_between(np.linspace(0, 299, 299), to_plot_UP_mean + patch*to_plot_UP_err/np.sqrt(UP_DOWN_channels.size), to_plot_UP_mean - patch*to_plot_UP_err/np.sqrt(PSTH_resp_channels.size), alpha = 0.1, color = 'r')
    ax.set_xlim([50,250])
    ax.set_xlabel('time from stim (ms)', size = 16)
    ax.set_ylabel('MUA (Hz)', size = 16)
    ax.set_xticks([50,100,150,200,250])
    ax.set_xticklabels(['-50', '0', '50', '100', '150'], size = 16)
    ax.tick_params(axis="y", labelsize=16)    
    plt.tight_layout()
    plt.savefig('PSTH during UP vs DOWN.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('PSTH during UP vs DOWN.jpg', dpi = 1000, format = 'jpg')

    
    
    
    # ------------------------------------------------------------------------- timecourse of UP vs DOWN deliveries
    if day == '121121':
        to_plot_1_PSTH = [1,2,3]
    if day == '160426_D1':
        to_plot_1_PSTH = [1,2,3]
    if day == '281021':
        to_plot_1_PSTH = [1,2,3]
    else:
        to_plot_1_PSTH = [0,1,2,3]
    
    to_plot_2_PSTH = [4,5,6,7,8,9]
    
    #get LFP min, rel min, rel change of just UP and DOWN states (for overall plot)
    # LFP min peaks and std per sweep
    LFP_min_UP = np.empty([10, 64])
    LFP_min_UP[:] = np.NaN
    LFP_min_rel_UP = np.empty([10, 64])
    LFP_min_rel_UP[:] = np.NaN
    
    LFP_min_DOWN = np.empty([10, 64])
    LFP_min_DOWN[:] = np.NaN
    LFP_min_rel_DOWN = np.empty([10, 64])
    LFP_min_rel_DOWN[:] = np.NaN
    
    #get PSTH peak and amplitude and relative change of just UP and DOWN states (for overall plot)
    PSTH_peak_UP = np.empty([10, 64])
    PSTH_peak_UP[:] = np.NaN
    PSTH_peak_UP_rel = np.empty([10, 64])
    PSTH_peak_UP_rel[:] = np.NaN
    
    PSTH_magn_UP = np.empty([10, 64])
    PSTH_magn_UP[:] = np.NaN
    PSTH_magn_UP_rel = np.empty([10, 64])
    PSTH_magn_UP_rel[:] = np.NaN
    
    PSTH_peak_DOWN = np.empty([10, 64])
    PSTH_peak_DOWN[:] = np.NaN
    PSTH_peak_DOWN_rel = np.empty([10, 64])
    PSTH_peak_DOWN_rel[:] = np.NaN
    
    PSTH_magn_DOWN = np.empty([10, 64])
    PSTH_magn_DOWN[:] = np.NaN
    PSTH_magn_DOWN_rel = np.empty([10, 64])
    PSTH_magn_DOWN_rel[:] = np.NaN
    
    for sweep in range(len(LFP_all_sweeps)):
        LFP_min_UP[sweep,:] = np.abs(np.min(LFP_average([sweep], stims = UP_stims)[:,210:300], 1) - LFP_average([sweep], stims = UP_stims)[:,210])
        LFP_min_DOWN[sweep,:] = np.abs(np.min(LFP_average([sweep], stims = DOWN_stims)[:,210:300], 1) - LFP_average([sweep], stims = DOWN_stims)[:,210])
    
        # need to smooth psth for peak
        peak_UP = PSTH_matrix([sweep], stims = UP_stims, artifact_locs=artifacts)[100:180,:]
        peak_DOWN = PSTH_matrix([sweep], stims = DOWN_stims, artifact_locs=artifacts)[100:180,:]
        for chan in range(64):
            PSTH_peak_UP[sweep,chan] = np.max(smooth(peak_UP[:,chan], 6))
            PSTH_peak_DOWN[sweep,chan] = np.max(smooth(peak_DOWN[:,chan], 6))
        
        if day == '201121' or day == '160519_B2' or day == '281021':
            PSTH_magn_UP[sweep,:] = np.sum(PSTH_matrix([sweep], stims = UP_stims, artifact_locs=artifacts)[100:180,:], axis = 0) - np.sum(PSTH_matrix([sweep], stims = UP_stims, artifact_locs=artifacts)[50:90,:], axis = 0)
            PSTH_magn_DOWN[sweep,:] = np.sum(PSTH_matrix([sweep], stims = DOWN_stims, artifact_locs=artifacts)[100:180,:], axis = 0) - np.sum(PSTH_matrix([sweep], stims = DOWN_stims, artifact_locs=artifacts)[50:90,:], axis = 0)
        else:
            PSTH_magn_UP[sweep,:] = np.sum(PSTH_matrix([sweep], stims = UP_stims, artifact_locs=artifacts)[100:180,:], axis = 0)
            PSTH_magn_DOWN[sweep,:] = np.sum(PSTH_matrix([sweep], stims = DOWN_stims, artifact_locs=artifacts)[100:180,:], axis = 0)

    LFP_min_rel_UP = LFP_min_UP/np.nanmean(LFP_min_UP[to_plot_1_LFP,:], axis = 0)
    LFP_min_rel_change_UP = np.nanmean(LFP_min_rel_UP[to_plot_2_LFP,:], axis = 0) - np.nanmean(LFP_min_rel_UP[to_plot_1_LFP,:], axis = 0)
    
    LFP_min_rel_DOWN = LFP_min_DOWN/np.nanmean(LFP_min_DOWN[to_plot_1_LFP,:], axis = 0)
    LFP_min_rel_change_DOWN = np.nanmean(LFP_min_rel_DOWN[to_plot_2_LFP,:], axis = 0) - np.nanmean(LFP_min_rel_DOWN[to_plot_1_LFP,:], axis = 0)
        
    PSTH_resp_magn_UP_rel = np.transpose(np.asarray([PSTH_magn_UP[:,chan]/np.nanmean(PSTH_magn_UP[np.where(PSTH_magn_UP[to_plot_1_PSTH,chan] > 0)[0],chan]) for chan in range(64)]))
    PSTH_resp_magn_UP_rel_change = np.nanmean(PSTH_resp_magn_UP_rel[to_plot_2_PSTH,:], axis = 0) - np.nanmean(PSTH_resp_magn_UP_rel[to_plot_1_PSTH,:], axis = 0)
    
    PSTH_resp_magn_DOWN_rel = np.transpose(np.asarray([PSTH_magn_DOWN[:,chan]/np.nanmean(PSTH_magn_DOWN[np.where(PSTH_magn_DOWN[to_plot_1_PSTH,chan] > 0)[0],chan]) for chan in range(64)]))
    PSTH_resp_magn_DOWN_rel_change = np.nanmean(PSTH_resp_magn_DOWN_rel[to_plot_2_PSTH,:], axis = 0) - np.nanmean(PSTH_resp_magn_DOWN_rel[to_plot_1_PSTH,:], axis = 0)
    
    PSTH_resp_peak_UP_rel = np.transpose(np.asarray([PSTH_peak_UP[:,chan]/np.nanmean(PSTH_peak_UP[np.where(PSTH_peak_UP[to_plot_1_PSTH,chan] > 0)[0],chan]) for chan in range(64)]))
    PSTH_resp_peak_UP_rel_change = np.nanmean(PSTH_resp_peak_UP_rel[to_plot_2_PSTH,:], axis = 0) - np.nanmean(PSTH_resp_peak_UP_rel[to_plot_1_PSTH,:], axis = 0)
    
    PSTH_resp_peak_DOWN_rel = np.transpose(np.asarray([PSTH_peak_DOWN[:,chan]/np.nanmean(PSTH_peak_DOWN[np.where(PSTH_peak_DOWN[to_plot_1_PSTH,chan] > 0)[0],chan]) for chan in range(64)]))
    PSTH_resp_peak_DOWN_rel_change = np.nanmean(PSTH_resp_peak_DOWN_rel[to_plot_2_PSTH,:], axis = 0) - np.nanmean(PSTH_resp_peak_DOWN_rel[to_plot_1_PSTH,:], axis = 0)
    
    #average difference in responsive channels for UP vs DOWN, during baseline, as relative change
    # UP_DOWN_LFP_diff_baseline_rel = (np.mean(np.mean(LFP_min_UP[np.asarray(to_plot_1_LFP)[:,None], LFP_resp_channels], axis =0), axis = 0) - np.mean(np.mean(LFP_min_DOWN[np.asarray(to_plot_1_LFP)[:,None], LFP_resp_channels], axis =0), axis = 0))/np.mean(np.mean(LFP_min_DOWN[np.asarray(to_plot_1_LFP)[:,None], LFP_resp_channels], axis =0), axis = 0)
    # UP_DOWN_PSTH_magn_diff_baseline_rel = (np.mean(np.mean(PSTH_magn_UP[np.asarray(to_plot_1_PSTH)[:,None], PSTH_resp_channels], axis =0), axis = 0) - np.mean(np.mean(PSTH_magn_DOWN[np.asarray(to_plot_1_PSTH)[:,None], PSTH_resp_channels], axis =0), axis = 0))/np.mean(np.mean(PSTH_magn_DOWN[np.asarray(to_plot_1_PSTH)[:,None], PSTH_resp_channels], axis =0), axis = 0)
    # UP_DOWN_PSTH_peak_diff_baseline_rel = (np.mean(np.mean(PSTH_peak_UP[np.asarray(to_plot_1_PSTH)[:,None], PSTH_resp_channels], axis =0), axis = 0) - np.mean(np.mean(PSTH_peak_DOWN[np.asarray(to_plot_1_PSTH)[:,None], PSTH_resp_channels], axis =0), axis = 0))/np.mean(np.mean(PSTH_peak_DOWN[np.asarray(to_plot_1_PSTH)[:,None], PSTH_resp_channels], axis =0), axis = 0)
    
    # take out channels with artifacts that fuck up the average change
    # UP_DOWN_LFP_change_channels = LFP_resp_channels
    # UP_DOWN_LFP_change_channels = [59,35,13,61,63]
    # plot strength of response during UP/DOWN deliveries over time (as % of baseline response). Error bars across stimulations are probably messy as would have to get the individual stims and if there is noise or other it's impossible. Would have to try maybe though.
    # better to do error bars across channels or mice for the average plot.

# --------------------------------------------------------------- plot time course of UP vs DOWN stim strength

    if plot:
        fig, ax = plt.subplots(8,8)
        fig.suptitle('LFP response')
        for chan in range(64):                        
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(LFP_min_rel_DOWN[:,chan], 'b')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(LFP_min_rel_UP[:,chan], 'r')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
            if chan in LFP_resp_channels_cutoff:
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
                
        fig, ax = plt.subplots(8,8)
        fig.suptitle('PSTH peak')
        for chan in range(64):                        
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(PSTH_resp_peak_DOWN_rel[:,chan], 'b')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(PSTH_resp_peak_UP_rel[:,chan], 'r')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
            if chan in PSTH_resp_channels:
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
                
        fig, ax = plt.subplots(8,8)
        fig.suptitle('PSTH magn')
        for chan in range(64):                        
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(PSTH_resp_magn_DOWN_rel[:,chan], 'b')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(PSTH_resp_magn_UP_rel[:,chan], 'r')
            ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan))
            if chan in PSTH_resp_channels:
                ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")    
        
        



    # os.chdir(home_directory)
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    np.savetxt('UP_stims_freq.csv', UP_stims_freq, delimiter = ',')
    np.savetxt('LFP_min_UP.csv', LFP_min_UP, delimiter = ',')
    np.savetxt('LFP_min_DOWN.csv', LFP_min_DOWN, delimiter = ',')
    np.savetxt('LFP_min_rel_UP.csv', LFP_min_rel_UP, delimiter = ',')
    np.savetxt('LFP_min_rel_change_UP.csv', LFP_min_rel_change_UP, delimiter = ',')
    np.savetxt('LFP_min_rel_DOWN.csv', LFP_min_rel_DOWN, delimiter = ',')
    np.savetxt('LFP_min_rel_change_DOWN.csv', LFP_min_rel_change_DOWN, delimiter = ',')
    # np.save('UP_DOWN_LFP_diff_baseline_rel.npy', UP_DOWN_LFP_diff_baseline_rel)
    # np.save('UP_DOWN_LFP_change_channels.npy', UP_DOWN_LFP_change_channels)
    
    np.savetxt('PSTH_peak_UP.csv', PSTH_peak_UP, delimiter = ',')
    np.savetxt('PSTH_peak_DOWN.csv', PSTH_peak_DOWN, delimiter = ',')
    np.savetxt('PSTH_resp_peak_UP_rel.csv', PSTH_resp_peak_UP_rel, delimiter = ',')
    np.savetxt('PSTH_resp_peak_UP_rel_change.csv', PSTH_resp_peak_UP_rel_change, delimiter = ',')
    np.savetxt('PSTH_resp_peak_DOWN_rel.csv', PSTH_resp_peak_DOWN_rel, delimiter = ',')
    np.savetxt('PSTH_resp_peak_DOWN_rel_change.csv', PSTH_resp_peak_DOWN_rel_change, delimiter = ',')
    # np.save('UP_DOWN_PSTH_peak_diff_baseline_rel.npy', UP_DOWN_PSTH_peak_diff_baseline_rel)
    
    np.savetxt('PSTH_magn_UP.csv', PSTH_magn_UP, delimiter = ',')
    np.savetxt('PSTH_magn_DOWN.csv', PSTH_magn_DOWN, delimiter = ',')
    np.savetxt('PSTH_resp_magn_UP_rel.csv', PSTH_resp_magn_UP_rel, delimiter = ',')
    np.savetxt('PSTH_resp_magn_UP_rel_change.csv', PSTH_resp_magn_UP_rel_change, delimiter = ',')
    np.savetxt('PSTH_resp_magn_DOWN_rel.csv', PSTH_resp_magn_DOWN_rel, delimiter = ',')
    np.savetxt('PSTH_resp_magn_DOWN_rel_change.csv', PSTH_resp_magn_DOWN_rel_change, delimiter = ',')
    # np.save('UP_DOWN_PSTH_magn_diff_baseline_rel.npy', UP_DOWN_PSTH_magn_diff_baseline_rel)
    
    np.save('tolerance_before_1.npy', tolerance_before_1)
    np.save('tolerance_before_2.npy', tolerance_before_2)
    np.save('UP_DOWN_channels', UP_DOWN_channels)
    
    os.chdir('..')
    os.chdir('..')

    

#%% SW and delta power for no stim sweep

delta_lower = 0.5
delta_upper = 4

exclude_before = 0.1
exclude_after = 1.4
    
# for day in ['160310', '160414_D1', '160426_D1', '160519_B2', '160624_B2', '160628_D1', '221220_3']:
# for day in ['160218', '160308', '160322', '160331', '160420', '160427', '221213', '221216', '221219_1']:
# for day in ['221220_3']:
for day in ['221213', '221216', '221219_1']:

    os.chdir(day)
    if [i for i in os.listdir() if 'pairing_nowhisker'].__len__() == 0:
        os.chdir('..')
        continue
    
    # -------------------------------------------------------------- extract spikes and resample LFP if not done already --------------------------------------
    # LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    os.chdir([i for i in os.listdir() if 'pairing_nowhisker' in i][0])
    if os.path.exists('LFP_resampled') == False:
        j = 0
        channels = [s for s in os.listdir() if 'amp' in s and '.dat' in s]
        #figure out if some channels are not recorded from:
        if len(channels) < 63:
            chans_not_recorded = [chan for chan in list(range(64)) if chan not in list(map(int, [chan[-6:-4] for chan in channels]))]
            # for chan in chans_not_recorded:
            #     curr_spikes = {f'{chan}' :np.array([]) for chan in list(map(str, chans_not_recorded))}
        else:
            chans_not_recorded = []
        curr_spikes = {}
        
        for ind_channel, channel in enumerate(channels):
            print(channel)
            with open(channel,'rb') as f:
                curr_LFP = np.fromfile(f, np.int16)
                
                # take out spikes, from kilosort or no
                highpass = elephant.signal_processing.butter(curr_LFP,highpass_frequency = 250, sampling_frequency=30000)
                std = np.std(highpass)
                crossings = np.argwhere(highpass<-highpass_cutoff*std)
                # take out values within half a second of each other
                crossings = crossings[np.roll(crossings,-1) - crossings > 20]
                curr_spikes[channel[-6:-4]] = crossings/resample_factor
                
                #resample for LFP
                curr_LFP = scipy.signal.resample(curr_LFP, int(np.ceil(len(curr_LFP)/resample_factor)))
                
            if ind_channel == 0:
                LFP = copy.deepcopy(curr_LFP)
            elif ind_channel > 0:                
                LFP = np.vstack((LFP,curr_LFP))
                
            # add empty channels if not recorded
            if len(chans_not_recorded) > 0: 
                curr_LFP[:] = 0
                
            if ind_channel+1 > (len(channels) - 1) and int(channel[-6:-4]) == 63:
                continue
            
            elif ind_channel+1 > (len(channels) - 1) and int(channel[-6:-4]) < 63:
                for i in range(int(channels[ind_channel+1]) - 63):
                    LFP = np.vstack((LFP,curr_LFP))
                    curr_spikes[f'{chans_not_recorded[j]}'] = np.array([])
                    j += 1
            elif int(channels[ind_channel+1][-6:-4]) - int(channel[-6:-4]) > 1:
                for i in range(int(channels[ind_channel+1][-6:-4]) - int(channel[-6:-4]) - 1):
                    LFP = np.vstack((LFP,curr_LFP))
                    curr_spikes[f'{chans_not_recorded[j]}'] = np.array([])
                    j += 1
        
        pickle.dump(LFP, open('LFP_resampled','wb'))
        pickle.dump(curr_spikes, open('pairing_spikes','wb'))

    LFP = pickle.load(open('LFP_resampled','rb'))
    spikes = pickle.load(open('pairing_spikes','rb'))
       
    
    # CREATE ARTIFICAL STIMS
    stim_times_nostim = np.arange(1000, LFP.shape[1] - 1000, 5000)
    
    fftfreq = np.fft.fftfreq(int((5 - exclude_before - exclude_after)*new_fs), d = (1/new_fs))
    hanning_window = np.tile(np.hanning((5 - exclude_before - exclude_after)*new_fs), (64, 1))
    hamming_window = np.tile(np.hamming((5 - exclude_before - exclude_after)*new_fs), (64, 1))

    #IF DOING ON WHOLE RECORDING WITHOUT ARTIFICIAL STIMS
    # fftfreq = np.fft.fftfreq(LFP.shape[1], d = (1/new_fs))
    # hanning_window = np.tile(np.hanning(LFP.shape[1]), (64, 1))
    # hamming_window = np.tile(np.hamming(LFP.shape[1]), (64, 1))
    
    # FFT_current_sweep = np.fft.fft(hanning_window*LFP, axis = 1)
    # PSD = np.abs(FFT_current_sweep)**2 
    # delta_power = np.nanmean(PSD[:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 1)
    
    # fig, ax = plt.subplots()
    # ax.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]], smooth(np.mean(PSD[:,np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]], axis = 0), 1000), 'b')
    
    #EXCLUDE first and last stim just in case there isnt enough time, makes it easier
    FFT_current_sweep = np.zeros([len(stim_times_nostim - 2), 64, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
    FFT_current_sweep[:] = np.NaN
    for ind_stim, stim in enumerate(list(stim_times_nostim[1:-1])):
        if stim == 0:
            print(f'{ind_stim}: continue')
            continue
        # apply hamming (or hanning?) window first
        FFT_current_sweep[ind_stim,:,:] = np.fft.fft(hanning_window*LFP[:, int(stim+exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs)], axis = 1)

    PSD = np.nanmean(np.abs(FFT_current_sweep)**2, axis = 0) # average over stims

    delta_power = np.nanmean(PSD[:,np.where(np.logical_and(delta_lower <= fftfreq, delta_upper >= fftfreq))[0]], axis = 1)

    
    #the average you want as a numpy array to manipulate later on
    SW_waveform_sweeps = [[] for i in range(64)]
    SW_spiking_sweeps = [[] for i in range(64)]
    Peak_dur_sweeps = [[] for i in range(64)]
    SW_fslope_sweeps = [[] for i in range(64)]
    SW_sslope_sweeps = [[] for i in range(64)]
    SW_famp_sweeps = [[] for i in range(64)]
    SW_samp_sweeps = [[] for i in range(64)]
    UP_Cross_sweeps = [[] for i in range(64)]

    SW_frequency_sweeps_avg = np.zeros([64])
    SW_waveform_sweeps_avg = np.zeros([64, 1000])
    SW_spiking_sweeps_avg = np.zeros([64, 1000])
    Peak_dur_sweeps_avg = np.zeros([64])
    SW_fslope_sweeps_avg = np.zeros([64])
    SW_sslope_sweeps_avg = np.zeros([64])
    SW_famp_sweeps_avg = np.zeros([64])
    SW_samp_sweeps_avg = np.zeros([64])
    
    duration_criteria = 100
    UP_states_cutoff = 1.3 #how many std for definition of up state
    # filter in slow wave range, then find every time it goes under 2xSD i.e.=  upstate
    LFP_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 2*pq.Hz).as_array()
    #EXCLUDE first and last stim just in case there isnt enough time, makes it easier
    for ind_stim, stim in enumerate(list(stim_times_nostim[1:-1])):
        print(ind_stim)
        if stim == 0:
            continue
        curr_LFP_filt_total = LFP_filt[int(stim):int(stim + 5*new_fs), :]
        curr_LFP_filt = LFP_filt[int(stim + exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs), :]
        for chan in range(64):
            # print(chan)
            # because spiking is saved as dict of channels need to convert it to list to be able to access channels
            chan_spiking = list(spikes.values())[chan]
                
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
            
            UP_LFP = np.where(curr_LFP_filt[:,chan] < -UP_states_cutoff*np.std(curr_LFP_filt[:,chan]))[0]
            
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
                UP_Cross_sweeps[chan].append(UP_Cross_after[i] + int(stim + exclude_after*new_fs))
                
                Peak_dur_sweeps[chan].append(DOWN_Cross_after[i] - DOWN_Cross_before[i])
                
                #save filtered LFP
                SW_waveform_sweeps[chan].append(curr_LFP_filt_total[int(UP_Cross_after[i] - 0.5*new_fs + exclude_after*new_fs) : int(UP_Cross_after[i] + 0.5*new_fs + exclude_after*new_fs), chan])
                
                #save spiking (as 1ms bins)
                temp_spiking = np.zeros(1000)
                # set all spikes there as 1. So take out spikes within 500ms of UP crossing, then subtract 500ms before UP crossing to start at 0
                temp_spiking[np.round(chan_spiking[np.logical_and(int(UP_Cross_after[i] + exclude_after*new_fs + stim - 0.5*new_fs) < chan_spiking, int(UP_Cross_after[i] + exclude_after*new_fs + stim + 0.5*new_fs) > chan_spiking)] - int(UP_Cross_after[i] + exclude_after*new_fs + stim - 0.5*new_fs) - 1).astype(int)] = 1
                SW_spiking_sweeps[chan].append(temp_spiking)
                
                idx_peak = np.argmax(curr_LFP_filt[UP_Cross_after[i]:DOWN_Cross_after[i],chan])
                idx_trough = np.argmin(curr_LFP_filt[DOWN_Cross_before[i]:UP_Cross_after[i],chan])
                
                SW_fslope_sweeps[chan].append(np.mean(np.diff(curr_LFP_filt[DOWN_Cross_before[i]:DOWN_Cross_before[i] + idx_trough])))
                SW_sslope_sweeps[chan].append(np.mean(np.diff(curr_LFP_filt[DOWN_Cross_before[i] + idx_trough:UP_Cross_after[i]+idx_peak, chan])))
                
                SW_famp_sweeps[chan].append(np.abs(min(curr_LFP_filt[DOWN_Cross_before[i]:UP_Cross_after[i],chan])))
                SW_samp_sweeps[chan].append(np.abs(max(curr_LFP_filt[UP_Cross_after[i]:DOWN_Cross_after[i],chan])))
    
    
    
    # if doing it without the artificial stims:
    # for chan in range(64):
    #     print(chan)
    #     # because spiking is saved as dict of channels need to convert it to list to be able to access channels
    #     chan_spiking = list(spikes.values())[chan]
            
    #     # print(chan)
    #     DOWN_Cross = np.where(np.diff((LFP_filt[:,chan] < 0).astype(int)) == 1)[0]
    #     UP_Cross = np.where(np.diff((LFP_filt[:,chan] < 0).astype(int)) == -1)[0]
        
    #     if DOWN_Cross.size == 0:
    #         continue
        
    #     #if no Down crossing before or after:
    #     UP_Cross = np.delete(UP_Cross, UP_Cross < DOWN_Cross[0])
    #     UP_Cross = np.delete(UP_Cross, UP_Cross > DOWN_Cross[-1])
        
    #     # If too close to end (need to be able to take out 500ms after for waveform)
    #     UP_Cross = np.delete(UP_Cross, UP_Cross > (LFP_filt.shape[0] - 1500))
        
    #     if UP_Cross.size == 0:
    #         continue
        
    #     #define UP states
    #     UP_LFP = np.where(LFP_filt[:,chan] < -UP_states_cutoff*np.std(LFP_filt[:,chan]))[0]
        
    #     # If no UP crossing after
    #     UP_LFP = np.delete(UP_LFP, UP_LFP > UP_Cross[-1])
        
    #     # only LFP points within 500ms of a UP Crossing afterwards
    #     for i in range(len(UP_LFP)):
    #        diff_to_crossing = UP_Cross - UP_LFP[i]
    #        if min(diff_to_crossing[diff_to_crossing > 0]) > 249:
    #            UP_LFP[i] = 0
    #     UP_LFP = np.delete(UP_LFP, UP_LFP == 0)
        
    #     #if no DOWN crossing before
    #     UP_LFP = np.delete(UP_LFP, UP_LFP < DOWN_Cross[0])
        
    #     #take out continuous numbers, so just left with first one that fits all the criteria
    #     UP_LFP = np.delete(UP_LFP, np.where(np.diff(UP_LFP) == 1)[0] + 1)

    #     if UP_LFP.size == 0:
    #         continue
        
    #     DOWN_Cross_before = []
    #     UP_Cross_after = []
    #     DOWN_Cross_after = []
        
    #     #find the Crossings before and after. Here also apply the duration criteria: Down before and UP after must be separated by at least 100ms (so 100ms UP state duration)
    #     for i in range(len(UP_LFP)):
    #         idx_down = np.argmin(np.abs(UP_LFP[i] - DOWN_Cross))
    #         idx_up = np.argmin(np.abs(UP_LFP[i] - UP_Cross))
            
    #         if DOWN_Cross[idx_down] < UP_LFP [i]:
    #             curr_DOWN_Cross_before = DOWN_Cross[idx_down]
    #             curr_DOWN_Cross_after = DOWN_Cross[idx_down + 1]

    #         elif DOWN_Cross[idx_down] > UP_LFP [i]:
    #             curr_DOWN_Cross_before = DOWN_Cross[idx_down - 1]
    #             curr_DOWN_Cross_after = DOWN_Cross[idx_down]
                
    #         if UP_Cross[idx_up] > UP_LFP[i]:
    #             curr_UP_Cross_after = UP_Cross[idx_up]
    #         elif UP_Cross[idx_up] < UP_LFP[i]:
    #             curr_UP_Cross_after = UP_Cross[idx_up + 1]
            
    #         # duration criteria:
    #         if curr_UP_Cross_after - curr_DOWN_Cross_before < duration_criteria:
    #             continue
    #         else:
    #             DOWN_Cross_before.append(curr_DOWN_Cross_before)
    #             DOWN_Cross_after.append(curr_DOWN_Cross_after)
    #             UP_Cross_after.append(curr_UP_Cross_after)
            
            
    #         # save UP_Cross_after in list of lists to get the spiking with the slow wave. remember UP_cross after is indexed with a 0.5s offset from stim start.
    #         UP_Cross_sweeps[chan].append(UP_Cross_after[i])
            
    #         Peak_dur_sweeps[chan].append(DOWN_Cross_after[i] - DOWN_Cross_before[i])
            
    #         #save filtered LFP
    #         SW_waveform_sweeps[chan].append(LFP_filt[int(UP_Cross_after[i] - 0.5*new_fs) : int(UP_Cross_after[i] + 0.5*new_fs), chan])
            
    #         #save spiking (as 1ms bins)
    #         temp_spiking = np.zeros(1000)
    #         # set all spikes there as 1. So take out spikes within 500ms of UP crossing, then subtract 500ms before UP crossing to start at 0
    #         temp_spiking[np.round(chan_spiking[np.logical_and(int(UP_Cross_after[i] - 0.5*new_fs) < chan_spiking, int(UP_Cross_after[i] + 0.5*new_fs) > chan_spiking)] - UP_Cross_after[i] - 1).astype(int)] = 1
    #         SW_spiking_sweeps[chan].append(temp_spiking)
            
    #         idx_peak = np.argmax(LFP_filt[UP_Cross_after[i]:DOWN_Cross_after[i],chan])
    #         idx_trough = np.argmin(LFP_filt[DOWN_Cross_before[i]:UP_Cross_after[i],chan])
            
    #         SW_fslope_sweeps[chan].append(np.mean(np.diff(LFP_filt[DOWN_Cross_before[i]:DOWN_Cross_before[i] + idx_trough])))
    #         SW_sslope_sweeps[chan].append(np.mean(np.diff(LFP_filt[DOWN_Cross_before[i] + idx_trough:UP_Cross_after[i]+idx_peak, chan])))
            
    #         SW_famp_sweeps[chan].append(np.abs(min(LFP_filt[DOWN_Cross_before[i]:UP_Cross_after[i],chan])))
    #         SW_samp_sweeps[chan].append(np.abs(max(LFP_filt[UP_Cross_after[i]:DOWN_Cross_after[i],chan])))
            
            
# average over stims, so 1 value per sweep. MAYBE ALSO TAKE OUT OUTLIERS?? so  maybe first concatenate before and after, take out outliers and average then 
    for chan in range(64):
        SW_waveform_sweeps_avg[chan,:] = np.mean(np.asarray([i for i in SW_waveform_sweeps[chan] if i.size == 1000]), axis = 0)
        SW_frequency_sweeps_avg[chan] = len(Peak_dur_sweeps[chan])/(len(stim_times_nostim) - 2) # -2 because exclude first and last stim
        SW_spiking_sweeps_avg[chan,:] = np.mean(np.asarray(SW_spiking_sweeps[chan]), axis = 0)
        Peak_dur_sweeps_avg[chan] = np.mean(np.asarray(Peak_dur_sweeps[chan]))
        SW_fslope_sweeps_avg[chan] = np.mean(np.asarray(SW_fslope_sweeps[chan]))
        SW_sslope_sweeps_avg[chan] = np.mean(np.asarray(SW_sslope_sweeps[chan]))
        SW_famp_sweeps_avg[chan] = np.mean(np.asarray(SW_famp_sweeps[chan]))
        SW_samp_sweeps_avg[chan] = np.mean(np.asarray(SW_samp_sweeps[chan]))

    os.chdir('..')
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    np.save('delta_power_nostim.npy', delta_power)
    np.save('PSD_nostim.npy', PSD)
    np.save('fftfreq_nostim.npy', PSD)

    np.save('SW_waveform_sweeps_avg_nostim.npy', SW_waveform_sweeps_avg)
    np.save('SW_frequency_sweeps_avg_nostim.npy', SW_frequency_sweeps_avg)
    np.save('SW_spiking_sweeps_avg_nostim.npy', SW_spiking_sweeps_avg)
    np.save('Peak_dur_sweeps_avg_nostim.npy', Peak_dur_sweeps_avg)
    np.save('SW_fslope_sweeps_avg_nostim.npy', SW_fslope_sweeps_avg)
    np.save('SW_sslope_sweeps_avg_nostim.npy', SW_sslope_sweeps_avg)
    np.save('SW_famp_sweeps_avg_nostim.npy', SW_famp_sweeps_avg)
    np.save('SW_samp_sweeps_avg_nostim.npy', SW_samp_sweeps_avg)


    #redo values with the mean waveforms:
    Peak_dur_sweeps_avg_overall = np.zeros([64])
    SW_fslope_sweeps_avg_overall = np.zeros([64])
    SW_sslope_sweeps_avg_overall = np.zeros([64])
    SW_famp_sweeps_avg_overall = np.zeros([64])
    SW_samp_sweeps_avg_overall = np.zeros([64])
    

    for chan in range(64):
        Peak_dur_sweeps_avg_overall[chan] = np.argmax(SW_waveform_sweeps_avg[chan,:]) - np.argmin(SW_waveform_sweeps_avg[chan,:])
        SW_fslope_sweeps_avg_overall[chan] = np.nanmean(np.diff(SW_waveform_sweeps_avg[chan,250:np.argmin(SW_waveform_sweeps_avg[chan,:])]))
        SW_sslope_sweeps_avg_overall[chan] = np.nanmean(np.diff(SW_waveform_sweeps_avg[chan,np.argmin(SW_waveform_sweeps_avg[chan,:500]):np.argmax(SW_waveform_sweeps_avg[chan,500:])+500]))
        SW_famp_sweeps_avg_overall[chan] = np.min(SW_waveform_sweeps_avg[chan,:])
        SW_samp_sweeps_avg_overall[chan] = np.max(SW_waveform_sweeps_avg[chan,:])
    
    np.save('Peak_dur_sweeps_avg_overall_nostim.npy', Peak_dur_sweeps_avg_overall)
    np.save('SW_fslope_sweeps_avg_overall_nostim.npy', SW_fslope_sweeps_avg_overall)
    np.save('SW_sslope_sweeps_avg_overall_nostim.npy', SW_sslope_sweeps_avg_overall)
    np.save('SW_famp_sweeps_avg_overall_nostim.npy', SW_famp_sweeps_avg_overall)
    np.save('SW_samp_sweeps_avg_overall_nostim.npy', SW_samp_sweeps_avg_overall)
    
    os.chdir('..')
    os.chdir('..')
            


#%% EXAMPLE slow waves before and after all chans for figure
to_plot_1_SW = [0,1,2,3]
to_plot_2_SW = [4,5,6,7,8,9]

SW_waveform_sweeps_avg = np.load('SW_waveform_sweeps_avg.npy')
LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',', dtype = int)
#average waveforms before vs after on same axis
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

#average waveforms before vs after on same axis
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





patch = 2 # how many times standard deviation
SW_spiking_avg = np.load('SW_spiking_sweeps_avg.npy')
SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',', dtype = int)
#average waveforms before vs after on same axis
fig, ax = plt.subplots()
to_plot_before = np.mean(SW_spiking_avg[to_plot_1_SW,:,:], axis = 0)*1000
ax.plot(np.mean(to_plot_before[SW_spiking_channels,:], axis = 0), color = 'k')
ax.fill_between(list(range(SW_spiking_avg.shape[2])), np.mean(to_plot_before[SW_spiking_channels,:], axis = 0) + patch*np.nanstd(to_plot_before[SW_spiking_channels,:], axis = 0), np.mean(to_plot_before[SW_spiking_channels,:], axis = 0) - patch*np.nanstd(to_plot_before[SW_spiking_channels,:], axis = 0), alpha = 0.1, color = 'k')
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
plt.savefig('SW spiking before.pdf', format = 'pdf')
plt.savefig('SW spiking before.jpg', format = 'jpg')

fig, ax = plt.subplots()
to_plot_after = np.mean(SW_spiking_avg[to_plot_2_SW,:,:], axis = 0)*1000
ax.plot(np.mean(to_plot_after[SW_spiking_channels,:], axis = 0), color = 'k')
ax.fill_between(list(range(SW_spiking_avg.shape[2])), np.mean(to_plot_after[SW_spiking_channels,:], axis = 0) + patch*np.nanstd(to_plot_after[SW_spiking_channels,:], axis = 0), np.mean(to_plot_after[SW_spiking_channels,:], axis = 0) - patch*np.nanstd(to_plot_after[SW_spiking_channels,:], axis = 0), alpha = 0.1, color = 'k')
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
plt.savefig('SW spiking after.pdf', format = 'pdf')
plt.savefig('SW spiking after.jpg', format = 'jpg')






#average SW waveform for slope and amplitude example
fig, ax = plt.subplots(figsize = (8,3))
ax.plot(np.mean(SW_waveform_sweeps_avg[1,LFP_resp_channels_cutoff,:], axis = 0), color = 'k')
ax.set_ylim([-400,400])
ax.set_yticks([-250,0,250])
ax.set_yticklabels(list(map(str, list(ax.get_yticks()))), size = 16)
ax.set_ylabel('Amplitude (yV)', size = 16)
ax.set_xticks([0,500,1000])
ax.set_xticklabels(list(map(str,[-500,0,500])), size = 16)
ax.set_xlabel('time from SW onset (ms)', size = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('SW example.pdf', format = 'pdf')
plt.savefig('SW example.jpg', format = 'jpg')


