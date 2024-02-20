# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:07:16 2023

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

#%% PSTH

# --------------------------------------------------------------------------------- PSTH ------------------------------------------------------------------------------------------------------------
highpass_cutoff = 4
smooth_over = 10
# PSTH_resp_channels = LFP_resp_channels
# PSTH_resp_channels = [37,35,13,33,15,63,62]
# PSTH_resp_channels = [27,43,25,41,23,39,21,37,19,35,17]

# artifacts = [96,97,98,99,100]

# os.chdir(home_directory)
reanalyze = True
plot = False

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i) and 'old' not in i]
# for day in ['221220_2']:
for day in days:
    os.chdir(day) 
    print(day)
    if day in ['160310', '160414_D1', '160426_D1', '160519_B2', '160624_B2', '160628_D1', '160128', '160202', '160218', '160308', '160322', '160331', '160420', '160427']:        
        artifacts = []
    else:
        # artifacts = []
        artifacts = list(np.linspace(97,123,27,dtype = int))                 
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    
    # if day in ['221220_3']:
    #     highpass_cutoff = 5 # last sweep gets a bit noisy which reduces channel detection and increases STTC
    # else:
    #     highpass_cutoff = 4
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

    # calculate 
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

    # os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    if os.path.exists(fr'analysis_{day}\PSTH_resp_channels.csv') and reanalyze == False:
        os.chdir(f'analysis_{day}')
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
    
    # os.chdir('..')
    

    if plot:
        # os.chdir(home_directory)
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
            ax1.axvline(3)
        plt.savefig(f'Spiking peak all chans', dpi = 1000)

        fig, ax = plt.subplots(8,8) 
        fig.suptitle('PSTH timecourse magn')
        for ind, ax1 in enumerate(list(ax.flatten())):                        
            ax1.plot(PSTH_resp_magn_rel[:,chanMap[ind]])
            if chanMap[ind] in PSTH_resp_channels:
                ax1.set_facecolor("y")
            ax1.set_title(str(chanMap[ind]))
            ax1.axvline(3)
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

        # #change over time in all channels (relative PSTH_magn)
        # fig, ax = plt.subplots(8,8) 
        # fig.suptitle('PSTH timecourse magn')
        # for ind, ax1 in enumerate(list(ax.flatten())):                        
        #     ax1.plot(PSTH_resp_magn_rel[:,chanMap[ind]])
        #     if chanMap[ind] in PSTH_resp_channels:
        #         ax1.set_facecolor("y")
        #     ax1.set_title(str(chanMap[ind]))
        #     ax1.axvline(3)
        
        
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
    
    
    # os.chdir(home_directory)
    # example colorplot of PSTH min magnitude
    # fig, ax = plt.subplots()
    # ax.imshow(np.reshape(PSTH_resp_magn[4,chanMap], (8, 8)), cmap = 'Blues')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.savefig('PSTH max colormap.pdf', dpi = 1000, format = 'pdf')
    # plt.savefig('PSTH max colormap.jpg', dpi = 1000, format = 'jpg')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    
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

    plot_before = (PSTH_matrix(to_plot_1_PSTH, artifact_locs = artifacts)*1000).T # *1000 for instantaneous spike rate
    plot_after = (PSTH_matrix(to_plot_2_PSTH, artifact_locs = artifacts)*1000).T
    fig, ax = plt.subplots(8,8, sharey = True, constrained_layout = True)
    for ind, ax1 in enumerate(list(ax.flatten())):
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        if chanMap[ind] in PSTH_resp_channels:
            ax1.plot(smooth(plot_before[chanMap[ind],50:200], 1),'b', linewidth = 1)
            ax1.plot(smooth(plot_after[chanMap[ind],50:200], 1),'r', linewidth = 1)
    plt.savefig('overall PSTH before and after chanMap.pdf', dpi = 1000, format = 'pdf')
    plt.savefig('overall PSTH before and after chanMap.jpg', dpi = 1000, format = 'jpg')


    # os.chdir(home_directory)
    # os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    # np.savetxt('PSTH_resp_channels.csv', PSTH_resp_channels, delimiter = ',')
    # np.savetxt('PSTH_resp_magn.csv', PSTH_resp_magn, delimiter = ',')
    # np.savetxt('PSTH_resp_peak.csv', PSTH_resp_peak, delimiter = ',')
    # np.savetxt('PSTH_resp_magn_rel.csv', PSTH_resp_magn_rel, delimiter = ',')
    # np.savetxt('PSTH_resp_peak_rel.csv', PSTH_resp_peak_rel, delimiter = ',')
    # np.savetxt('PSTH_resp_magn_rel_change.csv', PSTH_resp_magn_rel_change, delimiter = ',')
    # np.savetxt('PSTH_resp_peak_rel_change.csv', PSTH_resp_peak_rel_change, delimiter = ',')
    # np.savetxt('to_plot_1_PSTH.csv', to_plot_1_PSTH, delimiter = ',')
    # np.savetxt('to_plot_2_PSTH.csv', to_plot_2_PSTH, delimiter = ',')
    # np.savetxt('PSTH_artifacts.csv', np.asarray(artifacts))
    # np.save('PSTH_responses.npy', PSTH_responses)
    # os.chdir('..')
    
    os.chdir('..')
    
    cl()
    

