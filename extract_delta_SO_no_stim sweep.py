# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 19:43:25 2023

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


#%% estract SW parameters and delta power for no stim sweep

delta_lower = 0.5
delta_upper = 4

exclude_before = 0.1
exclude_after = 1.4

duration_criteria = 100
UP_states_cutoff = 1.75 #how many std for definition of up state

zero_mean = False

# for day in ['160310', '160414_D1', '160426_D1', '160519_B2', '160624_B2', '160628_D1', '221220_3']:
# for day in ['160218', '160308', '160322', '160331', '160420', '160427', '221213', '221216', '221219_1']:
# for day in ['221220_3']:
# for day in ['221213', '221216', '221219_1']:
    
days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]
for day in days:
    os.chdir(day)
    if [i for i in os.listdir() if 'pairing_nowhisker' in i].__len__() == 0:
        print(f'{day} doesnt have nowhisker')
        os.chdir('..')
        continue
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_resp_channels_cutoff = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',').astype(int)
    os.chdir('..')

    # -------------------------------------------------------------- extract spikes and resample LFP if not done already --------------------------------------
    # LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
    print(day)
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
       
    
    
    
    # --------------------------------------------------------------------- DELTA POWER ANALYSIS -------------------------------------------------------
    
    #IF DOING ON WHOLE RECORDING WITHOUT ARTIFICIAL STIMS
    # fftfreq = np.fft.fftfreq(LFP.shape[1], d = (1/new_fs))
    # hanning_window = np.tile(np.hanning(LFP.shape[1]), (64, 1))
    # hamming_window = np.tile(np.hamming(LFP.shape[1]), (64, 1))
    
    # FFT_current_sweep = np.fft.fft(hanning_window*LFP, axis = 1)
    # PSD = np.abs(FFT_current_sweep)**2 
    # delta_power = np.nanmean(PSD[:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]], axis = 1)
    
    # fig, ax = plt.subplots() # PLOTTING PSD
    # ax.plot(fftfreq[np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]], smooth(np.mean(PSD[:,np.where(np.logical_and(0.1 <= fftfreq , 50 >= fftfreq))[0]], axis = 0), 1000), 'b')


    # CREATE ARTIFICAL STIMS
    stim_times_nostim = np.arange(1000, LFP.shape[1] - 1000, 5000)
    
    fftfreq = np.fft.fftfreq(int((5 - exclude_before - exclude_after)*new_fs), d = (1/new_fs))
    hanning_window = np.tile(np.hanning((5 - exclude_before - exclude_after)*new_fs), (64, 1))
    hamming_window = np.tile(np.hamming((5 - exclude_before - exclude_after)*new_fs), (64, 1))

    FFT_current_sweep = np.zeros([len(stim_times_nostim), 64, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
    FFT_current_sweep[:] = np.NaN
    FFT_current_sweep_auto_outliers = np.zeros([len(stim_times_nostim), 64, int((5 - exclude_before - exclude_after)*new_fs)], dtype = complex)
    FFT_current_sweep_auto_outliers[:] = np.NaN

    all_stims_delta = np.zeros([64, len(stim_times_nostim)])
    all_stims_delta_auto_outliers = np.zeros([64, len(stim_times_nostim)])

    auto_outlier_stims_indices = []
    
    for ind_stim, stim in enumerate(list(stim_times_nostim)):
        
        if ind_stim == 0 or ind_stim == len(stim_times_nostim) - 1:
            continue
        
        # apply hamming window first
        FFT_current_sweep[ind_stim,:,:] = np.fft.fft(hanning_window*LFP[:, int(stim+exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs)], axis = 1)
        FFT_current_sweep_auto_outliers[ind_stim,:,:] = np.fft.fft(hanning_window*LFP[:, int(stim+exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs)], axis = 1)
        
        all_stims_delta[:,ind_stim] = np.transpose(np.nanmean(np.abs(FFT_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]]**2), axis = 0))
        all_stims_delta_auto_outliers[:,ind_stim] = np.transpose(np.nanmean(np.abs(FFT_current_sweep[ind_stim,:,np.where(np.logical_and(delta_lower <= fftfreq , delta_upper >= fftfreq))[0]]**2), axis = 0))

    # define auto outlier stims as exceeding outlier threshold within each sweep in each LFP responsive channel
    for chan in range(64):
        curr_delta = all_stims_delta[chan, :]
        outliers = (curr_delta > (np.percentile(curr_delta, 75) + 1.5*(np.abs(np.percentile(curr_delta, 75) - np.percentile(curr_delta, 25)))))
        if len(np.where(outliers == True)[0]) > 0:
            all_stims_delta_auto_outliers[chan, np.where(outliers == True)[0]] = 0
            FFT_current_sweep_auto_outliers[np.where(outliers == True)[0],:,:] = np.NaN
        auto_outlier_stims_indices.append(np.where(outliers == True)[0])

    PSD = np.nanmean(np.abs(FFT_current_sweep)**2, axis = 0) # average over stims
    delta_power = np.nanmean(PSD[:,np.where(np.logical_and(delta_lower <= fftfreq, delta_upper >= fftfreq))[0]], axis = 1)

    PSD_auto_outliers = np.nanmean(np.abs(FFT_current_sweep_auto_outliers)**2, axis = 0) # average over stims
    delta_power_auto_outliers = np.nanmean(PSD_auto_outliers[:,np.where(np.logical_and(delta_lower <= fftfreq, delta_upper >= fftfreq))[0]], axis = 1)
    PSD_median_auto_outliers = np.nanmedian(np.abs(FFT_current_sweep_auto_outliers)**2, axis = 0) # average over stims
    delta_power_median_auto_outliers = np.nanmean(PSD_median_auto_outliers[:,np.where(np.logical_and(delta_lower <= fftfreq, delta_upper >= fftfreq))[0]], axis = 1)

    
    #delta power timecourse over whole recording
    fig, ax = plt.subplots(8,8, figsize = (12,10)) 
    fig.suptitle(f'delta in all stims {day}')
    for ind, ax1 in enumerate(list(ax.flatten())):                        
        ax1.plot(all_stims_delta_auto_outliers[chanMap[ind],:], linewidth = 1)
        # ax1.axhline(450000, linestyle = '--')
        if chanMap[ind] in LFP_resp_channels_cutoff:
            ax1.set_facecolor("y")
        ax1.set_yticks([])
        ax1.set_xticks([])
        ax1.set_title(str(chanMap[ind]), size = 4)
    plt.savefig('delta power auto outliers no y-share', dpi = 1000)

    
    
    # ---------------------------------------------------------------- SLOW WAVE EXTRACTION ---------------------------------------------------------

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
    
    SW_waveform_sweeps_median = np.zeros([64, 1000])
    SW_spiking_sweeps_median = np.zeros([64, 1000])
    Peak_dur_sweeps_median = np.zeros([64])
    SW_fslope_sweeps_median = np.zeros([64])
    SW_sslope_sweeps_median = np.zeros([64])
    SW_famp_sweeps_median = np.zeros([64])
    SW_samp_sweeps_median = np.zeros([64])
    
    # filter in slow wave range, then find every time it goes under 2xSD i.e.=  upstate
    LFP_filt = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 2*pq.Hz).as_array()
    
    for ind_stim, stim in enumerate(list(stim_times_nostim)):
        print(ind_stim)
        
        if ind_stim == 0 or ind_stim == len(stim_times_nostim) - 1:
            continue
        
        curr_LFP_filt_total = LFP_filt[int(stim):int(stim + 5*new_fs), :]
        curr_LFP_filt = LFP_filt[int(stim + exclude_after*new_fs):int(stim+(5 - exclude_before)*new_fs), :]
        
        for chan in range(64):

            # if detected as outlier in delta power analysis
            if ind_stim in auto_outlier_stims_indices[chan]:
                continue
            
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
                
                SW_fslope_sweeps[chan].append(np.mean(np.diff(curr_LFP_filt[DOWN_Cross_before[i]:DOWN_Cross_before[i] + idx_trough, chan])))
                SW_sslope_sweeps[chan].append(np.mean(np.diff(curr_LFP_filt[DOWN_Cross_before[i] + idx_trough:UP_Cross_after[i]+idx_peak, chan])))
                
                SW_famp_sweeps[chan].append(np.abs(min(curr_LFP_filt[DOWN_Cross_before[i]:UP_Cross_after[i],chan])))
                SW_samp_sweeps[chan].append(np.abs(max(curr_LFP_filt[UP_Cross_after[i]:DOWN_Cross_after[i],chan])))
    
    
                
    # average over stims, so 1 value per sweep
    for chan in range(64):
        SW_frequency_sweeps_avg[chan] = len(Peak_dur_sweeps[chan])/(len(stim_times_nostim) - 2 - len(auto_outlier_stims_indices[chan])) 

        SW_waveform_sweeps_avg[chan,:] = np.mean(np.asarray([i for i in SW_waveform_sweeps[chan] if i.size == 1000]), axis = 0)
        SW_spiking_sweeps_avg[chan,:] = np.mean(np.asarray(SW_spiking_sweeps[chan]), axis = 0)
        Peak_dur_sweeps_avg[chan] = np.mean(np.asarray(Peak_dur_sweeps[chan]))
        SW_fslope_sweeps_avg[chan] = np.mean(np.asarray(SW_fslope_sweeps[chan]))
        SW_sslope_sweeps_avg[chan] = np.mean(np.asarray(SW_sslope_sweeps[chan]))
        SW_famp_sweeps_avg[chan] = np.mean(np.asarray(SW_famp_sweeps[chan]))
        SW_samp_sweeps_avg[chan] = np.mean(np.asarray(SW_samp_sweeps[chan]))

        SW_waveform_sweeps_median[chan,:] = np.median(np.asarray([i for i in SW_waveform_sweeps[chan] if i.size == 1000]), axis = 0)
        SW_spiking_sweeps_median[chan,:] = np.median(np.asarray(SW_spiking_sweeps[chan]), axis = 0)
        Peak_dur_sweeps_median[chan] = np.median(np.asarray(Peak_dur_sweeps[chan]))
        SW_fslope_sweeps_median[chan] = np.median(np.asarray(SW_fslope_sweeps[chan]))
        SW_sslope_sweeps_median[chan] = np.median(np.asarray(SW_sslope_sweeps[chan]))
        SW_famp_sweeps_median[chan] = np.median(np.asarray(SW_famp_sweeps[chan]))
        SW_samp_sweeps_median[chan] = np.median(np.asarray(SW_samp_sweeps[chan]))


    # fig, ax = plt.subplots(8,8, figsize = (15,12))
    # fig.suptitle(f'Slow Waves')
    # for ind, ax1 in enumerate(list(ax.flatten())):  
    #     ax1.tick_params(axis='both', which='minor', labelsize=4)
    #     ax1.tick_params(axis='both', which='major', labelsize=4)
    #     ax1.set_xticks([])
    #     if chanMap[ind] in LFP_resp_channels_cutoff:                     
    #         ax1.plot(np.asarray(SW_waveform_sweeps[chanMap[ind]]).T, linewidth = 0.45)
    #         ax1.set_title(str(chanMap[ind]), size = 5)
    # plt.tight_layout()
    # plt.savefig(f'Slow Waves no stim', dpi = 1000)
    # cl()

    os.chdir('..')
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
    np.save('delta_power_nostim.npy', delta_power)
    np.save('delta_power_auto_outliers_nostim.npy', delta_power_auto_outliers)
    np.save('delta_power_median_auto_outliers_nostim.npy', delta_power_median_auto_outliers)

    np.save('PSD_nostim.npy', PSD)
    np.save('fftfreq_nostim.npy', PSD)

    np.save('SW_frequency_sweeps_avg_nostim.npy', SW_frequency_sweeps_avg)
    
    np.save('SW_waveform_sweeps_avg_nostim.npy', SW_waveform_sweeps_avg)
    np.save('SW_spiking_sweeps_avg_nostim.npy', SW_spiking_sweeps_avg)
    np.save('Peak_dur_sweeps_avg_nostim.npy', Peak_dur_sweeps_avg)
    np.save('SW_fslope_sweeps_avg_nostim.npy', SW_fslope_sweeps_avg)
    np.save('SW_sslope_sweeps_avg_nostim.npy', SW_sslope_sweeps_avg)
    np.save('SW_famp_sweeps_avg_nostim.npy', SW_famp_sweeps_avg)
    np.save('SW_samp_sweeps_avg_nostim.npy', SW_samp_sweeps_avg)

    np.save('SW_waveform_sweeps_median_nostim.npy', SW_waveform_sweeps_median)
    np.save('SW_spiking_sweeps_median_nostim.npy', SW_spiking_sweeps_median)
    np.save('Peak_dur_sweeps_median_nostim.npy', Peak_dur_sweeps_median)
    np.save('SW_fslope_sweeps_median_nostim.npy', SW_fslope_sweeps_median)
    np.save('SW_sslope_sweeps_median_nostim.npy', SW_sslope_sweeps_median)
    np.save('SW_famp_sweeps_median_nostim.npy', SW_famp_sweeps_median)
    np.save('SW_samp_sweeps_median_nostim.npy', SW_samp_sweeps_median)



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


#%%
