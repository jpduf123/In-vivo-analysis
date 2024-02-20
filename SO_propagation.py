# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:18:51 2022

@author: Mann Lab
"""
# matplotlib 3.4.2 originally

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
# import natsort
from statistics import mean
# import xml.etree.ElementTree as ET
# from load_intan_rhd_format import *
# from operator import itemgetter
# import pandas as pd
from scipy.linalg import lstsq
import pycircstat
import matplotlib.colors as colors

overall_path = r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS'
# overall_path = r'C:\One_Drive\OneDrive\Dokumente\SWS\FOR_ANALYSIS'
os.chdir(overall_path)

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

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def cl():
    plt.close('all')

reanalyze = True

plot = True


def get_plane_equation_from_points(P, Q, R):  
    x1, y1, z1 = P
    x2, y2, z2 = Q
    x3, y3, z3 = R
    a1 = x2 - x1 
    b1 = y2 - y1 
    c1 = z2 - z1 
    a2 = x3 - x1 
    b2 = y3 - y1 
    c2 = z3 - z1 
    a = b1 * c2 - b2 * c1 
    b = a2 * c1 - a1 * c2 
    c = a1 * b2 - b1 * a2 
    d = (- a * x1 - b * y1 - c * z1) 
    return a, b, c, d

#%% ------------------------------------------------------------------------------- fit plane for all detected SOs and plot each mouse individually

# home_directory = r'D:\JP OneDrive\OneDrive\Dokumente\SWS\FOR_ANALYSIS\UP_pairing\160426_D1'
# os.chdir(home_directory)
# day = os.getcwd()[-6:]
to_plot_1_LFP = [0,1,2,3]
to_plot_2_LFP = [4,5,6,7,8,9]

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

redo = True

# fit just using the channels with detected slow waves
fit_detected_only = False

# outliers for slope fit
exclude_outliers = True
outliers_cutoff = 1.5 # outlier channels before fit (how many times IQR of LFP min timepoint across channels)

# fit significance cutoff for further analysis
significance_for_analysis = 0.05

# 1/x how many channels have to have slow wave then
channel_occurrence_fraction = 4

# speed cutoff of outliers
speed_cutoff = 50 


speed_mean_ALL = np.zeros([len(days),10])
R2_mean_ALL = np.zeros([len(days),10])
angle_mean_ALL = np.zeros([len(days),10])
speed_mean_ALL[:] = np.NaN
R2_mean_ALL[:] = np.NaN
angle_mean_ALL[:] = np.NaN

speed_median_ALL = np.zeros([len(days),10])
R2_median_ALL = np.zeros([len(days),10])
angle_median_ALL = np.zeros([len(days),10])
speed_median_ALL[:] = np.NaN
R2_median_ALL[:] = np.NaN
angle_median_ALL[:] = np.NaN


speed_difference_mean_ALL = np.zeros([len(days),2])
R2_difference_mean_ALL = np.zeros([len(days),2])
speed_difference_mean_ALL[:] = np.NaN
R2_difference_mean_ALL[:] = np.NaN
angle_difference_mean_ALL = np.zeros([len(days),2])
angle_difference_mean_ALL[:] = np.NaN


speed_difference_median_ALL = np.zeros([len(days),2])
R2_difference_median_ALL = np.zeros([len(days),2])
speed_difference_median_ALL[:] = np.NaN
R2_difference_median_ALL[:] = np.NaN
angle_difference_median_ALL = np.zeros([len(days),2])
angle_difference_median_ALL[:] = np.NaN


for day_ind, day in enumerate(days):
# for day_ind, day in enumerate(['160427']):

    print(day)
    os.chdir(day)
    
    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    lfp_cutoff_resp_channels = 200
    
    unique_times = pickle.load(open('unique_times_tol_500_chans_200', 'rb'))
    SW_occurence_list = pickle.load(open('SW_occurence_list', 'rb'))
    UP_Cross_sweeps = pickle.load(open('UP_Cross_sweeps', 'rb'))
    
    
    # LFP_resp_channels = np.loadtxt('LFP_resp_channels.csv', delimiter = ',') 
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    LFP_resp_channels =  np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
    
    # maximize number of channels for correlation analysis (include good channels with slightly smaller LFP response below the cutoff too)
    # DOWN
    if '160308' in os.getcwd():
        chans_to_append = [55,10,12,14,17,19,21,23,25,53,51]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
    if '160420' in os.getcwd():
        chans_to_append = [9,59,61,15,14,62,60,58,52,50,63,0,2,4,6,8,10]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)     
    if '160427' in os.getcwd():
        chans_to_append = [1,5,3,13,12,2,4,56,26]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)     
    if '221212' in os.getcwd():
        chans_to_append = [22,1,49,0,51,2,4,6,15]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)    
        #some channels have small spiking artifacts not tied to SW activity which artificially increase STTC metric in the lower timesteps (taking these out biases against own hypothesis)
        # chans_to_delete = [25,33,35,21,39,37,35,33,21]
        # for chan in chans_to_delete:
        #     LFP_resp_channels = np.delete(LFP_resp_channels, np.where(LFP_resp_channels == chan)[0])
    if '221216' in os.getcwd():
        to_plot_1_corr = [3]
        chans_to_append = [30,28]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)    
        #some channels have small spiking artifacts not tied to SW activity which artificially increase STTC metric in the lower timesteps (taking these out biases against own hypothesis)
        chans_to_delete = [9,57]
        for chan in chans_to_delete:
            LFP_resp_channels = np.delete(LFP_resp_channels, np.where(LFP_resp_channels == chan)[0])
    
    # UP
    if '160414' in os.getcwd():
        chans_to_append = [41,43,9,45,27,58]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)     
    if '160426' in os.getcwd():
        chans_to_append = [33,63,14,22,23,25,9,7,11,41,5,53,55,1]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)            
    if '121121' in os.getcwd():
        chans_to_append = [46,44,42,62,25]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
    if '221220_3' in os.getcwd():
        to_plot_2_corr = [4,5,6,7,8]
        chans_to_append = [57,8,23]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)


    
    LFP_resp_channels_map = np.isin(channelMapArray, LFP_resp_channels)

    # go through SW occurence and make new list with how many LFP resp channels have a slow wave then:
    SW_occurence_number = []
    for ind_sweep in range(10):
        temp = []
        for SW in range(len(SW_occurence_list[ind_sweep])):
            temp.append(np.sum(SW_occurence_list[ind_sweep][SW][np.isin(channelMapArray, LFP_resp_channels)]))
        SW_occurence_number.append(np.asarray(temp))

    
    os.chdir('..')
    if redo == False and os.path.exists(f'R2_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl') and os.path.exists(f'angle_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl'):
        R2 = pickle.load(open(f'R2_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
        slope = pickle.load(open(f'slope_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
        p_fit = pickle.load(open(f'p_fit_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
        angle = pickle.load(open(f'angle_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
        
    else:
        LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))
        # spikes_allsweeps = pickle.load(open('spikes_allsweeps','rb'))
        stim_times = pickle.load(open('stim_times','rb'))
        
        to_plot_1_LFP = [0,1,2,3]
        to_plot_2_LFP = list(np.linspace(4,len(LFP_all_sweeps) - 1, len(LFP_all_sweeps) - 4, dtype = int))
        
        os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    
        Z_heatmap_all = [[] for j in range(10)]
        Z_heatmap_all_no_outliers = [[] for j in range(10)]
        slope = [[] for j in range(10)]
        R2 = [[] for j in range(10)]
        R2_refit_allchans = [[] for j in range(10)]
        angle = [[] for j in range(10)]
        p_fit = [[] for j in range(10)]
        
        for sweep in range(10):
        # for sweep in range(4):
            Z_heatmap_all[sweep] = np.zeros([len(unique_times[sweep]),8,8])#
            Z_heatmap_all_no_outliers[sweep] = np.zeros([len(unique_times[sweep]),8,8])#
            slope[sweep] = np.zeros([len(unique_times[sweep])])
            R2[sweep] = np.zeros([len(unique_times[sweep])])
            R2_refit_allchans[sweep] = [[] for j in range(10)]
            angle[sweep] = np.zeros([len(unique_times[sweep])])
            p_fit[sweep] = np.zeros([len(unique_times[sweep])])
            
            Z_heatmap_all[sweep][:] = np.NaN
            Z_heatmap_all_no_outliers[sweep][:] = np.NaN
            slope[sweep][:] = np.NaN
            R2[sweep][:] = np.NaN
            angle[sweep][:] = np.NaN
            p_fit[sweep][:] = np.NaN
        
            
            LFP_filt_delta = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep]), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 2*pq.Hz).as_array()
            LFP_filt_gamma = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep]), units = 'mV', sampling_rate = new_fs*pq.Hz), highpass_frequency = 250).as_array()
            
            for SW in range(len(unique_times[sweep])):
            
            
        # PLOTTING INDIVIDUAL SLOW WAVE
        # sweep = 7
        # LFP_filt_delta = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep]), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 2*pq.Hz).as_array()
        # SW = 53
        # fig, ax = plt.subplots(8,8)
        # for chan in range(64):
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(LFP_filt_delta[unique_times[sweep][SW] - 500:unique_times[sweep][SW] + 500,chan])
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 8)
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_ylim([np.min(LFP_filt_delta[unique_times[sweep][SW] - 500:unique_times[sweep][SW] + 500,LFP_resp_channels]), np.max(LFP_filt_delta[unique_times[sweep][SW] - 500:unique_times[sweep][SW] + 500,LFP_resp_channels])])
        #     if chan in LFP_resp_channels:
        #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
        #     if SW_occurence_list[sweep][SW][np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]] == 1:
        #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("r")
        #     if SW_occurence_list[sweep][SW][np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]] == 1 and chan in LFP_resp_channels:
        #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("c")
    
        # to_plot = 1
        # #FIRST DERIVATIVE
        # fig, ax = plt.subplots(8,8,sharey = True)
        # for chan in range(64):
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(np.gradient(LFP_filt_delta[unique_times[0][to_plot] - 500:unique_times[0][to_plot] + 500,chan]))
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 8)
        #     if chan in LFP_resp_channels:
        #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
        
        # #SECOND DERIVATIVE
        # fig, ax = plt.subplots(8,8,sharey = True)
        # for chan in range(64):
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(np.gradient(np.gradient(LFP_filt_delta[unique_times[0][to_plot] - 500:unique_times[0][to_plot] + 500,chan])))
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 8)
        #     if chan in LFP_resp_channels:
        #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
        
            # # #GAMMA
            # LFP_filt_gamma = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep]), units = 'mV', sampling_rate = new_fs*pq.Hz), highpass_frequency = 250).as_array()
            # fig, ax = plt.subplots(8,8,sharey = True)
            # for chan in range(64):
            #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(LFP_filt_gamma[unique_times[sweep][SW] - 500:unique_times[sweep][SW] + 500,chan])
            #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 8)
            #     if chan in LFP_resp_channels:
            #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
        
         
        # fig, ax = plt.subplots(8,8)
        # for chan in range(64):
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(LFP_all_sweeps[0][chan,unique_times[0][85] - 500:unique_times[0][85] + 500])
        #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 4)
        #     if chan in LFP_resp_channels:
        #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
    
    
                # extract LFP peak:
                # first find peak of first derivative (inflection point of UP to DOWN state).Because the UP to DOWN slope is in most cases very prominent good starting point in all channels
                # then from there the first peak looking backwards of second derivative (=UP peak)
                first_dev_peak = [np.argmax(np.gradient(LFP_filt_delta[unique_times[sweep][SW] - 500:unique_times[sweep][SW] + 500,chan])) for chan in range(64)]
                first_dev_peak_heatmap = np.zeros([8,8])
                first_dev_peak_heatmap[:] = np.NaN
                for chan_ind, chan in enumerate(LFP_resp_channels):
                    first_dev_peak_heatmap[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]] = first_dev_peak[chan]
                
                #cant do list comprehension because of possible empty values
                LFP_UP_peak_idx = []
                for chan in range(64):
                    # if clearly not a slow wave in that channel
                    if first_dev_peak[chan] < 10:
                        LFP_UP_peak_idx.append(np.NaN)
                        continue
                    # find UP state peak (peak of second derivative) just before peak of first derivative (inflection point)
                    second_dev_peaks = scipy.signal.find_peaks(np.gradient(np.gradient(LFP_filt_delta[unique_times[sweep][SW] - 500:unique_times[sweep][SW] - 500 + first_dev_peak[chan],chan])))[0]
                    if len(second_dev_peaks) == 0:
                        LFP_UP_peak_idx.append(np.NaN)
                    else:
                        LFP_UP_peak_idx.append(max(second_dev_peaks))
                
                if fit_detected_only == False:
                    Z = np.asarray([LFP_UP_peak_idx[chan] for chan in LFP_resp_channels])
                    X = np.asarray([np.where(channelMapArray == chan)[0][0] for chan in LFP_resp_channels])
                    Y = np.asarray([np.where(channelMapArray == chan)[1][0] for chan in LFP_resp_channels])

                else:
                    chans_to_fit = [list(LFP_resp_channels)[i] for i in range(len(LFP_resp_channels)) if SW_occurence_list[sweep][SW][np.where(channelMapArray == list(LFP_resp_channels)[i])[0][0], np.where(channelMapArray == list(LFP_resp_channels)[i])[1][0]] == 1]
                    Z = np.asarray([LFP_UP_peak_idx[chan] for chan in chans_to_fit])
                    X = np.asarray([np.where(channelMapArray == chan)[0][0] for chan in chans_to_fit])
                    Y = np.asarray([np.where(channelMapArray == chan)[1][0] for chan in chans_to_fit])

                #plot as a heatmap, easier to see
                Z_heatmap = np.zeros([8,8])
                Z_heatmap[:] = np.NaN

                if fit_detected_only == False:
                    for chan_ind, chan in enumerate(LFP_resp_channels):
                        Z_heatmap[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]] = Z[chan_ind]
                else:
                    # only use channels where a SW was detected in the first place for the fit
                    for chan_ind, chan in enumerate(chans_to_fit):
                        Z_heatmap[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]] = Z[chan_ind]

                Z_heatmap_all[sweep][SW,:,:] = Z_heatmap
                # plt.imshow(Z_heatmap)
                
                # take out np.NaN values
                X = X[~np.isnan(Z)]
                Y = Y[~np.isnan(Z)]
                Z = Z[~np.isnan(Z)]
                
                # if not enough point to make a fit
                if len(Z) < 4:
                    continue
                
                # take out outlier channels
                outliers = np.where(np.logical_or(Z > (np.percentile(Z, 75) + outliers_cutoff*(np.abs(np.percentile(Z, 75) - np.percentile(Z, 25)))), Z < (np.percentile(Z, 75) - outliers_cutoff*(np.abs(np.percentile(Z, 75) - np.percentile(Z, 25))))))[0]
                if exclude_outliers:
                    X = np.delete(X, outliers)
                    Y = np.delete(Y, outliers)
                    Z_no_outliers = np.delete(Z, outliers)
                
                #heatmap again this time without outliers
                Z_heatmap_no_outliers = np.zeros([8,8])
                Z_heatmap_no_outliers[:] = np.NaN
                for chan_ind in range(len(X)):
                    Z_heatmap_no_outliers[X[chan_ind], Y[chan_ind]] = Z_no_outliers[chan_ind]
                Z_heatmap_all_no_outliers[sweep][SW,:,:] = Z_heatmap_no_outliers
        
                # do fit
                tmp_A = []
                tmp_b = []
                for i in range(len(X)):
                    tmp_A.append([X[i], Y[i], 1])
                    tmp_b.append(Z_no_outliers[i])
                b = np.matrix(tmp_b).T
                A = np.matrix(tmp_A)
                
                fit, residual, rnk, s = np.linalg.lstsq(A, b)
                
                X_plane, Y_plane = np.meshgrid(np.arange(0,8),
                                               np.arange(0,8))
                Z_plane = np.zeros(X_plane.shape)
                for r in range(X_plane.shape[0]):
                    for c in range(X_plane.shape[1]):
                        Z_plane[r,c] = fit[0] * X_plane[r,c] + fit[1] * Y_plane[r,c] + fit[2]
                
                # point1 = [X_plane[0,0], Y_plane[0,0], Z_plane[0,0]]
                # point2 = [X_plane[0,1], Y_plane[0,1], Z_plane[0,1]]
                # point3 = [X_plane[4,3], Y_plane[4,3], Z_plane[4,3]]
                # a1,b1,c1,d1 = get_plane_equation_from_points(point1,point2,point3)
                
                # e = np.sqrt(a1**2 + b1**2 + c1**2)
                # sum([(np.linalg.norm(np.array([a1*X[i], b1*Y[i], c1*Z_no_outliers[i], d1]))/e)**2 for i in range(len(Z_no_outliers))])
                # sum([(np.abs(a1*X_plane[X[i], Y[i]] + b1*Y_plane[X[i], Y[i]] + c1*Z_no_outliers[i] + d1)/e)**2 for i in range(len(Z_no_outliers))])

                # e = np.sqrt(fit[0]**2 + fit[1]**2 + 1).item(0)
                # sum([(np.linalg.norm([-fit[0]*X[i], -fit[1]*Y[i], Z_no_outliers[i], -fit[2]])/e)**2 for i in range(len(X))])

                
                # --------------------------------------------------------- PLOT EXAMPLE SLOW WAVE FITTING
                # # plot in 3D to visualize. Just LFP responsive channels
                # if SW % 10 == 0:
                # plt.figure()
                # ax = plt.subplot(111, projection='3d')
                # ax.scatter(X, Y, Z_no_outliers, color='b')
                # ax.plot_wireframe(X_plane, Y_plane, Z_plane, color='k')
        
                # calculate R2
                if residual.size == 0: # if for some reason perfect fit, don't include
                    continue
                R2[sweep][SW] = 1 - residual.item(0) / sum((Z_no_outliers - mean(Z_no_outliers))**2)
                print(R2[sweep][SW], outliers)
                
                
                # # calculate R2 of the fitted plane on all channels
                # Z = np.asarray([LFP_UP_peak_idx[chan] for chan in LFP_resp_channels])
                # X = np.asarray([np.where(channelMapArray == chan)[0][0] for chan in LFP_resp_channels])
                # Y = np.asarray([np.where(channelMapArray == chan)[1][0] for chan in LFP_resp_channels])
                # X = X[~np.isnan(Z)]
                # Y = Y[~np.isnan(Z)]
                # Z = Z[~np.isnan(Z)]
                # # take out outlier channels
                # outliers = np.where(np.logical_or(Z > (np.percentile(Z, 75) + outliers_cutoff*(np.abs(np.percentile(Z, 75) - np.percentile(Z, 25)))), Z < (np.percentile(Z, 75) - outliers_cutoff*(np.abs(np.percentile(Z, 75) - np.percentile(Z, 25))))))[0]
                # if exclude_outliers:
                #     X = np.delete(X, outliers)
                #     Y = np.delete(Y, outliers)
                #     Z = np.delete(Z, outliers)
                # plt.figure()
                # ax = plt.subplot(111, projection='3d')
                # ax.scatter(X, Y, Z, color='b')
                # ax.plot_wireframe(X_plane, Y_plane, Z_plane, color='k')

                # # check you're calculating residual correctly
                # e = np.sqrt(fit[0]**2 + fit[1]**2 + 1).item(0)
                # sum([(np.linalg.norm(fit[0]*X[i] + fit[1]*Y[i] + Z_no_outliers[i] - fit[2])/np.sqrt(fit[0]**2 + fit[1]**2 + 1))**2 for i in range(len(X))])
                # residual_allchans = sum([(np.abs(fit[0]*X[i] + fit[1]*Y[i] + Z[i] - fit[2])/e)**2 for i in range(len(X))])
                # R2_refit_allchans[sweep][SW] = 1 - residual_allchans/sum((Z - mean(Z))**2)
                
                
                
                # calculate slope and direction
                # slope is going to be in the direction of the x and y fit vector
                slope[sweep][SW] = np.sqrt(fit[0]**2 + fit[1]**2).item(0)
                
                # X direction: up to down on channel array
                # Y direction: left to right on channel array
                # from arctan2: STRAIGHT RIGHT ON CHANNELMAP (VECTOR 0,1) IS 0 PI. DOWN (1,0) IS 1/2 PI, STRAIGHT LEFT 1PI, STRAIGHT UP -1/2PI
                angle[sweep][SW] = np.arctan2(fit[0].item(0), fit[1].item(0))
                
                # significant fit? (F statistic and then p value). H0 is that R2 is 0 (doesn't fit any better than the model with coefficients = 0).
                F = (R2[sweep][SW]/2)/((1 - R2[sweep][SW])/(b.size - 3))
                p_fit[sweep][SW] = 1 - scipy.stats.f.cdf(F, dfn=2, dfd=b.size - 3)
            
        
        os.chdir('..')
        pickle.dump(R2, open(f'R2_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'wb'))
        pickle.dump(slope, open(f'slope_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'wb'))
        pickle.dump(p_fit, open(f'p_fit_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'wb'))
        pickle.dump(angle, open(f'angle_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'wb'))
        pickle.dump(Z_heatmap_all_no_outliers, open(f'Z_heatmap_all_{exclude_outliers}.pkl', 'wb'))
    
        
    
    # SW to include for analysis: have to be a SW in at least XX of the channels, and have to have a significant fit
    SW_for_analysis = []
    for sweep in range(10):
        
        # occurring in enough channels?
        occ = SW_occurence_number[sweep] > np.floor(len(LFP_resp_channels)/channel_occurrence_fraction)
        
        # significant fit? (F statistic and then p value). H0 is that R2 is 0 (doesn't fit any better than the model with coefficients = 0).
        sign = p_fit[sweep] < significance_for_analysis
        SW_for_analysis.append(np.asarray(np.logical_and(occ == True, sign == True)))
    
    
    # find speed outliers in each sweep, so you can check manually
    #with normal distribution thing, doesnt really work as not a normal distribution.
    # outliers = []
    # for sweep in range(10):
    #     SW = 200/slope[sweep][SW_for_analysis[sweep]]
    #     outliers.append(np.where(np.logical_or(SW > (np.percentile(SW, 75) + 1.5*(np.abs(np.percentile(SW, 75) - np.percentile(SW, 25)))), SW < (np.percentile(SW, 75) - 1.5*(np.abs(np.percentile(SW, 75) - np.percentile(SW, 25))))))[0])
    
    # probably best to just take a cutoff for speed outliers, like 100ym/s
    outliers = []
    for sweep in range(10):
        SW = 200/slope[sweep][SW_for_analysis[sweep]]
        outliers.append(SW > speed_cutoff)
    
    
    slope_with_outliers = [slope[sweep][SW_for_analysis[sweep]] for sweep in range(10)]
    slope_without_outliers = [slope[sweep][SW_for_analysis[sweep]][~outliers[sweep]] for sweep in range(10)]
    
    unique_times_for_analysis_with_outliers = [np.asarray(unique_times[i])[SW_for_analysis[i]] for i in range(10)]
    unique_times_for_analysis_without_outliers = [np.asarray(unique_times[i])[SW_for_analysis[i]][~outliers[i]] for i in range(10)]
    
    R2_with_outliers = [R2[i][SW_for_analysis[i]] for i in range(10)]
    R2_without_outliers = [R2[i][SW_for_analysis[i]][~outliers[i]] for i in range(10)]
    
    angle_with_outliers = [angle[i][SW_for_analysis[i]] for i in range(10)]
    angle_without_outliers = [angle[i][SW_for_analysis[i]][~outliers[i]] for i in range(10)]
    
    # average across SWs to get one value per sweep
    speed_mean_ALL[day_ind,:] = [200/np.mean(slope_without_outliers[sweep]) for sweep in range(10)]
    R2_mean_ALL[day_ind,:] = [np.mean(R2_without_outliers[sweep]) for sweep in range(10)]
    angle_mean_ALL[day_ind,:] = [np.mean(angle_without_outliers[sweep]) for sweep in range(10)]
    speed_median_ALL[day_ind,:] = [200/np.median(slope_without_outliers[sweep]) for sweep in range(10)]
    R2_median_ALL[day_ind,:] = [np.median(R2_without_outliers[sweep]) for sweep in range(10)]
    angle_median_ALL[day_ind,:] = [np.median(angle_without_outliers[sweep]) for sweep in range(10)]




    #plot single slow wave
    # sweep = 0
    # SW = 24
    # LFP_filt_delta = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep]), units = 'mV', sampling_rate = new_fs*pq.Hz), lowpass_frequency = 2*pq.Hz).as_array()
    # fig, ax = plt.subplots(8,8)
    # for chan in range(64):
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(LFP_filt_delta[unique_times_for_analysis_without_outliers[sweep][SW] - 500:unique_times_for_analysis_without_outliers[sweep][SW] + 500,chan])
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 8)
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_ylim([np.min(LFP_filt_delta[unique_times_for_analysis_without_outliers[sweep][SW] - 500:unique_times_for_analysis_without_outliers[sweep][SW] + 500,LFP_resp_channels_cutoff]), np.max(LFP_filt_delta[unique_times_for_analysis_without_outliers[sweep][SW] - 500:unique_times_for_analysis_without_outliers[sweep][SW] + 500,LFP_resp_channels_cutoff])])
    #     if chan in LFP_resp_channels_cutoff:
    #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
    #     if SW_occurence_list[sweep][SW][np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]] == 1:
    #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("r")
    #     if SW_occurence_list[sweep][SW][np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]] == 1 and chan in LFP_resp_channels_cutoff:
    #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("c")
    
    # # #GAMMA
    # LFP_filt_gamma = elephant.signal_processing.butter(neo.core.AnalogSignal(np.transpose(LFP_all_sweeps[sweep]), units = 'mV', sampling_rate = new_fs*pq.Hz), highpass_frequency = 250).as_array()
    # fig, ax = plt.subplots(8,8,sharey = True)
    # for chan in range(64):
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].plot(LFP_filt_gamma[unique_times[sweep][SW] - 500:unique_times[sweep][SW] + 500,chan])
    #     ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_title(str(chan), size = 8)
    #     if chan in LFP_resp_channels_cutoff:
    #         ax[np.where(channelMapArray == chan)[0][0], np.where(channelMapArray == chan)[1][0]].set_facecolor("y")
    
    # b_notch, a_notch = scipy.signal.iirnotch(50, 100, 1000)
    # x_range = 4000
    # i = 1
    # fig, ax = plt.subplots()
    # for i_ind, i in enumerate(list(LFP_resp_channels_cutoff)):
    #     ax.plot(scipy.signal.filtfilt(b_notch, a_notch, LFP_all_sweeps[sweep][i,:])[int(unique_times_for_analysis_without_outliers[sweep][SW] - x_range/2):int(unique_times_for_analysis_without_outliers[sweep][SW] + x_range/2)] + i_ind *1000 * np.ones_like(x_range), linewidth = 0.5)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     i += 1 
    
    
    
    # ---------------------------------------------------------------------- PLOTTING SPEED AND ANGLES FOR EVERY MOUSE INDIVIDUALLY
    # # if not enough channels for SW propagation
    # if len(LFP_resp_channels) < 4 or any([len(slope_without_outliers[i]) < 1 for i in range(10)]) == True:
    #     os.chdir('..')
    #     continue 
    
    # else:
    
        # -------------------------------------------------------------------------- plot speed histogram
        
        # fig, ax = plt.subplots(figsize = (6,3))
        # bins_to_plot = 20
        # slope_to_plot = slope_without_outliers 
        # range_to_plot = [0,40]
        # # range_to_plot = (min([min(200/slope_to_plot[i]) for i in range(10)]), max([max(200/slope_to_plot[i]) for i in range(10)]))
        # to_plot = np.histogram(200/np.concatenate(slope_to_plot), bins = bins_to_plot, range = range_to_plot)
        # ax.bar(to_plot[1][:-1], to_plot[0]/sum(to_plot[0]), width = (range_to_plot[1] - range_to_plot[0])/bins_to_plot, color = 'k')
        # ax.tick_params(axis = 'x', labelsize = 14)
        # ax.tick_params(axis = 'y', labelsize = 14)
        # ax.set_xlabel('SO propagation speed (mm/s)', size = 16)
        # ax.set_ylabel('proportion of SOs', size = 16)
        # plt.tight_layout()
        # plt.savefig('speed histogram example.pdf', dpi = 1000, format = 'pdf')
        # plt.savefig('speed histogram example.jpg', dpi = 1000, format = 'jpg')
 
        # all sweeps in individual subplots
        # fig, ax = plt.subplots(5,2,sharex = True, sharey = True)
        # fig.suptitle(f'{day} speed')
        # bins_to_plot = 50
        # slope_to_plot = slope_without_outliers 
        # range_to_plot = (min([min(200/slope_to_plot[i]) for i in range(10)]), max([max(200/slope_to_plot[i]) for i in range(10)]))
        # for sweep, ax1 in enumerate(list(ax.flatten())):
        #     to_plot = np.histogram(200/slope_to_plot[sweep], bins = bins_to_plot, range = range_to_plot)
        #     ax1.bar(to_plot[1][:-1], to_plot[0]/sum(to_plot[0]), width = (range_to_plot[1] - range_to_plot[0])/bins_to_plot)
        #     ax1.axvline(np.median(200/slope_to_plot[sweep]), color = 'red')
        
        
        
        
        # # slope before vs after
        # fig, ax = plt.subplots(2, 1, sharex = True, sharey = True)
        # fig.suptitle(f'{day} slope')
        # bins_to_plot = 50
        # slope_to_plot = slope_without_outliers
        # range_to_plot = (min([min(200/slope_to_plot[i]) for i in range(10)]), max([max(200/slope_to_plot[i]) for i in range(10)]))
        # #before
        # to_plot = np.histogram(200/np.concatenate([slope_to_plot[sweep] for sweep in [0,1,2,3]]), bins = bins_to_plot, range = range_to_plot)
        # ax[0].bar(to_plot[1][:-1], to_plot[0]/sum(to_plot[0]), width = (range_to_plot[1] - range_to_plot[0])/bins_to_plot)
        # ax[0].axvline(np.median(200/np.concatenate([slope_to_plot[sweep] for sweep in [0,1,2,3]])), color = 'red')
        # #after
        # to_plot = np.histogram(200/np.concatenate([slope_to_plot[sweep] for sweep in [4,5,6,7,8,9]]), bins = bins_to_plot, range = range_to_plot)
        # ax[1].bar(to_plot[1][:-1], to_plot[0]/sum(to_plot[0]), width = (range_to_plot[1] - range_to_plot[0])/bins_to_plot)
        # ax[1].axvline(np.median(200/np.concatenate([slope_to_plot[sweep] for sweep in [4,5,6,7,8,9]])), color = 'red')
        # print(np.median(200/np.concatenate([slope_to_plot[sweep] for sweep in [0,1,2,3]])), np.median(200/np.concatenate([slope_to_plot[sweep] for sweep in [4,5,6,7,8,9]])))
        # print(np.mean(200/np.concatenate([slope_to_plot[sweep] for sweep in [0,1,2,3]])), np.mean(200/np.concatenate([slope_to_plot[sweep] for sweep in [4,5,6,7,8,9]])))
        # speed_difference_mean_ALL[day_ind, 0] = np.mean(200/np.concatenate([slope_to_plot[sweep] for sweep in [0,1,2,3]]))
        # speed_difference_mean_ALL[day_ind, 1] = np.mean(200/np.concatenate([slope_to_plot[sweep] for sweep in [4,5,6,7,8,9]]))
        # speed_difference_median_ALL[day_ind, 0] = np.median(200/np.concatenate([slope_to_plot[sweep] for sweep in [0,1,2,3]]))
        # speed_difference_median_ALL[day_ind, 1] = np.median(200/np.concatenate([slope_to_plot[sweep] for sweep in [4,5,6,7,8,9]]))
                                                       
                
        # ANGLE 
        # fig, ax = plt.subplots(5,2, sharex = True, subplot_kw={'projection': 'polar'})
        # fig.suptitle(f'{day} angle')
        # bins_to_plot = 15
        # theta = np.linspace(-np.pi, np.pi, bins_to_plot) # angles for the plots in radians
        # width = (2*np.pi)/bins_to_plot
        # for sweep, ax1 in enumerate(list(ax.flatten())):
        #     ax1.bar(theta, np.histogram(angle_without_outliers[sweep], bins = bins_to_plot)[0], width = width)
        #     # ax = plt.subplot(111, polar = True)
        #     # bars = ax.bar(theta, np.histogram(angle[sweep][SW_for_analysis[sweep]], bins = bins_to_plot)[0], width = width)
        
        # fig, ax = plt.subplots(2, 1, sharex = True, subplot_kw={'projection': 'polar'})
        # fig.suptitle(f'{day} angle')
        # bins_to_plot = 15
        # theta = np.linspace(-np.pi, np.pi, bins_to_plot) # angles for the plots in radians
        # width = (2*np.pi)/bins_to_plot
        # #before
        # ax[0].bar(theta, np.histogram(np.concatenate([angle_without_outliers[sweep] for sweep in [0,1,2,3]]), bins = bins_to_plot)[0], width = width)
        # ax[0].plot([0,pycircstat.mean(np.concatenate([angle_without_outliers[sweep] for sweep in [0,1,2,3]]))], [0,25], color = 'r')
        # #after
        # ax[1].bar(theta, np.histogram(np.concatenate([angle_without_outliers[sweep] for sweep in [4,5,6,7,8,9]]), bins = bins_to_plot)[0], width = width)
        # ax[1].plot([0,pycircstat.mean(np.concatenate([angle_without_outliers[sweep] for sweep in [4,5,6,7,8,9]]))], [0,25], color = 'r')
        # angle_difference_mean_ALL[day_ind,0] = pycircstat.mean(np.concatenate([angle_without_outliers[sweep] for sweep in [0,1,2,3]]))
        # angle_difference_mean_ALL[day_ind,1] = pycircstat.mean(np.concatenate([angle_without_outliers[sweep] for sweep in [4,5,6,7,8,9]]))
        # angle_difference_median_ALL[day_ind,0] = pycircstat.median(np.concatenate([angle_without_outliers[sweep] for sweep in [0,1,2,3]]))
        # angle_difference_median_ALL[day_ind,1] = pycircstat.median(np.concatenate([angle_without_outliers[sweep] for sweep in [4,5,6,7,8,9]]))
        
        # from arctan2: STRAIGHT RIGHT ON CHANNELMAP (VECTOR 0,1) IS 0 PI. DOWN (1,0) IS 1/2 PI, STRAIGHT LEFT 1PI, STRAIGHT UP -1/2PI
        # overall angle for example plot
        # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # # fig.suptitle(f'{day} angle')
        # bins_to_plot = 15
        # theta = np.linspace(np.pi, -np.pi, bins_to_plot) # angles for the plots in radians
        # width = (2*np.pi)/bins_to_plot
        # mean_angle = -pycircstat.mean(np.concatenate([angle_without_outliers[sweep] for sweep in [0,1,2,3,4,5,6,7,8,9]]))
        # resultant_length = pycircstat.resultant_vector_length(np.concatenate([angle_without_outliers[sweep] for sweep in [0,1,2,3,4,5,6,7,8,9]]))/len(np.concatenate([angle_without_outliers[sweep] for sweep in [0,1,2,3,4,5,6,7,8,9]]))
        # ax.bar(theta, np.histogram(np.concatenate([angle_without_outliers[sweep] for sweep in [0,1,2,3,4,5,6,7,8,9]]), bins = bins_to_plot)[0]/len(np.concatenate([angle_without_outliers[sweep] for sweep in [0,1,2,3,4,5,6,7,8,9]])), width = width, color = 'black')
        # ax.plot([0, mean_angle], [0,resultant_length*100], color = 'r')
        # plt.savefig('angle histogram example.pdf', dpi = 1000, format = 'pdf')
        # plt.savefig('angle histogram example.jpg', dpi = 1000, format = 'jpg')


        # to_plot = np.histogram(200/np.concatenate([slope_to_plot[sweep] for sweep in [4,5,6,7,8,9]]), bins = bins_to_plot, range = range_to_plot)
        # ax[1].bar(to_plot[1][:-1], to_plot[0]/sum(to_plot[0]), width = (range_to_plot[1] - range_to_plot[0])/bins_to_plot)
        # ax[1].axvline(np.median(200/np.concatenate([slope_to_plot[sweep] for sweep in [4,5,6,7,8,9]])), color = 'red')
        # print(np.median(200/np.concatenate([slope_to_plot[sweep] for sweep in [0,1,2,3]])), np.median(200/np.concatenate([slope_to_plot[sweep] for sweep in [4,5,6,7,8,9]])))
        # print(np.mean(200/np.concatenate([slope_to_plot[sweep] for sweep in [0,1,2,3]])), np.mean(200/np.concatenate([slope_to_plot[sweep] for sweep in [4,5,6,7,8,9]])))
        # speed_difference_mean_ALL[day_ind, 0] = np.mean(200/np.concatenate([slope_to_plot[sweep] for sweep in [0,1,2,3]]))
        # speed_difference_mean_ALL[day_ind, 1] = np.mean(200/np.concatenate([slope_to_plot[sweep] for sweep in [4,5,6,7,8,9]]))
        # speed_difference_median_ALL[day_ind, 0] = np.median(200/np.concatenate([slope_to_plot[sweep] for sweep in [0,1,2,3]]))
        # speed_difference_median_ALL[day_ind, 1] = np.median(200/np.concatenate([slope_to_plot[sweep] for sweep in [4,5,6,7,8,9]]))
       
        
        # R2 fit
        # fig, ax = plt.subplots(2, 1, sharex = True, sharey = True)
        # fig.suptitle(f'{day} R2')
        # bins_to_plot = 50
        # R2_to_plot = R2_without_outliers
        # range_to_plot = (min([min(R2_to_plot[i]) for i in range(10)]), max([max(R2_to_plot[i]) for i in range(10)]))
        # #before
        # to_plot = np.histogram(np.concatenate([R2_to_plot[sweep] for sweep in [0,1,2,3]]), bins = bins_to_plot, range = range_to_plot)
        # ax[0].bar(to_plot[1][:-1], to_plot[0]/sum(to_plot[0]), width = (range_to_plot[1] - range_to_plot[0])/bins_to_plot)
        # ax[0].axvline(np.median(np.concatenate([R2_to_plot[sweep] for sweep in [0,1,2,3]])), color = 'red')
        # #after
        # to_plot = np.histogram(np.concatenate([R2_to_plot[sweep] for sweep in [4,5,6,7,8,9]]), bins = bins_to_plot, range = range_to_plot)
        # ax[1].bar(to_plot[1][:-1], to_plot[0]/sum(to_plot[0]), width = (range_to_plot[1] - range_to_plot[0])/bins_to_plot)
        # ax[1].axvline(np.median(np.concatenate([R2_to_plot[sweep] for sweep in [4,5,6,7,8,9]])), color = 'red')
        # print(np.median(np.concatenate([R2_to_plot[sweep] for sweep in [0,1,2,3]])), np.median(np.concatenate([R2_to_plot[sweep] for sweep in [4,5,6,7,8,9]])))
        # print(np.mean(np.concatenate([R2_to_plot[sweep] for sweep in [0,1,2,3]])), np.mean(np.concatenate([R2_to_plot[sweep] for sweep in [4,5,6,7,8,9]])))
        # print(len(np.concatenate([R2_to_plot[sweep] for sweep in [0,1,2,3]])), len(np.concatenate([R2_to_plot[sweep] for sweep in [4,5,6,7,8,9]])))
        # # print(np.mean(200/np.concatenate([to_plot[sweep] for sweep in [0,1,2,3]])), np.mean(200/np.concatenate([to_plot[sweep] for sweep in [4,5,6,7,8,9]])))
        # R2_difference_mean_ALL[day_ind, 0] = np.mean(np.concatenate([R2_to_plot[sweep] for sweep in [0,1,2,3]]))
        # R2_difference_mean_ALL[day_ind, 1] = np.mean(np.concatenate([R2_to_plot[sweep] for sweep in [4,5,6,7,8,9]]))
        # R2_difference_median_ALL[day_ind, 0] = np.median(np.concatenate([R2_to_plot[sweep] for sweep in [0,1,2,3]]))
        # R2_difference_median_ALL[day_ind, 1] = np.median(np.concatenate([R2_to_plot[sweep] for sweep in [4,5,6,7,8,9]]))
        

    os.chdir('..')
    
np.savetxt('speed_mean_ALL.csv', speed_mean_ALL, delimiter = ',')
np.savetxt('angle_mean_ALL.csv', angle_mean_ALL, delimiter = ',')
np.savetxt('R2_mean_ALL.csv', R2_mean_ALL, delimiter = ',')
np.savetxt('speed_median_ALL.csv', speed_median_ALL, delimiter = ',')
np.savetxt('angle_median_ALL.csv', angle_median_ALL, delimiter = ',')
np.savetxt('R2_median_ALL.csv', R2_median_ALL, delimiter = ',')


#%% EXAMPLE SW AND PLANE FITS FOR THESIS
# --------------------------------------------------------------------------- example slow wave -----------------------------------------------------------------------------

to_plot_1_LFP = [0,1,2,3]
to_plot_2_LFP = [4,5,6,7,8,9]

days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

# outliers for slope fit
exclude_outliers = True
outliers_cutoff = 1.5 # outlier channels before fit (how many times IQR of LFP min timepoint across channels)
fit_detected_only = True

# for day_ind, day in enumerate(days):
for day in ['160427']:
    print(day)
    os.chdir(day)
    LFP_all_sweeps = pickle.load(open('LFP_resampled','rb'))

    os.chdir([i for i in os.listdir() if 'analysis' in i][0])
    LFP_min = np.loadtxt('LFP_min.csv', delimiter = ',')
    lfp_cutoff_resp_channels = 200
    LFP_resp_channels =  np.asarray([chan for chan in range(64) if (LFP_min[to_plot_1_LFP, chan] > lfp_cutoff_resp_channels).all() and (LFP_min[to_plot_2_LFP,chan] > lfp_cutoff_resp_channels).all()], dtype = int)
    SW_spiking_channels = np.loadtxt('SW_spiking_channels.csv', delimiter = ',').astype(int)
    # LFP_resp_channels = np.loadtxt('LFP_resp_channels_cutoff.csv', delimiter = ',') 

    unique_times = pickle.load(open('unique_times_tol_500_chans_200', 'rb'))
    SW_occurence_list = pickle.load(open('SW_occurence_list', 'rb'))
    UP_Cross_sweeps = pickle.load(open('UP_Cross_sweeps', 'rb'))
    os.chdir('..')
    
    R2 = pickle.load(open(f'R2_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
    slope = pickle.load(open(f'slope_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
    p_fit = pickle.load(open(f'p_fit_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
    angle = pickle.load(open(f'angle_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
    heatmap = pickle.load(open(f'Z_heatmap_all_{exclude_outliers}.pkl', 'rb'))

    # maximize number of channels for correlation analysis (include good channels with slightly smaller LFP response below the cutoff too)
    # DOWN
    if '160308' in os.getcwd():
        chans_to_append = [55,10,12,14,17,19,21,23,25,53,51]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
    if '160420' in os.getcwd():
        chans_to_append = [9,59,61,15,14,62,60,58,52,50,63,0,2,4,6,8,10]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)     
    if '160427' in os.getcwd():
        chans_to_append = [1,5,3,13,12,2,4,56,26]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)     
    
    # UP
    if '160414' in os.getcwd():
        chans_to_append = [41,43,9,45,27,58]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)     
    if '160426' in os.getcwd():
        chans_to_append = [33,63,14,22,23,25,9,7,11,41,5,53,55,1]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)
            LFP_resp_channels = np.append(LFP_resp_channels, chan)            
    if '121121' in os.getcwd():
        chans_to_append = [46,44,42,62,25]
        chans_to_append = [i for i in chans_to_append if i not in LFP_resp_channels]
        for chan in chans_to_append:
            LFP_resp_channels = np.append(LFP_resp_channels, chan)

    LFP_resp_channels_map = np.isin(channelMapArray, np.intersect1d(LFP_resp_channels, SW_spiking_channels))

    # go through SW occurence and make new list with how many LFP resp channels have a slow wave then:
    SW_occurence_number = []
    for ind_sweep in range(10):
        temp = []
        for SW in range(len(SW_occurence_list[ind_sweep])):
            temp.append(np.sum(SW_occurence_list[ind_sweep][SW][np.isin(channelMapArray, LFP_resp_channels)]))
        SW_occurence_number.append(np.asarray(temp))
    
sweep = 3
good_SW = np.argwhere(np.logical_and(p_fit[sweep] < 0.01, np.asarray([np.sum(i) for i in SW_occurence_number[sweep]]) > 30, R2[sweep] > 0.75))


# ---------------------------------------------- plot fitted plane ------------------------------------------------------------------ 

to_plot_SW = 1 # SW to plot from good_SW list
# to_plot_SW = 6 # SW to plot from good_SW list

# do fit
X = np.argwhere(np.isnan(heatmap[sweep][good_SW[to_plot_SW]]) == False)[:,1]
Y = np.argwhere(np.isnan(heatmap[sweep][good_SW[to_plot_SW]]) == False)[:,2]
Z = heatmap[sweep][good_SW[to_plot_SW]].flatten()[~np.isnan(heatmap[sweep][good_SW[to_plot_SW]].flatten())]
Z = Z-np.min(Z)
tmp_A = []
tmp_b = []
for i in range(len(X)):
    tmp_A.append([X[i], Y[i], 1])
    tmp_b.append(Z[i])
b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)
fit, residual, rnk, s = np.linalg.lstsq(A, b)
X_plane, Y_plane = np.meshgrid(np.arange(min(X), max(X)+1),
                               np.arange(min(Y), max(Y)+1))
Z_plane = np.zeros(X_plane.shape)
for r in range(X_plane.shape[0]):
    for c in range(X_plane.shape[1]):
        Z_plane[r,c] = fit[0] * X_plane[r,c] + fit[1] * Y_plane[r,c] + fit[2]

plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(X, Y, Z, color='blue', s = 20)
ax.plot_wireframe(X_plane, Y_plane, Z_plane, color='black', rstride = 1, alpha = 0.5)
ax.plot_surface(X_plane, Y_plane, Z_plane, alpha = 0.25, cmap = 'plasma')
ax.view_init(elev=20, azim=-135)
ax.set_zlim([0,40])
ax.set_xlim([0,8])
ax.set_ylim([0,8])
# ax.set_xticks(np.linspace(0,7,8))
ax.set_xticklabels(list(map(str, np.linspace(0,16,9).astype('int')/10)), size = 12, rotation = -25)
# ax.set_yticks(np.linspace(0,7,8))
ax.set_yticklabels(list(map(str, np.linspace(0,16,9).astype('int')/10)), size = 12, rotation = 25)
ax.tick_params(axis="z", labelsize=12)
plt.tight_layout()
plt.savefig('SW plane fit example.jpg', format = 'jpg', dpi = 1000)
plt.savefig('SW plane fit example.pdf', format = 'pdf', dpi = 1000)



# plot sequence of LFP heatmap
UP_cross = unique_times[sweep][int(good_SW[to_plot_SW])]
SW_LFP = (LFP_all_sweeps[sweep][:,UP_cross + 300:UP_cross + 1000]).T
norm_SW_LFP = np.divide(SW_LFP, np.min(SW_LFP, axis = 0))# normalize to biggest LFP deflection in each channel
# do different timepoints along the slow wave time
time_to_avg = 10
fig, ax = plt.subplots(9,3, figsize = (10,10))
for ax_ind, ax1 in enumerate(list(ax.flatten())):
    # to_plot = np.nanmean(norm_SW_LFP[time_to_avg*ax_ind:time_to_avg*ax_ind + time_to_avg,:], axis = 0)[chanMap]
    to_plot = np.nanmean(-SW_LFP[time_to_avg*ax_ind:time_to_avg*ax_ind + time_to_avg,:], axis = 0)[chanMap]
    to_plot = to_plot.reshape(8,8)
    # to_plot[0,5] = np.min(to_plot)
    # to_plot[0,2] = np.min(to_plot)
    # to_plot[1,2] = np.min(to_plot)
    ax1.imshow(to_plot, vmin = -200, vmax = 200)
    ax1.set_xticks([])
    ax1.set_yticks([])
plt.tight_layout()
plt.savefig('SW example.jpg', format = 'jpg', dpi = 1000)
plt.savefig('SW example.pdf', format = 'pdf', dpi = 1000)

fig, ax = plt.subplots(figsize = (2,5))
norm = colors.Normalize(vmin=0, vmax=1)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'),
              cax=ax, ticks = [0,0.25,0.5,0.75,1])
ax.set_yticklabels(list(map(str, np.linspace(-500, 500, 5).astype(int))), size = 18)
ax.set_ylabel('LFP (uV)', size = 16)
plt.tight_layout()
plt.savefig('SW example colormap legend.pdf', dpi = 1000, format = 'pdf')
plt.savefig('SW example colormap legend.jpg', dpi = 1000, format = 'jpg')



#%% Angle polar histogram all mice and speed histogram for all mice

# fit just using the channels with detected slow waves
fit_detected_only = True

# outliers for slope fit
exclude_outliers = True
outliers_cutoff = 1.5 # outlier channels before fit (how many times IQR of LFP min timepoint across channels)
speed_cutoff = 40 

speed_all_mice = []
angle_all_mice = []
resultant_all_mice = []

for cond in ['UP_pairing', 'DOWN_pairing']:
    os.chdir(cond)
    days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

    for day_ind, day in enumerate(days):
        if day in ['061221', '160218', '160624_B2', '160628_D1', '191121']: # mice not included in LFP cross correlation analysis
            continue
    # for day_ind, day in enumerate(['160427']):
    
        print(day)
        os.chdir(day)
        
        R2 = pickle.load(open(f'R2_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
        slope = pickle.load(open(f'slope_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
        p_fit = pickle.load(open(f'p_fit_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
        angle = pickle.load(open(f'angle_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
        
        outliers = []
        for sweep in range(10):
            SW = 200/slope[sweep]
            outliers.append(SW > speed_cutoff)
        
        slope_without_outliers = [slope[sweep][~outliers[sweep]] for sweep in range(10)]
        
        angle_without_outliers = [angle[sweep][~outliers[sweep]] for sweep in range(10)]
        angle_without_outliers = [i[~np.isnan(i)] for i in angle_without_outliers]
        
        speed_all_mice.append(np.concatenate(slope_without_outliers))
        angle_all_mice.append(-pycircstat.mean(np.concatenate(angle_without_outliers)))
        resultant_all_mice.append(pycircstat.resultant_vector_length(np.concatenate(angle_without_outliers))/len(np.concatenate(angle_without_outliers)))
        
        os.chdir('..')
    os.chdir('..')

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize = (3,3))
# fig.suptitle(f'{day} angle')
for vector_to_plot, angle_to_plot in zip(resultant_all_mice, angle_all_mice):
    ax.plot([0, angle_to_plot], [0,vector_to_plot*100], linewidth = 3)#
# ax.set_xticks([])
ax.set_yticks([])
# plt.savefig('angle histogram all mice.pdf', dpi = 1000, format = 'pdf')
# plt.savefig('angle histogram all mice.jpg', dpi = 1000, format = 'jpg')


fig, ax = plt.subplots(figsize = (1,1))
ax.boxplot([np.nanmedian(i) for i in speed_all_mice], widths = 0.5)
# ax.set_yticks([0.4, 0.7])
# ax.set_ylim([0.35, 0.85])
ax.set_yticks([10,20,30])
ax.set_ylim([8,40])
ax.tick_params(axis="y", labelsize=14)    
ax.set_xticks([])
plt.tight_layout()
# plt.savefig('SO speed all mice.jpg', dpi = 1000, format = 'jpg')
# plt.savefig('SO speed all mice.pdf', dpi = 1000, format = 'pdf')



#%% SPEED UP VS DOWN

mice_UP = [0,1,2,3,4,8,9,10,11]
mice_DOWN = [2,3,4,5,6,7,8,9,10,11]
def normalize(array):
    return(np.transpose(np.transpose(array)/np.nanmean(array[:,[0,1,2,3]], axis = 1)))



fig1, ax1 = plt.subplots(figsize = (10,4))
# fig1.suptitle('speed mean')
# fig2, ax2 = plt.subplots(figsize = (10,6))
# fig2.suptitle('angle mean')
# fig3, ax3 = plt.subplots(figsize = (10,6))
# fig3.suptitle('R2 mean')
fig4, ax4 = plt.subplots(figsize = (10,4))
# fig4.suptitle('speed median')
# fig5, ax5 = plt.subplots(figsize = (10,6))
# fig5.suptitle('angle median')
# fig6, ax6 = plt.subplots(figsize = (10,6))
# fig6.suptitle('R2 median')

for group in ['UP_pairing', 'DOWN_pairing']:
    os.chdir(overall_path)
    os.chdir(group)
    speed_mean_ALL = np.loadtxt('speed_mean_ALL.csv', delimiter = ',')
    angle_mean_ALL = np.loadtxt('angle_mean_ALL.csv', delimiter = ',')
    R2_mean_ALL = np.loadtxt('R2_mean_ALL.csv', delimiter = ',')
    speed_median_ALL = np.loadtxt('speed_median_ALL.csv', delimiter = ',')
    angle_median_ALL = np.loadtxt('angle_median_ALL.csv', delimiter = ',')
    R2_median_ALL = np.loadtxt('R2_median_ALL.csv', delimiter = ',')

    if 'UP_pairing' in os.getcwd():
        mice_to_plot = mice_UP
        color = 'red'
    elif 'DOWN_pairing' in os.getcwd():
        mice_to_plot = mice_DOWN
        color = 'black'
    
    speed_mean_normalized = normalize(speed_mean_ALL[mice_to_plot,:])
    speed_median_normalized = normalize(speed_median_ALL[mice_to_plot,:])
    # np.savetxt('speed_mean_for_ANOVA.csv', speed_mean_normalized, delimiter = ',')
    # np.savetxt('speed_median_for_ANOVA.csv', speed_median_normalized, delimiter = ',')
    numb = len(mice_to_plot)
    os.chdir(overall_path)
    # plot speed
    ax1.plot(np.nanmean(speed_mean_normalized, axis = 0), color = color)
    ax1.fill_between(list(range(10)), np.nanmean(speed_mean_normalized, axis = 0) + 2*np.nanstd(speed_mean_normalized, axis = 0)/np.sqrt(numb), np.nanmean(speed_mean_normalized, axis = 0) - 2*np.nanstd(speed_mean_normalized, axis = 0)/np.sqrt(numb), color = color, alpha = 0.1)
    ax1.axvline(3.5, linestyle = '--', color = 'k')
    ax1.set_ylim([.45, 1.45])
    ax1.set_yticks([.5,.75,1,1.25,1.5])
    ax1.set_yticklabels(list(map(str, (ax1.get_yticks()*100).astype(int))), size = 16)
    ax1.set_ylabel('propagation speed (% baseline)', size = 16)
    ax1.set_xticks([0,2,5,7,9])
    ax1.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 16)
    ax1.set_xlabel('time from pairing (min)', size = 16)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.tight_layout()
    
    ax4.plot(np.nanmean(speed_median_normalized, axis = 0), color = color)
    ax4.fill_between(list(range(10)), np.nanmean(speed_median_normalized, axis = 0) + 2*np.nanstd(speed_median_normalized, axis = 0)/np.sqrt(numb), np.nanmean(speed_median_normalized, axis = 0) - 2*np.nanstd(speed_median_normalized, axis = 0)/np.sqrt(numb), color = color, alpha = 0.1)
    ax4.axvline(3.5, linestyle = '--', color = 'k')
    ax4.set_ylim([.45, 1.45])
    ax4.set_yticks([.5,.75,1,1.25,1.5])
    ax4.set_yticklabels(list(map(str, (ax1.get_yticks()*100).astype(int))), size = 16)
    ax4.set_ylabel('propagation speed (% baseline)', size = 16)
    ax4.set_xticks([0,2,5,7,9])
    ax4.set_xticklabels(['-40', '-20', '20', '40', '60'], size = 16)
    ax4.set_xlabel('time from pairing (min)', size = 16)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    plt.tight_layout()
    

# fig1.savefig('SW speed mean.pdf', dpi = 1000, format = 'pdf')
# fig1.savefig('SW speed mean.jpg', dpi = 1000, format = 'jpg')    

# fig4.savefig('SW speed median.pdf', dpi = 1000, format = 'pdf')
# fig4.savefig('SW speed median.jpg', dpi = 1000, format = 'jpg')  



#%% ANGLE UP VS DOWN

bins_to_plot = 15

# fit just using the channels with detected slow waves
fit_detected_only = True

# outliers for slope fit
exclude_outliers = True
outliers_cutoff = 1.5 # outlier channels before fit (how many times IQR of LFP min timepoint across channels)
speed_cutoff = 40 

angle_all_mice_all_sweeps_UP_DOWN = []
resultant_all_mice_all_sweeps_UP_DOWN = []
angles_histogram_all_sweeps_UP_DOWN = []

mice_UP = [0,1,2,3,4,8,9,10,11]
mice_DOWN = [2,3,4,5,6,7,8,9,10,11]

for group in ['UP_pairing', 'DOWN_pairing']:
    os.chdir(overall_path)
    os.chdir(group)
    
    angle_all_mice_all_sweeps = []
    resultant_all_mice_all_sweeps = []
    angles_histogram_all_mice_all_sweeps = []
    
    angle_mean_ALL = np.loadtxt('angle_mean_ALL.csv', delimiter = ',')
    angle_median_ALL = np.loadtxt('angle_median_ALL.csv', delimiter = ',')

    if 'UP_pairing' in os.getcwd():
        mice_to_plot = mice_UP
        color = 'red'
    elif 'DOWN_pairing' in os.getcwd():
        mice_to_plot = mice_DOWN
        color = 'black'

    days = [i for i in os.listdir() if 'not_used' not in i and os.path.isdir(i)]

    for day_ind, day in enumerate(days):
        if day in ['061221', '160218', '160624_B2', '160628_D1', '191121']: # mice not included in LFP cross correlation analysis
            continue
    # for day_ind, day in enumerate(['160427']):
    
        print(day)
        os.chdir(day)
        slope = pickle.load(open(f'slope_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
        angle = pickle.load(open(f'angle_propagation_excloutliers_{exclude_outliers}_cutoff_{outliers_cutoff}_onlydetectfit_{fit_detected_only}.pkl', 'rb'))
        outliers = []
        for sweep in range(10):
            SW = 200/slope[sweep]
            outliers.append(SW > speed_cutoff)
                
        angle_without_outliers = [angle[i][~outliers[i]] for i in range(10)]
        angle_without_outliers = [i[~np.isnan(i)] for i in angle_without_outliers]
        
        angles_histogram_all_mice_all_sweeps.append([np.histogram(angle_without_outliers[sweep], bins = bins_to_plot)[0]/len(angle_without_outliers[sweep]) for sweep in [0,1,2,3,4,5,6,7,8,9]])

        
        angle_all_mice_all_sweeps.append([-pycircstat.mean(i) for i in angle_without_outliers])
        resultant_all_mice_all_sweeps.append([pycircstat.resultant_vector_length(i)/len(i) for i in angle_without_outliers])
        
        os.chdir('..')
        
    angle_all_mice_all_sweeps_UP_DOWN.append(angle_all_mice_all_sweeps)
    resultant_all_mice_all_sweeps_UP_DOWN.append(resultant_all_mice_all_sweeps)
    angles_histogram_all_sweeps_UP_DOWN.append(angles_histogram_all_mice_all_sweeps)
    os.chdir('..')



fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='polar')
UP_before_resultant = np.asarray([np.nanmean(i[0:4]) for i in resultant_all_mice_all_sweeps_UP_DOWN[0]])/4
UP_before_angle = np.asarray([pycircstat.mean(i[0:4]) for i in angle_all_mice_all_sweeps_UP_DOWN[0]])
UP_after_resultant = np.asarray([np.nanmean(i[4:]) for i in resultant_all_mice_all_sweeps_UP_DOWN[0]])/4
UP_after_angle = np.asarray([pycircstat.mean(i[4:]) for i in angle_all_mice_all_sweeps_UP_DOWN[0]])
U_b = ax.errorbar(pycircstat.mean(UP_before_angle), np.mean(UP_before_resultant)*100, xerr = pycircstat.std(UP_before_angle)/np.sqrt(len(mice_UP)), yerr=np.mean(UP_before_resultant)*100/np.sqrt(len(mice_UP)), c="black", linewidth = 4, capsize = 10, marker = 'o', markersize = 15)
U_d = ax.errorbar(pycircstat.mean(UP_after_angle), np.mean(UP_after_resultant)*100, xerr = pycircstat.std(UP_after_angle)/np.sqrt(len(mice_UP)), yerr=np.mean(UP_after_resultant)*100/np.sqrt(len(mice_UP)), c="cyan", linewidth = 4, capsize = 10, marker = 'o', markersize = 15)
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(projection='polar')
theta = np.linspace(np.pi, -np.pi, bins_to_plot) # angles for the plots in radians
width = (2*np.pi)/bins_to_plot
UP_radius_mean_all_before = np.mean(np.asarray([np.mean(np.asarray(i[0:4]), axis = 0) for i in angles_histogram_all_sweeps_UP_DOWN[0]]), axis = 0)
UP_radius_sem_all_before = np.std(np.asarray([np.mean(np.asarray(i[0:4]), axis = 0) for i in angles_histogram_all_sweeps_UP_DOWN[0]]), axis = 0)/np.sqrt(len(mice_UP))
ax.plot(theta, UP_radius_mean_all_before, 'ko')
ax.errorbar(theta, UP_radius_mean_all_before, yerr = UP_radius_sem_all_before, color = 'k')
plt.fill_between(theta, np.repeat(0, len(theta)), UP_radius_mean_all_before, color = 'k', alpha = 0.1)
UP_radius_mean_all_after = np.mean(np.asarray([np.mean(np.asarray(i[4:]), axis = 0) for i in angles_histogram_all_sweeps_UP_DOWN[0]]), axis = 0)
UP_radius_sem_all_after = np.std(np.asarray([np.mean(np.asarray(i[4:]), axis = 0) for i in angles_histogram_all_sweeps_UP_DOWN[0]]), axis = 0)/np.sqrt(len(mice_UP))
ax.plot(theta, UP_radius_mean_all_after, 'co')
ax.errorbar(theta, UP_radius_mean_all_after, yerr = UP_radius_sem_all_after, color = 'c')
plt.fill_between(theta, np.repeat(0, len(theta)), UP_radius_mean_all_after, color = 'c', alpha = 0.1)
plt.tight_layout()
# plt.savefig('SO angles UP before vs after.jpg', format = 'jpg', dpi = 1000)
# plt.savefig('SO angles UP before vs after.pdf', format = 'pdf', dpi = 1000)


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='polar')
DOWN_before_resultant = np.asarray([np.nanmean(i[0:4]) for i in resultant_all_mice_all_sweeps_UP_DOWN[1]])/4
DOWN_before_angle = np.asarray([pycircstat.mean(i[0:4]) for i in angle_all_mice_all_sweeps_UP_DOWN[1]])
DOWN_after_resultant = np.asarray([np.nanmean(i[4:]) for i in resultant_all_mice_all_sweeps_UP_DOWN[1]])/4
DOWN_after_angle = np.asarray([pycircstat.mean(i[4:]) for i in angle_all_mice_all_sweeps_UP_DOWN[1]])
U_b = ax.errorbar(pycircstat.mean(DOWN_before_angle), np.mean(DOWN_before_resultant)*100, xerr = pycircstat.std(DOWN_before_angle)/np.sqrt(len(mice_DOWN)), yerr = np.mean(DOWN_before_resultant)*100/np.sqrt(len(mice_DOWN)), c="black", linewidth = 4, capsize = 10, marker = 'o', markersize = 15)
U_d = ax.errorbar(pycircstat.mean(DOWN_after_angle), np.mean(DOWN_after_resultant)*100, xerr = pycircstat.std(DOWN_after_angle)/np.sqrt(len(mice_DOWN)), yerr = np.mean(DOWN_after_resultant)*100/np.sqrt(len(mice_DOWN)), c="cyan", linewidth = 4, capsize = 10, marker = 'o', markersize = 15)
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(projection='polar')
theta = np.linspace(np.pi, -np.pi, bins_to_plot) # angles for the plots in radians
width = (2*np.pi)/bins_to_plot
DOWN_radius_mean_all_before = np.mean(np.asarray([np.mean(np.asarray(i[0:4]), axis = 0) for i in angles_histogram_all_sweeps_UP_DOWN[1]]), axis = 0)
DOWN_radius_sem_all_before = np.std(np.asarray([np.mean(np.asarray(i[0:4]), axis = 0) for i in angles_histogram_all_sweeps_UP_DOWN[1]]), axis = 0)/np.sqrt(len(mice_DOWN))
ax.plot(theta, DOWN_radius_mean_all_before, 'ko')
ax.errorbar(theta, DOWN_radius_mean_all_before, yerr = DOWN_radius_sem_all_before, color = 'k')
plt.fill_between(theta, np.repeat(0, len(theta)), DOWN_radius_mean_all_before, color = 'k', alpha = 0.1)
DOWN_radius_mean_all_after = np.mean(np.asarray([np.mean(np.asarray(i[4:]), axis = 0) for i in angles_histogram_all_sweeps_UP_DOWN[1]]), axis = 0)
DOWN_radius_sem_all_after = np.std(np.asarray([np.mean(np.asarray(i[4:]), axis = 0) for i in angles_histogram_all_sweeps_UP_DOWN[1]]), axis = 0)/np.sqrt(len(mice_DOWN))
ax.plot(theta, DOWN_radius_mean_all_after, 'co')
ax.errorbar(theta, DOWN_radius_mean_all_after, yerr = DOWN_radius_sem_all_after, color = 'c')
plt.fill_between(theta, np.repeat(0, len(theta)), DOWN_radius_mean_all_after, color = 'c', alpha = 0.1)
plt.tight_layout()
# plt.savefig('SO angles DOWN before vs after.jpg', format = 'jpg', dpi = 1000)
# plt.savefig('SO angles DOWN before vs after.pdf', format = 'pdf', dpi = 1000)

angle_for_ANOVA_UP = np.vstack((UP_before_angle, UP_after_angle))
angle_for_ANOVA_DOWN = np.vstack((DOWN_before_angle, DOWN_after_angle))

# np.savetxt('angle_for_ANOVA_UP.csv', angle_for_ANOVA_UP.T, delimiter = ',')
# np.savetxt('angle_for_ANOVA_DOWN.csv', angle_for_ANOVA_DOWN.T, delimiter = ',')

# th_err = 1
# for th,  _r in zip(theta, r):
#     local_theta = np.linspace(-th_err, th_err, 15) + th
#     local_r = np.ones(15) * _r
#     ax.plot(local_theta, local_r, color='k', marker='')



#%% R2 UP vs DOWN

mice_UP = [0,1,2,3,4,8,9,10,11]
mice_DOWN = [2,3,4,5,6,7,8,9,10,11]
def normalize(array):
    return(np.transpose(np.transpose(array)/np.nanmean(array[:,[0,1,2,3]], axis = 1)))

# fig1, ax1 = plt.subplots(figsize = (10,4))
# fig1.suptitle('speed mean')
# fig2, ax2 = plt.subplots(figsize = (10,6))
# fig2.suptitle('angle mean')
# fig3, ax3 = plt.subplots(figsize = (10,6))
# fig3.suptitle('R2 mean')
# fig4, ax4 = plt.subplots(figsize = (10,4))
# fig4.suptitle('speed median')
# fig5, ax5 = plt.subplots(figsize = (10,6))
# fig5.suptitle('angle median')
# fig6, ax6 = plt.subplots(figsize = (10,6))
# fig6.suptitle('R2 median')

for group in ['UP_pairing', 'DOWN_pairing']:
    os.chdir(overall_path)
    os.chdir(group)
    R2_mean_ALL = np.loadtxt('R2_mean_ALL.csv', delimiter = ',')
    R2_median_ALL = np.loadtxt('R2_median_ALL.csv', delimiter = ',')

    if 'UP_pairing' in os.getcwd():
        mice_to_plot = mice_UP
        color = 'red'
        R2_before_UP = np.mean(R2_mean_ALL[:,[0,1,2,3]], axis = 1)
        R2_after_UP = np.mean(R2_mean_ALL[:,[4,5,6,7,8,9]], axis = 1)

    elif 'DOWN_pairing' in os.getcwd():
        mice_to_plot = mice_DOWN
        color = 'black'
        R2_before_DOWN = np.mean(R2_mean_ALL[:,[0,1,2,3]], axis = 1)
        R2_after_DOWN = np.mean(R2_mean_ALL[:,[4,5,6,7,8,9]], axis = 1)
os.chdir('..')

print(scipy.stats.ttest_rel(R2_before_UP[~np.isnan(R2_before_UP)], R2_after_UP[~np.isnan(R2_before_UP)]))
print(scipy.stats.ttest_rel(R2_before_DOWN[~np.isnan(R2_before_DOWN)], R2_after_DOWN[~np.isnan(R2_before_DOWN)]))

fig, ax = plt.subplots()
ax.errorbar([0,1], [np.nanmean(R2_before_UP), np.nanmean(R2_after_UP)], yerr = [np.nanstd(R2_before_UP)/np.sqrt(12), np.nanstd(R2_after_UP)/np.sqrt(12)], color = 'r')
ax.errorbar([0,1], [np.nanmean(R2_before_DOWN[2:]), np.nanmean(R2_after_DOWN[2:])], yerr = [np.nanstd(R2_before_DOWN[2:])/np.sqrt(12), np.nanstd(R2_after_DOWN[2:])/np.sqrt(12)], color = 'k')

fig, ax = plt.subplots(figsize = (2,4))
ax.plot([np.repeat(1,len(R2_before_UP)), np.repeat(2,len(R2_before_UP))], [R2_before_UP, R2_after_UP], color = 'k', linewidth = 1)
ax.scatter(np.concatenate([np.repeat(1,len(R2_before_UP)), np.repeat(2,len(R2_before_UP))]), np.concatenate([R2_before_UP, R2_after_UP]), color = 'k')
ax.tick_params(axis = 'y', labelsize = 18)
ax.set_xticklabels([])
ax.set_xlim([0.5, 2.5])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# plt.savefig('R2 pre vs post pairing UP.jpg', format = 'jpg', dpi = 1000)
# plt.savefig('R2 pre vs post pairing UP.pdf', format = 'pdf', dpi = 1000)

fig, ax = plt.subplots(figsize = (2.5,4))
ax.plot([np.repeat(1,len(R2_before_DOWN[2:])), np.repeat(2,len(R2_before_DOWN[2:]))], [R2_before_DOWN[2:], R2_after_DOWN[2:]], color = 'k', linewidth = 1)
ax.scatter(np.concatenate([np.repeat(1,len(R2_before_DOWN[2:])), np.repeat(2,len(R2_before_DOWN[2:]))]), np.concatenate([R2_before_DOWN[2:], R2_after_DOWN[2:]]), color = 'k')
ax.set_yticks([0.4, 0.5, 0.6])
ax.tick_params(axis = 'y', labelsize = 18)
ax.set_xticklabels([])
ax.set_xlim([0.5, 2.5])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# plt.savefig('R2 pre vs post pairing DOWN.jpg', format = 'jpg', dpi = 1000)
# plt.savefig('R2 pre vs post pairing DOWN.pdf', format = 'pdf', dpi = 1000)


R2_for_ANOVA_UP = np.vstack((R2_before_UP, R2_after_UP))
R2_for_ANOVA_DOWN = np.vstack((R2_before_DOWN, R2_after_DOWN))
np.savetxt('R2_for_ANOVA_UP.csv', R2_for_ANOVA_UP.T, delimiter = ',')
np.savetxt('R2_for_ANOVA_DOWN.csv', R2_for_ANOVA_DOWN.T, delimiter = ',')
