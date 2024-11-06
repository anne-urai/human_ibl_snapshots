"""
Created on October 16 2024
@author: phijoh
Leiden University

requirements: ibllib https://int-brain-lab.github.io/iblenv/notebooks_external/data_download.html
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from io import StringIO
#import h5py
import cv2
import mne

import time

from pathlib import Path
from tqdm import tqdm

import utils

from audio_extract import extract_audio
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
from scipy import signal

from datetime import datetime, timedelta

def run_fft(data, samplerate):
    d = 1/samplerate
    y = fft(data)
    n = len(data)
    power = 2.0/n * np.abs(y[0:n//2])
    freqs = fftfreq(n, d)[:n//2]
    return power, freqs

def find_audio_onsets(spectogram_data, onset_freq_index, power_threshold, minimum_gap):
    impulse_inds = np.where(spectogram_data[onset_freq_index,:]>power_threshold)[0]
    audio_onsets = [impulse_inds[0]]
    for i,x in enumerate(impulse_inds[1:]):
        if x - impulse_inds[i] < minimum_gap:
            continue
        else:
            audio_onsets.append(x)
    return np.array(audio_onsets)

def detect_freq_onsets(data:np.ndarray, samplerate:float, target_freq:float, min_seconds_gap:float, power_thresh:float):
    """
    detect onsets at a certain frequency within timeseries data
    """    

    # short time FFT
    f, t, Sxx = signal.spectrogram(data, samplerate)
    spectogram_samplerate = len(t)/len(data_short)*samplerate

    # find audio_onsets of groups
    onset_f_ind = np.argmin(np.abs(f-target_freq))
    onsets = find_audio_onsets(Sxx, onset_f_ind, power_thresh, spectogram_samplerate*min_seconds_gap) 

    return np.array(onsets), t

def extract_onsets(Fs, ys, ms=80, threshold=0.0225):
    ''' Onset extraction from high SNR audio signal.
    
    See p368/369 (p10/11) of
        https://www.cell.com/cms/10.1016/j.cub.2016.12.031/attachment/9fc19c51-d255-43a6-800d-2d4a993155e8/mmc1.pdf
    '''
    ms_gap = int(ms * (Fs/1000.0))
    hold = float('inf')
 
    # Get the power of the signal to increase SNR
    ys = np.abs(ys)**2
 
    # Get the threshold from the average of 20 highest peaks
    sig_sorted = np.sort(np.abs(ys))
    thresh = np.mean(sig_sorted[-20:]) * threshold
 
    xs = []
    for i, y in enumerate(np.abs(ys)):
        if hold < ms_gap:
            hold += 1
            continue
        elif hold and y > thresh:
            hold = 0
            xs.append(i/Fs)
    return (xs, thresh)

def find_extra_onsets(onset_list_1, onset_list_2):
    # compare lists
    set_1 = set(np.round(onset_list_1))
    set_2 = set(np.round(onset_list_2))
    extra_set_1 = list(sorted(set_1 - set_2))
    extra_set_2 = list(sorted(set_2 - set_1))

    extra_detections = []
    for i in extra_set_1:
        diff = extra_set_2 - i
        if (diff==1).any() or (diff==-1).any():
            continue
        else:
            extra_detections.append(i)

    extra_onset_ind = [np.argmin(np.abs(audio_onsets_shifted-i)) for i in extra_detections]

    return extra_onset_ind

def make_mne_events_array(event_times_ms:np.ndarray, event_code):
    return np.vstack((np.rint(event_times_ms), np.zeros_like(event_times_ms, dtype=int), np.full_like(event_times_ms, event_code))).T

onset_freq = 5000
correct_freq = 2000
timeout_freq = 567
min_seconds_gap = 0.1

folder_path = Path('data/subjects')
sub_names = os.listdir(folder_path)
for sub in sub_names:
    print(f'processing sub {sub}...')
    data_dir = folder_path / sub
    all_files = os.listdir(data_dir)

    # load trials info
    downloaded_files = utils.get_files_from_folder(data_dir)
    for file_name, file_content in downloaded_files:
        trials_data = pd.read_csv(StringIO(file_content))

    # convert to ONE convention
    trials_df = utils.convert_psychopy_one(trials_data, file_name)
    n_trials = len(trials_df)

    # find time that video starts
    vid_filename = [x for x in all_files if '.mkv' in x]
    vid_path = folder_path / sub / vid_filename[0]
    vid_start_str = str(vid_path)[-12:-4]
    vid_start_time = datetime.strptime(vid_start_str, '%H-%M-%S')

    # extract audio if not done
    audio_path = f'{str(vid_path)[:-3]}wav'
    if audio_path.split('\\')[-1]  not in all_files:
        extract_audio(input_path=vid_path, output_path=audio_path, output_format='wav', overwrite=False)

    # load audio
    samplerate, data = wavfile.read(audio_path)
    data = data.astype('float')

    # find time that experiment starts
    exp_start_str = trials_data[trials_data['session_start'].notna()]['session_start'].unique()[0][-12:-1]
    exp_start_time = datetime.strptime(exp_start_str, '%Hh%M.%S.%f') - timedelta(seconds=5)
    if 'session_end' in trials_df.keys():
        exp_end_str = trials_data[trials_data['session_end'].notna()]['session_end'].unique()[0][-12:]
        exp_end_time = datetime.strptime(exp_end_str, '%Hh%M.%S.%f') + timedelta(seconds=5)
    else:
        exp_end_time = exp_start_time + timedelta(minutes=np.floor(len(data)/samplerate/60))
    exp_dur = (exp_end_time-exp_start_time).total_seconds()

    # cut off unnecessary data
    cutoff_start_s = (exp_start_time-vid_start_time).total_seconds()
    cutoff_end_s = cutoff_start_s + exp_dur
    data_short = data[int(cutoff_start_s*samplerate,):int(cutoff_end_s*samplerate),0]

    audio_onsets, t = detect_freq_onsets(data_short, samplerate, onset_freq, min_seconds_gap, 8000)
    audio_onsets_relative_to_start = t[audio_onsets] + cutoff_start_s

    audio_onsets_shifted = t[audio_onsets]-t[audio_onsets[0]]

    grating_onsets = trials_df['sound_trial_start.started'].values
    grating_onsets_shifted = grating_onsets-grating_onsets[0]

    if len(audio_onsets) > len(grating_onsets):
        extra_inds = find_extra_onsets(audio_onsets_shifted, grating_onsets_shifted)
        audio_onsets = np.delete(audio_onsets, extra_inds)
        audio_onsets_shifted = t[audio_onsets]-t[audio_onsets[0]]

    # check equal number of onsets and onsets within 100ms
    if len(grating_onsets) == len(audio_onsets):
        print(f'success: {len(grating_onsets)} onsets detected')
        if not np.isclose(audio_onsets_shifted, grating_onsets_shifted, atol=0.15).all():
            print('detected onsets not within 150ms of psychopy onsets')
        else:
            max_offset = np.round(np.max(np.abs(audio_onsets_shifted-grating_onsets_shifted))*1000, decimals=1)
            print(f'maximum offset between psychopy and audio is {max_offset} ms')
    else:
        print(f'audio {len(audio_onsets)} onsets, psychopy {len(grating_onsets)} onsets')

    make_plots = True
    if make_plots:
            
        timepoints = np.arange(0, len(data_short)/samplerate, 1/samplerate)
        seconds = pd.to_datetime(timepoints, unit='s')

        len_segment_s = 60
        segment_start = 10*60*samplerate
        len_segment_samples = samplerate*len_segment_s # 10 secs
        segment = data_short[segment_start:segment_start+len_segment_samples]
        segment_seconds = seconds[segment_start:segment_start+len_segment_samples]

        correct_feedback_times = t[audio_onsets[trials_df['feedbackType']==1]] + trials_df[trials_df['feedbackType']==1]['response_time']
        error_feedback_times = t[audio_onsets[trials_df['feedbackType']==-1]] + trials_df[trials_df['feedbackType']==-1]['response_time']

        # convert to mne for easy epoching
        mne_info = mne.create_info(ch_names=['audio', 'stim'], sfreq=samplerate, ch_types=['misc', 'stim'])
        raw = mne.io.RawArray(np.vstack((data_short,np.zeros_like(data_short))), mne_info)
        # raw_downsampled = raw.copy().resample(sfreq=5000)

        onset_events = make_mne_events_array(np.rint(t[audio_onsets]*samplerate), 1)
        correct_events = make_mne_events_array(np.rint(t[audio_onsets[trials_df['feedbackType']==1]]*samplerate+trials_df[trials_df['feedbackType']==1]['response_time']*samplerate), 2)
        error_events = make_mne_events_array(np.rint(t[audio_onsets[trials_df['feedbackType']==-1]]*samplerate+trials_df[trials_df['feedbackType']==-1]['response_time']*samplerate), 3)
        
        all_events = np.vstack((onset_events, correct_events, error_events))
        raw.add_events(all_events, stim_channel='stim')
        events = mne.find_events(raw, stim_channel='stim')  
        fb_event_dict = {'correct':2, 'error':3}   
        onset_event_dict = {'onset':1}  
        onset_epochs = mne.Epochs(raw, events=events, event_id=onset_event_dict, tmin=-.1, tmax=.5, baseline=None, reject=None)
        fb_epochs = mne.Epochs(raw, events=events, event_id=fb_event_dict, tmin=-.1, tmax=.5, baseline=None, reject=None)
        # epochs_spec = onset_epochs.compute_tfr(picks='audio', method='morlet', freqs=np.arange(3600, 5600, 100))
        # epochs_spec.average().plot(picks='audio') 
        onset_epochs_df = onset_epochs.to_data_frame()
        fb_epochs_df = fb_epochs.to_data_frame()

        sns.set_style('darkgrid')
        fig, ax = plt.subplot_mosaic([['A','A'],['B', 'C'],['B','D'],['B','E']], height_ratios=[0.4, 0.2, 0.2, 0.2], layout='constrained', figsize=(12,8))

        offsets = audio_onsets_shifted - grating_onsets_shifted
        ax['B'].scatter(range(len(offsets)), offsets*1000, c='purple') 
        ax['B'].set_xlabel('trial')
        ax['B'].set_ylabel('offset (audio - psychopy) [ms]')   
        
        ax['A'].plot(segment_seconds, segment)
        ax['A'].set_title('audio snippet')
        ax['A'].set_xlabel('time from session start (mm:ss)')
        ax['A'].set_ylabel('audio amplitude')
        ax['A'].vlines(pd.to_datetime(t[audio_onsets], unit='s'), ymin=np.min(segment), ymax=np.max(segment), color='k', linestyle=':', label='onset')
        ax['A'].vlines(pd.to_datetime(correct_feedback_times, unit='s'), ymin=np.min(segment), ymax=np.max(segment), color='g', linestyle=':', label='correct')
        ax['A'].vlines(pd.to_datetime(error_feedback_times, unit='s'), ymin=np.min(segment), ymax=np.max(segment), color='r', linestyle=':', label='error')
        ax['A'].set_xlim(segment_seconds[0], segment_seconds[-1])
        ax['A'].xaxis.set_major_formatter(mdates.DateFormatter("%M:%S"))
        ax['A'].margins(x=0)
        ax['A'].legend()

        sns.lineplot(onset_epochs_df.groupby(['time','condition']).mean(), x='time',y='audio', color='k', ax=ax['C'])
        sns.lineplot(fb_epochs_df[fb_epochs_df['condition']=='correct'], x='time',y='audio', estimator='mean', errorbar=None, color='green', ax=ax['D'])
        sns.lineplot(fb_epochs_df[fb_epochs_df['condition']=='error'], x='time',y='audio', estimator='mean', errorbar=None, color='red', ax=ax['E'])
        ax['C'].set_title('locked to onset')        
        ax['D'].set_title('locked to correct')        
        ax['E'].set_title('locked to error')        
        ax['C'].sharex(ax['D'])
        ax['D'].sharex(ax['E'])
        ax['E'].set_xlabel('time from event (s)')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        sns.despine(trim=True)
        plt.show()
    # correct_onsets, _ = detect_freq_onsets(data_short, samplerate, correct_freq, min_seconds_gap, 5000)
    # n_correct = (trials_df['feedbackType']==1).sum()
    
    # feedback_times = trials_df[trials_df['feedbackType']==1]['feedback_sound.started'].values
    # feedback_times_shifted = feedback_times - feedback_times[0]

    # correct_onsets_shifted = t[audio_onsets]-t[audio_onsets[0]]

    # if len(correct_onsets) > len(feedback_times):
    #     extra_inds = find_extra_onsets(correct_onsets_shifted, feedback_times_shifted)
    #     feedback_times = np.delete(feedback_times, extra_inds)
    #     feedback_times_shifted = t[feedback_times_shifted]-t[feedback_times_shifted[0]]

    # if n_correct == len(correct_onsets):
    #     print(f'success: {n_correct} onsets detected')
    #     if not np.isclose(correct_onsets_shifted, feedback_times_shifted, atol=0.1).all():
    #         print('detected onsets not within 100ms of psychopy onsets')
    # else:
    #     print(f'audio {len(correct_onsets)} onsets, psychopy {n_correct} onsets')

timepoints = np.arange(0, len(data_short)/samplerate, 1/samplerate)
seconds = pd.to_datetime(timepoints, unit='s')

len_segment_s = 60
len_segment_samples = samplerate*len_segment_s # 10 secs
segment = data_short[:len_segment_samples]
segment_seconds = seconds[:len_segment_samples]

plot_segment = False
if plot_segment:
    fig, ax = plt.subplots()
    ax.plot(segment_seconds, np.square(segment))
    ax.set_title('Original Signal (Time Domain)')
    ax.set_xlabel('Time [secs]')
    ax.set_ylabel('Amplitude')
    ax.vlines(pd.to_datetime(t[audio_onsets], unit='s'), ymin=np.min(segment), ymax=np.max(segment), color='k', linestyle=':', label='onset')
    ax.vlines(pd.to_datetime(correct_feedback_times, unit='s'), ymin=np.min(segment), ymax=np.max(segment), color='g', linestyle=':', label='correct')
    ax.vlines(pd.to_datetime(error_feedback_times, unit='s'), ymin=np.min(segment), ymax=np.max(segment), color='r', linestyle=':', label='error')
    ax.set_xlim(0, segment_seconds[-1])
    ax.margins(x=0)
    plt.show()

# get the threshold from the segment
ys = np.square(data_short)
y_sorted = np.sort(ys)
n_onsets = len(grating_onsets_shifted[grating_onsets_shifted<len_segment_s])
thresh = np.mean(y_sorted[-20:]) * .95
min_gap = int(trials_df['response_time'].min()*samplerate)
sample = np.random.choice(ys, 1_000_000, replace=False)
plt.hist(sample)
plt.show()

# apply to the whole data
xs=np.full_like(ys, False)
candidates = ys > thresh
i = 0
while i < len(data_short):
    if candidates[i]:
        xs[i] = True
        i = i + min_gap
    else:
        i += 1
np.sum(xs==True)

hold = float('inf')
for i, y in enumerate(np.square(data_short)):
    if hold < min_gap:
        hold += 1
    elif y > thresh:
        hold = 0
        xs.append(i/samplerate)


run_fft = False
if run_fft:
    # fft signal segment
    power, xf = run_fft(data_short, samplerate)
    peaks, properties = signal.find_peaks(power, prominence=12)

    # plot power spectrum
    plt.plot(xf, power, color='blue')
    # plt.scatter(xf[peaks], power[peaks], c='orange',marker='x')
    plt.axvline(onset_freq,0,1,linestyle=':', color='deeppink') # onset
    plt.annotate('onset', (onset_freq, np.max(power)), color='deeppink')
    plt.axvline(timeout_freq,0,1,linestyle=':',color='red') # timeout
    plt.annotate('timeout', (timeout_freq, np.max(power)*0.8), color='red')
    plt.axvline(2000,0,1,linestyle=':',color='purple') # correct
    plt.annotate('correct', (correct_freq, np.max(power)*0.9), color='purple')
    plt.show()

# # fft of white noise error tone
# samplerate_error, error_tone = wavfile.read('data/whitenoise.wav')
# error_power, error_freqs = run_fft(error_tone, samplerate_error)

# # plot power spectrum of error tone
# plt.plot(error_freqs, error_power, color='blue')
# plt.axvline(onset_freq,0,1,linestyle=':', color='deeppink') # onset
# plt.annotate('onset', (onset_freq, np.max(error_power)), color='deeppink')
# plt.axvline(timeout_freq,0,1,linestyle=':',color='red') # timeout
# plt.annotate('timeout', (timeout_freq, np.max(error_power)*0.9), color='red')
# plt.axvline(2000,0,1,linestyle=':',color='purple') # correct
# plt.annotate('correct', (correct_freq, np.max(error_power)*0.95), color='purple')
# plt.show()

# short time FFT
f, t, Sxx = signal.spectrogram(data_short, samplerate)
spectogram_samplerate = len(t)/len(data_short)*samplerate

# plot a segment of spectrogram
plt.pcolormesh(t[:], f[:30], Sxx[:30,:], shading='gouraud')
plt.xlim(1350,1400)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# find audio_onsets of groups
onset_f_ind = np.argmin(np.abs(f-onset_freq))
other_fs = np.where(f!=f[onset_f_ind])
other_f_ind = np.argmin(np.abs(f-onset_freq-500))
audio_onsets = find_audio_onsets(Sxx, onset_f_ind, 8000, spectogram_samplerate/10) # FIXME: hardcoded threshold for finding audio_onsets 

audio_onsets_shifted = t[audio_onsets]-t[audio_onsets[0]]

grating_onsets = trials_df['sound_trial_start.started'].values
grating_onsets_shifted = grating_onsets-grating_onsets[0]

# check equal number of onsets
assert len(grating_onsets) == len(audio_onsets), 'different number of onsets detected'

# check onsets are within 100ms of each other
assert np.isclose(audio_onsets_shifted, grating_onsets_shifted, atol=0.1).all(), 'detected onsets not within 100ms of psychopy onsets'

# plot only power of onset frequency
fig,ax = plt.subplots()
plt.plot(t, Sxx[onset_f_ind,:])#-Sxx[other_f_ind,:])
plt.vlines(t[audio_onsets],0,np.max(Sxx[onset_f_ind,:]) ,color='k', linestyles=':')
plt.xlim(1350,1400)
plt.show()

# extra_onset = 1171


import scipy.ndimage
import scipy.signal

from ibllib.io.extractors.training_audio import welchogram, extract_sound
detect_kwargs = {'threshold':0.8}
tscale, fscale, W, onsets = welchogram(samplerate, data_short, detect_kwargs=detect_kwargs) # does not work

# grating_idx = 0
# audio_idx = 0
# extra_onsets = []
# while (audio_idx <= len(extra_audio) & grating_idx <= len(extra_grating)):
#     if np.isclose(extra_audio[audio_idx], extra_grating[grating_idx], atol=1):
#         grating_idx += 1
#         audio_idx += 1
#     else:
#         extra_onsets.append(extra_audio[audio_idx])

plt.scatter(audio_onsets_shifted, grating_onsets_shifted)
plt.show()

# overlay onset times
fig,ax = plt.subplots()
plt.vlines(audio_onsets_shifted, 0, 1, color='b', linestyles=':')
plt.vlines(grating_onsets_shifted, 0, 1 ,color='r', linestyles=':')
plt.xlim(100,110)
plt.show()

# plot histogram of itis
itis_trials = np.diff(grating_onsets)
spectogram_timepoints = np.arange(0,len(t)/spectogram_samplerate,step=1/spectogram_samplerate)
itis_audio = np.diff(spectogram_timepoints[audio_onsets])
plt.hist(itis_trials, range=(0,7), bins=14, alpha=0.5, color='darkseagreen', label='trials')
plt.hist(itis_audio, range=(0,7), bins=14, alpha=0.4, color='steelblue', label='audio')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(grating_onsets, t[audio_onsets])
ax.axline((0, 0), slope=1, linestyle=':', color='k')
plt.show()

### repeat for correct
# find audio_onsets of groups
correct_onsets, t = detect_freq_onsets(data_short, samplerate, correct_freq, 0.1, 6000)

correct_f_ind = np.argmin(np.abs(f-correct_freq))
audio_onsets_correct = find_audio_onsets(Sxx, correct_f_ind, 5000, spectogram_samplerate/10) # FIXME: hardcoded threshold for finding correct tones

# plot only power of onset frequency
fig,ax = plt.subplots()
plt.plot(t, Sxx[correct_f_ind,:])
plt.vlines(t[audio_onsets_correct],0,np.max(Sxx[correct_f_ind,:]) ,color='k', linestyles=':')
plt.show()

# check itis against trials table
correct_in_exp = audio_onsets_correct[t[audio_onsets_correct] > 200] # FIXME: hardcoded time where experiment starts



if False:
    ### other SFFT approach
    w = signal.windows.gaussian(50, std=8, sym=True)  # symmetric Gaussian window
    SFT = signal.ShortTimeFFT(w, hop=10, fs=samplerate, mfft=200, scale_to='magnitude')
    Sx = SFT.stft(segment)

    # bandpass filter for onset
    order = 6
    hp_cutoff = xf[onset_freq-1000] # desired cutoff frequency of the filter, Hz
    lp_cutoff = xf[onset_freq+1000]

    hp_filtered_audio = butter_highpass_filter(data[:,0], hp_cutoff, samplerate, order)
    bp_filtered_audio = butter_lowpass_filter(hp_filtered_audio, lp_cutoff, samplerate, order)

    coefficients = signal.cwt(bp_filtered_audio, wavelet=signal.morlet, widths=[1]) # what is the width here?

    # plot bandpassed signal
    fig, ax = plt.subplots(2,1,figsize=(11, 5), sharex=False)
    ax[0].plot(seconds, bp_filtered_audio)
    ax[0].set_title('Original Signal (Time Domain)')
    ax[0].set_xlabel('Time [mins]')
    ax[0].set_ylabel('Amplitude')
    ax[0].margins(x=0)

    ax[1].imshow(np.abs(coefficients), cmap='PRGn', aspect='auto',
            vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
    ax[1].set_title('Wavelet Transform (Time-Frequency Domain)')
    ax[1].set_xlabel('Time [samples]')
    ax[1].set_ylabel('Scale')
    plt.tight_layout()
    plt.show()

    # select places where coeffs are 
    impulse_inds = np.where(np.abs(coefficients)>100)[1] # TODO: check trial table for how many trials were actually shown
    plt.plot(segment_seconds, segment)
    plt.scatter(segment_seconds[impulse_inds], segment[impulse_inds],marker='x', c='orange')
    plt.show()

    # diff = np.diff(impulse_inds)
    # groups = [[impulse_inds[0]]]
    # for x in impulse_inds[1:]:
    #     if x - groups[-1][0] < min_gap:
    #         groups[-1].append(x)
    #     else:
    #         groups.append([x])

    min_gap = samplerate/10 # 200ms
    impulse_inds = np.where(np.abs(coefficients)>150)[1]
    audio_onsets = [impulse_inds[0]]
    for i,x in enumerate(impulse_inds[1:]):
        if x - impulse_inds[i] < min_gap:
            continue
        else:
            audio_onsets.append(x)

    plt.plot(segment_seconds, segment)
    plt.scatter(segment_seconds[audio_onsets], segment[audio_onsets],marker='x', c='orange', s=100, zorder=3)
    plt.show()


    order = 6
    last_peak = peaks[-1] # 16666 for feedback
    hp_cutoff = xf[last_peak-10] # desired cutoff frequency of the filter, Hz
    lp_cutoff = xf[last_peak+10]

    hp_filtered_audio = butter_highpass_filter(data[:,0], hp_cutoff, samplerate, order)
    bp_filtered_audio = butter_lowpass_filter(hp_filtered_audio, lp_cutoff, samplerate, order)



    from matplotlib.ticker import AutoMinorLocator

    peaks, properties = signal.find_peaks_cwt(segment,widths=np.arange(1,5))

    fig, ax = plt.subplots()
    ax.plot(segment_seconds, segment)
    ax.scatter(segment_seconds[peaks], segment[peaks], c='orange',marker='x')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(visible=True,which='both')
    plt.show()


    # there is a stimulus at 7mins 9secs

    ### video stuff



    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(str(vid_path)))

    # coordinates of mirror - 400:600,1400:1560

    # Check if camera opened successfully
    if (capture.isOpened()== False): 
        print("Error opening video stream or file")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) 
    print(f'Total number of frames: {total_frames}')
    capture.release()

    
    # contrast_list = []
    # # Read until video is completed
    # while(capture.isOpened()):
    #   # Capture frame-by-frame
    #   ret, frame = capture.read()
    #   if ret == True:
    
    #     # Display the resulting frame
    #     # cv2.imshow('Frame',frame)

    #     # get contrast of mirrow
    #     mirror = frame[400:600,1400:1560,:]
    #     mirror_bw = Y = cv2.cvtColor(mirror, cv2.COLOR_BGR2YUV)[:,:,0]

    #     # compute min and max of Y
    #     min = np.min(mirror_bw)
    #     max = np.max(mirror_bw)

    #     # compute contrast
    #     contrast = (max-min)/(max+min)
    #     contrast_list.append(contrast)
    
    #     # Press Q on keyboard to  exit
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #       break
    
    #   # Break the loop
    #   else: 
    #     break


    # contrast_list = np.empty((total_frames, 1))
    # frame_list = []
    # start = time.time()
    # for frame_number in range(total_frames):
    #     _ = capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
    #     res, frame = capture.read()

    #     mirror = frame[400:600,1400:1560,:]
    #     mirror_bw = cv2.cvtColor(mirror, cv2.COLOR_BGR2YUV)[:,:,0]

    #     # compute min and max of Y
    #     min = np.min(mirror_bw)
    #     max = np.max(mirror_bw)

    #     # compute contrast
    #     contrast = (max-min)/(max+min)
    #     contrast_list[frame_number] = contrast

    #     frame_list.append(mirror)
    # end = time.time()
    # length = end-start

    # print(f"It took {length} segment_seconds, that's {length/200} secs per frame")
    # cv2.imshow('Frame', frame_list[np.argmin(contrast_list)])


    import imageio.v3 as iio

    # with iio.imopen(str(vid_path), "r", plugin="pyav") as file:
    #         n_frames = file.properties().shape[0]
    #         read_indices = np.linspace(0, n_frames-1, 10, dtype=int)
    #         for count, idx in enumerate(read_indices):
    #             frame = file.read(index=idx)
    #             mirror = frame[400:600,1400:1560,:]

    ## this is the quick one
    contrast_list = np.empty((total_frames, 1))
    for idx, frame in enumerate(tqdm(iio.imiter(str(vid_path)))):

        mirror = frame[400:600,1400:1560,:]
        mirror_bw = cv2.cvtColor(mirror[50:125, 25:125], cv2.COLOR_BGR2YUV)[:,:,0]

        # compute min and max of Y
        min = np.min(mirror_bw)
        max = np.max(mirror_bw)

        # compute contrast
        contrast = (max-min)/(max+min)
        contrast_list[idx] = contrast

        if idx == 1000:
            break


    frame = iio.imread(vid_path, index=30200)
    plt.imshow(frame)
    plt.imshow(frame[400:600, 1400:1560])
    plt.imshow(cv2.cvtColor(frame[450:580, 1430:1530], cv2.COLOR_BGR2YUV)[:,:,0])
    plt.show()

    plt.plot(contrast_list)
    plt.ylim(0,.5)
    plt.xlabel('video frame')
    plt.ylabel('contrast')
    plt.show()

    dst = cv2.cornerHarris(cv2.cvtColor(mirror, cv2.COLOR_BGR2YUV)[:,:,0],2,3,0.04)
    dst = cv2.dilate(dst,None)

    corners = mirror.copy()
    corners[dst>0.01*dst.max()]=[0,0,255]
    cv2.imshow('dst',corners)

    subframe = frame[:,1250:,:]
    edges = cv2.Canny(subframe,100,200)
    
    plt.subplot(121),plt.imshow(subframe,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
    plt.show()


