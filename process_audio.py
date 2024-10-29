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

import time

from pathlib import Path
from tqdm import tqdm

import utils

from audio_extract import extract_audio
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from matplotlib.ticker import AutoMinorLocator
from scipy import signal

from datetime import datetime

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_highpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='high', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

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

folder_path = Path('data/subjects')
sub_names = os.listdir(folder_path)
for sub in sub_names:
    data_dir = folder_path / sub
    data = os.listdir(data_dir)

# load trials info
downloaded_files = utils.get_files_from_folder(data_dir)
for file_name, file_content in downloaded_files:
    trials_data = pd.read_csv(StringIO(file_content))
exp_start_str = trials_data[trials_data['session_start'].notna()]['session_start'].unique()[0][-12:-1]
exp_end_str = trials_data[trials_data['session_end'].notna()]['session_end'].unique()[0][-12:]
exp_start_time = datetime.strptime(exp_start_str, '%Hh%M.%S.%f')
exp_end_time = datetime.strptime(exp_end_str, '%Hh%M.%S.%f')
exp_dur = (exp_end_time-exp_start_time).total_seconds()

# time_last_practice = trials_data[trials_data['trial_practice.stopped'].notna()]['trial_practice.stopped'].tail(1).values[0]
# start_session_time = trials_data[trials_data['start_session.started'].notna()]['start_session.started']
# end_session_time = trials_data[trials_data['start_session.stopped'].notna()]['start_session.stopped']

# convert to ONE convention
trials_df = utils.convert_psychopy_one(trials_data, file_name)
n_trials = len(trials_df)

vid_path = folder_path / sub / data[-1]
audio_path = f'{str(vid_path)[:-3]}wav'
vid_start_str = str(vid_path)[-12:-4]
vid_start_time = datetime.strptime(vid_start_str, '%H-%M-%S')

cutoff_start_s = (exp_start_time-vid_start_time).total_seconds()
cutoff_end_s = cutoff_start_s + exp_dur

extract_audio(input_path=vid_path, output_path=audio_path, output_format='wav', overwrite=False)

# load audio
samplerate, data = wavfile.read(audio_path)

data_short = data[int(cutoff_start_s*samplerate,):int(cutoff_end_s*samplerate),0]
timepoints = np.arange(0,len(data_short)/samplerate,step=1/samplerate)
seconds = pd.to_datetime(timepoints, unit='s')

onset_freq = 5000
correct_freq = 2000
timeout_freq = 567

# take a segment of data
len_segment = samplerate*10 # 10 secs
segment = data_short[:len_segment]
segment_seconds = seconds[:len_segment]

# plot segment
# coefficients = signal.cwt(segment, wavelet=signal.morlet, widths=[2])
fig, ax = plt.subplots()
ax.plot(segment_seconds, segment)
ax.set_title('Original Signal (Time Domain)')
ax.set_xlabel('Time [mins]')
ax.set_ylabel('Amplitude')
ax.margins(x=0)
plt.show()

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

# fft of white noise error tone
samplerate_error, error_tone = wavfile.read('data/whitenoise.wav')
error_power, error_freqs = run_fft(error_tone, samplerate_error)

# plot power spectrum of error tone
plt.plot(error_freqs, error_power, color='blue')
plt.axvline(onset_freq,0,1,linestyle=':', color='deeppink') # onset
plt.annotate('onset', (onset_freq, np.max(error_power)), color='deeppink')
plt.axvline(timeout_freq,0,1,linestyle=':',color='red') # timeout
plt.annotate('timeout', (timeout_freq, np.max(error_power)*0.9), color='red')
plt.axvline(2000,0,1,linestyle=':',color='purple') # correct
plt.annotate('correct', (correct_freq, np.max(error_power)*0.95), color='purple')
plt.show()

# short time FFT
f, t, Sxx = signal.spectrogram(data_short, samplerate)
spectogram_samplerate = len(t)/len(data_short)*samplerate

# plot a segment of spectrogram
plt.pcolormesh(t[90000:91071], f[:30], Sxx[:30,90000:91071], shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# find audio_onsets of groups
onset_f_ind = np.argmin(np.abs(f-onset_freq))
other_fs = np.where(f!=f[onset_f_ind])
other_f_ind = np.argmin(np.abs(f-onset_freq-500))
audio_onsets = find_audio_onsets(Sxx, onset_f_ind, 10000, spectogram_samplerate/10) # FIXME: hardcoded threshold for finding audio_onsets 

audio_onsets_shifted = t[audio_onsets]-t[audio_onsets[0]]
# audio_onsets_seconds = audio_onsets_shifted/spectogram_samplerate

# plot only power of onset frequency
fig,ax = plt.subplots()
plt.plot(t, Sxx[onset_f_ind,:])#-Sxx[other_f_ind,:])
plt.vlines(t[audio_onsets],0,np.max(Sxx[onset_f_ind,:]) ,color='k', linestyles=':')
plt.show()

# check itis against trials table
grating_onsets = trials_df['sound_trial_start.started'].values
grating_onsets_shifted = grating_onsets-grating_onsets[0]
plt.scatter(audio_onsets_shifted, grating_onsets_shifted)
plt.show()

# exp_duration = trials_df['trial.stopped'].tail(1) - grating_onsets[0]
itis_trials = np.diff(grating_onsets)

audio_onsets_in_exp = audio_onsets[t[audio_onsets] > 200] # FIXME: hardcoded time where experiment starts
first_onset = audio_onsets[0]
time_from_start = t[audio_onsets_in_exp]-t[first_onset]
# audio_onsets_in_exp = audio_onsets_in_exp[np.where(time_from_start < exp_duration.values)]
spectogram_timepoints = np.arange(0,len(t)/spectogram_samplerate,step=1/spectogram_samplerate)
itis_audio = np.diff(spectogram_timepoints[audio_onsets_in_exp])

# plot histogram of itis
plt.hist(itis_trials, range=(0,7), bins=14, alpha=0.5, color='darkseagreen', label='trials')
plt.hist(itis_audio, range=(0,7), bins=14, alpha=0.4, color='steelblue', label='audio')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(grating_onsets, t[audio_onsets_in_exp])
ax.axline((0, 0), slope=1, linestyle=':', color='k')
plt.show()

### repeat for correct
# find audio_onsets of groups
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


