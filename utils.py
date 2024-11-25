import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 

import os
import shutil
from io import StringIO
import re
from pathlib import Path
import mne

from audio_extract import extract_audio
from scipy.io import wavfile
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy import signal
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# https://int-brain-lab.github.io/iblenv/_autosummary/brainbox.behavior.pyschofit.html, has moved to its own package instead of brainbox
try:
    import psychofit as psy 
except:
    import brainbox.behavior.psychofit as psy


def get_files_from_folder(folder_path, extension='.csv'):
    # Get all files in the folder
    files = os.listdir(folder_path)
    # Filter only CSV files
    csv_files = [file for file in files if file.endswith(extension)]
    # Create absolute paths for CSV files
    csv_paths = [os.path.join(folder_path, file) for file in csv_files]
    
    downloaded_files = []
    for csv_path in csv_paths:
        with open(csv_path, 'r') as file:
            file_content = file.read()
            downloaded_files.append((csv_path, file_content))
    return downloaded_files

#%% =============================== #
# plotting functions
# ================================= #

def plot_psychometric(df, color='black', **kwargs):
    
    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        ax = plt.gca()
    
    # from https://github.com/int-brain-lab/paper-behavior/blob/master/paper_behavior_functions.py#L391
    # summary stats - average psychfunc
    df2 = df.groupby(['signed_contrast']).agg(count=('choice', 'count'),
                                              mean=('choice', 'mean')).reset_index()    
    # fit psychfunc
    pars, L = psy.mle_fit_psycho(df2.transpose().values,  # extract the data from the df
                                 P_model='erf_psycho_2gammas',
                                 parstart=np.array(
                                     [0, 2., 0.05, 0.05]),
                                 parmin=np.array(
                                     [df2['signed_contrast'].min(), 0, 0., 0.]),
                                 parmax=np.array([df2['signed_contrast'].max(), 4., 1, 1]))

    # plot psychfunc
    xrange = np.max(np.abs(df['signed_contrast']))
    xlims = np.linspace(-xrange, xrange, num=100)
    sns.lineplot(x=xlims, y=psy.erf_psycho_2gammas(pars, xlims), 
                 color=color, zorder=10, **kwargs)
    
    # plot datapoints on top
    sns.lineplot(data=df, 
                  x='signed_contrast', y='choice', err_style="bars", 
                  linewidth=0, mew=0.5, zorder=20,
                  marker='o', errorbar=('ci',68), color=color, **kwargs)
    
    # paramters in title
    ax.set_title(r'$\mu=%.2f, \sigma=%.2f, \gamma=%.2f, \lambda=%.2f$'%tuple(pars),
              fontsize='x-small')

# ================================= #
def convert_psychopy_one(data, file_name):

    session_df = data.copy() # to extract session-level info
    
    # recode some things
    if 'eccentricity' in data.keys(): # do this if mouse responses
        # this code merges the columns that are now spread over trials_1, trials_2, etc
        response_mappings = pd.DataFrame({'eccentricity':[15,15,-15,-15], 'correct':[1,0,1,0], 'response':[1,0,0,1]})
        data = data.merge(response_mappings, on=['eccentricity', 'correct'], how='left') # right response = 1, left response = 0
    else: # do this if keyboard responses (older code from Annika)
        data['response'] = data['key_resp.keys'].map({'x': 1, 'm': 0}, na_action=None)

    # drop practice trials
    session_start_ind = np.max(np.argwhere(data['session_start'].notnull()))+1
    data = data.iloc[session_start_ind:]
    
    # add new column that counts all trials
    data['trials.allN'] = data.reset_index().index
    # print(f"number of trials: {data['trials.allN'].iloc[-1]}")

    # drop rows with no stimuli
    data = data[data['eccentricity'].notna()]

    # add new column that counts all trials
    data['trial'] = data.reset_index().index
    
    # rename some columns
    # use the same as IBL dataset types https://docs.google.com/document/d/1OqIqqakPakHXRAwceYLwFY9gOrm8_P62XIfCTnHwstg/edit#heading=h.nvzaz0fozs8h
    data.rename(columns={'reaction_time':'firstMovement_time', 
                            'contrastDelta':'stimContrast',
                            'leftCont':'contrastLeft',
                            'rightCont':'contrastRight',
                            'response': 'choice',
                            'correct': 'feedbackType',
                            'participant': 'subject'}, inplace=True)
    data['stimSide'] = (data['eccentricity'] > 1).astype(int) # check this is the correct side
    data['feedbackType'] = data['feedbackType'].replace(0, -1)

    # insert the biased block column from excel if not present
    if 'bias' in data.columns:
        data.rename(columns={'bias':'probabilityLeft'}, inplace=True)
    else:
        pregen = pd.read_excel('pregen_params_1.xlsx')
        try:
            assert (pregen.eccentricity.values == data.eccentricity.values).all() # confirm we are matching correctly
            data['probabilityLeft'] = pregen.bias   
        except: # if the trial numbers dont match
            data['probabilityLeft'] = 0.5 # pretend it was all one neutral session

    # drop some columns
    data.dropna(axis=1, how='all', inplace=True) # all columns with only nans
    data = data[data.columns.drop(list(data.filter(regex='og')))]
    data = data[data.columns.drop(list(data.filter(regex='this')))]

    # change unit of response times
    data[['response_time','firstMovement_time']] /= 1000
    # data['firstMovement_times_from_stim'] /= 1000

    # recode no response trials
    max_rt = data['response_time'].max()
    data['response_times_max'] = data['response_time'].fillna(value=max_rt)

    # set session number
    data['session'] = ''.join(re.findall(r'\d+', file_name[-15:]))
        
    ## now also extract session-level info
    cols2save = ['date', 'expName', 'participant', 
                 'expStart', 'session_end', 
                 'psychopyVersion', 'win_size', #'frameRate',
                 'block_breaks', 'instructions',
                 'age_slider.response'] \
                + [c for c in session_df.keys() if 'clicked_name' in c and \
                   not 'continue_txt' in session_df[c].dropna().unique()[0] and \
                  len(session_df[c].dropna().unique()[0]) > 0] \
                + [c for c in session_df.keys() if '.text' in c]

    col_dict = {}
    for c in cols2save:
        if c in session_df.keys(): # if this can't be found, list nan
            item = session_df[c].dropna().unique()
            try: item = item.item()
            except: continue

            if isinstance(item, list):
                item = item[0]
            col_dict.update({c: item})
        else:
           col_dict.update({c: np.nan})
    session_df = pd.DataFrame(col_dict.items(), columns=['name', 'value'])

    return data, session_df

# ================================= #
def plot_snapshot_behavior(data, folder_save, file_name_save):

        # ============================= %
    # from https://github.com/int-brain-lab/IBL-pipeline/blob/master/prelim_analyses/behavioral_snapshots/behavior_plots.py
    # https://github.com/int-brain-lab/IBL-pipeline/blob/7da7faf40796205f4d699b3b6d14d3bf08e81d4b/prelim_analyses/behavioral_snapshots/behavioral_snapshot.py
    
    plt.close('all')

    fig, ax = plt.subplots(ncols=3, nrows=2, width_ratios=[1,1,2], figsize=(12,8))
    
    # 1. psychometric
    plot_psychometric(data, ax=ax[0,0])
    ax[0,0].set(xlabel='Signed contrast', ylabel='Choice (fraction)',
                ylim=[-0.05, 1.05])
    
    # 2. chronometric
    sns.lineplot(data=data, ax=ax[0,1],
                    x='signed_contrast', y='response_time', err_style="bars", 
                    linewidth=1, estimator=np.median, 
                    mew=0.5,
                    marker='o', errorbar=('ci',68), color='black')
    sns.lineplot(data=data, ax=ax[0,1],
                    x='signed_contrast', y='firstMovement_time', err_style="bars", 
                    linewidth=1, estimator=np.median, 
                    mew=0.5, alpha=0.5,
                    marker='o', errorbar=('ci',68), color='black')
    ax[0,1].set(xlabel='Signed contrast', ylabel='Response time (black) / movement initiation (grey) (s)', 
                ylim=[0, 1.6])

    # 3. time on task
    sns.scatterplot(data=data, ax=ax[0,2], 
                    x='trials.allN', y='response_time', 
                    style='feedbackType', hue='feedbackType',
                    palette={1.:"#009E73", -1.:"#D55E00"}, 
                    markers={1.:'o', -1.:'X'}, s=10, edgecolors='face',
                    alpha=.5, legend=False)
    
    # running median overlaid
    sns.lineplot(data=data[['trials.allN', 'response_time']].rolling(10).median(), 
                    ax=ax[0,2],
                    x='trials.allN', y='response_time', color='black', errorbar=None)
    
    sns.lineplot(data=data[['trials.allN', 'firstMovement_time']].rolling(10).median(), 
                    ax=ax[0,2],
                    x='trials.allN', y='firstMovement_time', color='grey', errorbar=None)
    
    # add lines to mark breaks, if these were included
    if 'block_breaks' in data.columns:
        if data['block_breaks'].iloc[0] == 'y':
            [ax[0,2].axvline(x, color='blue', alpha=0.2, linestyle='--') 
             for x in np.arange(100,data['trials.allN'].iloc[-1],step=100)]
    
    ax[0,2].set(xlabel="Trial number", ylabel="RT / MiT (s)", ylim=[0.01, 10])
    ax[0,2].set_yscale("log")
    ax[0,2].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos:
        ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))

    # ============================= %
    # BIASED BLOCKS
    # ============================= %

    cols = {0.2: 'orange', 0.5:'grey', 0.8:'purple'}

    for gr, dat in data.groupby(['probabilityLeft']):
        color = cols[gr[0]]

        # 1. psychometric
        plot_psychometric(dat, color=color, ax=ax[1,0])
        ax[1,0].set(xlabel='Signed contrast', ylabel='Choice (fraction)',
                ylim=[-0.05, 1.05])

        # 2. chronometric
        sns.lineplot(data=dat, ax=ax[1,1],
                    x='signed_contrast', y='response_time', err_style="bars", 
                    linewidth=1, estimator=np.median, 
                    mew=0.5,
                    marker='o', errorbar=('ci',68), color=color)
        sns.lineplot(data=dat, ax=ax[1,1],
                    x='signed_contrast', y='firstMovement_time', err_style="bars", 
                    linewidth=1, estimator=np.median, 
                    mew=0.5, alpha=0.5,
                    marker='o', errorbar=('ci',68), color=color)
        ax[1,1].set(xlabel='Signed contrast', ylabel='Response time (black) / movement initiation (grey) (s)')
                # ylim=[0, 1.6])

    # 3. biased blocks
    sns.scatterplot(data=data, ax=ax[1,2], 
                    x='trials.allN', y='probabilityLeft', 
                    style='probabilityLeft', hue='probabilityLeft',
                    palette=cols, 
                    marker='o', s=10, edgecolors='face',
                    alpha=.5, legend=False)
    
    # running median overlaid
    sns.lineplot(data=data[['trials.allN', 'stimSide']].rolling(10).mean(), 
                    ax=ax[1,2],
                    x='trials.allN', y='stimSide', color='grey', errorbar=None)
    
    sns.lineplot(data=data[['trials.allN', 'choice']].rolling(10).mean(), 
                    ax=ax[1,2],
                    x='trials.allN', y='choice', color='black', errorbar=None)
    
    # add lines to mark breaks
    if 'block_breaks' in data.columns:
        if data['block_breaks'].iloc[0] == 'y':        
            [plt.axvline(x, color='blue', alpha=0.2, linestyle='--') 
             for x in np.arange(100,data['trials.allN'].iloc[-1],step=100)]
    ax[1,2].set(xlabel="Trial number", ylabel= "Stim (grey) / choice (black)")

    # ============================= %

    fig.suptitle(file_name_save)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    sns.despine(trim=True)
    fig.savefig(os.path.join(folder_save, file_name_save))

def plot_snapshot_pupil(file_name, folder_save, fig_name):
    # %%
    # load pupil data:
    raw_et = mne.io.read_raw_eyelink(file_name)
    raw_et_df = raw_et.to_data_frame()
    interp_et = mne.preprocessing.eyetracking.interpolate_blinks(raw_et, buffer=(0.05, 0.1), 
                                                                    interpolate_gaze=True)
    interp_et_df = interp_et.to_data_frame()

    # find the channels, these may be left or right eye
    pupil_chan = [c for c in interp_et_df.columns if 'pupil' in c][0]
    x_chan = [c for c in interp_et_df.columns if 'x' in c][0]
    y_chan = [c for c in interp_et_df.columns if 'y' in c][0]

    # TODO: add temporal filters, see code JW
    interp_et = interp_et.filter(0.01, 10, 
                                    picks=pupil_chan,
                                    method='iir',
                                    fir_design='firwin', 
                                    skip_by_annotation='edge')
    
    # get raw annotations
    annot = raw_et.annotations.to_data_frame()
    annot['onset'] = raw_et.annotations.onset

    # lock to stimulus onset, response and feedback
    # get events - note that mne epoching squashes everything below 0 to 0, so use integers here for hte contrast levels
    stim_events_dict = {"signed_contrast -0.02": -2, "signed_contrast -0.05": -5, "signed_contrast -0.10": -10, 
                        "signed_contrast -0.20": -20, "signed_contrast 0.00": 0, "signed_contrast 0.02": 2, "signed_contrast 0.05": 5,
                        "signed_contrast 0.10": 10, "signed_contrast 0.20": 20}
    events, _ = mne.events_from_annotations(raw_et, event_id=stim_events_dict)
    epochs = mne.Epochs(interp_et, events,  tmin=-1.5, tmax=5, 
                        baseline=(-1, 0), preload=True, reject=None)   
    stim_epochs_df = epochs.to_data_frame()
    stim_epochs_df['abs_contrast'] = np.abs(stim_epochs_df['condition'].astype(float))
    stim_epochs_df['stim_side'] = np.sign(stim_epochs_df['condition'].astype(float))

    response_events_dict = {"response 1":1, "response -1":-1}
    events, _ = mne.events_from_annotations(raw_et, event_id=response_events_dict)
    epochs = mne.Epochs(interp_et, events,  tmin=-1, tmax=3, baseline=None, preload=True, reject=None)   
    resp_epochs_df = epochs.to_data_frame()

    feedback_events_dict = {"feedbackType 1":1, "feedbackType 0":0}
    events, _ = mne.events_from_annotations(raw_et, event_id=feedback_events_dict)
    epochs = mne.Epochs(interp_et, events,  tmin=-1.5, tmax=5, baseline=None, preload=True, reject=None)   
    fb_epochs_df = epochs.to_data_frame()

    # ================================= #
    # make a snapshot figure
    # ================================= #

    sns.set_style('darkgrid')
    fig, axes = plt.subplot_mosaic([['a', 'a'], ['b', 'c'], ['d', 'e'], ['f', 'g']],
                                layout='constrained', figsize=(12,8))

    #sns.lineplot(, x='time', y='pupil_left')
    sns.lineplot(raw_et_df, x='time', y=pupil_chan, color='darkgrey', ax=axes['a'])
    sns.lineplot(interp_et_df, x='time', y=pupil_chan, color='deepskyblue', ax=axes['a'])
    axes['a'].set_xlabel('Time in session (seconds)')

    # stimulus-locked pupil
    for axn, var in zip(['b', 'd', 'f'], [pupil_chan, x_chan, y_chan]):
        # sns.lineplot(data=stim_epochs_df, x='time', y=var, estimator=None, 
        #             hue='condition', palette='rocket', alpha=0.1, linewidth=0.5,
        #             legend=False,
        #             units='epoch', ax=axes[axn])
        sns.lineplot(data=stim_epochs_df, x='time', y=var, estimator='mean',
                    err_style='band', errorbar=("pi", 50),
                    # style='stim_side',
                    hue='abs_contrast', palette='viridis', alpha=1, linewidth=2,
                    legend=False, ax=axes[axn])

    # feedback-locked pupil
    for axn, var in zip(['c', 'e', 'g'], [pupil_chan, x_chan, y_chan]):
        # sns.lineplot(data=fb_epochs_df, x='time', y=var, estimator=None, 
        #             hue='condition', legend=False, alpha=0.5,
        #             palette=['green', 'red'], hue_order=['1','0'],
        #             units='epoch', ax=axes[axn])
        sns.lineplot(data=fb_epochs_df, x='time', y=var, 
                        estimator='mean',                         
                        err_style='band', errorbar=("pi", 50),
                    hue='condition', legend=False, alpha=1, linewidth=2,
                    palette=['green', 'red'], hue_order=['1','0'],
                    ax=axes[axn])
    # link some axes
    axes['b'].sharex(axes['d'])
    axes['d'].sharex(axes['f'])
    axes['f'].set_xlabel('Time from stimulus onset (s)')

    axes['c'].sharex(axes['e'])
    axes['e'].sharex(axes['g'])
    axes['g'].set_xlabel('Time from feedback (s)')

    # figname = os.path.split(file_name)[1].split('.asc')[0]
    # fig.suptitle(figname)
    # fig.savefig(str(figpath) + 'pupil_' + figname + '.png')

    fig.suptitle(file_name)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    sns.despine(trim=True)
    fig.savefig(os.path.join(folder_save, fig_name))


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
    spectogram_samplerate = len(t)/len(data)*samplerate

    # find audio_onsets of groups
    onset_f_ind = np.argmin(np.abs(f-target_freq))
    onsets = find_audio_onsets(Sxx, onset_f_ind, power_thresh, spectogram_samplerate*min_seconds_gap) 

    return np.array(onsets), t

def make_mne_events_array(event_times_samples:np.ndarray, event_code):
    return np.vstack((np.rint(event_times_samples), np.zeros_like(event_times_samples, dtype=int), np.full_like(event_times_samples, event_code))).T

def trim_video(processed_path, vid_path, cutoff_start_s, cutoff_end_s):
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    short_vid_path = processed_path + 'short_' + os.path.split(vid_path)[-1]
    if not os.path.exists(short_vid_path):
        ffmpeg_extract_subclip(vid_path, cutoff_start_s, cutoff_end_s, targetname=short_vid_path)
    return short_vid_path

def plot_snapshot_audio(data_path, folder_save, fig_name, onset_freq = 5000):

    # load trials
    behavior_file_name = [s for s in os.listdir(data_path) if s.endswith('.csv')]
    trials_data = pd.read_csv(os.path.join(data_path, behavior_file_name[0]))

    # convert to ONE convention
    trials_df = convert_psychopy_one(trials_data, behavior_file_name[0])
    grating_onsets = trials_df['sound_trial_start.started'].values
    grating_onsets_shifted = grating_onsets-grating_onsets[0]

    # find time that video starts
    vid_filename = [x for x in os.listdir(data_path) if '.mkv' in x]
    vid_path = os.path.join(data_path, vid_filename[0])
    vid_start_str = str(vid_path)[-12:-4]
    vid_start_time = datetime.strptime(vid_start_str, '%H-%M-%S')

    # find time that experiment starts
    exp_start_str = trials_data[trials_data['session_start'].notna()]['session_start'].unique()[0][-12:-1]
    exp_start_time = datetime.strptime(exp_start_str, '%Hh%M.%S.%f') - timedelta(seconds=5)
    if 'session_end' in trials_df.keys():
        exp_end_str = trials_data[trials_data['session_end'].notna()]['session_end'].unique()[0][-12:]
        exp_end_time = datetime.strptime(exp_end_str, '%Hh%M.%S.%f') + timedelta(seconds=5)
    else:
        exp_end_time = exp_start_time + timedelta(seconds=(trials_df['feedback_sound.stopped'].tail(1).values-trials_df['sound_trial_start.started'].head(1).values)[0]+10)
    exp_dur = (exp_end_time-exp_start_time).total_seconds()
    cutoff_start_s = (exp_start_time-vid_start_time).total_seconds()
    cutoff_end_s = cutoff_start_s + exp_dur
    
    processed_path = data_path + '\processed\\'
    short_vid_path = trim_video(processed_path, vid_path, cutoff_start_s, cutoff_end_s)

    # extract audio if not done
    audio_path = f'{str(short_vid_path)[:-3]}wav'
    if audio_path.split('\\')[-1] not in os.listdir(processed_path):
        extract_audio(input_path=short_vid_path, output_path=audio_path, output_format='wav', overwrite=False)

    # load audio
    samplerate, data = wavfile.read(audio_path)
    data_short = data.astype('float')[:0]

    audio_onsets_selected = []
    threshold = 3000
    min_gap = np.min(trials_df['response_time'])
    while (len(audio_onsets_selected) < len(grating_onsets)) & (threshold > 500):
        threshold -= 500
        audio_onsets, t = detect_freq_onsets(data_short, samplerate, onset_freq, 0.1, threshold)
        _, onset_indices = find_best_shift(t[audio_onsets], grating_onsets_shifted, (0,20), min_gap)
        audio_onsets_selected = audio_onsets[onset_indices>-1]

    audio_onsets_relative_to_start = t[audio_onsets_selected]
    np.save(processed_path + 'av_onsets', audio_onsets_relative_to_start)

    audio_onsets_shifted = t[audio_onsets_selected]-t[audio_onsets_selected[0]]

    # check equal number of onsets and onsets within 100ms
    assert len(grating_onsets) == len(audio_onsets_selected), f'audio {len(audio_onsets_selected)} onsets, psychopy {len(grating_onsets)} onsets'
    print(f'success: {len(grating_onsets)} onsets detected')
    if not np.isclose(audio_onsets_shifted, grating_onsets_shifted, atol=0.15).all():
        print('detected onsets not within 150ms of psychopy onsets')
    else:
        max_offset = np.round(np.max(np.abs(audio_onsets_shifted-grating_onsets_shifted))*1000, decimals=1)
        print(f'maximum offset between psychopy and audio is {max_offset} ms')

    timepoints = np.arange(0, len(data_short)/samplerate, 1/samplerate)
    seconds = pd.to_datetime(timepoints, unit='s')

    len_segment_s = 60
    segment_start = 10*60*samplerate
    len_segment_samples = samplerate*len_segment_s
    segment = data_short[segment_start:segment_start+len_segment_samples]
    segment_seconds = seconds[segment_start:segment_start+len_segment_samples]

    correct_feedback_times = t[audio_onsets_selected[trials_df['feedbackType']==1]] + trials_df[trials_df['feedbackType']==1]['response_time']
    error_feedback_times = t[audio_onsets_selected[trials_df['feedbackType']==-1]] + trials_df[trials_df['feedbackType']==-1]['response_time']

    # convert to mne for easy epoching
    mne_info = mne.create_info(ch_names=['audio', 'stim'], sfreq=samplerate, ch_types=['misc', 'stim'])
    raw = mne.io.RawArray(np.vstack((data_short,np.zeros_like(data_short))), mne_info)

    onset_events = make_mne_events_array(np.rint(t[audio_onsets_selected]*samplerate), 1)
    correct_events = make_mne_events_array(np.rint(t[audio_onsets_selected[trials_df['feedbackType']==1]]*samplerate+trials_df[trials_df['feedbackType']==1]['response_time']*samplerate), 2)
    error_events = make_mne_events_array(np.rint(t[audio_onsets_selected[trials_df['feedbackType']==-1]]*samplerate+trials_df[trials_df['feedbackType']==-1]['response_time']*samplerate), 3)
    
    all_events = np.vstack((onset_events, correct_events, error_events))
    raw.add_events(all_events, stim_channel='stim')
    events = mne.find_events(raw, stim_channel='stim')  
    fb_event_dict = {'correct':2, 'error':3}   
    onset_event_dict = {'onset':1}  
    onset_epochs = mne.Epochs(raw, events=events, event_id=onset_event_dict, tmin=-.1, tmax=.5, baseline=None, reject=None)
    fb_epochs = mne.Epochs(raw, events=events, event_id=fb_event_dict, tmin=-.1, tmax=.5, baseline=None, reject=None)
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
    ax['A'].vlines(pd.to_datetime(t[audio_onsets_selected], unit='s'), ymin=np.min(segment), ymax=np.max(segment), color='k', linestyle=':', label='onset')
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
    fig.savefig(os.path.join(folder_save, fig_name))


def find_best_shift(audio_onsets, grating_onsets_shifted, shift_start_end=(0,20), delta=0.2):
    shifts = range(shift_start_end[0], shift_start_end[1])
    len_shifts = len(shifts)
    len_audio_onsets = len(audio_onsets)
    shift_errors = np.empty(len_shifts, dtype=float)
    shift_matches = np.empty((len_shifts, len_audio_onsets), dtype=np.int64)

    for j,s in enumerate(shifts):
        shifted = audio_onsets-audio_onsets[s]
        abs_errors = np.empty(len_audio_onsets)

        for i,n in enumerate(shifted):
            errors = np.minimum(np.square(grating_onsets_shifted-n), np.square(delta))
            is_outlier = np.all(np.isclose(errors, np.square(delta)))
            shift_matches[j,i] = np.argmin(errors) if not is_outlier else -1
            abs_errors[i] = errors[shift_matches[j,i]] if not is_outlier else np.square(delta)

        shift_errors[j] = np.sum(abs_errors)

    return shifts[np.argmin(shift_errors)], shift_matches[np.argmin(shift_errors),:]

import imageio.v3 as iio
from tqdm import tqdm
import cv2
import argparse
from moviepy.video.io import VideoFileClip
from moviepy.editor import VideoFileClip    
from moviepy.video.fx import crop
from moviepy.video.fx.all import blackwhite
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

def plot_contour(image, contours, idx):
    approx = cv2.approxPolyDP(contours[idx], 0.02 * cv2.arcLength(contours[idx], True), True)  
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [approx], -1, 255, -1)
    plt.imshow(mask)

def plot_frame(clip, idx):
    plt.imshow(clip.get_frame(idx*1.0/clip.fps)[:,:,0], cmap='gray')

def find_mirror(frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morphed = cv2.morphologyEx(frame,cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(morphed,100,200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        approx = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)  
        # Check bounding box and aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.8 < aspect_ratio < 1.2:  # Adjust as needed
                mask = np.zeros_like(frame)
                cv2.drawContours(mask, [approx], -1, 255, -1)
                mean_intensity = cv2.mean(frame, mask=mask)[0]
                _, max_intensity,_,_ = cv2.minMaxLoc(frame, mask=mask)
                if (max_intensity > 220) & (mean_intensity > 150):
                    print(i, max_intensity, mean_intensity)
                    mirror_loc = [x,y,w,h]
    return mirror_loc

def compute_contrast(frame):
    f = frame[:, :, 0]
    min_val = np.min(f)
    max_val = np.max(f)
    return (max_val - min_val) / (max_val + min_val)

def compute_max(frame):
    f = frame[:, :, 0]
    return np.max(f)

def process_frames(clip):
    """
    Function to process the frames in parallel using multiprocessing.
    """
    with Pool() as pool:
        # Use Pool's map function to parallelize the contrast calculation for each frame
        contrast_list = pool.map(compute_contrast, clip.iter_frames())
    
    return np.array(contrast_list)

subj = '008'
data_path = os.path.join(folder_path, subj)

def get_contrast_from_video(data_path):
    vid_path = data_path + '\processed\\'
    vid_filename = [x for x in os.listdir(vid_path) if '.mkv' in x][0]
    onsets_filename = [x for x in os.listdir(vid_path) if '.npy' in x][0]
    audio_onsets = np.load(vid_path + onsets_filename)

    clip = VideoFileClip.VideoFileClip(vid_path+vid_filename)
    clip_bw = blackwhite(clip=clip, preserve_luminosity=True)
    frame = clip_bw.get_frame(500*1.0/clip.fps)[:,:,0] 
    x,y,w,h = find_mirror(frame)
    plt.imshow(frame[y:y+w, x:x+h])
    mirror_crop = crop.crop(clip=clip_bw, x1=x, y1=y, width=w, height=h)

    # get some info from video
    vid_samplerate = clip.fps
    total_frames = clip.reader.nframes

    plt.imshow(mirror_crop.get_frame(500*1.0/clip.fps)[:,:,0], cmap='gray') # plot 500th frame

    # Use ProcessPoolExecutor to parallelize the task
    contrast_list = np.empty((total_frames, 1))
    with ProcessPoolExecutor() as executor:
        # Map each frame to the compute_contrast function
        results = list(tqdm(executor.map(compute_contrast, mirror_crop.iter_frames()), total=total_frames))
        # TODO: find progress bar that works
    # Store the results into the contrast_list
    for idx, result in enumerate(results):
        contrast_list[idx] = result

    contrast_list = np.empty((total_frames, 1))
    f = np.empty((w,h))
    for idx, f in enumerate(mirror_crop.iter_frames()):
        # compute_contrast(f)
        f_bw = f[:,:,0]
        # compute min and max of Y
        min = np.min(f_bw)
        max = np.max(f_bw)
        # compute contrast
        contrast_list[idx] = (max-min)/(max+min)
        if idx>500:
            break

    for idx in range(total_frames): # is this faster?
        f_bw = mirror_crop.get_frame(idx*1.0/clip.fps)[:,:,0]
        # compute min and max of Y
        min = np.min(f_bw)
        max = np.max(f_bw)
        # compute contrast
        contrast_list[idx] = (max-min)/(max+min)
        if idx>500:
            break

    # find contrast for frames where onsets should be happening
    epoch_start = -0.15
    epoch_end = 0.1
    contrast_epochs = np.empty((len(audio_onsets), len(np.arange(epoch_start, epoch_end, step=1/clip.fps))+1))
    for i,o in enumerate(audio_onsets[:10]):
        print(i)
        for j,t in enumerate(np.arange(o+epoch_start, o+epoch_end, step=1/clip.fps)):
            f = mirror_crop.get_frame(t)
            contrast_epochs[i,j] = compute_max(f)
    
    plt.plot(np.arange(epoch_start, epoch_end, step=1/clip.fps), contrast_epochs[:10,:24].T-contrast_epochs[:10,0].T)

    # do the same thing but use the grating onsets - this is better
    # load trials
    behavior_file_name = [s for s in os.listdir(data_path) if s.endswith('.csv')]
    trials_data = pd.read_csv(os.path.join(data_path, behavior_file_name[0]))

    # convert to ONE convention
    trials_df = convert_psychopy_one(trials_data, behavior_file_name[0])
    grating_onsets = trials_df['grating_l.started'].values # sound_trial_start.started
    grating_onsets_shifted = grating_onsets-grating_onsets[0]
    onsets_shifted = grating_onsets_shifted + audio_onsets[0]

    contrast_epochs = np.empty((len(audio_onsets), len(np.arange(epoch_start, epoch_end, step=1/clip.fps))+1))
    for i,o in enumerate(onsets_shifted[:10]):
        print(i)
        for j,t in enumerate(np.arange(o+epoch_start, o+epoch_end, step=1/clip.fps)):
            f = mirror_crop.get_frame(t)
            contrast_epochs[i,j] = compute_max(f)
    
    plt.plot(np.arange(epoch_start, epoch_end, step=1/clip.fps), contrast_epochs[:10,:24].T-contrast_epochs[:10,0].T)



    if False: # some plotting stuff
        edges = cv2.Canny(frame,100,200)
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame = clip_bw.get_frame(500*1.0/clip.fps)[:,:,0] # make sure this is bw




        plt.subplot(121),plt.imshow(frame,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

        plt.show()