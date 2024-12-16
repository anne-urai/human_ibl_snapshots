import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 

import os
import shutil
from io import StringIO
import re
import mne
import cv2

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

def load_trials(folder_path, subj, data_type='df'):
    if data_type == 'df':
        return pd.read_csv(os.path.join(folder_path, subj, 'alf', 'trials_table.csv'))
    elif data_type == 'raw':
        raw_path = os.path.join(folder_path, subj, 'raw_behavior_data')
        raw_file_name = [s for s in os.listdir(raw_path) if s.endswith('.csv')][0]
        return pd.read_csv(os.path.join(raw_path, raw_file_name))
    else:
        raise ValueError('data_type should be "df" or "raw"')


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
    short_vid_path = os.path.join(processed_path, 'short_' + os.path.split(vid_path)[-1])
    if not os.path.exists(short_vid_path):
        ffmpeg_extract_subclip(vid_path, cutoff_start_s, cutoff_end_s, targetname=short_vid_path)
    return short_vid_path

def plot_snapshot_audio(folder_path, subj, folder_save, fig_name):
    
    alf_path = os.path.join(folder_path, subj, 'alf')
    audio_filename = [x for x in os.listdir(alf_path) if '.wav' in x][0]

    # load audio, onsets, timepoints
    samplerate, data = wavfile.read(os.path.join(alf_path, audio_filename))
    data_short = data.astype('float')[:,0]
    audio_onsets = np.load(os.path.join(alf_path, 'audio_onsets.npy'))
    audio_onsets_shifted = audio_onsets - audio_onsets[0]
    seconds = np.load(os.path.join(alf_path, 'audio_times.npy'))

    # load trials, get grating onsets
    trials_df = load_trials(folder_path, subj)
    grating_onsets = trials_df['sound_trial_start.started'].values
    grating_onsets_shifted = grating_onsets-grating_onsets[0]

    # snippet to plot
    len_segment_s = 60
    segment_start = 10*60*samplerate
    len_segment_samples = samplerate*len_segment_s
    segment = data_short[segment_start:segment_start+len_segment_samples]
    segment_seconds = seconds[segment_start:segment_start+len_segment_samples]

    # get feedback times from trials
    correct_feedback_times = audio_onsets[trials_df['feedbackType']==1] + trials_df[trials_df['feedbackType']==1]['response_time']
    error_feedback_times = audio_onsets[trials_df['feedbackType']==-1] + trials_df[trials_df['feedbackType']==-1]['response_time']

    # convert to mne for easy epoching
    mne_info = mne.create_info(ch_names=['audio', 'stim'], sfreq=samplerate, ch_types=['misc', 'stim'])
    raw = mne.io.RawArray(np.vstack((data_short,np.zeros_like(data_short))), mne_info)

    onset_events = make_mne_events_array(np.rint(audio_onsets*samplerate), 1)
    correct_events = make_mne_events_array(np.rint(audio_onsets[trials_df['feedbackType']==1]*samplerate+trials_df[trials_df['feedbackType']==1]['response_time']*samplerate), 2)
    error_events = make_mne_events_array(np.rint(audio_onsets[trials_df['feedbackType']==-1]*samplerate+trials_df[trials_df['feedbackType']==-1]['response_time']*samplerate), 3)
    
    all_events = np.vstack((onset_events, correct_events, error_events))
    raw.add_events(all_events, stim_channel='stim')
    events = mne.find_events(raw, stim_channel='stim')  
    fb_event_dict = {'correct':2, 'error':3}   
    onset_event_dict = {'onset':1}  
    onset_epochs = mne.Epochs(raw, events=events, event_id=onset_event_dict, tmin=-.1, tmax=.5, baseline=None, reject=None)
    fb_epochs = mne.Epochs(raw, events=events, event_id=fb_event_dict, tmin=-.1, tmax=.5, baseline=None, reject=None)
    onset_epochs_df = onset_epochs.to_data_frame()
    fb_epochs_df = fb_epochs.to_data_frame()

    # ================================= #
    # make a snapshot figure
    # ================================= #

    sns.set_style('darkgrid')
    fig, ax = plt.subplot_mosaic([['A','A'],['B', 'C'],['B','D'],['B','E']], height_ratios=[0.4, 0.2, 0.2, 0.2], layout='constrained', figsize=(12,8))

    offsets = audio_onsets_shifted - grating_onsets_shifted
    ax['B'].scatter(range(len(offsets)), offsets*1000, c='purple') 
    ax['B'].set_xlabel('trial')
    ax['B'].set_ylabel('offset (audio - psychopy) [ms]')   
    
    ax['A'].plot(segment_seconds, segment)
    ax['A'].set_title('audio snippet')
    ax['A'].set_xlabel('time from session start (s)')
    ax['A'].set_ylabel('audio amplitude')
    ax['A'].vlines(audio_onsets, ymin=np.min(segment), ymax=np.max(segment), color='k', linestyle=':', label='onset')
    ax['A'].vlines(correct_feedback_times, ymin=np.min(segment), ymax=np.max(segment), color='g', linestyle=':', label='correct')
    ax['A'].vlines(error_feedback_times, ymin=np.min(segment), ymax=np.max(segment), color='r', linestyle=':', label='error')
    ax['A'].set_xlim(segment_seconds[0], segment_seconds[-1])
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

def process_audio(folder_path, subj, onset_freq = 5000):

    raw_video_path = os.path.join(folder_path, subj, 'raw_video_data')
    alf_path = os.path.join(folder_path, subj, 'alf')

    # load trials
    trials_data = load_trials(folder_path, subj, 'raw')
    trials_df = load_trials(folder_path, subj)

    # get grating onsets
    grating_onsets = trials_df['sound_trial_start.started'].values
    grating_onsets_shifted = grating_onsets-grating_onsets[0]

    # find time that video starts
    vid_filename = [x for x in os.listdir(raw_video_path) if '.mkv' in x]
    vid_path = os.path.join(raw_video_path, vid_filename[0])
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
    np.save(os.path.join(alf_path, 'cutoff_start_end'), np.array([cutoff_start_s, cutoff_end_s]))
    
    short_vid_path = trim_video(alf_path, vid_path, cutoff_start_s, cutoff_end_s)

    # extract audio if not done
    audio_path = f'{str(short_vid_path)[:-3]}wav'
    if audio_path.split('\\')[-1] not in os.listdir(alf_path):
        extract_audio(input_path=short_vid_path, output_path=audio_path, output_format='wav', overwrite=False)

    # load audio
    samplerate, data = wavfile.read(audio_path)
    data_short = data.astype('float')[:,0]

    audio_onsets_selected = []
    threshold = 3000
    min_gap = np.min(trials_df['response_time'])
    while (len(audio_onsets_selected) < len(grating_onsets)) & (threshold > 500):
        threshold -= 500
        audio_onsets, t = detect_freq_onsets(data_short, samplerate, onset_freq, 0.1, threshold)
        _, onset_indices = find_best_shift(t[audio_onsets], grating_onsets_shifted, (0,20), min_gap)
        audio_onsets_selected = audio_onsets[onset_indices>-1]

    audio_onsets_relative_to_start = t[audio_onsets_selected]
    np.save(os.path.join(alf_path, 'audio_onsets'), audio_onsets_relative_to_start)

    audio_onsets_shifted = t[audio_onsets_selected]-t[audio_onsets_selected[0]]

    # check equal number of onsets and onsets within 150ms
    assert len(grating_onsets) == len(audio_onsets_selected), f'audio {len(audio_onsets_selected)} onsets, psychopy {len(grating_onsets)} onsets'
    print(f'success: {len(grating_onsets)} audio onsets detected')
    if not np.isclose(audio_onsets_shifted, grating_onsets_shifted, atol=0.15).all():
        print('detected onsets not within 150ms of psychopy onsets')
    else:
        max_offset = np.round(np.max(np.abs(audio_onsets_shifted-grating_onsets_shifted))*1000, decimals=1)
        print(f'maximum offset between psychopy and audio is {max_offset} ms')

    timepoints = np.arange(0, len(data_short)/samplerate, 1/samplerate)
    # seconds = pd.to_datetime(timepoints, unit='s')
    # seconds_array = np.array(pd.to_timedelta(seconds - seconds[0], unit='s').total_seconds())
    np.save(os.path.join(alf_path, 'audio_times'), timepoints) 


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

def plot_contour(image, contours, idx):
    approx = cv2.approxPolyDP(contours[idx], 0.02 * cv2.arcLength(contours[idx], True), True)  
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [approx], -1, 255, -1)
    plt.imshow(mask)

def find_mirror(frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morphed = cv2.morphologyEx(frame,cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(morphed,100,200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mirror_loc = []
    for i, contour in enumerate(contours):
        approx = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)  
        # Check bounding box and aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.8 < aspect_ratio < 1.2:  # Adjust as needed
                # print(i, aspect_ratio)
                mask = np.zeros_like(frame)
                cv2.drawContours(mask, [approx], -1, 255, -1)
                mean_intensity = cv2.mean(frame, mask=mask)[0]
                _, max_intensity,_,_ = cv2.minMaxLoc(frame, mask=mask)
                if (max_intensity > 220) & (mean_intensity > 150):
                    # print(i, max_intensity, mean_intensity)
                    mirror_loc = [x,y,w,h]
    return mirror_loc

def compute_contrast(frame):
    min_val = np.min(frame)
    max_val = np.max(frame)
    return (max_val - min_val) / (max_val + min_val)

def compute_max(frame):
    f = frame[:, :, 0]
    return np.max(f)

def plot_snapshot_video(folder_path, subj, folder_save, fig_name):

    alf_path = os.path.join(folder_path, subj, 'alf')

    # load stim brightness, times
    stim_brightness = np.load(os.path.join(alf_path, 'stim_brightness.npy'))
    video_times = np.load(os.path.join(alf_path, 'video_times.npy'))
    video_onsets = np.load(os.path.join(alf_path, 'video_onsets.npy'))

    onsets_not_nan = ~np.isnan(video_onsets)
    if np.any(~onsets_not_nan):
        video_onsets[~onsets_not_nan] = np.nan

    # load trials, get psychopy onsets
    trials_df = load_trials(folder_path, subj)
    audio_onsets_filename = [x for x in os.listdir(alf_path) if 'audio_onsets' in x][0]
    audio_onsets = np.load(os.path.join(alf_path, audio_onsets_filename))
    grating_onsets = trials_df['grating_l.started'].values
    grating_onsets_shifted = grating_onsets - grating_onsets[0] + audio_onsets[0]

    # open video
    vid_filename = [x for x in os.listdir(alf_path) if '.mkv' in x][0]
    cap = cv2.VideoCapture(os.path.join(alf_path,vid_filename))
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # get some info from video
    vid_samplerate = cap.get(cv2.CAP_PROP_FPS) 
    vid_onsets_samples = (video_onsets * vid_samplerate).astype(int)

    # run through video to make avg mirror plot - 30 seconds
    x,y,w,h = np.load(os.path.join(alf_path, 'mirror_coords.npy'))
    video_array = np.empty((2000, h, w))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print('getting avg frame for plot...')    
    for i in range(2000):
        ret, frame = cap.read()
        if not ret:
            break
        video_array[i] = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

    cap.release()

    # convert to mne for easy epoching
    mne_info = mne.create_info(ch_names=['video', 'stim'], sfreq=vid_samplerate, ch_types=['eyegaze', 'stim'])
    raw = mne.io.RawArray(np.hstack((stim_brightness,np.zeros_like(stim_brightness))).T, mne_info)
    onset_events = make_mne_events_array(vid_onsets_samples[onsets_not_nan], 1)
    raw.add_events(onset_events, stim_channel='stim')
    events = mne.find_events(raw, stim_channel='stim')  
    onset_event_dict = {'onset':1}  
    onset_epochs = mne.Epochs(raw, events=events, event_id=onset_event_dict, tmin=-.15, tmax=.15, baseline=(-.15,-.10), reject=None) 
    onset_epochs_df = onset_epochs.to_data_frame()

    # ================================= #
    # make a snapshot figure
    # ================================= #

    sns.set_style('darkgrid')
    fig, ax = plt.subplot_mosaic([['A','A', 'A'],['B','F', 'D'], ['B','F', 'E']], height_ratios=[0.3, 0.2, 0.2], layout='constrained', figsize=(12,8))

    # make avg image of first 5000 frames of mirror
    ax['B'].imshow(np.mean(video_array, axis=0), cmap='gray')
    ax['B'].grid(None)
    ax['B'].set_title('selected mirror crop')
    ax['B'].axis('off')

    t_start = 10 # mins
    t_dur = 1 # min
    sample_start = int(t_start * 60 * vid_samplerate)
    sample_dur = int(t_dur * 60 * vid_samplerate)
    segment = stim_brightness[sample_start:sample_start+sample_dur]
    segment_seconds = video_times[sample_start:sample_start+sample_dur]

    ax['A'].plot(segment_seconds, segment)
    ax['A'].set_title('max brightness snippet')
    ax['A'].set_xlabel('time from session start (s)')
    ax['A'].set_ylabel('mirror max brightness')
    ax['A'].vlines(grating_onsets_shifted, ymin=np.min(segment), ymax=np.max(segment), color='orange', linestyle='--', label='psychopy onset')
    ax['A'].vlines(video_onsets, ymin=np.min(segment), ymax=np.max(segment), color='k', linestyle=':', label='video onset')
    ax['A'].set_xlim(segment_seconds[0], segment_seconds[-1])
    ax['A'].margins(x=0)
    ax['A'].legend()
    
    trials_reorg = trials_df.reset_index()
    trials_reorg.index.name = 'epoch'
    merged = onset_epochs_df.merge(trials_reorg, on='epoch')

    sns.lineplot(merged, x='time',y='video', color='k', style='epoch', hue='contrastLeft', alpha=0.5, ax=ax['D'], palette='viridis', dashes='')
    ax['D'].set_title(f'all onsets, {len(vid_onsets_samples)} detected')
    ax['D'].set_ylabel('change in brightness')
    ax['D'].set_xlabel('time from onset')
    ax['D'].legend().remove()
    
    sns.lineplot(merged, x='time',y='video', color='k', hue='contrastLeft', alpha=0.5, ax=ax['E'], palette='viridis')
    ax['E'].set_title('mean over left contrast')
    ax['E'].set_ylabel('change in brightness')
    ax['E'].set_xlabel('time from onset')
    # barplot version
    # sns.barplot(merged[(merged.time>0)&(merged.time<.05)], x='contrastLeft', y='video')

    ax['F'].scatter(np.arange(len(video_onsets))[onsets_not_nan], video_onsets[onsets_not_nan] - grating_onsets_shifted[onsets_not_nan], c='purple') 
    ax['F'].set_xlim(0,len(video_onsets))
    ax['F'].set_xlabel('trial')
    ax['F'].set_ylabel('offset (video - psychopy) [s]')   

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    sns.despine(trim=True)
    fig.savefig(os.path.join(folder_save, fig_name))

    return fig_name

def process_video(folder_path, subj):
    
    alf_path = os.path.join(folder_path, subj, 'alf')
    vid_filename = [x for x in os.listdir(alf_path) if '.mkv' in x][0]

    # open video
    cap = cv2.VideoCapture(os.path.join(alf_path,vid_filename))
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # get some info from video
    vid_samplerate = cap.get(cv2.CAP_PROP_FPS) 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # crop video to just include mirror
    if os.path.exists(os.path.join(alf_path, 'mirror_coords.npy')):
        x, y, w, h = np.load(os.path.join(alf_path, 'mirror_coords.npy'))
    else:
        attempt = 0
        coords = []
        while not coords and attempt < 10:
            selected_idx = np.random.randint(total_frames)
            print(f'attempt {attempt+1}, trying frame {selected_idx}')
            cap.set(cv2.CAP_PROP_POS_FRAMES, selected_idx)
            ret, frame = cap.read()
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            coords = find_mirror(frame_bw)
            attempt += 1
        x, y, w, h = coords
        np.save(os.path.join(alf_path, 'mirror_coords'), coords)

    # compute and save max brightness of mirror for every frame
    if os.path.exists(os.path.join(alf_path, 'stim_brightness.npy')):
        stim_brightness = np.load(os.path.join(alf_path, 'stim_brightness.npy'))
    else:  # 20 mins or so
        stim_brightness = np.empty((total_frames, 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print('computing max of frames... this will take a while...')    
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_crop = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            stim_brightness[i] = frame_crop.max()
        np.save(os.path.join(alf_path, 'stim_brightness'), stim_brightness)

    if os.path.exists(os.path.join(alf_path, 'video_times.npy')):
        t = np.load(os.path.join(alf_path, 'video_times.npy'))
    else:
        t = np.arange(0, total_frames/vid_samplerate, step=1/vid_samplerate)
        np.save(os.path.join(alf_path, 'video_times'), t)

    cap.release()

    # use the grating onsets and audio onsets to epoch vid contrast
    audio_onsets_filename = [x for x in os.listdir(alf_path) if 'audio_onsets' in x][0]
    audio_onsets = np.load(os.path.join(alf_path, audio_onsets_filename))

    #  load trials
    trials_df = load_trials(folder_path, subj)

    # find onset times from psychopy and audio
    grating_onsets = trials_df['grating_l.started'].values
    grating_onsets_shifted = grating_onsets-grating_onsets[0]
    onsets_shifted = grating_onsets_shifted + audio_onsets[0]

    # find onset times from video
    vid_onsets_samples_nans = detect_video_onsets(stim_brightness, onsets_shifted)
    onsets_not_nan = ~np.isnan(vid_onsets_samples_nans)
    vid_onsets_samples = vid_onsets_samples_nans[onsets_not_nan].astype(int)

    vid_onsets_secs = np.zeros_like(vid_onsets_samples_nans, dtype=np.float64)
    vid_onsets_secs[onsets_not_nan] = t[vid_onsets_samples]
    if np.any(~onsets_not_nan):
        vid_onsets_secs[~onsets_not_nan] = np.nan
    np.save(os.path.join(alf_path, 'video_onsets'), vid_onsets_secs)

    print(f'success: {len(np.sum(onsets_not_nan))} video onsets detected')


def detect_video_onsets(frame_data, psychopy_onsets, search_window_t=0.2, sampling_rate=60):
    psychopy_onsets_samples = psychopy_onsets * sampling_rate
    search_window_samples = int(search_window_t * sampling_rate)
    onset_idx = []
    for o in psychopy_onsets_samples:
        search_window = frame_data[int(o)-search_window_samples:int(o)+search_window_samples]
        # plt.plot(search_window)
        if np.any(np.argmax((search_window - np.roll(search_window,1))[1:]>1)): 
            increase_idx = np.argmax((search_window - np.roll(search_window,1))[1:]>1) + 1
            onset_idx.append(int(o)-search_window_samples + increase_idx)
            # plt.scatter(increase_idx, 250)
        else: 
            onset_idx.append(np.nan)
    return np.array(onset_idx)