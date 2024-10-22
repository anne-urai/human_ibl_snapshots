import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 

import os
from io import StringIO
import re
from pathlib import Path
import mne

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
    
    # recode some things
    if 'mouse.x' in data.keys(): # do this if mouse responses
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
        
    # TODO: return separate smaller df with session- and participant-level info
    # how to save these in ALF on J-drive?

    return data

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
