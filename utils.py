import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 

import os
from io import StringIO
import re
from pathlib import Path

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
# plotting function
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
