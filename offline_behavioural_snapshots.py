# -*- coding: utf-8 -*-
"""
Created on March 26 2024
@author: cami-uche-enwe
Leiden University
"""
#%% =============================== #
# import packages
# ================================= #
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 
sns.set_style('darkgrid')
import brainbox.behavior.pyschofit as psy

import os
from io import StringIO

#%% =============================== #
# get files and file contents
# ================================= #

def get_files_from_folder(folder_path):
    # Get all files in the folder
    files = os.listdir(folder_path)
    # Filter only CSV files
    csv_files = [file for file in files if file.endswith('.csv')]
    # Create absolute paths for CSV files
    csv_paths = [os.path.join(folder_path, file) for file in csv_files]
    
    downloaded_files = []
    for csv_path in csv_paths:
        with open(csv_path, 'r') as file:
            file_content = file.read()
            downloaded_files.append((csv_path, file_content))
    print(downloaded_files)
    return downloaded_files

# Specify the path to the folder containing your CSV files
folder_path = "./data"
# Extract file names and contents from the specified folder
downloaded_files = get_files_from_folder(folder_path)


#%% =============================== #
# plotting function
# ================================= #
def plot_psychometric(df, **kwargs):
    
    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        ax = plt.gca()
    
    # from https://github.com/int-brain-lab/paper-behavior/blob/master/paper_behavior_functions.py#L391
    # summary stats - average psychfunc
    df2 = df.groupby(['signed_contrast']).agg(count=('response', 'count'),
                                              mean=('response', 'mean')).reset_index()    
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
                 color='black', zorder=10, **kwargs)
    
    # plot datapoints on top
    sns.lineplot(data=df, 
                  x='signed_contrast', y='response', err_style="bars", 
                  linewidth=0, mew=0.5, zorder=20,
                  marker='o', errorbar=('ci',68), color='black', **kwargs)
    
    # paramters in title
    ax.set_title(r'$\mu=%.2f, \sigma=%.2f, \gamma=%.2f, \lambda=%.2f$'%tuple(pars),
              fontsize='x-small')


#%% =============================== #
# plot and save figures
# ================================= #

# loop over file name and content of csv 
for file_name, file_content in downloaded_files:    
    try:
        fig_name = file_name.split('/')[-1].split('_202')
        fig_name = os.path.join(os.getcwd(), 'offline_snapshot_figures', 
                                '202'+ fig_name[1].split('.csv')[0] + '_' +  fig_name[0].replace('data\\','') + '.png')
        
        if os.path.exists(fig_name):
            print("skipping ", file_name, ", already exists")
        else:
            # type(file_content) == string
            # parse string using CSV format into a Python Pandas Dataframe
            data = pd.read_csv(StringIO(file_content)) # string IO pretends to be a file handle
            print("reading in ", file_name)
            
            # recode some things
            if 'mouse.x' in data.keys(): # do this if mouse responses
                response_mappings = pd.DataFrame({'eccentricity':[15,15,-15,-15], 'correct':[1,0,1,0], 'response':[1,0,0,1]})
                data = data.merge(response_mappings, on=['eccentricity', 'correct'], how='left') # right response = 1, left response = 0
            else: # do this if keyboard responses
                data['response'] = data['key_resp.keys'].map({'x': 1, 'm': 0}, na_action=None)

            block_breaks = True if data['block_breaks'].iloc[0] == 'y' else False

            # drop practice trials
            session_start_ind = np.max(np.argwhere(data['session_start'].notnull()))
            data = data.iloc[session_start_ind:]
            
            # add new column that counts all trials
            data['trials.allN'] = data.reset_index().index

            # ============================= %
            # from https://github.com/int-brain-lab/IBL-pipeline/blob/master/prelim_analyses/behavioral_snapshots/behavior_plots.py
            # https://github.com/int-brain-lab/IBL-pipeline/blob/7da7faf40796205f4d699b3b6d14d3bf08e81d4b/prelim_analyses/behavioral_snapshots/behavioral_snapshot.py
            plt.close('all')
            fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(10,5))
            
            # 1. psychometric
            plot_psychometric(data, ax=ax[0])
            ax[0].set(xlabel='Signed contrast', ylabel='Choice (fraction)')
          
            # 2. chronometric
            sns.lineplot(data=data, ax=ax[1],
                         x='signed_contrast', y='response_time', err_style="bars", 
                         linewidth=1, estimator=np.median, 
                         mew=0.5,
                         marker='o', errorbar=('ci',68), color='black')
            ax[1].set(xlabel='Signed contrast', ylabel='RT (ms)',) #ylim=[0,2])
    
            # 3. time on task
            sns.scatterplot(data=data, ax=ax[2], 
                            x='trials.allN', y='response_time', 
                            style='correct', hue='correct',
                            palette={1:"#009E73", 0:"#D55E00"}, 
                            markers={1:'o', 0:'X'}, s=10, edgecolors='face',
                            alpha=.5, legend=False)
            
            # running median overlaid
            sns.lineplot(data=data[['trials.allN', 'response_time']].rolling(10).median(), 
                         ax=ax[2],
                         x='trials.allN', y='response_time', color='black', errorbar=None, 
                         label='median response time')
            sns.lineplot(data=data[['trials.allN', 'reaction_time']].rolling(10).median(), 
                         ax=ax[2],
                         x='trials.allN', y='reaction_time', color='black', alpha=0.2, errorbar=None, 
                         label='median reaction time')
            if block_breaks:
                [plt.axvline(x, color='blue', alpha=0.2, linestyle='--') for x in np.arange(100,data['trials.allN'].iloc[-1],step=100)]
            ax[2].set(xlabel="Trial number", ylabel="RT (ms)",) #ylim=[0.1, 10])
            ax[2].set_yscale("log")
            ax[2].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos:
                ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    
            fig.suptitle(fig_name.split('\\')[-1])
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            sns.despine(trim=True)
            fig.savefig(fig_name)
            
    except  Exception as e:
        print("skipped file with error", file_name, e)


# %%
