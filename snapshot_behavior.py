# -*- coding: utf-8 -*-
"""
Created on March 26 2024
@author: cami-uche-enwe
Leiden University

requirements: ibllib https://int-brain-lab.github.io/iblenv/notebooks_external/data_download.html

"""
#%% =============================== #
# import packages
# ================================= #

import os
from io import StringIO
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 
sns.set_style('darkgrid')
import utils

#%% =============================== #
# get files and file contents
# ================================= #

# Specify the path to the folder containing your CSV files
folder_path = "./data"
# Extract file names and contents from the specified folder
downloaded_files = utils.get_files_from_folder(folder_path)

#%% =============================== #
# plot and save figures
# ================================= #

# loop over file name and content of csv 
for file_name, file_content in downloaded_files:    
    try:
        fig_name = file_name.split('/')[-1].split('_202')
        fig_name = os.path.join(os.getcwd(), 'figures', 
                                '202'+ fig_name[1].split('.csv')[0] + '_' +  fig_name[0].replace('data\\','') + '.png')
        
        if os.path.exists(fig_name):
            print("skipping ", file_name, ", already exists")
        else:
            # type(file_content) == string
            # parse string using CSV format into a Python Pandas Dataframe
            data = pd.read_csv(StringIO(file_content)) # string IO pretends to be a file handle
            print("reading in ", file_name)
            
            # convert to ONE convention
            block_breaks = True if data['block_breaks'].iloc[0] == 'y' else False
            data = utils.convert_psychopy_one(data, file_name)

            # ============================= %
            # from https://github.com/int-brain-lab/IBL-pipeline/blob/master/prelim_analyses/behavioral_snapshots/behavior_plots.py
            # https://github.com/int-brain-lab/IBL-pipeline/blob/7da7faf40796205f4d699b3b6d14d3bf08e81d4b/prelim_analyses/behavioral_snapshots/behavioral_snapshot.py
            
            plt.close('all')

            fig, ax = plt.subplots(ncols=3, nrows=2, width_ratios=[1,1,2], figsize=(12,8))
            
            # 1. psychometric
            utils.plot_psychometric(data, ax=ax[0,0])
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
            
            # add lines to mark breaks
            if block_breaks: 
                [ax[0,2].axvline(x, color='blue', alpha=0.2, linestyle='--') for x in np.arange(100,data['trials.allN'].iloc[-1],step=100)]
            
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
                utils.plot_psychometric(dat, color=color, ax=ax[1,0])
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
                ax[1,1].set(xlabel='Signed contrast', ylabel='Response time (black) / movement initiation (grey) (s)', 
                        ylim=[0, 1.6])
        
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
            if block_breaks: 
                [plt.axvline(x, color='blue', alpha=0.2, linestyle='--') for x in np.arange(100,data['trials.allN'].iloc[-1],step=100)]
            ax[1,2].set(xlabel="Trial number", ylabel= "Stim (grey) / choice (black)")

            # ============================= %

            fig.suptitle(os.path.split(fig_name)[-1])
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            sns.despine(trim=True)
            fig.savefig(fig_name)
            
    except  Exception as e:
        print("skipped file with error", file_name, e)


# %%
