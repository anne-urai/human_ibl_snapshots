# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:17:56 2022

@author: ann-k, anne-urai
Leiden University
"""

# import useful packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 
sns.set_style('darkgrid')
import brainbox.behavior.pyschofit as psy

import os, urllib # library to acess url links
from io import StringIO

#%% ================================== #
# functions
# ================================== #

# function to acquire list of csv files from pavlovia gitlab
def get_files(contents):
    import re # regular expressions
    actual_regular_expression = 'href="(.*?)"' # () => group  => extract this part ".*" => indicates that all kind of special letters should be included. ? => take letters until "
    matches = re.findall(actual_regular_expression, contents) # use regular expression to extract all links
    # matches =>all links that were found in href (also not CSV)
    csv_files = list(filter(lambda match: match.endswith('.csv'), matches)) # from found links, filter everything that doesnt end with a .csv extension
    raw_csv_links = list(map(lambda link: link.replace("blob", "raw"), csv_files)) # basically clicking the raw link in the HTML so we only get what we want
    
    downloaded_files = [] #make a list
    for file_name in raw_csv_links: # loop through each CSV file link
        downloaded_files.append((file_name,
          urllib.request.urlopen(f"https://gitlab.pavlovia.org/{file_name}").read().decode("UTF-8"))) # read URL into a String
    return downloaded_files


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
    sns.lineplot(xlims,psy.erf_psycho_2gammas(pars, xlims), 
                 color='black', zorder=10, **kwargs)
    
    # plot datapoints on top
    sns.lineplot(data=df, 
                  x='signed_contrast', y='response', err_style="bars", 
                  linewidth=0, mew=0.5, zorder=20,
                  marker='o', ci=68, color='black', **kwargs)
    
    # paramters in title
    ax.set_title(r'$\mu=%.2f, \sigma=%.2f, \gamma=%.2f, \lambda=%.2f$'%tuple(pars),
              fontsize='x-small')


#%% ================================== #
# determine which plots to make
# ================================== #

# read html code of gitlab file list
contents = urllib.request.urlopen("https://gitlab.pavlovia.org/Anninas/human_ibl_piloting/tree/master/data").read().decode("UTF-8")
# extract file names and file contents from gitlab
downloaded_files = get_files(contents) 

# loop over file name and string content of csv 
for file_name, file_content in downloaded_files:    
    try:
        fig_name = os.path.join('behavioral_snapshot_figures', file_name.split('/')[-1].replace('csv', 'png'))
        if os.path.exists(fig_name):
            print("skipping ", file_name, ", already exists")
        else:
            # type(file_content) == string
            # parse string using CSV format into a Python Pandas Dataframe
            data = pd.read_csv(StringIO(file_content)) #string IO pretends to be a file handle
            print("reading in ", file_name)
            
            # recode some things
            data['response'] = data['key_resp.keys'].map({'x': 1, 'm': 0}, na_action=np.nan)
            
            # ============================= %
            # from https://github.com/int-brain-lab/IBL-pipeline/blob/master/prelim_analyses/behavioral_snapshots/behavior_plots.py
            # https://github.com/int-brain-lab/IBL-pipeline/blob/7da7faf40796205f4d699b3b6d14d3bf08e81d4b/prelim_analyses/behavioral_snapshots/behavioral_snapshot.py
            plt.close('all')
            fig, ax = plt.subplots(ncols=3, nrows=1)
            
            # 1. psychometric
            plot_psychometric(data, ax=ax[0])
            ax[0].set(xlabel='Signed contrast', ylabel='Choice (fraction)')
          
            # 2. chronometric
            sns.lineplot(data=data, ax=ax[1],
                         x='signed_contrast', y='key_resp.rt', err_style="bars", 
                         linewidth=1, estimator=np.median, 
                         mew=0.5,
                         marker='o', ci=68, color='black')
            ax[1].set(xlabel='Signed contrast', ylabel='RT (s)', ylim=[0,2])
    
            # 4. time on task
            sns.scatterplot(data=data, ax=ax[2], 
                            x='trials.thisN', y='key_resp.rt', 
                            style='key_resp.corr', hue='key_resp.corr',
                            palette={1:"#009E73", 0:"#D55E00"}, 
                            markers={1:'o', 0:'X'}, s=10, edgecolors='face',
                            alpha=.5, legend=False)
            
            # running median overlaid
            sns.lineplot(data=data[['trials.thisTrialN', 'key_resp.rt']].rolling(10).median(), ax=ax[2],
                         x='trials.thisN', y='key_resp.rt', color='black', ci=None, )
            ax[2].set(xlabel="Trial number", ylabel="RT (s)", ylim=[0.1, 10])
            ax[2].set_yscale("log")
            ax[2].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos:
                ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    
            fig.suptitle(file_name.split('/')[-1])
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            sns.despine(trim=True)
            fig.savefig(fig_name)
    except  Exception as e:
        print("skipped file with error", file_name, e)
        
