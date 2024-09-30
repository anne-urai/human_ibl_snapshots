"""
Use an MNE environment to run the script

@anne-urai
"""

#%% =============================== #

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from io import StringIO
#import h5py

from pathlib import Path
import mne
import utils

# Specify the path to the folder containing your CSV files
folder_path = "./data/human_pupil"
# Extract file names and contents from the specified folder
downloaded_files = utils.get_files_from_folder(folder_path, extension='.asc')

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

            # load pupil data:
            raw_et = mne.io.read_raw_eyelink(file_name)
            raw_et_df = raw_et.to_data_frame()
            interp_et = mne.preprocessing.eyetracking.interpolate_blinks(raw_et, buffer=(0.05, 0.1), interpolate_gaze=True)
            interp_et_df = interp_et.to_data_frame()

            # find the channels, these may be left or right eye
            pupil_chan = [c for c in interp_et_df.columns if 'pupil' in c][0]
            x_chan = [c for c in interp_et_df.columns if 'x' in c][0]
            y_chan = [c for c in interp_et_df.columns if 'y' in c][0]

            # get raw annotations
            annot = raw_et.annotations.to_data_frame()
            annot['onset'] = raw_et.annotations.onset

            # lock to stimulus onset, response and feedback
            # get events
            stim_events_dict = {"signed_contrast -0.02": -0.02, "signed_contrast -0.05": -0.05, "signed_contrast -0.10": -0.10, 
                                "signed_contrast -0.20": -0.20, "signed_contrast 0.00": 0.00, "signed_contrast 0.02": 0.02, "signed_contrast 0.05": 0.05,
                                "signed_contrast 0.10": 0.10, "signed_contrast 0.20": 0.20}
            events, _ = mne.events_from_annotations(raw_et, event_id=stim_events_dict)
            epochs = mne.Epochs(interp_et, events,  tmin=-1, tmax=3, baseline=None, preload=True, reject=None)   
            stim_epochs_df = epochs.to_data_frame()

            response_events_dict = {"response 1":1, "response -1":-1}
            events, _ = mne.events_from_annotations(raw_et, event_id=response_events_dict)
            epochs = mne.Epochs(interp_et, events,  tmin=-1, tmax=3, baseline=None, preload=True, reject=None)   
            resp_epochs_df = epochs.to_data_frame()

            feedback_events_dict = {"feedback 1":1, "feedback 0":0}
            events, _ = mne.events_from_annotations(raw_et, event_id=feedback_events_dict)
            epochs = mne.Epochs(interp_et, events,  tmin=-1, tmax=3, baseline=None, preload=True, reject=None)   
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
                sns.lineplot(data=stim_epochs_df, x='time', y=var, estimator=None, 
                            hue='condition', palette='rocket', alpha=0.5,
                            legend=False,
                            units='epoch', ax=axes[axn])

            for axn, var in zip(['c', 'e', 'g'], [pupil_chan, x_chan, y_chan]):
                sns.lineplot(data=fb_epochs_df, x='time', y=var, estimator=None, 
                            hue='condition', legend=False, alpha=0.5,
                            palette=['green', 'red'], hue_order=['1','0'],
                            units='epoch', ax=axes[axn])

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

            fig.suptitle(os.path.split(fig_name)[-1])
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            sns.despine(trim=True)
            fig.savefig(fig_name)

    except  Exception as e:
        print("skipped file with error", file_name, e)
