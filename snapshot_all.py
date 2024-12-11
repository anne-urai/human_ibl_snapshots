# -*- coding: utf-8 -*-
"""
Created on March 26 2024
@author: cami-uche-enwe
Leiden University

requirements: MNE installed; psychofit installed

"""

#%% =============================== #
# import packages
# ================================= #

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 
sns.set_style('darkgrid')
from pathlib import Path
import utils
from audio_extract import extract_audio
from tqdm import tqdm

# flexible paths
from utils_path import folder_path, figures_folder

#%% =============================== #
# get files and file contents
# ================================= #

# loop over all subject folders
subjects = sorted(os.listdir(folder_path))
print(subjects)
# subjects = subjects[0:10]

#%% =============================== #
# preprocess and extract
# ================================= #

for subj in tqdm(subjects):

    # make the folders necessary
    if not os.path.exists(os.path.join(folder_path, subj, 'raw_behavior_data')):
        for f in ['raw_behavior_data', 'raw_video_data', 'raw_eyelink_data', 'alf']:
            os.mkdir(os.path.join(folder_path, subj, f))

        # move files to the correct folders
        for exts in ['csv', 'psydat', 'log']:
            file_names = [s for s in os.listdir(os.path.join(folder_path, subj)) if s.endswith('.' + exts) and not s.startswith('.')]
            for f in file_names:
                os.rename(os.path.join(folder_path, subj, f), os.path.join(folder_path, subj, 'raw_behavior_data', f))    
        for exts in ['asc', 'EDF']:
            file_names = [s for s in os.listdir(os.path.join(folder_path, subj)) if s.endswith('.' + exts) and not s.startswith('.')]
            for f in file_names:
                os.rename(os.path.join(folder_path, subj, f), os.path.join(folder_path, subj, 'raw_eyelink_data', f))
        for exts in ['mkv', 'wav']:
            file_names = [s for s in os.listdir(os.path.join(folder_path, subj)) if s.endswith('.' + exts) and not s.startswith('.')]
            for f in file_names:
                os.rename(os.path.join(folder_path, subj, f), os.path.join(folder_path, subj, 'raw_video_data', f))

#%%
for subj in tqdm(subjects):

    print(subj)

    try:
        # extract psychopy file into usable trials df and session df
        behavior_file_name = [s for s in os.listdir(os.path.join(folder_path, subj, 'raw_behavior_data')) if s.endswith('.csv') and not s.startswith('.')]
        data = pd.read_csv(os.path.join(folder_path, subj, 'raw_behavior_data', behavior_file_name[0]))
        trials_df, session_df = utils.convert_psychopy_one(data, behavior_file_name[0])
        trials_df.to_csv(os.path.join(folder_path, subj, 'alf', 'trials_table.csv'))
        session_df.to_csv(os.path.join(folder_path, subj, 'alf', 'session_info.csv'))
    except:
        continue

    # # extract audio if not done (takes a while)
    # vid_file_name = [s for s in os.listdir(os.path.join(folder_path, subj, 'raw_video_data')) if s.endswith('.mkv') and not s.startswith('.')]
    # vid_path = os.path.join(folder_path, subj, 'raw_video_data', vid_file_name[0])
    # audio_path = f'{str(vid_path)[:-3]}wav'
    # if audio_path.split('\\')[-1] not in os.listdir(folder_path):
    #     extract_audio(input_path=vid_path, output_path=audio_path, output_format='wav', overwrite=False)
    # # extract pupil data


#%% =============================== #
# make snapshot figures
# ================================= #

#%%
for subj in subjects:

    # don't redo if the snapshots are there already
    if not os.path.exists(os.path.join(figures_folder, subj + '_behavior_snapshot.png')):
        try:
            # # BEHAVIORAL DATA FROM PSYCHOPY CSV FILE
            behavior_file_name = [s for s in os.listdir(os.path.join(folder_path, subj, 'alf')) if s.endswith('trials_df.csv') and not s.startswith('.')]
            data = pd.read_csv(os.path.join(folder_path, subj, behavior_file_name[0]))
            utils.plot_snapshot_behavior(data, figures_folder, subj + '_behavior_snapshot.png')
        except Exception as e:
            print("skipped subject with error", subj, e)
    else:
        continue

    if not os.path.exists(os.path.join(figures_folder, subj + '_pupil_snapshot.png')):
        try:
            # EYETRACKING DATA FROM EYELINK ASC FILE
            pupil_file_name = [s for s in os.listdir(os.path.join(folder_path, subj)) if s.endswith('.asc')]
            utils.plot_snapshot_pupil(os.path.join(folder_path, subj, pupil_file_name[0]),
                                    figures_folder, subj + '_pupil_snapshot.png')
        except Exception as e:
            print("skipped subject with error", subj, e)
    else:
        continue

    if not os.path.exists(os.path.join(figures_folder, subj + '_audio_snapshot.png')):
        try:
            utils.plot_snapshot_audio(os.path.join(folder_path, subj), figures_folder, subj + '_audio_snapshot.png')
        except Exception as e:
            print("skipped subject with error", subj, e)
    else:
        continue

    #TODO: webcam snapshot
    
# %%
