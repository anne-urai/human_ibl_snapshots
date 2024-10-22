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
from io import StringIO
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 
sns.set_style('darkgrid')
from pathlib import Path
import utils

#%% =============================== #
# get files and file contents
# ================================= #

# Specify the path to the folder containing your CSV files
usr = os.path.expanduser("~")
if usr == '/Users/uraiae': # mounted using mountainduck
    folder_path = '/Volumes/macOS/Users/uraiae/VISUAL-DECISIONS.localized/subjects/'
else: # for local data
    folder_path = "./data"

figures_folder = os.path.join(os.getcwd(), 'figures') # to save

# loop over all subject folders
subjects = os.listdir(folder_path)
print(subjects)

for subj in subjects:

    # don't redo if the snapshots are there already
    if os.path.exists(os.path.join(figures_folder, subj + '_behavior_snapshot.png')):
         continue

    try:
        # # BEHAVIORAL DATA FROM PSYCHOPY CSV FILE
        behavior_file_name = [s for s in os.listdir(os.path.join(folder_path, subj)) if s.endswith('.csv')]
        data = pd.read_csv(os.path.join(folder_path, subj, behavior_file_name[0]))
        data = utils.convert_psychopy_one(data, behavior_file_name[0])
        utils.plot_snapshot_behavior(data, figures_folder, subj + '_behavior_snapshot.png')

    except Exception as e:
        print("skipped subject with error", subj, e)

    try:
        # EYETRACKING DATA FROM EYELINK ASC FILE
        pupil_file_name = [s for s in os.listdir(os.path.join(folder_path, subj)) if s.endswith('.asc')]
        utils.plot_snapshot_pupil(os.path.join(folder_path, subj, pupil_file_name[0]),
                                figures_folder, subj + '_pupil_snapshot.png')
    except Exception as e:
        print("skipped subject with error", subj, e)

    #TODO: webcam snapshot
    
# %%
