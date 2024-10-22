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
usr = os.path.expanduser("~")
if usr == '/Users/uraiae': # mounted using mountainduck
    folder_path = '/Volumes/macOS/Users/uraiae/VISUAL-DECISIONS.localized/subjects/006'
else: # for local data
    folder_path = "./data"

# Extract file names and contents from the specified folder
downloaded_files = utils.get_files_from_folder(folder_path)
# print(downloaded_files)

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
            
            data = utils.convert_psychopy_one(data, file_name)

 
            
    except  Exception as e:
        print("skipped file with error", file_name, e)


# %%
