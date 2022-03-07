# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:17:56 2022

@author: ann-k
"""

#####import useful packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import datetime
import seaborn as sns 



def get_files(contents):
    import re
    matches = re.findall('href="(.*?)"', contents)
    csv_files = list(filter(lambda match: match.endswith('.csv'), matches)) #find the csv files
    raw_csv_links = list(map(lambda link: link.replace("blob", "raw"), csv_files))
    
    downloaded_files = [] #make a list
    for file_name in raw_csv_links: #loop through the raw csv links
        downloaded_files.append((file_name, urllib.request.urlopen(f"https://gitlab.pavlovia.org/{file_name}").read().decode("UTF-8")))
    return downloaded_files

import urllib #library to acess url links
contents = urllib.request.urlopen("https://gitlab.pavlovia.org/Anninas/human_ibl_piloting/tree/master/data").read().decode("UTF-8")

files = get_files(contents)
print(files)



from io import StringIO
import csv
sns.set_theme()
for file_name, file_content in files:    
    print("looking at", file_name)
    data = pd.read_csv(StringIO(file_content)) #string IO pretends to be a file handle
    sns.relplot(#command to plot
            data=data,
            x="difference", y="mouse.time"
            
               )