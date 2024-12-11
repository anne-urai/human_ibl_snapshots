
import os
#ToDo: rewrite with Path

# Specify the path to the folder containing your CSV files
#ToDo: store this info in a better file
usr = os.path.expanduser("~")
if usr == '/Users/uraiae': # mounted using mountainduck
    folder_path = '/Volumes/macOS/Users/uraiae/VISUAL-DECISIONS.localized/subjects/'
elif usr == 'C:\\Users\\Philippa':
    folder_path = 'D:\winshare\workgroups\FSW\VISUAL-DECISIONS\subjects'
    # folder_path = "./data/subjects"
else: # for local data
    folder_path = "./data/subjects"

# figures stay in the repo for now
figures_folder = os.path.join(os.getcwd(), 'figures') # to save
