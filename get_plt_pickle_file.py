# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:54:08 2025

@author: imrul
"""

import os


    

def get_plt_pickle_file(folder_path, key = 'pl_bf'):
    latest_folder = None
    latest_time = 0
    for folder in os.listdir(folder_path):
        if key in folder:  # Check if the keyword is in the folder name
            full_folder_path = os.path.join(folder_path, folder)
            if os.path.isdir(full_folder_path):  # Ensure it's a directory
                # Iterate through files in the folder
                for file in os.listdir(full_folder_path):
                    if file.endswith('.pickle'):  # Check for pickle files
                        full_file_path = os.path.join(full_folder_path, file)
                        file_time = os.path.getmtime(full_file_path)  # Get modification time
                        if file_time > latest_time:  # Update if it's the latest
                            latest_time = file_time
                            latest_file_path = full_file_path
    
    # Print the latest pickle file path, if any
    try:
        if latest_file_path:
            print(f"Latest pickle file: {latest_file_path}")
            return latest_file_path
        else:
            print("No pickle files found in the specified folders.")  
            return None
    except:
        return None
    
    


folder_paths = r'W:\raspberrypi\photos\Juravinski\2025-01-16'

key = 'pl_bf'

pickle_files = []


for folder in os.listdir(folder_paths):
    
    folder_path = os.path.join(folder_paths,folder)
    
    if os.path.isdir(folder_path):    
        pickle_files.append(get_plt_pickle_file(folder_path))
    

pickle_files = [item for item in pickle_files if item is not None]    
for pickle_file in pickle_files:
    print(pickle_file)
    
    # pickle_files.append(get_plt_pickle_file(folder_path))
    
    
    
