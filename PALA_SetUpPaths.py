import sys
import os

def PALA_SetUpPaths():
    """
    Define paths for data and addons. This function sets up the environment paths required
    to run the PALA scripts.
    """
    # Define the addons directory on your computer
    PALA_addons_folder = '/Users/eric/Desktop/PALA/PALA_addons'
    
    # Define the data directory on your computer
    PALA_data_folder = '/Users/eric/Desktop/PALA_data'
    
    # Add the addons folder to the system path
    if os.path.exists(PALA_addons_folder):
        sys.path.append(PALA_addons_folder)
        print(f"Addons folder added to path: {PALA_addons_folder}")
    else:
        print(f"Addons folder does not exist: {PALA_addons_folder}")
    
    # Add the data folder to the system path
    if os.path.exists(PALA_data_folder):
        sys.path.append(PALA_data_folder)
        print(f"Data folder added to path: {PALA_data_folder}")
    else:
        print(f"Data folder does not exist: {PALA_data_folder}")

    return PALA_addons_folder, PALA_data_folder
