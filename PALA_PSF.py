# PALA_PSF.py

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import time
from PALA_SetUpPaths import PALA_SetUpPaths
from PALA_AddNoiseInIQ import PALA_AddNoiseInIQ
from PALA_multiULM_mesh import PALA_multiULM_mesh

def print_structure(var, indent=0):
    """Recursively prints the structure of a variable."""
    if isinstance(var, np.ndarray):
        if var.dtype.names:
            for name in var.dtype.names:
                print('  ' * indent + f"{name}:")
                print_structure(var[name], indent + 1)
        else:
            print('  ' * indent + str(var))
    elif isinstance(var, dict):
        for key, value in var.items():
            print('  ' * indent + f"{key}:")
            print_structure(value, indent + 1)
    else:
        print('  ' * indent + str(var))

# Set up paths and get the data folder
PALA_addons_folder, PALA_data_folder = PALA_SetUpPaths()

# Select IQ and Media folder
print('Running PALA_SilicoPSF.py')

workingdir = os.path.join(PALA_data_folder, 'PALA_data_InSilicoPSF')
if not os.path.exists(workingdir):
    os.makedirs(workingdir)
os.chdir(workingdir)

filename = 'PALA_InSilicoPSF'
myfilepath = os.path.join(workingdir, filename)

# Load data from the .mat file
data = loadmat(f'{myfilepath}_sequence.mat')
P, PData, Media = data['P'], data['PData'], data['Media']

# Check PData['Size'] structure
print(f"PData['Size']: {PData['Size']}, shape: {PData['Size'].shape}")

# Access the nested array within PData['Size']
size_array = PData['Size'][0][0]  # Extract the array from the nested structure
print(f"Extracted size array: {size_array}")  # Print to check its contents

# Flatten the array and access the z and x sizes
size_array = size_array.flatten()  # Flatten the array to get [9, 9, 1]
size_z = size_array[0]  # z size (9 in this case)
size_x = size_array[1]  # x size (9 in this case)

# Extract scalar values for P['BlocSize'] and P['numBloc']
bloc_size = P['BlocSize'][0, 0].item()  # Convert to Python scalar
num_bloc = P['numBloc'][0, 0].item()    # Convert to Python scalar

ULM = {
    'numberOfParticles': 1,
    'res': 10,
    'fwhm': [3, 3],
    'size': [size_z, size_x, int(bloc_size * num_bloc)],  # Using the extracted scalars
    'scale': [1, 1, 1],
    'algorithm': 'wa',
    'numberOfFramesProcessed': int(bloc_size * num_bloc),
    'parameters': {}  # Added to prevent KeyError
}

res = ULM['res']

# Investigate PData['PDelta'] structure
print(f"PData['PDelta']: {PData['PDelta']}, shape: {PData['PDelta'].shape}")

# Extract the correct element from PData['PDelta'] for the x and z dimensions
PDelta_values = PData['PDelta'][0][0].flatten()  # Flatten the array to ensure proper access
PDelta_x = PDelta_values[0]  # x dimension value (1)
PDelta_z = PDelta_values[2]  # z dimension value (1)

# Extract scalar origin values
origin = PData['Origin'][0][0]  # Shape: (1, 3)
origin_x = origin[0, 0]         # Extract x origin
origin_z = origin[0, 2]         # Extract z origin

# Calculate lx and lz
lx = origin_x + np.arange(size_x) * PDelta_x
lz = origin_z + np.arange(size_z) * PDelta_z

# --- Start of Fix ---

# Inspect the Media structure
print("Inspecting Media structure:")
print_structure(Media)

# Extract unique x and z positions from ListPos
list_pos_array = Media['ListPos'][0][0]  # Shape: (N, 4)

# Limit to first 100 points to match localization output
list_pos_subset = list_pos_array[:100, :]  # Shape: (100, 4)
unique_x = np.unique(list_pos_subset[:, 0])
unique_z = np.unique(list_pos_subset[:, 2])

ll_x = unique_x
ll_z = unique_z

print(f"Unique x positions (ll_x): {ll_x}")
print(f"Unique z positions (ll_z): {ll_z}")
print(f"Number of unique x positions: {len(ll_x)}")
print(f"Number of unique z positions: {len(ll_z)}")

Npoints = len(list_pos_subset)  # Total number of simulated IQ positions (100)
print(f"Npoints (Total IQ Positions): {Npoints}")

# Define list of algorithms and compute Nalgo
listAlgo = ['no_shift', 'wa', 'interp_cubic', 'interp_lanczos', 'interp_spline', 'gaussian_fit', 'radial']
Nalgo = len(listAlgo)

# Simulated Noise parameters
NoiseParam = {
    'Power': -2,
    'Impedance': 0.2,
    'SigmaGauss': 1.5,
    'clutterdB': -30,
    'amplCullerdB': 10
}

# List of SNR values
listdB = [-60, -40, -30, -25, -20, -15, -10]

# Display noised IQ example
temp = loadmat(f'{myfilepath}_IQ001.mat')
IQ_speckle = PALA_AddNoiseInIQ(np.abs(temp['IQ']), NoiseParam)

plt.figure(1)
plt.imshow(np.abs(IQ_speckle[:, :, 1]), cmap='gray')
plt.title('Noised IQ Example (Frame 1)')
plt.colorbar()
plt.show()

# Display IQ noise and scatter positions
dB = 20 * np.log10(np.abs(IQ_speckle)) - np.max(20 * np.log10(np.abs(IQ_speckle)))
plt.figure(2)
for ii in range(0, min(100, len(list_pos_subset)), 10):
    plt.imshow(dB[:, :, ii], cmap='gray', vmin=-30, vmax=0)
    plt.colorbar()
    plt.scatter(list_pos_subset[ii, 0], list_pos_subset[ii, 2], color='r')
    plt.title(f'IQ Noise with Scatter Positions (Frame {ii})')
    plt.pause(0.5)
plt.show()

# Load and localize data
t_start = time.time()
for idB, dB_val in enumerate(listdB):
    print(f'Simulation {idB+1}/{len(listdB)}: SNR = {abs(dB_val)} dB')
    NoiseParam['clutterdB'] = dB_val
    MatLocFull = []  # Initialize as a list to collect MatTracking arrays

    temp = loadmat(f'{myfilepath}_IQ001.mat')
    IQ = temp['IQ']

    Nt = 2  # Number of repetitions of noise occurrence
    for irp in range(Nt):
        print(f'Iteration {irp+1}: ')
        IQ_speckle = PALA_AddNoiseInIQ(np.abs(IQ), NoiseParam)
        MatTracking = PALA_multiULM_mesh(IQ_speckle, listAlgo, ULM, PData)
        MatLocFull.append(MatTracking)  # Append each MatTracking to the list

    # After collecting all MatTracking arrays, stack them along axis=3
    try:
        MatLocFull = np.stack(MatLocFull, axis=3)  # Shape: (2, Nalgo, 100, Nt)
    except ValueError as e:
        print(f"Stacking Error: {e}")
        print(f"MatLocFull list length: {len(MatLocFull)}")
        for idx, mat in enumerate(MatLocFull):
            print(f"MatLocFull[{idx}].shape: {mat.shape}")
        raise

    savemat(f'{myfilepath}_LocalMesh{abs(NoiseParam["clutterdB"])}dB.mat', {
        'MatLocFull': MatLocFull,
        'Media': Media,
        'Nt': Nt,
        'ULM': ULM,
        'P': P,
        'listAlgo': listAlgo,
        'NoiseParam': NoiseParam,
        'Nalgo': Nalgo,
        'll_x': ll_x,  # Save unique x positions
        'll_z': ll_z   # Save unique z positions
    })

    # Reshape and compute errors
    MatLocFull_w = MatLocFull.copy()

    # Correctly reshape MatPosSim using the subset
    MatPosSim = list_pos_subset[:, [2, 0]].T[:, np.newaxis, :, np.newaxis]  # Shape: (2,1,100,1)
    print(f"MatPosSim shape: {MatPosSim.shape}")  # Debugging statement

    # Tile MatPosSim to match Nt
    MatPosSim = np.tile(MatPosSim, (1, 1, 1, Nt))  # Shape: (2,1,100,2)

    print(f"MatPosSim shape after tiling: {MatPosSim.shape}")  # Debugging statement

    # Adjust MatLocFull_w to match MatPosSim dimensions
    # No slicing needed as both have N=100
    print(f"MatLocFull_w shape before subtraction: {MatLocFull_w.shape}")  # Debugging statement

    # Compute localization errors
    try:
        ErrorFull = MatLocFull_w - MatPosSim  # Shape: (2, Nalgo, 100, Nt)
    except ValueError as e:
        print(f"Subtraction Error: {e}")
        raise

    # Reshape errors for analysis
    try:
        ErrorFull = ErrorFull.reshape((2, Nalgo, Npoints, -1))  # Shape: (2, Nalgo, 100, Nt)
    except ValueError as e:
        print(f"Reshape Error: {e}")
        print(f"ErrorFull shape before reshape: {ErrorFull.shape}")
        print(f"Expected reshape dimensions: (2, {Nalgo}, {Npoints}, ...)")
        raise

    # Static error removal and computation
    ListMean = np.mean(ErrorFull, axis=3)  # Shape: (2, Nalgo, 100)
    ListMean = np.concatenate((ListMean, np.mean(np.sqrt(np.sum(ErrorFull ** 2, axis=0)), axis=2)[None, :]), axis=0)  # Shape: (3, Nalgo, 100)

    if abs(NoiseParam['clutterdB']) == 60:
        StaticError = np.mean(ListMean, axis=2)  # Shape: (3, Nalgo)
    else:
        temp_static = loadmat(f'{myfilepath}_LocalMesh60dB.mat')
        StaticError = temp_static['StaticError']

    ListMean -= StaticError[:, :, np.newaxis]  # Shape: (3, Nalgo, 100)

    # Calculate RMSE
    ListMean[2, :, :] = np.mean(np.sqrt(np.sum((ErrorFull - StaticError[:2, :, np.newaxis, np.newaxis]) ** 2, axis=0)), axis=2)

    # Calculate variance
    ListVar = np.var(ErrorFull, axis=3)  # Shape: (2, Nalgo, 100)
    ListVar = np.concatenate((ListVar, np.var(np.sqrt(np.sum(ErrorFull ** 2, axis=0)), axis=2)[None, :]), axis=0)  # Shape: (3, Nalgo, 100)
    ListVar[2, :, :] = np.var(np.sqrt(np.sum(ErrorFull ** 2, axis=0)), axis=2)

    savemat(f'{myfilepath}_LocalMesh{abs(NoiseParam["clutterdB"])}dB.mat', {
        'StaticError': StaticError,
        'ListMean': ListMean
    }, appendmat=True)

    # Display errors' maps
    # Define color axis ranges
    Caxis = {
        'x': [0, 0.5],
        'y': [0, 0.5],
        'RMSE': [0, 0.5]
    }

    # Initialize the figure with appropriate dimensions
    fig, axes = plt.subplots(3, Nalgo, figsize=(4 * Nalgo, 12))

    # Reshape ListMean to create ErrorMaps
    # Assuming ListMean has shape (3, Nalgo, 100)
    # Reshape to (3, Nalgo, 10, 10) assuming a 10x10 grid
    try:
        grid_size = 10  # Since 10x10=100
        ErrorMaps = np.abs(ListMean).reshape((3, Nalgo, grid_size, grid_size))
    except ValueError as e:
        print(f"Reshape Error for ErrorMaps: {e}")
        print(f"ListMean shape: {ListMean.shape}")
        print(f"Expected reshape dimensions: (3, {Nalgo}, {grid_size}, {grid_size})")
        raise

    ErrorMaps = np.transpose(ErrorMaps, [2, 3, 0, 1])  # Now shape: (10, 10, 3, Nalgo)

    for ialgo in range(Nalgo):
        # Plot Lateral Error
        ax = axes[0, ialgo]
        im = ax.imshow(ErrorMaps[:, :, 1, ialgo], vmin=Caxis['x'][0], vmax=Caxis['x'][1], cmap='viridis', 
                       extent=(ll_x.min(), ll_x.max(), ll_z.min(), ll_z.max()), origin='lower')
        ax.set_title(listAlgo[ialgo])
        if ialgo == 0:
            ax.set_ylabel('Lateral Error')

        # Plot Axial Error
        ax = axes[1, ialgo]
        im = ax.imshow(ErrorMaps[:, :, 0, ialgo], vmin=Caxis['y'][0], vmax=Caxis['y'][1], cmap='viridis',
                       extent=(ll_x.min(), ll_x.max(), ll_z.min(), ll_z.max()), origin='lower')
        if ialgo == 0:
            ax.set_ylabel('Axial Error')

        # Plot RMSE
        ax = axes[2, ialgo]
        im = ax.imshow(ErrorMaps[:, :, 2, ialgo], vmin=Caxis['RMSE'][0], vmax=Caxis['RMSE'][1], cmap='viridis',
                       extent=(ll_x.min(), ll_x.max(), ll_z.min(), ll_z.max()), origin='lower')
        if ialgo == 0:
            ax.set_ylabel('RMSE')

    # Adjust layout and add colorbars
    plt.tight_layout()
    plt.show()

# End of simulation
t_end = time.time()
elapsed_hours = int((t_end - t_start) // 3600)
elapsed_minutes = ((t_end - t_start) % 3600) / 60
print(f'PALA_SilicoPSF.py performed in {elapsed_hours} hours and {elapsed_minutes:.1f} minutes')

# Run the figure plotting script
exec(open('PALA_SilicoPSF_fig.py').read())
