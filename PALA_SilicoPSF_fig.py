# PALA_SilicoFlow_fig.py

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.ndimage import gaussian_filter, maximum_filter
import tifffile
from matplotlib import gridspec

# Import custom modules
from PALA_SetUpPaths import PALA_SetUpPaths
from PALA_GetFormat import PALA_GetFormat
from WriteTif import write_tif  # Ensure the function is correctly named in WriteTif.py
from PALA_boxplot_multi import PALA_boxplot_multi  # Assume this is a custom function

# ---------------------------
# Helper Functions
# ---------------------------

def get_colormap(cmap_name='hot', num_colors=256):
    """
    Returns a sampled RGB array from the specified colormap.
    
    Parameters:
        cmap_name (str): Name of the Matplotlib colormap.
        num_colors (int): Number of color samples.
    
    Returns:
        np.ndarray: Array of shape (num_colors, 3) with RGB values.
    """
    cmap = plt.get_cmap(cmap_name)
    return cmap(np.linspace(0, 1, num_colors))[:, :3]

def convert_matout_to_float32(matout_object_array):
    """
    Converts an object array containing numpy arrays to a list of float32 numpy arrays.
    
    Parameters:
        matout_object_array (np.ndarray): Object array containing sub-arrays.
    
    Returns:
        list: List of float32 numpy arrays.
    """
    converted_matout = []
    for idx, subarr in enumerate(matout_object_array.flatten()):
        if isinstance(subarr, np.ndarray):
            try:
                converted_subarr = subarr.astype(np.float32)
                converted_matout.append(converted_subarr)
                print(f"Converted MatOut_multi['MatOut'][{idx}] to float32")
            except Exception as e:
                print(f"Failed to convert MatOut_multi['MatOut'][{idx}] to float32: {e}")
                converted_matout.append(subarr)  # Append original if conversion fails
        else:
            print(f"Element MatOut_multi['MatOut'][{idx}] is not a numpy array, skipping.")
            converted_matout.append(subarr)  # Or handle accordingly
    return converted_matout

# ---------------------------
# Main Execution
# ---------------------------

if __name__ == "__main__":
    # Set up paths
    PALA_addons_folder, PALA_data_folder = PALA_SetUpPaths()

    # Define working directories and filenames
    print('Running PALA_SilicoFlow_fig.py - Images generation')
    
    workingdir = os.path.join(PALA_data_folder, 'PALA_data_InSilicoFlow')
    os.makedirs(workingdir, exist_ok=True)
    os.chdir(workingdir)
    
    filename = 'PALA_InSilicoFlow'
    myfilepath = os.path.join(workingdir, filename)
    myfilepath_data = os.path.join(workingdir, 'Results', filename)
    myfilepath_fig = os.path.join(myfilepath_data, 'img')
    
    os.makedirs(myfilepath_fig, exist_ok=True)
    
    # Load variables from sequence.mat
    listVar = ['P', 'PData', 'Trans', 'Media', 'UF', 'Resource', 'Receive', 'filetitle']
    data_sequence = loadmat(f'{myfilepath}_sequence.mat')
    P = data_sequence['P']
    PData = data_sequence['PData']
    Trans = data_sequence['Trans']
    Media = data_sequence['Media']
    UF = data_sequence['UF']
    Resource = data_sequence['Resource']
    Receive = data_sequence['Receive']
    filetitle = data_sequence['filetitle']
    
    # Load and convert MatOutTarget
    MatOutTarget = loadmat(f'{workingdir}/{filename}_v3_config.mat')['MatOut']
    print(f"MatOutTarget dtype before conversion: {MatOutTarget.dtype}")
    MatOutTarget = MatOutTarget.astype(np.float32)  # Convert to float32
    print(f"MatOutTarget dtype after conversion: {MatOutTarget.dtype}")
    
    # Load Stats_multi and MatOut_multi
    Stats_multi = loadmat(f'{myfilepath_data}_Stats_multi30dB.mat')
    MatOut_multi = loadmat(f'{myfilepath_data}_MatOut_multi_30dB.mat')
    
    # Ensure MatOut_multi contains float32 data
    for key in MatOut_multi:
        # Skip MATLAB metadata keys
        if key in ['__header__', '__version__', '__globals__']:
            print(f"MatOut_multi['{key}'] is not a numpy array, skipping conversion.")
            continue
        
        # Handle 'MatOut' key separately
        if key == 'MatOut':
            if isinstance(MatOut_multi[key], np.ndarray) and MatOut_multi[key].dtype == 'object':
                print(f"Converting MatOut_multi['{key}'] elements to float32")
                MatOut_multi[key] = convert_matout_to_float32(MatOut_multi[key])
                print(f"MatOut_multi['{key}'] is now a list of float32 numpy arrays with length {len(MatOut_multi[key])}")
            else:
                try:
                    MatOut_multi[key] = MatOut_multi[key].astype(np.float32)
                    print(f"MatOut_multi['{key}'] dtype after conversion: {MatOut_multi[key].dtype}")
                except Exception as e:
                    print(f"Failed to convert MatOut_multi['{key}'] to float32: {e}")
        
        # Exclude structured arrays like 'MatOut_vel' and 'Nalgo'
        elif key in ['MatOut_vel', 'Nalgo']:
            print(f"MatOut_multi['{key}'] is a structured array, skipping conversion.")
            continue
        
        # For other keys, attempt conversion if they are not object or structured arrays
        else:
            if isinstance(MatOut_multi[key], np.ndarray):
                if MatOut_multi[key].dtype != 'object' and not MatOut_multi[key].dtype.names:
                    try:
                        MatOut_multi[key] = MatOut_multi[key].astype(np.float32)
                        print(f"MatOut_multi['{key}'] dtype after conversion: {MatOut_multi[key].dtype}")
                    except Exception as e:
                        print(f"Failed to convert MatOut_multi['{key}'] to float32: {e}")
                else:
                    print(f"MatOut_multi['{key}'] is a structured or object array, skipping conversion.")
            else:
                print(f"MatOut_multi['{key}'] is not a numpy array, skipping conversion.")
    
    # Get formatting details
    listAlgoName, ListColor, ListMarker, ListShortName = PALA_GetFormat()
    labelSize = 15
    
    # Define other necessary variables
    num_algorithms = len(listAlgoName)
    ListGap_pix = None  # Will be defined later
    vv = None  # Replace with actual data or remove if not needed
    
    # ---------------------------
    # FIG 5: MATOUT target
    # ---------------------------
    fig39 = plt.figure(39, figsize=(10, 8))
    plt.clf()
    fig39.set_size_inches(MatOutTarget.shape[1]/100, MatOutTarget.shape[0]/100)
    fig39.set_dpi(100)
    
    # Compute the square root
    img_sqrt = np.sqrt(MatOutTarget)
    print(f"img_sqrt dtype: {img_sqrt.dtype}")
    
    # Apply Gaussian filter
    img_filtered = gaussian_filter(img_sqrt, sigma=1)
    print(f"img_filtered dtype: {img_filtered.dtype}")
    
    plt.imshow(img_filtered, cmap='hot')
    plt.clim(0, 10)
    plt.axis('image')
    plt.axis('off')
    
    posScale = [60, MatOutTarget.shape[0]-70]
    plt.gca().imshow(np.maximum(MatOutTarget[posScale[1]:posScale[1]+10, posScale[0]:posScale[0]+100], 10), cmap='hot')
    plt.text(posScale[0]+50, posScale[1], '1 mm', color='w', fontsize=labelSize-2, va='bottom', ha='center', backgroundcolor='k')
    
    figname = 'fig5_MatOutTarget'
    plt.savefig(os.path.join(myfilepath_fig, figname + '.png'), dpi=300, bbox_inches='tight')
    
    # Sample the 'hot' colormap
    OutPutColorMap = get_colormap('hot', 256)
    
    # Convert the array to float32 before writing
    write_tif(
        MatIn=np.minimum(img_filtered, 10).astype(np.float32),  # Ensure float32
        OutPutColorMap=OutPutColorMap,                           # Sampled color map as NumPy array
        tif_filename=os.path.join(myfilepath_fig, figname + '.tif'),
        overwrite=True
    )
    
    print(f"TIFF file saved successfully: {os.path.join(myfilepath_fig, figname + '.tif')}")
    
    # ---------------------------
    # FIG 5: MATOUT target colorbar
    # ---------------------------
    fig38 = plt.figure(38, figsize=(2, 8))
    plt.clf()
    cax = plt.axes([0.1, 0.05, 0.8, 0.9])
    plt.colorbar(plt.cm.ScalarMappable(cmap='hot'), cax=cax, ticks=[0, 5, 10])
    cax.set_yticklabels(['0', '5', '10'])
    cax.set_ylabel('Counts', fontsize=labelSize-4)
    plt.axis('off')
    figname_clb = 'fig5_MatOutTarget_clb'
    plt.savefig(os.path.join(myfilepath_fig, figname_clb + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(myfilepath_fig, figname_clb + '.eps'), format='eps', dpi=300)
    
    # ---------------------------
    # FIG 5: MATOUTS zoom horseshoe
    # ---------------------------
    fig40 = plt.figure(40, figsize=(11, 6))
    plt.clf()
    gs = gridspec.GridSpec(2, 3, wspace=0.02, hspace=0.005)
    list2Disp = [1, 2, 3, 6, 7, 8]
    
    xcrop = slice(120, 800)
    ycrop = slice(780, 800)
    figname_horseshoe = 'fig5_MatOut_horseshoe_30dB'
    
    for ii, ialgo in enumerate(list2Disp, 1):
        ax = fig40.add_subplot(gs[ii-1])
        if ialgo == 1:
            img = gaussian_filter(np.sqrt(MatOutTarget[ycrop, xcrop]), sigma=0.001)
            print(f"Algorithm {ialgo} img dtype: {img.dtype}")
            ax.imshow(img, cmap='hot', vmin=0, vmax=10)
            ax.text(3, 3, 'Target MatOut', color='w', fontsize=labelSize-2, va='top', ha='left', backgroundcolor='k')
        else:
            matout_index = ialgo - 2
            if matout_index < len(MatOut_multi['MatOut']):
                subarr = MatOut_multi['MatOut'][matout_index]
                img = gaussian_filter(np.sqrt(subarr[ycrop, xcrop]), sigma=0.001)
                print(f"Algorithm {ialgo} img dtype: {img.dtype}")
                ax.imshow(img, cmap='hot', vmin=0, vmax=10)
                # Conditional label assignment to prevent IndexError
                if (ialgo - 1) < len(listAlgoName):
                    label = listAlgoName[ialgo-1]
                else:
                    label = f'Algorithm {ialgo}'  # Default label if out of range
                ax.text(3, 3, label, color='w', fontsize=labelSize-2, va='top', ha='left', backgroundcolor='k')
            else:
                print(f"Algorithm index {matout_index} out of bounds for 'MatOut_multi['MatOut']'")
                label = f'Algorithm {ialgo}'
                ax.text(3, 3, label, color='w', fontsize=labelSize-2, va='top', ha='left', backgroundcolor='k')
                continue  # Skip to next iteration or handle appropriately
        
        # Add scale bar for first subplot
        if ii == 1:
            ax.imshow(np.maximum(img[-5:-4, 8:58], 10), cmap='hot')
            ax.text(25, -2, '500 µm', color='w', fontsize=labelSize-2, va='bottom', ha='center')
            # Conditional ListShortName access
            if (ialgo - 1) < len(ListShortName):
                short_name = ListShortName[ialgo-1]
            else:
                short_name = f'Algorithm{ialgo}'
            write_tif(
                MatIn=img.astype(np.float32),  # Ensure float32
                OutPutColorMap=OutPutColorMap,  # Reuse the sampled colormap
                tif_filename=os.path.join(myfilepath_fig, figname_horseshoe + ('Target' if ialgo ==1 else short_name) + '.tif'),
                overwrite=True
            )
        else:
            # Conditional ListShortName access
            if (ialgo - 1) < len(ListShortName):
                short_name = ListShortName[ialgo-1]
            else:
                short_name = f'Algorithm{ialgo}'
            write_tif(
                MatIn=img.astype(np.float32),  # Ensure float32
                OutPutColorMap=OutPutColorMap,  # Reuse the sampled colormap
                tif_filename=os.path.join(myfilepath_fig, figname_horseshoe + short_name + '.tif'),
                overwrite=True
            )
        
        ax.axis('image')
        ax.axis('off')
    
    plt.savefig(os.path.join(myfilepath_fig, figname_horseshoe + '.png'), dpi=300, bbox_inches='tight')
    
    # ---------------------------
    # FIG 5: MATOUTS zoom 1
    # ---------------------------
    fig42 = plt.figure(42, figsize=(18, 8))
    plt.clf()
    gs_zoom = gridspec.GridSpec(2, 3, wspace=0.01, hspace=0.01)
    MatZoom = {'x': slice(250, 650), 'z': slice(350, 590)}
    figname_zoom = 'fig5_MatOut_zoom_30dB'
    
    for ii, ialgo in enumerate(list2Disp, 1):
        ax = fig42.add_subplot(gs_zoom[ii-1])
        if ialgo ==1:
            img = gaussian_filter(np.sqrt(MatOutTarget[MatZoom['z'], MatZoom['x']]), sigma=0.5)
            print(f"Algorithm {ialgo} zoom img dtype: {img.dtype}")
            ax.imshow(img, cmap='hot', vmin=0, vmax=10)
            ax.text(5, 5, 'Target MatOut', color='w', fontsize=labelSize, va='top', ha='left', backgroundcolor='k')
        else:
            matout_index = ialgo - 2
            if matout_index < len(MatOut_multi['MatOut']):
                subarr = MatOut_multi['MatOut'][matout_index]
                img = gaussian_filter(np.sqrt(subarr[MatZoom['z'], MatZoom['x']]), sigma=0.001)
                print(f"Algorithm {ialgo} zoom img dtype: {img.dtype}")
                ax.imshow(img, cmap='hot', vmin=0, vmax=10)
                # Conditional label assignment to prevent IndexError
                if (ialgo - 1) < len(listAlgoName):
                    label = listAlgoName[ialgo-1]
                else:
                    label = f'Algorithm {ialgo}'  # Default label if out of range
                ax.text(5, 5, label, color='w', fontsize=labelSize, va='top', ha='left', backgroundcolor='k')
            else:
                print(f"Algorithm index {matout_index} out of bounds for 'MatOut_multi['MatOut']'")
                label = f'Algorithm {ialgo}'
                ax.text(5, 5, label, color='w', fontsize=labelSize, va='top', ha='left', backgroundcolor='k')
                continue  # Skip to next iteration or handle appropriately
        
        # Add scale bar for first subplot
        if ii == 1:
            ax.imshow(np.maximum(img[-5:-2, 10:60], 10), cmap='hot')
            ax.text(25, -2, '500 µm', color='w', fontsize=labelSize-2, va='bottom', ha='center')
            # Conditional ListShortName access
            if (ialgo - 1) < len(ListShortName):
                short_name = ListShortName[ialgo-1]
            else:
                short_name = f'Algorithm{ialgo}'
            write_tif(
                MatIn=img.astype(np.float32),  # Ensure float32
                OutPutColorMap=OutPutColorMap,  # Reuse the sampled colormap
                tif_filename=os.path.join(myfilepath_fig, figname_zoom + ('Target' if ialgo ==1 else short_name) + '.tif'),
                overwrite=True
            )
        else:
            # Conditional ListShortName access
            if (ialgo - 1) < len(ListShortName):
                short_name = ListShortName[ialgo-1]
            else:
                short_name = f'Algorithm{ialgo}'
            write_tif(
                MatIn=img.astype(np.float32),  # Ensure float32
                OutPutColorMap=OutPutColorMap,  # Reuse the sampled colormap
                tif_filename=os.path.join(myfilepath_fig, figname_zoom + short_name + '.tif'),
                overwrite=True
            )
        
        ax.axis('image')
        ax.axis('off')
    
    plt.savefig(os.path.join(myfilepath_fig, figname_zoom + '.png'), dpi=300, bbox_inches='tight')
    
    # ---------------------------
    # FIG 3: Error histograms
    # ---------------------------
    print("Loading error statistics...")
    Stats_multi = loadmat(f'{myfilepath_data}_Stats_multi30dB.mat')
    ErrList = Stats_multi['ErrList']
    
    fig43 = plt.figure(20, figsize=(10, 6))
    plt.clf()
    gs_hist = gridspec.GridSpec(2, 7, wspace=0.09, hspace=0.015)
    xedge = np.linspace(-0.55, 0.55, 200)
    labelSize = 16
    
    # Precision histograms
    for ii in range(len(listAlgoName)):
        ax = fig43.add_subplot(gs_hist[0, ii])
        if ii < ErrList.shape[0]:
            if ErrList[ii].dtype.names and 'x' in ErrList[ii].dtype.names:
                data_x = ErrList[ii]['x'].flatten()
                ax.hist(data_x, bins=xedge, density=True, color=ListColor[ii], alpha=1, edgecolor='none')
                ax.set_xlim([-1, 1])
                ax.set_ylim([0, 0.06])
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.text(0.02, 0.059, f'σ={data_x.std():.2f}', fontsize=labelSize, ha='left', va='top')
                ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
                ax.set_xticklabels(['-$\lambda$/2', '', '0', '', '$\lambda$/2'])
                ax.set_yticklabels([])
                ax.tick_params(axis='both', which='both', direction='in', length=4)
                if ii ==0:
                    ax.set_ylabel('Lateral error', fontsize=labelSize)
            else:
                print(f"ErrList[{ii}] does not have 'x' field or is not a structured array.")
        else:
            print(f"ErrList does not have index {ii}")
    
    # Sensitivity histograms
    for ii in range(len(listAlgoName)):
        ax = fig43.add_subplot(gs_hist[1, ii])
        if ii < ErrList.shape[0]:
            if ErrList[ii].dtype.names and 'z' in ErrList[ii].dtype.names:
                data_z = ErrList[ii]['z'].flatten()
                ax.hist(data_z, bins=xedge, density=True, color=ListColor[ii], alpha=1, edgecolor='none')
                ax.set_xlim([-1, 1])
                ax.set_ylim([0, 0.03])
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.text(0.02, 0.029, f'σ={data_z.std():.2f}', fontsize=labelSize, ha='left', va='top')
                ax.set_xticks([-1, -0.5, 0, 0.5, 1])
                ax.set_xticklabels(['-$\lambda$', '0', '$+\lambda$'])
                ax.set_yticklabels([])
                ax.tick_params(axis='both', which='both', direction='in', length=4)
                if ii ==0:
                    ax.set_ylabel('Axial error', fontsize=labelSize)
            else:
                print(f"ErrList[{ii}] does not have 'z' field or is not a structured array.")
        else:
            print(f"ErrList does not have index {ii}")
    
    plt.suptitle('Error Distributions', fontsize=labelSize+2)
    figname_distrib = 'fig3_distrib_30dB'
    plt.savefig(os.path.join(myfilepath_fig, figname_distrib + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(myfilepath_fig, figname_distrib + '.eps'), format='eps', dpi=300)
    
    # ---------------------------
    # FIG 3: TRUE POSITIVE
    # ---------------------------
    Stats_multi = loadmat(f'{myfilepath_data}_Stats_multi30dB.mat')
    Stat_class = Stats_multi['Stat_class']
    
    TP = Stat_class[2, :, :]
    FN = Stat_class[3, :, :]
    FP = Stat_class[4, :, :]
    
    J_p = TP / (FP + TP) * 100  # precision
    J_r = TP / (FN + TP) * 100  # sensitivity (recall)
    J_ac = TP / (FP + FN + TP) * 100  # Jaccard
    J_p_mean = J_p.mean(axis=1)
    J_r_mean = J_r.mean(axis=1)
    J_ac_mean = J_ac.mean(axis=1)
    TP_sum = TP.sum(axis=1)
    FN_sum = FN.sum(axis=1)
    FP_sum = FP.sum(axis=1)
    
    # FIG 3: TP, FN, FP Barh
    fig50 = plt.figure(50, figsize=(10, 5))
    plt.clf()
    plt.barh(range(1, 8), TP_sum * 1e-3, color=ListColor, label='True Positive')
    plt.barh(range(1, 8), -FN_sum * 1e-3, color=ListColor, alpha=0.5, label='False Negative')
    plt.barh(range(1, 8), -FP_sum * 1e-3, color=ListColor, alpha=0.2, label='False Positive')
    
    for ii in range(len(listAlgoName)):
        plt.text(TP_sum[ii]*1e-3 + 20, ii+1, f'{TP_sum[ii]/1000:.0f}k', va='center', ha='left', fontsize=labelSize)
        plt.text(-FN_sum[ii]*1e-3/2, ii+1, f'{FN_sum[ii]/1000:.0f}k', va='center', ha='center', fontsize=labelSize)
        plt.text(-FP_sum[ii]*1e-3/2 - FN_sum[ii]*1e-3, ii+1, f'{FP_sum[ii]/1000:.0f}k', va='center', ha='center', fontsize=labelSize)
    
    plt.xlim([-700, 500])
    plt.ylim([0.5, 7.5])
    plt.yticks([])
    plt.xlabel('Counts (k)', fontsize=labelSize)
    plt.legend(loc='upper left', frameon=False)
    plt.grid(True, axis='x', linestyle='--', linewidth=0.5)
    
    NbDetectableBulles = (TP_sum + FN_sum).mean()
    plt.axvline(x=NbDetectableBulles/1000, linestyle='--', color='black', linewidth=1, alpha=0.2)
    plt.text(NbDetectableBulles/1000, 0.5, 'Max', ha='center', va='bottom')
    
    figname_TP_FP_FN = 'fig3_TP_FP_FN_30dB'
    plt.savefig(os.path.join(myfilepath_fig, figname_TP_FP_FN + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(myfilepath_fig, figname_TP_FP_FN + '.eps'), format='eps', dpi=300)
    
    # ---------------------------
    # FIG 3: JACCARD
    # ---------------------------
    fig52 = plt.figure(52, figsize=(12, 4))
    plt.clf()
    gs_jaccard = gridspec.GridSpec(1, 3, wspace=0.02)
    
    # Precision
    ax1 = fig52.add_subplot(gs_jaccard[0])
    b_p = ax1.barh(range(1, 8), J_p_mean, color=ListColor)
    ax1.errorbar(J_p_mean, range(1,8), xerr=J_p.std(axis=1), fmt='none', ecolor='k', capsize=10)
    ax1.set_xlim([0, 90])
    ax1.set_ylim([0.5, 7.5])
    ax1.set_yticks([])
    ax1.set_xlabel('Precision (%)', fontsize=labelSize)
    ax1.grid(True, axis='x', linestyle='--', linewidth=0.5)
    ax1.set_title('Precision', fontsize=labelSize)
    
    # Sensitivity
    ax2 = fig52.add_subplot(gs_jaccard[1])
    b_r = ax2.barh(range(1, 8), J_r_mean, color=ListColor)
    ax2.errorbar(J_r_mean, range(1,8), xerr=J_r.std(axis=1), fmt='none', ecolor='k', capsize=10)
    ax2.set_xlim([0, 80])
    ax2.set_ylim([0.5, 7.5])
    ax2.set_yticks([])
    ax2.set_xlabel('Sensitivity (%)', fontsize=labelSize)
    ax2.grid(True, axis='x', linestyle='--', linewidth=0.5)
    ax2.set_title('Sensitivity', fontsize=labelSize)
    
    # Jaccard Index
    ax3 = fig52.add_subplot(gs_jaccard[2])
    b_ac = ax3.barh(range(1, 8), J_ac_mean, color=ListColor)
    ax3.errorbar(J_ac_mean, range(1,8), xerr=J_ac.std(axis=1), fmt='none', ecolor='k', capsize=10)
    ax3.set_xlim([0, 100])
    ax3.set_ylim([0.5, 7.5])
    ax3.set_yticks([])
    ax3.set_xlabel('Jaccard Index (%)', fontsize=labelSize)
    ax3.grid(True, axis='x', linestyle='--', linewidth=0.5)
    ax3.set_title('Jaccard Index', fontsize=labelSize)
    
    plt.tight_layout()
    figname_Jaccard = 'fig3_Jacc_30dB'
    plt.savefig(os.path.join(myfilepath_fig, figname_Jaccard + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(myfilepath_fig, figname_Jaccard + '.eps'), format='eps', dpi=300)
    
    # ---------------------------
    # FIG 2: BRANCHES MATOUT
    # ---------------------------
    print("Processing branches MATOUT...")
    MM = MatOutTarget[780:800, 120:800]
    DistNul = np.zeros((MM.shape[1]//2, 8))
    Gap = np.zeros(MM.shape[1]//2)
    ListGap = np.zeros(8)
    Threshold_p = 0.03
    Threshold_val = Threshold_p * MM.max()
    
    for ialgo in range(1, 9):
        if ialgo < 8:
            if ialgo-1 < len(MatOut_multi['MatOut']):
                MM_algo = MatOut_multi['MatOut'][ialgo-1][780:800, 220:800].astype(np.float32)
                print(f"Processing MatOut_multi['MatOut'][{ialgo-1}]")
            else:
                print(f"No MatOut data for algorithm index {ialgo-1}")
                continue
        else:
            MM_algo = MatOutTarget[780:800, 220:800].astype(np.float32)
            print(f"Processing MatOutTarget for ialgo {ialgo}")
        lz = np.arange(0, MM_algo.shape[1], 2)
        for ind, pos in enumerate(lz):
            Lseq = MM_algo[:, pos]
            Lseq[Lseq < Threshold_val] = 0
            non_zero = np.where(Lseq > Threshold_val)[0]
            if non_zero.size > 0:
                Lseq = Lseq[non_zero[0]:non_zero[-1]+1]
            else:
                Lseq = np.array([])
            DistNul[ind, ialgo-1] = np.sum(Lseq < Threshold_val)
            if ialgo ==8 and Lseq.size > 0:
                im = maximum_filter(Lseq, size=1) == Lseq
                posMax = np.where(im)[0]
                Kepmax = np.insert(np.diff(posMax) !=1, 0, True)
                posMax = posMax[Kepmax]
                val = Lseq[im]
                if len(posMax) >=2:
                    sorted_indices = np.argsort(val)[::-1]
                    Gap[ind] = np.abs(posMax[sorted_indices[0]] - posMax[sorted_indices[1]])
                else:
                    Gap[ind] = 0
        mingap = np.where(DistNul[:, ialgo-1] ==0)[0]
        if mingap.size >0:
            mingap = mingap[-1] +1
        else:
            mingap = 0
        ListGap[ialgo-1] = mingap +1
    
    SR2conv = 1 / 10  # ULM.res = 10
    DistNul_wv = DistNul * SR2conv
    Gap_linear = 0.0  # Placeholder, needs proper calculation based on 'polyfit'
    
    # Define ListGap_pix
    ListGap_pix = ListGap * SR2conv  # Convert gaps to pixels
    
    fig26 = plt.figure(26, figsize=(16, 6))
    plt.clf()
    ax_main = fig26.add_subplot(1,1,1)
    for ialgo in range(7):
        if ialgo < len(listAlgoName):
            ax_main.plot(DistNul_wv[:, ialgo], ListGap[ialgo], marker=ListMarker[ialgo], color=ListColor[ialgo], label=listAlgoName[ialgo])
            ax_main.scatter(ListGap_pix[ialgo], 0, marker='+', color=ListColor[ialgo])
            ax_main.text(ListGap_pix[ialgo], 0, f'{ListGap_pix[ialgo]:.2f}', ha='center', va='top')
        else:
            print(f"Algorithm index {ialgo} out of range.")
    
    ax_main.legend(loc='lower left')  # Changed 'northwest' to 'lower left'
    ax_main.set_xlim([3*SR2conv, 9*SR2conv])
    ax_main.set_ylim([0, 12*SR2conv])
    ax_main.set_xlabel('Simulated canal to canal distance')
    ax_main.set_ylabel('Measured canal to canal distance [λ]')
    ax_main.set_title('Size of the measured gap on MatOut (threshold 0.03, 3%)')
    ax_main.grid(True, linestyle='--', linewidth=0.5)
    
    figname_gap = 'fig2_gap_30dB'
    plt.savefig(os.path.join(myfilepath_fig, figname_gap + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(myfilepath_fig, figname_gap + '.eps'), format='eps', dpi=300)
    
    # ---------------------------
    # FIG 3: RMSE(dB) silico
    # ---------------------------
    val_Err = {}
    RMSE_m = []
    ClutterList = [30]  # Adjust based on your actual clutter list
    
    for ii, clutter in enumerate(ClutterList):
        data_err = loadmat(f'{myfilepath_data}_Stats_multi{abs(clutter)}dB.mat')['ErrList']
        for ialgo in range(len(listAlgoName)):
            if ialgo not in val_Err:
                val_Err[ialgo] = {}
            # Check if 'value' exists and convert
            if 'value' in data_err[ii].dtype.names:
                try:
                    val_Err[ialgo][ii] = data_err[ii]['value'].astype(np.float32)
                    print(f"Converted ErrList for algorithm {ialgo}, clutter {ii} to float32")
                except Exception as e:
                    print(f"Failed to convert ErrList for algorithm {ialgo}, clutter {ii} to float32: {e}")
                    val_Err[ialgo][ii] = data_err[ii]['value']  # Or handle accordingly
            else:
                print(f"No 'value' field for ErrList in algorithm {ialgo}, clutter {ii}")
                val_Err[ialgo][ii] = data_err[ii]['value']  # Or handle accordingly
            if clutter == -30:
                if ialgo < len(listAlgoName):
                    RMSE_m.append(np.mean(val_Err[ialgo][ii]))
                else:
                    print(f"Algorithm index {ialgo} out of range for RMSE calculation")
    
    # Build boxplot
    fig60 = plt.figure(60, figsize=(11.5, 8))
    plt.clf()
    plt.subplot(1,1,1)
    
    # Convert val_Err values to float32 if necessary
    val_Err_converted = {}
    for algo, clutter_dict in val_Err.items():
        val_Err_converted[algo] = {}
        for clutter, values in clutter_dict.items():
            if isinstance(values, np.ndarray):
                try:
                    val_Err_converted[algo][clutter] = values.astype(np.float32)
                    print(f"Converted val_Err_converted[{algo}][{clutter}] to float32")
                except Exception as e:
                    print(f"Failed to convert val_Err_converted[{algo}][{clutter}] to float32: {e}")
                    val_Err_converted[algo][clutter] = values  # Or handle accordingly
            else:
                val_Err_converted[algo][clutter] = values  # Or handle accordingly
    
    # Convert ListColor to NumPy array if not already
    ListColor_array = np.array(ListColor)  # Shape: (num_algorithms, 3)
    
    # Call PALA_boxplot_multi with correct parameters
    PALA_boxplot_multi(
        label_in=listAlgoName,
        val_Err=val_Err_converted,
        labelSize=15,
        InPutColor=ListColor_array,  # Ensure correct shape
        txtsize=10,
        dispVal=False  # Set to False or True based on your preference
    )
    plt.ylim([0, 0.56])
    plt.xticks([])
    plt.xlabel('Algorithms', fontsize=11)
    plt.ylabel('RMSE', fontsize=11)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.15)
    figname_RMSEsilico = 'fig2_RMSEsilico'
    plt.savefig(os.path.join(myfilepath_fig, figname_RMSEsilico + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(myfilepath_fig, figname_RMSEsilico + '.eps'), format='eps', dpi=300)
    
    # ---------------------------
    # 2D plot RMSE SNR
    # ---------------------------
    fig61 = plt.figure(61, figsize=(8, 5))
    plt.clf()
    for ialgo in range(num_algorithms):
        if ialgo < len(RMSE_m):
            plt.plot(
                abs(ClutterList), 
                RMSE_m[ialgo],  # Use RMSE_m directly
                marker=ListMarker[ialgo] if ialgo < len(ListMarker) else 'o', 
                linestyle='.-',
                color=ListColor[ialgo] if ialgo < len(ListColor) else 'k', 
                label=listAlgoName[ialgo] if ialgo < len(listAlgoName) else f'Algorithm {ialgo+1}', 
                linewidth=1.5, 
                markersize=10
            )
        else:
            print(f"No RMSE data available for algorithm index {ialgo}")
    
    plt.ylabel('RMSE')
    plt.xlabel('SNR [dB]')
    plt.ylim([0.06, 0.35])
    plt.gca().invert_xaxis()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc='lower left')
    figname_RMSEsilicoSNR = 'fig2_RMSEsilicoSNR'
    plt.savefig(os.path.join(myfilepath_fig, figname_RMSEsilicoSNR + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(myfilepath_fig, figname_RMSEsilicoSNR + '.eps'), format='eps', dpi=300)
    
    # ---------------------------
    # Save data for Global Score
    # ---------------------------
    Radar = {
        'RMSE': RMSE_m,
        'Jaccard': J_ac_mean,
        'precision': J_p_mean,
        'Gap': ListGap_pix,
        'listName': listAlgoName
    }
    savemat(os.path.join(workingdir, f'{filename}_scores.mat'), {'Radar': Radar, 'Nalgo': num_algorithms})
    
    print('PALA_SilicoFlow_fig.py done.')
