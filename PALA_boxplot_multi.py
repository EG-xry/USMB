# PALA_boxplot_multi.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def PALA_boxplot_multi(label_in, val_Err, labelSize=12, InPutColor=None, txtsize=10, dispVal=False):
    """
    Creates a customized multi-box plot with optional annotations.
    
    Parameters:
        label_in (list of str): Labels for the x-axis categories.
        val_Err (list of lists or numpy.ndarray): 
            - If list of lists: val_Err[i][j] contains data for category i, box j.
            - If numpy.ndarray:
                - 3D array: shape (categories, boxes, data_points)
                - 4D array: shape (categories, boxes, frames, data_points) for RGB or similar.
        labelSize (int, optional): Font size for axis labels. Default is 12.
        InPutColor (numpy.ndarray, optional): 
            - Array of shape (num_categories, 3) representing RGB colors for each category.
            - If None, default colors are used.
        txtsize (int, optional): Font size for text annotations. Default is 10.
        dispVal (bool, optional): If True, display mean ± std annotations. Default is False.
    
    Returns:
        val (numpy.ndarray): Concatenated mean and std values. Shape depends on input.
    """
    
    # Validate InPutColor
    if InPutColor is None:
        # Generate default colors using a colormap
        cmap = plt.get_cmap('tab10')
        InPutColor = cmap(np.linspace(0, 1, len(label_in)))[:,:3]  # Shape: (num_categories, 3)
    else:
        InPutColor = np.array(InPutColor)
        print(f"Original InPutColor shape: {InPutColor.shape}")
        if InPutColor.shape[0] == 3 and InPutColor.shape[1] == len(label_in):
            InPutColor = InPutColor.T  # Transpose to (num_categories, 3)
            print(f"Transposed InPutColor shape: {InPutColor.shape}")
        elif InPutColor.shape[1] == 4:
            # Remove alpha channel
            InPutColor = InPutColor[:,:3]
            print(f"Removed alpha channel. New InPutColor shape: {InPutColor.shape}")
        elif InPutColor.shape[1] !=3:
            raise ValueError("InPutColor should have shape (num_categories, 3) representing RGB colors.")
        print(f"Adjusted InPutColor shape: {InPutColor.shape}")
    
    # Compute statistics
    if isinstance(val_Err, list):
        num_categories = len(val_Err)
        num_boxes = max(len(boxes) for boxes in val_Err)
        
        # Initialize statistics arrays
        val_mean = np.zeros((num_categories, num_boxes))
        val_std = np.zeros((num_categories, num_boxes))
        val_med = np.zeros((num_categories, num_boxes))
        val_d1 = np.zeros((num_categories, num_boxes))
        val_d9 = np.zeros((num_categories, num_boxes))
        val_q1 = np.zeros((num_categories, num_boxes))
        val_q3 = np.zeros((num_categories, num_boxes))
        
        for i in range(num_categories):
            for j in range(len(val_Err[i])):
                data = np.array(val_Err[i][j])
                val_med[i, j] = np.median(data)
                val_mean[i, j] = np.mean(data)
                val_std[i, j] = np.std(data)
                val_d1[i, j] = np.quantile(data, 0.05)
                val_d9[i, j] = np.quantile(data, 0.95)
                val_q1[i, j] = np.quantile(data, 0.25)
                val_q3[i, j] = np.quantile(data, 0.75)
                
    elif isinstance(val_Err, np.ndarray):
        if val_Err.ndim == 3:
            # 3D array: (categories, boxes, data_points)
            val_mean = np.mean(val_Err, axis=2)
            val_std = np.std(val_Err, axis=2)
            val_med = np.median(val_Err, axis=2)
            val_d1 = np.quantile(val_Err, 0.05, axis=2)
            val_d9 = np.quantile(val_Err, 0.95, axis=2)
            val_q1 = np.quantile(val_Err, 0.25, axis=2)
            val_q3 = np.quantile(val_Err, 0.75, axis=2)
        elif val_Err.ndim == 4:
            # 4D array: (categories, boxes, frames, data_points)
            # Reshape to (categories, boxes, frames * data_points)
            categories, boxes, frames, data_points = val_Err.shape
            print(f"Reshaping val_Err from shape {val_Err.shape} to ({categories}, {boxes}, {frames * data_points})")
            val_Err = val_Err.reshape(categories, boxes, frames * data_points)
            print(f"New val_Err shape: {val_Err.shape}")
            val_mean = np.mean(val_Err, axis=2)
            val_std = np.std(val_Err, axis=2)
            val_med = np.median(val_Err, axis=2)
            val_d1 = np.quantile(val_Err, 0.05, axis=2)
            val_d9 = np.quantile(val_Err, 0.95, axis=2)
            val_q1 = np.quantile(val_Err, 0.25, axis=2)
            val_q3 = np.quantile(val_Err, 0.75, axis=2)
        else:
            raise ValueError("val_Err should be a list of lists or a 3D/4D numpy array.")
    else:
        raise ValueError("val_Err should be a list of lists or a numpy array.")
    
    # Debugging: print shapes
    print(f"val_q1 shape: {val_q1.shape}")
    print(f"val_q3 shape: {val_q3.shape}")
    print(f"val_med shape: {val_med.shape}")
    print(f"val_mean shape: {val_mean.shape}")
    print(f"val_std shape: {val_std.shape}")
    
    # Concatenate mean and std for return value
    val = np.concatenate((val_mean, val_std), axis=1)
    
    # Displaying
    fig, ax = plt.subplots(figsize=(10, 6))
    
    num_categories = len(label_in)
    if isinstance(val_Err, list):
        num_boxes = max(len(boxes) for boxes in val_Err)
    else:
        num_boxes = val_mean.shape[1]
    
    box_width = 0.94 / num_boxes
    spacing = box_width / 2 * (num_boxes - 1)
    posX = np.linspace(-spacing, spacing, num_boxes)
    
    for i in range(num_categories):
        for j in range(num_boxes):
            if isinstance(val_Err, list) and j >= len(val_Err[i]):
                continue  # Skip if this category has fewer boxes
            # Define position
            x = i + 1 + posX[j]
            
            # Define color
            color = InPutColor[i] * 0.8
            color = color * (1 + 0.5 * (j) / num_boxes)
            color = np.clip(color, 0, 1)
            color = tuple(color)  # Ensure it's a tuple
            
            # Define rectangle parameters
            lower = val_q1[i, j]
            height = val_q3[i, j] - val_q1[i, j]
            
            # Debugging: Print the parameters
            print(f"Category {i}, Box {j}: x={x}, lower={lower}, height={height}, color={color}")
            
            # Ensure lower and height are scalars
            if not np.isscalar(lower) or not np.isscalar(height):
                raise ValueError(f"Non-scalar value encountered: lower={lower}, height={height}")
            
            # Ensure color has exactly 3 elements
            if len(color) != 3:
                raise ValueError(f"Color must have 3 elements, got {len(color)}: {color}")
            
            rect = Rectangle((x - box_width / 2, lower), box_width, height,
                             linewidth=0.5, edgecolor='k', facecolor=color)
            ax.add_patch(rect)
            
            # Plot median
            ax.plot([x - box_width / 2, x + box_width / 2], 
                    [val_med[i, j], val_med[i, j]], 'k-', linewidth=1)
            
            # Plot mean
            ax.plot(x, val_mean[i, j], 'ko', markersize=4, markerfacecolor='k')
            
            # Plot whiskers
            ax.plot([x, x], [val_d1[i, j], val_d9[i, j]], 'k.-', linewidth=1)
            
            # Optional: Display mean ± std
            if dispVal:
                text = f"{val_mean[i, j]:.2f}±{val_std[i, j]:.2f}"
                ax.text(x + box_width / 2 - 0.1, val_d9[i, j] + 0.02, text,
                        fontsize=txtsize, ha='left', va='bottom', rotation=90, color='k')
    
    # Customize plot
    ax.set_xlim(0.5, num_categories + 0.5)
    ax.set_xticks(range(1, num_categories + 1))
    ax.set_xticklabels(label_in, fontsize=labelSize)
    ax.tick_params(axis='both', which='both', colors='k', labelsize=labelSize)
    ax.set_ylabel('Values', fontsize=labelSize)
    ax.set_xlabel('Categories', fontsize=labelSize)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return val

if __name__ == "__main__":
    # Example Usage
    import numpy as np
    from matplotlib import colormaps

    # Example 1: 2D Data with Colormap (list of lists)
    label_in = ['Category 1', 'Category 2', 'Category 3']
    val_Err = [
        [np.random.normal(5, 1, 100), np.random.normal(6, 1.2, 100)],
        [np.random.normal(4, 0.8, 100), np.random.normal(5.5, 1.1, 100)],
        [np.random.normal(6, 1.5, 100), np.random.normal(7, 1.3, 100)]
    ]
    InPutColor = [
        [1, 0, 0],  # Red for Category 1
        [0, 1, 0],  # Green for Category 2
        [0, 0, 1]   # Blue for Category 3
    ]
    
    PALA_boxplot_multi(
        label_in=label_in,
        val_Err=val_Err,
        labelSize=12,
        InPutColor=InPutColor,
        txtsize=10,
        dispVal=True
    )
    
    # Example 2: 3D Data with Numpy Array
    label_in = ['A', 'B']
    val_Err = np.random.rand(2, 3, 100) * 10  # 2 categories, 3 boxes each, 100 data points
    
    InPutColor = np.array([
        [1, 0, 0],  # Red for Category A
        [0, 1, 0]   # Green for Category B
    ])
    
    PALA_boxplot_multi(
        label_in=label_in,
        val_Err=val_Err,
        labelSize=14,
        InPutColor=InPutColor,
        txtsize=12,
        dispVal=True
    )
    
    # Example 3: 4D RGB Data (Not typical for box plots, but included for completeness)
    label_in = ['X', 'Y']
    val_Err = np.random.rand(2, 2, 4, 100) * 10  # 2 categories, 2 boxes each, 4 frames, 100 data points
    
    InPutColor = np.array([
        [1, 0, 0],  # Red for Category X
        [0, 0, 1]   # Blue for Category Y
    ])
    
    PALA_boxplot_multi(
        label_in=label_in,
        val_Err=val_Err,
        labelSize=12,
        InPutColor=InPutColor,
        txtsize=10,
        dispVal=True
    )
