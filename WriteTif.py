# WriteTif.py

import os
import numpy as np
from matplotlib import colormaps
import tifffile
from tifffile import TiffWriter
import warnings

def write_tif(MatIn, OutPutColorMap, tif_filename, overwrite=False, caxis=None, voxel_size_mm=None):
    """
    Writes a 2D, 3D, or 4D NumPy array to a TIFF file with optional color mapping, scaling, and voxel size metadata.

    Parameters:
        MatIn (numpy.ndarray): Input matrix.
            - If OutPutColorMap is provided:
                - 2D array: Single image.
                - 3D array: Multiple frames (slices along the third axis).
            - If OutPutColorMap is None:
                - 3D array with shape (height, width, 3): Single RGB image.
                - 4D array with shape (height, width, frames, 3): Multiple RGB frames.
        OutPutColorMap (numpy.ndarray or None): Colormap to apply.
            - Should be an (N, 3) array where N is the number of colors.
            - If None, MatIn is assumed to be RGB.
        tif_filename (str): Path to the output TIFF file. Must end with '.tif'.
        overwrite (bool, optional): If True, overwrite the file if it exists. Default is False.
        caxis (float or list/tuple of two floats, optional): 
            - If float, scales the color axis by this factor (multiplies [0, max(MatIn)] by the factor).
            - If list or tuple of two floats, sets the color axis to [min, max].
            - If None, defaults to [0, max(MatIn)].
        voxel_size_mm (float or list/tuple of two floats, optional):
            - Specifies the voxel size in millimeters.
            - If float, applies the same size to both X and Y axes.
            - If list or tuple, specifies [X_size_mm, Y_size_mm].
            - If None, resolution tags are not set.

    Raises:
        ValueError: If tif_filename does not end with '.tif' or if input dimensions are incorrect.
        FileExistsError: If the file exists and overwrite is False.
    """

    # Debugging: Print the shape of MatIn
    print(f"Input MatIn shape: {MatIn.shape}")

    # Validate file extension
    if not tif_filename.lower().endswith('.tif'):
        raise ValueError("tif_filename should end with '.tif'")

    # Handle overwriting
    if os.path.exists(tif_filename):
        if overwrite:
            os.remove(tif_filename)
            print(f"Overwriting existing file: {tif_filename}")
        else:
            raise FileExistsError(f"The file '{tif_filename}' already exists. Set overwrite=True to overwrite it.")

    # Handle caxis
    if caxis is None:
        c0 = [0, MatIn.max()]
    elif isinstance(caxis, (int, float)):
        c0 = [0, MatIn.max() * caxis]
    elif isinstance(caxis, (list, tuple, np.ndarray)) and len(caxis) == 2:
        c0 = [caxis[0], caxis[1]]
    else:
        raise ValueError("caxis must be None, a float, or a list/tuple of two floats.")

    # Initialize frames list
    frames = []

    if OutPutColorMap is not None:
        if MatIn.ndim not in [2, 3]:
            raise ValueError("MatIn should be a 2D or 3D array when OutPutColorMap is provided.")

        # Clip the input matrix
        MatIn_sat = np.clip(MatIn, c0[0], c0[1])

        # Normalize
        MatIn_norm = (MatIn_sat - c0[0]) / (c0[1] - c0[0])

        # Scale to colormap indices
        num_colors = OutPutColorMap.shape[0]
        MatIn_scaled = np.round(MatIn_norm * (num_colors - 1)).astype(np.int32)
        MatIn_scaled = np.clip(MatIn_scaled, 0, num_colors - 1)

        # Map indices to RGB
        if MatIn.ndim == 2:
            RGB = OutPutColorMap[MatIn_scaled]
            # Convert to uint8
            RGB = (RGB * 255).astype(np.uint8)
            # Ensure RGB has shape (height, width, 3)
            RGB = RGB[:, :, :3]
            # Expand to (height, width, 3)
            frames.append(RGB)
            print(f"Processed 2D MatIn with colormap into 1 RGB frame.")
        else:
            # MatIn is 3D: (height, width, frames)
            num_frames = MatIn_scaled.shape[2]
            for i in range(num_frames):
                frame = OutPutColorMap[MatIn_scaled[:, :, i]]
                frame = (frame * 255).astype(np.uint8)
                frame = frame[:, :, :3]  # Ensure RGB
                frames.append(frame)
            print(f"Processed 3D MatIn with colormap into {num_frames} RGB frames.")
    else:
        # Assume MatIn is already RGB
        if MatIn.ndim == 3:
            if MatIn.shape[2] == 3:
                # Single RGB image
                frame = MatIn
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                frames.append(frame)
                print(f"Processed 3D MatIn as a single RGB frame.")
            else:
                # Unsupported shape for 3D array without colormap
                raise ValueError("MatIn has an unsupported shape. For multiple RGB frames, provide a 4D array with shape (height, width, frames, 3).")
        elif MatIn.ndim == 4:
            if MatIn.shape[3] !=3:
                raise ValueError("MatIn's last dimension must be 3 for RGB data.")
            num_frames = MatIn.shape[2]
            for i in range(num_frames):
                frame = MatIn[:, :, i, :]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                frames.append(frame)
            print(f"Processed 4D MatIn into {num_frames} RGB frames.")
        else:
            raise ValueError("MatIn should be a 2D, 3D, or 4D array when OutPutColorMap is None.")

    # Write to TIFF using 'write' method
    with TiffWriter(tif_filename, bigtiff=True) as tif:
        for idx, frame in enumerate(frames):
            tif.write(
                frame,
                photometric='rgb',
                compression='lzw',
                description='ArthurChavignon'
            )
            print(f"Wrote frame {idx+1}/{len(frames)} to TIFF.")

    # Handle voxel size metadata
    if voxel_size_mm is not None:
        with tifffile.TiffFile(tif_filename, mode='r+') as tif_file:
            for page_idx, page in enumerate(tif_file.pages):
                # Set XResolution and YResolution in pixels per mm
                if isinstance(voxel_size_mm, (int, float)):
                    voxel_size_mm_list = [voxel_size_mm, voxel_size_mm]
                elif isinstance(voxel_size_mm, (list, tuple, np.ndarray)) and len(voxel_size_mm) == 2:
                    voxel_size_mm_list = list(voxel_size_mm)
                else:
                    raise ValueError("voxel_size_mm must be a float or a list/tuple of two floats.")

                x_res = 1 / voxel_size_mm_list[0]  # pixels per mm
                y_res = 1 / voxel_size_mm_list[1]  # pixels per mm

                # Update tags
                page.tags['XResolution'].value = (x_res, 1)
                page.tags['YResolution'].value = (y_res, 1)
                page.tags['ResolutionUnit'].value = 3  # 3 = millimeter
                print(f"Set voxel size metadata for page {page_idx+1}.")

    print(f"TIFF file saved successfully: {tif_filename}")

# Example Usage
if __name__ == "__main__":
    import numpy as np
    from matplotlib import colormaps

    # Example 2D MatIn with colormap
    MatIn_2D = np.random.rand(100, 100) * 20  # Example data ranging from 0 to 20

    # Example colormap (hot with 256 colors)
    # Correctly sample the colormap without passing lut
    cmap = colormaps.get_cmap('hot')  # Get the 'hot' colormap without specifying lut
    OutPutColorMap = cmap(np.linspace(0, 1, 256))[:, :3]  # Shape: (256, 3)

    # Save as TIFF
    write_tif(
        MatIn=MatIn_2D,
        OutPutColorMap=OutPutColorMap,
        tif_filename='example_output_2D.tif',
        overwrite=True,
        caxis=[0, 10],          # Optional: set color axis to [0, 10]
        voxel_size_mm=0.5        # Optional: set voxel size to 0.5 mm
    )

    # Example 3D MatIn with colormap (multiple frames)
    MatIn_3D = np.random.rand(100, 100, 5) * 20  # 5 frames

    # Save as TIFF with colormap
    write_tif(
        MatIn=MatIn_3D,
        OutPutColorMap=OutPutColorMap,
        tif_filename='example_output_3D.tif',
        overwrite=True,
        caxis=[0, 10],          # Optional: set color axis to [0, 10]
        voxel_size_mm=[0.5, 0.5] # Optional: set voxel size to [0.5 mm, 0.5 mm]
    )

    # Example RGB MatIn without colormap
    MatIn_RGB = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Save as TIFF without colormap
    write_tif(
        MatIn=MatIn_RGB,
        OutPutColorMap=None,
        tif_filename='example_output_RGB.tif',
        overwrite=True,
        voxel_size_mm=[0.5, 0.5] # Optional: set voxel size to [0.5 mm, 0.5 mm]
    )

    # Example 4D RGB MatIn (multiple RGB frames)
    MatIn_4D_RGB = np.random.randint(0, 256, (100, 100, 5, 3), dtype=np.uint8)

    # Save as TIFF without colormap
    write_tif(
        MatIn=MatIn_4D_RGB,
        OutPutColorMap=None,
        tif_filename='example_output_4D_RGB.tif',
        overwrite=True,
        voxel_size_mm=[0.5, 0.5] # Optional: set voxel size to [0.5 mm, 0.5 mm]
    )
