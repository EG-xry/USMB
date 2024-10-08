# PALA_GetFormat.py

import numpy as np

def PALA_GetFormat():
    """
    Returns algorithm names, colors, markers, and short names for PALA_SilicoFlow_fig.

    Returns:
        ListAlgoName (list of str): Names of algorithms.
        ListColor (numpy.ndarray): RGB colors for each algorithm.
        ListMarker (list of str): Marker symbols for each algorithm.
        ListShortName (list of str): Short names for algorithms.
    """
    # Define the full list of algorithm names and their short names
    ListAlgoName = ['No Shift', 'WA', 'Cub-Interp', 'Lz-Interp', 'Sp-Interp', 'Gauss-Fit', 'RS']
    ListShortName = ['NS', 'WA', 'CI', 'LI', 'SI', 'GF', 'RS']

    # Define the initial list of colors (RGB values normalized between 0 and 1)
    ListColor = np.array([
        [0.9047, 0.1918, 0.1988],
        [0.2941, 0.5447, 0.7494],
        [0.3718, 0.7176, 0.3612],
        [1.0000, 0.5482, 0.1000],
        [0.8650, 0.8110, 0.4330],
        [0.6859, 0.4035, 0.2412],
        [0.9718, 0.5553, 0.7741],
        [0.6400, 0.6400, 0.6400]
    ])

    # Remove the 7th color (index 6) to match the number of algorithms
    ListColor = np.delete(ListColor, 6, axis=0)

    # Swap the first and the last colors in the updated ListColor
    ListColor[[0, -1], :] = ListColor[[-1, 0], :]

    # Define the list of markers for each algorithm
    ListMarker = ['.', 'o', 's', '^', 'd', 'p', 'h']

    return ListAlgoName, ListColor, ListMarker, ListShortName

# Example usage:
if __name__ == "__main__":
    algo_names, colors, markers, short_names = PALA_GetFormat()
    print("Algorithm Names:", algo_names)
    print("Colors:\n", colors)
    print("Markers:", markers)
    print("Short Names:", short_names)
