# PALA_multiULM_mesh.py

import numpy as np

def PALA_multiULM_mesh(IQ_in, listAlgo, ULM, PData):
    """
    Performs ULM on a block of IQ data for all ULM algorithms in listAlgo and stores results.

    Parameters:
    IQ_in: numpy array
        Block of image data where bubbles are located in the center.
    listAlgo: list
        List of algorithms to be compared.
    ULM: dict
        Dictionary with various ULM parameters.
    PData: dict
        Dictionary with pixel information of IQ images (including PDelta and Origin).

    Returns:
    MatTrackingTot: numpy array
        Scatter position [z, x] for each frame and each algorithm.
        Dim 1: position [z, x]
        Dim 2: algorithm [1:len(listAlgo)]
        Dim 3: frame
    """

    MatTrackingTot = []

    for ialgo in range(len(listAlgo)):
        algo = listAlgo[ialgo].lower()

        # Determine the localization method based on the algorithm
        if algo in ['wa', 'weighted_average']:
            ULM['LocMethod'] = 'WA'
        elif algo in ['radial', 'radial_vivo', 'radial_silicio']:
            ULM['LocMethod'] = 'Radial'
        elif algo in ['interp_cubic', 'interp_cubic_silicio']:
            ULM['LocMethod'] = 'Interp'
            ULM['parameters']['InterpMethod'] = 'cubic'
        elif algo in ['interp_lanczos', 'interp_lanczos_silicio']:
            ULM['LocMethod'] = 'Interp'
            ULM['parameters']['InterpMethod'] = 'lanczos3'
        elif algo in ['interp_spline', 'interp_spline_silicio']:
            ULM['LocMethod'] = 'Interp'
            ULM['parameters']['InterpMethod'] = 'spline'
        elif algo == 'gaussian_fit':
            ULM['LocMethod'] = 'CurveFitting'
        elif algo in ['interp_bilinear', 'interp_bilinear_silicio', 'no_localization', 'no_shift']:
            ULM['LocMethod'] = 'NoLocalization'
        else:
            raise ValueError(f"Wrong method selected: {algo}")

        # Call the ULM localization function (this should be implemented elsewhere)
        MatTracking = ULM_localization2D_mesh(IQ_in, ULM)

        # Correctly extract PDelta and Origin values
        # PData['PDelta'] and PData['Origin'] are structured as [[array([[x, y, z]])]], shape (1, 1)
        # Extract the inner array first
        PDelta_array = PData['PDelta'][0][0]  # shape: (1, 3)
        Origin_array = PData['Origin'][0][0]  # shape: (1, 3)

        # Flatten to 1D arrays
        PDelta_values = PDelta_array.flatten()  # [PDelta_x, PDelta_y, PDelta_z]
        Origin_values = Origin_array.flatten()  # [Origin_x, Origin_y, Origin_z]

        # Extract PDelta_z and PDelta_x
        PDelta_z = PDelta_values[2]  # Third element
        PDelta_x = PDelta_values[0]  # First element

        # Extract Origin_z and Origin_x
        Origin_z = Origin_values[2]  # Third element
        Origin_x = Origin_values[0]  # First element

        # Convert MatTracking to physical coordinates
        # Assuming MatTracking has shape (N, 2) where N is the number of detections
        MatTracking = (MatTracking - np.array([1, 1])) * np.array([PDelta_z, PDelta_x]) \
                      + np.array([Origin_z, Origin_x])

        # Store the result
        MatTrackingTot.append(MatTracking)

    print('Localization performed')

    # Concatenate and permute the results
    # Convert the list of arrays into a 3D numpy array
    # Each element in MatTrackingTot is of shape (N, 2), where N is number of detections
    # Stack along the third dimension (algorithm)
    MatTrackingTot = np.stack(MatTrackingTot, axis=2)  # Shape: (N, 2, numAlgo)

    # Permute to [position, algo, frame]
    # Here, frame dimension is ambiguous. Assuming 'frame' corresponds to algorithm
    MatTrackingTot = np.transpose(MatTrackingTot, (1, 2, 0))  # Shape: (2, numAlgo, N)

    return MatTrackingTot

# Placeholder function for ULM_localization2D_mesh (needs actual implementation)
def ULM_localization2D_mesh(IQ_in, ULM):
    """
    Placeholder for the actual ULM localization function.
    Replace this with the real implementation.
    """
    # For demonstration, it returns random values simulating localization
    # Assume it returns an array of shape (number_of_detections, 2)
    # where each row is [z, x] in physical coordinates
    number_of_detections = 100  # Example value; replace with actual detections
    return np.random.rand(number_of_detections, 2)

# Example usage (for testing purposes)
if __name__ == "__main__":
    IQ_in = np.random.rand(100, 100)  # Replace with actual IQ data
    listAlgo = ['wa', 'radial', 'interp_cubic']
    ULM = {
        'LocMethod': '',
        'parameters': {}
    }
    PData = {
        'PDelta': [[np.array([[1, 0, 1]], dtype=np.uint8)]],  # Example values
        'Origin': [[np.array([[0, 0, 0]], dtype=float)]]      # Example values
    }

    MatTrackingTot = PALA_multiULM_mesh(IQ_in, listAlgo, ULM, PData)
    print(MatTrackingTot.shape)  # Expected shape: (2, numAlgo, N)
