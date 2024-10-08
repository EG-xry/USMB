import numpy as np
from scipy.ndimage import gaussian_filter

def generate_wgn(shape, power, impedance):
    """
    Generate white Gaussian noise (WGN) given the power and impedance.

    Parameters:
    shape: tuple
        Shape of the noise array.
    power: float
        Power in dBW.
    impedance: float
        Impedance in ohms.

    Returns:
    noise: numpy array
        Generated white Gaussian noise.
    """
    # Generate white Gaussian noise with a standard deviation based on the power and impedance
    std_dev = np.sqrt(10 ** (power / 10) / impedance)
    return std_dev * np.random.randn(*shape)

def PALA_AddNoiseInIQ(IQ, NoiseParam):
    """
    Adds noise to IQ data simulating clutter noise based on NoiseParam.

    Parameters:
    IQ: numpy array
        Raw IQ data (2D or 3D array).
    NoiseParam: dict
        Dictionary containing noise parameters:
        - 'Power': Power in dBW.
        - 'Impedance': Impedance in ohms.
        - 'SigmaGauss': Standard deviation for Gaussian filtering.
        - 'clutterdB': Clutter level in dB.
        - 'amplCullerdB': Amplitude of clutter in dB.

    Returns:
    IQ_speckle: numpy array
        IQ data with added noise.
    """

    # Take the absolute value of the IQ data
    IQ = np.abs(IQ)

    # Create noise component using Gaussian white noise
    noise = generate_wgn(IQ.shape, NoiseParam['Power'], NoiseParam['Impedance'])

    # Scale the noise
    noise_scaled = noise * np.max(IQ) * 10 ** ((NoiseParam['amplCullerdB'] + NoiseParam['clutterdB']) / 20)

    # Add clutter to the noise
    clutter = np.max(IQ) * 10 ** (NoiseParam['clutterdB'] / 20)
    noisy_IQ = IQ + noise_scaled + clutter

    # Apply Gaussian filter
    IQ_speckle = gaussian_filter(noisy_IQ, sigma=NoiseParam['SigmaGauss'])

    return IQ_speckle

# Example usage
NoiseParam = {
    'Power': -2,
    'Impedance': 0.2,
    'SigmaGauss': 1.5,
    'clutterdB': -20,
    'amplCullerdB': 10
}

# Assume IQ is a 2D or 3D numpy array
IQ = np.random.rand(100, 100)  # Replace with actual IQ data
IQ_speckle = PALA_AddNoiseInIQ(IQ, NoiseParam)
