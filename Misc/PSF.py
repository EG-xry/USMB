import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
import os

def create_circular_object(image_size, radius):
    """
    Creates a binary image with a filled circle.

    Parameters:
    - image_size: Tuple of (height, width)
    - radius: Radius of the circle

    Returns:
    - 2D NumPy array with a circular object
    """
    Y, X = np.ogrid[:image_size[0], :image_size[1]]
    center = (image_size[0] / 2, image_size[1] / 2)
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    mask = dist_from_center <= radius
    image = np.zeros(image_size)
    image[mask] = 1
    return image

def create_gaussian_psf(psf_size, sigma):
    """
    Creates a Gaussian Point Spread Function (PSF).

    Parameters:
    - psf_size: Tuple of (height, width)
    - sigma: Standard deviation of the Gaussian

    Returns:
    - 2D NumPy array representing the PSF
    """
    ax = np.linspace(-(psf_size[1] // 2), psf_size[1] // 2, psf_size[1])
    ay = np.linspace(-(psf_size[0] // 2), psf_size[0] // 2, psf_size[0])
    X, Y = np.meshgrid(ax, ay)
    kernel = np.exp(-(X**2 + Y**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def main():
    # Parameters
    image_size = (256, 256)  # Size of the image
    circle_radius = 50       # Radius of the circular object
    psf_size = (51, 51)      # Size of the PSF kernel
    psf_sigma = 5            # Standard deviation for Gaussian PSF
    output_dir = "output_images"  # Directory to save images

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the circular object
    object_image = create_circular_object(image_size, circle_radius)

    # Create the PSF
    psf = create_gaussian_psf(psf_size, psf_sigma)

    # Convolve the object with the PSF using FFT for efficiency
    blurred_image = fftconvolve(object_image, psf, mode='same')

    # Alternatively, you can use a Gaussian filter directly
    # blurred_image = gaussian_filter(object_image, sigma=psf_sigma)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original Object
    axes[0].imshow(object_image, cmap='gray')
    axes[0].set_title('Original Circular Object')
    axes[0].axis('off')

    # PSF
    axes[1].imshow(psf, cmap='gray')
    axes[1].set_title('Point Spread Function (PSF)')
    axes[1].axis('off')

    # Blurred Image
    axes[2].imshow(blurred_image, cmap='gray')
    axes[2].set_title('Blurred Image (Object * PSF)')
    axes[2].axis('off')

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, "circular_object_psf.png")
    plt.savefig(output_path)
    print(f"Image saved to {output_path}")

    plt.show()

if __name__ == "__main__":
        main()
