import jax.numpy as np
from scipy import fftpack
from jax.scipy.ndimage import map_coordinates
from dLux.utils import polar2cart, cart2polar, get_pixel_positions
from typing import TypeVar

matrix = TypeVar("matrix")


def rotate(
        image: matrix, 
        rotation: float):
    """
    Rotate an image by some amount.

    Parameters
    ----------
    image: matrix
        The image to rotate.
    rotation: float, radians
        The amount to rotate clockwise from the positive x axis. 

    Returns 
    -------
    image: matrix
        The rotated image. 
    """
    npix = image.shape[0]
    centre = (npix - 1) / 2
    x_pixels, y_pixels = get_pixel_positions(npix)
    rs, phis = cart2polar(x_pixels, y_pixels)
    phis += rotation
    coordinates_rot = np.roll(polar2cart(rs, phis) + centre, 
        shift=1, axis=0)
    rotated = map_coordinates(image, coordinates_rot, order=1)
    return rotated


def conserve_information_and_rotate(
        image: matrix, 
        alpha: float, 
        pad: int = 4) -> matrix:
    """
    A rotation code that conserbes the information in the image. 

    Parameters
    ----------
    image: matrix
        
    """
    # We need to add some extra rows since np.rot90 has a different definition of the centre
    in_shape = image.shape
    image_shape = np.array(in_shape, dtype=int) + 3 
    image = np.full(image_shape, np.nan, dtype=float)\
        .at[1 : in_shape[0] + 1, 1 : in_shape[1] + 1]\
        .set(image)

    # NOTE: Correct until here. 
    # pyplot.imshow(image)
    # pyplot.show()

    # FFT rotation only work in the -45:+45 range
    # So I need to work out how to determine the quadrant that alpha is in and hence the 
    # number of required pi/2 rotations and angle in radians. 
    half_pi_to_1st_quadrant = alpha // (np.pi / 2)
    angle_in_1st_quadrant = - alpha + (half_pi_to_1st_quadrant * np.pi / 2)

    image = np.rot90(image, half_pi_to_1st_quadrant)\
        .at[:-1, :-1]\
        .get()  

    width, height = image.shape
    # Calculate the position that the input array will be in the padded array to simplify
    #  some lines of code later 
    left_corner = int(((pad - 1) / 2.) * width)
    right_corner = int(((pad + 1) / 2.) * width)
    top_corner = int(((pad - 1) / 2.) * height)
    bottom_corner = int(((pad + 1) / 2.) * height)

    # Make the padded array 
    out_shape = (width * pad, height * pad)
    padded_image = np.full(out_shape, np.nan, dtype=float)\
        .at[left_corner : right_corner, top_corner : bottom_corner]\
        .set(image)

    padded_mask = np.ones(out_shape, dtype=bool)\
        .at[left_corner : right_corner, top_corner : bottom_corner]\
        .set(np.where(np.isnan(image), True, False))
    
    # Rotate the mask, to know what part is actually the image
    padded_mask = rotate(padded_mask, -angle_in_1st_quadrant)

    # Replace part outside the image which are NaN by 0, and go into Fourier space.
    padded_image = np.where(np.isnan(padded_image), 0. , padded_image)

    # NOTE: Correct Unitl here. 
    # pyplot.imshow(padded_image)
    # pyplot.show()

    uncentered_angular_displacement = np.tan(angle_in_1st_quadrant / 2.)
    centered_angular_displacement = -np.sin(angle_in_1st_quadrant)

    uncentered_frequencies = np.fft.fftfreq(out_shape[0])
    centered_frequencies = np.arange(-out_shape[0] / 2., out_shape[0] / 2.)

    # a = uncentered_angular_displacement
    # b = centered_angular_displacement
    # N = uncentered_frequencies
    # X = centered_frequencies

    pi_factor = -2.j * np.pi * np.ones(out_shape, dtype=float)

    uncentered_phase = np.exp(
        uncentered_angular_displacement *\
        ((pi_factor * uncentered_frequencies).T *\
        centered_frequencies).T)

    centered_phase = np.exp(
        centered_angular_displacement *\
        (pi_factor * centered_frequencies).T *\
        uncentered_frequencies)

    # NOTE: To be honest the stuff above also looked alright. 
    # I need to double check but it seemed fine. 

    f1 = np.fft.ifft(
        (np.fft.fft(padded_image, axis=0).T * uncentered_phase).T, axis=0)
    
    f2 = np.fft.ifft(
        np.fft.fft(f1, axis=1) * centered_phase, axis=1)

    rotated_image = np.fft.ifft(
        (np.fft.fft(f2, axis=0).T * uncentered_phase).T, axis=0)\
        .at[padded_mask]\
        .set(np.nan)
    
    return np.real(rotated_image\
        .at[left_corner + 1 : right_corner - 1, 
            top_corner + 1 : bottom_corner - 1]\
        .get()).copy()

shape = (100, 100)
quadrant_1 = np.full(shape, 1., dtype=float)
quadrant_2 = np.full(shape, 2., dtype=float)
quadrant_3 = np.full(shape, 3., dtype=float)
quadrant_4 = np.full(shape, 4., dtype=float)

top = np.vstack([quadrant_2, quadrant_1])
bottom = np.vstack([quadrant_3, quadrant_4])
image = np.hstack([top, bottom])

import matplotlib.pyplot as pyplot

pyplot.imshow(image)
pyplot.show()

image = conserve_information_and_rotate(image, np.pi / 4)

print("Shape After:", image.shape)
pyplot.imshow(image)
pyplot.show()
