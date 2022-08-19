import jax.numpy as np
from scipy import fftpack
from jax.scipy.ndimage import map_coordinates
from dLux.utils import polar2cart, cart2polar, get_pixel_positions
from typing import TypeVar

matrix = TypeVar("matrix")


def rotate(
        image: matrix, 
        rotation: float):
    npix = image.shape[0]
    centre = (npix - 1) / 2
    x_pixels, y_pixels = get_pixel_positions(npix)
    rs, phis = cart2polar(x_pixels, y_pixels)
    phis += rotation
    coordinates_rot = np.roll(polar2cart(rs, phis) + centre, 
        shift=1, axis=0)
    rotated = map_coordinates(array, coordinates_rot, order=1)
    return rotated


def conserve_information_and_rotate(
        image: matrix, 
        alpha: float, 
        pad: int = 4) -> matrix:
	# We need to add some extra rows since np.rot90 has a different definition of the centre
    in_shape = image.shape
    image_shape = map(lambda x: x + 3, in_shape)
	image = np.full(shape, np.nan, dtype=float)\
        .at[1 : in_shape[0] + 1, 1 : in_shape[1] + 1]\
        .set(image)

	# FFT rotation only work in the -45:+45 range
    # So I need to work out how to determine the quadrant that alpha is in and hence the 
    # number of required pi/2 rotations and angle in radians. 
    half_pi_to_1st_quadrant = (alpha + np.pi / 4) // (np.pi / 2)
    angle_in_first_quadrant = angle - (half_pi_to_1st_quadrant * np.pi / 2)
    
    image = np.rot90(image, half_pi_to_1st_quadrant)\
        .at[:-1, :-1]\
        .get()	

    width, height = np.array(image.shape, dtype=int)
	# Calculate the position that the input array will be in the padded array to simplify
	#  some lines of code later 
    # NOTE: This is the location to which I have reached. 
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
	padded_mask = rotate(padded_mask, -angle_in_first_quadrant)

	# Replace part outside the image which are NaN by 0, and go into Fourier space.
	padded_frame = np.where(np.isnan(padded_image), 0. , padded_image)

	uncentered_angular_displacement = np.tan(angle_in_first_quadrant / 2.)
	centered_angular_displacement = -np.sin(angle_in_first_quadrant)

	uncentered_frequencies = np.fft.fftfreq(out_shape[0])
    centered_frequencies = np.arange(-out_shape[0] / 2., out_shape[0] / 2.)

    uncentered_phase = np.exp(
        uncentered_angular_displacement *\
        ((-2.j * np.pi * uncentered_frequencies).T *\
        centered_frequecies).T)).T
    centered_phase = np.exp(
        centered_angular_displacement *\
        ((-2.j * np.pi * centered_frequencies).T *\
        uncentered_frequencies)

    rotated_image = np.fft.ifft(
        np.fft.fft(
            np.fft.ifft(
                np.fft.fft(
                    np.fft.ifft(
                        np.fft.fft(padded_image, axis=0).T * uncentered_phase, 
                        axis=0), 
                    axis=1) * centered_phase, 
                axis=1), 
            axis=0).T * uncentered_phase, 
        axis=0)\
        .at[padded_mask]\
        .set(np.nan)

    return np.real(rotated_image).copy()
